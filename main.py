# main.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn import set_config

from src.utils import set_seed, sequential_split_by_period_keyed
from src.models import ModelSpec, get_model
from src.predict import fit_and_predict, predict_only
from src.evaluate import mse, mae, oos_r2
from src.tune import tune_grid  # still available if you select method: grid
from src.report import (
    save_predictions, metrics_table, save_metrics_table,
    ols_table, ridge_coeffs_from_pipeline,
    generate_shap_reports, generate_linear_pipeline_diagnostics,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import sys, os, warnings
from sklearn.exceptions import ConvergenceWarning

# make stdout line-buffered/unbuffered so prints don't lag stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# show where warnings come from (or silence them)
warnings.simplefilter("default", ConvergenceWarning)   # or "ignore" to silence


# keep pandas through pipeline steps wherever supported (sklearn >=1.2)
set_config(transform_output="pandas")

# =========================
# Helpers
# =========================
def load_config(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        cfg = json.loads(path.read_text())
    else:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise SystemExit(f"Install pyyaml or use JSON config. Could not import yaml: {e}")
        cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or not cfg:
        raise SystemExit(f"Config '{path}' is empty or invalid.")

    return cfg

def _to_col(v):
    a = np.asarray(v, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def load_data(cfg: dict) -> pd.DataFrame:
    ds = cfg["data"]
    df = pd.read_csv(ds["path"])

    # ---- side & cp settings ----
    # Prefer 'callput' (your YAML), fall back to 'side' if present
    side = (ds.get("callput", ds.get("side", "all")) or "all").strip().lower()
    cp_col = ds.get("cp_col", "cp")
    call_vals = set(ds.get("call_values", [1, "C", "Call"]))
    put_vals  = set(ds.get("put_values",  [0, -1, "P", "Put"]))

    # ---- side filter ----
    if side in {"call", "put"}:
        allowed = call_vals if side == "call" else put_vals
        before = len(df)
        df = df[df[cp_col].isin(allowed)].copy()
        print(f"[data] side={side} kept {len(df)}/{before} rows")
    else:
        print(f"[data] side=all (no side filter)")

    # ---- moneyness filter (cp-aware; works for all/call/put) ----
    m_filter = (ds.get("moneyness", "all") or "all").strip().lower()
    if m_filter != "all":
        if "moneyness" not in df.columns:
            print("[data] 'moneyness' column missing; skipping moneyness filter.")
        else:
            band = float(ds.get("atm_band", 0.02))  # e.g., 0.02 -> ±2%
            m = df["moneyness"].astype(float)
            is_call = df[cp_col].isin(call_vals)
            is_put  = df[cp_col].isin(put_vals)

            if m_filter == "atm":
                mask = (m >= 1 - band) & (m <= 1 + band)
            elif m_filter == "otm":
                # calls OTM: m<1-band; puts OTM: m>1+band
                mask = (is_call & (m < 1 - band)) | (is_put & (m > 1 + band))
            elif m_filter == "itm":
                # calls ITM: m>1+band; puts ITM: m<1-band
                mask = (is_call & (m > 1 + band)) | (is_put & (m < 1 - band))
            else:
                print(f"[data] unknown moneyness '{m_filter}', skipping.")
                mask = pd.Series(True, index=df.index)

            before = len(df)
            df = df[mask].copy()
            print(f"[data] moneyness={m_filter} (band=±{band:.2%}) kept {len(df)}/{before} rows")
    else:
        print("[data] moneyness=all (no moneyness filter)")

    # ---- dropna (as configured) ----
    if ds.get("dropna", True):
        before = len(df)
        df = df.dropna(axis=0).copy()
        print(f"[data] dropna removed {before - len(df)} rows")

    # ---- date parsing & ordering ----
    date_col = ds.get("date_col", None)
    if not date_col:
        raise SystemExit("data.date_col must be set in config.")
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df.set_index("row_id")

def get_feature_list(df: pd.DataFrame, cfg: dict) -> List[str]:
    target = cfg["data"]["target"]
    date_col = cfg["data"]["date_col"]
    feats = cfg["data"].get("features", "auto")
    if feats == "auto" or feats is None:
        candidates = [c for c in df.columns if c not in (target, date_col)]
        numeric = df[candidates].select_dtypes(include=["number", "bool", "boolean"]).columns.tolist()
        return numeric
    return [c for c in feats if c not in (target, date_col)]

def ensure_series(y: Union[pd.Series, pd.DataFrame], name: str = "y") -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            s = y.iloc[:, 0]
            s.name = name
            return s
        raise SystemExit(f"Target must be a single column; got {y.shape[1]} columns.")
    y = y.copy()
    y.name = name
    return y

def maybe_apply_fixed_params(spec: ModelSpec, fixed: Optional[dict]) -> ModelSpec:
    if not fixed:
        return spec
    merged = dict(spec.params or {})
    merged.update(fixed)
    return ModelSpec(spec.name, merged)

# ---- y-transform helpers ----
class _IdentityY:
    def fit(self, y): return self
    def transform(self, y): return y
    def fit_transform(self, y): return y
    def inverse_transform(self, z): return z

class _AsinhY:
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
    def fit(self, y): return self
    def transform(self, y):
        return np.arcsinh(np.asarray(y, dtype=float) / self.scale)
    def fit_transform(self, y):
        return self.transform(y)
    def inverse_transform(self, z):
        return np.sinh(np.asarray(z, dtype=float)) * self.scale

def build_y_transformer(cfg: dict):
    tcfg = (cfg.get("train", {}) or {})
    name = str(tcfg.get("y_transform", "")).strip().lower()
    if name == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson", standardize=False)
    if name in {"asinh", "arcsinh"}:
        return _AsinhY(scale=float(tcfg.get("y_asinh_scale", 1.0)))
    return _IdentityY()

def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d: 
            return default
        d = d[k]
    return d

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, default="configs/base.yaml")
    args = ap.parse_args()

    # ---- config & rng
    cfg = load_config(Path(args.config))
    set_seed(cfg.get("train", {}).get("random_state", 42))

    # ---- data
    df = load_data(cfg)
    date_col = cfg["data"]["date_col"]
    target_col = cfg["data"]["target"]
    features = get_feature_list(df, cfg)
    if not features:
        raise SystemExit("No numeric features found. Check config 'features' or your data types.")

    # Optional: forecast horizon shift (in periods of `split.period`)
    H = int(cfg.get("forecast", {}).get("horizon_periods", 0))
    if H and H > 0:
        df[target_col] = df[target_col].shift(-H)

    # drop rows with missing target after any shift
    df = df.dropna(subset=[target_col])

    # ---- calendar-time split (returns *row_id* indices)
    split_cfg = cfg.get("split", {})
    train_frac = float(split_cfg.get("train_frac", 0.70))
    val_frac   = float(split_cfg.get("val_frac", 0.15))
    period     = split_cfg.get("period", "D")  # "D", "W", "M", ...

    train_idx, val_idx, test_idx = sequential_split_by_period_keyed(
        df, date_col=date_col, train=train_frac, val=val_frac, period=period
    )

    # ----- choose modeling target and inverse_y from config -----
    orig_target_col = cfg["data"].get("target", "ret_dh_w_adj")  # evaluation target (original units)

    # which scaled target?
    scale_kind = (cfg["data"].get("target_scale", {}) or {}).get("kind", "none").lower()
    if   scale_kind == "vega":
        y_col    = "ret_target_vega"
        scale_col= "target_scale_vega"
    elif scale_kind == "gamma":
        y_col    = "ret_target_gamma"
        scale_col= "target_scale_gamma"
    else:
        y_col    = orig_target_col
        scale_col= None
    print(f"[data] modeling target: '{y_col}' (scale kind: {scale_kind})")

    # build matrices/targets
    X_train = df.loc[train_idx, features]
    X_val   = df.loc[val_idx,   features]
    X_test  = df.loc[test_idx,  features]

    y_train = df.loc[train_idx, y_col].astype(float)
    y_val   = df.loc[val_idx,   y_col].astype(float)
    y_test  = df.loc[test_idx,  orig_target_col].astype(float)  # always evaluate in original units

    # TARGET PROCESSING (winsor + transform)
    # ================================
    winsor = float(cfg.get("train", {}).get("y_winsor_p", 0.0))
    if winsor > 0:
        lo, hi = y_train.quantile([winsor, 1 - winsor])
        y_train_fit = y_train.clip(lo, hi)
        y_val_fit   = y_val.clip(lo, hi)   # use TRAIN caps for val
    else:
        y_train_fit, y_val_fit = y_train, y_val

    tname = (cfg.get("train", {}).get("y_transform") or "").lower()

    def _identity(x):
        return pd.Series(x, index=getattr(x, "index", None)) if isinstance(x, pd.Series) else x

    inv_y = _identity

    if tname == "yeo-johnson":
        _pt = PowerTransformer(method="yeo-johnson", standardize=False)

        y_train_tr = pd.Series(
            np.asarray(_pt.fit_transform(_to_col(y_train_fit))).ravel(),
            index=y_train_fit.index
        )
        y_val_tr = pd.Series(
            np.asarray(_pt.transform(_to_col(y_val_fit))).ravel(),
            index=y_val_fit.index
        )
        y_test_tr = pd.Series(
            np.asarray(_pt.transform(_to_col(y_test))).ravel(),
            index=y_test.index
        )

        def inv_y(s):
            arr = np.asarray(s, dtype=float)
            if arr.ndim == 1: arr = arr.reshape(-1, 1)
            out = _pt.inverse_transform(arr).ravel()
            return pd.Series(out, index=getattr(s, "index", None))
    elif tname == "asinh":
        k = float(cfg.get("train", {}).get("y_asinh_scale",
                                           float(np.nanmedian(np.abs(y_train_fit)))))
        y_train_tr = pd.Series(np.arcsinh(np.asarray(y_train_fit)/k), index=y_train_fit.index)
        y_val_tr   = pd.Series(np.arcsinh(np.asarray(y_val_fit)/k),   index=y_val_fit.index)
        y_test_tr  = pd.Series(np.arcsinh(np.asarray(y_test)/k),      index=y_test.index)
        def inv_y(s):
            return pd.Series(np.sinh(np.asarray(s, dtype=float)) * k, index=getattr(s, "index", None))
    else:
        y_train_tr, y_val_tr = y_train_fit, y_val_fit
        y_test_tr            = y_test
        inv_y = _identity
    # ---- compose inverse_y with target_scale (if we trained on a scaled target) ----
    if scale_col is not None:
        # keep a stable reference to the *base* inverse before wrapping
        base_inv_y = inv_y

        def _inv_y_with_scale(s):
            # undo any y-transform using the base, NOT the wrapper
            out = base_inv_y(s)

            # determine an index to align with df[scale_col]
            idx = getattr(out, "index", None)
            if idx is None:
                # s may be a numpy array; try to use y_test's index if lengths match
                arr = np.asarray(out).ravel()
                n = len(arr)
                idx = y_test.index if len(y_test) == n else pd.RangeIndex(n)
                out = pd.Series(arr, index=idx)
            else:
                # ensure Series
                out = pd.Series(np.asarray(out).ravel(), index=idx)

            # fetch scale aligned on df, allow for any missing values
            scale = df.loc[idx, scale_col].astype(float).fillna(1.0)
            return out * scale

        # rebind public inverse to the safe wrapper
        inv_y = _inv_y_with_scale


    # ---- TRAIN-ONLY coverage filter (avoids leakage)
    cov_thr = float(cfg.get("train", {}).get("min_feature_coverage", 0.90))
    coverage = 1 - X_train.isna().mean()
    keep = coverage[coverage >= cov_thr].index.tolist()
    if not keep:
        raise SystemExit(
            f"No features meet the TRAIN coverage threshold {cov_thr:.2f}. "
            f"Lower it (train.min_feature_coverage) or specify features explicitly."
        )
    X_train = X_train[keep]
    X_val   = X_val[keep]
    X_test  = X_test[keep]
    features = keep
    print(f"[debug] kept {len(keep)} features after train-coverage filter (>={cov_thr:.0%})")

    # ---- choose REFIT set and align to kept columns
    refit_on = cfg["train"].get("refit_on", "train_val").lower()
    if refit_on not in {"train_val", "train_only"}:
        refit_on = "train_val"
    X_refit = pd.concat([X_train, X_val]) if refit_on == "train_val" else X_train
    y_refit = pd.concat([y_train_tr, y_val_tr]) if refit_on == "train_val" else y_train_tr

    # ---- drop zero-variance features on REFIT (measured on numeric cols)
    zero_var = X_refit.var(numeric_only=True)
    zero_var = zero_var[zero_var == 0.0].index.tolist()
    if zero_var:
        print(f"[warn] dropping zero-variance features on refit: {zero_var[:12]}{' ...' if len(zero_var)>12 else ''}")
        X_train = X_train.drop(columns=zero_var, errors="ignore")
        X_val   = X_val.drop(columns=zero_var, errors="ignore")
        X_test  = X_test.drop(columns=zero_var, errors="ignore")
        X_refit = X_refit.drop(columns=zero_var, errors="ignore")
        features = [f for f in features if f not in zero_var]

    print(f"[debug] shapes: X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}  X_refit={X_refit.shape}")

    def report_split_stats(name, y):
        yf = y[np.isfinite(y.values)]
        print(f"[{name}] n={y.size}  finite={yf.size}  n_nan={int(y.size - yf.size)}")
        if yf.size:
            qs = yf.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
            sk = yf.skew()
            from scipy.stats import kurtosis
            ku = float(kurtosis(yf, fisher=False, nan_policy="omit"))
            print(f"[{name}] quantiles={qs}  skew={sk:.3f}  kurtosis={ku:.3f}")

    for (nm, y_) in [("train", y_train), ("val", y_val), ("test", y_test), ("refit", y_refit)]:
        report_split_stats(nm, y_)


    # ---- outputs
    out_cfg = cfg["outputs"]
    OUTDIR = Path(out_cfg["dir"]); OUTDIR.mkdir(parents=True, exist_ok=True)
    CVDIR = OUTDIR / out_cfg.get("cv_dir", "cv"); CVDIR.mkdir(parents=True, exist_ok=True)
    index_label = out_cfg.get("index_label", "row_id")

 # ---- training / tuning setup
    models_to_run = [m.lower().strip() for m in cfg["train"]["models"]]

    tune_cfg = cfg.get("tuning", {}) or {}
    search_method = (tune_cfg.get("method", "grid")).lower()
    scoring       = (tune_cfg.get("scoring", "mse")).lower()
    n_iter        = int(tune_cfg.get("n_iter", 50))

    # === Default spaces (used if you didn't specify them in YAML) ===
    DEFAULT_SPACES = {
        "ridge": {
            "alpha": {"dist": "loguniform", "low": 1e-6, "high": 1e-2},
            "fit_intercept": {"values": [True]}
        },
        "huber": {
            "alpha":   {"dist": "loguniform", "low": 1e-6, "high": 1e-2},
            "epsilon": {"values": [1.1, 1.35, 1.5, 2.0]}
        },
        "random_forest": {
            "n_estimators":     {"values": [300, 600, 1000]},
            "max_depth":        {"values": [None, 6, 10, 15]},
            "min_samples_leaf": {"values": [1, 2, 5, 10]},
            "max_features":     {"values": ["sqrt", "log2", 0.5]},
            "bootstrap":        {"values": [True]}
        },
        "ffn": {
            "hidden_layer_sizes": {"values": [[64], [128], [64,64], [128,128], [64,64,64]]},
            "activation":         {"values": ["relu", "tanh"]},
            "alpha":              {"dist": "loguniform", "low": 1e-6, "high": 1e-2},
            "learning_rate_init": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
            "early_stopping":     {"values": [True]},
            "n_iter_no_change":   {"values": [10]},
            "max_iter":           {"values": [300]}
        },
        # keep your existing ones in YAML for enet_pairs / lgbm_* (user overrides will win)
    }

    # read spaces/grids **from under tuning** and merge defaults (user overrides win)
    spaces_user = tune_cfg.get("spaces", {}) or {}
    grids_user  = tune_cfg.get("grids",  {}) or {}
    spaces = {**DEFAULT_SPACES, **spaces_user}
    grids  = {**grids_user}

    cvcfg  = tune_cfg.get("cv", {}) or {}
    n_splits   = int(cvcfg.get("n_splits", 5))
    min_trainP = int(cvcfg.get("min_train_periods", 250))
    valP       = int(cvcfg.get("val_periods", 60))
    cv_period  = cvcfg.get("period", "D")

    # selection config (keeps MSE primary; improves R2_oos via tie-break)
    select_cfg   = tune_cfg.get("select", {}) or {}
    select_rule  = (select_cfg.get("rule", "one_se")).lower()       # "one_se" or "min"
    tie_break    = (select_cfg.get("tie_breaker", "r2_oos")).lower()

    print(f"[tune-setup] method={search_method}  spaces={list(spaces.keys())}  grids={list(grids.keys())}  n_iter={n_iter}")

    # dates for CV folds (aligned to TRAIN subset)
    dates_train = df.loc[X_train.index, date_col]

    fitted: Dict[str, object] = {}
    yhat_test_map: Dict[str, pd.Series] = {}

    print("y_test quantiles:", y_test.quantile([.01,.05,.5,.95,.99]).to_dict())
    print("skew(y_test)=", float(y_test.skew()), "  kurtosis(y_test)=", float(y_test.kurtosis()))
    
    
    for name in models_to_run:
        if search_method == "random" and name in spaces and spaces[name]:
            from src.tune import tune_random
            res = tune_random(
                model_name=name,
                X=X_train, y=y_train_tr,
                space=spaces[name],
                n_iter=n_iter,
                n_splits=n_splits,
                min_train=min_trainP,
                val_size=valP,
                period=cv_period,
                dates=dates_train,
                scoring=scoring,
                random_state=cfg.get("train", {}).get("random_state", 42),
                verbose=True,
                inverse_y=inv_y,                    # <<< NEW: evaluate in original units
                select_rule=select_rule,            # <<< NEW
                tie_breaker=tie_break               # <<< NEW
            )
            res.cv_table.to_csv(CVDIR / f"cv_random_{name}.csv", index=False)
            print(f"[tune] best {name} params:", res.best_spec.params)
            spec = res.best_spec

        elif search_method == "grid" and name in grids and grids[name]:
            from src.tune import tune_grid
            res = tune_grid(
                model_name=name,
                X=X_train, y=y_train_tr,
                grid=grids[name],
                n_splits=n_splits,
                min_train=min_trainP,
                val_size=valP,
                period=cv_period,
                dates=dates_train,
                scoring=scoring,
                inverse_y=inv_y,                    # <<< NEW: evaluate in original units
                select_rule=select_rule,
                tie_breaker=tie_break
            )
            res.cv_table.to_csv(CVDIR / f"cv_{name}.csv", index=False)
            print(f"[tune] best {name} params:", res.best_spec.params)
            spec = res.best_spec
        else:
            spec = ModelSpec(name, params={})


        # fixed overrides from config
        fixed_overrides = cfg.get("model_params", {}).get(name, {})
        spec = maybe_apply_fixed_params(spec, fixed_overrides)

        m_refit = np.isfinite(y_refit.values)
        if m_refit.sum() < len(m_refit):
            n_bad = int((~m_refit).sum())
            print(f"[clean] dropping {n_bad} rows with non-finite REFIT target")
        X_refit = X_refit.loc[m_refit]
        y_refit = y_refit.loc[m_refit]

        # fit & predict in transformed space, then invert for evaluation
        model = get_model(spec)
        model, yhat_test_tr = fit_and_predict(model, X_refit, y_refit, X_test)
        if not isinstance(yhat_test_tr, pd.Series):
            yhat_test_tr = pd.Series(yhat_test_tr, index=X_test.index)
        yhat_test = inv_y(yhat_test_tr)

        assert yhat_test.index.equals(y_test.index), f"Index mismatch for {name}"

        fitted[name] = model
        yhat_test_map[name] = yhat_test

        # ---- optional benchmark predictions
    bench_name = (cfg.get("benchmark") or "").lower().strip() or None
    yhat_bench = None
    if bench_name == "zero":
        bench_model = get_model(ModelSpec("zero", params={}))
        yhat_bench = predict_only(bench_model, X_test)
        yhat_test_map["zero"] = yhat_bench

    # ---- save predictions & base metrics
    save_predictions(
        y=y_test,
        yhat_map=yhat_test_map,
        path=OUTDIR / out_cfg["predictions_csv"],
        index_label=index_label
    )
    tbl = metrics_table(y_test, yhat_test_map, bench=yhat_bench)
    save_metrics_table(tbl, OUTDIR / out_cfg["metrics_csv"], index_label="model")
    print("\n=== Test metrics (pooled) ===")
    print(tbl)

    # ---- cross-sectional OOS R^2 (by date)
    from src.evaluate import oos_r2_xs

    dates_test = pd.to_datetime(df.loc[y_test.index, date_col]).dt.normalize()

    xs_rows = []
    for mname, yhat in yhat_test_map.items():
        avg_xs, per_date = oos_r2_xs(y_test, yhat, groups=dates_test, weight="n")
        xs_rows.append({"model": mname, "OOS_R2_xs": avg_xs})
        # keep a tidy per-date file per model
        per_date.sort_index().to_csv(OUTDIR / f"r2_oos_xs_per_date__{mname}.csv", index=True)

    xs_df = pd.DataFrame(xs_rows)

    # merge & re-save (option A: overwrite; option B: save as a separate file)
    tbl_aug = (tbl.reset_index()
                .merge(xs_df, on="model", how="left")
                .set_index("model"))

    # overwrite original metrics.csv
    save_metrics_table(tbl_aug, OUTDIR / out_cfg["metrics_csv"], index_label="model")

    # or, alternatively:
    # save_metrics_table(tbl_aug, OUTDIR / "metrics_test_with_xs.csv", index_label="model")

    print("\n=== Cross-sectional OOS R^2 (avg over dates) ===")
    print(xs_df.set_index("model").sort_values("OOS_R2_xs"))

    print("\n=== Test metrics (pooled + OOS_R2_xs) ===")
    print(tbl_aug)

    # ---- SHAP (single consolidated call)
    try:
        shap_dir = OUTDIR / "shap"
        generate_shap_reports(fitted, X_refit, X_test, shap_dir)
    except Exception as e:
        print(f"[shap] failed to generate reports: {e}")


    # ---- coefficients / diagnostics
    if "ols" in fitted:
        try:
            ols_df = ols_table(fitted["ols"])
            ols_df.to_csv(OUTDIR / out_cfg["coeffs_ols_csv"], index=True, index_label="term")
        except Exception as e:
            print("[warn] OLS coefficients not saved:", e)

    for ridge_name in ("ridge", "ridge_cv"):
        if ridge_name in fitted:
            try:
                r_df = ridge_coeffs_from_pipeline(
                    fitted[ridge_name],
                    feature_names=X_test.columns.tolist(),
                    original_units=True
                )
                r_df.to_csv(OUTDIR / out_cfg["coeffs_ridge_csv"], index=True, index_label="term")
                break
            except Exception as e:
                print(f"[warn] {ridge_name} coefficients not saved:", e)

    # --- Diagnostics ---
    diag_cfg = cfg.get("diagnostics", {})
    do_generic = bool(diag_cfg.get("generic", True))
    do_linear  = bool(diag_cfg.get("linear", False))
    topk_scatter = int(diag_cfg.get("topk_scatter", 8))

    if do_generic:
        from src.report import generate_model_diagnostics_generic
        try:
            diag_all_dir = OUTDIR / "diag_all"
            print("[diag-all] fitted models:", list(fitted.keys()))
            for mname, mdl in fitted.items():
                generate_model_diagnostics_generic(
                    name=mname,
                    model=mdl,
                    X_test=X_test,
                    y_test=y_test,
                    yhat_test=yhat_test_map.get(mname),
                    outdir=diag_all_dir,
                    topk_scatter=topk_scatter
                )
        except Exception as e:
            print(f"[warn] generic diagnostics failed: {e}")

    if do_linear:
        from src.report import generate_linear_pipeline_diagnostics
        try:
            lin_diag_dir = OUTDIR / "linear_diag"
            print("[diag-linear] fitted models:", list(fitted.keys()))
            for m in ("ridge", "ridge_cv", "huber", "enet_pairs"):
                if m in fitted:
                    generate_linear_pipeline_diagnostics(
                        name=m,
                        pipe=fitted[m],
                        X_refit=X_refit, y_refit=y_refit,
                        X_test=X_test,   y_test=y_test,
                        yhat_test=yhat_test_map.get(m),
                        outdir=lin_diag_dir,
                        topk_scatter=topk_scatter
                    )
        except Exception as e:
            print(f"[warn] linear diagnostics failed: {e}")


    # ---- bucketed metrics (robust to missing cols)
    from src.report import bucketed_metrics
    df_test = df.loc[y_test.index].copy()
    if "moneyness" in df_test.columns:
        df_test["abs_moneyness"] = df_test["moneyness"].abs()
    buckets = {}
    if "abs_moneyness" in df_test.columns:
        buckets["abs_moneyness_q"] = {"col": "abs_moneyness", "q": [0, .33, .66, 1.0]}
    if "tau" in df_test.columns:
        buckets["tau_q"] = {"col": "tau", "q": [0, .25, .5, .75, 1.0]}
    if "cp" in df_test.columns:
        buckets["side"] = {"col": "cp", "bins": [-0.5, 0.5, 1.5]}

    if buckets:
        bkt = bucketed_metrics(df_test, y_test, yhat_test_map, buckets, min_n=50, var_floor=1e-6)
        bkt.to_csv(OUTDIR / "metrics_bucketed.csv", index=False)
        print("\n=== Bucketed metrics (head) ===")
        print(bkt.head(12))

    # ---- console summary
    for mname, yhat in yhat_test_map.items():
        print(f"{mname:>10} | MSE={mse(y_test, yhat):.6f}  MAE={mae(y_test, yhat):.6f}  "
              f"OOS_R2 vs 0={oos_r2(y_test, yhat):.4f}")

if __name__ == "__main__":
    main()