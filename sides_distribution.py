# analysis/sides_stratified_distribution.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import yaml
except Exception as e:
    raise SystemExit(f"Install pyyaml: {e}")

# ----------------- utils -----------------

def load_cfg(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return yaml.safe_load(path.read_text())

def standardize_side(df: pd.DataFrame, cfg: dict) -> pd.Series:
    ds = cfg.get("data", {})
    cp_col = ds.get("cp_col", "cp")
    if cp_col not in df.columns:
        raise SystemExit(f"Column '{cp_col}' not in data for call/put classification.")

    s = df[cp_col]
    call_vals = set(ds.get("call_values", [1, "C", "Call", "CALL", "call"]))
    put_vals  = set(ds.get("put_values",  [0, -1, "P", "Put", "PUT", "put"]))

    # Start with object dtype to allow mixing strings/NA safely
    side = pd.Series(pd.NA, index=df.index, dtype="object")

    # Numeric mask (handles 1/0/-1/1.0/0.0 etc.)
    call_num = [v for v in call_vals if not isinstance(v, str)]
    put_num  = [v for v in put_vals  if not isinstance(v, str)]
    call_mask = s.isin(call_num) if call_num else pd.Series(False, index=df.index)
    put_mask  = s.isin(put_num)  if put_num  else pd.Series(False, index=df.index)

    # String mask (handles 'C'/'P'/'Call'/'Put', case/space-insensitive)
    s_str = s.astype(str).str.strip().str.upper()
    call_str = {str(v).strip().upper() for v in call_vals if isinstance(v, str)}
    put_str  = {str(v).strip().upper() for v in put_vals  if isinstance(v, str)}
    call_mask |= s_str.isin(call_str)
    put_mask  |= s_str.isin(put_str)

    side[call_mask] = "call"
    side[put_mask]  = "put"

    unknown = int(side.isna().sum())
    if unknown:
        print(f"[warn] {unknown} rows with unknown side (neither in call_values nor put_values); dropping in side splits.")

    return side.astype("string")  # pandas string dtype (nullable)


def robust_bins(y: pd.Series, nbins: int = 80) -> np.ndarray:
    y = y.dropna()
    if y.empty:
        return np.linspace(-1, 1, nbins)
    qlo, qhi = y.quantile([0.005, 0.995])
    if not np.isfinite(qlo): qlo = float(y.min())
    if not np.isfinite(qhi): qhi = float(y.max())
    if qhi <= qlo: qlo, qhi = float(y.min()), float(y.max())
    if qhi == qlo: qhi = qlo + 1e-6
    return np.linspace(qlo, qhi, nbins)

def summarize_series(y: pd.Series) -> dict:
    y = y.dropna()
    if y.empty:
        return {}
    q = y.quantile([0.01,0.05,0.25,0.5,0.75,0.95,0.99])
    return {
        "n": int(y.size),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)),
        "skew": float(y.skew()),
        "kurtosis": float(y.kurtosis()),
        "min": float(y.min()),
        "max": float(y.max()),
        "q01": float(q.loc[0.01]), "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]), "q50": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]), "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "zero_bench_mse": float((y**2).mean()),
        "mad": float((y - y.median()).abs().median()),
    }

def ecdf(y: pd.Series):
    y = np.sort(y.dropna().values)
    n = y.size
    if n == 0:
        return np.array([]), np.array([])
    return y, np.arange(1, n+1) / n

def two_sample_tests(y_call: pd.Series, y_put: pd.Series) -> dict:
    a, b = y_call.dropna().values, y_put.dropna().values
    out = {}
    if len(a) and len(b):
        ks = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
        out["ks_stat"] = float(ks.statistic); out["ks_pvalue"] = float(ks.pvalue)
        lev = stats.levene(a, b, center="median")
        out["levene_stat"] = float(lev.statistic); out["levene_pvalue"] = float(lev.pvalue)
        try:
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            out["mw_stat"] = float(mw.statistic); out["mw_pvalue"] = float(mw.pvalue)
        except Exception:
            pass
    return out

# ----------------- bucketing -----------------

def make_bucket(series: pd.Series, *, q: list[float] | None = None, bins: list[float] | None = None, labels=None) -> pd.Categorical:
    s = series.copy()
    if q is not None:
        return pd.qcut(s, q=q, duplicates="drop", labels=labels)
    elif bins is not None:
        return pd.cut(s, bins=bins, labels=labels)
    else:
        raise ValueError("Provide either q=quantiles or bins=edges.")

def bucketed_summary(df: pd.DataFrame, target: str, side_name: str,
                     bucket_name: str, cats: pd.Categorical, min_n: int = 50) -> pd.DataFrame:
    rows = []
    yfull = df[target]
    for g in cats.cat.categories:
        mask = (cats == g)
        n = int(mask.sum())
        if n < min_n:
            continue
        y = yfull.loc[mask]
        summ = summarize_series(y)
        if summ:
            summ.update({"side": side_name, "bucket": bucket_name, "group": str(g), "n": n})
            rows.append(summ)
    return pd.DataFrame(rows)

# ----------------- plotting -----------------

def save_side_plots(y: pd.Series, outdir: Path, title_prefix: str, nbins: int = 80):
    bins = robust_bins(y, nbins)
    # histogram
    plt.figure()
    plt.hist(y.dropna(), bins=bins, density=True)
    plt.title(f"{title_prefix} – Histogram (density)")
    plt.xlabel(y.name); plt.ylabel("density")
    plt.tight_layout(); plt.savefig(outdir / f"hist_{title_prefix}.png", dpi=150); plt.close()
    # ECDF
    x, F = ecdf(y)
    plt.figure()
    if x.size:
        plt.plot(x, F)
    plt.title(f"{title_prefix} – ECDF")
    plt.xlabel(y.name); plt.ylabel("F(y)")
    plt.tight_layout(); plt.savefig(outdir / f"ecdf_{title_prefix}.png", dpi=150); plt.close()
    # QQ
    plt.figure()
    stats.probplot(y.dropna(), dist="norm", plot=plt)
    plt.title(f"{title_prefix} – QQ vs Normal")
    plt.tight_layout(); plt.savefig(outdir / f"qq_{title_prefix}.png", dpi=150); plt.close()

def save_bucket_overlay_hist(df: pd.DataFrame, target: str, cats: pd.Categorical,
                             out: Path, title: str, nbins: int = 80):
    y = df[target]
    bins = robust_bins(y, nbins)
    plt.figure()
    for g in cats.cat.categories:
        mask = (cats == g)
        if mask.sum() == 0:
            continue
        plt.hist(y.loc[mask].dropna(), bins=bins, density=True, histtype="step", label=str(g))
    plt.legend()
    plt.title(f"{title} – Histogram overlay (density)")
    plt.xlabel(target); plt.ylabel("density")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def save_bucket_overlay_ecdf(df: pd.DataFrame, target: str, cats: pd.Categorical,
                             out: Path, title: str):
    plt.figure()
    for g in cats.cat.categories:
        mask = (cats == g)
        if mask.sum() == 0:
            continue
        x, F = ecdf(df.loc[mask, target])
        if x.size:
            plt.plot(x, F, label=str(g))
    plt.legend()
    plt.title(f"{title} – ECDF overlay")
    plt.xlabel(target); plt.ylabel("F(y)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def save_bucket_boxplot(df: pd.DataFrame, target: str, cats: pd.Categorical,
                        out: Path, title: str):
    data = [df.loc[(cats == g), target].dropna().values for g in cats.cat.categories]
    labels = [str(g) for g in cats.cat.categories]
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(f"{title} – Boxplot (no outliers)")
    plt.xlabel("bucket"); plt.ylabel(target)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, required=True)
    ap.add_argument("--outdir", "-o", type=str, default="results/diagnostics_sides_stratified")
    ap.add_argument("--min_n", type=int, default=50, help="min observations per bucket group to include")
    ap.add_argument("--nbins", type=int, default=80, help="histogram bins (robust range)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    ds = cfg["data"]
    target = ds["target"]
    df = pd.read_csv(ds["path"])
    df = df[pd.notna(df[target])].copy()

    # side labels
    df["side"] = standardize_side(df, cfg)

    # optional convenience: abs moneyness
    if "moneyness" in df.columns:
        df["abs_moneyness"] = df["moneyness"].abs()

    # output
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- overall summaries per side --------
    parts = []
    for side_name, mask in [("all", slice(None)),
                            ("call", df["side"] == "call"),
                            ("put",  df["side"] == "put")]:
        y = pd.Series(df.loc[mask, target].values, index=df.index[mask], name=target)
        summ = summarize_series(y)
        if summ:
            summ["which"] = side_name
            parts.append(summ)
        # plots per side
        save_side_plots(y, outdir, f"{side_name}", nbins=args.nbins)

    summary = pd.DataFrame(parts).set_index("which")
    summary.to_csv(outdir / "summary_by_side.csv")
    print("\n=== Summary by side ===")
    cols = ["n","mean","std","skew","kurtosis","q01","q05","q50","q95","q99","zero_bench_mse"]
    print(summary[cols])

    # tests calls vs puts
    tests = two_sample_tests(
        df.loc[df["side"]=="call", target],
        df.loc[df["side"]=="put",  target]
    )
    (outdir / "two_sample_tests.json").write_text(json.dumps(tests, indent=2))
    print("\n=== Two-sample tests (calls vs puts) ===")
    print(tests or "(insufficient data)")

    # -------- stratified within each side --------
    by_bucket_rows = []

    for side_name, mask in [("all", slice(None)),
                            ("call", df["side"] == "call"),
                            ("put",  df["side"] == "put")]:
        df_s = df.loc[mask].copy()
        if df_s.empty:
            continue
        side_dir = outdir / f"side_{side_name}"
        side_dir.mkdir(parents=True, exist_ok=True)

        # abs(moneyness) quantile buckets (if available)
        if "abs_moneyness" in df_s.columns:
            try:
                cats_mq = make_bucket(df_s["abs_moneyness"], q=[0, .33, .66, 1.0])
                title = f"{target} – {side_name} – abs(moneyness) quantiles"
                save_bucket_overlay_hist(df_s, target, cats_mq, side_dir / "hist_abs_moneyness_q.png", title, args.nbins)
                save_bucket_overlay_ecdf(df_s, target, cats_mq, side_dir / "ecdf_abs_moneyness_q.png", title)
                save_bucket_boxplot(df_s, target, cats_mq, side_dir / "box_abs_moneyness_q.png", title)
                by_bucket_rows.append(bucketed_summary(df_s, target, side_name, "abs_moneyness_q", cats_mq, args.min_n))
            except Exception as e:
                print(f"[warn] abs_moneyness quantile bucketing failed for {side_name}: {e}")

        # tau fixed bins (if available): 0–7d, 7–30d, 30–90d, 90d+
        if "tau" in df_s.columns:
            try:
                tau_bins = [0, 7/365, 30/365, 90/365, np.inf]
                cats_tf = make_bucket(df_s["tau"], bins=tau_bins,
                                      labels=["≤1w","1–4w","1–3m",">3m"])
                title = f"{target} – {side_name} – tau fixed"
                save_bucket_overlay_hist(df_s, target, cats_tf, side_dir / "hist_tau_fixed.png", title, args.nbins)
                save_bucket_overlay_ecdf(df_s, target, cats_tf, side_dir / "ecdf_tau_fixed.png", title)
                save_bucket_boxplot(df_s, target, cats_tf, side_dir / "box_tau_fixed.png", title)
                by_bucket_rows.append(bucketed_summary(df_s, target, side_name, "tau_fixed", cats_tf, args.min_n))
            except Exception as e:
                print(f"[warn] tau fixed bucketing failed for {side_name}: {e}")

            # tau quantiles (balanced counts)
            try:
                cats_tq = make_bucket(df_s["tau"], q=[0, .25, .5, .75, 1.0])
                title = f"{target} – {side_name} – tau quantiles"
                save_bucket_overlay_hist(df_s, target, cats_tq, side_dir / "hist_tau_q.png", title, args.nbins)
                save_bucket_overlay_ecdf(df_s, target, cats_tq, side_dir / "ecdf_tau_q.png", title)
                save_bucket_boxplot(df_s, target, cats_tq, side_dir / "box_tau_q.png", title)
                by_bucket_rows.append(bucketed_summary(df_s, target, side_name, "tau_q", cats_tq, args.min_n))
            except Exception as e:
                print(f"[warn] tau quantile bucketing failed for {side_name}: {e}")

    if by_bucket_rows:
        by_bucket = pd.concat(by_bucket_rows, ignore_index=True)
        by_bucket = by_bucket.sort_values(["side","bucket","group"]).reset_index(drop=True)
        by_bucket.to_csv(outdir / "summary_by_side_bucket.csv", index=False)
        print("\n=== Stratified summary (head) ===")
        print(by_bucket.head(12))
    else:
        print("\n(no stratified summaries produced)")

    print(f"\nWrote outputs to: {outdir}")

if __name__ == "__main__":
    main()
