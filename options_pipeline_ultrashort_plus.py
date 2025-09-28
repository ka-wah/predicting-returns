#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra‑short‑dated BTC options pipeline (0–3 DTE focus)
------------------------------------------------------

Builds leakage‑safe features tailored to 0–3D horizons, extending a Black–76
baseline with:
  • Robust IV fill + hierarchical smoothing
  • Constant‑maturity ATM IV (2D default) + *changes* and *term slope*
  • Smile/skew proxy around ATM (OTM put – OTM call IV)
  • Leakage‑safe past RV over each contract's DTE
  • Greeks (model) + scaling + convexity interactions
  • Option microstructure: relative spread at quote level
  • Positioning: OI concentration, put/call imbalance, distance to OI “wall”
  • Dealer‑style exposures (GEX proxy from gamma × OI)
  • Daily signals with changes and rolling z‑scores (rv, basis, baspread, sentiment)
  • Calendar effects (weekend/fri/long‑gap dummies)

Notes
-----
- All per‑row features are computed using information available at date *t*.
- Any forward‑looking RV is avoided; we only use past windows.
- OI‑based features require an `openinterest` column. If missing, they are NaN.
- Funding rates can be merged as an extra daily series if desired (see TODO hook).

CLI (example)
-------------
python options_pipeline_ultrashort_plus.py \
  --options data/raw/options.csv \
  --futures data/raw/yahoo-futurebtc.csv --fut_date_col date --fut_price_col close \
  --spot data/raw/yahoo-btcusd.csv \
  --bitw data/raw/yahoo-bitw.csv \
  --public data/processed/google-trends-btc.csv \
  --rtrs data/processed/rtrs-sentiment.csv \
  --reddit data/processed/reddit-sentiment.csv \
  --outdir data/processed --build_returns --consecutive_only --forward_fill_futures

Outputs
-------
- options_with_returns.csv  (row‑level target + features)
- diagnostics_returns.csv   (summary diagnostics)
"""

import argparse, os, math
import numpy as np
import pandas as pd
from typing import Iterable

import json, gzip
from pathlib import Path

import numpy as np
import pandas as pd

def _clean_delta_series(df: pd.DataFrame, col: str,
                        sentinels=(-99.99, -999, 999, -1e9, 1e9)) -> pd.Series:
    """
    Return a cleaned Series for df[col]:
      - if column missing → all-NaN Series aligned to df.index
      - coerce non-numeric to NaN
      - map sentinel codes to NaN
      - drop impossible magnitudes (|delta|>1)
    """
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index)
    if sentinels:
        s = s.replace(list(sentinels), np.nan)
    s = s.where(s.between(-1.0, 1.0), np.nan)
    return s


def _build_target_scale(df, kind: str, use_abs: bool, floor: float) -> pd.Series:
    """
    Returns a scale factor per row based on Greeks.
    - kind='none' -> ones
    - kind='vega' -> (abs) vega
    - kind='gamma'-> (abs) gamma
    Clips by 'floor' to avoid exploding scaled returns.
    """
    if kind == "none":
        s = pd.Series(1.0, index=df.index)
    elif kind == "vega":
        col = "vega" if "vega" in df.columns else None
        if col is None:
            raise ValueError("target_scale_kind=vega but 'vega' column not present.")
        s = df[col].astype(float)
    elif kind == "gamma":
        col = "gamma" if "gamma" in df.columns else None
        if col is None:
            raise ValueError("target_scale_kind=gamma but 'gamma' column not present.")
        s = df[col].astype(float)
    else:
        raise ValueError(f"Unknown target_scale_kind={kind}")

    if use_abs:
        s = s.abs()
    s = s.clip(lower=float(floor))
    s.replace([np.inf, -np.inf], np.nan, inplace=True)
    s = s.fillna(float(floor))
    return s



def _clean_delta(series: pd.Series, sentinels=(-99.99, -999, 999, -1e9, 1e9)) -> pd.Series:
    """Return delta with sentinels → NaN and values outside [-1, 1] → NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if sentinels:
        s = s.replace(list(sentinels), np.nan)
    # drop impossible magnitudes (handles bad vendor rows)
    s = s.where(s.between(-1.0, 1.0), np.nan)
    return s


def _write_csv_with_config(df, path, cfg, key: str = "cfg"):
    """
    Write CSV with line 1 containing the configuration as a JSON comment:
    # cfg: {"arg":"value", ...}

    - Works with .csv and .csv.gz
    - cfg can be argparse.Namespace or dict
    """
    path = Path(path)
    if not isinstance(cfg, dict):
        try:
            cfg = vars(cfg)  # argparse.Namespace -> dict
        except Exception:
            cfg = {"cfg": str(cfg)}

    # make JSON compact & robust
    cfg_json = json.dumps(cfg, separators=(",", ":"), default=str)
    header_line = f"# {key}: {cfg_json}\n"

    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(header_line)
            df.to_csv(f, index=False)
    else:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(header_line)
            df.to_csv(f, index=False)

# ===================== numerics & Black–76 =====================
SQRT2PI = math.sqrt(2.0 * math.pi)

def phi(x):
    return np.exp(-0.5 * x * x) / SQRT2PI

try:
    from numpy import erf as _erf
except Exception:  # pragma: no cover
    from math import erf as _erf

def N(x):
    try:
        return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    except TypeError:
        vf = np.vectorize(_erf)
        return 0.5 * (1.0 + vf(x / np.sqrt(2.0)))

def b76_price(F, K, tau, sigma, cp, df=1.0):
    try:
        if cp not in (1, -1) or not (F > 0 and K > 0 and tau > 0 and sigma > 0):
            return np.nan
        vs = sigma * np.sqrt(tau)
        d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau) / vs
        d2 = d1 - vs
        return df * (F * N(d1) - K * N(d2)) if cp == 1 else df * (K * N(-d2) - F * N(-d1))
    except Exception:
        return np.nan

def b76_greeks_vec(F, K, tau, sigma, cp, r=0.0):
    F = np.asarray(F, float); K = np.asarray(K, float)
    tau = np.asarray(tau, float); sigma = np.asarray(sigma, float)
    cp = np.asarray(cp, float)
    df = np.exp(-r * tau)
    vs = sigma * np.sqrt(tau)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau) / vs
        nd1 = phi(d1); Nd1 = N(d1)
        delta = np.where(cp == 1, df * Nd1, -df * (1.0 - Nd1))
        gamma = df * nd1 / (F * vs)
        vega  = df * F * nd1 * np.sqrt(tau)                    # per 1.00 vol
        theta = -df * (F * nd1 * sigma) / (2.0 * np.sqrt(tau)) # core (r≈0)
        bad = ~((F>0)&(K>0)&(tau>0)&(sigma>0)&np.isin(cp,[-1,1]))
        for arr in (delta, gamma, vega, theta):
            arr[bad] = np.nan
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

# ===================== helpers =====================

def infer_option_key(df: pd.DataFrame):
    for c in ["optionid","securityid","optionsymbol"]:
        if c in df.columns: return [c]
    cols = ["optionroot","futuresymbol","expiration","K","callput","cp","strike"]
    key  = [c for c in cols if c in df.columns]
    if "K" not in key and "strike" in df.columns: key.append("strike")
    return key if key else ["expiration","K","cp"]


def corwin_schultz_spread(df: pd.DataFrame, overnight_adjust: bool = True) -> pd.Series:
    req = {"high","low","close"}
    if not req.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    h = df["high"].astype(float).copy()
    l = df["low"].astype(float).copy()
    c_prev = df["close"].shift(1).astype(float)
    if overnight_adjust:
        up = (c_prev - h).clip(lower=0)
        dn = (l - c_prev).clip(lower=0)
        h = h + up - dn
        l = l + up - dn
    l = l.where(l > 0)
    hl = np.log(h / l)
    beta = hl.pow(2) + hl.shift(1).pow(2)
    Hmax = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    Lmin = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(Hmax / Lmin).pow(2)
    den = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0)


def fisher_skew(a: np.ndarray) -> float:
    n = a.size
    if n < 3: return np.nan
    m = np.nanmean(a)
    s = np.nanstd(a, ddof=1)
    if not np.isfinite(s) or s <= 0: return np.nan
    return np.nanmean(((a - m)/s)**3)

# --- BEGIN PATCH: options_pipeline_ultrashort_plus.py (helper) ---
def _merge_futures_and_compute_moneyness(wrk, args):
    """
    Do the futures merge and (log) moneyness computation exactly once.
    """
    wrk, fut_df = merge_futures(
        wrk,
        args.futures,
        args.fut_date_col,
        args.fut_price_col,
        forward_fill=getattr(args, "forward_fill_futures", True),
        max_ffill=getattr(args, "max_ffill", 5),
    )
    # compute log-moneyness and bucket once
    with np.errstate(divide="ignore", invalid="ignore"):
        wrk["logM"] = np.log(wrk["F"] / wrk["K"])
    wrk["lm_bucket"] = (wrk["logM"] / 0.05).round().astype("Int64")
    return wrk, fut_df
# --- END PATCH ---


# ===================== cleaners & BBO =====================

def standardize_options(opt: pd.DataFrame) -> pd.DataFrame:
    inv = {c.lower(): c for c in opt.columns}
    ren = {}
    for want, alts in {
        "date": ["date","quotedate","datadate","observationdate"],
        "expiration": ["expiration","expiry","maturity"],
        "callput": ["callput","cp","type"],
        "settlementprice": ["settlementprice","settle","price","closeprice","optionprice","close"],
        "bid": ["bid","bestbid"],
        "offer": ["offer","bestask","ask"],
        "iv": ["iv","impliedvol","impl_vol","impliedvolatility","sigma"],
        # Vendor‑specific names can be added here:
        "F": ["F","fut","futures","underlyingfut","underlying","underlyingprice","futuresettlementprice"],
        "K": ["K","strike","strikeprice"],
        "dte": ["dte","days","days_to_expiration","daystomaturity","days"],
        "openinterest": ["openinterest","oi"],
        "volume": ["volume","vol"],
        "delta": ["delta"], "gamma": ["gamma"], "vega": ["vega"], "theta": ["theta"],
        "fut_sym": ["futuresymbol","future","symbol","contract","root"],
        "optionsymbol": ["optionsymbol","symbol_option","optionroot"]
    }.items():
        for a in alts:
            if a in inv:
                ren[inv[a]] = want
                break

    df = opt.rename(columns=ren).copy()

    # normalize types
    for c in ["date","expiration"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    for c in [
        "settlementprice","bid","offer","iv","F","K","dte",
        "openinterest","volume","delta","gamma","vega","theta"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace({-99.99: np.nan, -99.990000: np.nan})

    # tau (years)
    if "dte" in df.columns:
        df["tau"] = df["dte"] / 365.0

    # cp ∈ {+1,-1}
    r = df["callput"].astype(str).str.strip().str.upper() if "callput" in df.columns else pd.Series(index=df.index, dtype=str)
    df["cp"] = np.where(r.str[0].isin(["C","+","1"]), 1, np.where(r.str[0].isin(["P","-","0"]), -1, np.nan))

    # log‑moneyness + bucket (prelim; recomputed after futures merge)
    if "F" in df.columns and "K" in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["logM"] = np.log(df["F"] / df["K"])
    else:
        df["logM"] = np.nan
    df["lm_bucket"] = (df["logM"]/0.05).round().astype("Int64")

    # Best bid/offer & option‑relative spread
    if "bid" in df.columns and "offer" in df.columns:
        bid = df["bid"].where(df["bid"] > 0)
        ask = df["offer"].where(df["offer"] > 0)
        mid = (bid + ask) / 2.0
        df["bbo_mid"] = mid
        with np.errstate(divide='ignore', invalid='ignore'):
            df["opt_spread_rel"] = (ask - bid) / mid
    else:
        df["bbo_mid"] = np.nan
        df["opt_spread_rel"] = np.nan

    # Detect min tick (for later tick filters)
    pos = df["settlementprice"][df["settlementprice"] > 0] if "settlementprice" in df.columns else pd.Series([], dtype=float)
    df["min_tick_detected"] = float(pos.min()) if len(pos) > 0 else np.nan
    return df


# ---------- robust IV inversion from observed prices ----------

def impvol_b76(price, F, K, tau, cp, df=1.0, lo=1e-6, hi_start=2.0, hi_cap=10.0, tol=1e-7, max_iter=80):
    """Robust bisection implied vol for Black–76 with dynamic upper bracket expansion."""
    if not (pd.notna(price) and pd.notna(F) and pd.notna(K) and pd.notna(tau) and pd.notna(cp)):
        return np.nan
    if not (price > 0 and F > 0 and K > 0 and tau > 0 and cp in (1, -1)):
        return np.nan

    plo = b76_price(F, K, tau, lo, cp, df)
    hi = hi_start
    phi_ = b76_price(F, K, tau, hi, cp, df)
    tries = 0
    while (not np.isfinite(phi_) or phi_ < price) and hi < hi_cap and tries < 20:
        hi *= 2.0
        phi_ = b76_price(F, K, tau, hi, cp, df)
        tries += 1

    if not np.isfinite(plo) or not np.isfinite(phi_) or plo > price or phi_ < price:
        return np.nan

    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        pm = b76_price(F, K, tau, m, cp, df)
        if not np.isfinite(pm): return np.nan
        if abs(pm - price) < tol: return max(m, lo)
        if pm > price: b = m
        else: a = m
    return max(0.5 * (a + b), lo)


def fill_iv_from_prices(df: pd.DataFrame) -> pd.Series:
    """Back out IV from best available observed price per row: BBO mid → settlement → closeprice → cleaned_close.
    Safe if required columns are missing (returns all-NaN series)."""
    # Need these to invert; otherwise return NaNs
    need = ["F", "K", "tau", "cp"]
    if not all(c in df.columns for c in need):
        return pd.Series(np.nan, index=df.index, name="iv_from_price")

    cand = pd.Series(index=df.index, dtype=float)
    sources = []
    if "bbo_mid" in df.columns: sources.append(df["bbo_mid"])
    if "settlementprice" in df.columns: sources.append(df["settlementprice"])
    if "closeprice" in df.columns: sources.append(pd.to_numeric(df["closeprice"], errors="coerce"))
    if "cleaned_close" in df.columns: sources.append(pd.to_numeric(df["cleaned_close"], errors="coerce"))
    if sources:
        cand = sources[0].copy()
        for s in sources[1:]:
            cand = cand.where(cand.notna(), s)

    out = []
    for p, F, K, tau, cp in zip(cand, df.get("F"), df.get("K"), df.get("tau"), df.get("cp")):
        out.append(impvol_b76(p, F, K, tau, cp) if pd.notna(p) else np.nan)
    return pd.Series(out, index=df.index, name="iv_from_price")


# ---------- hierarchical IV smoothing ----------

def build_iv_surface(df: pd.DataFrame) -> pd.Series:
    """Build iv_hat with hierarchical fallbacks: (date,expiration,cp,lm_bucket) → (date,expiration,cp) → (date,cp) → (date) → global median."""
    base_iv = df["iv"].copy()
    if "iv_from_price" in df.columns:
        base_iv = base_iv.fillna(df["iv_from_price"])

    ivh = base_iv.copy()
    # 1) (date, expiration, cp, lm_bucket)
    g = df.groupby(["date","expiration","cp","lm_bucket"])['iv']
    ivh = ivh.fillna(g.transform("median"))
    # 2) (date, expiration, cp)
    g = df.groupby(["date","expiration","cp"])['iv']
    ivh = ivh.fillna(g.transform("median"))
    # 3) (date, cp)
    g = df.groupby(["date","cp"])['iv']
    ivh = ivh.fillna(g.transform("median"))
    # 4) (date)
    g = df.groupby(["date"])['iv']
    ivh = ivh.fillna(g.transform("median"))
    # 5) global
    glob = ivh.median()
    ivh = ivh.fillna(glob)
    return ivh.clip(lower=0.01, upper=3.00)


def build_proxy_mid(df: pd.DataFrame) -> pd.DataFrame:
    # model prices (if inputs available)
    if all(c in df.columns for c in ["F","K","tau","cp"]):
        px_ivhat = [b76_price(F,K,t,ivh,cp)  for F,K,t,ivh,cp in zip(df["F"],df["K"],df["tau"],df["iv_hat"],df["cp"])]
        px_iv    = [b76_price(F,K,t,iv, cp)  for F,K,t,iv, cp in zip(df["F"],df["K"],df["tau"],df.get("iv"),df["cp"])]
    else:
        px_ivhat = px_iv = [np.nan]*len(df)

    s_bbo    = df.get("bbo_mid", pd.Series(np.nan, index=df.index))
    s_ivhatm = pd.Series(px_ivhat, index=df.index)
    s_ivm    = pd.Series(px_iv,    index=df.index)
    s_settle = df.get("settlementprice", pd.Series(np.nan, index=df.index))

    proxy = s_bbo.copy()
    proxy = proxy.where(proxy.notna(), s_ivhatm)
    proxy = proxy.where(proxy.notna(), s_ivm)
    proxy = proxy.where(proxy.notna(), s_settle)

    # Tag price source in priority order
    src = pd.Series(np.nan, index=df.index, dtype=object)
    m = s_bbo.notna();                  src.loc[m] = "BBO"
    m = src.isna() & s_ivhatm.notna();  src.loc[m] = "MODEL@IV_hat"
    m = src.isna() & s_ivm.notna();     src.loc[m] = "MODEL@IV"
    m = src.isna() & s_settle.notna();  src.loc[m] = "SETTLE"

    df["proxy_mid"] = proxy
    df["px_source"] = src
    return df




def model_greeks(df: pd.DataFrame, r: float = 0.0) -> pd.DataFrame:
    need = ["F","K","tau","cp","iv_hat"]
    if all(c in df.columns for c in need):
        G = b76_greeks_vec(df["F"], df["K"], df["tau"], df["iv_hat"], df["cp"], r=r)
        for k, v in G.items():
            df[k+"_model"] = v
    return df

def ensure_model_greeks(df: pd.DataFrame, r: float = 0.0) -> pd.DataFrame:
    """Ensure *_model greek columns exist. Compute via Black-76 if possible, else fall back to vendor greeks."""
    need = ["F","K","tau","cp","iv_hat"]
    if all(c in df.columns for c in need):
        try:
            G = b76_greeks_vec(df["F"], df["K"], df["tau"], df["iv_hat"], df["cp"], r=r)
            for k, v in G.items():
                col = k + "_model"
                if col not in df.columns or df[col].isna().all():
                    df[col] = v
        except Exception:
            pass
    for k in ("delta","gamma","vega","theta"):
        col = k + "_model"
        if col not in df.columns and k in df.columns:
            df[col] = df[k]
    return df


# ===================== daily loaders =====================

def load_daily_csv(path: str|None, date_col: str = "date", cols: list[str]|None = None) -> pd.DataFrame|None:
    if not path: return None
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{path}: missing date column '{date_col}'")
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    if cols:
        keep = [c for c in cols if c in df.columns] + [date_col]
        df = df[keep]
    return df.sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)


def merge_futures(wrk: pd.DataFrame, fut_csv: str, date_col: str = "date", price_col: str = "close",
                  forward_fill: bool = False, max_ffill: int = 3):
    fut = pd.read_csv(fut_csv, parse_dates=[date_col])
    fut[date_col] = pd.to_datetime(fut[date_col]).dt.normalize()
    fut = fut.sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)
    if price_col not in fut.columns:
        raise ValueError(f"Futures CSV missing '{price_col}'")
    fut = fut.rename(columns={price_col:"close"})
    wrk = wrk.sort_values("date")
    wrk["F"] = wrk["date"].map(fut["close"])
    if forward_fill:
        wrk["F"] = wrk["F"].ffill(limit=max_ffill)
    return wrk, fut

# ===================== ATM const‑maturity & RV helpers =====================

def build_atm_const_maturity(wrk: pd.DataFrame, target_days: float = 2.0) -> pd.Series:
    """
    Per-date, compute an ATM (|logM| small) IV aggregated near a target maturity (in days).
    Weight options by inverse |tau_days - target_days|, and take median within the top weights.
    """
    df = wrk.copy()
    df = df.loc[df["iv_hat"].notna() & df["tau"].notna() & df["logM"].notna()]
    if df.empty:
        return pd.Series(dtype=float)

    df["tau_days"] = df["tau"] * 365.0
    df["w"] = 1.0 / (1.0 + (df["tau_days"] - float(target_days)).abs())
    # restrict to near-ATM buckets
    df = df.loc[df["lm_bucket"].abs() <= 1].copy()

    # normalize weights per date
    df["w"] = df.groupby("date")["w"].transform(lambda x: x / x.max())
    # top-quantile weights per date then median
    q = df.groupby("date")["w"].transform(lambda s: s.quantile(0.8))
    sel = df["w"] >= q
    atm = df.loc[sel].groupby("date")["iv_hat"].median().sort_index()
    return atm


def compute_option_level_rv_past(paired: pd.DataFrame, fut_df: pd.DataFrame) -> pd.Series:
    """Annualized realized vol over the *previous* DTE days ending at the option date (inclusive). Leakage‑safe."""
    if "close" not in fut_df.columns: return pd.Series(np.nan, index=paired.index)
    fut_sorted = fut_df.sort_index()
    fut_sorted["logret"] = np.log(fut_sorted["close"]).diff()
    fut_sorted["r2"] = fut_sorted["logret"]**2
    csum = fut_sorted["r2"].fillna(0).cumsum()
    dates = fut_sorted.index.to_numpy()

    pos = np.searchsorted(dates, pd.to_datetime(paired["date"]).to_numpy(), side="left")
    out = np.empty(len(paired)); out[:] = np.nan
    dte = np.maximum(paired["dte"].fillna(0).to_numpy().astype(int), 1)
    for i, (p, h) in enumerate(zip(pos, dte)):
        if p < 0 or p >= len(dates): continue
        a = max(1, p - h + 1)
        b = p
        if b < a: continue
        s = csum.iloc[b] - csum.iloc[a-1]
        heff = (b - a + 1)
        if heff <= 0: continue
        out[i] = np.sqrt((s / heff) * 252.0)
    return pd.Series(out, index=paired.index)

# ===================== daily predictors (S/T features) =====================

def _add_changes_and_zscores(df: pd.DataFrame, cols: Iterable[str], win: int = 21) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns: continue
        df[c+"_chg"] = df[c].diff()
        roll = df[c].rolling(win, min_periods=max(5, win//3))
        mu, sd = roll.mean(), roll.std(ddof=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            df[c+"_z"] = (df[c] - mu) / sd
    return df


def build_daily_predictors_df(
    fut_daily: pd.DataFrame,
    spot_daily: pd.DataFrame|None = None,
    bitw_daily: pd.DataFrame|None = None,
    public_daily: pd.Series|None = None,
    rtrs_daily: pd.DataFrame|None = None,
    reddit_daily: pd.DataFrame|None = None,
    # funding_daily: pd.Series|None = None,  # TODO: add if available
) -> pd.DataFrame:
    out = pd.DataFrame(index=fut_daily.index.copy())
    out["logret"] = np.log(fut_daily["close"]).diff()
    out["r2"] = out["logret"]**2
    out["rv"] = out["r2"].rolling(window=7, min_periods=7).sum()
    out["realskew"] = out["logret"].rolling(window=21, min_periods=21).apply(fisher_skew, raw=True)
    try:
        tmp = fut_daily[["high","low","close"]].copy()
        out["baspread"] = corwin_schultz_spread(tmp, overnight_adjust=True)
    except Exception:
        out["baspread"] = np.nan
    if spot_daily is not None and "close" in spot_daily.columns:
        pfb = fut_daily[["close"]].rename(columns={"close":"fut_close"}).join(
            spot_daily[["close"]].rename(columns={"close":"spot_close"}), how="left"
        )
        out["pfb"] = pfb["fut_close"] - pfb["spot_close"]
    if bitw_daily is not None and "close" in bitw_daily.columns:
        # using level as a simple trend proxy; we will add change/z as well
        out["indmom"] = bitw_daily["close"]
    if public_daily is not None:
        out["public"] = public_daily
    if rtrs_daily is not None:
        for k in ["positive","negative","neutral"]:
            col = f"rtrs_{k[:3]}"
            out[col] = rtrs_daily[k] if k in rtrs_daily.columns else np.nan
        out["rtrs_total"] = out[[c for c in out.columns if c.startswith("rtrs_") and len(c)==8]].sum(axis=1)
    if reddit_daily is not None:
        for k in ["positive","negative","neutral"]:
            col = f"reddit_{k[:3]}"
            out[col] = reddit_daily[k] if k in reddit_daily.columns else np.nan
        out["reddit_total"] = out[[c for c in out.columns if c.startswith("reddit_") and len(c)==10]].sum(axis=1)

    # Calendar dummies
    idx = out.index
    out["is_weekend"] = (idx.weekday >= 5).astype(int)
    out["is_friday"] = (idx.weekday == 4).astype(int)

    # Changes + z‑scores (leakage‑safe: only up to t)
    out = _add_changes_and_zscores(out, [
        "rv","baspread","pfb","indmom","realskew","public",
        "rtrs_pos","rtrs_neg","rtrs_neu","rtrs_total",
        "reddit_pos","reddit_neg","reddit_neu","reddit_total",
    ])
    return out.sort_values(out.index.name or "date")

# ===================== smile/term and OI/positioning helpers =====================

def build_smile_and_term_features_per_date(wrk: pd.DataFrame, short_days: float = 2.0, long_days: float = 7.0,
                                           skew_bucket: int = 1) -> pd.DataFrame:
    """Return a per‑date frame with ATM IV at two horizons and a simple smile slope proxy.
    - ATM_short, ATM_long from near‑ATM buckets around requested target days.
    - smile_slope ≈ median IV(OTM put @ bucket=−skew_bucket) − median IV(OTM call @ bucket=+skew_bucket)
    """
    out = pd.DataFrame(index=sorted(wrk["date"].dropna().unique()))
    atm_s = build_atm_const_maturity(wrk, target_days=short_days)
    atm_l = build_atm_const_maturity(wrk, target_days=long_days)
    out["atm_iv_short"] = atm_s
    out["atm_iv_long"] = atm_l
    out["atm_term_slope"] = out["atm_iv_short"] - out["atm_iv_long"]

    # Smile: pick near target short maturity and cp‑specific OTM buckets
    df = wrk.copy()
    df = df.loc[df["iv_hat"].notna() & df["tau"].notna() & df["logM"].notna() & df["cp"].notna()]
    if df.empty:
        out["smile_slope"] = np.nan
        return out
    df["tau_days"] = df["tau"] * 365.0
    df["w"] = 1.0 / (1.0 + (df["tau_days"] - float(short_days)).abs())
    df = df[df["lm_bucket"].abs() <= max(3, skew_bucket+1)]

    def _cp_bucket_median(gdf, cp_val, bucket_val):
        sel = (gdf["cp"]==cp_val) & (gdf["lm_bucket"]==bucket_val)
        if sel.any():
            x = gdf.loc[sel]
            wq = x["w"] >= x["w"].quantile(0.8)
            return x.loc[wq, "iv_hat"].median()
        return np.nan

    smiles = []
    for d, gdf in df.groupby("date"):
        pc = _cp_bucket_median(gdf, -1, -abs(skew_bucket))  # OTM puts (K>S ⇒ logM<0 ⇒ bucket negative)
        cc = _cp_bucket_median(gdf, +1, +abs(skew_bucket))  # OTM calls (K<S ⇒ logM>0 ⇒ bucket positive)
        smiles.append((d, pc - cc if (pd.notna(pc) and pd.notna(cc)) else np.nan))
    if smiles:
        s = pd.Series({d:v for d,v in smiles})
        out["smile_slope"] = s
    else:
        out["smile_slope"] = np.nan
    return out


def build_oi_positioning_per_expiry(wrk: pd.DataFrame, contract_multiplier: float) -> pd.DataFrame:
    """Per (date, expiration) aggregates: OI concentration, PCR, distance to OI wall, GEX proxy.
    Returns a frame indexed by (date, expiration) with columns to be mapped to rows.
    """
    need = ["date","expiration","K","openinterest"]
    if not all(c in wrk.columns for c in need):
        idx = pd.MultiIndex.from_tuples([], names=["date","expiration"])
        return pd.DataFrame(index=idx)

    df = wrk.copy()
    df = df.dropna(subset=["date","expiration","K"])
    # fill OI NaNs with 0 for aggregation
    df["openinterest"] = df["openinterest"].fillna(0.0)

    # median F per (date,expiration) to compute distances
    F_med = df.groupby(["date","expiration"])["F"].median()

    # OI by strike & totals
    grp = df.groupby(["date","expiration","K"]).agg({"openinterest":"sum"}).reset_index()
    tot = grp.groupby(["date","expiration"])['openinterest'].sum().rename("oi_total")
    grp = grp.merge(tot, on=["date","expiration"], how="left")
    grp["share"] = np.where(grp["oi_total"]>0, grp["openinterest"] / grp["oi_total"], np.nan)

    # OI concentration (Herfindahl) & wall
    herf = grp.groupby(["date","expiration"])['share'].apply(lambda s: np.nansum(s.values**2)).rename("oi_herf")
    # top‑N strikes by OI and nearest to F
    def _nearest_wall(sub, d, e):
        sub = sub.copy().sort_values("openinterest", ascending=False).head(5)
        # median futures at this (date, expiration)
        f = F_med.loc[(d, e)] if (d, e) in F_med.index else np.nan
        if not pd.notna(f):
            return pd.Series({"K_wall": np.nan, "dist_to_wall": np.nan})
        k_near = sub.assign(dist=(sub["K"] - f).abs()).sort_values("dist").iloc[0]["K"]
        signed = f - k_near
        return pd.Series({"K_wall": k_near, "dist_to_wall": signed / f})

    # iterate groups to avoid pandas include_groups pitfalls
    _wall_rows = []
    for (d, e), _sub in grp.groupby(["date", "expiration"], sort=False):
        ser = _nearest_wall(_sub.drop(columns=["date","expiration"], errors="ignore"), d, e)
        ser["date"] = d
        ser["expiration"] = e
        _wall_rows.append(ser)

    wall = pd.DataFrame(_wall_rows).set_index(["date","expiration"])

    # Put/Call imbalance (by OI)
    if "cp" in df.columns:
        pc = df.groupby(["date","expiration","cp"])['openinterest'].sum().unstack("cp")
        p = pc.get(-1, pd.Series(dtype=float))
        c = pc.get(+1, pd.Series(dtype=float))
        pcr = (p / (p + c)).rename("oi_put_frac")
    else:
        pcr = pd.Series(dtype=float, name="oi_put_frac")

    # Dealer‑style gamma exposure proxy
    # gex ≈ sum(gamma_model × OI × S^2 × contract_multiplier)
    if "gamma_model" in df.columns:
        df_g = df.copy()
        df_g["dgamma_notional"] = df_g["gamma_model"] * (df_g["F"]**2) * contract_multiplier
        gex = df_g.groupby(["date","expiration"])['dgamma_notional'].sum().rename("gex_proxy")
    else:
        gex = pd.Series(dtype=float, name="gex_proxy")

    out = pd.concat([herf, wall, pcr, gex], axis=1)
    return out

# ===================== returns + per‑row features =====================

def compute_returns_and_features(
    wrk: pd.DataFrame, fut_df: pd.DataFrame,
    daily_feats: pd.DataFrame | None,
    outdir: str, multiplier: float,
    consecutive_only: bool, min_price: float, drop_leq_ticks: float,
    winsorize_q: float, trim_q: float, force_model_delta: bool,
    atm_target_days: float,
    consistent_model_pairs_only: bool = False,
    strict_bbo_only: bool = False,
    max_rel_spread: float = 0.05,
    compute_debiased_return: bool = False,
    # --- NEW: target scaling controls ---
    target_scale_kind: str = "none",     # "none" | "vega" | "gamma"
    target_scale_abs: bool = True,
    target_scale_floor: float = 1e-6,
    cfg_header=None
):
    key = infer_option_key(wrk)
    wrk = wrk.sort_values(key+["date"])
    g = wrk.groupby(key, dropna=False, sort=False)

    wrk["date_next"]      = g["date"].shift(-1)
    wrk["proxy_mid_next"] = g["proxy_mid"].shift(-1)
    
    if "px_source" in wrk.columns:
        wrk["px_source_next"] = g["px_source"].shift(-1)
    if "opt_spread_rel" in wrk.columns:
        wrk["opt_spread_rel_next"] = g["opt_spread_rel"].shift(-1)
    if "bid" in wrk.columns:
        wrk["bid_next"] = g["bid"].shift(-1)
    if "offer" in wrk.columns:
        wrk["offer_next"] = g["offer"].shift(-1)

    wrk["H"]      = wrk["Px"] if "Px" in wrk.columns else wrk.get("F")
    wrk["H_next"] = g["H"].shift(-1)

    # intrinsic at expiry if next is expiration and price missing
    if all(c in wrk.columns for c in ("expiration","K","cp")):
        mask = wrk["proxy_mid_next"].isna() & wrk["date_next"].notna() & (wrk["date_next"]==wrk["expiration"])
        intrinsic = np.maximum(wrk["cp"]*(wrk["H_next"]-wrk["K"]), 0.0)
        wrk.loc[mask, "proxy_mid_next"] = intrinsic[mask]

    ok = wrk["proxy_mid"].notna() & wrk["proxy_mid_next"].notna() & wrk["H"].notna() & wrk["H_next"].notna()
    paired = wrk.loc[ok].copy()

    # Ensure model greeks exist (recompute or fall back to vendor greeks)
    paired = ensure_model_greeks(paired, r=0.0)

    # consecutive filter & gap dummies
    paired["days_gap"] = (paired["date_next"] - paired["date"]).dt.days
    if consecutive_only:
        paired = paired.loc[paired["days_gap"] == 1].copy()
    paired["gap_gt1"] = (paired["days_gap"] > 1).astype(int)

    # --- Delta selection with vendor sentinel handling ---
    paired["delta_vendor"]      = _clean_delta_series(paired, "delta")
    paired["delta_vendor_next"] = _clean_delta_series(paired, "delta_next")
    paired["delta_model"]       = _clean_delta_series(paired, "delta_model")
    paired["delta_model_next"]  = _clean_delta_series(paired, "delta_model_next")

    # Prefer vendor unless forced; fall back to the other if NaN
    if force_model_delta:
        d0 = paired["delta_model"].where(paired["delta_model"].notna(), paired["delta_vendor"])
        d1 = paired["delta_model_next"].where(paired["delta_model_next"].notna(), paired["delta_vendor_next"])
    else:
        d0 = paired["delta_vendor"].where(paired["delta_vendor"].notna(), paired["delta_model"])
        d1 = paired["delta_vendor_next"].where(paired["delta_vendor_next"].notna(), paired["delta_model_next"])

    paired["delta_for_hedge"]      = d0
    paired["delta_for_hedge_next"] = d1

    # Drop rows that still lack delta on either day
    mask_delta_ok = paired["delta_for_hedge"].notna()   # only need t-day delta

    dropped = int((~mask_delta_ok).sum())
    if dropped:
        print(f"[delta] dropping {dropped} rows due to missing delta (after sentinel cleaning)")
    paired = paired.loc[mask_delta_ok].copy()


    paired["option_pl"] = paired["proxy_mid_next"] - paired["proxy_mid"]
    paired["hedge_pl"]  = paired["delta_for_hedge"] * (paired["H_next"] - paired["H"])
    paired["pl_dh"]     = (paired["option_pl"] - paired["hedge_pl"]) * float(multiplier)

    paired["ret_dh"] = np.where(paired["proxy_mid"] >= float(min_price),
                                paired["pl_dh"] / (paired["proxy_mid"] * float(multiplier)),
                                np.nan)
    
    paired["ret_dh_w"] = paired["ret_dh"]
    if winsorize_q and 0 < winsorize_q < 0.5:
        ql, qh = paired["ret_dh"].quantile([winsorize_q, 1 - winsorize_q])
        paired["ret_dh_w"] = paired["ret_dh"].clip(lower=ql, upper=qh)
    if trim_q and 0 < trim_q < 0.5:
        ql, qh = paired["ret_dh"].quantile([trim_q, 1 - trim_q])
        paired = paired.loc[(paired["ret_dh"] >= ql) & (paired["ret_dh"] <= qh) | (paired["ret_dh"].isna())].copy()

    #         # ----- OPTIONAL: de-bias mid→mid return (add back half the rel. spread both days)
    # if compute_debiased_return and all(c in paired.columns for c in ["opt_spread_rel","opt_spread_rel_next"]):
    #     paired["ret_dh_w_adj"] = paired["ret_dh_w"] + 0.5*(paired["opt_spread_rel"] + paired["opt_spread_rel_next"])

    if compute_debiased_return:
        # placeholder for XS de-mean; will recompute after filtering
        paired["ret_dh_w_adj"] = paired["ret_dh_w"]
    else:
        paired["ret_dh_w_adj"] = paired["ret_dh_w"]


    # ----- OPTIONAL: sample restrictions (do these AFTER ret_dh_w exists)
    if consistent_model_pairs_only and {"px_source","px_source_next"}.issubset(paired.columns):
        paired = paired[(paired["px_source"]=="MODEL@IV_hat") & (paired["px_source_next"]=="MODEL@IV_hat")].copy()

    if strict_bbo_only and {"px_source","px_source_next","opt_spread_rel","opt_spread_rel_next"}.issubset(paired.columns):
        cond = (paired["px_source"]=="BBO") & (paired["px_source_next"]=="BBO") \
            & (paired["opt_spread_rel"]<=max_rel_spread) & (paired["opt_spread_rel_next"]<=max_rel_spread)
        if {"bid","offer","bid_next","offer_next"}.issubset(paired.columns):
            cond &= (paired["bid"]>0) & (paired["offer"]>0) & (paired["bid_next"]>0) & (paired["offer_next"]>0)
        paired = paired.loc[cond].copy()

    # --- Greek-based target scaling (write both variants) ---
    def _build_target_scale(df, col, use_abs=True, floor=1e-6):
        import numpy as np, pandas as pd
        if col not in df.columns:
            return pd.Series(1.0, index=df.index)  # graceful fallback
        s = pd.to_numeric(df[col], errors="coerce")
        if use_abs:
            s = s.abs()
        # avoid division blow-ups
        s = s.clip(lower=float(floor)).fillna(float(floor)).replace([np.inf, -np.inf], float(floor))
        return s

    # scales (vega abs; gamma as dollar-gamma = |Γ|·F²; adjust floors if needed)
    paired["target_scale_vega"]  = _build_target_scale(paired, "vega",  use_abs=True, floor=1e-6)
    paired["target_scale_gamma"] = (_build_target_scale(paired, "gamma", use_abs=True, floor=1e-12) * (paired["F"].astype(float)**2)).astype(float).clip(lower=1e-6)

    # training targets (we keep the unscaled debiased target too)
    # Recompute XS de-mean by date AFTER all sample filters so per-day means are ~0
    if compute_debiased_return:
        paired["ret_dh_w_adj"] = paired["ret_dh_w"] - paired.groupby("date")["ret_dh_w"].transform("mean")

    paired["ret_dh_w_adj"]    = paired["ret_dh_w_adj"].astype(float)
    paired["ret_target_vega"]  = paired["ret_dh_w_adj"] / paired["target_scale_vega"]
    paired["ret_target_gamma"] = paired["ret_dh_w_adj"] / paired["target_scale_gamma"]

    print(
        "[target] wrote scaled targets:",
        "median vega scale =", float(paired["target_scale_vega"].median()),
        "median gamma scale =", float(paired["target_scale_gamma"].median()),
    )


    # Sanity: per-date mean of adjusted returns should be ~0 by construction
    try:
        _dm = paired.groupby("date")["ret_dh_w_adj"].mean()
        print("[sanity] day-mean ret_dh_w_adj (abs mean, max):",
              float(_dm.abs().mean()), float(_dm.abs().max()))
    except Exception as _e:
        print("[sanity] day-mean check failed:", str(_e))

    # === predictors on option rows ===
    paired["corp"] = (paired["callput"].astype(str).str.upper() == "C").astype(int)
    S_under = paired.get("F", paired.get("Px"))
    paired["moneyness"] = np.where(paired["corp"]==1, S_under/paired["K"], paired["K"]/S_under)
    # $gamma per 1% move approx; fall back to vendor gamma if needed
    _gamma_src = paired.get("gamma_model", paired.get("gamma"))
    paired["gamma_scaled"] = _gamma_src * S_under / 100.0  # $gamma per 1% move approx


    # DTE bucket dummies (1/2/3+)
    paired["dte_1"] = (paired["dte"]==1).astype('Int64')
    paired["dte_2"] = (paired["dte"]==2).astype('Int64')
    paired["dte_3p"] = (paired["dte"]>=3).astype('Int64')

    # Constant‑maturity ATM IV per date and its changes/term slope & smile
    atm_by_date = build_atm_const_maturity(wrk, target_days=atm_target_days)
    term_smile = build_smile_and_term_features_per_date(wrk, short_days=atm_target_days, long_days=7.0)
    paired["atm_iv"] = paired["date"].map(atm_by_date)
    paired["atm_iv_chg"] = paired["date"].map(term_smile["atm_iv_short"].diff())
    paired["atm_term_slope"] = paired["date"].map(term_smile["atm_term_slope"])
    paired["smile_slope"] = paired["date"].map(term_smile["smile_slope"])

    # Leakage‑safe past RV over each option’s DTE + IV–RV gap features
    rv_ann_past = compute_option_level_rv_past(paired, fut_df)
    paired["rv_ann_past"]   = rv_ann_past
    paired["ivrv_pred"]     = paired["atm_iv"] - paired["rv_ann_past"]
    paired["ivrv_ratio_pred"] = paired["atm_iv"] / paired["rv_ann_past"]
    paired.loc[paired["rv_ann_past"] <= 1e-12, ["ivrv_pred","ivrv_ratio_pred"]] = np.nan

    # Expected move and convexity interaction proxies
    with np.errstate(invalid='ignore'):
        expmove_pct = paired["atm_iv"] * np.sqrt(paired["tau"])  # ≈ E[|return|]
        paired["convexity_proxy"] = paired["gamma_scaled"] * (expmove_pct ** 2)
        # Vega × change in ATM IV
        paired["vega_ivchg_proxy"] = paired["vega_model"] * paired["atm_iv_chg"]

    # Option‑level relative spread and a high‑spread flag (per date decile)
    if "opt_spread_rel" in paired.columns:
        paired["opt_spread_rel"] = paired["opt_spread_rel"].clip(lower=0)
        q90 = paired.groupby("date")["opt_spread_rel"].transform(lambda s: s.quantile(0.90))
        paired["opt_spread_hi"] = (paired["opt_spread_rel"] >= q90).astype('Int64')

    # Map daily features (levels + chg + z)
    if daily_feats is not None and not daily_feats.empty:
        cols_to_map = [
            # levels
            "logret","baspread","rv","pfb","indmom","realskew","public",
            "rtrs_pos","rtrs_neg","rtrs_neu","rtrs_total","reddit_pos","reddit_neg","reddit_neu","reddit_total",
            # changes
            "rv_chg","baspread_chg","pfb_chg","indmom_chg","realskew_chg","public_chg",
            "rtrs_pos_chg","rtrs_neg_chg","rtrs_neu_chg","rtrs_total_chg",
            "reddit_pos_chg","reddit_neg_chg","reddit_neu_chg","reddit_total_chg",
            # z‑scores
            "rv_z","baspread_z","pfb_z","indmom_z","realskew_z","public_z",
            "rtrs_pos_z","rtrs_neg_z","rtrs_neu_z","rtrs_total_z",
            "reddit_pos_z","reddit_neg_z","reddit_neu_z","reddit_total_z",
            # calendar
            "is_weekend","is_friday",
        ]
        for col in cols_to_map:
            if col in daily_feats.columns:
                paired[col] = paired["date"].map(daily_feats[col])

    # OI/positioning per (date,expiration)
    oi_pos = build_oi_positioning_per_expiry(wrk, contract_multiplier=multiplier)
    if not oi_pos.empty:
        idx2 = list(zip(paired["date"], paired["expiration"]))
        paired["oi_herf"] = [oi_pos.get("oi_herf").get((d,e), np.nan) for d,e in idx2]
        paired["oi_put_frac"] = [oi_pos.get("oi_put_frac").get((d,e), np.nan) for d,e in idx2]
        paired["dist_to_wall"] = [oi_pos.get("dist_to_wall").get((d,e), np.nan) for d,e in idx2]
        paired["gex_proxy"] = [oi_pos.get("gex_proxy").get((d,e), np.nan) for d,e in idx2]


    print("[pairs] start:", len(paired))
    print("  nonnull mids:", int((paired['proxy_mid'].notna() & paired['proxy_mid_next'].notna()).sum()))
    print("  price>=min_price:", int((paired['proxy_mid'] >= float(min_price)).sum()))
    if 'opt_spread_rel' in paired and 'opt_spread_rel_next' in paired:
        tight = (paired['opt_spread_rel']<=max_rel_spread) & (paired['opt_spread_rel_next']<=max_rel_spread)
        print(f"  spread<= {max_rel_spread:.2f} both days:", int(tight.sum()))
    if 'px_source' in paired and 'px_source_next' in paired:
        print("  BBO both days:", int(((paired['px_source']=='BBO') & (paired['px_source_next']=='BBO')).sum()))
        print("  MODEL@IV_hat both days:", int(((paired['px_source']=='MODEL@IV_hat') & (paired['px_source_next']=='MODEL@IV_hat')).sum()))


    # ===== assemble output =====
    out_cols = [
        # identifiers & target
        "date","expiration","callput","cp","F","K","dte","tau",
        "proxy_mid","proxy_mid_next","px_source","px_source_next",
        "opt_spread_rel","opt_spread_rel_next",
        "ret_dh_w","ret_dh_w_adj",  # only populated if --compute_debiased_return
        "ret_target_vega","ret_target_gamma",
        "target_scale_vega","target_scale_gamma",
        # (optional) raw quotes if you have them:
        "bid","offer","bid_next","offer_next",
        # contract‑level basics
        "corp","moneyness","iv","atm_iv","atm_iv_chg","atm_term_slope","smile_slope",
        "delta_model","vega_model","theta_model","gamma_scaled",
        # leakage‑safe volatility features
        "rv_ann_past","ivrv_pred","ivrv_ratio_pred",
        # convexity/vega interactions
        "convexity_proxy","vega_ivchg_proxy",
        # option microstructure
        "opt_spread_rel","opt_spread_hi",
        # positioning
        "oi_herf","oi_put_frac","dist_to_wall","gex_proxy",
        # daily features (levels, changes, z, calendar)
        "baspread","baspread_chg","baspread_z",
        "rv","rv_chg","rv_z",
        "pfb","pfb_chg","pfb_z",
        "indmom","indmom_chg","indmom_z",
        "realskew","realskew_chg","realskew_z",
        "public","public_chg","public_z",
        "rtrs_pos","rtrs_pos_chg","rtrs_pos_z",
        "rtrs_neg","rtrs_neg_chg","rtrs_neg_z",
        "rtrs_neu","rtrs_neu_chg","rtrs_neu_z",
        "rtrs_total","rtrs_total_chg","rtrs_total_z",
        "reddit_pos","reddit_pos_chg","reddit_pos_z",
        "reddit_neg","reddit_neg_chg","reddit_neg_z",
        "reddit_neu","reddit_neu_chg","reddit_neu_z",
        "reddit_total","reddit_total_chg","reddit_total_z",
        "is_weekend","is_friday","gap_gt1",
        # DTE buckets
        "dte_1","dte_2","dte_3p",
    ]
    existing = [c for c in out_cols if c in paired.columns]
    out = paired[existing].copy()

    # right before building 'diag'
    day_means = paired.groupby("date")["ret_dh_w_adj"].mean()

    diag = pd.DataFrame({
        "metric":[
            "n_input_rows","n_output_rows",
            "n_missing_iv","n_iv_from_price","n_missing_iv_hat","n_missing_atm_iv",
            "mean_ret_dh_w","median_ret_dh_w",
            "mean_ret_dh_w_adj","median_ret_dh_w_adj",
            "avg_abs_day_mean_adj","max_abs_day_mean_adj"
        ],
        "value":[
            len(wrk), len(out),
            int(wrk["iv"].isna().sum()) if "iv" in wrk.columns else -1,
            int(wrk["iv_from_price"].notna().sum()) if "iv_from_price" in wrk.columns else -1,
            int(wrk["iv_hat"].isna().sum()) if "iv_hat" in wrk.columns else -1,
            int(out["atm_iv"].isna().sum()) if "atm_iv" in out.columns else -1,
            float(np.nanmean(out["ret_dh_w"])),
            float(np.nanmedian(out["ret_dh_w"])),
            float(np.nanmean(out["ret_dh_w_adj"])),
            float(np.nanmedian(out["ret_dh_w_adj"])),
            float(np.nanmean(np.abs(day_means))),
            float(np.nanmax(np.abs(day_means)))
        ]
    })


    out_path = os.path.join(outdir, "options_with_returns.csv")
    diag_path = os.path.join(outdir, "diagnostics_returns.csv")
    _write_csv_with_config(out,  out_path,  cfg_header or {})
    if diag is not None:
        _write_csv_with_config(diag, diag_path, cfg_header or {})
    print("\n[RETURNS] Saved:\n ", out_path, "\n ", diag_path)
    print(diag.to_string(index=False))

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", required=True)
    ap.add_argument("--futures", required=True)
    ap.add_argument("--spot", default=None)
    ap.add_argument("--bitw", default=None)
    ap.add_argument("--public", default=None)
    ap.add_argument("--rtrs", default=None)
    ap.add_argument("--reddit", default=None)
    ap.add_argument("--fut_date_col", default="date")
    ap.add_argument("--fut_price_col", default="close")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--multiplier", type=float, default=5.0)


    # returns controls
    ap.add_argument("--build_returns", action="store_true")
    ap.add_argument("--consecutive_only", action="store_true")
    ap.add_argument("--min_price", type=float, default=2.0)
    ap.add_argument("--drop_leq_ticks", type=float, default=2.0)   # (accepted; filtering can be applied inside compute_returns_and_features)
    ap.add_argument("--winsorize", type=float, default=0.01)
    ap.add_argument("--trim", type=float, default=0.00)
    ap.add_argument("--force_model_delta", action="store_true")
    ap.add_argument("--forward_fill_futures", action="store_true")
    ap.add_argument("--max_ffill", type=int, default=3)
    ap.add_argument("--atm_target_days", type=float, default=2.0)  # const-maturity target
    ap.add_argument("--compute_debiased_return", action="store_true",
                help="If set, add ret_dh_w_adj by demeaning ret_dh_w (per date).")
    ap.add_argument("--consistent_model_pairs_only", action="store_true")
    ap.add_argument("--strict_bbo_only", action="store_true")
    ap.add_argument("--max_rel_spread", type=float, default=0.05)

    ap.add_argument("--target_scale_kind", choices=["none", "vega", "gamma"],
                default="none",
                help="Scale the target by this Greek (divide by |Greek|).")
    ap.add_argument("--target_scale_abs", action="store_true",
                    help="Use absolute value of the Greek when scaling (recommended).")
    ap.add_argument("--target_scale_floor", type=float, default=1e-6,
                    help="Floor for the scale to avoid division by ~0.")


    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load & standardize option panel
    raw = pd.read_csv(args.options)
    wrk = standardize_options(raw)

    # 2) Merge futures ONCE (need F for IV inversion / pricing); compute log-moneyness ONCE
    wrk, fut_df = merge_futures(
        wrk,
        args.futures,
        args.fut_date_col,
        args.fut_price_col,
        forward_fill=args.forward_fill_futures,
        max_ffill=args.max_ffill,
    )
    if "K" in wrk.columns and "F" in wrk.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            wrk["logM"] = np.log(wrk["F"] / wrk["K"])
        wrk["lm_bucket"] = (wrk["logM"] / 0.05).round().astype("Int64")

    # 3) IV: invert from prices, then smooth → surface
    wrk["iv_from_price"] = fill_iv_from_prices(wrk)
    wrk["iv_hat"] = build_iv_surface(wrk)

    # 4) Proxy mids & Greeks
    wrk = build_proxy_mid(wrk)
    wrk = model_greeks(wrk, r=0.0)

    # 5) Build daily predictor blocks (futures always present; others optional)
    fut_daily  = load_daily_csv(args.futures, date_col=args.fut_date_col, cols=None)
    spot_daily = load_daily_csv(args.spot,    date_col="date", cols=None) if args.spot else None
    bitw_daily = load_daily_csv(args.bitw,    date_col="date", cols=None) if args.bitw else None

    # Flexible "public" series: use 'value' if present; else first numeric column
    public_daily = None
    if args.public:
        pub = pd.read_csv(args.public, parse_dates=["date"])
        pub["date"] = pd.to_datetime(pub["date"]).dt.normalize()
        if "value" in pub.columns:
            public_daily = (
                pub.sort_values("date")
                   .drop_duplicates("date", keep="last")
                   .set_index("date")["value"]
            )
        else:
            numcols = [c for c in pub.columns if c != "date" and pd.api.types.is_numeric_dtype(pub[c])]
            public_daily = (
                pub.sort_values("date")
                   .drop_duplicates("date", keep="last")
                   .set_index("date")[numcols[0]]
                if numcols else None
            )

    rtrs_daily   = load_daily_csv(args.rtrs,   cols=["positive", "negative", "neutral"]) if args.rtrs else None
    reddit_daily = load_daily_csv(args.reddit, cols=["positive", "negative", "neutral"]) if args.reddit else None

    daily_feats = (
        build_daily_predictors_df(fut_daily, spot_daily, bitw_daily, public_daily, rtrs_daily, reddit_daily)
        if fut_daily is not None else None
    )

    # 6) Build targets + features once, with filters applied ONCE inside
    if args.build_returns:
        compute_returns_and_features(
            wrk.copy(),
            fut_df.copy(),
            daily_feats,
            args.outdir,
            args.multiplier,
            args.consecutive_only,
            args.min_price,
            args.drop_leq_ticks,
            args.winsorize,
            args.trim,
            args.force_model_delta,
            args.atm_target_days,
            consistent_model_pairs_only=args.consistent_model_pairs_only,
            strict_bbo_only=args.strict_bbo_only,
            max_rel_spread=args.max_rel_spread,
            compute_debiased_return=args.compute_debiased_return,
            target_scale_kind=args.target_scale_kind,
            target_scale_abs=args.target_scale_abs,
            target_scale_floor=args.target_scale_floor,
            cfg_header=args
        )
    else:
        print("Nothing to do: add --build_returns")

if __name__ == "__main__":
    main()