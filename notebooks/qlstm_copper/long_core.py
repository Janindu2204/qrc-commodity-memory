"""Long-daily copper pipeline core (see long_prereg.json). Preliminary, AI-written, not independently re-derived."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "daily_long"
DATA = ROOT / "flagged" / "HG_F_1d.csv"
DATA2 = ROOT / "flagged_v2" / "HG_F_1d.csv"   # v2 cleaning: roll-suspect 2005-05-27, 2005-11-29; 2004-10-13 only marked
OUT = Path(__file__).with_name("long_preds")
OUT_ROLL = Path(__file__).with_name("long_preds_rolling")   # long_prereg_rolling.json: 1-year blocks, test years ROLL_YEARS
ROLL_YEARS = list(range(2008, 2025))
H = 5
FLOOR_FIT_END = "2006-12-31"
ORIGINS = {2007: 6, 2013: 6, 2019: 6}   # test start year -> test length (years); val = 2 years before; test ends before 2025


def _lag_strict(ser, dates):
    """value of ser from the latest date STRICTLY before each copper date (outside markets close after copper: avoids the close-time artefact)."""
    pos = np.clip(ser.index.searchsorted(dates, side="left") - 1, 0, len(ser) - 1)
    return ser.values[pos]


def _exog(dates):
    rd = lambda n: pd.read_csv(ROOT / f"{n}_1d.csv", parse_dates=["date"]).dropna(subset=["close"]).set_index("date")
    out = {"lvix_l1": _lag_strict(np.log(rd("_VIX").close), dates)}
    for name, nm in (("CL_F", "lcl_l1"), ("DX-Y.NYB", "ldx_l1")):
        e = rd(name); r = np.log(e.high / e.low) ** 2 / (4 * np.log(2)); r = r.where(r > 0)
        out[nm] = _lag_strict(np.log(r.rolling(5, min_periods=3).mean()).ffill(), dates)
    return pd.DataFrame(out, index=dates)


def prepare(version, origin, mv=False, length=None, data=None):
    """version: range|gap (v1: floor, old data file) or range_m|gap_m (v2: zero variance -> missing, v2 data file). mv adds 3 lagged outside inputs."""
    m_mode = version.endswith("_m"); base = version[:-2] if m_mode else version
    d = (pd.read_csv(DATA2 if m_mode else DATA, parse_dates=["date"]).set_index("date")) if data is None else data.copy()
    rng = np.log(d.high / d.low) ** 2 / (4 * np.log(2))
    rv = rng + d.gap_clean.pow(2).fillna(0) if base == "gap" else rng
    rv = rv.where(np.isfinite(rv))
    floored = (rv <= 0).values
    if m_mode:
        rv = rv.where(rv > 0)                                  # zero variance -> missing; windows skip it
        mp5, mp22, mpf = 3, 14, 3
        floor = float("nan")
    else:
        floor = float(rv[(rv > 0) & (rv.index <= FLOOR_FIT_END)].quantile(0.01))
        rv = pd.Series(np.where(rv.notna() & (rv <= 0), floor, rv), index=d.index)
        mp5, mp22, mpf = 5, 22, H
    y = np.log(rv[::-1].rolling(H, min_periods=mpf).mean()[::-1].shift(-1)).values
    f = pd.DataFrame(index=d.index)
    f["lvR5"] = np.log(rv.rolling(5, min_periods=mp5).mean()); f["lvR1"] = np.log(rv).fillna(f["lvR5"]); f["lvR22"] = np.log(rv.rolling(22, min_periods=mp22).mean())
    f = f[["lvR1", "lvR5", "lvR22"]]
    if mv:
        ex = _exog(d.index)
        for c in ex.columns: f[c] = ex[c].values
    n = len(d); idx = np.arange(n); dt = d.index
    t0, t1 = pd.Timestamp(f"{origin}-01-01"), pd.Timestamp(f"{origin + (ORIGINS[origin] if length is None else length)}-01-01")
    v0 = pd.Timestamp(f"{origin - 2}-01-01")
    okrow = np.isfinite(y) & np.isfinite(f.values).all(1)
    tr = idx[(dt < v0) & okrow][:-H]; va = idx[(dt >= v0) & (dt < t0) & okrow][:-H]; te = idx[(dt >= t0) & (dt < t1) & okrow]
    A = f.values; mu, sd = np.nanmean(A[tr], 0), np.nanstd(A[tr], 0)
    M = np.nan_to_num((A - mu) / sd, nan=0.0)
    flag = (floored | d.is_ambiguous.values.astype(bool)).astype(float)
    return dict(d=d, rv=rv, y=y, anchor=f["lvR5"].values, M=M, tr=tr, va=va, te=te, floor=floor, flag=flag, n_floored=int(floored.sum()))


def qlike_vec(f, ylev):
    r = ylev / f
    return r - np.log(r) - 1
