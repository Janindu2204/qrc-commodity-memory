"""v2 pipeline core (see v2_prereg.json): range-based variance proxy, extra features, QLIKE-trained linear/GBM models, calibration.

All 'levels' are variance forecasts per bar (mean over the next 24 valid bars). Preliminary, AI-written, not independently re-derived."""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
import hourly_baselines as hb
import mc_lib as mc

OUT = Path(__file__).with_name(os.environ.get("V2_OUT", "v2_preds"))
DAILY = Path(__file__).resolve().parents[2] / "data" / "daily_commod"
H = 24
# FOMC meeting decision days (second day), from memory: UNVERIFIED, used only inside feature set X and ablated
FOMC = ["2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]
ENERGY_WED = {"CL_F", "HO_F"}


def range_variance(d, r):
    rv = np.log(d.high / d.low) ** 2 / (4 * np.log(2))
    gap = np.log(d.open / d.close.shift(1)) ** 2
    ok = d.index.to_series().diff() <= pd.Timedelta("2h")
    return (rv + gap.where(ok, 0.0)).where(np.isfinite(rv))


def fwd_mean(v, h=H):
    return v[::-1].rolling(h, min_periods=int(0.8 * h)).mean()[::-1].shift(-1)


def prev_day_series(idx, name):
    s = pd.read_csv(DAILY / f"{name}_1d.csv", parse_dates=["date"]).set_index("date").close
    dates = pd.DatetimeIndex(idx.normalize())
    pos = s.index.searchsorted(dates, side="left") - 1           # strictly earlier trading day
    pos = np.clip(pos, 0, len(s) - 1)
    return pd.Series(s.values[pos], index=idx)


def prepare_v2(asset):
    S = mc.prepare(asset); d, r, F = S["d"], S["r"], S["F"]; idx = d.index
    rv = range_variance(d, r)
    yfull = np.log(fwd_mean(rv) + 1e-12)
    sq = r ** 2
    feats = pd.DataFrame(index=idx)
    for w in (24, 120, 480):
        feats[f"lvR{w}"] = np.log(rv.rolling(w, min_periods=int(0.8 * w)).mean() + 1e-12)
    lvol = np.log1p(d.volume.astype(float))
    feats["lvolu"] = lvol - lvol.rolling(120, min_periods=60).mean()
    hr = idx.hour.values + idx.minute.values / 60
    feats["h_sin"] = np.sin(2 * np.pi * hr / 24); feats["h_cos"] = np.cos(2 * np.pi * hr / 24)
    feats["fri"] = (idx.dayofweek == 4).astype(float)
    vix, ovx = np.log(prev_day_series(idx, "_VIX")), np.log(prev_day_series(idx, "_OVX"))
    feats["lvix"], feats["lovx"] = vix.values, ovx.values
    feats["dvix5"] = (vix - prev_day_series(idx, "_VIX").shift(120).pipe(np.log)).values
    wed = (idx.dayofweek == 2) & (idx.hour >= 10) & (idx.hour <= 12); thu = (idx.dayofweek == 3) & (idx.hour >= 10) & (idx.hour <= 12)
    feats["eia"] = (wed & (asset in ENERGY_WED)).astype(float); feats["ngst"] = (thu & (asset == "NG_F")).astype(float)
    fd = pd.to_datetime(FOMC); feats["fomc"] = (np.isin(idx.normalize(), fd) & (idx.hour >= 14) & (idx.hour <= 16)).astype(float)
    drv = F[[c for c in F.columns if c.startswith(("er_", "ev_"))]]
    tr, va, te = S["tr"], S["va"], S["te"]
    okrow = np.isfinite(yfull.values) & np.isfinite(feats[["lvR24", "lvR120", "lvR480"]].values).all(1)
    keep = lambda rows: rows[okrow[rows]]
    tr, va, te = keep(tr), keep(va), keep(te)
    sets = {"HAR": ["lvR24", "lvR120", "lvR480"],
            "X": ["lvR24", "lvR120", "lvR480", "lvolu", "h_sin", "h_cos", "fri", "lvix", "lovx", "dvix5", "eia", "ngst", "fomc"],
            "Xnoev": ["lvR24", "lvR120", "lvR480", "lvolu", "h_sin", "h_cos", "fri", "lvix", "lovx", "dvix5"]}
    M = {}
    for k, cols in sets.items():
        A = feats[cols].values.astype(float); mu = np.nanmean(A[tr], 0); sd = np.nanstd(A[tr], 0); sd[sd < 1e-9] = 1.0
        M[k] = np.nan_to_num((A - mu) / sd, nan=0.0)
    A = np.hstack([feats[sets["X"]].values, drv.values]).astype(float); mu = np.nanmean(A[tr], 0); sd = np.nanstd(A[tr], 0); sd[sd < 1e-9] = 1.0
    M["ALL"] = np.nan_to_num((A - mu) / sd, nan=0.0)
    ret_z = ((r.fillna(0.0) - r.fillna(0.0).values[tr].mean()) / (r.fillna(0.0).values[tr].std() + 1e-12)).values
    return dict(asset=asset, d=d, r=r, rv=rv, y=yfull.values, anchor=feats["lvR24"].values, M=M, feats=feats, tr=tr, va=va, te=te,
                ret_z=ret_z, wins=None, S=S, sq=sq)


def qlike_loss(y, eta):
    return float(np.mean(np.exp(y - eta) + eta - y - 1))


def fit_linear_qlike(X, y, anchor, lam):
    """eta = anchor + X w + b minimising mean(exp(y - eta) + eta) + lam |w|^2 (convex)."""
    n, k = X.shape
    def f(p):
        w, b = p[:k], p[k]; eta = anchor + X @ w + b; e = np.exp(y - eta)
        return np.mean(e + eta) + lam * (w @ w), np.r_[((1 - e) @ X) / n + 2 * lam * w, np.mean(1 - e)]
    res = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B")
    return res.x


def predict_linear(p, X, anchor):
    return anchor + X @ p[:-1] + p[-1]


def calib_scale(y_level_val, lev_val):
    """QLIKE-optimal multiplicative scale on validation rows: argmin_c mean(y/(c f) + log(c f)) = mean(y/f)."""
    return float(np.mean(y_level_val / lev_val))


def best_lambda_linear(X, y, anchor, tr, va, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    best = None
    for lam in grid:
        p = fit_linear_qlike(X[tr], y[tr], anchor[tr], lam)
        ev = np.exp(predict_linear(p, X[va], anchor[va])); c = calib_scale(np.exp(y[va]), ev)
        sc = qlike_loss(y[va], np.log(c * ev))
        if best is None or sc < best[0]: best = (sc, lam, p)
    return best
