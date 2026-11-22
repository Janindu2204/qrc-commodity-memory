"""One intraday block of long_prereg_lin.json (fix A, stronger benchmark). Usage: python lin_run.py GRID YEAR.
Rows, target, benchmark, splits and purge: dom_run.py code copied unchanged above the model section (checked by diff).
LIN200: HAR + a ridge readout (hourly_databento_qrc.ridge_fit, penalty chosen on validation as for every reservoir) on the last 200 values of
ln RV (missing values hold the last value, standardised with training rows), fitted to the HAR residual. LIN50 the same with 50 values.
Writes long_preds_lin/<GRID>/<YEAR>.npz. Scores nothing."""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import hourly_databento_qrc as qrc, dom_lib as dl

GRID, YEAR = sys.argv[1], int(sys.argv[2]); DRY = os.environ.get("DRY") == "1"; NRES = 2 if DRY else 10
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent
PURGE_H = 24; SEQ = 10
LONG = {"H": (24, 120), "M": (96, 480), "F": (288, 1440)}[GRID]

z = np.load(ROOT / "data" / "hf" / f"{GRID}.npz"); t = pd.to_datetime(z["t"]); N = len(t)
ret, lrv0, lvol, hs, hc, adj, rvpos, hodbin = (z[k] for k in ("ret", "lrv", "lvol", "hs", "hc", "adj", "rvpos", "hodbin"))
lrv = np.where(rvpos, lrv0, np.nan)
S = pd.Series(lrv)
l_a = S.rolling(LONG[0], min_periods=LONG[0] // 2).mean().to_numpy(); l_b = S.rolling(LONG[1], min_periods=LONG[1] // 2).mean().to_numpy()
CH = np.stack([ret, lrv, lvol, hs, hc], 1)

yr_bin = np.r_[t.year.values[1:], 0]
pos = np.arange(SEQ - 1, N - 1); pos = pos[(yr_bin[pos] >= YEAR - 4) & (yr_bin[pos] <= YEAR)]
okw = np.ones(len(pos), bool)
for k in range(SEQ): okw &= np.isfinite(CH[pos - k]).all(1)
okw &= adj[pos + 1] & rvpos[pos + 1] & np.isfinite(l_a[pos]) & np.isfinite(l_b[pos])
pos = pos[okw]; tt = t[pos + 1]; yr = tt.year.values
y = lrv[pos + 1]
Hm = np.column_stack([np.ones(len(pos)), lrv[pos], np.mean([lrv[pos - k] for k in range(SEQ)], 0), l_a[pos], l_b[pos],
                      pd.get_dummies(hodbin[pos + 1]).reindex(columns=range(24), fill_value=0).astype(float).values[:, 1:]])

te_m = yr == YEAR; va_m = yr == YEAR - 1; tr_m = (yr >= YEAR - 4) & (yr <= YEAR - 2)
w = pd.Timedelta(hours=PURGE_H)
tr_m &= tt < (tt[va_m].min() - w) if va_m.any() else False
va_m &= tt < (tt[te_m].min() - w)
tr, va, te = np.where(tr_m)[0], np.where(va_m)[0], np.where(te_m)[0]
assert len(tr) > 1000, (GRID, YEAR, len(tr))
print(GRID, YEAR, "rows train/val/test", len(tr), len(va), len(te), flush=True)

ytr, yva, yte = y[tr], y[va], y[te]
beta = np.linalg.lstsq(Hm[tr], ytr, rcond=None)[0]; base = {k: Hm[i] @ beta for k, i in (("tr", tr), ("va", va), ("te", te))}

# delay-line features: the last K values of ln RV along the block's bins
x = dl.hold(lrv); xtr = lrv[pos[tr]]; xs = (x - np.nanmean(xtr)) / np.nanstd(xtr); t0 = time.time(); P = {}
for K in (50, 200):
    assert pos.min() >= K - 1
    F = lambda rows: np.stack([xs[pos[rows] - k] for k in range(K)], 1)
    P[f"LIN{K}"] = base["te"] + qrc.ridge_fit(F(tr), ytr - base["tr"], F(va), yva - base["va"], F(te))
    print(GRID, YEAR, f"LIN{K} done {time.time() - t0:.0f}s", flush=True)
out = HERE / "long_preds_lin" / GRID; out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{YEAR}.npz", y=yte, base=base["te"], t=tt[te].values.astype("datetime64[ns]").astype(np.int64), **{f"p_{k}": v for k, v in P.items()})
