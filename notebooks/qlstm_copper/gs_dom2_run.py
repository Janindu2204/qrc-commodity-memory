"""One intraday block of long_prereg_dom2.json. Usage: python dom2_run.py GRID YEAR    GRID = H | M | F; YEAR = 2013..2025, or 2026 (held-out step).
Rows, target, benchmark, splits, input series and scaling are those of dom_run.py (code copied unchanged above the model section). New models:
ISING (transverse-field Ising reservoir, dom2_lib, 10 seeds, no tuning) and ESN28u_t (the 28-unit single-input echo state network of dom_run.py, now TUNED:
12 configurations spectral radius {0.6,0.9,1.2} x leak {0.1,0.2,0.5,1.0}, chosen by validation MSE with tuning seeds 100,101, then seeds 0-9).
Writes long_preds_dom2/<GRID>/<YEAR>.npz. Scores nothing on test rows. DRY=1: 3000 bins, shapes only.
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import json
import hourly_databento_qrc as qrc, dom_lib as dl, dom2_lib as d2

GRID, YEAR = sys.argv[1], int(sys.argv[2]); DRY = os.environ.get("DRY") == "1"; NRES = 2 if DRY else 10
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent
PURGE_H = 24; SEQ = 10
LONG = {"H": (24, 120), "M": (96, 480), "F": (288, 1440)}[GRID]

ASSET = os.environ["GS_ASSET"]; z = np.load(ROOT / "data" / f"hf_{ASSET}" / f"{GRID}.npz"); t = pd.to_datetime(z["t"]); N = len(t)
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

# single input series along the block's bins: ln RV of the bin, missing values hold the last value; scaling constants from TRAINING rows only
i0, i1 = int(pos[0]), int(pos[-1]) + 1; x = dl.hold(lrv[i0:i1]); xtr = lrv[pos[tr]]
lo, hi = float(xtr.min()), float(xtr.max()); u = np.clip((x - lo) / (hi - lo), 0.0, 1.0); zs = (x - xtr.mean()) / xtr.std()
rel = pos - i0; R = {"tr": rel[tr], "va": rel[va], "te": rel[te]}
if DRY: u, zs = u[:3000], zs[:3000]

t0 = time.time(); fit = lambda f: base["te"] + qrc.ridge_fit(f[R["tr"]], ytr - base["tr"], f[R["va"]], yva - base["va"], f[R["te"]])
if DRY:
    print({"ISING": d2.ising_features(u, 0).shape, "ESN": dl.esn1(zs, 100, leak=0.1, rho=1.2).shape}); sys.exit()
best, cfg = np.inf, None
for rho in (0.6, 0.9, 1.2):
    for leak in (0.1, 0.2, 0.5, 1.0):
        pv = []
        for s in (100, 101):
            f = dl.esn1(zs, s, leak=leak, rho=rho); pv.append(base["va"] + qrc.ridge_fit(f[R["tr"]], ytr - base["tr"], f[R["va"]], yva - base["va"], f[R["va"]]))
        e = float(np.mean((yva - np.mean(pv, 0)) ** 2))
        if e < best: best, cfg = e, dict(rho=rho, leak=leak)
print("ESN28u_t chosen", cfg, f"val mse {best:.5f}", f"{time.time() - t0:.0f}s", flush=True)
preds = {"ISING": [], "ESN28u_t": []}
for s in range(NRES):
    preds["ESN28u_t"].append(fit(dl.esn1(zs, s, leak=cfg["leak"], rho=cfg["rho"]))); preds["ISING"].append(fit(d2.ising_features(u, s)))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
out = HERE / "long_preds_gs" / ASSET / "dom2" / GRID; out.mkdir(parents=True, exist_ok=True)
res = {f"p_{m}": np.mean(v, 0) for m, v in preds.items()}; seedp = {f"seedp_{m}": np.array(v, dtype=np.float32) for m, v in preds.items()}
np.savez(out / f"{YEAR}.npz", y=yte, base=base["te"], t=tt[te].values.astype("datetime64[ns]").astype(np.int64), vol=np.nanmean(lrv[pos[te] + 1]), config=json.dumps(cfg), **res, **seedp)
print(GRID, YEAR, "done", f"{time.time() - t0:.0f}s", flush=True)
