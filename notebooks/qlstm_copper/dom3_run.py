"""One intraday block of long_prereg_dom3.json. Usage: python dom3_run.py GRID YEAR    GRID = H | M | F; YEAR = 2013..2025, or 2026 (held-out step).
Rows, target, benchmark, splits, input series and scaling are those of dom_run.py / dom2_run.py (code copied unchanged above the model section). New models, all residual add-ons to the benchmark:
ESNq_t  : 28-unit echo state network fed the two input functions of the quantum encoding (u and sqrt(u(1-u)), standardised); 12 configurations, tuned as ESN28u_t.
SAS28_t : 28-number state-affine system (dom3_lib.sas); 12 configurations gain {0.3,0.5,0.65} x leak {0.1,0.2,0.5,1.0}, tuned the same way.
LSTM, GRU : one recurrent layer + linear head on the last 10 bins of the 5 channels (return, ln RV, ln volume, hour sine, hour cosine; the window of the stored ESN);
            12 configurations hidden {8,16,32} x learning rate {3e-4,1e-3,3e-3,1e-2}, chosen by validation MSE with tuning seed 100, then seeds 0-9.
Writes long_preds_dom3/<GRID>/<YEAR>.npz. Scores nothing on test rows. DRY=1: 3000 bins, shapes only.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import hourly_databento_qrc as qrc, dom_lib as dl, dom3_lib as d3

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

# single input series along the block's bins: ln RV of the bin, missing values hold the last value; scaling constants from TRAINING rows only
i0, i1 = int(pos[0]), int(pos[-1]) + 1; x = dl.hold(lrv[i0:i1]); xtr = lrv[pos[tr]]
lo, hi = float(xtr.min()), float(xtr.max()); u = np.clip((x - lo) / (hi - lo), 0.0, 1.0); zs = (x - xtr.mean()) / xtr.std()
rel = pos - i0; R = {"tr": rel[tr], "va": rel[va], "te": rel[te]}
if DRY: u, zs = u[:3000], zs[:3000]

cc = np.sqrt(u * (1 - u)); rt = R["tr"][R["tr"] < len(u)]; Z = np.column_stack([zs, (cc - cc[rt].mean()) / (cc[rt].std() + 1e-9)])     # rt = all training rows unless DRY
SEQA = np.stack([CH[pos - (SEQ - 1) + k] for k in range(SEQ)], 1); smu, ssd = SEQA[tr].reshape(-1, 5).mean(0), SEQA[tr].reshape(-1, 5).std(0) + 1e-9
rsd = float((ytr - base["tr"]).std()); T32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
Xs = {k: T32((SEQA[i] - smu) / ssd) for k, i in (("tr", tr), ("va", va), ("te", te))}; del SEQA
rs = {"tr": T32((ytr - base["tr"]) / rsd), "va": T32((yva - base["va"]) / rsd)}

t0 = time.time(); fit = lambda f: base["te"] + qrc.ridge_fit(f[R["tr"]], ytr - base["tr"], f[R["va"]], yva - base["va"], f[R["te"]])
RES = {"ESNq_t": (d3.ESN_GRID, lambda s, c: d3.esn_k(Z, s, leak=c["leak"], rho=c["rho"])), "SAS28_t": (d3.SAS_GRID, lambda s, c: d3.sas(u, s, leak=c["leak"], gain=c["gain"]))}
if DRY:
    print({m: fn(100, g[0]).shape for m, (g, fn) in RES.items()}, {k: tuple(v.shape) for k, v in Xs.items()})
    print({kind: d3.fit_seq(kind, d3.NN_GRID[0], 100, Xs["tr"][:2000], rs["tr"][:2000], Xs["va"][:500], rs["va"][:500], Xs["te"][:500], max_epochs=1)[2].shape for kind in ("LSTM", "GRU")}); sys.exit()
chosen, preds = {}, {}
for m, (grid, fn) in RES.items():
    best, cfg = np.inf, None
    for c in grid:
        pv = []
        for s in (100, 101):
            f = fn(s, c); pv.append(base["va"] + qrc.ridge_fit(f[R["tr"]], ytr - base["tr"], f[R["va"]], yva - base["va"], f[R["va"]]))
        e = float(np.mean((yva - np.mean(pv, 0)) ** 2))
        if e < best: best, cfg = e, c
    chosen[m] = cfg; preds[m] = [fit(fn(s, cfg)) for s in range(NRES)]
    print(m, "chosen", cfg, f"val mse {best:.5f}", f"{time.time() - t0:.0f}s", flush=True)
for kind in ("LSTM", "GRU"):
    vals = [d3.fit_seq(kind, c, 100, Xs["tr"], rs["tr"], Xs["va"], rs["va"])[0] for c in d3.NN_GRID]; cfg = d3.NN_GRID[int(np.argmin(vals))]
    chosen[kind] = cfg; preds[kind] = [base["te"] + rsd * d3.fit_seq(kind, cfg, s, Xs["tr"], rs["tr"], Xs["va"], rs["va"], Xs["te"])[2] for s in range(NRES)]
    print(kind, "chosen", cfg, f"val mse (standardised residual) {min(vals):.5f}", f"{time.time() - t0:.0f}s", flush=True)
out = HERE / "long_preds_dom3" / GRID; out.mkdir(parents=True, exist_ok=True)
res = {f"p_{m}": np.mean(v, 0) for m, v in preds.items()}; seedp = {f"seedp_{m}": np.array(v, dtype=np.float32) for m, v in preds.items()}
np.savez(out / f"{YEAR}.npz", y=yte, base=base["te"], t=tt[te].values.astype("datetime64[ns]").astype(np.int64), vol=np.nanmean(lrv[pos[te] + 1]), config=json.dumps(chosen), **res, **seedp)
print(GRID, YEAR, "done", f"{time.time() - t0:.0f}s", flush=True)
