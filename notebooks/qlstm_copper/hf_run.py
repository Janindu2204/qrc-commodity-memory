"""One rolling block of the high-frequency design (long_prereg_hf.json).
Usage: python hf_run.py GRID YEAR      GRID = H (60-minute bins) or M (15-minute bins); YEAR = test year 2013..2025.   DRY=1: 600/150/150 rows, 2 seeds.
Target: ln realised variance (sum of squared 1-minute returns) of the NEXT bin. Benchmark: OLS HAR + hour-of-day on train.
Models (all as residual add-ons to the benchmark, ridge on the 36 reservoir features, lambda on validation, forecasts averaged over seeds):
QRC (8 qubits, exact), QRC_noent, RandFeat, ESN, TrigFeat. Circuit/ESN/ridge code is imported from hourly_databento_qrc.py. Writes long_preds_hf/<GRID>/<YEAR>.npz.
"""
import os, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
import hourly_databento_qrc as qrc

qrc.CHUNK = 1000
GRID, YEAR = sys.argv[1], int(sys.argv[2]); DRY = os.environ.get("DRY") == "1"; NRES = 2 if DRY else int(os.environ.get("NRES", 10))
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent
PURGE_H = 24; SEQ = 10; NQ = 8; F_ = 36
LONG = (24, 120) if GRID == "H" else (96, 480)

z = np.load(ROOT / "data" / "hf" / f"{GRID}.npz"); t = pd.to_datetime(z["t"]); N = len(t)
ret, lrv0, lvol, hs, hc, adj, rvpos, hodbin = (z[k] for k in ("ret", "lrv", "lvol", "hs", "hc", "adj", "rvpos", "hodbin"))
lrv = np.where(rvpos, lrv0, np.nan)
S = pd.Series(lrv)
l_a = S.rolling(LONG[0], min_periods=LONG[0] // 2).mean().to_numpy(); l_b = S.rolling(LONG[1], min_periods=LONG[1] // 2).mean().to_numpy()
CH = np.stack([ret, lrv, lvol, hs, hc], 1)                      # (N, 5); channel order ret, lrv, lvol, hs, hc

pos = np.arange(SEQ - 1, N - 1)                                  # window ends at bin i, target is bin i+1
okw = np.ones(len(pos), bool)
for k in range(SEQ):
    okw &= np.isfinite(CH[pos - k]).all(1)
okw &= adj[pos + 1] & rvpos[pos + 1] & np.isfinite(l_a[pos]) & np.isfinite(l_b[pos])
pos = pos[okw]; tt = t[pos + 1]; yr = tt.year.values
y = lrv[pos + 1]
Hm = np.column_stack([np.ones(len(pos)), lrv[pos], np.mean([lrv[pos - k] for k in range(SEQ)], 0), l_a[pos], l_b[pos],
                      pd.get_dummies(hodbin[pos + 1]).reindex(columns=range(24), fill_value=0).astype(float).values[:, 1:]])
Q8 = np.column_stack([ret[pos], lrv[pos], lvol[pos], hs[pos], hc[pos], ret[pos - 1], lrv[pos - 1], lvol[pos - 1]])
SEQA = np.stack([CH[pos - (SEQ - 1) + k] for k in range(SEQ)], 1)  # (n, 10, 5) oldest first

te_m = yr == YEAR; va_m = yr == YEAR - 1; tr_m = (yr >= YEAR - 4) & (yr <= YEAR - 2)
w = pd.Timedelta(hours=PURGE_H)
tr_m &= tt < (tt[va_m].min() - w) if va_m.any() else False
va_m &= tt < (tt[te_m].min() - w)
tr, va, te = np.where(tr_m)[0], np.where(va_m)[0], np.where(te_m)[0]
if DRY: tr, va, te = tr[:600], va[:150], te[:150]
assert len(tr) > 1000 or DRY, (GRID, YEAR, len(tr))
out = HERE / "long_preds_hf" / GRID; out.mkdir(parents=True, exist_ok=True)
print(GRID, YEAR, "rows train/val/test", len(tr), len(va), len(te), flush=True)

ytr, yva, yte = y[tr], y[va], y[te]
beta = np.linalg.lstsq(Hm[tr], ytr, rcond=None)[0]; base = {k: Hm[i] @ beta for k, i in (("tr", tr), ("va", va), ("te", te))}
mu, sd = Q8[tr].mean(0), Q8[tr].std(0) + 1e-9
Q = {k: (Q8[i] - mu) / sd for k, i in (("tr", tr), ("va", va), ("te", te))}
smu, ssd = SEQA[tr].reshape(-1, 5).mean(0), SEQA[tr].reshape(-1, 5).std(0) + 1e-9
Sq = {k: (SEQA[i] - smu) / ssd for k, i in (("tr", tr), ("va", va), ("te", te))}
PAIRS = list(itertools.combinations(range(NQ), 2))


def trig(Qk, s):
    rng = np.random.default_rng(2000 + s); th = np.arctan(Qk); phi = rng.uniform(0, 2 * np.pi, F_); sg = rng.choice([-1.0, 1.0], len(PAIRS))
    Fm = np.zeros((len(Qk), F_))
    for i in range(NQ): Fm[:, i] = np.cos(th[:, i] + phi[i])
    for k, (i, j) in enumerate(PAIRS): Fm[:, NQ + k] = np.cos(th[:, i] + sg[k] * th[:, j] + phi[NQ + k])
    return Fm


names = ["QRC", "QRC_noent", "RandFeat", "ESN", "TrigFeat"]; preds = {m: [] for m in names}; t0 = time.time()
for s in range(NRES):
    rng = np.random.default_rng(1000 + s); W, b = rng.normal(size=(NQ, F_)) / np.sqrt(NQ), rng.uniform(0, 2 * np.pi, F_)
    Fm = {"QRC": {k: qrc.reservoir_features(Q[k], NQ, s, True) for k in Q}, "QRC_noent": {k: qrc.reservoir_features(Q[k], NQ, s, False) for k in Q},
          "RandFeat": {k: np.tanh(np.arctan(Q[k]) @ W + b) for k in Q}, "ESN": {k: qrc.esn_features(Sq[k], s) for k in Sq},
          "TrigFeat": {k: trig(Q[k], s) for k in Q}}
    for m, f in Fm.items():
        preds[m].append(base["te"] + qrc.ridge_fit(f["tr"], ytr - base["tr"], f["va"], yva - base["va"], f["te"]))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
res = {f"p_{m}": np.mean(v, 0) for m, v in preds.items()}
seedmse = {f"seedmse_{m}": np.array([np.mean((yte - q) ** 2) for q in v]) for m, v in preds.items()}
if not DRY:
    np.savez(out / f"{YEAR}.npz", y=yte, base=base["te"], t=tt[te].values.astype("datetime64[ns]").astype(np.int64), vol=np.nanmean(lrv[pos[te] + 1]), **res, **seedmse)
else:
    print({k: float(np.mean((yte - v) ** 2) / np.mean((yte - base["te"]) ** 2)) for k, v in res.items()})
print(GRID, YEAR, "done", f"{time.time() - t0:.0f}s", flush=True)
