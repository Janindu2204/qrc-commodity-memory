"""One block of long_prereg_hf2.json (extends hf_run.py; same data, task, models, readout).
Usage: python hf2_run.py MODE GRID YEAR     MODE = fixed | tuned ; GRID = H (60 min) | M (15 min) | F (5 min) ; YEAR = 2013..2025 or 2026 (held-out step).
fixed: the registered untuned configuration of long_prereg_hf.json (QRC scale 1, 2 layers; RandFeat scale 1 W-scale 1; ESN rho .9 leak .5; TrigFeat scale 1 freq 1), 10 seeds.
tuned: each model picks one configuration from a 12-point grid by validation MSE (2 tuning seeds 100,101, features on train+validation only), then 10 final seeds (0-9).
Writes long_preds_hf2/<MODE>/<GRID>/<YEAR>.npz. Scores nothing.
"""
import os, sys, time, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
import hourly_databento_qrc as qrc

qrc.CHUNK = 1000
MODE, GRID, YEAR = sys.argv[1], sys.argv[2], int(sys.argv[3]); DRY = os.environ.get("DRY") == "1"; NRES = 2 if DRY else 10; NTUNE = 1 if DRY else 2
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent
PURGE_H = 24; SEQ = 10; NQ = 8; F_ = 36
LONG = {"H": (24, 120), "M": (96, 480), "F": (288, 1440)}[GRID]
assert MODE in ("fixed", "tuned") and GRID in ("H", "M", "F")

z = np.load(ROOT / "data" / "hf" / f"{GRID}.npz"); t = pd.to_datetime(z["t"]); N = len(t)
ret, lrv0, lvol, hs, hc, adj, rvpos, hodbin = (z[k] for k in ("ret", "lrv", "lvol", "hs", "hc", "adj", "rvpos", "hodbin"))
lrv = np.where(rvpos, lrv0, np.nan)
S = pd.Series(lrv)
l_a = S.rolling(LONG[0], min_periods=LONG[0] // 2).mean().to_numpy(); l_b = S.rolling(LONG[1], min_periods=LONG[1] // 2).mean().to_numpy()
CH = np.stack([ret, lrv, lvol, hs, hc], 1)

yr_bin = np.r_[t.year.values[1:], 0]                                   # year of bin i+1 (target bin of a window ending at i)
pos = np.arange(SEQ - 1, N - 1); pos = pos[(yr_bin[pos] >= YEAR - 4) & (yr_bin[pos] <= YEAR)]   # keep only years this block can use (memory)
okw = np.ones(len(pos), bool)
for k in range(SEQ): okw &= np.isfinite(CH[pos - k]).all(1)
okw &= adj[pos + 1] & rvpos[pos + 1] & np.isfinite(l_a[pos]) & np.isfinite(l_b[pos])
pos = pos[okw]; tt = t[pos + 1]; yr = tt.year.values
y = lrv[pos + 1]
Hm = np.column_stack([np.ones(len(pos)), lrv[pos], np.mean([lrv[pos - k] for k in range(SEQ)], 0), l_a[pos], l_b[pos],
                      pd.get_dummies(hodbin[pos + 1]).reindex(columns=range(24), fill_value=0).astype(float).values[:, 1:]])
Q8 = np.column_stack([ret[pos], lrv[pos], lvol[pos], hs[pos], hc[pos], ret[pos - 1], lrv[pos - 1], lvol[pos - 1]])
SEQA = np.stack([CH[pos - (SEQ - 1) + k] for k in range(SEQ)], 1)

te_m = yr == YEAR; va_m = yr == YEAR - 1; tr_m = (yr >= YEAR - 4) & (yr <= YEAR - 2)
w = pd.Timedelta(hours=PURGE_H)
tr_m &= tt < (tt[va_m].min() - w) if va_m.any() else False
va_m &= tt < (tt[te_m].min() - w)
tr, va, te = np.where(tr_m)[0], np.where(va_m)[0], np.where(te_m)[0]
if DRY: tr, va, te = tr[:600], va[:150], te[:150]
assert len(tr) > 1000 or DRY, (GRID, YEAR, len(tr))
out = HERE / "long_preds_hf2" / MODE / GRID; out.mkdir(parents=True, exist_ok=True)
print(MODE, GRID, YEAR, "rows train/val/test", len(tr), len(va), len(te), flush=True)

ytr, yva, yte = y[tr], y[va], y[te]
beta = np.linalg.lstsq(Hm[tr], ytr, rcond=None)[0]; base = {k: Hm[i] @ beta for k, i in (("tr", tr), ("va", va), ("te", te))}
mu, sd = Q8[tr].mean(0), Q8[tr].std(0) + 1e-9
Q = {k: (Q8[i] - mu) / sd for k, i in (("tr", tr), ("va", va), ("te", te))}
smu, ssd = SEQA[tr].reshape(-1, 5).mean(0), SEQA[tr].reshape(-1, 5).std(0) + 1e-9
Sq = {k: (SEQA[i] - smu) / ssd for k, i in (("tr", tr), ("va", va), ("te", te))}
del SEQA, Hm, Q8
PAIRS = list(itertools.combinations(range(NQ), 2))


def feats(model, cfg, s, keys):
    """features of `model` under configuration cfg for seed s on the splits in keys"""
    if model in ("QRC", "QRC_noent"):
        return {k: qrc.reservoir_features(Q[k] * cfg["scale"], NQ, s, model == "QRC", cfg["layers"]) for k in keys}
    if model == "RandFeat":
        rng = np.random.default_rng(1000 + s); W, b = cfg["wscale"] * rng.normal(size=(NQ, F_)) / np.sqrt(NQ), rng.uniform(0, 2 * np.pi, F_)
        return {k: np.tanh(np.arctan(Q[k] * cfg["scale"]) @ W + b) for k in keys}
    if model == "ESN":
        return {k: qrc.esn_features(Sq[k], s, units=F_, leak=cfg["leak"], rho=cfg["rho"]) for k in keys}
    rng = np.random.default_rng(2000 + s); phi = rng.uniform(0, 2 * np.pi, F_); sg = rng.choice([-1.0, 1.0], len(PAIRS)); o = {}
    for k in keys:
        th = np.arctan(Q[k] * cfg["scale"]); Fm = np.zeros((len(th), F_))
        for i in range(NQ): Fm[:, i] = np.cos(cfg["freq"] * th[:, i] + phi[i])
        for j, (a, c) in enumerate(PAIRS): Fm[:, NQ + j] = np.cos(cfg["freq"] * (th[:, a] + sg[j] * th[:, c]) + phi[NQ + j])
        o[k] = Fm
    return o


names = ["QRC", "QRC_noent", "RandFeat", "ESN", "TrigFeat"]
FIXED = {"QRC": dict(scale=1.0, layers=2), "QRC_noent": dict(scale=1.0, layers=2), "RandFeat": dict(scale=1.0, wscale=1.0), "ESN": dict(rho=0.9, leak=0.5), "TrigFeat": dict(scale=1.0, freq=1.0)}
GRIDS = {"QRC": [dict(scale=a, layers=l) for a in (0.25, 0.5, 1.0, 2.0) for l in (1, 2, 3)], "QRC_noent": [dict(scale=a, layers=l) for a in (0.25, 0.5, 1.0, 2.0) for l in (1, 2, 3)],
         "RandFeat": [dict(scale=a, wscale=c) for a in (0.25, 0.5, 1.0, 2.0) for c in (0.5, 1.0, 2.0)], "ESN": [dict(rho=r, leak=l) for r in (0.6, 0.9, 1.2) for l in (0.1, 0.2, 0.5, 1.0)],
         "TrigFeat": [dict(scale=a, freq=f) for a in (0.25, 0.5, 1.0, 2.0) for f in (0.5, 1.0, 2.0)]}
if DRY: GRIDS = {m: g[:2] for m, g in GRIDS.items()}
t0 = time.time(); chosen = {}
for m in names:
    if MODE == "fixed": chosen[m] = FIXED[m]; continue
    best, bc = np.inf, None
    for cfg in GRIDS[m]:
        pv = []
        for s in range(NTUNE):
            f = feats(m, cfg, 100 + s, ("tr", "va"))
            pv.append(base["va"] + qrc.ridge_fit(f["tr"], ytr - base["tr"], f["va"], yva - base["va"], f["va"]))
        e = float(np.mean((yva - np.mean(pv, 0)) ** 2))
        if e < best: best, bc = e, cfg
    chosen[m] = bc; print(m, "chosen", bc, f"val mse {best:.5f}", f"{time.time() - t0:.0f}s", flush=True)

preds = {m: [] for m in names}
for s in range(NRES):
    for m in names:
        f = feats(m, chosen[m], s, ("tr", "va", "te"))
        preds[m].append(base["te"] + qrc.ridge_fit(f["tr"], ytr - base["tr"], f["va"], yva - base["va"], f["te"]))
    print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
res = {f"p_{m}": np.mean(v, 0) for m, v in preds.items()}
seedmse = {f"seedmse_{m}": np.array([np.mean((yte - q) ** 2) for q in v]) for m, v in preds.items()}
if DRY:
    print({k: float(np.mean((yte - v) ** 2) / np.mean((yte - base["te"]) ** 2)) for k, v in res.items()})
else:
    np.savez(out / f"{YEAR}.npz", y=yte, base=base["te"], t=tt[te].values.astype("datetime64[ns]").astype(np.int64), vol=np.nanmean(lrv[pos[te] + 1]), config=json.dumps(chosen), **res, **seedmse)
print(MODE, GRID, YEAR, "done", f"{time.time() - t0:.0f}s", flush=True)
