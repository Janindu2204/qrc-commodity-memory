"""One block of long_prereg_smalln.json (DRAFT until frozen). Usage: python smalln_run.py YEAR     DRY=1: year as given, n=100 only, tuned arm, 2 final seeds, saves nothing, prints timing only.
Training set = last n eligible training rows (n in 50,100,250,500,1000); validation and test unchanged. Arms: tuned (12-config grids on validation, tuning seeds 100-102, final seeds 0-9) and fixed.
Writes long_preds_smalln/<arm>/<n>/<YEAR>/<model>.npz (lv_val, lv_te, config). Scores nothing."""
import os, sys, time, itertools, json
from pathlib import Path
import numpy as np
import sur_lib as L, v2_core as vc

YEAR = int(sys.argv[1]); DRY = os.environ.get("DRY") == "1"; NFIN = 2 if DRY else 10; TUNE = (100, 101, 102) if not DRY else (100,)
SIZES = (100,) if DRY else (50, 100, 250, 500, 1000); ARMS = ("tuned",) if DRY else ("tuned", "fixed")
D, y, anc, M, X0, G, tr_full, va, te = L.load_block(YEAR); n_all = len(M); rows = np.concatenate([tr_full, va, te])
root = Path(__file__).with_name("long_preds_smalln"); t0 = time.time(); cache = {}


def feats(model, cfg, s):
    key = (model, json.dumps(cfg, sort_keys=True), s)
    if key not in cache:
        if model in ("QRC", "QRC_noent", "QRC_L2", "QRC_L1", "QRC_noent_L2"): F = L.qrc_feats(G, rows, s, model not in ("QRC_noent", "QRC_noent_L2"), cfg["layers"], cfg["scale"])
        elif model == "RandFeat": F = L.randfeat_feats(G, rows, s, cfg["scale"], cfg["wscale"])
        elif model == "ESN": F = L.esn_feats(X0, s, cfg["radius"], cfg["leak"])
        elif model == "TrigFeat": F = L.trig_feats(G, rows, s, cfg["scale"], cfg["freq"])
        elif model == "Basis2": F = L.basis_feats(G, rows, 2, cfg["scale"])
        else: F = None
        cache[key] = F
    return cache[key]


GRID = {"QRC": [dict(layers=l, scale=a) for l in (1, 2, 3) for a in (0.1, 0.25, 0.5, 1.0)], "QRC_noent": [dict(layers=l, scale=a) for l in (1, 2, 3) for a in (0.1, 0.25, 0.5, 1.0)],
        "ESN": [dict(radius=r, leak=k) for r in (0.6, 0.9, 1.2) for k in (0.1, 0.2, 0.5, 1.0)], "RandFeat": [dict(scale=a, wscale=w) for a in (0.1, 0.25, 0.5, 1.0) for w in (0.5, 1.0, 2.0)],
        "TrigFeat": [dict(scale=a, freq=f) for a in (0.1, 0.25, 0.5, 1.0) for f in (0.5, 1.0, 2.0)], "Basis2": [dict(scale=1.0)], "HAR_only": [dict()]}
FIXED = {"QRC_L2": dict(layers=2, scale=1.0), "QRC_L1": dict(layers=1, scale=1.0), "QRC_noent_L2": dict(layers=2, scale=1.0), "RandFeat": dict(scale=1.0, wscale=1.0),
         "ESN": dict(radius=0.9, leak=0.5), "TrigFeat": dict(scale=1.0, freq=1.0), "Basis2": dict(scale=1.0), "HAR_only": dict()}


def forecast(model, cfg, seeds, tr, va):
    if model in ("Basis2", "HAR_only"): return L.readout(M, feats(model, cfg, 0), y, anc, tr, va)
    return np.mean([L.readout(M, feats(model, cfg, s), y, anc, tr, va) for s in seeds], 0)


def val_score(lv, va):
    ok = np.isfinite(lv[va]) & (lv[va] > 0); c = vc.calib_scale(np.exp(y[va][ok]), lv[va][ok]); return vc.qlike_loss(y[va][ok], np.log(c * lv[va][ok]))


for arm in ARMS:
    for n in SIZES:
        tr = tr_full[-n:]; od = root / arm / str(n) / str(YEAR); od.mkdir(parents=True, exist_ok=True) if not DRY else None
        for model in (GRID if arm == "tuned" else FIXED):
            if arm == "tuned":
                sc = np.array([val_score(forecast(model, c, TUNE, tr, va), va) for c in GRID[model]]); sc[~np.isfinite(sc)] = np.inf; cfg = GRID[model][int(np.argmin(sc))]     # ties -> first; non-finite scores never chosen
            else: cfg = FIXED[model]
            lv = forecast(model, cfg, range(NFIN), tr, va); assert np.isfinite(lv[te]).all() and (lv[te] > 0).all(), (arm, n, model)
            if not DRY: np.savez(od / f"{model}.npz", lv_val=lv[va], lv_te=lv[te], config=json.dumps(cfg))
        print(f"{arm} n={n} done {time.time() - t0:.0f}s", flush=True)
print("done", f"{time.time() - t0:.0f}s")
