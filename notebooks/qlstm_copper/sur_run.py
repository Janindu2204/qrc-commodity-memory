"""One rolling block of long_prereg_surrogate.json (DRAFT until frozen). Usage: python sur_run.py YEAR     DRY=1: 2 seeds, prints timing and the deterministic span sanity check only, saves nothing.
Writes long_preds_sur/<YEAR>/<model>.npz (lv_val, lv_te = forecasts of the variance level) and span_<YEAR>.npz. Scores nothing."""
import os, sys, time
from pathlib import Path
import numpy as np
import sur_lib as L

YEAR = int(sys.argv[1]); DRY = os.environ.get("DRY") == "1"; NSEED = 2 if DRY else 10
D, y, anc, M, X0, G, tr, va, te = L.load_block(YEAR); n = len(M); rows = np.concatenate([tr, va, te])
out = Path(__file__).with_name("long_preds_sur") / str(YEAR); out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); print("block", YEAR, "rows train/val/test", len(tr), len(va), len(te), flush=True)
B = {d: L.basis_feats(G, rows, d) for d in (2, 3, 4)}; print("basis columns", {d: B[d].shape[1] for d in B}, flush=True)


def avg(fn):
    return np.mean([L.readout(M, fn(s), y, anc, tr, va) for s in range(NSEED)], 0)


MODELS = {
    "QRC_L1": lambda: avg(lambda s: L.qrc_feats(G, rows, s, True, 1)), "QRC_L2": lambda: avg(lambda s: L.qrc_feats(G, rows, s, True, 2)),
    "QRC_L3": lambda: avg(lambda s: L.qrc_feats(G, rows, s, True, 3)), "QRC_noent_L2": lambda: avg(lambda s: L.qrc_feats(G, rows, s, False, 2)),
    "Basis2": lambda: L.readout(M, B[2], y, anc, tr, va), "Basis3": lambda: L.readout(M, B[3], y, anc, tr, va), "Basis4": lambda: L.readout(M, B[4], y, anc, tr, va),
    "RandProj_2": lambda: avg(lambda s: L.randproj_feats(B[2], tr, rows, s)), "RandProj_3": lambda: avg(lambda s: L.randproj_feats(B[3], tr, rows, s)),
    "RandProj_4": lambda: avg(lambda s: L.randproj_feats(B[4], tr, rows, s)),
    "TrigFeat": lambda: avg(lambda s: L.trig_feats(G, rows, s)), "HAR_only": lambda: L.readout(M, None, y, anc, tr, va)}
for name, fn in MODELS.items():
    lv = fn(); assert np.isfinite(lv[te]).all() and (lv[te] > 0).all(), name
    if not DRY: np.savez(out / f"{name}.npz", lv_val=lv[va], lv_te=lv[te])
    print(f"{name:14s} done {time.time() - t0:.0f}s", flush=True)
# deterministic span diagnostic on ALL block rows (not forecasts)
sp = {}
for k in (1, 2, 3):
    for ent in (True, False):
        r = np.array([[L.span_r2(L.qrc_feats(G, rows, s, ent, k)[rows], B[d][rows]) for d in (2, 3, 4)] for s in range(NSEED)])    # seeds x degrees x 21
        sp[f"L{k}_{'ent' if ent else 'noent'}_min"] = r.min(axis=(0, 2)); sp[f"L{k}_{'ent' if ent else 'noent'}_mean"] = r.mean(axis=(0, 2))
tfr = np.array([L.span_r2(L.qrc_feats(G, rows, s, True, 1)[rows], L.trig_feats(G, rows, s)[rows]) for s in range(NSEED)]); sp["L1_on_TrigFeat_min"] = tfr.min(); sp["L1_on_TrigFeat_mean"] = tfr.mean()
if DRY: print({k: np.round(v, 4) for k, v in sp.items()})
else: np.savez(out / f"span_{YEAR}.npz", **sp)
print("done", f"{time.time() - t0:.0f}s")
