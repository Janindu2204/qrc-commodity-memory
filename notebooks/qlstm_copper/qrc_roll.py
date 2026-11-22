"""QRC + matched classical controls on the long daily series, WITH validation-window tuning (long_prereg_qrc.json).
Usage: python qrc_roll.py TAG YEAR [LENGTH]   TAG = roll (1-year blocks, LENGTH 1) or orig (original 6-year origins, LENGTH omitted)
Per block: every model gets the same budget of 12 configurations, scored on the validation window ONLY (disjoint tuning seeds 100-102,
forecasts averaged over those seeds), best config refit with 10 final seeds. Writes long_preds_qrc/<TAG>/range_m/<YEAR>/<model>.npz and appends
the chosen configs to qrc_tuned_configs.csv. Prints no score."""
import sys, itertools, csv
from pathlib import Path
import numpy as np
import long_core as lc, v2_core as vc, hourly_qrc as hq, hourly_tune_qrc as tq

tag, year = sys.argv[1], int(sys.argv[2]); length = int(sys.argv[3]) if len(sys.argv) > 3 else None
VERSION, NSEED, NQ, UNITS = "range_m", 10, 6, 21
TUNE_SEEDS = (100, 101, 102)
D = lc.prepare(VERSION, year, length=length); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
X0 = M[:, :3]                                                       # standardised lvR1, lvR5, lvR22 (train-row statistics)
G = np.hstack([X0, np.vstack([np.zeros((1, 3)), X0[:-1]])])         # today and previous day: 6 inputs
rows_all = np.concatenate([tr, va, te])
out = Path(__file__).with_name("long_preds_qrc") / tag / VERSION / str(year); out.mkdir(parents=True, exist_ok=True)


def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
    return np.exp(vc.predict_linear(p, X, anc))


def qrc(entangle, layers, scale):
    def f(s):
        F = np.zeros((len(M), UNITS)); F[rows_all] = hq.reservoir_features(G[rows_all] * scale, NQ, s, entangle, layers); return readout(F)
    return f


def esn(radius, leak):
    return lambda s: readout(tq.esn(X0, s, radius, leak, UNITS))


def randfeat(scale, wscale):
    def f(s):
        rng = np.random.default_rng(1000 + s); W = wscale * rng.normal(size=(6, UNITS)) / np.sqrt(6); b = rng.uniform(0, 2 * np.pi, UNITS)
        return readout(np.tanh(np.arctan(G * scale) @ W + b))
    return f


GRIDS = {   # 12 configurations each (equal tuning budget)
    "QRC_angle": [(dict(layers=l, scale=s), qrc(True, l, s)) for l, s in itertools.product((1, 2, 3), (0.1, 0.25, 0.5, 1.0))],
    "QRC_noent": [(dict(layers=l, scale=s), qrc(False, l, s)) for l, s in itertools.product((1, 2, 3), (0.1, 0.25, 0.5, 1.0))],
    "ESN": [(dict(radius=r, leak=k), esn(r, k)) for r, k in itertools.product((0.6, 0.9, 1.2), (0.1, 0.2, 0.5, 1.0))],
    "RandFeat": [(dict(scale=s, wscale=w), randfeat(s, w)) for s, w in itertools.product((0.1, 0.25, 0.5, 1.0), (0.5, 1.0, 2.0))],
}


def val_score(fn):
    lv = np.mean([fn(s) for s in TUNE_SEEDS], 0)[va]; ev = lv[np.isfinite(lv) & (lv > 0)]
    c = vc.calib_scale(np.exp(y[va][np.isfinite(lv) & (lv > 0)]), ev); return vc.qlike_loss(y[va][np.isfinite(lv) & (lv > 0)], np.log(c * ev))


cfg_log = Path(__file__).with_name("qrc_tuned_configs.csv"); new = not cfg_log.exists()
with open(cfg_log, "a", newline="") as fh:
    w = csv.writer(fh)
    if new: w.writerow(["tag", "year", "model", "config", "n_configs"])
    for name, grid in GRIDS.items():
        scores = [val_score(fn) for _, fn in grid]; k = int(np.argmin(scores))          # ties -> first (smallest) config
        cfg, fn = grid[k]; w.writerow([tag, year, name, cfg, len(grid)])
        lv = np.mean([fn(s) for s in range(NSEED)], 0); np.savez(out / f"{name}.npz", lv_val=lv[va], lv_te=lv[te])
print(tag, year, "QRC tuned + done", flush=True)
