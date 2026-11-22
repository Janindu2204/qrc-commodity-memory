"""Databento-based daily target (long_prereg_dbdaily.json). Usage: python db_run.py TARGET YEAR   TARGET = range | rv ; YEAR = test year 2015..2024 (one-year block).
Same models, grids, tuning protocol (12 configs per model on the validation window only, seeds 100-102, 10 final seeds) as qrc_roll.py / qrc_mvtrig.py part B.
Also computes EWMA and HARq (as long_base.py). Writes long_preds_qrc/dbdaily/<TARGET>/<YEAR>/*.npz and appends configs to qrc_dbdaily_configs.csv. Prints no score."""
import sys, itertools, csv
from pathlib import Path
import numpy as np, pandas as pd
import long_core as lc, v2_core as vc, hourly_qrc as hq, hourly_tune_qrc as tq

target, year = sys.argv[1], int(sys.argv[2]); NSEED, NQ, UNITS, TUNE = 10, 6, 21, (100, 101, 102)
data = pd.read_csv(lc.ROOT.parents[0] / "daily_databento" / f"HG_db_1d_{target}.csv", parse_dates=["date"]).set_index("date")
D = lc.prepare("range_m", year, length=1, data=data); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]; rv = D["rv"].values
out = Path(__file__).with_name("long_preds_qrc") / "dbdaily" / target / str(year); out.mkdir(parents=True, exist_ok=True)
X0 = M[:, :3]; G = np.hstack([X0, np.vstack([np.zeros((1, 3)), X0[:-1]])]); rows_all = np.concatenate([tr, va, te]); PAIRS = list(itertools.combinations(range(6), 2))
save = lambda name, lv: np.savez(out / f"{name}.npz", lv_val=np.asarray(lv[va], float), lv_te=np.asarray(lv[te], float))

init = float(np.nanmean(rv[tr])); best = None
for lam in (0.90, 0.94, 0.97, 0.99, 0.995):
    s = init; e = np.empty(len(rv))
    for i, x in enumerate(rv):
        if np.isfinite(x): s = lam * s + (1 - lam) * x
        e[i] = s
    c = vc.calib_scale(np.exp(y[va]), e[va]); sc = vc.qlike_loss(y[va], np.log(c * e[va]))
    if best is None or sc < best[0]: best = (sc, e)
save("EWMA", best[1])
sc, lamr, p = vc.best_lambda_linear(M, y, anc, tr, va); save("HARq", np.exp(vc.predict_linear(p, M, anc)))


def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)); return np.exp(vc.predict_linear(p, X, anc))


def qrc(ent, layers, scale):
    def f(s):
        F = np.zeros((len(M), UNITS)); F[rows_all] = hq.reservoir_features(G[rows_all] * scale, NQ, s, ent, layers); return readout(F)
    return f


def esn(radius, leak): return lambda s: readout(tq.esn(X0, s, radius, leak, UNITS))


def randfeat(scale, wscale):
    def f(s):
        rng = np.random.default_rng(1000 + s); W = wscale * rng.normal(size=(6, UNITS)) / np.sqrt(6); b = rng.uniform(0, 2 * np.pi, UNITS)
        return readout(np.tanh(np.arctan(G * scale) @ W + b))
    return f


def trig(scale, freq):
    def f(s):
        rng = np.random.default_rng(2000 + s); th = np.arctan(G * scale); phi = rng.uniform(0, 2 * np.pi, UNITS); sg = rng.choice([-1.0, 1.0], len(PAIRS)); F = np.zeros((len(M), UNITS))
        for i in range(6): F[:, i] = np.cos(freq * th[:, i] + phi[i])
        for k, (i, j) in enumerate(PAIRS): F[:, 6 + k] = np.cos(freq * (th[:, i] + sg[k] * th[:, j]) + phi[6 + k])
        return readout(F)
    return f


GRIDS = {
    "QRC_angle": [(dict(layers=l, scale=s), qrc(True, l, s)) for l, s in itertools.product((1, 2, 3), (0.1, 0.25, 0.5, 1.0))],
    "QRC_noent": [(dict(layers=l, scale=s), qrc(False, l, s)) for l, s in itertools.product((1, 2, 3), (0.1, 0.25, 0.5, 1.0))],
    "ESN": [(dict(radius=r, leak=k), esn(r, k)) for r, k in itertools.product((0.6, 0.9, 1.2), (0.1, 0.2, 0.5, 1.0))],
    "RandFeat": [(dict(scale=s, wscale=w), randfeat(s, w)) for s, w in itertools.product((0.1, 0.25, 0.5, 1.0), (0.5, 1.0, 2.0))],
    "TrigFeat": [(dict(scale=s, freq=fq), trig(s, fq)) for s, fq in itertools.product((0.1, 0.25, 0.5, 1.0), (0.5, 1.0, 2.0))],
}


def val_score(fn):
    lv = np.mean([fn(s) for s in TUNE], 0)[va]; ok = np.isfinite(lv) & (lv > 0); ev = lv[ok]
    c = vc.calib_scale(np.exp(y[va][ok]), ev); return vc.qlike_loss(y[va][ok], np.log(c * ev))


with open(Path(__file__).with_name("qrc_dbdaily_configs.csv"), "a", newline="") as fh:
    w = csv.writer(fh)
    for name, grid in GRIDS.items():
        k = int(np.argmin([val_score(fn) for _, fn in grid])); cfg, fn = grid[k]; w.writerow([target, year, name, cfg, len(grid)])
        save(name, np.mean([fn(s) for s in range(NSEED)], 0))
print(target, year, "dbdaily done", flush=True)
