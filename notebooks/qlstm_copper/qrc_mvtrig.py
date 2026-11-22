"""Multivariate QRC vs multivariate ESN (Part A) and trigonometric-feature control (Part B), see long_prereg_mvtrig.json.
Usage: python qrc_mvtrig.py YEAR      (one-year rolling block, test year YEAR, 2008..2024)
Same tuning protocol as qrc_roll.py: 12 configurations per model, scored on the validation window ONLY (tuning seeds 100-102),
best config refit with 10 final seeds. Writes long_preds_qrc/mvtrig/range_m/<YEAR>/<model>.npz and appends the chosen
configs to qrc_mvtrig_configs.csv. Prints no score.
Part A (outside inputs, mv=True): QRCmv_angle, QRCmv_noent, ESNmv, RandFeatmv, plus the HARx reference.
Part B (univariate inputs as in qrc_roll.py): TrigFeat."""
import sys, itertools, csv
from pathlib import Path
import numpy as np
import long_core as lc, v2_core as vc, hourly_qrc as hq, hourly_tune_qrc as tq

year = int(sys.argv[1]); VERSION, NSEED, NQ, UNITS = "range_m", 10, 6, 21
TUNE_SEEDS = (100, 101, 102)
out = Path(__file__).with_name("long_preds_qrc") / "mvtrig" / VERSION / str(year); out.mkdir(parents=True, exist_ok=True)


def setup(mv):
    D = lc.prepare(VERSION, year, length=1, mv=mv)
    return D, D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]


def make_readout(M, y, anc, tr, va):
    def readout(F):
        X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
        sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
        return np.exp(vc.predict_linear(p, X, anc))
    return readout


def val_score(fn, y, va):
    lv = np.mean([fn(s) for s in TUNE_SEEDS], 0)[va]; ok = np.isfinite(lv) & (lv > 0); ev = lv[ok]
    c = vc.calib_scale(np.exp(y[va][ok]), ev); return vc.qlike_loss(y[va][ok], np.log(c * ev))


def tune_and_save(name, grid, y, va, te, w, tag):
    scores = [val_score(fn, y, va) for _, fn in grid]; k = int(np.argmin(scores))     # ties -> first
    cfg, fn = grid[k]; w.writerow([tag, year, name, cfg, len(grid)])
    lv = np.mean([fn(s) for s in range(NSEED)], 0); np.savez(out / f"{name}.npz", lv_val=lv[va], lv_te=lv[te])


cfg_log = Path(__file__).with_name("qrc_mvtrig_configs.csv"); new = not cfg_log.exists()
with open(cfg_log, "a", newline="") as fh:
    w = csv.writer(fh)
    if new: w.writerow(["part", "year", "model", "config", "n_configs"])

    # ---------------- Part A: multivariate (6 inputs = lvR1, lvR5, lvR22, lagged VIX, crude, dollar index; today only) ----------------
    D, y, anc, M, tr, va, te = setup(True); readout = make_readout(M, y, anc, tr, va)
    X0 = M[:, :6]; rows_all = np.concatenate([tr, va, te])
    sc, lam, p = vc.best_lambda_linear(M, y, anc, tr, va); lv = np.exp(vc.predict_linear(p, M, anc))
    np.savez(out / "HARx.npz", lv_val=lv[va], lv_te=lv[te])

    def qrc(entangle, layers, scale):
        def f(s):
            F = np.zeros((len(M), UNITS)); F[rows_all] = hq.reservoir_features(X0[rows_all] * scale, NQ, s, entangle, layers); return readout(F)
        return f

    def esn(radius, leak):
        return lambda s: readout(tq.esn(X0, s, radius, leak, UNITS))

    def randfeat(scale, wscale):
        def f(s):
            rng = np.random.default_rng(1000 + s); W = wscale * rng.normal(size=(6, UNITS)) / np.sqrt(6); b = rng.uniform(0, 2 * np.pi, UNITS)
            return readout(np.tanh(np.arctan(X0 * scale) @ W + b))
        return f

    GA = {
        "QRCmv_angle": [(dict(layers=l, scale=s), qrc(True, l, s)) for l, s in itertools.product((1, 2, 3), (0.1, 0.25, 0.5, 1.0))],
        "QRCmv_noent": [(dict(layers=l, scale=s), qrc(False, l, s)) for l, s in itertools.product((1, 2, 3), (0.1, 0.25, 0.5, 1.0))],
        "ESNmv": [(dict(radius=r, leak=k), esn(r, k)) for r, k in itertools.product((0.6, 0.9, 1.2), (0.1, 0.2, 0.5, 1.0))],
        "RandFeatmv": [(dict(scale=s, wscale=wv), randfeat(s, wv)) for s, wv in itertools.product((0.1, 0.25, 0.5, 1.0), (0.5, 1.0, 2.0))],
    }
    for name, grid in GA.items(): tune_and_save(name, grid, y, va, te, w, "A")

    # ---------------- Part B: trigonometric-feature control on the univariate inputs of qrc_roll.py ----------------
    D, y, anc, M, tr, va, te = setup(False); readout = make_readout(M, y, anc, tr, va)
    X0 = M[:, :3]; G = np.hstack([X0, np.vstack([np.zeros((1, 3)), X0[:-1]])]); rows_all = np.concatenate([tr, va, te])
    PAIRS = list(itertools.combinations(range(6), 2))                                 # 15 pairs: same count as the 15 ZZ features

    def trig(scale, freq):
        def f(s):
            rng = np.random.default_rng(2000 + s); th = np.arctan(G * scale)
            phi = rng.uniform(0, 2 * np.pi, UNITS); sg = rng.choice([-1.0, 1.0], len(PAIRS))
            F = np.zeros((len(M), UNITS))
            for i in range(6): F[:, i] = np.cos(freq * th[:, i] + phi[i])             # 6 one-input terms (cf. 6 single-qubit Z)
            for k, (i, j) in enumerate(PAIRS): F[:, 6 + k] = np.cos(freq * (th[:, i] + sg[k] * th[:, j]) + phi[6 + k])   # 15 two-input terms (cf. ZZ)
            return readout(F)
        return f

    GB = {"TrigFeat": [(dict(scale=s, freq=fq), trig(s, fq)) for s, fq in itertools.product((0.1, 0.25, 0.5, 1.0), (0.5, 1.0, 2.0))]}
    for name, grid in GB.items(): tune_and_save(name, grid, y, va, te, w, "B")
print(year, "mvtrig tuned + done", flush=True)
