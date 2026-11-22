"""v2 per-asset non-network models (see v2_prereg.json). Usage: python v2_run.py ASSET  (gpuvenv python: needs lightgbm).
Saves v2_preds/<asset>/<name>.npz with UNCALIBRATED variance levels for validation and test rows (lv_val, lv_te)."""
import sys
import numpy as np, pandas as pd
import lightgbm as lgb
from arch import arch_model
import hourly_qrc as hq, hourly_qrc_mem as mem, hourly_tune_qrc as tq
import v2_core as vc

asset = sys.argv[1]
H = vc.H
D = vc.prepare_v2(asset); d, r, rv, y, anc, M = D["d"], D["r"], D["rv"], D["y"], D["anchor"], D["M"]
tr, va, te = D["tr"], D["va"], D["te"]
(vc.OUT / asset).mkdir(parents=True, exist_ok=True)


def save(name, lv_all):
    np.savez(vc.OUT / asset / f"{name}.npz", lv_val=np.asarray(lv_all[va], float), lv_te=np.asarray(lv_all[te], float))


# ---- naive and EWMA on the range-variance series
rvv = rv.values; init = float(np.nanmean(rvv[tr]))
save("naive", np.exp(anc))
def ewma(lam):
    s = init; out = np.empty(len(rvv))
    for i, x in enumerate(rvv):
        if np.isfinite(x): s = lam * s + (1 - lam) * x
        out[i] = s
    return out
best = None
for lam in (0.90, 0.94, 0.97, 0.99, 0.995):
    e = ewma(lam); c = vc.calib_scale(np.exp(y[va]), e[va]); sc = vc.qlike_loss(y[va], np.log(c * e[va]))
    if best is None or sc < best[0]: best = (sc, lam, e)
save("EWMA", best[2]); lam_best = best[1]
save("EWMA_multi", np.mean([ewma(l) for l in (0.94, 0.97, 0.99)], 0))

# ---- GARCH(1,1)-t fit once on returns up to the training cut (levels for all rows via analytic 24-step mean)
valid = r.notna().values; cnt = np.cumsum(valid); xv = r.values[valid] * 100.0
cut = int(cnt[max(tr.max(), va.max())])
res = arch_model(xv, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(last_obs=cut, disp="off")
first = int(cnt[min(va.min(), te.min())] - 1)
fc = res.forecast(horizon=H, start=first, reindex=False, method="analytic").variance.values.mean(1) / 1e4
lvg = np.full(len(d), np.nan)
rows = np.r_[va, te]; lvg[rows] = fc[np.clip(cnt[rows] - 1 - first, 0, len(fc) - 1)]
save("GARCH11_t", lvg)

# ---- QLIKE-trained linear models (HAR, X, ALL, Xnoev), lambda on validation
for name, key in (("HARq", "HAR"), ("HARq_X", "X"), ("HARq_ALL", "ALL"), ("HARq_Xnoev", "Xnoev")):
    sc, lam, p = vc.best_lambda_linear(M[key], y, anc, tr, va)
    save(name, np.exp(vc.predict_linear(p, M[key], anc)))

# ---- rolling QLIKE HAR (windows 500/1000/2000 matured rows, refit every 120 rows), averaged over windows
Xh = M["HAR"]; okr = np.isfinite(y); allpos = np.where(okr)[0]
rows_eval = np.r_[va, te]; order = np.sort(rows_eval)
roll = {W: np.full(len(d), np.nan) for W in (500, 1000, 2000)}
for W in roll:
    last = None; lastblk = -1
    for t in order:
        blk = t // 120
        if blk != lastblk:
            avail = allpos[allpos <= t - H][-W:]
            if len(avail) >= 300: last = vc.fit_linear_qlike(Xh[avail], y[avail], anc[avail], 1e-3)
            lastblk = blk
        if last is not None: roll[W][t] = np.exp(vc.predict_linear(last, Xh[t:t + 1], anc[t:t + 1]))[0]
save("HARq_roll", np.nanmean([roll[W] for W in roll], 0))

# ---- LightGBM with QLIKE objective on residual of the anchor (X set), small val-grid, 5 seeds
yl = np.exp(y)
def obj_factory(anchor, ylev):
    def obj(raw, _):
        eta = anchor + raw; e = ylev * np.exp(-eta)
        return 1 - e, np.maximum(e, 1e-6)
    return obj
def feval_factory(anchor, ylev, y_):
    def fe(raw, _):
        eta = anchor + raw; return "qlike", float(np.mean(ylev * np.exp(-eta) + eta - y_ - 1)), False
    return fe
Xg = M["X"]; bestg = None
for leaves, lr, n in ((4, 0.03, 200), (8, 0.03, 200), (4, 0.05, 300)):
    prm = dict(num_leaves=leaves, learning_rate=lr, min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=0)
    dtr = lgb.Dataset(Xg[tr], label=y[tr], free_raw_data=False)
    bst = lgb.train({**prm, "objective": obj_factory(anc[tr], yl[tr])}, dtr, n, feval=feval_factory(anc[va], yl[va], y[va]), valid_sets=[lgb.Dataset(Xg[va], label=y[va])])
    ev = np.exp(anc[va] + bst.predict(Xg[va], raw_score=True)); c = vc.calib_scale(yl[va], ev); sc = vc.qlike_loss(y[va], np.log(c * ev))
    if bestg is None or sc < bestg[0]: bestg = (sc, prm, n)
ps = []
for sd in range(5):
    prm = {**bestg[1], "seed": sd}
    bst = lgb.train({**prm, "objective": obj_factory(anc[tr], yl[tr])}, lgb.Dataset(Xg[tr], label=y[tr]), bestg[2])
    ps.append(np.exp(anc + bst.predict(Xg, raw_score=True)))
save("LGBM_q", np.mean(ps, 0))

# ---- quantum reservoirs with QLIKE linear readout (features appended to the HAR set)
cz = np.column_stack([D["ret_z"], M["HAR"]])
def with_feats(Fres, s, name_prefix):
    X = np.hstack([M["HAR"], Fres]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    return vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)), X
mem.TAU, mem.JSCALE = 1.0, 0.25
ps = []
for s in range(10):
    f = mem.run_quantum(cz, mem.unitary(s), True)
    (sc, lam, p), X = with_feats(f, s, "ising"); ps.append(np.exp(vc.predict_linear(p, X, anc)))
save("Ising_q", np.mean(ps, 0))
prev = np.vstack([np.zeros((1, 4)), cz[:-1]]); G = np.hstack([cz, prev]); ps = []
rows_all = np.concatenate([tr, va, te])
for s in range(10):
    f = np.zeros((len(d), 36)); f[rows_all] = hq.reservoir_features(G[rows_all] * 0.25, 8, s, True, 3)
    (sc, lam, p), X = with_feats(f, s, "angle"); ps.append(np.exp(vc.predict_linear(p, X, anc)))
save("QRCangle_q", np.mean(ps, 0))
print(asset, "v2_run done; ewma lambda", lam_best, flush=True)
