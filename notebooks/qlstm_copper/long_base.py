"""Non-network models for one version/origin: python long_base.py VERSION ORIGIN -> long_preds/<version>/<origin>/{EWMA,HARq,GARCH11_t}.npz (uncalibrated levels on val+test)."""
import sys
import numpy as np
from arch import arch_model
import v2_core as vc
import long_core as lc

version, origin = sys.argv[1], int(sys.argv[2])
D = lc.prepare(version, origin); rv, y, anc, M, tr, va, te, d = D["rv"].values, D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"], D["d"]
out = lc.OUT / version / str(origin); out.mkdir(parents=True, exist_ok=True)
save = lambda name, lv: np.savez(out / f"{name}.npz", lv_val=np.asarray(lv[va], float), lv_te=np.asarray(lv[te], float))

init = float(np.nanmean(rv[tr]))
def ewma(lam):
    s = init; o = np.empty(len(rv))
    for i, x in enumerate(rv):
        if np.isfinite(x): s = lam * s + (1 - lam) * x
        o[i] = s
    return o
best = None
for lam in (0.90, 0.94, 0.97, 0.99, 0.995):
    e = ewma(lam); c = vc.calib_scale(np.exp(y[va]), e[va]); sc = vc.qlike_loss(y[va], np.log(c * e[va]))
    if best is None or sc < best[0]: best = (sc, lam, e)
save("EWMA", best[2])

sc, lamr, p = vc.best_lambda_linear(M, y, anc, tr, va)
save("HARq", np.exp(vc.predict_linear(p, M, anc)))

x = d.ret_clean.fillna(0.0).values * 100.0
res = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(last_obs=int(va.max()) + 1, disp="off")
first = int(min(va.min(), te.min()))
fc = res.forecast(horizon=lc.H, start=first, reindex=False, method="analytic").variance.values.mean(1) / 1e4
lvg = np.full(len(d), np.nan); rows = np.r_[va, te]; lvg[rows] = fc[rows - first]
save("GARCH11_t", lvg)
print(version, origin, "EWMA lambda", best[1], "HAR ridge", lamr, "floor", f"{D['floor']:.2e}", "floored days", D["n_floored"], "rows tr/va/te", len(tr), len(va), len(te), flush=True)
