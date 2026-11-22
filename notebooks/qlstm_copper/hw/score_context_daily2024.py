"""Context arm of hw_prereg_daily2024.json (NOT part of the decision): hardware seed-0 and exact seed-0 QRC vs the saved 10-seed averages of QRC_angle, QRC_noent, ESN, RandFeat for the 2024 block. Main venv, PYTHONPATH=."""
import numpy as np, statsmodels.api as sm
import long_core as lc, v2_core as vc
D = lc.prepare("range_m", 2024, length=1); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
r = np.load("hw/daily2024_rebuild.npz"); F0 = r["F0"].copy()
X = np.hstack([M, F0]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
sc, lam, p = vc.best_lambda_linear((X - mu) / sd, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
def fc(Fte): Z = X.copy(); Z[te, M.shape[1]:] = Fte; return np.exp(vc.predict_linear(p, (Z - mu) / sd, anc))
Fh = np.full((len(te), F0.shape[1]), np.nan)
for j in ("0", "1", "2", "3"): z = np.load(f"hw/main_job_{j}.npz"); Fh[z["rows"]] = z["f"]
ex, hw = fc(F0[te]), fc(Fh); cal = vc.calib_scale(np.exp(y[va]), ex[va]); ok = np.isfinite(y[te]) & np.isfinite(ex[te]) & (ex[te] > 0)
rr = np.exp(y[te][ok])
def row_loss(f): f = f[ok]; return rr / f - np.log(rr / f) - 1
arms = {"hardware seed0": row_loss(cal * hw[te]), "exact seed0": row_loss(cal * ex[te])}
for m in ("QRC_angle", "QRC_noent", "ESN", "RandFeat"):
    z = np.load(f"long_preds_qrc/roll/range_m/2024/{m}.npz"); c = vc.calib_scale(np.exp(y[va][np.isfinite(z['lv_val'])]), z["lv_val"][np.isfinite(z["lv_val"])])
    arms[m + " (10-seed avg)"] = row_loss(c * z["lv_te"])
ref = arms["hardware seed0"]; print(f"{'arm':26s} mean QLIKE   hardware/arm ratio [95% HAC CI]")
for k, e in arms.items():
    d = ref - e; m = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 10}); md, se = float(m.params[0]), float(m.bse[0])
    print(f"{k:26s} {e.mean():.4f}      " + ("-" if k == "hardware seed0" else f"{1 + md / e.mean():.3f} [{1 + (md - 1.96 * se) / e.mean():.3f}, {1 + (md + 1.96 * se) / e.mean():.3f}]"))
