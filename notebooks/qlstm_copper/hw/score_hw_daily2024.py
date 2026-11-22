"""Score the frozen hardware registration (hw_prereg_daily2024.json + addendum1). Main venv, PYTHONPATH=. from qlstm_copper/.
Readout fitted on EXACT seed-0 train/val features; test features = hardware (main jobs 0-3, one measurement per row) or exact.
Primary: ratio of mean QLIKE hardware/exact with HAC (lag 10) interval on per-row loss differences; verdict on [0.97, 1.03]."""
import numpy as np, statsmodels.api as sm
import long_core as lc, v2_core as vc
D = lc.prepare("range_m", 2024, length=1); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
r = np.load("hw/daily2024_rebuild.npz"); F0 = r["F0"].copy(); nz = np.load("hw/daily2024_noisy.npz")
X = np.hstack([M, F0]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
sc, lam, p = vc.best_lambda_linear((X - mu) / sd, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
def fc(Fte):
    Z = X.copy(); Z[te, M.shape[1]:] = Fte; return np.exp(vc.predict_linear(p, (Z - mu) / sd, anc))
base = fc(F0[te]); c = vc.calib_scale(np.exp(y[va]), base[va]); ok = np.isfinite(y[te]) & np.isfinite(base[te]) & (base[te] > 0)
def loss(pred): f = c * pred[te][ok]; rr = np.exp(y[te][ok]); return rr / f - np.log(rr / f) - 1
def compare(Fte, label):
    e0, e1 = loss(base), loss(fc(Fte)); d = e1 - e0
    m = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 10}); md, se = float(m.params[0]), float(m.bse[0]); ref = e0.mean()
    ratio, lo, hi = 1 + md / ref, 1 + (md - 1.96 * se) / ref, 1 + (md + 1.96 * se) / ref
    v = "equivalent" if lo > 0.97 and hi < 1.03 else "better" if hi < 1 else "worse" if lo > 1 else "inconclusive"
    print(f"{label:34s} QLIKE exact {e0.mean():.4f}  this {e1.mean():.4f}  ratio {ratio:.4f} [{lo:.4f}, {hi:.4f}]  {v}")
    return ratio
Fh = np.full((len(te), F0.shape[1]), np.nan); got = 0
for j in ("0", "1", "2", "3"):
    try: z = np.load(f"hw/main_job_{j}.npz"); Fh[z["rows"]] = z["f"]; got += len(z["rows"])
    except FileNotFoundError: pass
print("hardware rows measured:", got, "of", len(te))
if got == len(te):
    fe = np.abs(Fh - F0[te]); print(f"feature error hardware vs exact: mean {fe.mean():.4f} max {fe.max():.4f}")
    compare(Fh, "PRIMARY hardware vs exact")
    for k in ("fakefez_2000",): print(f"fake-chip sim feature error {np.abs(nz[k] - F0[te]).mean():.4f}"); compare(nz[k], "context: FakeFez sim vs exact")
    try:
        zr = np.load("hw/main_job_repeat.npz"); rr = zr["rows"]; print(f"drift: repeat-vs-first mean |diff| {np.abs(zr['f'] - Fh[rr]).mean():.4f}  (sampling-only expectation about {np.sqrt(2) * 0.016:.4f})")
    except FileNotFoundError: print("repeat job not available")
    for pr in ("probe_result", "probe2_result"):
        zz = np.load(f"hw/{pr}.npz"); rows = zz["rows"] if "rows" in zz.files else np.array([0]); print(pr, "vs main-run same rows, mean |diff|", round(float(np.abs(np.atleast_2d(zz['f']) - Fh[rows]).mean()), 4))
