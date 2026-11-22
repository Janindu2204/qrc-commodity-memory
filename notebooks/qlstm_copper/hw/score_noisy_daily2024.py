"""Main venv. Readout fitted on EXACT seed-0 features (train+val as in qrc_roll.py); test features swapped for sampled/noisy ones. Reports the
free-simulation version of the planned hardware test: QLIKE (validation-calibrated scale) and ratio to the exact-simulator forecast."""
import numpy as np
import long_core as lc, v2_core as vc, hourly_qrc as hq
from scipy import stats
D = lc.prepare("range_m", 2024, length=1); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
r = np.load("hw/daily2024_rebuild.npz"); F0 = r["F0"].copy(); nz = np.load("hw/daily2024_noisy.npz")
X = np.hstack([M, F0]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
sc, lam, p = vc.best_lambda_linear((X - mu) / sd, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
def fc(Fte):
    Z = X.copy(); Z[te, M.shape[1]:] = Fte; return np.exp(vc.predict_linear(p, (Z - mu) / sd, anc))
base = fc(F0[te]); c = vc.calib_scale(np.exp(y[va]), base[va]); ok = np.isfinite(y[te]) & np.isfinite(base[te]) & (base[te] > 0)
def q(pred): return vc.qlike_loss(y[te][ok], np.log(c * pred[te][ok]))
def qvec(pred):   # per-row loss for DM: QLIKE of variance forecast f vs realised exp(y): r/f - log(r/f) - 1
    f = c * pred[te][ok]; rr = np.exp(y[te][ok]); return rr / f - np.log(rr / f) - 1
print(f"exact seed-0 QLIKE {q(base):.4f} on {ok.sum()} test rows; saved 10-seed avg lv_te vs seed-0 differ by {np.nanmean(np.abs(np.load('hw/daily2024_rebuild.npz')['lv_avg'] - base[te])):.2e} (mean abs)")
e0 = qvec(base)
for k in nz.files:
    if k == "exact": continue
    pr = fc(nz[k]); e1 = qvec(pr); d = e1 - e0
    print(f"{k:14s} QLIKE {q(pr):.4f}  ratio to exact {e1.mean() / e0.mean():.4f}  mean-diff t (naive) {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}  max|forecast change| {np.max(np.abs(pr[te] / base[te] - 1)):.3f}")
