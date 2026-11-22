"""Part 4 (long_prereg_mech.json): task profile of copper's HAR residual. Usage: python mech_copper.py GRID   (H | M | F).
Rows, target and HAR benchmark (with hour-of-day) are built exactly as in dom_run.py (code copied), for ONE split: fit on target years
2013-2022, evaluate on 2023-2025 (24-hour purge). 2026 is not read. Then the HAR residual r is regressed (ridge, lambda = 1e-3 * n, on
standardised features) on: L(K) = the last K values of ln RV (K = 1, 2, 5, 10, 20, 50, 100, 200) and Q(K) = L(K) plus all degree-2 products of
the last K values (K = 2, 5, 10). Score: out-of-sample R^2 of r, and the implied MSE ratio (HAR + fit) / HAR on 2023-2025.
Writes mech_out/copper_<GRID>.json. These are descriptive regressions, not a model of the paper's comparison."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import dom_lib as dl

GRID = sys.argv[1]; ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent; PURGE_H = 24; SEQ = 10
LONG = {"H": (24, 120), "M": (96, 480), "F": (288, 1440)}[GRID]
z = np.load(ROOT / "data" / "hf" / f"{GRID}.npz"); t = pd.to_datetime(z["t"]); N = len(t)
ret, lrv0, lvol, hs, hc, adj, rvpos, hodbin = (z[k] for k in ("ret", "lrv", "lvol", "hs", "hc", "adj", "rvpos", "hodbin"))
lrv = np.where(rvpos, lrv0, np.nan); S = pd.Series(lrv)
l_a = S.rolling(LONG[0], min_periods=LONG[0] // 2).mean().to_numpy(); l_b = S.rolling(LONG[1], min_periods=LONG[1] // 2).mean().to_numpy()
CH = np.stack([ret, lrv, lvol, hs, hc], 1); yr_bin = np.r_[t.year.values[1:], 0]
pos = np.arange(SEQ - 1, N - 1); pos = pos[(yr_bin[pos] >= 2013) & (yr_bin[pos] <= 2025)]
okw = np.ones(len(pos), bool)
for k in range(SEQ): okw &= np.isfinite(CH[pos - k]).all(1)
okw &= adj[pos + 1] & rvpos[pos + 1] & np.isfinite(l_a[pos]) & np.isfinite(l_b[pos]); pos = pos[okw]; tt = t[pos + 1]; yr = tt.year.values; y = lrv[pos + 1]
Hm = np.column_stack([np.ones(len(pos)), lrv[pos], np.mean([lrv[pos - k] for k in range(SEQ)], 0), l_a[pos], l_b[pos],
                      pd.get_dummies(hodbin[pos + 1]).reindex(columns=range(24), fill_value=0).astype(float).values[:, 1:]])
te_m = yr >= 2023; tr_m = (yr <= 2022) & (tt < tt[te_m].min() - pd.Timedelta(hours=PURGE_H)); tr, te = np.where(tr_m)[0], np.where(te_m)[0]
beta = np.linalg.lstsq(Hm[tr], y[tr], rcond=None)[0]; base = Hm @ beta; r = y - base; mse_har = float(np.mean(r[te] ** 2))
x = dl.hold(lrv); xm, xs = float(np.nanmean(lrv[pos[tr]])), float(np.nanstd(lrv[pos[tr]])); x = (x - xm) / xs
print(GRID, "rows train/test", len(tr), len(te), flush=True)


def feats(rows, K, quad):
    L = np.stack([x[pos[rows] - k] for k in range(K)], 1).astype(np.float64)
    if not quad: return L
    return np.column_stack([L] + [L[:, i] * L[:, j] for i in range(K) for j in range(i, K)])


def fit_score(K, quad, chunk=200000):
    p = feats(tr[:10], K, quad).shape[1]; G = np.zeros((p + 1, p + 1)); b = np.zeros(p + 1); mu = np.zeros(p); m2 = np.zeros(p); n = 0
    for i in range(0, len(tr), chunk):
        F = feats(tr[i:i + chunk], K, quad); mu += F.sum(0); m2 += (F ** 2).sum(0); n += len(F)
    mu /= n; sd = np.sqrt(m2 / n - mu ** 2) + 1e-12
    for i in range(0, len(tr), chunk):
        F = np.column_stack([np.ones(min(chunk, len(tr) - i)), (feats(tr[i:i + chunk], K, quad) - mu) / sd]); G += F.T @ F; b += F.T @ r[tr[i:i + chunk]]
    R = 1e-3 * n * np.eye(p + 1); R[0, 0] = 0; w = np.linalg.solve(G + R, b); sse = 0.0
    for i in range(0, len(te), chunk):
        F = np.column_stack([np.ones(min(chunk, len(te) - i)), (feats(te[i:i + chunk], K, quad) - mu) / sd]); sse += float(((r[te[i:i + chunk]] - F @ w) ** 2).sum())
    mse = sse / len(te); return dict(K=K, quad=quad, p=p, r2=1 - mse / float(np.var(r[te])), ratio=mse / mse_har)


out = []; t0 = time.time()
for K in (1, 2, 5, 10, 20, 50, 100, 200): out.append(fit_score(K, False)); print(out[-1], f"{time.time() - t0:.0f}s", flush=True)
for K in (2, 5, 10): out.append(fit_score(K, True)); print(out[-1], f"{time.time() - t0:.0f}s", flush=True)
(HERE / "mech_out").mkdir(exist_ok=True); json.dump(dict(grid=GRID, n_train=int(len(tr)), n_test=int(len(te)), mse_har=mse_har, fits=out), open(HERE / "mech_out" / f"copper_{GRID}.json", "w"), indent=1)
