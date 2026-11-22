"""Rebuild the daily copper 2024 rolling-block QRC_angle forecast (qrc_roll.py, TAG roll, YEAR 2024, chosen config layers=1 scale=1.0) and
compare it with the saved long_preds_qrc/roll/range_m/2024/QRC_angle.npz. Also saves per-seed features/forecasts for the hardware plan. Run from qlstm_copper/ with the main venv, PYTHONPATH=."""
import numpy as np
import long_core as lc, v2_core as vc, hourly_qrc as hq
YEAR, NQ, UNITS, LAY, SC = 2024, 6, 21, 1, 1.0
D = lc.prepare("range_m", YEAR, length=1); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
X0 = M[:, :3]; G = np.hstack([X0, np.vstack([np.zeros((1, 3)), X0[:-1]])]); rows_all = np.concatenate([tr, va, te])
print("rows train/val/test", len(tr), len(va), len(te))
def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
    return np.exp(vc.predict_linear(p, X, anc))
LV, FS = [], {}
for s in range(10):
    F = np.zeros((len(M), UNITS)); F[rows_all] = hq.reservoir_features(G[rows_all] * SC, NQ, s, True, LAY); FS[s] = F; LV.append(readout(F))
avg = np.mean(LV, 0); z = np.load(f"long_preds_qrc/roll/range_m/{YEAR}/QRC_angle.npz")
print("max |rebuilt - saved| test:", np.nanmax(np.abs(avg[te] - z["lv_te"])), " val:", np.nanmax(np.abs(avg[va] - z["lv_val"])), " scale of lv_te:", np.nanmean(np.abs(z["lv_te"])))
np.savez("hw/daily2024_rebuild.npz", G=G, te=te, va=va, tr=tr, y=y, anc=anc, lv_seed0=LV[0][te], lv_avg=avg[te], F0=FS[0], inputs_te=G[te] * SC, ytest=y[te])
