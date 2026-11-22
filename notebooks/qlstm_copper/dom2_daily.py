"""One daily block of long_prereg_dom2.json. Usage: python dom2_daily.py YEAR   YEAR = 2015..2024, or 2025 / 2026 (held-out step).
Data, target, blocks, input series, scaling and QLIKE readout as dom_daily.py. New models: ISING (dom2_lib, 10 seeds, no tuning) and ESN28u_t
(12 configurations radius {0.6,0.9,1.2} x leak {0.1,0.2,0.5,1.0}, chosen by validation QLIKE with tuning seeds 100-102 as db_run.py, then seeds 0-9).
Writes long_preds_dom2/D/<YEAR>/<MODEL>.npz and appends the chosen configuration to dom2_daily_configs.csv. Scores nothing on test rows.
"""
import sys, csv, itertools
from pathlib import Path
import numpy as np, pandas as pd
import long_core as lc, v2_core as vc, dom_lib as dl, dom2_lib as d2

year = int(sys.argv[1]); NSEED = 10; TUNE = (100, 101, 102)
data = pd.read_csv(lc.ROOT.parents[0] / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date")
D = lc.prepare("range_m", year, length=1, data=data); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
out = Path(__file__).with_name("long_preds_dom2") / "D" / str(year); out.mkdir(parents=True, exist_ok=True)
x = dl.hold(np.log(D["rv"].values)); lo, hi = float(x[tr].min()), float(x[tr].max()); u = np.clip((x - lo) / (hi - lo), 0.0, 1.0); zs = (x - x[tr].mean()) / x[tr].std()


def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)); return np.exp(vc.predict_linear(p, X, anc))


def val_score(fn):
    lv = np.mean([fn(s) for s in TUNE], 0)[va]; ok = np.isfinite(lv) & (lv > 0); ev = lv[ok]
    c = vc.calib_scale(np.exp(y[va][ok]), ev); return vc.qlike_loss(y[va][ok], np.log(c * ev))


grid = [dict(rho=r, leak=k) for r, k in itertools.product((0.6, 0.9, 1.2), (0.1, 0.2, 0.5, 1.0))]
cfg = grid[int(np.argmin([val_score(lambda s, c=c: readout(dl.esn1(zs, s, leak=c["leak"], rho=c["rho"]))) for c in grid]))]
with open(Path(__file__).with_name("dom2_daily_configs.csv"), "a", newline="") as fh: csv.writer(fh).writerow([year, "ESN28u_t", cfg, len(grid)])
for name, fn in (("ESN28u_t", lambda s: dl.esn1(zs, s, leak=cfg["leak"], rho=cfg["rho"])), ("ISING", lambda s: d2.ising_features(u, s))):
    lv = np.mean([readout(fn(s)) for s in range(NSEED)], 0); np.savez(out / f"{name}.npz", lv_val=np.asarray(lv[va], float), lv_te=np.asarray(lv[te], float))
print(year, "dom2 daily done", cfg, flush=True)
