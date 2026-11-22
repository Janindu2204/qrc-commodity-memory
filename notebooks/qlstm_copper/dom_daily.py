"""One daily block of long_prereg_dom.json (Databento daily range target, as long_prereg_dbdaily.json). Usage: python dom_daily.py YEAR   YEAR = 2015..2024, or 2025 / 2026 (held-out step).
Data preparation, target (ln of 5-day forward mean range variance), blocks, QLIKE readout (HAR features + 28 reservoir features, ridge on validation) are those of db_run.py.
New models: DOM, DOMG1 (dom_lib, one input = ln daily range variance, run along the whole series) and ESN28u (28-unit echo state network on the same input).
Writes long_preds_dom/D/<YEAR>/<MODEL>.npz (lv_val, lv_te). Scores nothing.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import long_core as lc, v2_core as vc, dom_lib as dl

year = int(sys.argv[1]); NSEED = 10
data = pd.read_csv(lc.ROOT.parents[0] / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date")
D = lc.prepare("range_m", year, length=1, data=data); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
out = Path(__file__).with_name("long_preds_dom") / "D" / str(year); out.mkdir(parents=True, exist_ok=True)
x = dl.hold(np.log(D["rv"].values)); lo, hi = float(x[tr].min()), float(x[tr].max()); u = np.clip((x - lo) / (hi - lo), 0.0, 1.0); zs = (x - x[tr].mean()) / x[tr].std()


def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)); return np.exp(vc.predict_linear(p, X, anc))


for name, fn in (("DOM", lambda s: dl.features(u, s, "G3")), ("DOMG1", lambda s: dl.features(u, s, "G1")), ("ESN28u", lambda s: dl.esn1(zs, s))):
    lv = np.mean([readout(fn(s)) for s in range(NSEED)], 0); np.savez(out / f"{name}.npz", lv_val=np.asarray(lv[va], float), lv_te=np.asarray(lv[te], float))
print(year, "rows train/val/test", len(tr), len(va), len(te), "dom daily done", flush=True)
