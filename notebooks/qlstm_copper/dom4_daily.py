"""One daily block of long_prereg_dom4.json. Usage: python dom4_daily.py YEAR   YEAR = 2015..2024, or 2025 / 2026 (held-out step).
Data, target, blocks, input series, scaling, readout and tuning as dom3_daily.py. New models: ESNb_t, SASb_t (dom4_lib, 256-number state, 28 projections);
LSTM1, GRU1 (dom3_lib.fit_daily reading ONE input, the last 5 days of ln daily range variance (lvR1); anchor lvR5 as before).
Writes long_preds_dom4/D/<YEAR>/<MODEL>.npz and appends the chosen configurations to dom4_daily_configs.csv. Scores nothing on test rows.
"""
import sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import long_core as lc, v2_core as vc, dom_lib as dl, dom3_lib as d3, dom4_lib as d4

year = int(sys.argv[1]); NSEED = 10; TUNE = (100, 101, 102)
data = pd.read_csv(lc.ROOT.parents[0] / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date")
D = lc.prepare("range_m", year, length=1, data=data); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
out = Path(__file__).with_name("long_preds_dom4") / "D" / str(year); out.mkdir(parents=True, exist_ok=True)
x = dl.hold(np.log(D["rv"].values)); lo, hi = float(x[tr].min()), float(x[tr].max()); u = np.clip((x - lo) / (hi - lo), 0.0, 1.0); zs = (x - x[tr].mean()) / x[tr].std()
cc = np.sqrt(u * (1 - u)); Z = np.column_stack([zs, (cc - cc[tr].mean()) / (cc[tr].std() + 1e-9)])


def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)); return np.exp(vc.predict_linear(p, X, anc))


def val_score(fn):
    lv = np.mean([fn(s) for s in TUNE], 0)[va]; ok = np.isfinite(lv) & (lv > 0); ev = lv[ok]
    if len(ev) < 20: return np.inf
    c = vc.calib_scale(np.exp(y[va][ok]), ev); return vc.qlike_loss(y[va][ok], np.log(c * ev))


MODELS = {"ESNb_t": (d3.ESN_GRID, lambda s, c: readout(d4.esn_big(Z, s, leak=c["leak"], rho=c["rho"]))), "SASb_t": (d3.SAS_GRID, lambda s, c: readout(d4.sas_big(u, s, leak=c["leak"], gain=c["gain"]))),
          "LSTM1": (d3.NN_GRID, lambda s, c: d3.fit_daily("LSTM", c, s, M[:, :1], anc, y, tr, va)), "GRU1": (d3.NN_GRID, lambda s, c: d3.fit_daily("GRU", c, s, M[:, :1], anc, y, tr, va))}
for name, (grid, fn) in MODELS.items():
    cfg = grid[int(np.argmin([val_score(lambda s, c=c: fn(s, c)) for c in grid]))]
    with open(Path(__file__).with_name("dom4_daily_configs.csv"), "a", newline="") as fh: csv.writer(fh).writerow([year, name, cfg, len(grid)])
    lv = np.nanmean([fn(s, cfg) for s in range(NSEED)], 0); np.savez(out / f"{name}.npz", lv_val=np.asarray(lv[va], float), lv_te=np.asarray(lv[te], float))
    print(year, name, cfg, flush=True)
print(year, "dom4 daily done", flush=True)
