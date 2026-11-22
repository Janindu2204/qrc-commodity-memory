"""One daily block of long_prereg_dom3.json. Usage: python dom3_daily.py YEAR   YEAR = 2015..2024, or 2025 / 2026 (held-out step).
Data, target, blocks, input series, scaling and QLIKE readout as dom_daily.py / dom2_daily.py. New models:
ESNq_t, SAS28_t (dom3_lib; 12 configurations each, chosen by validation QLIKE with tuning seeds 100-102, then seeds 0-9; HAR features + 28 features, linear QLIKE readout);
LSTM, GRU (dom3_lib.fit_daily: last 5 days of the three HAR inputs, QLIKE loss anchored on lvR5 as long_nn.py; 12 configurations, tuning seeds 100-102, then seeds 0-9).
Writes long_preds_dom3/D/<YEAR>/<MODEL>.npz and appends the chosen configurations to dom3_daily_configs.csv. Scores nothing on test rows.
"""
import sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import long_core as lc, v2_core as vc, dom_lib as dl, dom3_lib as d3

year = int(sys.argv[1]); NSEED = 10; TUNE = (100, 101, 102)
data = pd.read_csv(lc.ROOT.parents[0] / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date")
D = lc.prepare("range_m", year, length=1, data=data); y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
out = Path(__file__).with_name("long_preds_dom3") / "D" / str(year); out.mkdir(parents=True, exist_ok=True)
x = dl.hold(np.log(D["rv"].values)); lo, hi = float(x[tr].min()), float(x[tr].max()); u = np.clip((x - lo) / (hi - lo), 0.0, 1.0); zs = (x - x[tr].mean()) / x[tr].std()
cc = np.sqrt(u * (1 - u)); Z = np.column_stack([zs, (cc - cc[tr].mean()) / (cc[tr].std() + 1e-9)])


def readout(F):
    X = np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)); return np.exp(vc.predict_linear(p, X, anc))


def val_score(fn):
    lv = np.mean([fn(s) for s in TUNE], 0)[va]; ok = np.isfinite(lv) & (lv > 0); ev = lv[ok]
    if len(ev) < 20: return np.inf
    c = vc.calib_scale(np.exp(y[va][ok]), ev); return vc.qlike_loss(y[va][ok], np.log(c * ev))


MODELS = {"ESNq_t": (d3.ESN_GRID, lambda s, c: readout(d3.esn_k(Z, s, leak=c["leak"], rho=c["rho"]))), "SAS28_t": (d3.SAS_GRID, lambda s, c: readout(d3.sas(u, s, leak=c["leak"], gain=c["gain"]))),
          "LSTM": (d3.NN_GRID, lambda s, c: d3.fit_daily("LSTM", c, s, M, anc, y, tr, va)), "GRU": (d3.NN_GRID, lambda s, c: d3.fit_daily("GRU", c, s, M, anc, y, tr, va))}
for name, (grid, fn) in MODELS.items():
    cfg = grid[int(np.argmin([val_score(lambda s, c=c: fn(s, c)) for c in grid]))]
    with open(Path(__file__).with_name("dom3_daily_configs.csv"), "a", newline="") as fh: csv.writer(fh).writerow([year, name, cfg, len(grid)])
    lv = np.nanmean([fn(s, cfg) for s in range(NSEED)], 0); np.savez(out / f"{name}.npz", lv_val=np.asarray(lv[va], float), lv_te=np.asarray(lv[te], float))
    print(year, name, cfg, flush=True)
print(year, "dom3 daily done", flush=True)
