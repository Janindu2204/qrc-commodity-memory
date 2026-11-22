"""Parts 1-2 (long_prereg_budget.json). Usage: python budget_run.py   (reads saved forecasts only; writes budget_out/).
Step 1: weekly loss tables per grid. Step 2: all 64 protocols on the real labels, Shapley decomposition. Step 3: null with B = 200 week-swap
replicates for the 32 protocols with REF = control. Step 4: seed lottery on the 50-seed hourly Databento QLSTMT and LSTM forecasts."""
import json, time, glob, os
from pathlib import Path
import numpy as np
import budget_lib as bl

HERE = Path(__file__).parent; OUT = HERE / "budget_out"; OUT.mkdir(exist_ok=True); B = int(os.environ.get("BUDGET_B", 200)); t0 = time.time()
tabs, refsel = {}, {}
for g in bl.GRIDS:
    W, cnt, wy, ids = bl.weekly_tables(g); tabs[g] = (W, cnt, wy, ids); refsel[g] = bl.ref_selection(W, wy)
    print(g, "weeks", len(ids), "refs (MSE):", refsel[g]["MSE"], f"{time.time() - t0:.0f}s", flush=True)
json.dump({g: {L: {str(Y): r for Y, r in v.items()} for L, v in refsel[g].items()} for g in bl.GRIDS}, open(OUT / "refsel.json", "w"), indent=1)
P = bl.all_protocols(); obs = {}
for f in P: obs[tuple(f[k] for k in bl.FACTORS)] = bl.protocol_ratio(tabs, None, refsel, f)
json.dump([dict(zip(bl.FACTORS, k), ratio=v) for k, v in obs.items()], open(OUT / "observed.json", "w"), indent=1)
ph = bl.shapley({k: float(np.log(v)) for k, v in obs.items()}); json.dump(ph, open(OUT / "shapley.json", "w"), indent=1)
print("observed done; control", obs[(False,) * 6], "all pitfalls", obs[(True,) * 6], "shapley", ph, f"{time.time() - t0:.0f}s", flush=True)
# common calendar weeks across grids for the shared flips
allw = np.unique(np.concatenate([tabs[g][3] for g in bl.GRIDS])); sel = [f for f in P if not f["REF"]]; null = {tuple(f[k] for k in bl.FACTORS): [] for f in sel}
for b in range(B):
    coin = np.random.default_rng(41000 + b).random(len(allw)) < 0.5; flips = {g: coin[np.searchsorted(allw, tabs[g][3])] for g in bl.GRIDS}
    for f in sel: null[tuple(f[k] for k in bl.FACTORS)].append(bl.protocol_ratio(tabs, flips, refsel, f))
    if b % 20 == 0: print("null replicate", b, f"{time.time() - t0:.0f}s", flush=True)
json.dump([dict(zip(bl.FACTORS, k), ratios=v) for k, v in null.items()], open(OUT / "null.json", "w"))
# seed lottery (hourly Databento v5, 50 seeds each)
def load(kind):
    fs = sorted(glob.glob(str(HERE / "hourly_databento_preds_v5" / f"seeds2_{kind}_s*.npz")), key=lambda s: int(s.split("_s")[-1][:-4]))
    zs = [np.load(f) for f in fs]; y = zs[0]["y"]; assert all(np.array_equal(z["y"], y) for z in zs); return y, np.stack([z["pred"] for z in zs])
yq, Pq = load("QLSTMT_h16"); yl, Pl = load("LSTM_h5"); assert np.array_equal(yq, yl); y = yq
mse = lambda p: float(np.mean((y - p) ** 2)); lot = {"n_seeds": [len(Pq), len(Pl)], "all_seed_ratio": mse(Pq.mean(0)) / mse(Pl.mean(0)), "k": {}}
rng = np.random.default_rng(42000)
for k in (1, 3, 5, 10, 25):
    r = np.array([mse(Pq[rng.choice(len(Pq), k, replace=False)].mean(0)) / mse(Pl[rng.choice(len(Pl), k, replace=False)].mean(0)) for _ in range(2000)])
    lot["k"][k] = dict(median=float(np.median(r)), p_below_1=float((r < 1).mean()), p_below_097=float((r < 0.97).mean()), p_above_103=float((r > 1.03).mean()), q05=float(np.quantile(r, 0.05)), q95=float(np.quantile(r, 0.95)))
json.dump(lot, open(OUT / "seed_lottery.json", "w"), indent=1); print("seed lottery", lot["all_seed_ratio"], f"{time.time() - t0:.0f}s", flush=True)
