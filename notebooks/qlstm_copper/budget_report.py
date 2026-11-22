"""Report for long_prereg_budget.json: applies the registered rules to budget_out/*.json."""
import json
from pathlib import Path
import numpy as np
import budget_lib as bl
O = Path(__file__).parent / "budget_out"; pr = print
obs = {tuple(d[k] for k in bl.FACTORS): d["ratio"] for d in json.load(open(O / "observed.json"))}
null = {tuple(d[k] for k in bl.FACTORS): np.array(d["ratios"]) for d in json.load(open(O / "null.json"))}; ph = json.load(open(O / "shapley.json"))
key = lambda **on: tuple(bool(on.get(k, False)) for k in bl.FACTORS); A = lambda r: 100 * (1 - r)
c = obs[key()]; pr(f"== control protocol (DOM, 10 seeds, pooled 2014-2025, geometric mean over F/M/H, MSE, past-selected classical) ==\n  ratio {c:.4f}  apparent advantage {A(c):+.2f}%")
pr("\n== single pitfalls (others at control): observed apparent advantage; null median and 95th percentile (selection pitfalls) ==")
for fct in bl.FACTORS:
    k = key(**{fct: True}); o = obs[k]
    if fct == "REF": pr(f"  {fct:6s} observed {A(o):+.2f}%  (systematic: no null; change from control {A(o) - A(c):+.2f} points)"); continue
    n = null[k]; pr(f"  {fct:6s} observed {A(o):+.2f}%  null median {A(np.median(n)):+.2f}%  null 95th {A(np.quantile(n, 0.05)):+.2f}%  ->",
                    "within what selection alone produces" if o >= np.quantile(n, 0.05) else "beyond what selection alone produces")
sel_all = key(SEED=True, WINDOW=True, GRID=True, LOSS=True, QMODEL=True); full = key(SEED=True, WINDOW=True, GRID=True, LOSS=True, QMODEL=True, REF=True)
n = null[sel_all]; o = obs[sel_all]
pr(f"\n== all five selection pitfalls (REF control) ==\n  observed {A(o):+.2f}%  null median {A(np.median(n)):+.2f}%  null 95th {A(np.quantile(n, 0.05)):+.2f}%  ->",
   "the apparent advantage is no larger than selection alone produces when neither model is better" if o >= np.quantile(n, 0.05) else "the apparent advantage exceeds what selection alone produces")
pr(f"\n== all six pitfalls ==\n  observed apparent advantage {A(obs[full]):+.2f}%  (control {A(c):+.2f}%)")
tot = sum(ph.values()); pr("  Shapley decomposition of the change in log ratio, control -> all six (share of total; negative log = adds apparent advantage):")
for k, v in sorted(ph.items(), key=lambda kv: kv[1]): pr(f"    {k:6s} {100 * -v:+.2f} points  ({100 * v / tot:5.1f}% of the total)")
pr(f"    sum {100 * -tot:+.2f} points = log change {np.log(obs[full]) - np.log(c):+.4f}")
lot = json.load(open(O / "seed_lottery.json")); pr(f"\n== seed lottery, hourly QLSTMT vs LSTM (50 seeds each): all-seed ratio {lot['all_seed_ratio']:.4f} ==")
for k, d in lot["k"].items(): pr(f"  k={k:>2}: median {d['median']:.4f}  5-95% [{d['q05']:.4f}, {d['q95']:.4f}]  P(<1) {d['p_below_1']:.2f}  P(<0.97) {d['p_below_097']:.2f}  P(>1.03) {d['p_above_103']:.2f}")
