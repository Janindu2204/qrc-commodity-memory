"""Fix B (long_prereg_budget2.json): year-block sign-flip null for the manufactured-advantage budget. Usage: python budget2_run.py
budget_lib unchanged. In each replicate, whole test years (2014-2025, shared across grids) are swapped between the quantum model and the
reference with probability 1/2, which keeps the real size of year-to-year differences and makes the two models exchangeable by year.
Protocols: control, each single selection pitfall (REF control), all five selection pitfalls. Gain = log ratio(control) - log ratio(protocol),
computed inside the same replicate. B = 1000 draws (seeds 43000+b). Writes budget_out/null_year.json and prints the report."""
import json, time
from pathlib import Path
import numpy as np
import budget_lib as bl
HERE = Path(__file__).parent; OUT = HERE / "budget_out"; t0 = time.time(); B = 1000
tabs, refsel = {}, {}
for g in bl.GRIDS:
    W, cnt, wy, ids = bl.weekly_tables(g); tabs[g] = (W, cnt, wy, ids); refsel[g] = bl.ref_selection(W, wy)
print("tables", f"{time.time() - t0:.0f}s", flush=True)
key = lambda **on: {k: bool(on.get(k, False)) for k in bl.FACTORS}
PROT = {"control": key(), **{f: key(**{f: True}) for f in bl.FACTORS[:5]}, "ALL5": key(SEED=True, WINDOW=True, GRID=True, LOSS=True, QMODEL=True)}
obs = {n: bl.protocol_ratio(tabs, None, refsel, f) for n, f in PROT.items()}
null = {n: [] for n in PROT}
for b in range(B):
    fy = dict(zip(bl.TEST, np.random.default_rng(43000 + b).random(len(bl.TEST)) < 0.5)); flips = {g: np.array([fy.get(int(Y), False) for Y in tabs[g][2]]) for g in bl.GRIDS}
    for n, f in PROT.items(): null[n].append(bl.protocol_ratio(tabs, flips, refsel, f))
    if b % 100 == 0: print("replicate", b, f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(observed=obs, null=null), open(OUT / "null_year.json", "w"))
print("\n== year-block sign-flip null (B = 1000): gain a pitfall adds over the control, in log points ==")
c0 = np.log(obs["control"]); cn = np.log(np.array(null["control"]))
for n in PROT:
    if n == "control": continue
    go = 100 * (c0 - np.log(obs[n])); gn = 100 * (cn - np.log(np.array(null[n]))); q95 = float(np.quantile(gn, 0.95)); pv = float(np.mean(gn >= go))
    print(f"  {n:6s} observed gain {go:6.2f}  null median {np.median(gn):6.2f}  null 95th {q95:6.2f}  share of null >= observed {pv:.3f}  ->", "beyond chance" if go > q95 else "within chance")
print(f"  null control ratio median {np.median(null['control']):.4f} (observed control {obs['control']:.4f})")
