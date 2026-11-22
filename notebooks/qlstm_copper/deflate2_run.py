"""long_prereg_deflate2.json. Usage: python deflate2_run.py   Writes deflate2_out.json; prints the report (deflate2_report.txt)."""
import json
import numpy as np
from scipy import stats
import deflate_lib as dl, deflate2_lib as d2
ALL = {}
for k, f in d2.LOADERS.items():
    D = f(); C = [d2.finish(c) for c in dl.cells(D)]
    if k.startswith("smalln"): C += [d2.finish(c) for c in d2.trainsize_cells(D)]
    for c in C: c["key"] = k; c.setdefault("dataset", D["name"])
    ALL[k] = C
json.dump(ALL, open(dl.HERE / "deflate2_out.json", "w"), indent=1, default=float)
pct = lambda x: f"{100 * x:+.2f}"
for k, C in ALL.items():
    print(f"\n== {k}: {C[0]['dataset']} ==")
    for c in C:
        h = {True: "pass", False: "FAIL", None: "n/a"}[c["H1_pass"]]
        print(f"  {c['choice']:9s} {c['unit']:8s} N={c['N']:3d}  G_obs {pct(c['G_obs'])}  G_pred {pct(c['G_pred'])}  deflated-honest {pct(c['H1_err'])}  H1 {h}  bound-honest {pct(c['bound'] - c['rc'])}")
print("\n== H2 replication (Spearman across choices, unit-averaged) ==")
n_ok = 0
for k, C in ALL.items():
    ch = sorted(set(c["choice"] for c in C)); gp = [np.mean([c["G_pred"] for c in C if c["choice"] == x]) for x in ch]; go = [np.mean([c["G_obs"] for c in C if c["choice"] == x]) for x in ch]
    rho = float(stats.spearmanr(gp, go).statistic); n_ok += rho >= 0.6
    print(f"  {k:13s} {ch}  Spearman {rho:+.2f}  pred {[round(100 * v, 2) for v in gp]}  obs {[round(100 * v, 2) for v in go]}")
print(f"  datasets with Spearman >= 0.6: {n_ok} of {len(ALL)} ->", "H2 REPLICATED" if n_ok >= 3 else "H2 NOT REPLICATED")
H = [c for C in ALL.values() for c in C if c["H1_pass"] is not None]
print(f"\n== H1 re-test (non-seed cells, N >= 4): {sum(c['H1_pass'] for c in H)}/{len(H)} pass ->", "SUPPORTED" if np.mean([c["H1_pass"] for c in H]) >= 0.75 else "NOT SUPPORTED")
