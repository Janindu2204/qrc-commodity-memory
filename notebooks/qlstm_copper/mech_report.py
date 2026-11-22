"""Report for long_prereg_mech.json: applies the registered decision rules to mech_out/*.json."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent / "mech_out"
pr = print; cap = {f.stem[4:]: json.load(open(f)) for f in sorted(HERE.glob("cap_*.json"))}
pr("== A. capacity on i.i.d. input (median over seeds/configs; range) ==")
H90 = {}
for fam, rows in cap.items():
    mc = np.array([r["MC"] for r in rows]); c2 = np.array([r["C2"] for r in rows]); h = np.array([r["H90"] for r in rows]); H90[fam] = float(np.median(h))
    pr(f"  {fam:6s} n={len(rows):3d}  MC {np.median(mc):6.2f} [{mc.min():.2f}, {mc.max():.2f}]  C2 {np.median(c2):6.2f} [{c2.min():.2f}, {c2.max():.2f}]  H90 {np.median(h):5.0f} [{h.min()}, {h.max()}]")
if "DOMG1" in cap:
    pr("DOMG1 rule:", "DOMG1 is NOT a memory control (MC < 0.5 for all seeds)" if max(r["MC"] for r in cap["DOMG1"]) < 0.5 else "DOMG1 has memory on some seed")
pr("\n== B. copper HAR-residual task profile (fit 2013-2022, test 2023-2025) ==")
for f in sorted(HERE.glob("copper_*.json")):
    d = json.load(open(f)); L = [x for x in d["fits"] if not x["quad"]]; Q = [x for x in d["fits"] if x["quad"]]
    pr(f"  {d['grid']}: n train {d['n_train']}, test {d['n_test']}")
    for x in d["fits"]: pr(f"    {'Q' if x['quad'] else 'L'}({x['K']:3d}) p={x['p']:5d}  R2 {x['r2']:+.4f}  ratio vs HAR {x['ratio']:.4f}")
    best = max(x["r2"] for x in L)
    if best <= 0: ks = None; pr("    K*: undefined -> no linear lag signal left")
    else:
        ks = min(x["K"] for x in L if x["r2"] >= 0.9 * best); pr(f"    K* = {ks}")
        for q in ("DOM", "ISING"):
            if q in H90: pr(f"    vs {q} H90 {H90[q]:.0f}:", "within the reservoir's memory reach; memory is not what limits it" if ks <= H90[q] else "beyond the reservoir's memory; its memory is too short")
    l10 = next(x for x in L if x["K"] == 10); q10 = next(x for x in Q if x["K"] == 10)
    pr("    nonlinearity:", "short-lag degree-2 structure" if l10["ratio"] - q10["ratio"] > 0.005 else "no short-lag degree-2 structure detectable", f"(L10 {l10['ratio']:.4f}, Q10 {q10['ratio']:.4f})")
    b = min(x["ratio"] for x in d["fits"]); pr("    signal:", f"best ratio {b:.4f} ->", "less than 1% of HAR's error is predictable from lagged ln RV by these regressions" if b > 0.99 else "at least 1% predictable")
