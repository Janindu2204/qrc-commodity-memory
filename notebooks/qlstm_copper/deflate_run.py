"""long_prereg_deflate.json. Usage: python deflate_run.py   Writes deflate_out.json and prints the report (deflate_report.txt)."""
import json
import numpy as np
from scipy import stats
import deflate_lib as dl
ALL = []
for k in ["copper", "panel", "kernels", "adaptive", "hourly_roll"]:
    D = dl.LOADERS[k](); C = dl.cells(D)
    for c in C: c["key"] = k; c["insample"] = D["insample"]; c["null_median"], c["null_pred"] = dl.signflip_null(c) if c["N"] >= 2 else (None, None)
    ALL += C
json.dump(ALL, open(dl.HERE / "deflate_out.json", "w"), indent=1)
pct = lambda x: f"{100 * x:+.2f}"
for k in ["copper", "panel", "kernels", "adaptive", "hourly_roll"]:
    C = [c for c in ALL if c["key"] == k]; print(f"\n== {C[0]['dataset']} ({'in-sample' if C[0]['insample'] else 'OUT-OF-SAMPLE'}) ==")
    for c in C:
        h = {True: "pass", False: "FAIL", None: "n/a (N<4)"}[c["H1_pass"]]
        print(f"  {c['choice']:6s} {c['unit']:10s} N={c['N']:3d}  G_obs {pct(c['G_obs'])}  G_pred {pct(c['G_pred'])}  deflated-honest {pct(c['err'])} (tol {100 * c['tol']:.2f})  H1 {h}"
              f"  | null median {pct(c['null_median'])} vs rms rule {pct(c['null_pred'])}")
print("\n== H1 (deflation recovers the honest result), cells with N >= 4 ==")
for name, sel in (("in-sample (copper, panel)", lambda c: c["insample"]), ("OUT-OF-SAMPLE (kernels, adaptive, hourly_roll)", lambda c: not c["insample"])):
    H = [c for c in ALL if sel(c) and c["H1_pass"] is not None]; print(f"  {name}: {sum(c['H1_pass'] for c in H)}/{len(H)} pass ({100 * np.mean([c['H1_pass'] for c in H]):.0f}%)")
oos = [c for c in ALL if not c["insample"] and c["H1_pass"] is not None]; per = {k: np.mean([c["H1_pass"] for c in oos if c["key"] == k]) for k in ("kernels", "adaptive", "hourly_roll")}
ok = np.mean([c["H1_pass"] for c in oos]) >= 0.75 and min(per.values()) >= 0.5
print("  per out-of-sample dataset:", {k: f"{100 * v:.0f}%" for k, v in per.items()}, "->", "H1 SUPPORTED" if ok else "H1 NOT SUPPORTED")
print("\n== H2 (ranking of choices by G_pred matches ranking by G_obs; unit-averaged per choice) ==")
res = {}
for k in ["copper", "panel", "kernels", "adaptive", "hourly_roll"]:
    C = [c for c in ALL if c["key"] == k]; ch = sorted(set(c["choice"] for c in C))
    gp = [np.mean([c["G_pred"] for c in C if c["choice"] == x]) for x in ch]; go = [np.mean([c["G_obs"] for c in C if c["choice"] == x]) for x in ch]
    rho = float(stats.spearmanr(gp, go).statistic) if len(ch) >= 3 else float("nan"); res[k] = rho
    print(f"  {k:12s} choices {ch}  Spearman {rho:+.2f}  pred {[round(100 * v, 2) for v in gp]}  obs {[round(100 * v, 2) for v in go]}")
n_ok = sum(res[k] >= 0.6 for k in ("kernels", "adaptive", "hourly_roll")); print("  out-of-sample datasets with Spearman >= 0.6:", n_ok, "of 3 ->", "H2 SUPPORTED" if n_ok >= 2 else "H2 NOT SUPPORTED")
