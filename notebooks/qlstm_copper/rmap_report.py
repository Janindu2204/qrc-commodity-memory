"""Report for long_prereg_rmap.json. Usage: python rmap_report.py   (reads rmap_preds/*.npz; writes rmap_report.csv, rmap_report.txt, rmap_map.png).
Per dataset: test MSE of each model. CB = the best classical student (ESN28u_t, SAS28_t, LSTM1) chosen by VALIDATION MSE.
Per cell (memory, structure, share): 10 independent datasets; ratio = exp(mean log MSE ratio), 95% t interval (9 df), two-sided one-sample t
test on the log ratios; Holm within each family over all cells. Categories exactly as dom3_report.category (+/-3% band).
Copper placement reads (does not recompute) the single-input LSTM1 and GRU1 vs HAR pooled rolling ratios in dom4_report_rolling.csv.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

import os
HERE = Path(__file__).parent; PD = Path(os.environ.get("RMAP_PREDS", HERE / "rmap_preds")); OUTD = Path(os.environ.get("RMAP_OUT", HERE)); MARGIN = 0.03; CLS = ["ESN28u_t", "SAS28_t", "LSTM1"]


def category(r):
    if r["p_holm"] < 0.05 and r["ci_hi"] < 1 - MARGIN: return "meaningful edge"
    if r["p_holm"] < 0.05 and r["ci_hi"] < 1: return "detectable, below band"
    if r["p_holm"] < 0.05 and r["ci_lo"] > 1: return "detectably worse"
    return "no detectable difference"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
rows = []
for f in sorted(PD.glob("*.npz")):
    z = np.load(f); mem, st, sh, rep = f.stem.split("_"); V = json.loads(str(z["val"])); y = z["y"]
    mse = {k[2:]: float(np.mean((y - z[k]) ** 2)) for k in z.files if k.startswith("p_")}; cb = min(CLS, key=lambda m: V[m])
    rows.append(dict(memory=mem, structure=st, share=float(sh), rep=int(rep), cb=cb, **{f"mse_{k}": v for k, v in mse.items()}, mse_CB=mse[cb]))
D = pd.DataFrame(rows)
COMP = {"Q1 DOM vs best classical": ("DOM", "CB"), "Q1 ISING vs best classical": ("ISING", "CB"),
        "Q2 DOM vs LSTM1": ("DOM", "LSTM1"), "Q2 ISING vs LSTM1": ("ISING", "LSTM1"), "Q2 DOM vs SAS28_t": ("DOM", "SAS28_t"), "Q2 ISING vs SAS28_t": ("ISING", "SAS28_t"),
        "R best classical vs HAR": ("CB", "HAR"), "R DOM vs HAR": ("DOM", "HAR"), "R ISING vs HAR": ("ISING", "HAR"), "R oracle vs HAR": ("ORACLE", "HAR"),
        "R teacher circuit vs best classical": ("OR", "CB")}
out = []
for fam, (m, ref) in COMP.items():
    for (mem, st, sh), g in D.groupby(["memory", "structure", "share"]):
        mm = {"OR": "DOM_or" if st == "QDOM" else "ISING_or"}.get(m, m)
        if f"mse_{mm}" not in g or g[f"mse_{mm}"].isna().all(): continue
        lr = np.log(g[f"mse_{mm}"].to_numpy() / g[f"mse_{ref}"].to_numpy()); n = len(lr); mu = lr.mean(); se = lr.std(ddof=1) / np.sqrt(n)
        q = stats.t.ppf(0.975, n - 1); p = float(2 * stats.t.sf(abs(mu / se), n - 1)) if se > 0 else 1.0
        out.append(dict(family=fam, model=mm, ref=ref, memory=mem, structure=st, share=sh, n=n, ratio=np.exp(mu), ci_lo=np.exp(mu - q * se), ci_hi=np.exp(mu + q * se), p=p))
O = pd.DataFrame(out); O["p_holm"] = np.nan
for fam, g in O.groupby("family"): O.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
O["category"] = [category(r) for _, r in O.iterrows()]; O.to_csv(OUTD / "rmap_report.csv", index=False)

pr = lambda *a: print(*a)
pr(f"datasets read: {len(D)}  (cells: {D.groupby(['memory', 'structure', 'share']).ngroups}; reps per cell min {D.groupby(['memory', 'structure', 'share']).size().min()})")
pr("best classical chosen by validation:", D["cb"].value_counts().to_dict())
for fam in COMP:
    g = O[O.family == fam]
    if g.empty: continue
    pr(f"\n== {fam} ==")
    for _, r in g.sort_values(["memory", "structure", "share"]).iterrows():
        pr(f"  {r.memory:5s} {r.structure:6s} share {r.share:4.2f}  ratio {r.ratio:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]  p_holm {r.p_holm:.3g}  {r.category}")

# decision rules (fixed in the registration)
Q1 = O[O.family.str.startswith("Q1")]; win = Q1[Q1.category == "meaningful edge"]; det = Q1[Q1.category == "detectable, below band"]
orc = O[(O.family == "R teacher circuit vs best classical") & (O.share >= 0.15)]
valid = bool((orc.category == "meaningful edge").any() or (orc.category == "detectable, below band").any())
pr("\n== decision ==")
pr("pipeline check (teacher's own circuit beats best classical, detectably, in at least one quantum-generated cell with share >= 0.15):", "PASSED" if valid else "FAILED -> map inconclusive")
pr("win region (meaningful edge, Q1):", "none" if win.empty else "; ".join(f"{r.model} {r.memory} {r.structure} {r.share}" for _, r in win.iterrows()))
pr("detectable but below band (Q1):", "none" if det.empty else "; ".join(f"{r.model} {r.memory} {r.structure} {r.share}" for _, r in det.iterrows()))
qn = {"QDOM", "QISING"}
def rule_of(sel):
    if sel.empty: return "no edge anywhere"
    return "edge only on quantum-generated data" if set(sel.structure) <= qn else "edge also on classically generated data"
rule = "W0 inconclusive" if not valid else {"no edge anywhere": "W1", "edge only on quantum-generated data": "W2", "edge also on classically generated data": "W3"}[rule_of(win)] + " " + rule_of(win)
pr("secondary (detectable, below band):", rule_of(det))
pr("outcome rule:", rule)

# descriptive: share of the attainable gain over HAR that each model captures, (HAR - model) / (HAR - ORACLE), mean over datasets
pr("\n== descriptive: share of attainable gain captured (mean over datasets; cells with share >= 0.05) ==")
cap = D[D.share >= 0.05].copy()
for m in ["DOM", "ISING", "ESN28u_t", "SAS28_t", "LSTM1", "CB"]: cap[m] = (cap.mse_HAR - cap[f"mse_{m}"]) / (cap.mse_HAR - cap.mse_ORACLE)
print(cap.groupby(["memory", "structure", "share"])[["DOM", "ISING", "ESN28u_t", "SAS28_t", "LSTM1", "CB"]].mean().round(3).to_string())

# copper placement (read only)
C = pd.read_csv(HERE / "dom4_report_rolling.csv"); C = C[(C.ref == "HAR") & C.model.isin(["LSTM1", "GRU1"]) & (C.stratum == "all")]
cb_har = O[O.family == "R best classical vs HAR"]
pr("\n== copper placement (copper numbers read from dom4_report_rolling.csv; nothing recomputed) ==")
for grid, g in C.groupby("grid"):
    cu = float(g.ratio.min()); near = cb_har[(cb_har.ratio - cu).abs() <= 0.01]
    near_q = Q1.merge(near[["memory", "structure", "share"]], on=["memory", "structure", "share"])
    pr(f"  {grid}: copper best single-input network / HAR = {cu:.4f}; synthetic cells within +/-0.01: {len(near)}; their Q1 categories: {near_q.groupby('category').size().to_dict()}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for i, mem in enumerate(["long", "short"]):
        for j, q in enumerate(["DOM", "ISING"]):
            g = Q1[(Q1.memory == mem) & (Q1.model == q) & (Q1.structure != "NULL")]; S = sorted(g.structure.unique()); sh = sorted(g.share.unique())
            if g.empty or len(g) != len(S) * len(sh): ax[i, j].set_axis_off(); continue
            M = np.array([[g[(g.structure == s) & (g.share == h)].ratio.mean() for h in sh] for s in S])
            im = ax[i, j].imshow(M, cmap="RdBu_r", vmin=0.94, vmax=1.06, aspect="auto"); ax[i, j].set_xticks(range(len(sh)), sh); ax[i, j].set_yticks(range(len(S)), S)
            for a_ in range(len(S)):
                for b_ in range(len(sh)):
                    cat = g[(g.structure == S[a_]) & (g.share == sh[b_])].category.iloc[0]; mark = {"meaningful edge": "**", "detectable, below band": "*", "detectably worse": "x"}.get(cat, "")
                    ax[i, j].text(b_, a_, f"{M[a_, b_]:.3f}{mark}", ha="center", va="center", fontsize=8)
            ax[i, j].set_title(f"{q} / best classical, {mem} memory"); ax[i, j].set_xlabel("signal share")
    fig.colorbar(im, ax=ax, shrink=0.6); fig.suptitle("Regime map: quantum reservoir vs validation-selected classical model (** meaningful edge, * below band, x worse)")
    fig.savefig(OUTD / "rmap_map.png", dpi=130, bbox_inches="tight")
except Exception as e:
    pr("figure skipped:", e)
