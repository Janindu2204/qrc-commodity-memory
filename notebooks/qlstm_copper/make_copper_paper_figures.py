"""Figures for DOCUMENTATION/ongoing/ieee_access_copper_paper (reads saved report/json files only; fits nothing).
Usage: python make_copper_paper_figures.py  ->  ../../../../DOCUMENTATION/ongoing/ieee_access_copper_paper/figs/sanda1.pdf ... sanda5.pdf"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import budget_lib as bl

HERE = Path(__file__).parent; ROOT = HERE.parents[1]; OUT = ROOT / "paper/figs"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
NAME = {"SEED": "Seeds", "WINDOW": "Test year", "GRID": "Sampling frequency", "LOSS": "Loss function", "QMODEL": "Quantum model", "REF": "Weak reference"}

# Fig 1: apparent advantage under control, each single pitfall, five selection pitfalls, all six; and Shapley shares
obs = {tuple(d[k] for k in bl.FACTORS): d["ratio"] for d in json.load(open(HERE / "budget_out/observed.json"))}; ph = json.load(open(HERE / "budget_out/shapley.json"))
key = lambda **on: tuple(bool(on.get(k, False)) for k in bl.FACTORS); A = lambda r: 100 * (1 - r)
labels = ["Sound protocol"] + [NAME[f] for f in bl.FACTORS] + ["Five selection\npitfalls", "All six"]
vals = [A(obs[key()])] + [A(obs[key(**{f: True})]) for f in bl.FACTORS] + [A(obs[key(SEED=1, WINDOW=1, GRID=1, LOSS=1, QMODEL=1)]), A(obs[(True,) * 6])]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
y = np.arange(len(vals))[::-1]; ax[0].barh(y, vals, color=["0.35"] + ["0.65"] * 6 + ["0.2", "0.2"]); ax[0].axvline(0, color="k", lw=0.6)
ax[0].set_yticks(y, labels); ax[0].set_xlabel("Apparent quantum advantage (%)"); ax[0].set_title("(a) Apparent advantage of DOM by protocol", loc="left")
for yy, v in zip(y, vals): ax[0].text(v + (0.4 if v >= 0 else -0.4), yy, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
ax[0].set_xlim(min(vals) - 3, max(vals) + 4)
tot = sum(ph.values()); order = sorted(ph, key=lambda k: ph[k]); sh = [100 * ph[k] / tot for k in order]
ax[1].barh(range(len(order)), sh, color="0.5"); ax[1].set_yticks(range(len(order)), [NAME[k] for k in order]); ax[1].axvline(0, color="k", lw=0.6)
ax[1].set_xlabel("Share of the swing (%)"); ax[1].set_title("(b) Shapley decomposition", loc="left")
for i, v in enumerate(sh): ax[1].text(v + (1 if v >= 0 else -1), i, f"{v:.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
ax[1].set_xlim(min(sh) - 12, max(sh) + 12); fig.tight_layout(); fig.savefig(OUT / "sanda1.pdf"); plt.close(fig)

# Fig 2: observed gain of each selection pitfall vs the year-block null; seed lottery
d = json.load(open(HERE / "budget_out/null_year.json")); o, n = d["observed"], d["null"]; c0 = np.log(o["control"]); cn = np.log(np.array(n["control"]))
names = ["SEED", "WINDOW", "GRID", "LOSS", "QMODEL", "ALL5"]; lab2 = ["Seeds", "Test\nyear", "Fre-\nquency", "Loss", "Quantum\nmodel", "All\nfive"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
G = [100 * (cn - np.log(np.array(n[k]))) for k in names]; go = [100 * (c0 - np.log(o[k])) for k in names]
ax[0].boxplot(G, positions=range(len(names)), widths=0.5, showfliers=False, whis=(5, 95)); ax[0].scatter(range(len(names)), go, color="k", marker="D", s=14, zorder=3, label="Observed on copper")
ax[0].set_xticks(range(len(names)), lab2); ax[0].set_ylabel("Gain over sound protocol (log points)"); ax[0].set_yscale("symlog", linthresh=1)
ax[0].set_title("(a) Gain vs. chance (year-block null)", loc="left"); ax[0].legend(frameon=False, fontsize=7)
lot = json.load(open(HERE / "budget_out/seed_lottery.json")); ks = list(lot["k"].keys())
ax[1].plot([int(k) for k in ks], [lot["k"][k]["p_below_1"] for k in ks], "o-", color="k", label="P(quantum looks better)")
ax[1].plot([int(k) for k in ks], [lot["k"][k]["p_above_103"] for k in ks], "s--", color="0.5", label="P(quantum looks >3% worse)")
ax[1].set_xscale("log"); ax[1].set_xlabel("Seeds per model"); ax[1].set_ylabel("Probability"); ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
ax[1].set_title("(b) Seed lottery, QLSTM vs. LSTM", loc="left"); fig.tight_layout(); fig.savefig(OUT / "sanda2.pdf"); plt.close(fig)

# Fig 3: memory functions and copper's residual signal by lag
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
for fam, lab, st in (("DOM", "DOM (quantum)", "k-"), ("ISING", "ISING (quantum)", "k--"), ("ESN", "Tuned ESN (classical)", "0.5"), ("SAS", "State-affine twin", "0.7")):
    rows = json.load(open(HERE / f"mech_out/cap_{fam}.json")); mf = np.median(np.array([r["MF"] for r in rows]), 0)
    ax[0].plot(np.arange(len(mf))[:61], mf[:61], st if fam in ("DOM", "ISING") else "-", color=None if fam in ("DOM", "ISING") else st, label=lab)
ax[0].set_xlabel("Lag (steps)"); ax[0].set_ylabel("Memory function"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) Memory of past inputs", loc="left")
for g, mk in (("F", "o"), ("M", "s"), ("H", "^")):
    c = json.load(open(HERE / f"mech_out/copper_{g}.json")); L = [x for x in c["fits"] if not x["quad"]]
    ax[1].plot([x["K"] for x in L], [100 * (1 - x["ratio"]) for x in L], marker=mk, color="k", lw=0.8, label={"F": "5-minute", "M": "15-minute", "H": "Hourly"}[g])
ax[1].axvspan(0.8, 8, color="0.85", label="Quantum memory (90%)"); ax[1].set_xscale("log"); ax[1].set_xlabel("Number of past values used, K")
ax[1].set_ylabel("Gain over HAR (%)"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("(b) Signal left in copper after HAR", loc="left")
fig.tight_layout(); fig.savefig(OUT / "sanda3.pdf"); plt.close(fig)

# Fig 4: regime map (corrected run)
R = pd.read_csv(HERE / "rmap2_report.csv", keep_default_na=False); Q = R[R.family.str.startswith("Q1")]
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.8), sharey=True)
for i, mem in enumerate(["long", "short"]):
    for j, q in enumerate(["DOM", "ISING"]):
        g = Q[(Q.memory == mem) & (Q.model == q) & (Q.structure != "NULL")]; S = ["QDOM", "QISING", "CSAS", "CESN", "POLY2", "POLY3"]; sh = sorted(g.share.unique())
        M = np.array([[g[(g.structure == s) & (g.share == h)].ratio.iloc[0] for h in sh] for s in S])
        im = ax[i, j].imshow(M, cmap="RdBu_r", vmin=0.9, vmax=1.1, aspect="auto"); ax[i, j].set_xticks(range(len(sh)), sh); ax[i, j].set_yticks(range(len(S)), S)
        for a in range(len(S)):
            for b in range(len(sh)):
                cat = g[(g.structure == S[a]) & (g.share == sh[b])].category.iloc[0]; mark = {"meaningful edge": "**", "detectable, below band": "*", "detectably worse": "x"}.get(cat, "")
                ax[i, j].text(b, a, f"{M[a, b]:.3f}{mark}", ha="center", va="center", fontsize=6.5, color="w" if abs(M[a, b] - 1) > 0.06 else "k")
        ax[i, j].set_title(f"{q}, {mem} memory", loc="left"); ax[i, j].set_xlabel("Signal share")
fig.subplots_adjust(hspace=0.45, wspace=0.08); fig.colorbar(im, ax=ax, shrink=0.6, label="MSE ratio, quantum / best classical"); fig.savefig(OUT / "sanda4.pdf", bbox_inches="tight"); plt.close(fig)

# Fig 5: prevalence of pitfalls in published papers
P = pd.read_csv(ROOT / "notes/31_prevalence_pilot_coding_2026-11-18.csv"); P = P[P.full_text_read.astype(str).str.lower() == "yes"]
PN = {"P9": "No test or no correction", "P2": "One seed or no repeats", "P10": "Single train/test split", "P8": "No naive or strong benchmark", "P11": "Level metric, no naive comparison",
      "P1": "Test-period selection", "P12": "Scaling on full data", "P3": "Untuned classical baseline", "P5": "Unequal inputs"}
rows = []
for k, lab in PN.items():
    v = P[k].astype(str).str.strip(); pres = int((v == "present").sum()); cod = int(v.isin(["present", "absent"]).sum()); nr = int((v == "not_reported").sum()); rows.append((lab, pres, cod, nr))
rows.sort(key=lambda r: r[1] / max(r[2], 1))
fig, ax = plt.subplots(figsize=(3.5, 2.8)); yy = np.arange(len(rows))
ax.barh(yy, [100 * r[1] / max(r[2], 1) for r in rows], color="0.45"); ax.set_yticks(yy, [f"{r[0]} ({r[1]}/{r[2]})" for r in rows], fontsize=7)
ax.set_xlabel("Share of codable papers (%)"); ax.set_xlim(0, 100); ax.set_title(f"Pitfalls in {len(P[P.n_present.notna()])} papers", loc="left", fontsize=8)
fig.savefig(OUT / "sanda5.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("sanda*.pdf")))

# Fig 6: registered held-out test on gold and silver (OSF yj6w4): R1 ratios with 95% intervals, R3 ratios with 90% intervals and margins
G = json.load(open(HERE / "long_preds_gs/gs_report.json")); GR = {"F": "5-min", "M": "15-min", "H": "60-min"}; MET = {"GC": "Gold", "SI": "Silver"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
cells = [(a, g) for a in ("GC", "SI") for g in ("F", "M", "H")]; off = {("DOM", "LIN200"): -0.27, ("ISING", "LIN200"): -0.09, ("DOM", "LSTM1"): 0.09, ("ISING", "LSTM1"): 0.27}
sty = {("DOM", "LIN200"): ("o", "k"), ("ISING", "LIN200"): ("s", "k"), ("DOM", "LSTM1"): ("o", "0.55"), ("ISING", "LSTM1"): ("s", "0.55")}
for r in G["R1"]:
    i = cells.index((r["asset"], r["grid"])); k = (r["model"], r["ref"]); m, c = sty[k]
    ax[0].errorbar(i + off[k], r["ratio"], yerr=[[r["ratio"] - r["lo"]], [r["hi"] - r["ratio"]]], fmt=m, color=c, ms=3.5, lw=0.8, capsize=1.5,
                   label=f"{k[0]} / {k[1]}" if i == 0 else None)
ax[0].axhline(1, color="k", lw=0.6); ax[0].axhspan(0.97, 1.03, color="0.92", zorder=0, label="+/-3% band")
ax[0].set_xticks(range(6), [f"{MET[a]}\n{GR[g]}" for a, g in cells], fontsize=7); ax[0].set_ylabel("Loss ratio (95% interval)")
ax[0].set_title("(a) R1: quantum reservoir vs. strong classical models", loc="left"); ax[0].legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
for r in G["R3"]:
    i = cells.index((r["asset"], r["grid"]))
    ax[1].errorbar(i, r["ratio"], yerr=[[r["ratio"] - r["lo"]], [r["hi"] - r["ratio"]]], fmt="D", color="k", ms=3.5, lw=0.8, capsize=1.5)
    ax[1].plot([i - 0.3, i + 0.3], [1 + r["margin"]] * 2, color="0.5", lw=0.8); ax[1].plot([i - 0.3, i + 0.3], [1 - r["margin"]] * 2, color="0.5", lw=0.8)
ax[1].axhline(1, color="k", lw=0.6); ax[1].set_xticks(range(6), [f"{MET[a][0]}{GR[g][:-4]}" for a, g in cells], fontsize=7)
ax[1].set_ylabel("A/B loss ratio (90% interval)"); ax[1].set_title("(b) R3: combination test", loc="left")
fig.tight_layout(); fig.savefig(OUT / "sanda6.pdf"); plt.close(fig)
print("wrote sanda6.pdf")
