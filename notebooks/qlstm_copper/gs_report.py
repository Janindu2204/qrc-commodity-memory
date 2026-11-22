"""Report for the gold/silver registration (PROGRESS/34_osf_registration_gold_silver_*.md). Usage: python gs_report.py   (reads long_preds_gs/<ASSET>/...).
R1: DOM, ISING vs LIN200 and vs LSTM1, rolling years (LIN200 hourly 2015-2025, otherwise 2013-2025), 6 cells x 4 = 24 comparisons, Holm within each cell
    (family of 4), categories with the +/-3% band; supported if no 'meaningful edge'.
R2: K* from gs_mech (ridge on last K lags, fit 2013-2022, test 2023-2025) > 8 in >= 5 of 6 cells.
R3: combination LIN200+DOM (A) vs LIN200+ESNb_t (B), weight in {0,...,1} step 0.05 chosen on the previous test year; pooled test-loss ratio A/B with a
    Newey-West 90% interval; pass if the interval lies inside +/-0.25% (5-min) or +/-0.5% (15, 60-min); supported if >= 5 of 6 cells pass.
Sensitivity: R1 and R3 on never-touched years only (gold 2018-2019; silver 2013-2019)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
HERE = Path(__file__).parent / "long_preds_gs"; LAGS = {"F": 288, "M": 96, "H": 24}; MARGIN = 0.03; R3M = {"F": 0.0025, "M": 0.005, "H": 0.005}
SENS = {"GC": [2018, 2019], "SI": list(range(2013, 2020))}; W = np.linspace(0, 1, 21)
SRC = {"DOM": "dom", "ISING": "dom2", "ESNb_t": "dom4", "LSTM1": "dom4", "LIN200": "lin"}


def nw(ea, eb, lag, z=1.96):
    d = eb - ea; n = len(d); md = d.mean(); x = d - md; s = x @ x / n
    for k in range(1, min(lag, n - 1) + 1): s += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    se = np.sqrt(max(s, 0) / n); ref = eb.mean(); r = 1 - md / ref; h = z * se / ref; p = float(2 * stats.norm.sf(abs(md / se))) if se > 0 else 1.0
    return r, r - h, r + h, p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0.0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); a[i] = min(1.0, run)
    return a


def cat(ph, lo, hi):
    if ph < 0.05 and hi < 1 - MARGIN: return "meaningful edge"
    if ph < 0.05 and hi < 1: return "detectable, below band"
    if ph < 0.05 and lo > 1: return "detectably worse"
    return "no detectable difference"


def load(a, g, years):
    D = {}
    for Y in years:
        f = {m: np.load(HERE / a / s / g / f"{Y}.npz") for m, s in SRC.items()}; y = f["LIN200"]["y"]
        for m in f: assert np.array_equal(f[m]["y"], y), (a, g, Y, m)
        D[Y] = dict(y=y, **{m: f[m][f"p_{m}"] for m in SRC})
    return D


def r1(D):
    E = {m: np.concatenate([(d["y"] - d[m]) ** 2 for d in D.values()]) for m in SRC}; return E


out = {"R1": [], "R2": [], "R3": [], "sens": []}
for a in ["GC", "SI"]:
    for g in ["F", "M", "H"]:
        yrs = list(range(2015 if g == "H" else 2013, 2026)); D = load(a, g, yrs)
        for tag, sel in (("all", yrs), ("never-touched", [y for y in yrs if y in SENS[a]])):
            if not sel: continue
            Dsel = {y: D[y] for y in sel}; E = r1(Dsel)
            res = [(q, ref, *nw(E[q], E[ref], LAGS[g])) for q in ("DOM", "ISING") for ref in ("LIN200", "LSTM1")]; ph = holm([r[5] for r in res])
            rows = [dict(asset=a, grid=g, years=tag, model=q, ref=ref, ratio=r, lo=lo, hi=hi, p_holm=float(h), category=cat(h, lo, hi)) for (q, ref, r, lo, hi, p), h in zip(res, ph)]
            (out["R1"] if tag == "all" else out["sens"]).extend(rows)
            # R3 (needs a previous year inside the selection)
            ea, eb = [], []
            for Y in sel:
                if Y - 1 not in D: continue
                p0, p1 = D[Y - 1], D[Y]
                w1 = W[np.argmin([np.mean((p0["y"] - ((1 - w) * p0["LIN200"] + w * p0["DOM"])) ** 2) for w in W])]
                w2 = W[np.argmin([np.mean((p0["y"] - ((1 - w) * p0["LIN200"] + w * p0["ESNb_t"])) ** 2) for w in W])]
                ea.append((p1["y"] - ((1 - w1) * p1["LIN200"] + w1 * p1["DOM"])) ** 2); eb.append((p1["y"] - ((1 - w2) * p1["LIN200"] + w2 * p1["ESNb_t"])) ** 2)
            if ea:
                r, lo, hi, _ = nw(np.concatenate(ea), np.concatenate(eb), LAGS[g], z=1.645); ok = (lo > 1 - R3M[g]) and (hi < 1 + R3M[g])
                (out["R3"] if tag == "all" else out["sens"]).append(dict(asset=a, grid=g, years=tag, test="R3", ratio=r, lo=lo, hi=hi, margin=R3M[g], pass_=bool(ok)))
        m = json.load(open(HERE / a / f"mech_{g}.json")); L = [x for x in m["fits"] if not x["quad"]]; best = max(x["r2"] for x in L)
        ks = None if best <= 0 else min(x["K"] for x in L if x["r2"] >= 0.9 * best); out["R2"].append(dict(asset=a, grid=g, K_star=ks, pass_=bool(ks is not None and ks > 8)))
json.dump(out, open(HERE / "gs_report.json", "w"), indent=1)
print("== R1 (no meaningful quantum edge), rolling years ==")
for r in out["R1"]: print(f"  {r['asset']} {r['grid']} {r['model']:5s} vs {r['ref']:6s} {r['ratio']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] p_holm {r['p_holm']:.3g}  {r['category']}")
n_edge = sum(r["category"] == "meaningful edge" for r in out["R1"]); print("  R1:", "SUPPORTED" if n_edge == 0 else f"REFUTED ({n_edge} meaningful edges)")
print("== R2 (K* > 8) ==")
for r in out["R2"]: print(f"  {r['asset']} {r['grid']} K* = {r['K_star']}  {'pass' if r['pass_'] else 'FAIL'}")
print("  R2:", "SUPPORTED" if sum(r["pass_"] for r in out["R2"]) >= 5 else "REFUTED")
print("== R3 (combination with DOM vs with ESNb_t) ==")
for r in out["R3"]: print(f"  {r['asset']} {r['grid']} A/B {r['ratio']:.4f} 90% [{r['lo']:.4f}, {r['hi']:.4f}] margin +/-{100 * r['margin']:.2f}%  {'pass' if r['pass_'] else 'FAIL'}")
print("  R3:", "SUPPORTED" if sum(r["pass_"] for r in out["R3"]) >= 5 else "REFUTED")
print("== Sensitivity (never-touched years; reported, not a decision) ==")
for r in out["sens"]:
    if "category" in r: print(f"  {r['asset']} {r['grid']} {r['model']:5s} vs {r['ref']:6s} {r['ratio']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]  {r['category']}")
    else: print(f"  {r['asset']} {r['grid']} R3 A/B {r['ratio']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]  {'pass' if r['pass_'] else 'FAIL'}")
