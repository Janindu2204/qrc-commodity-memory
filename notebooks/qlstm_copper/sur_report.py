"""Score long_prereg_surrogate.json (DRAFT until frozen). Usage: python sur_report.py    Reads long_preds_sur/<2008..2024>/*.npz and span_<year>.npz. 2025-26 is never read.
Writes sur_report.txt-style output to stdout, sur_report.csv, sur_report_blocks.csv."""
import os
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

YEARS = list(lc.ROLL_YEARS); LAG, MARGIN = 10, 0.03; P = Path(os.environ.get("SUR_PREDS", Path(__file__).with_name("long_preds_sur")))
MODELS = ["QRC_L1", "QRC_L2", "QRC_L3", "QRC_noent_L2", "Basis2", "Basis3", "Basis4", "RandProj_2", "RandProj_3", "RandProj_4", "TrigFeat", "HAR_only"]


def verdict(ratio, lo, hi):
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    if hi < 1: return "better"
    if lo > 1: return "worse"
    return "inconclusive"


assert verdict(0.9, 0.85, 0.95) == "better" and verdict(1.1, 1.05, 1.2) == "worse" and verdict(1.0, 0.99, 1.01) == "equivalent" and verdict(1.0, 0.9, 1.1) == "inconclusive"
missing = [(y, m) for y in YEARS for m in MODELS if not (P / str(y) / f"{m}.npz").exists()]
assert not missing, f"incomplete output: {missing[:5]} ..."
Lz, row_unit, vol, per = {m: [] for m in MODELS}, [], {}, []
for u in YEARS:
    D = lc.prepare("range_m", u, length=1); yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te])
    vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values))); row_unit += [u] * len(te); row = {"unit": u, "n_test": len(te)}
    for m in MODELS:
        z = np.load(P / str(u) / f"{m}.npz"); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); c = vc.calib_scale(np.exp(yy[va][ok]), v[ok])
        Lz[m].append(lc.qlike_vec(c * t, yl)); row[m] = Lz[m][-1].mean()
    per.append(row)
row_unit = np.array(row_unit); Lz = {m: np.concatenate(Lz[m]) for m in MODELS}
blk = pd.DataFrame(per)
for m in MODELS: blk[f"{m}/HAR_only"] = blk[m] / blk["HAR_only"]
order = sorted(YEARS, key=lambda y: (vol[y], y)); regime = {y: ("calm" if i < 6 else "intermediate" if i < 11 else "turbulent") for i, y in enumerate(order)}
blk["regime"] = blk.unit.map(regime); blk.round(4).to_csv("sur_report_blocks.csv", index=False)


def compare(a, b, mask):
    d = Lz[b][mask] - Lz[a][mask]; r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); md, se = float(r.params[0]), float(r.bse[0]); ref = Lz[b][mask].mean()
    rel = -md / ref; half = 1.96 * se / ref; ratio, lo, hi = 1 + rel, 1 + rel - half, 1 + rel + half
    return dict(model=a, ref=b, ratio=ratio, ci_lo=lo, ci_hi=hi, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict(ratio, lo, hi), n=int(mask.sum()))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


FAM = {"A exact surrogate": [("QRC_L1", "Basis2"), ("QRC_L2", "Basis4"), ("QRC_L3", "Basis4")],
       "B capacity-matched": [("QRC_L1", "RandProj_2"), ("QRC_L2", "RandProj_4"), ("QRC_L3", "RandProj_4")],
       "C existing controls": [(f"QRC_L{k}", c) for k in (1, 2, 3) for c in ("TrigFeat", "HAR_only")] + [("QRC_L2", "QRC_noent_L2")],
       "D extra (not decision)": [("QRC_L2", "Basis2"), ("QRC_L2", "Basis3"), ("QRC_L2", "RandProj_2"), ("QRC_L2", "RandProj_3"), ("QRC_L3", "Basis3"), ("Basis4", "Basis2")]}
strata = {"all": np.ones(len(row_unit), bool)}
for g in ("calm", "intermediate", "turbulent"): strata[g] = np.isin(row_unit, [u for u in YEARS if regime[u] == g])
out = []
for tag, mk in strata.items():
    for fam, pairs in FAM.items():
        rows = holm([compare(a, b, mk) for a, b in pairs])
        for r in rows: r.update(stratum=tag, family=fam)
        out += rows
res = pd.DataFrame(out); res.to_csv("sur_report.csv", index=False); pd.set_option("display.width", 250)
print(res[["stratum", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict"]].round(4).to_string(index=False))
al = res[res.stratum == "all"].set_index(["model", "ref"])
print("\n=== span diagnostic (min / mean R2 of QRC features on Basis2, Basis3, Basis4; mean over blocks) ===")
sp = [np.load(P / str(u) / f"span_{u}.npz") for u in YEARS]
for k in sp[0].files: print(f"{k:22s}", np.round(np.mean([s[k] for s in sp], 0), 6))
print("\n=== registered decision rules (stratum all) ===")
for k, (basis, rp) in {1: ("Basis2", "RandProj_2"), 2: ("Basis4", "RandProj_4"), 3: ("Basis4", "RandProj_4")}.items():
    q = f"QRC_L{k}"; a, b, t, h = al.loc[(q, basis)], al.loc[(q, rp)], al.loc[(q, "TrigFeat")], al.loc[(q, "HAR_only")]
    eq = a.verdict == "equivalent" and not (b.ci_hi < 1)
    edge = all(r.ci_hi < 1 and r.p_holm < 0.05 for r in (a, b, t, h))
    cap = (a.verdict == "better" and b.verdict == "equivalent")
    print(f"{q}: exact-surrogate equivalence: {eq} | quantum-specific edge: {edge} | capacity-artifact flag: {cap}")
