"""Score long_prereg_smalln.json (DRAFT until frozen). Usage: python smalln_report.py    Reads long_preds_smalln/<arm>/<n>/<2008..2024>/*.npz. 2025-26 is never read.
Primary: tuned arm, QRC vs {ESN, RandFeat, QRC_noent, Basis2, HAR_only} at each n, Holm over 25 tests. Writes smalln_report.csv."""
import os
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

YEARS = list(lc.ROLL_YEARS); LAG, MARGIN = 10, 0.03; SIZES = (50, 100, 250, 500, 1000); P = Path(os.environ.get("SMALLN_PREDS", Path(__file__).with_name("long_preds_smalln")))
SET = {"tuned": ["QRC", "QRC_noent", "RandFeat", "ESN", "TrigFeat", "Basis2", "HAR_only"], "fixed": ["QRC_L2", "QRC_L1", "QRC_noent_L2", "RandFeat", "ESN", "TrigFeat", "Basis2", "HAR_only"]}
CONTROLS = {"tuned": ["ESN", "RandFeat", "QRC_noent", "Basis2", "HAR_only"], "fixed": ["ESN", "RandFeat", "QRC_noent_L2", "Basis2", "HAR_only"]}


def verdict(ratio, lo, hi):
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    if hi < 1: return "better"
    if lo > 1: return "worse"
    return "inconclusive"


assert verdict(0.9, 0.85, 0.95) == "better" and verdict(1.1, 1.05, 1.2) == "worse" and verdict(1.0, 0.99, 1.01) == "equivalent" and verdict(1.0, 0.9, 1.1) == "inconclusive"
blocks = {y: lc.prepare("range_m", y, length=1) for y in YEARS}


def losses(arm, n):
    miss = [(y, m) for y in YEARS for m in SET[arm] if not (P / arm / str(n) / str(y) / f"{m}.npz").exists()]
    if miss: return None
    Lz = {m: [] for m in SET[arm]}
    for u in YEARS:
        D = blocks[u]; yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te])
        for m in SET[arm]:
            z = np.load(P / arm / str(n) / str(u) / f"{m}.npz"); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); c = vc.calib_scale(np.exp(yy[va][ok]), v[ok]); Lz[m].append(lc.qlike_vec(c * t, yl))
    return {m: np.concatenate(Lz[m]) for m in SET[arm]}


def compare(Lz, a, b):
    d = Lz[b] - Lz[a]; r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); md, se = float(r.params[0]), float(r.bse[0]); ref = Lz[b].mean()
    rel = -md / ref; half = 1.96 * se / ref; ratio, lo, hi = 1 + rel, 1 + rel - half, 1 + rel + half
    return dict(model=a, ref=b, ratio=ratio, ci_lo=lo, ci_hi=hi, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict(ratio, lo, hi))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


out = []
for arm in ("tuned", "fixed"):
    q = "QRC" if arm == "tuned" else "QRC_L2"; rows = []
    for n in SIZES:
        Lz = losses(arm, n)
        if Lz is None: print("MISSING blocks for", arm, n); continue
        for c in CONTROLS[arm]: rows.append(dict(compare(Lz, q, c), arm=arm, n_train=n, family="primary"))
        for c in ("TrigFeat",) + (("QRC_L1",) if arm == "fixed" else ()): rows.append(dict(compare(Lz, q, c), arm=arm, n_train=n, family="secondary"))
        rows.append(dict(compare(Lz, "HAR_only", "HAR_only") if False else dict(model="HAR_only", ref="(mean QLIKE)", ratio=float(Lz["HAR_only"].mean()), ci_lo=np.nan, ci_hi=np.nan, p=np.nan, verdict="-"), arm=arm, n_train=n, family="level"))
    prim = [r for r in rows if r["family"] == "primary"]
    if prim: holm(prim)
    out += rows
res = pd.DataFrame(out); res.to_csv("smalln_report.csv", index=False); pd.set_option("display.width", 250)
print(res[["arm", "n_train", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict"]].round(4).to_string(index=False))
t = res[(res.arm == "tuned") & (res.family == "primary")]
if len(t):
    print("\n=== registered decision (tuned arm) ===")
    ok_n = {}
    for n in SIZES:
        s = t[t.n_train == n]; ok_n[n] = (len(s) == 5 and bool((s.ci_hi < 1).all()) and bool((s.p_holm < 0.05).all()), len(s) == 5 and bool((s.ratio < 1).all()))
    found = [n for n in SIZES if n <= 500 and ok_n[n][0] and n != SIZES[-1] and ok_n[SIZES[SIZES.index(n) + 1]][1]]
    print("n with QRC better than all five controls (upper CI<1, Holm<.05):", [n for n in SIZES if ok_n[n][0]], "| sizes satisfying the full rule:", found)
    print("SMALL-SAMPLE NICHE SUPPORTED" if found else "no small-sample niche found")
