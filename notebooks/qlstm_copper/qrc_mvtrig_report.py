"""Score long_prereg_mvtrig.json (17 rolling one-year blocks 2008-2024). Usage: python qrc_mvtrig_report.py
Writes qrc_mvtrig_report_blocks.csv and qrc_mvtrig_report.csv. Never touches 2025-26."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc
import long_core as lc

VERSION = "range_m"; LAG = 10; MARGIN = 0.03; UNITS = list(lc.ROLL_YEARS)
NEW = Path(__file__).with_name("long_preds_qrc") / "mvtrig"; OLD = Path(__file__).with_name("long_preds_qrc") / "roll"
NEWM = ["HARx", "QRCmv_angle", "QRCmv_noent", "ESNmv", "RandFeatmv", "TrigFeat"]; OLDM = ["QRC_angle", "QRC_noent", "ESN", "RandFeat"]


def load(u, m):
    if m in NEWM: f = NEW / VERSION / str(u) / f"{m}.npz"
    elif m in OLDM: f = OLD / VERSION / str(u) / f"{m}.npz"
    else: f = lc.OUT_ROLL / VERSION / str(u) / f"{m}.npz"
    z = np.load(f); return z["lv_val"], z["lv_te"]


MODELS = ["HARq", "EWMA"] + OLDM + NEWM
L, row_unit, vol, per = {m: [] for m in MODELS}, [], {}, []
for u in UNITS:
    D = lc.prepare(VERSION, u, length=1); yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te])
    vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values)))
    row_unit += [u] * len(te); row = {"unit": u, "n_test": len(te)}
    for m in MODELS:
        v, t = load(u, m); ok = np.isfinite(v) & (v > 0); c = vc.calib_scale(np.exp(yy[va][ok]), v[ok])
        L[m].append(lc.qlike_vec(c * t, yl)); row[m] = L[m][-1].mean()
    per.append(row)
row_unit = np.array(row_unit); L = {m: np.concatenate(L[m]) for m in MODELS}
blk = pd.DataFrame(per)
for m in MODELS: blk[f"{m}/HAR"] = blk[m] / blk["HARq"]
order = sorted(UNITS, key=lambda y: (vol[y], y)); regime = {y: ("calm" if i < 6 else "intermediate" if i < 11 else "turbulent") for i, y in enumerate(order)}
blk["vol"] = blk.unit.map(vol); blk["regime"] = blk.unit.map(regime)
blk.round(4).to_csv("qrc_mvtrig_report_blocks.csv", index=False)
pd.set_option("display.width", 250); print(blk[["unit", "regime"] + [f"{m}/HAR" for m in MODELS if m != "HARq"]].round(3).to_string(index=False))
print("\nblocks with ratio<1 vs HARq, exact binomial p vs 0.5 (descriptive):")
for m in MODELS:
    if m != "HARq":
        k = int((blk[f"{m}/HAR"] < 1).sum()); print(f"  {m}: {k}/{len(UNITS)}  p={stats.binomtest(k, len(UNITS), 0.5).pvalue:.3f}")


def hac(diff):
    r = sm.OLS(diff, np.ones(len(diff))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); return float(r.params[0]), float(r.bse[0])


def verdict_of(lo, hi): return "equivalent (CI inside +/-3%)" if (lo > -MARGIN and hi < MARGIN) else ("worse" if lo > 0 else ("better" if hi < 0 else "inconclusive"))


def compare(a, b, mask):      # ratio < 1: a better than b
    mean_d, se = hac(L[b][mask] - L[a][mask]); ref = L[b][mask].mean(); rel = -mean_d / ref; half = 1.96 * se / ref
    p = 2 * (1 - stats.norm.cdf(abs(mean_d / se))); lo, hi = rel - half, rel + half
    return dict(model=a, ref=b, ratio=1 + rel, ci_lo=1 + lo, ci_hi=1 + hi, p=p, verdict=verdict_of(lo, hi), n=int(mask.sum()))


def did(a1, a0, b1, b0, mask):   # (a1 - a0) - (b1 - b0): does the multivariate step change a more than b? ratio < 1: a gains more
    d = (L[a1][mask] - L[a0][mask]) - (L[b1][mask] - L[b0][mask]); mean_d, se = hac(d); ref = L[b1][mask].mean()
    rel = mean_d / ref; half = 1.96 * se / ref; p = 2 * (1 - stats.norm.cdf(abs(mean_d / se))); lo, hi = rel - half, rel + half
    return dict(model=f"({a1}-{a0})", ref=f"({b1}-{b0})", ratio=1 + rel, ci_lo=1 + lo, ci_hi=1 + hi, p=p, verdict=verdict_of(lo, hi), n=int(mask.sum()))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


strata = {"all": np.ones(len(row_unit), bool)}
for g in ("calm", "intermediate", "turbulent"): strata[g] = np.isin(row_unit, [u for u in UNITS if regime[u] == g])
FAMS = [
    ("A primary (Holm)", [("QRCmv_angle", "ESNmv"), ("QRCmv_angle", "QRCmv_noent"), ("QRCmv_angle", "RandFeatmv"), ("QRCmv_angle", "HARx")]),
    ("A secondary: multivariate vs univariate, same rows (own Holm)", [("QRCmv_angle", "QRC_angle"), ("ESNmv", "ESN"), ("RandFeatmv", "RandFeat"), ("HARx", "HARq")]),
    ("B primary (Holm)", [("TrigFeat", "QRC_angle"), ("TrigFeat", "RandFeat"), ("TrigFeat", "ESN"), ("TrigFeat", "HARq")]),
]
out = []
for tag, mask in strata.items():
    for fam, pairs in FAMS:
        rows = [compare(a, b, mask) for a, b in pairs]
        for r in holm(rows): r["set"] = f"{tag}, {fam}"
        out += rows
    r = did("QRCmv_angle", "QRC_angle", "ESNmv", "ESN", mask); r["p_holm"] = r["p"]; r["set"] = f"{tag}, A secondary: difference-in-differences (single test)"; out.append(r)
res = pd.DataFrame(out); res.to_csv("qrc_mvtrig_report.csv", index=False)
print(res[["set", "model", "ref", "ratio", "ci_lo", "ci_hi", "p", "p_holm", "verdict", "n"]].round(4).to_string(index=False))
