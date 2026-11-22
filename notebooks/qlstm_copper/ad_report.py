"""Score long_prereg_adaptive.json. Usage: python ad_report.py
Pooling as ma_report.py (equal-weighted across assets, daily series of cross-asset mean relative loss difference, DM-HAC lag 10), but verdict words are computed on the RATIO interval:
ratio < 1 means the FIRST model is better; 'better' = whole interval below 1, 'worse' = whole interval above 1, 'equivalent' = interval inside 0.97-1.03."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

LAG = 10; P = Path(__file__).parent / "long_preds_qrc"; ELIG = pd.read_csv(lc.ROOT.parent / "daily_multi" / "eligibility.csv")
FIXED = ["HARq", "EWMA", "QRC_angle", "ESN", "TrigFeat"]; ADAPT = ["HAR_adapt", "QRC_adapt", "ESN_adapt", "TrigFeat_adapt"]; MODELS = FIXED + ADAPT
STRESS, POST = (2008, 2009, 2020), (2010, 2021)


def verdict_ratio(lo, hi): return "equivalent" if (lo > 0.97 and hi < 1.03) else ("worse" if lo > 1 else ("better" if hi < 1 else "inconclusive"))


assert verdict_ratio(0.90, 0.95) == "better" and verdict_ratio(1.05, 1.10) == "worse" and verdict_ratio(0.98, 1.02) == "equivalent" and verdict_ratio(0.9, 1.1) == "inconclusive"


def path(a, u, m):
    if m in ADAPT: return P / "adaptive" / a / str(u) / f"{m}.npz"
    if a != "HG": return P / "multiasset" / a / str(u) / f"{m}.npz"
    if m in ("QRC_angle", "ESN"): return P / "roll" / "range_m" / str(u) / f"{m}.npz"
    if m == "TrigFeat": return P / "mvtrig" / "range_m" / str(u) / f"{m}.npz"
    return lc.OUT_ROLL / "range_m" / str(u) / f"{m}.npz"


recs = []; ASSETS = ["HG"] + sorted(ELIG[ELIG.asset_included].asset.unique())
for a in ASSETS:
    data = None if a == "HG" else pd.read_csv(lc.ROOT.parent / "daily_multi" / f"{a}.csv", parse_dates=["date"]).set_index("date")
    years = list(range(2008, 2025)) if a == "HG" else sorted(ELIG[(ELIG.asset == a) & ELIG.eligible].year); vol = {}
    for u in years:
        D = lc.prepare("range_m", u, length=1, data=data); yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te]); vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values))); L = {}
        for m in MODELS:
            z = np.load(path(a, u, m)); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); L[m] = lc.qlike_vec(vc.calib_scale(np.exp(yy[va][ok]), v[ok]) * t, yl)
        recs.append(dict(asset=a, year=u, dates=D["d"].index[te], L=L))
    order = sorted(years, key=lambda y: (vol[y], y)); k = len(years) // 3
    for r in recs:
        if r["asset"] == a: r["regime"] = "calm" if r["year"] in order[:k] else ("turbulent" if r["year"] in order[len(order) - k:] else "intermediate")
print("assets", ASSETS, "blocks", len(recs), "rows", sum(len(r["dates"]) for r in recs))


def hac(d):
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); return float(r.params[0]), float(r.bse[0])


def pooled(a, b, sel, assets=None):
    parts = []
    for A in (assets or ASSETS):
        rs = [r for r in recs if r["asset"] == A and sel(r)]
        if not rs: continue
        La = np.concatenate([r["L"][a] for r in rs]); Lb = np.concatenate([r["L"][b] for r in rs]); dt = np.concatenate([r["dates"] for r in rs])
        parts.append(pd.Series((Lb - La) / Lb.mean(), index=dt).groupby(level=0).mean().rename(A))
    if not parts: return None
    rel = pd.concat(parts, axis=1).mean(axis=1).sort_index().values; md, se = hac(rel)
    lo, hi = 1 - (md + 1.96 * se), 1 - (md - 1.96 * se)
    return dict(ratio=1 - md, ci_lo=lo, ci_hi=hi, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict_ratio(lo, hi), n_assets=len(parts), n_days=len(rel))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


STRATA = {"all": lambda r: True, "global stress 2008/09/2020": lambda r: r["year"] in STRESS, "post-stress 2010/2021": lambda r: r["year"] in POST,
          "calm": lambda r: r["regime"] == "calm", "intermediate": lambda r: r["regime"] == "intermediate", "turbulent": lambda r: r["regime"] == "turbulent"}
PRIM = [("HAR_adapt", "HARq"), ("HAR_adapt", "EWMA"), ("QRC_adapt", "HAR_adapt"), ("QRC_adapt", "ESN_adapt"), ("QRC_adapt", "TrigFeat_adapt")]
SEC = [("QRC_adapt", "QRC_angle"), ("ESN_adapt", "ESN"), ("TrigFeat_adapt", "TrigFeat"), ("QRC_adapt", "EWMA"), ("EWMA", "HARq")]
out = []
for sname, sel in STRATA.items():
    for subset, assets in (("all assets", None), ("ex-copper", [a for a in ASSETS if a != "HG"])):
        for fam, pairs in (("primary (Holm)", PRIM), ("secondary (own Holm)", SEC)):
            rows = [dict(model=a, ref=b, **pooled(a, b, sel, assets)) for a, b in pairs]
            for r in holm(rows): r.update(stratum=sname, subset=subset, family=fam)
            out += rows
res = pd.DataFrame(out); res.to_csv("ad_report_pooled.csv", index=False); pd.set_option("display.width", 230)
print(res[["stratum", "subset", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict", "n_assets", "n_days"]].round(3).to_string(index=False))
print("\n=== per asset (single-asset DM, all blocks and stress/post-stress years): ratio<1 favours the first model ===")
pa = []
for A in ASSETS:
    for sname in ("all", "global stress 2008/09/2020", "post-stress 2010/2021"):
        for a, b in (("HAR_adapt", "HARq"), ("HAR_adapt", "EWMA"), ("QRC_adapt", "HAR_adapt")):
            r = pooled(a, b, STRATA[sname], [A])
            if r: pa.append(dict(asset=A, stratum=sname, model=a, ref=b, ratio=r["ratio"], verdict=r["verdict"]))
pa = pd.DataFrame(pa); pa.to_csv("ad_report_per_asset.csv", index=False)
print(pa.pivot_table(index="asset", columns=["stratum", "model", "ref"], values="ratio").round(3).to_string())
