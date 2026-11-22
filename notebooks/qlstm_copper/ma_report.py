"""Score long_prereg_multiasset.json. Usage: python ma_report.py
Pooling: for each asset, relative loss difference rel_it = (L_ref - L_model)/mean(L_ref over that asset's rows in the stratum); rel_t = mean over assets on each date;
estimate = mean_t rel_t with Diebold-Mariano HAC (lag 10) standard error; ratio = 1 - estimate. This handles cross-asset dependence on the same dates and weights assets equally."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

LAG, MARGIN = 10, 0.03; P = Path(__file__).parent / "long_preds_qrc"; ELIG = pd.read_csv(lc.ROOT.parent / "daily_multi" / "eligibility.csv")
MODELS = ["HARq", "EWMA", "QRC_angle", "QRC_noent", "ESN", "RandFeat", "TrigFeat"]; STRESS = (2008, 2009, 2020)


def path(asset, u, m):
    if asset != "HG": return P / "multiasset" / asset / str(u) / f"{m}.npz"
    if m in ("QRC_angle", "QRC_noent", "ESN", "RandFeat"): return P / "roll" / "range_m" / str(u) / f"{m}.npz"
    if m == "TrigFeat": return P / "mvtrig" / "range_m" / str(u) / f"{m}.npz"
    return lc.OUT_ROLL / "range_m" / str(u) / f"{m}.npz"


recs = []; ASSETS = ["HG"] + sorted(ELIG[ELIG.asset_included].asset.unique())
for a in ASSETS:
    data = None if a == "HG" else pd.read_csv(lc.ROOT.parent / "daily_multi" / f"{a}.csv", parse_dates=["date"]).set_index("date")
    years = list(range(2008, 2025)) if a == "HG" else sorted(ELIG[(ELIG.asset == a) & ELIG.eligible].year)
    vol = {}
    for u in years:
        D = lc.prepare("range_m", u, length=1, data=data); yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te]); dts = D["d"].index[te]
        vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values)))
        L = {}
        for m in MODELS:
            z = np.load(path(a, u, m)); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); L[m] = lc.qlike_vec(vc.calib_scale(np.exp(yy[va][ok]), v[ok]) * t, yl)
        recs.append(dict(asset=a, year=u, dates=dts, L=L, vol=vol[u]))
    order = sorted(years, key=lambda y: (vol[y], y)); k = len(years) // 3
    for r in recs:
        if r["asset"] == a: r["regime"] = "calm" if r["year"] in order[:k] else ("turbulent" if r["year"] in order[len(order) - k:] else "intermediate")
print("assets:", ASSETS, "| blocks per asset:", {a: sum(r["asset"] == a for r in recs) for a in ASSETS}, "| rows:", sum(len(r["dates"]) for r in recs))


def hac(d):
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); return float(r.params[0]), float(r.bse[0])


def verdict_of(lo, hi): return "equivalent" if (lo > -MARGIN and hi < MARGIN) else ("worse" if lo > 0 else ("better" if hi < 0 else "inconclusive"))


def pooled(a, b, sel, assets=None):
    parts = []
    for A in (assets or ASSETS):
        rs = [r for r in recs if r["asset"] == A and sel(r)]
        if not rs: continue
        La = np.concatenate([r["L"][a] for r in rs]); Lb = np.concatenate([r["L"][b] for r in rs]); dt = np.concatenate([r["dates"] for r in rs])
        parts.append(pd.Series((Lb - La) / Lb.mean(), index=dt).groupby(level=0).mean().rename(A))
    if not parts: return None
    rel = pd.concat(parts, axis=1).mean(axis=1).sort_index().values; md, se = hac(rel); lo, hi = md - 1.96 * se, md + 1.96 * se
    return dict(ratio=1 - md, ci_lo=1 - hi, ci_hi=1 - lo, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict_of(lo, hi), n_assets=len(parts), n_days=len(rel))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


STRATA = {"all": lambda r: True, "calm": lambda r: r["regime"] == "calm", "intermediate": lambda r: r["regime"] == "intermediate",
          "turbulent": lambda r: r["regime"] == "turbulent", "global stress 2008/09/2020": lambda r: r["year"] in STRESS}
PRIM = [("QRC_angle", "ESN"), ("QRC_angle", "QRC_noent"), ("QRC_angle", "RandFeat"), ("QRC_angle", "TrigFeat"), ("QRC_angle", "HARq")]
SEC = [("QRC_angle", "EWMA"), ("ESN", "HARq"), ("TrigFeat", "HARq"), ("EWMA", "HARq")]
out = []
for sname, sel in STRATA.items():
    for subset, assets in (("all assets", None), ("ex-copper (replication set)", [a for a in ASSETS if a != "HG"])):
        for fam, pairs in (("primary (Holm)", PRIM), ("secondary (own Holm)", SEC)):
            rows = [dict(model=a, ref=b, **pooled(a, b, sel, assets)) for a, b in pairs]
            for r in holm(rows): r.update(stratum=sname, subset=subset, family=fam)
            out += rows
res = pd.DataFrame(out); res.to_csv("ma_report_pooled.csv", index=False); pd.set_option("display.width", 220)
print(res[["stratum", "subset", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict", "n_assets", "n_days"]].round(3).to_string(index=False))

print("\n=== per asset, all eligible blocks (single-asset DM; ratio<1 favours the first model) ===")
pa = []
for A in ASSETS:
    for a, b in (("QRC_angle", "ESN"), ("QRC_angle", "TrigFeat"), ("QRC_angle", "QRC_noent"), ("QRC_angle", "HARq"), ("ESN", "HARq"), ("TrigFeat", "HARq")):
        r = pooled(a, b, lambda r: True, [A]); pa.append(dict(asset=A, model=a, ref=b, ratio=r["ratio"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], verdict=r["verdict"]))
pa = pd.DataFrame(pa); pa.to_csv("ma_report_per_asset.csv", index=False)
print(pa.pivot_table(index="asset", columns=["model", "ref"], values="ratio").round(3).to_string())
print("\nassets where QRC_angle < ESN (ratio<1):", int((pa[(pa.model == "QRC_angle") & (pa.ref == "ESN")].ratio < 1).sum()), "of", len(ASSETS))
print("assets where QRC_angle < HARq:", int((pa[(pa.model == "QRC_angle") & (pa.ref == "HARq")].ratio < 1).sum()), "| ESN < HARq:", int((pa[(pa.model == "ESN") & (pa.ref == "HARq")].ratio < 1).sum()), "| TrigFeat < HARq:", int((pa[(pa.model == "TrigFeat") & (pa.ref == "HARq")].ratio < 1).sum()))
print("\n=== leave-one-asset-out, QRC_angle vs ESN ===")
for sname in ("all", "turbulent"):
    lo = [(A, pooled("QRC_angle", "ESN", STRATA[sname], [x for x in ASSETS if x != A])["ratio"]) for A in ASSETS]
    print(sname, {A: round(r, 3) for A, r in lo})
