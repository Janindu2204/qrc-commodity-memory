"""Score long_prereg_dbdaily.json. Usage: python db_report.py TARGET   (range | rv). Compares pooled verdicts with the Yahoo run on the SAME test years."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

TARGET = sys.argv[1]; YEARS = list(range(2015, 2025)); LAG = 10; MARGIN = 0.03
P = Path(__file__).parent / "long_preds_qrc"; data = pd.read_csv(lc.ROOT.parent / "daily_databento" / f"HG_db_1d_{TARGET}.csv", parse_dates=["date"]).set_index("date")
MODELS = ["HARq", "EWMA", "QRC_angle", "QRC_noent", "ESN", "RandFeat", "TrigFeat"]


def path(src, u, m):
    if src == "db": return P / "dbdaily" / TARGET / str(u) / f"{m}.npz"
    if m in ("QRC_angle", "QRC_noent", "ESN", "RandFeat"): return P / "roll" / "range_m" / str(u) / f"{m}.npz"
    if m == "TrigFeat": return P / "mvtrig" / "range_m" / str(u) / f"{m}.npz"
    return lc.OUT_ROLL / "range_m" / str(u) / f"{m}.npz"


def collect(src):
    L = {m: [] for m in MODELS}; ru, vol = [], {}
    for u in YEARS:
        D = lc.prepare("range_m", u, length=1, data=(data if src == "db" else None)); yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te])
        vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values))); ru += [u] * len(te)
        for m in MODELS:
            z = np.load(path(src, u, m)); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); L[m].append(lc.qlike_vec(vc.calib_scale(np.exp(yy[va][ok]), v[ok]) * t, yl))
    return {m: np.concatenate(L[m]) for m in MODELS}, np.array(ru), vol


def hac(d):
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); return float(r.params[0]), float(r.bse[0])


def cmp(L, a, b, mask):
    md, se = hac(L[b][mask] - L[a][mask]); ref = L[b][mask].mean(); rel = -md / ref; half = 1.96 * se / ref; lo, hi = rel - half, rel + half
    v = "equivalent" if (lo > -MARGIN and hi < MARGIN) else ("worse" if lo > 0 else ("better" if hi < 0 else "inconclusive"))
    return dict(model=a, ref=b, ratio=1 + rel, ci_lo=1 + lo, ci_hi=1 + hi, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=v)


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


PAIRS = [("QRC_angle", "ESN"), ("QRC_angle", "QRC_noent"), ("QRC_angle", "RandFeat"), ("QRC_angle", "TrigFeat"), ("QRC_angle", "HARq")]
SEC = [("QRC_angle", "EWMA"), ("ESN", "HARq"), ("TrigFeat", "HARq"), ("EWMA", "HARq")]
res = []
for src in ("db", "yahoo"):
    L, ru, vol = collect(src); order = sorted(YEARS, key=lambda y: (vol[y], y)); reg = {y: ("calm" if i < 3 else "intermediate" if i < 7 else "turbulent") for i, y in enumerate(order)}
    if src == "db":
        print("Databento block volatility / regime:", {y: (round(vol[y], 3), reg[y]) for y in YEARS}); REG = reg
    strata = {"all": np.ones(len(ru), bool)}
    for g in ("calm", "intermediate", "turbulent"): strata[g] = np.isin(ru, [u for u in YEARS if REG[u] == g])      # same regime labels for both sources (Databento-defined)
    for tag, mask in strata.items():
        for fam, pairs in (("primary (Holm)", PAIRS), ("secondary (own Holm)", SEC)):
            rows = holm([cmp(L, a, b, mask) for a, b in pairs])
            for r in rows: r.update(source=src, stratum=tag, family=fam, n=int(mask.sum()))
            res += rows
df = pd.DataFrame(res); df.to_csv(f"db_report_{TARGET}.csv", index=False); pd.set_option("display.width", 220)
wide = df.pivot_table(index=["stratum", "family", "model", "ref"], columns="source", values=["ratio", "p_holm"], sort=False).round(3)
vd = df.pivot_table(index=["stratum", "family", "model", "ref"], columns="source", values="verdict", aggfunc="first", sort=False)
print(f"TARGET {TARGET}: years {YEARS[0]}-{YEARS[-1]}; ratio<1 favours the first model; db = Databento target, yahoo = earlier run on the same years")
print(pd.concat([wide, vd.add_prefix("verdict_")], axis=1).to_string())
