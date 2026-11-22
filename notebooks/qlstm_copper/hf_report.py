"""Score long_prereg_hf.json. Usage: python hf_report.py      Reads long_preds_hf/{H,M}/<year>.npz, writes hf_report_pooled.csv, hf_report_blocks.csv. 2026 is never read."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats

YEARS = list(range(2013, 2026)); MARGIN = 0.03; LAGS = {"H": 24, "M": 96}
MODELS = ["QRC", "QRC_noent", "RandFeat", "ESN", "TrigFeat"]
P = Path(__file__).parent / "long_preds_hf"


def verdict(ratio, lo, hi):                       # on the ratio interval; ratio < 1 means the first model is better
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    if hi < 1: return "better"
    if lo > 1: return "worse"
    return "inconclusive"


assert verdict(0.9, 0.85, 0.95) == "better" and verdict(1.1, 1.05, 1.2) == "worse" and verdict(1.0, 0.99, 1.01) == "equivalent" and verdict(1.0, 0.9, 1.1) == "inconclusive"


def cmp_arr(ea, eb, lag):
    d = eb - ea; r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag}); md, se = float(r.params[0]), float(r.bse[0]); ref = eb.mean()
    rel = -md / ref; half = 1.96 * se / ref; ratio, lo, hi = 1 + rel, 1 + rel - half, 1 + rel + half
    return dict(ratio=ratio, ci_lo=lo, ci_hi=hi, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict(ratio, lo, hi))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


pooled, blocks, decision = [], [], []
for G in ("H", "M"):
    if not all((P / G / f"{y}.npz").exists() for y in YEARS): print("MISSING blocks for", G, [y for y in YEARS if not (P / G / f"{y}.npz").exists()]); continue
    Z = {y: np.load(P / G / f"{y}.npz") for y in YEARS}
    vol = {y: float(Z[y]["vol"]) for y in YEARS}; order = sorted(YEARS, key=lambda y: (vol[y], y)); n = len(YEARS)
    regime = {y: ("calm" if i < 4 else "intermediate" if i < 9 else "turbulent") for i, y in enumerate(order)}
    E = {m: np.concatenate([(Z[y]["y"] - Z[y]["p_" + m]) ** 2 for y in YEARS]) for m in MODELS}; E["HAR"] = np.concatenate([(Z[y]["y"] - Z[y]["base"]) ** 2 for y in YEARS])
    yy = np.concatenate([[y] * len(Z[y]["y"]) for y in YEARS]); strata = {"all": np.ones(len(yy), bool)}
    for g in ("calm", "intermediate", "turbulent"): strata[g] = np.isin(yy, [y for y in YEARS if regime[y] == g])
    for tag, mk in strata.items():
        fams = {"A model vs HAR+hod": [(m, "HAR") for m in MODELS], "B QRC vs controls": [("QRC", c) for c in ("QRC_noent", "RandFeat", "ESN", "TrigFeat")]}
        for fam, pairs in fams.items():
            rows = holm([dict(cmp_arr(E[a][mk], E[b][mk], LAGS[G]), model=a, ref=b, n=int(mk.sum())) for a, b in pairs])
            for r in rows: r.update(grid=G, stratum=tag, family=fam)
            pooled += rows
    for y in YEARS:
        mk = yy == y; blocks.append(dict(grid=G, year=y, regime=regime[y], n=int(mk.sum()), **{m + "/HAR": E[m][mk].mean() / E["HAR"][mk].mean() for m in MODELS}))
    # registered decision rule
    pp = pd.DataFrame([r for r in pooled if r["grid"] == G])
    def row(st, a, b): return pp[(pp.stratum == st) & (pp.model == a) & (pp.ref == b)].iloc[0]
    ok_all = all(row("all", "QRC", c).ci_hi < 1 and row("all", "QRC", c).p_holm < 0.05 for c in ("RandFeat", "ESN", "TrigFeat")) and row("all", "QRC", "HAR").ci_hi < 1 and row("all", "QRC", "HAR").p_holm < 0.05
    ok_turb = all(row("turbulent", "QRC", c).ratio < 1 for c in ("RandFeat", "ESN", "TrigFeat", "HAR"))
    trig = row("all", "QRC", "TrigFeat")
    decision.append(dict(grid=G, quantum_edge=bool(ok_all and ok_turb), qrc_beats_all_controls_and_HAR_pooled=bool(ok_all), turbulent_same_direction=bool(ok_turb),
                         trig_mechanism=f"QRC vs TrigFeat {trig.ratio:.3f} [{trig.ci_lo:.3f},{trig.ci_hi:.3f}] {trig.verdict}"))
df = pd.DataFrame(pooled); df.to_csv("hf_report_pooled.csv", index=False); pd.DataFrame(blocks).to_csv("hf_report_blocks.csv", index=False)
pd.set_option("display.width", 250)
if len(df):
    print(df[["grid", "stratum", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict", "n"]].round(4).to_string(index=False))
    print("\n=== per-year ratio to HAR+hod ===\n", pd.DataFrame(blocks).round(3).to_string(index=False))
    print("\n=== registered decision ===\n", pd.DataFrame(decision).to_string(index=False))
