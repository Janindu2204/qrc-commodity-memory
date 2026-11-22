"""Score long_prereg_hf2.json. Usage: python hf2_report.py rolling | heldout.
rolling: tuned H, tuned M, fixed F over blocks 2013-2025 (+ tuned-vs-fixed on H and M). heldout: the single 2026 block for the five configurations (read ONLY by the 'heldout' call)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats

WHAT = sys.argv[1]; MARGIN = 0.03; LAGS = {"H": 24, "M": 96, "F": 288}; MODELS = ["QRC", "QRC_noent", "RandFeat", "ESN", "TrigFeat"]
HERE = Path(__file__).parent


def path(mode, g, y):
    return (HERE / "long_preds_hf" / g / f"{y}.npz") if (mode == "fixed" and g in "HM" and y != 2026) else (HERE / "long_preds_hf2" / mode / g / f"{y}.npz")


def verdict(lo, hi):
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    return "better" if hi < 1 else ("worse" if lo > 1 else "inconclusive")


def cmp_arr(ea, eb, lag):
    d = eb - ea; r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag}); md, se = float(r.params[0]), float(r.bse[0]); ref = eb.mean()
    rel = -md / ref; half = 1.96 * se / ref
    return dict(ratio=1 + rel, ci_lo=1 + rel - half, ci_hi=1 + rel + half, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict(1 + rel - half, 1 + rel + half))


assert verdict(0.85, 0.95) == "better" and verdict(1.05, 1.2) == "worse" and verdict(0.99, 1.01) == "equivalent" and verdict(0.9, 1.1) == "inconclusive"


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


def errs(mode, g, years):
    Z = {y: np.load(path(mode, g, y)) for y in years}
    E = {m: np.concatenate([(Z[y]["y"] - Z[y]["p_" + m]) ** 2 for y in years]) for m in MODELS}; E["HAR"] = np.concatenate([(Z[y]["y"] - Z[y]["base"]) ** 2 for y in years])
    return Z, E


out, decisions = [], []
if WHAT == "rolling":
    YEARS = list(range(2013, 2026)); sets = [("tuned", "H"), ("tuned", "M"), ("fixed", "F")]
else:
    YEARS = [2026]; sets = [("fixed", "H"), ("fixed", "M"), ("fixed", "F"), ("tuned", "H"), ("tuned", "M")]
for mode, g in sets:
    if not all(path(mode, g, y).exists() for y in YEARS): print("MISSING", mode, g, [y for y in YEARS if not path(mode, g, y).exists()]); continue
    Z, E = errs(mode, g, YEARS); yy = np.concatenate([[y] * len(Z[y]["y"]) for y in YEARS]); strata = {"all": np.ones(len(yy), bool)}
    if WHAT == "rolling":
        vol = {y: float(Z[y]["vol"]) for y in YEARS}; order = sorted(YEARS, key=lambda y: (vol[y], y)); reg = {y: ("calm" if i < 4 else "intermediate" if i < 9 else "turbulent") for i, y in enumerate(order)}
        for s in ("calm", "intermediate", "turbulent"): strata[s] = np.isin(yy, [y for y in YEARS if reg[y] == s])
    for tag, mk in strata.items():
        for fam, pairs in {"A model vs HAR+hod": [(m, "HAR") for m in MODELS], "B QRC vs controls": [("QRC", c) for c in ("QRC_noent", "RandFeat", "ESN", "TrigFeat")]}.items():
            rows = holm([dict(cmp_arr(E[a][mk], E[b][mk], LAGS[g]), model=a, ref=b, n=int(mk.sum())) for a, b in pairs])
            for r in rows: r.update(mode=mode, grid=g, stratum=tag, family=fam)
            out += rows
    pp = pd.DataFrame([r for r in out if r["grid"] == g and r["mode"] == mode])
    row = lambda st, a, b: pp[(pp.stratum == st) & (pp.model == a) & (pp.ref == b)].iloc[0]
    ok = all(row("all", "QRC", c).ci_hi < 1 and row("all", "QRC", c).p_holm < 0.05 for c in ("RandFeat", "ESN", "TrigFeat", "HAR"))
    turb = all(row("turbulent", "QRC", c).ratio < 1 for c in ("RandFeat", "ESN", "TrigFeat", "HAR")) if WHAT == "rolling" else None
    tr = row("all", "QRC", "TrigFeat")
    decisions.append(dict(mode=mode, grid=g, quantum_edge=bool(ok and (turb if turb is not None else True)), pooled_rule=bool(ok), turbulent_rule=turb, trig=f"QRC vs TrigFeat {tr.ratio:.3f} [{tr.ci_lo:.3f},{tr.ci_hi:.3f}] {tr.verdict}"))
    if WHAT == "rolling":
        cfgs = [json_ for json_ in (str(Z[y]["config"]) for y in YEARS)] if mode == "tuned" else []
        if mode == "tuned":                                   # descriptive: does tuning help each model? tuned vs fixed on the same rows
            _, Ef = errs("fixed", g, YEARS)
            for m in MODELS: out.append(dict(cmp_arr(E[m], Ef[m], LAGS[g]), model=m + " tuned", ref=m + " fixed", n=len(Ef[m]), mode=mode, grid=g, stratum="all", family="C tuned vs fixed (descriptive)", p_holm=np.nan))
df = pd.DataFrame(out); df.to_csv(f"hf2_report_{WHAT}.csv", index=False); pd.set_option("display.width", 250)
if len(df):
    print(df[["mode", "grid", "stratum", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict", "n"]].round(4).to_string(index=False))
    print("\n=== registered decision ===\n", pd.DataFrame(decisions).to_string(index=False))
