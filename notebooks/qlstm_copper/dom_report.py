"""Score long_prereg_dom.json. Usage: python dom_report.py rolling | heldout
rolling: intraday grids H, M, F over blocks 2013-2025 (squared error of ln RV) and the daily grid D over blocks 2015-2024 (QLIKE). heldout: intraday 2026; daily 2025 + 2026
(held-out files are read ONLY by the 'heldout' call). Earlier models are read from stored files of the fixed-configuration intraday runs (long_preds_hf, long_preds_hf2/fixed)
and of the daily Databento run (long_preds_qrc/dbdaily/range; held-out years from long_preds_dom/D_old). Writes dom_report_<what>.csv.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

WHAT = sys.argv[1]; MARGIN = 0.03; LAGS = {"H": 24, "M": 96, "F": 288, "D": 10}; HERE = Path(__file__).parent
NEW = ["DOM", "DOMG1", "ESN28u"]; OLD_I = ["QRC", "RandFeat", "ESN", "TrigFeat"]; OLD_D = ["QRC_angle", "QRC_noent", "ESN", "RandFeat", "TrigFeat", "EWMA"]


def verdict(lo, hi):
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    return "better" if hi < 1 else ("worse" if lo > 1 else "inconclusive")


assert verdict(0.85, 0.95) == "better" and verdict(1.05, 1.2) == "worse" and verdict(0.99, 1.01) == "equivalent" and verdict(0.9, 1.1) == "inconclusive"


def cmp_arr(ea, eb, lag):
    d = eb - ea; r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag}); md, se = float(r.params[0]), float(r.bse[0]); ref = eb.mean()
    rel = -md / ref; half = 1.96 * se / ref
    return dict(ratio=1 + rel, ci_lo=1 + rel - half, ci_hi=1 + rel + half, p=2 * (1 - stats.norm.cdf(abs(md / se))), verdict=verdict(1 + rel - half, 1 + rel + half))


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


def intraday(g, years):
    E = {m: [] for m in NEW + OLD_I + ["HAR"]}; yy, vol = [], {}
    for y in years:
        a = np.load(HERE / "long_preds_dom" / g / f"{y}.npz")
        b = np.load((HERE / "long_preds_hf" / g / f"{y}.npz") if (g in "HM" and y != 2026) else (HERE / "long_preds_hf2" / "fixed" / g / f"{y}.npz"))
        assert np.array_equal(a["t"], b["t"]) and np.allclose(a["y"], b["y"]) and np.allclose(a["base"], b["base"]), ("rows differ from the stored run", g, y)
        for m in NEW: E[m].append((a["y"] - a["p_" + m]) ** 2)
        for m in OLD_I: E[m].append((b["y"] - b["p_" + m]) ** 2)
        E["HAR"].append((a["y"] - a["base"]) ** 2); yy += [y] * len(a["y"]); vol[y] = float(a["vol"])
    return {m: np.concatenate(v) for m, v in E.items()}, np.array(yy), vol


def daily(years, held):
    data = pd.read_csv(lc.ROOT.parent / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date")
    L = {m: [] for m in NEW + OLD_D + ["HAR"]}; yy, vol = [], {}
    for u in years:
        D = lc.prepare("range_m", u, length=1, data=data); y_, va, te = D["y"], D["va"], D["te"]; yl = np.exp(y_[te])
        vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values))); yy += [u] * len(te)
        old = (HERE / "long_preds_dom" / "D_old" / str(u)) if held else (HERE / "long_preds_qrc" / "dbdaily" / "range" / str(u))
        for m in NEW + OLD_D + ["HAR"]:
            z = np.load((HERE / "long_preds_dom" / "D" / str(u) / f"{m}.npz") if m in NEW else (old / f"{'HARq' if m == 'HAR' else m}.npz"))
            v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); L[m].append(lc.qlike_vec(vc.calib_scale(np.exp(y_[va][ok]), v[ok]) * t, yl))
    return {m: np.concatenate(v) for m, v in L.items()}, np.array(yy), vol


FAM = {"A primary": [("DOM", "ESN28u"), ("DOM", "DOMG1"), ("DOM", "HAR")],
       "C descriptive": [("ESN28u", "HAR"), ("DOMG1", "HAR")]}
out, decisions, blocks = [], [], []
for g in ("H", "M", "F", "D"):
    years = ([2026] if g != "D" else [2025, 2026]) if WHAT == "heldout" else (list(range(2013, 2026)) if g != "D" else list(range(2015, 2025)))
    try:
        E, yy, vol = intraday(g, years) if g != "D" else daily(years, WHAT == "heldout")
    except FileNotFoundError as e:
        print("MISSING", g, e); continue
    oldq = "QRC" if g != "D" else "QRC_angle"
    fams = dict(FAM); fams["B stored models"] = [("DOM", oldq), ("DOM", "RandFeat"), ("DOM", "ESN"), ("DOM", "TrigFeat")]
    if g == "D" and WHAT == "heldout":
        fams["H earlier quantum reservoir, held-out"] = [("QRC_angle", "ESN"), ("QRC_angle", "QRC_noent"), ("QRC_angle", "RandFeat"), ("QRC_angle", "TrigFeat"), ("QRC_angle", "HAR")]
    strata = {"all": np.ones(len(yy), bool)}
    if WHAT == "rolling":
        order = sorted(years, key=lambda y: (vol[y], y)); n = len(years); cut = (4, 9) if n == 13 else (3, 7)
        reg = {y: ("calm" if i < cut[0] else "intermediate" if i < cut[1] else "turbulent") for i, y in enumerate(order)}
        for s in ("calm", "intermediate", "turbulent"): strata[s] = np.isin(yy, [y for y in years if reg[y] == s])
        for y in years:
            mk = yy == y; blocks.append(dict(grid=g, year=y, regime=reg[y], n=int(mk.sum()), **{m + "/HAR": E[m][mk].mean() / E["HAR"][mk].mean() for m in NEW}))
    for tag, mk in strata.items():
        for fam, pairs in fams.items():
            rows = holm([dict(cmp_arr(E[a][mk], E[b][mk], LAGS[g]), model=a, ref=b, n=int(mk.sum())) for a, b in pairs])
            for r in rows: r.update(grid=g, stratum=tag, family=fam)
            out += rows
    pp = pd.DataFrame([r for r in out if r["grid"] == g])
    row = lambda st, a, b: pp[(pp.stratum == st) & (pp.model == a) & (pp.ref == b)].iloc[0]
    ok = all(row("all", "DOM", c).ci_hi < 1 and row("all", "DOM", c).p_holm < 0.05 for c in ("ESN28u", "HAR"))
    turb = all(row("turbulent", "DOM", c).ratio < 1 for c in ("ESN28u", "HAR")) if WHAT == "rolling" else None
    cx = row("all", "DOM", "DOMG1")
    decisions.append(dict(grid=g, quantum_edge=bool(ok and (turb if turb is not None else True)), pooled_rule=bool(ok), turbulent_rule=turb,
                          complexity_claim=bool(cx.ci_hi < 1 and cx.p_holm < 0.05), dom_vs_g1=f"{cx.ratio:.4f} [{cx.ci_lo:.4f},{cx.ci_hi:.4f}] {cx.verdict}"))
df = pd.DataFrame(out); df.to_csv(HERE / f"dom_report_{WHAT}.csv", index=False); pd.set_option("display.width", 250)
if len(df):
    print("ratio < 1 favours the first model. H, M, F: mean squared error of ln RV; D: QLIKE.")
    print(df[["grid", "stratum", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict", "n"]].round(4).to_string(index=False))
    if blocks: print("\n=== per-year ratio to the benchmark (descriptive) ===\n", pd.DataFrame(blocks).round(4).to_string(index=False))
    print("\n=== registered decision ===\n", pd.DataFrame(decisions).to_string(index=False))
