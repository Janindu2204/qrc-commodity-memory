"""Score long_prereg_dom4.json. Usage: python dom4_report.py rolling | heldout
Reads DOM, ESN28u (long_preds_dom), ISING, ESN28u_t (long_preds_dom2), ESNq_t, SAS28_t, LSTM, GRU (long_preds_dom3), ESNb_t, SASb_t, LSTM1, GRU1 (long_preds_dom4) and the earlier quantum reservoir QRC from the stored
fixed-configuration intraday runs / the stored daily run (QRC_angle), as dom_report.py. Held-out files are read ONLY by the 'heldout' call. Writes dom4_report_<what>.csv.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

WHAT = sys.argv[1]; MARGIN = 0.03; LAGS = {"H": 24, "M": 96, "F": 288, "D": 10}; HERE = Path(__file__).parent
SRC = {"DOM": "long_preds_dom", "ESN28u": "long_preds_dom", "ISING": "long_preds_dom2", "ESN28u_t": "long_preds_dom2", "ESNq_t": "long_preds_dom3", "SAS28_t": "long_preds_dom3", "LSTM": "long_preds_dom3", "GRU": "long_preds_dom3",
       "ESNb_t": "long_preds_dom4", "SASb_t": "long_preds_dom4", "LSTM1": "long_preds_dom4", "GRU1": "long_preds_dom4"}
ALL = list(SRC) + ["QRC", "HAR"]


def verdict(lo, hi):
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    return "better" if hi < 1 else ("worse" if lo > 1 else "inconclusive")


def category(r):
    if r["p_holm"] < 0.05 and r["ci_hi"] < 1 - MARGIN: return "meaningful edge"
    if r["p_holm"] < 0.05 and r["ci_hi"] < 1: return "detectable, below band"
    if r["p_holm"] < 0.05 and r["ci_lo"] > 1: return "detectably worse"
    return "no detectable difference"


assert verdict(0.85, 0.95) == "better" and verdict(1.05, 1.2) == "worse" and verdict(0.99, 1.01) == "equivalent" and verdict(0.9, 1.1) == "inconclusive"
assert category(dict(p_holm=0.001, ci_hi=0.96, ci_lo=0.9)) == "meaningful edge" and category(dict(p_holm=0.001, ci_hi=0.999, ci_lo=0.99)) == "detectable, below band"
assert category(dict(p_holm=0.2, ci_hi=0.999, ci_lo=0.99)) == "no detectable difference" and category(dict(p_holm=0.01, ci_hi=1.02, ci_lo=1.001)) == "detectably worse"


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
    E = {m: [] for m in ALL}; yy, vol, cfg = [], {}, {}
    for y in years:
        Z = {d: np.load(HERE / d / g / f"{y}.npz") for d in set(SRC.values())}; a = Z["long_preds_dom"]
        b = np.load((HERE / "long_preds_hf" / g / f"{y}.npz") if (g in "HM" and y != 2026) else (HERE / "long_preds_hf2" / "fixed" / g / f"{y}.npz"))
        for c in list(Z.values()) + [b]: assert np.array_equal(a["t"], c["t"]) and np.allclose(a["y"], c["y"]) and np.allclose(a["base"], c["base"]), ("rows differ", g, y)
        for m, d in SRC.items(): E[m].append((a["y"] - Z[d]["p_" + m]) ** 2)
        E["QRC"].append((a["y"] - b["p_QRC"]) ** 2); E["HAR"].append((a["y"] - a["base"]) ** 2); yy += [y] * len(a["y"]); vol[y] = float(a["vol"]); cfg[y] = str(Z["long_preds_dom4"]["config"])
    return {m: np.concatenate(v) for m, v in E.items()}, np.array(yy), vol, cfg


def daily(years, held):
    data = pd.read_csv(lc.ROOT.parent / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date")
    L = {m: [] for m in ALL}; yy, vol = [], {}
    for u in years:
        D = lc.prepare("range_m", u, length=1, data=data); y_, va, te = D["y"], D["va"], D["te"]; yl = np.exp(y_[te])
        vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values))); yy += [u] * len(te)
        old = (HERE / "long_preds_dom" / "D_old" / str(u)) if held else (HERE / "long_preds_qrc" / "dbdaily" / "range" / str(u))
        for m in ALL:
            z = np.load((HERE / SRC[m] / "D" / str(u) / f"{m}.npz") if m in SRC else (old / ("HARq.npz" if m == "HAR" else "QRC_angle.npz")))
            v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); assert np.isfinite(t).all() and (t > 0).all(), (m, u)
            L[m].append(lc.qlike_vec(vc.calib_scale(np.exp(y_[va][ok]), v[ok]) * t, yl))
    return {m: np.concatenate(v) for m, v in L.items()}, np.array(yy), vol, {}


QM = ("DOM", "ISING"); SI = ("ESNb_t", "SASb_t"); NN = ("LSTM1", "GRU1")
FAM = {"M2 mechanism, larger stand-ins": [(q, s) for q in QM for s in SI],
       "T1 quantum vs single-input networks": [(q, n) for q in QM for n in NN],
       "L1 single-input networks": [("LSTM1", "HAR"), ("GRU1", "HAR"), ("LSTM1", "LSTM"), ("GRU1", "GRU")],
       "C descriptive": [("ESNb_t", "ESNq_t"), ("SASb_t", "SAS28_t"), ("ESNb_t", "ESN28u_t"), ("SASb_t", "ESN28u_t"), ("LSTM1", "ESN28u_t"), ("GRU1", "ESN28u_t"), ("LSTM1", "ESNb_t"), ("LSTM1", "SASb_t")]}
out, decisions, blocks, tiers = [], [], [], []
for g in ("H", "M", "F", "D"):
    years = ([2026] if g != "D" else [2025, 2026]) if WHAT == "heldout" else (list(range(2013, 2026)) if g != "D" else list(range(2015, 2025)))
    try:
        E, yy, vol, cfg = intraday(g, years) if g != "D" else daily(years, WHAT == "heldout")
    except FileNotFoundError as e:
        print("MISSING", g, e); continue
    strata = {"all": np.ones(len(yy), bool)}
    if WHAT == "rolling":
        order = sorted(years, key=lambda y: (vol[y], y)); cut = (4, 9) if len(years) == 13 else (3, 7)
        reg = {y: ("calm" if i < cut[0] else "intermediate" if i < cut[1] else "turbulent") for i, y in enumerate(order)}
        for s in ("calm", "intermediate", "turbulent"): strata[s] = np.isin(yy, [y for y in years if reg[y] == s])
        for y in years:
            mk = yy == y; blocks.append(dict(grid=g, year=y, regime=reg[y], n=int(mk.sum()), **{m + "/HAR": E[m][mk].mean() / E["HAR"][mk].mean() for m in ALL[:-1]}, config=cfg.get(y, "")))
    tiers.append(dict(grid=g, **{m: E[m].mean() / E["HAR"].mean() for m in ALL[:-1]}))
    for tag, mk in strata.items():
        for fam, pairs in FAM.items():
            rows = holm([dict(cmp_arr(E[a][mk], E[b][mk], LAGS[g]), model=a, ref=b, n=int(mk.sum())) for a, b in pairs])
            for r in rows: r.update(grid=g, stratum=tag, family=fam, category=category(r) if not fam.startswith("C") else "")
            out += rows
    pp = pd.DataFrame([r for r in out if r["grid"] == g]); row = lambda a, b: pp[(pp.stratum == "all") & (pp.model == a) & (pp.ref == b)].iloc[0]
    d = dict(grid=g)
    for q in QM:
        cats = [row(q, s).category for s in SI]
        d[q + "_mechanism"] = ("explained by a classical stand-in" if any(c in ("no detectable difference", "detectably worse") for c in cats) else "not explained") + f" ({SI[0]}: {cats[0]}; {SI[1]}: {cats[1]})"
    d["quantum_vs_best_single_input_network"] = "; ".join(f"{q}: {row(q, min(NN, key=lambda n: E[n].mean())).category}" for q in QM)
    decisions.append(d)
df = pd.DataFrame(out); df.to_csv(HERE / f"dom4_report_{WHAT}.csv", index=False); pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 200)
if len(df):
    print("ratio < 1 favours the first model. H, M, F: mean squared error of ln RV; D: QLIKE.")
    print(df[["grid", "stratum", "family", "model", "ref", "ratio", "ci_lo", "ci_hi", "p_holm", "verdict", "category", "n"]].round(4).to_string(index=False))
    print("\n=== mean loss relative to the benchmark, all blocks pooled (descriptive) ===\n", pd.DataFrame(tiers).round(4).to_string(index=False))
    if blocks: print("\n=== per-year ratio to the benchmark and chosen configurations (descriptive) ===\n", pd.DataFrame(blocks).round(4).to_string(index=False))
    print("\n=== registered decision ===\n", pd.DataFrame(decisions).to_string(index=False))
