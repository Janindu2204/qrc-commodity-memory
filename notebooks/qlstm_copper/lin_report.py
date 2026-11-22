"""Report for long_prereg_lin.json. Usage: python lin_report.py   Writes lin_report.txt (stdout) and lin_report.csv.
Rolling = test years 2013-2025 pooled; held-out = 2026. Ratio of mean MSE with a Newey-West 95% interval and two-sided p (lags H 24, M 96, F 288).
Families: B1 LIN200 and LIN50 vs HAR (Holm over 2); B2 every compared model vs LIN200 (Holm over 10). Categories as dom3_report (+/-3% band)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).parent; LAGS = {"H": 24, "M": 96, "F": 288}; MARGIN = 0.03
SRC = {"DOM": "long_preds_dom", "ISING": "long_preds_dom2", "ESN28u_t": "long_preds_dom2", "SAS28_t": "long_preds_dom3", "LSTM": "long_preds_dom3", "GRU": "long_preds_dom3",
       "ESNb_t": "long_preds_dom4", "SASb_t": "long_preds_dom4", "LSTM1": "long_preds_dom4", "GRU1": "long_preds_dom4"}


def nw(ea, eb, lag):
    d = eb - ea; n = len(d); md = d.mean(); x = d - md; s = x @ x / n
    for k in range(1, min(lag, n - 1) + 1): s += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    se = np.sqrt(max(s, 0) / n); ref = eb.mean(); r = 1 - md / ref; h = 1.96 * se / ref; p = float(2 * stats.norm.sf(abs(md / se))) if se > 0 else 1.0
    return r, r - h, r + h, p


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def category(p_holm, lo, hi):
    if p_holm < 0.05 and hi < 1 - MARGIN: return "meaningful edge"
    if p_holm < 0.05 and hi < 1: return "detectable, below band"
    if p_holm < 0.05 and lo > 1: return "detectably worse"
    return "no detectable difference"


rows = []
for g in ("F", "M", "H"):
    for per, years in (("rolling", range(2013, 2026)), ("heldout", [2026])):
        E = {}
        for Y in years:
            L = np.load(HERE / "long_preds_lin" / g / f"{Y}.npz"); y = L["y"]
            for m in ("LIN200", "LIN50"): E.setdefault(m, []).append((y - L[f"p_{m}"]) ** 2)
            E.setdefault("HAR", []).append((y - L["base"]) ** 2)
            for m, s in SRC.items():
                z = np.load(HERE / s / g / f"{Y}.npz"); assert np.array_equal(z["y"], y) and np.allclose(z["base"], L["base"]), (g, Y, m)
                E.setdefault(m, []).append((y - z[f"p_{m}"]) ** 2)
        E = {k: np.concatenate(v) for k, v in E.items()}
        for fam, pairs in (("B1 linear benchmark vs HAR", [("LIN200", "HAR"), ("LIN50", "HAR")]), ("B2 model vs LIN200", [(m, "LIN200") for m in SRC])):
            res = [(a, b, *nw(E[a], E[b], LAGS[g])) for a, b in pairs]; ph = holm([r[5] for r in res])
            for (a, b, r, lo, hi, p), q in zip(res, ph): rows.append(dict(grid=g, period=per, family=fam, model=a, ref=b, ratio=r, ci_lo=lo, ci_hi=hi, p=p, p_holm=q, category=category(q, lo, hi), n=len(E[a])))
D = pd.DataFrame(rows); D.to_csv(HERE / "lin_report.csv", index=False)
for (g, per), d in D.groupby(["grid", "period"], sort=False):
    print(f"\n== {g} {per} (n = {d.n.iloc[0]}) ==")
    for _, r in d.iterrows(): print(f"  {r.family[:2]} {r.model:9s} vs {r.ref:6s} {r.ratio:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]  p_holm {r.p_holm:.3g}  {r.category}")
print("\n== decision ==")
for (g, per), d in D.groupby(["grid", "period"], sort=False):
    b1 = d[(d.model == "LIN200") & (d.ref == "HAR")].iloc[0]; b2 = d[d.family.str.startswith("B2")]
    beat = b2[b2.category.isin(["meaningful edge", "detectable, below band"])]
    print(f"  {g} {per}: LIN200 vs HAR {b1.category} ({b1.ratio:.4f}); models that beat LIN200 detectably: {', '.join(beat.model) if len(beat) else 'none'}; meaningful edge over LIN200: {', '.join(b2[b2.category == 'meaningful edge'].model) or 'none'}")
