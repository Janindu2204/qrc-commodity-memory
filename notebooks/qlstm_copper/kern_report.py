"""Score long_prereg_kernel.json (DRAFT until frozen). Usage: python kern_report.py. Reads kern_preds/<ASSET>/<YEAR>/*.npz for every job in kern_jobs.txt. 2025-26 is never read.
Env KERN_PREDS / KERN_JOBS override paths (used only to test the script on fake data)."""
import os
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import v2_core as vc, long_core as lc

HERE = Path(__file__).parent; P = Path(os.environ.get("KERN_PREDS", HERE / "kern_preds")); JOBS = Path(os.environ.get("KERN_JOBS", HERE / "kern_jobs.txt"))
LAG, MARGIN = 10, 0.03; PARTB = ["HG", "CL", "GC", "HO", "NG", "ZC", "ZS", "ZW"]; PARTA = ["SPX", "AAPL", "GARAN", "KCHOL"]
MODELS = ["Q_L1", "Q_L2", "Q_L3", "Product", "RBF", "CosPoly", "FS_L2", "FS_L3", "RFF_RBF", "HAR_only"]
jobs = [tuple(l.split()) for l in JOBS.read_text().split("\n") if l.strip()]


def verdict(ratio, lo, hi):
    if lo > 1 - MARGIN and hi < 1 + MARGIN: return "equivalent"
    if hi < 1: return "better"
    if lo > 1: return "worse"
    return "inconclusive"


assert verdict(0.9, 0.85, 0.95) == "better" and verdict(1.1, 1.05, 1.2) == "worse" and verdict(1.0, 0.99, 1.01) == "equivalent" and verdict(1.0, 0.9, 1.1) == "inconclusive"
missing = [(a, y, m) for a, y in jobs for m in MODELS if not (P / a / y / f"{m}.npz").exists()]
assert not missing, f"incomplete output: {missing[:5]} ... ({len(missing)})"
Lz = {}
for a, y in jobs:
    if a not in Lz: Lz[a] = {m: [] for m in MODELS}
    data = None if a == "HG" else pd.read_csv(lc.ROOT.parents[0] / ("daily_kernel_audit" if a in PARTA else "daily_multi") / f"{a}.csv", parse_dates=["date"]).set_index("date")
    D = lc.prepare("range_m", int(y), length=1, data=data); yy, va, te = D["y"], D["va"], D["te"]; yl = np.exp(yy[te])
    for m in MODELS:
        z = np.load(P / a / y / f"{m}.npz"); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); c = vc.calib_scale(np.exp(yy[va][ok]), v[ok]); Lz[a][m].append(lc.qlike_vec(c * t, yl))
Lz = {a: {m: np.concatenate(v) for m, v in d.items()} for a, d in Lz.items()}


def compare(a, q, ref):
    d = Lz[a][ref] - Lz[a][q]; r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}); md, se = float(r.params[0]), float(r.bse[0]); base = Lz[a][ref].mean()
    rel = -md / base; half = 1.96 * se / base; ratio, lo, hi = 1 + rel, 1 + rel - half, 1 + rel + half
    return dict(asset=a, model=q, ref=ref, ratio=ratio, ci_lo=lo, ci_hi=hi, p=2 * (1 - stats.norm.cdf(abs(md / se))) if se > 0 else 1.0, verdict=verdict(ratio, lo, hi) if se > 0 else "equivalent")


def holm(rows):
    pv = np.array([r["p"] for r in rows]); o = np.argsort(pv); h = np.empty(len(pv)); run = 0
    for k, i in enumerate(o): run = max(run, (len(pv) - k) * pv[i]); h[i] = min(1, run)
    for r, x in zip(rows, h): r["p_holm"] = x
    return rows


refs = lambda q: ["Product", "RBF", "CosPoly", "RFF_RBF"] + ([f"FS_L{q[-1]}"] if q in ("Q_L2", "Q_L3") else [])
allrows, edge = [], {}
for a in sorted(Lz):
    for q in ("Q_L1", "Q_L2", "Q_L3"):
        rows = holm([compare(a, q, r) for r in refs(q)]); allrows += rows
        edge[(a, q)] = all(r["verdict"] == "better" and r["p_holm"] < 0.05 for r in rows)
R = pd.DataFrame(allrows); R.to_csv(Path(os.environ.get("KERN_OUT", HERE)) / "kern_report.csv", index=False)
print(R.round(4).to_string(index=False))
print("\n=== descriptive: quantum vs HAR_only, ratio (QLIKE) ===")
for a in sorted(Lz): print(a, {q: round(1 + (Lz[a]["HAR_only"].mean() - Lz[a][q].mean()) / Lz[a]["HAR_only"].mean() * -1, 4) for q in ("Q_L1", "Q_L2", "Q_L3")})
print("\n=== registered decision ===")
for part, names in (("B (main: copper + 7 commodities)", PARTB), ("A (paper's four tickers, our protocol)", PARTA)):
    names = [a for a in names if a in Lz]
    for q in ("Q_L1", "Q_L2", "Q_L3"):
        k = sum(edge[(a, q)] for a in names); print(f"Part {part} {q}: quantum-specific edge in {k}/{len(names)} assets", "-> CLAIM" if part.startswith("B") and k >= 5 else "")
    sub = R[R.asset.isin(names) & (R.model.isin(["Q_L2", "Q_L3"]))]
    print("  vs RBF verdicts:", sub[sub.ref == "RBF"].verdict.value_counts().to_dict(), "| vs RFF_RBF:", sub[sub.ref == "RFF_RBF"].verdict.value_counts().to_dict())
