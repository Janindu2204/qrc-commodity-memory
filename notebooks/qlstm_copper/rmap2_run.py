"""One synthetic dataset of long_prereg_rmap2.json (corrected generator rmap2_lib.py; otherwise rmap_run.py unchanged except the target and the output folder). Usage: python rmap2_run.py MEMORY STRUCT SHARE REP   (SHARE 0 uses STRUCT NULL).
Rows t = 50 .. n-2 predict x[t+1]. Splits by time: train t < 4000, validation 4024 <= t < 5000, test t >= 5024 (24-step purge).
Benchmark HAR = OLS on [1, x_t, mean of last 10, mean of last 50] (the true linear part has exactly this form).
Students, all residual add-ons to HAR, exactly the copper pipeline (dom_run.py, dom2_run.py, dom3_run.py, dom4_run.py):
  DOM (G3, seeds 0-9), ISING (seeds 0-9): ridge readout on 28 features, input u = min-max scaled with training rows.
  ESN28u_t, SAS28_t: tuned on 12 configurations with tuning seeds 100, 101 (validation MSE), then seeds 0-9.
  LSTM1: one channel, last 10 values, 12 configurations, tuning seed 100, then seeds 0-9 (dom3_lib.fit_seq unchanged).
Reference rows (not students): ORACLE = true conditional mean (linear part + a g); DOM_or / ISING_or = the teacher's own circuit and
scaling with a ridge readout (QDOM / QISING datasets only; checks that a quantum edge can be detected at all).
Target: y[t+1] = x[t+1] + a g_t (rmap2_lib). Writes rmap2_preds/<MEMORY>_<STRUCT>_<SHARE>_<REP>.npz. Scores nothing.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np, torch
import dom_lib as dl, dom2_lib as d2, dom3_lib as d3, rmap_lib as R, rmap2_lib as R2, hourly_databento_qrc as qrc

MEM, ST, SH, REP = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); HERE = Path(__file__).parent; NRES = 10; SEQ = 10
gen = "QDOM" if ST == "NULL" else ST; assert (ST == "NULL") == (SH == 0.0)
x, yv, g, a, c = R2.dataset(MEM, gen, SH, REP); n = len(x); t0 = time.time()
m10 = np.array([x[max(0, t - 9):t + 1].mean() for t in range(n)]); m50 = np.array([x[max(0, t - 49):t + 1].mean() for t in range(n)])
pos = np.arange(50, n - 1); y = yv[pos + 1]; H = np.column_stack([np.ones(len(pos)), x[pos], m10[pos], m50[pos]])
tr = np.where(pos < 4000)[0]; va = np.where((pos >= 4024) & (pos < 5000))[0]; te = np.where(pos >= 5024)[0]
ytr, yva, yte = y[tr], y[va], y[te]; beta = np.linalg.lstsq(H[tr], ytr, rcond=None)[0]; base = {k: H[i] @ beta for k, i in (("tr", tr), ("va", va), ("te", te))}
b1, b2, b3 = R.B[MEM]; oracle = b1 * x[pos] + b2 * m10[pos] + b3 * m50[pos] + a * g[pos]
xtr = x[pos[tr]]; lo, hi = float(xtr.min()), float(xtr.max()); u = np.clip((x - lo) / (hi - lo), 0, 1); zs = (x - xtr.mean()) / xtr.std()
Rr = {"tr": pos[tr], "va": pos[va], "te": pos[te]}


def ridge(f, rows="te"): return base[rows] + qrc.ridge_fit(f[Rr["tr"]], ytr - base["tr"], f[Rr["va"]], yva - base["va"], f[Rr[rows]])


P, V, chosen = {}, {}, {}
def add(m, fs):  # fs: list of feature matrices (one per seed); prediction = mean over seeds
    P[m] = np.mean([ridge(f) for f in fs], 0); V[m] = float(np.mean((yva - np.mean([ridge(f, "va") for f in fs], 0)) ** 2))


add("DOM", [dl.features(u, s, "G3") for s in range(NRES)]); add("ISING", [d2.ising_features(u, s) for s in range(NRES)])
RES = {"ESN28u_t": (d3.ESN_GRID, lambda s, c: dl.esn1(zs, s, leak=c["leak"], rho=c["rho"])), "SAS28_t": (d3.SAS_GRID, lambda s, c: d3.sas(u, s, leak=c["leak"], gain=c["gain"]))}
for m, (grid, fn) in RES.items():
    best, cfg = np.inf, None
    for cc in grid:
        e = float(np.mean((yva - np.mean([ridge(fn(s, cc), "va") for s in (100, 101)], 0)) ** 2))
        if e < best: best, cfg = e, cc
    chosen[m] = cfg; add(m, [fn(s, cfg) for s in range(NRES)])
W = np.stack([zs[pos - (SEQ - 1) + k] for k in range(SEQ)], 1)[:, :, None]; rsd = float((ytr - base["tr"]).std()); T32 = lambda q: torch.tensor(np.asarray(q, dtype=np.float32))
Xs = {k: T32(W[i]) for k, i in (("tr", tr), ("va", va), ("te", te))}; rs = {"tr": T32((ytr - base["tr"]) / rsd), "va": T32((yva - base["va"]) / rsd)}
vals = [d3.fit_seq("LSTM", cc, 100, Xs["tr"], rs["tr"], Xs["va"], rs["va"])[0] for cc in d3.NN_GRID]; cfg = d3.NN_GRID[int(np.argmin(vals))]; chosen["LSTM1"] = cfg
fits = [d3.fit_seq("LSTM", cfg, s, Xs["tr"], rs["tr"], Xs["va"], rs["va"], Xs["te"]) for s in range(NRES)]
P["LSTM1"] = base["te"] + rsd * np.mean([f[2] for f in fits], 0); V["LSTM1"] = float(np.mean((yva - (base["va"] + rsd * np.mean([f[1] for f in fits], 0))) ** 2))
P["HAR"] = base["te"]; V["HAR"] = float(np.mean((yva - base["va"]) ** 2)); P["ORACLE"] = oracle[te]; V["ORACLE"] = float(np.mean((yva - oracle[va]) ** 2))
if ST in ("QDOM", "QISING"):
    ut = np.clip((x - c["scale"][0]) / (c["scale"][1] - c["scale"][0]), 0, 1); Ua, Ub = dl.circuit(R.teacher_seed(REP), "G3") if ST == "QDOM" else d2.ising_unitaries(R.teacher_seed(REP))
    m = ST[1:] + "_or" if ST == "QISING" else "DOM_or"; add(m, [d2.features_u(ut, Ua, Ub)])
out = HERE / os.environ.get("RMAP2_OUT", "rmap2_preds"); out.mkdir(exist_ok=True)
np.savez(out / f"{MEM}_{ST}_{SH}_{REP}.npz", y=yte, a=a, consts=json.dumps(c), config=json.dumps(chosen), val=json.dumps(V), **{f"p_{k}": v for k, v in P.items()})
print(MEM, ST, SH, REP, "done", f"{time.time() - t0:.0f}s", {k: round(float(np.mean((yte - v) ** 2) / np.mean((yte - P['HAR']) ** 2)), 4) for k, v in P.items()} if "--show" in sys.argv else "", flush=True)
