"""long_prereg_deflate.json: tests of the deflation rule (PROGRESS/32_theory_deflated_quantum_advantage_2026-11-18.md) on saved forecasts only.
Each loader returns a standard dataset:
  dict(name, insample, quant=[...], qctrl, losses=[...], lctrl, units={u: dict(win=[...], S={L: {model: array (k, nwin)}}, ref={L: {w: model}})})
  where S[L][m][j, i] is the sum of loss L of model m in window i, j indexing 3-seed subsets (last row = full seed mean; one row if no seeds).
Losses on the log variance scale: MSE, MAE and QLIKE(y, f) = exp(y - f) - (y - f) - 1.
"""
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import integrate, stats

HERE = Path(__file__).parent


def emax(n):
    if n <= 1: return 0.0
    return float(integrate.quad(lambda x: n * x * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1), -12, 12)[0])


def loss(L, y, f):
    e = y - f
    with np.errstate(over="ignore", invalid="ignore"):
        return {"MSE": e ** 2, "MAE": np.abs(e), "QLIKE": np.exp(e) - e - 1}[L]


# ---------------- loaders ----------------
def copper():
    import budget_lib as bl
    units = {}
    for g in bl.GRIDS:
        W, cnt, wy, ids = bl.weekly_tables(g); rs = bl.ref_selection(W, wy); yrs = bl.TEST
        S = {L: {m: np.stack([W[L][m][:, wy == Y].sum(1) for Y in yrs], 1) for m in bl.MODELS} for L in bl.LOSSES}
        units[g] = dict(win=yrs, S=S, ref={L: {Y: rs[L][Y] for Y in yrs} for L in bl.LOSSES})
    return dict(name="copper ladder (DOM/ISING)", insample=True, quant=["DOM", "ISING"], qctrl="DOM", losses=["MSE", "MAE", "QLIKE"], lctrl="MSE", units=units)


def panel():
    import budget3_lib as b3
    units = {}
    for a in b3.ASSETS:
        T = b3.tables(b3.load_asset(a)); qs = T["quarters"]; good, _ = b3.refs(T)
        S = {L: {m: np.stack([T["S"][L][m][:, T["qm"] == q].sum(1) for q in qs], 1) for m in T["S"][L]} for L in b3.LOSSES}
        units[a] = dict(win=qs, S=S, ref={L: {q: good for q in qs} for L in b3.LOSSES})
    return dict(name="commodity panel (QLSTMT, QRC, Ising, kernels, QKAN)", insample=True, quant=[q for q in b3.QUANT], qctrl="QLSTMT",
                losses=["QLIKE", "MSE", "MAE"], lctrl="QLIKE", units=units)


def _blocks(asset, years, paths, models, folder):
    """daily blocks as kern_report.py / ad_report.py: y from long_core.prepare('range_m', year, length=1); forecasts calibrated on validation"""
    import long_core as lc, v2_core as vc
    data = None if asset == "HG" else pd.read_csv(lc.ROOT.parent / folder / f"{asset}.csv", parse_dates=["date"]).set_index("date")
    S = {L: {m: [] for m in models} for L in ("QLIKE", "MSE", "MAE")}; val = {m: [] for m in models}; wins = []
    for Y in years:
        D = lc.prepare("range_m", int(Y), length=1, data=data); yy, va, te = D["y"], D["va"], D["te"]; F = {}
        for m in models:
            z = np.load(paths(asset, Y, m)); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0)
            c = vc.calib_scale(np.exp(yy[va][ok]), v[ok])
            with np.errstate(divide="ignore", invalid="ignore"): F[m] = (np.log(c * t), np.log(c * v))
        if not all(np.isfinite(F[m][0]).all() for m in models): continue                      # block dropped if any forecast unusable (recorded by count)
        wins.append(int(Y))
        for m in models:
            for L in S: S[L][m].append(np.nansum(loss(L, yy[te], F[m][0])))
            vv = loss("QLIKE", yy[va], F[m][1]); val[m].append(float(np.nanmean(vv[np.isfinite(vv)])))
    S = {L: {m: np.array(v)[None, :] for m, v in d.items()} for L, d in S.items()}
    return wins, S, {m: np.array(v) for m, v in val.items()}


def kernels():
    MODELS = ["Q_L1", "Q_L2", "Q_L3", "Product", "RBF", "CosPoly", "FS_L2", "FS_L3", "RFF_RBF", "HAR_only"]; CL = MODELS[3:]
    jobs = [l.split() for l in (HERE / "kern_jobs.txt").read_text().split("\n") if l.strip()]; units = {}
    for a in sorted(set(j[0] for j in jobs)):
        yrs = sorted(int(j[1]) for j in jobs if j[0] == a); folder = "daily_kernel_audit" if a in ("SPX", "AAPL", "GARAN", "KCHOL") else "daily_multi"
        wins, S, val = _blocks(a, yrs, lambda a_, Y, m: HERE / "kern_preds" / a_ / str(Y) / f"{m}.npz", MODELS, folder)
        ref = {w: CL[int(np.argmin([val[m][i] for m in CL]))] for i, w in enumerate(wins)}
        units[a] = dict(win=wins, S=S, ref={L: ref for L in S}, nyears=len(yrs))
    return dict(name="daily quantum kernels (Q_L1-Q_L3)", insample=False, quant=["Q_L1", "Q_L2", "Q_L3"], qctrl="Q_L2", losses=["QLIKE", "MSE", "MAE"], lctrl="QLIKE", units=units)


def adaptive():
    import long_core as lc
    E = pd.read_csv(lc.ROOT.parent / "daily_multi" / "eligibility.csv"); P = HERE / "long_preds_qrc"
    FIXED = ["HARq", "EWMA", "QRC_angle", "ESN", "TrigFeat"]; ADAPT = ["HAR_adapt", "QRC_adapt", "ESN_adapt", "TrigFeat_adapt"]; MODELS = FIXED + ADAPT
    CL = [m for m in MODELS if not m.startswith("QRC")]

    def path(a, u, m):
        if m in ADAPT: return P / "adaptive" / a / str(u) / f"{m}.npz"
        if a != "HG": return P / "multiasset" / a / str(u) / f"{m}.npz"
        if m in ("QRC_angle", "ESN"): return P / "roll" / "range_m" / str(u) / f"{m}.npz"
        if m == "TrigFeat": return P / "mvtrig" / "range_m" / str(u) / f"{m}.npz"
        return lc.OUT_ROLL / "range_m" / str(u) / f"{m}.npz"
    units = {}
    for a in ["HG"] + sorted(E[E.asset_included].asset.unique()):
        yrs = list(range(2008, 2025)) if a == "HG" else sorted(E[(E.asset == a) & E.eligible].year)
        wins, S, val = _blocks(a, yrs, path, MODELS, "daily_multi")
        ref = {w: CL[int(np.argmin([val[m][i] for m in CL]))] for i, w in enumerate(wins)}
        units[a] = dict(win=wins, S=S, ref={L: ref for L in S}, nyears=len(yrs))
    return dict(name="daily quantum reservoirs, 8 assets (QRC fixed/adaptive)", insample=False, quant=["QRC_angle", "QRC_adapt"], qctrl="QRC_angle",
                losses=["QLIKE", "MSE", "MAE"], lctrl="QLIKE", units=units)


def hourly_roll():
    import glob
    yrs = sorted({int(Path(f).name[4:8]) for f in glob.glob(str(HERE / "hourly_databento_preds_v5" / "roll*_s*.npz"))})
    MOD = {"QLSTMT": "QLSTMT_h16", "LSTM": "LSTM_h5", "GRU": "GRU_h6"}; subs = [tuple(c) for c in itertools.combinations(range(5), 3)]
    S = {L: {m: np.zeros((len(subs) + 1, len(yrs))) for m in MOD} for L in ("QLIKE", "MSE", "MAE")}; ref = {}
    for i, Y in enumerate(yrs):
        val = {}
        for m, tag in MOD.items():
            zs = [np.load(HERE / "hourly_databento_preds_v5" / f"roll{Y}_{tag}_s{s}.npz") for s in range(5)]; y = zs[0]["y"]
            assert all(np.array_equal(z["y"], y) for z in zs); P = np.stack([z["pred"] for z in zs]); val[m] = float(np.mean([float(z["best_val"]) for z in zs]))
            for L in S:
                for j, c in enumerate(subs): S[L][m][j, i] = np.sum(loss(L, y, P[list(c)].mean(0)))
                S[L][m][-1, i] = np.sum(loss(L, y, P.mean(0)))
        ref[Y] = min(("LSTM", "GRU"), key=lambda m: val[m])
    return dict(name="hourly rolling QLSTMT (Databento copper, 5 seeds)", insample=False, quant=["QLSTMT"], qctrl="QLSTMT", losses=["MSE", "MAE", "QLIKE"], lctrl="MSE",
                units={"HG_hourly": dict(win=yrs, S=S, ref={L: ref for L in S})})


LOADERS = {"copper": copper, "panel": panel, "kernels": kernels, "adaptive": adaptive, "hourly_roll": hourly_roll}


# ---------------- option results r_o for single-choice protocols ----------------
def _pooled(U, q, L, j=-1, wins=None):
    idx = range(len(U["win"])) if wins is None else wins
    Q = sum(U["S"][L][q][j, i] for i in idx); C = sum(U["S"][L][U["ref"][L][U["win"][i]]][-1, i] for i in idx); return float(np.log(Q / C))


def cells(D):
    """returns list of dict(dataset, unit, choice, N, r_options, r_control) for single-choice protocols (others at control)"""
    out = []; q0, L0 = D["qctrl"], D["lctrl"]
    for u, U in D["units"].items():
        nw = len(U["win"])
        r = [_pooled(U, q0, L0, wins=[i]) for i in range(nw)]; out.append(dict(choice="WINDOW", unit=u, r=r, rc=_pooled(U, q0, L0)))
        k = U["S"][L0][q0].shape[0]
        if k > 1: out.append(dict(choice="SEED", unit=u, r=[_pooled(U, q0, L0, j=j) for j in range(k - 1)], rc=_pooled(U, q0, L0)))
        out.append(dict(choice="LOSS", unit=u, r=[_pooled(U, q0, L) for L in D["losses"]], rc=_pooled(U, q0, L0)))
        qs = [q for q in D["quant"] if q in U["S"][L0]]
        if len(qs) > 1: out.append(dict(choice="QMODEL", unit=u, r=[_pooled(U, q, L0) for q in qs], rc=_pooled(U, q0, L0)))
    if len(D["units"]) > 1:
        ru = [_pooled(U, q0, L0) for U in D["units"].values()]; out.append(dict(choice="UNIT", unit="all", r=ru, rc=float(np.mean(ru))))
    for c in out:
        r = np.array(c["r"]); c.update(dataset=D["name"], N=len(r), s=float(r.std(ddof=1)) if len(r) > 1 else 0.0, rms=float(np.sqrt(np.mean(r ** 2))),
                                       G_obs=float(c["rc"] - r.min()), r=[float(x) for x in r])
        c["G_pred"] = c["s"] * emax(c["N"]); c["deflated"] = float(r.min() + c["G_pred"]); c["err"] = c["deflated"] - c["rc"]
        c["tol"] = max(0.01, 0.3 * c["G_pred"]); c["H1_pass"] = bool(abs(c["err"]) <= c["tol"]) if c["N"] >= 4 else None
    return out


def signflip_null(c, B=2000, seed=0):
    """median of G under random sign flips of the option results (control = mean of flipped options); compared with rms x E[M_N]"""
    rng = np.random.default_rng(seed); r = np.array(c["r"]); s = rng.choice([-1, 1], size=(B, len(r))) * r
    return float(np.median(s.mean(1) - s.min(1))), c["rms"] * emax(len(r))


def selftest():
    assert abs(emax(2) - 1 / np.sqrt(np.pi)) < 1e-6
    rng = np.random.default_rng(0); n = 12; g = []
    for _ in range(4000):
        r = rng.normal(0, 0.05, n); g.append((r.mean() - r.min()) / r.std(ddof=1))
    assert abs(np.mean(g) / emax(n) - 1) < 0.08, np.mean(g) / emax(n)                          # s x E[M_N] tracks mean - min for normal draws
    return round(float(np.mean(g) / emax(n)), 3)


if __name__ == "__main__":
    print("selftest ratio (should be near 1):", selftest())
