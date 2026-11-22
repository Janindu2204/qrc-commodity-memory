"""long_prereg_deflate2.json: replication of H2 and a second test of the H1 rule on three datasets not used in long_prereg_deflate.json.
deflate_lib.py is unchanged; its loss, emax, cells() and signflip_null() are reused. New: three loaders and the tail-adjusted expected maximum.
  smalln      : daily copper, training size n in {50,100,250,500,1000} (new choice TRAINSIZE), years 2008-2024; each arm is one dataset;
                tuned arm: quantum QRC (control) and QRC_noent; fixed arm: QRC_L2 (control), QRC_L1, QRC_noent_L2; classical ESN, RandFeat, TrigFeat, Basis2, HAR_only; reference = validation-best classical per block.
                Control training size = 1000. Forecasts calibrated on validation as in smalln_report.py.
  roll_daily  : daily copper, target range_m (gap_m has no QLSTMT forecasts), years 2008-2024, QLSTMT 10 seeds (control) vs validation-best of LSTM (10 seeds), EWMA, GARCH11_t, HARq.
  mc_daily    : daily rolling origins 2020/2022/2024 (windows), 11 commodity futures (units); quantum IsingMem_anch (control) and QRC_angle_anch;
                no validation forecasts are saved, so the reference is fixed in advance: HAR_anch. Levels converted to logs.
"""
import itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import integrate, stats
import deflate_lib as dl

HERE = Path(__file__).parent


def emax_t(n, nu):
    """expected maximum of n iid Student-t(nu) variables scaled to unit variance; nu = inf gives dl.emax"""
    if not np.isfinite(nu): return dl.emax(n)
    sc = np.sqrt((nu - 2) / nu); f = lambda x: n * x * stats.t.pdf(x / sc, nu) / sc * stats.t.cdf(x / sc, nu) ** (n - 1)
    return float(integrate.quad(f, -60, 60, limit=200)[0])


def nu_from_kurtosis(r):
    """nu = 4 + 6 / k from the unbiased sample excess kurtosis k (normal if k <= 0), at least 3"""
    k = float(stats.kurtosis(r, fisher=True, bias=False)) if len(r) >= 4 else 0.0
    return np.inf if k <= 0 else max(3.0, 4 + 6 / k)


def _cal(yy, va, te, z):
    import v2_core as vc
    v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); c = vc.calib_scale(np.exp(yy[va][ok]), v[ok])
    with np.errstate(divide="ignore", invalid="ignore"): return np.log(c * t), np.log(c * v)


def smalln(arm):
    """one arm as one dataset (the arms saved different quantum models); unit = the arm"""
    import long_core as lc
    Q = {"tuned": ["QRC", "QRC_noent"], "fixed": ["QRC_L2", "QRC_L1", "QRC_noent_L2"]}[arm]; CL = ["ESN", "RandFeat", "TrigFeat", "Basis2", "HAR_only"]; M = Q + CL
    SIZES = [50, 100, 250, 500, 1000]; YEARS = list(range(2008, 2025)); blocks = {y: lc.prepare("range_m", y, length=1) for y in YEARS}
    S = {L: {m: np.zeros((1, len(YEARS))) for m in M} for L in ("QLIKE", "MSE", "MAE")}; SZ = {L: {sz: {m: np.zeros(len(YEARS)) for m in M} for sz in SIZES} for L in S}; ref = {}
    for i, y in enumerate(YEARS):
        D = blocks[y]; yy, va, te = D["y"], D["va"], D["te"]; val = {}
        for sz in SIZES:
            for m in M:
                f, fv = _cal(yy, va, te, np.load(HERE / "long_preds_smalln" / arm / str(sz) / str(y) / f"{m}.npz"))
                for L in S: SZ[L][sz][m][i] = np.sum(dl.loss(L, yy[te], f))
                if sz == 1000: vv = dl.loss("QLIKE", yy[va], fv); val[m] = float(np.nanmean(vv[np.isfinite(vv)]))
        ref[y] = min(CL, key=lambda m: val[m])
    for L in S:
        for m in M: S[L][m][0] = SZ[L][1000][m]
    return dict(name=f"daily copper training-size study, {arm} arm", insample=False, quant=Q, qctrl=Q[0], losses=["QLIKE", "MSE", "MAE"], lctrl="QLIKE",
                units={arm: dict(win=YEARS, S=S, ref={L: ref for L in S}, sizes=SZ)})


def roll_daily():
    import long_core as lc
    YEARS = list(range(2008, 2025)); CL = ["LSTM", "EWMA", "GARCH11_t", "HARq"]; units = {}; subs = [tuple(c) for c in itertools.combinations(range(10), 3)]
    for ver in ("range_m",):                                   # gap_m has no QLSTMT forecasts saved
        S = {L: {m: np.zeros((len(subs) + 1 if m in ("QLSTMT", "LSTM") else 1, len(YEARS))) for m in ["QLSTMT"] + CL} for L in ("QLIKE", "MSE", "MAE")}; ref = {}
        for i, y in enumerate(YEARS):
            D = lc.prepare(ver, y, length=1); yy, va, te = D["y"], D["va"], D["te"]; val = {}; P = HERE / "long_preds_rolling" / ver / str(y)
            for m in ["QLSTMT", "LSTM"]:
                F = [_cal(yy, va, te, np.load(P / f"{m}_s{s}.npz")) for s in range(10)]; T = np.stack([f[0] for f in F]); V = np.stack([f[1] for f in F])
                for L in S:
                    for j, c in enumerate(subs): S[L][m][j, i] = np.sum(dl.loss(L, yy[te], T[list(c)].mean(0)))
                    S[L][m][-1, i] = np.sum(dl.loss(L, yy[te], T.mean(0)))
                vv = dl.loss("QLIKE", yy[va], V.mean(0)); val[m] = float(np.nanmean(vv[np.isfinite(vv)]))
            for m in CL[1:]:
                f, fv = _cal(yy, va, te, np.load(P / f"{m}.npz"))
                for L in S: S[L][m][0, i] = np.sum(dl.loss(L, yy[te], f))
                vv = dl.loss("QLIKE", yy[va], fv); val[m] = float(np.nanmean(vv[np.isfinite(vv)]))
            ref[y] = min(CL, key=lambda m: val[m])
        units[ver] = dict(win=YEARS, S=S, ref={L: ref for L in S})
    return dict(name="daily copper rolling QLSTMT (range target)", insample=False, quant=["QLSTMT"], qctrl="QLSTMT", losses=["QLIKE", "MSE", "MAE"], lctrl="QLIKE", units=units)


def mc_daily():
    P = HERE / "mc_daily_preds"; M = ["IsingMem_anch", "QRC_angle_anch", "HAR_anch"]; units = {}
    for a in sorted(p.name for p in P.iterdir() if p.is_dir()):
        origins = sorted(p.name for p in (P / a).iterdir()); S = {L: {m: np.zeros((1, len(origins))) for m in M} for L in ("QLIKE", "MSE", "MAE")}
        for i, o in enumerate(origins):
            meta = np.load(P / a / o / "meta.npz"); y = meta["y"]
            for m in M:
                z = np.load(P / a / o / f"{m}.npz"); p = z["p"] if str(z["kind"]) == "log" else np.log(z["p"]); assert len(p) == len(y), (a, o, m)
                for L in S: S[L][m][0, i] = np.sum(dl.loss(L, y, p))
        units[a] = dict(win=origins, S=S, ref={L: {o: "HAR_anch" for o in origins} for L in S})
    return dict(name="daily multi-commodity rolling origins (IsingMem, QRC)", insample=False, quant=["IsingMem_anch", "QRC_angle_anch"], qctrl="IsingMem_anch",
                losses=["QLIKE", "MSE", "MAE"], lctrl="QLIKE", units=units)


LOADERS = {"smalln_tuned": lambda: smalln("tuned"), "smalln_fixed": lambda: smalln("fixed"), "roll_daily": roll_daily, "mc_daily": mc_daily}


def trainsize_cells(D):
    out = []
    for u, U in D["units"].items():
        L0, q0 = D["lctrl"], D["qctrl"]; r = []
        for sz in sorted(U["sizes"][L0]):
            Q = U["sizes"][L0][sz][q0].sum(); C = sum(U["sizes"][L0][sz][U["ref"][L0][w]][i] for i, w in enumerate(U["win"])); r.append(float(np.log(Q / C)))
        rc = r[-1]; out.append(dict(choice="TRAINSIZE", unit=u, r=r, rc=rc))
    return out


def finish(c):
    r = np.array(c["r"]); c.update(N=len(r), s=float(r.std(ddof=1)), rms=float(np.sqrt(np.mean(r ** 2))), G_obs=float(c["rc"] - r.min()))
    c["G_pred"] = c["s"] * dl.emax(c["N"]); tol = max(0.01, 0.3 * c["G_pred"]); err = r.min() + c["G_pred"] - c["rc"]
    c["H1_err"] = float(err); c["H1_pass"] = bool(abs(err) <= tol) if (c["N"] >= 4 and c["choice"] != "SEED") else None
    c["bound"] = float(r.min() + c["s"] * (c["N"] - 1) / np.sqrt(c["N"]))       # Samuelson: mean of options <= min + s (N-1)/sqrt(N), always
    return c


def selftest():
    assert abs(emax_t(10, np.inf) - dl.emax(10)) < 1e-9
    r = np.random.default_rng(1).standard_t(3, 17); assert r.mean() <= r.min() + r.std(ddof=1) * 16 / np.sqrt(17) + 1e-12   # Samuelson bound
    return "ok"


if __name__ == "__main__":
    print("selftest", selftest())
