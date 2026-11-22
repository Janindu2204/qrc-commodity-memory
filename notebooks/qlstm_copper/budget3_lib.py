"""Second dataset for the manufactured-advantage budget (long_prereg_budget3.json), and the 'typical researcher' statistic for both datasets.
Saved forecasts only (v2_preds, Yahoo hourly, 11 commodity futures, test 2025-05-01 .. 2026-10-08, target = log of the mean range variance
over the next 24 bars, as in v2_core.prepare_v2). No model is trained.

Factors (control / pitfall):
  SEED   QLSTMT 10-seed mean / best 3 of 10 seeds (only QLSTMT has seeds; other quantum models are deterministic)
  WINDOW pooled test period / best calendar quarter (quarters with >= 500 test rows)
  ASSET  geometric mean over the 11 assets / best asset
  LOSS   QLIKE / best of QLIKE, squared error and absolute error of the log variance
  QMODEL QLSTMT (fixed in advance) / best of QLSTMT, QRCangle, Ising, QKernel, QKernelPS, QKAN
  REF    classical model with the LOWEST validation QLIKE (past data) / classical model with the HIGHEST validation QLIKE
Classical candidates: LSTM (10-seed mean), ESN, RBF, RandFeat, MLP_ctrl, KAN_ctrl, LGBM, HARq, EWMA, GARCH11_t (all per-asset '_q' fits where named so).
'Best' = the option with the lowest loss ratio quantum/reference; 'typical' = the mean of the log ratio over the open options (a researcher who
picks an option at random). Null: whole calendar months swapped between the quantum model and the reference with probability 1/2, months
shared across assets; the gain over control is computed inside each replicate.
"""
import itertools
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
ASSETS = ["CL_F", "GC_F", "HG_F", "HO_F", "NG_F", "PA_F", "PL_F", "SI_F", "ZC_F", "ZS_F", "ZW_F"]
QUANT = ["QLSTMT", "QRCangle_q", "Ising_q", "QKernel_q", "QKernelPS_q", "QKAN_q"]
CLASS = ["LSTM", "ESN_q", "RBF_q", "RandFeat_q", "MLP_ctrl_q", "KAN_ctrl_q", "LGBM_q", "HARq", "EWMA", "GARCH11_t"]
LOSSES = ["QLIKE", "MSE", "MAE"]; FACTORS = ["SEED", "WINDOW", "ASSET", "LOSS", "QMODEL", "REF"]
SUBSETS = [tuple(c) for c in itertools.combinations(range(10), 3)]; FULL = len(SUBSETS)


def loss(kind, y, p):
    e = y - p
    return {"QLIKE": np.exp(e) - e - 1, "MSE": e ** 2, "MAE": np.abs(e)}[kind]


def _cal_log(lev_te, lev_va, y_va):
    """saved arrays are UNCALIBRATED variance levels (v2_run.py docstring). As in v2_report.py: scale c = mean(exp(y)/level) on valid
    validation rows (>= 200 needed), forecast = log(c * level). Returns None if the forecast is unusable (non-finite or non-positive)."""
    ok = np.isfinite(lev_va) & (lev_va > 0)
    if ok.sum() < 200 or not (np.isfinite(lev_te).all() and (lev_te > 0).all()): return None
    c = float(np.mean(np.exp(y_va[ok]) / lev_va[ok]))
    if not (np.isfinite(c) and c > 0): return None
    with np.errstate(divide="ignore"): v = np.log(c * lev_va)
    return np.log(c * lev_te), v


def load_asset(a):
    """returns dict with y_te, y_va, month, quarter, F[model] = (log forecasts on test (nsub+1 or 1, n), log forecast on validation)"""
    import v2_core as vc
    P = vc.prepare_v2(a); te, va = P["te"], P["va"]; idx = P["d"].index; y = P["y"]
    out = dict(y_te=y[te], y_va=y[va], month=np.array(idx[te].year * 12 + idx[te].month), F={}, dropped=[])
    out["quarter"] = np.array(idx[te].year * 10 + (idx[te].month - 1) // 3 + 1)
    for m, base in (("QLSTMT", "QLSTMT_q"), ("LSTM", "LSTM_q")):
        S = [np.load(HERE / "v2_preds" / a / f"{base}_s{s}.npz") for s in range(10)]; T = np.stack([z["lv_te"] for z in S]); V = np.stack([z["lv_val"] for z in S])
        assert T.shape[1] == len(te) and V.shape[1] == len(va), (a, m)
        res = [_cal_log(T[list(c)].mean(0), V[list(c)].mean(0), y[va]) for c in SUBSETS] + [_cal_log(T.mean(0), V.mean(0), y[va])]
        if any(r is None for r in res): out["dropped"].append(m); continue
        out["F"][m] = (np.stack([r[0] for r in res]), res[-1][1])
    for m in QUANT[1:] + CLASS[1:]:
        z = np.load(HERE / "v2_preds" / a / f"{m}.npz"); assert len(z["lv_te"]) == len(te) and len(z["lv_val"]) == len(va), (a, m)
        r = _cal_log(z["lv_te"], z["lv_val"], y[va])
        if r is None: out["dropped"].append(m); continue
        out["F"][m] = (r[0][None], r[1])
    return out


def tables(D):
    """loss sums per calendar month; a model whose loss is non-finite anywhere for this asset is dropped for this asset (recorded)"""
    months = np.unique(D["month"]); inv = np.searchsorted(months, D["month"]); S = {L: {} for L in LOSSES}; drop = list(D["dropped"])
    with np.errstate(over="ignore", invalid="ignore"):
        for m, F in D["F"].items():
            sums = {L: np.stack([np.bincount(inv, loss(L, D["y_te"], f), minlength=len(months)) for f in F[0]]) for L in LOSSES}
            vq = loss("QLIKE", D["y_va"], F[1]); vq = vq[np.isfinite(vq)]
            if not all(np.isfinite(v).all() for v in sums.values()) or len(vq) < 200: drop.append(m); continue
            for L in LOSSES: S[L][m] = sums[L]
    qm = np.array([D["quarter"][D["month"] == mo][0] for mo in months]); nq = pd.Series(D["quarter"]).value_counts()
    with np.errstate(over="ignore", invalid="ignore"):
        val = {m: float(np.nanmean(np.where(np.isfinite(l := loss("QLIKE", D["y_va"], D["F"][m][1])), l, np.nan))) for m in S["QLIKE"]}
    return dict(S=S, months=months, qm=qm, quarters=[q for q in sorted(set(qm)) if nq.get(q, 0) >= 500], val=val, dropped=drop)


def refs(T):
    c = {m: T["val"][m] for m in CLASS if m in T["val"]}; return min(c, key=c.get), max(c, key=c.get)


def per_asset(T, flip, f, stat):
    good, bad = refs(T); r = bad if f["REF"] else good; vals = []
    for q in ([x for x in QUANT if x in T["S"]["QLIKE"]] if f["QMODEL"] else ["QLSTMT"]):
        for L in (LOSSES if f["LOSS"] else ["QLIKE"]):
            Sq = T["S"][L][q]; subs = list(range(FULL)) if (f["SEED"] and q == "QLSTMT") else [Sq.shape[0] - 1]
            Rf = T["S"][L][r][-1]; Qf = T["S"][L][q][-1]
            for s in subs:
                qa = np.where(flip, Rf, Sq[s]) if flip is not None else Sq[s]; ra = np.where(flip, Qf, Rf) if flip is not None else Rf
                for w in ([[qq] for qq in T["quarters"]] if f["WINDOW"] else [None]):
                    c = np.ones(len(qa), bool) if w is None else np.isin(T["qm"], w); vals.append(np.log(qa[c].sum() / ra[c].sum()))
    vals = np.array(vals); return float(vals.min()) if stat == "best" else float(vals.mean())


def protocol(TT, flips, f, stat="best"):
    per = np.array([per_asset(TT[a], None if flips is None else flips[a], f, stat) for a in ASSETS])
    if f["ASSET"]: return float(np.exp(per.min() if stat == "best" else per.mean()))
    return float(np.exp(per.mean()))


def all_protocols(): return [dict(zip(FACTORS, b)) for b in itertools.product([False, True], repeat=6)]


def selftest():
    assert abs(loss("QLIKE", np.array([0.3]), np.array([0.3]))[0]) < 1e-12
    f = dict.fromkeys(FACTORS, False); assert sum(f.values()) == 0 and len(all_protocols()) == 64
    return "ok"


if __name__ == "__main__":
    print("selftest", selftest())
