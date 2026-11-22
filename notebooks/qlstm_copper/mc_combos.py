"""Score the pre-registered forecast combinations (mc_combos_prereg.json) against EWMA_tuned on the 10 commodities.
Averages variance LEVELS. Losses: QLIKE (primary, DM HAC 48 per asset) and log-MSE. Holm over combos x assets per window; pooled Stouffer Z
across assets plus a sign count. Weights for C9 use only realised losses at least 24 bars old. Preliminary, AI-written."""
import json
import numpy as np, pandas as pd
from scipy import stats
import hourly_baselines as hb
import hourly_opt_common as oc
import mc_lib as mc
import mc_report as rep
import statsmodels.api as sm
from arch.bootstrap import MCS

ASSETS_RUN = mc.ASSETS + ["HG_F"]          # copper is scored but kept OUT of the pooled 10-commodity statistics


def gw(dl, lags=48):
    """Giacomini-White (2006) test of equal conditional predictive ability, test function h_t = [1, dL_t]:
    moment Z_t = h_t * dL_{t+1}, Wald W = T * zbar' Omega^-1 zbar ~ chi2(2), Omega = Newey-West HAC (lag `lags`) of Z."""
    dl = np.asarray(dl, float); T = len(dl) - 1
    Z = np.c_[dl[1:], dl[:-1] * dl[1:]]
    zb = Z.mean(0); Zc = Z - zb
    om = Zc.T @ Zc / T
    for k in range(1, lags + 1):
        G = Zc[k:].T @ Zc[:-k] / T
        om += (1 - k / (lags + 1)) * (G + G.T)
    w = float(T * zb @ np.linalg.solve(om, zb))
    return float(1 - stats.chi2.cdf(w, 2))

P = json.load(open("mc_combos_prereg.json"))
FAM = P["families"]
MEMBERS = [m for f in FAM.values() for m in f]


def combos_for(levels, S):
    fam_mean = {f: np.mean([levels[m] for m in ms], 0) for f, ms in FAM.items()}
    out = {"C1_econ_mean": fam_mean["econometric"], "C2_ml_mean": fam_mean["ml"], "C3_qml_mean": fam_mean["qml"]}
    out["C4_mean_of_family_means"] = np.mean(list(fam_mean.values()), 0)
    out["C5_all_equal"] = np.mean([levels[m] for m in MEMBERS], 0)
    ew = levels["EWMA_tuned"]
    out["C6_ewma_plus_ml"] = 0.5 * ew + 0.5 * fam_mean["ml"]
    out["C7_ewma_plus_qml"] = 0.5 * ew + 0.5 * fam_mean["qml"]
    rest = np.mean([levels[m] for m in ("GARCH11_t", "GJR_t", "HAR_anch")], 0)
    out["C8_ewma_plus_econ_rest"] = 0.5 * ew + 0.5 * rest
    # C9 recursive inverse-loss weights, losses at least 24 bars old
    te = S["te"]; y = np.exp(S["yv"][te])
    L = np.stack([oc.qlike_vec(levels[m], y) for m in MEMBERS], 1)          # (T, M)
    cs = np.vstack([np.zeros((1, L.shape[1])), np.cumsum(L, 0)])
    avail = np.searchsorted(te, te - 24, side="right")                       # rows with te[j] <= te[i]-24
    W = np.full((len(te), len(MEMBERS)), 1.0 / len(MEMBERS))
    for i in range(len(te)):
        k = avail[i]
        if k >= 200:
            m = cs[k] / k
            w = 1.0 / np.maximum(m, 1e-9); W[i] = w / w.sum()
    out["C9_recursive_weighted"] = (W * np.stack([levels[m] for m in MEMBERS], 1)).sum(1)
    return out


def main():
    rows = []
    for a in ASSETS_RUN:
        S = mc.prepare(a); y = S["yv"][S["te"]]; s2 = S["s2"]
        raw = rep.load_asset(a); raw = {k: v for k, v in raw.items() if not k.endswith("_nseeds")}
        levels = {k: oc.level(p, kind, s2) for k, (p, kind) in raw.items()}
        need = set(MEMBERS) | {"EWMA_tuned"}
        miss = need - set(levels)
        assert not miss, (a, miss)
        allm = {**{m: levels[m] for m in sorted(need)}, **combos_for(levels, S)}
        for w, wm in S["wins"].items():
            ref = oc.qlike_vec(allm["EWMA_tuned"][wm], np.exp(y[wm]))
            ref_lm = np.mean((y[wm] - np.log(allm["EWMA_tuned"][wm])) ** 2)
            QL = pd.DataFrame({k: oc.qlike_vec(v[wm], np.exp(y[wm])) for k, v in allm.items()})
            try:
                m_ = MCS(QL, size=0.10, reps=500, block_size=48, method="R", seed=0); m_.compute(); inc = set(m_.included)
            except Exception:
                inc = set()
            for k, v in allm.items():
                ql = QL[k].values
                rows.append(dict(asset=a, window=w, model=k, is_combo=k.startswith("C"), QLIKE_rel_EWMA=ql.mean() / ref.mean(),
                                 logMSE_rel_EWMA=np.mean((y[wm] - np.log(v[wm])) ** 2) / ref_lm,
                                 DM_t=hb.hac_t(pd.Series(ref - ql), 48) if k != "EWMA_tuned" else np.nan,
                                 GW_p=gw(ref - ql) if k != "EWMA_tuned" else np.nan, in_MCS10=k in inc))
        print(a, "ok", flush=True)
    r = pd.DataFrame(rows)
    r["p"] = 1 - stats.norm.cdf(r.DM_t)
    r["holm"] = np.nan
    pooled = r[r.asset != "HG_F"]
    for w, g in pooled[pooled.is_combo].groupby("window"):
        r.loc[g.index, "holm"] = rep.holm_bh(g.p.values)[0]
    r.to_csv("mc_combos.csv", index=False)
    pd.set_option("display.width", 200)
    for w in ("all", "ex_shock"):
        g = r[(r.window == w) & (r.model != "EWMA_tuned") & (r.asset != "HG_F")]
        s = g.groupby("model").agg(is_combo=("is_combo", "first"), med_QLIKE_rel=("QLIKE_rel_EWMA", "median"), beats=("QLIKE_rel_EWMA", lambda x: int((x < 1).sum())),
                                   med_logMSE_rel=("logMSE_rel_EWMA", "median"), stouffer_Z=("DM_t", lambda t: t.sum() / np.sqrt(len(t))),
                                   raw_sig=("p", lambda x: int((x < .05).sum())), holm_sig=("holm", lambda x: int((x < .05).sum())),
                                   GW_sig=("GW_p", lambda x: int((x < .05).sum())), in_MCS=("in_MCS10", "sum")).sort_values("med_QLIKE_rel")
        s.index = [str(i) for i in s.index]
        print(f"\n== window {w} (reference EWMA_tuned; combos tested: {int(r[(r.window==w)&r.is_combo].shape[0])}; Holm<.05: {int((r[(r.window==w)].holm<.05).sum())})")
        print(s.round(3).to_string())
        c = r[(r.window == w) & (r.asset == "HG_F") & (r.is_combo)][["model", "QLIKE_rel_EWMA", "DM_t", "GW_p", "in_MCS10"]]
        print(f"-- copper (HG_F) alone, window {w}:"); print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
