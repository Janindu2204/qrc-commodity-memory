"""Final scoring of every stored prediction (hourly_opt_preds/*.npz) on the same 7,667 test rows.

Per window (all / ex_shock): error relative to fixed-origin HAR (log-MSE and QLIKE), Diebold-Mariano t on QLIKE-loss differences
(HAC lag 48, positive = better than HAR), Model Confidence Set (arch, QLIKE loss, 10%, block bootstrap 48, 1000 reps) membership,
then Holm / Benjamini-Hochberg over all one-sided DM tests. Log models use exp(p + s2/2) (s2 from HAR training residuals).
"""
import numpy as np, pandas as pd
from scipy import stats
from arch.bootstrap import MCS
import hourly_baselines as hb
import hourly_opt_common as oc

REF = "HAR_fixed"


def main():
    S = oc.setup(); yv, te, s2, wins = S["yv"], S["te"], S["s2"], S["wins"]
    preds = oc.load_all()
    y = yv[te]
    lev = {k: oc.level(p, kind, s2) for k, (p, kind) in preds.items()}
    logp = {k: (p if kind == "log" else np.log(p)) for k, (p, kind) in preds.items()}
    rows = []
    for w, wm in wins.items():
        QL = pd.DataFrame({k: oc.qlike_vec(v[wm], np.exp(y[wm])) for k, v in lev.items()})
        ql_ref = QL[REF].values; mse_ref = np.mean((y[wm] - logp[REF][wm]) ** 2)
        try:
            mcs = MCS(QL, size=0.10, reps=1000, block_size=48, method="R", seed=0); mcs.compute()
            inc = set(mcs.included)
        except Exception as e:
            print("MCS failed", e); inc = set()
        for k in QL.columns:
            dm = hb.hac_t(pd.Series(ql_ref - QL[k].values), 48) if k != REF else np.nan
            rows.append(dict(window=w, model=k, n=int(wm.sum()), QLIKE=QL[k].mean(), QLIKE_rel_HAR=QL[k].mean() / QL[REF].mean(),
                             logMSE_rel_HAR=np.mean((y[wm] - logp[k][wm]) ** 2) / mse_ref, DM_t_QLIKE=dm, in_MCS10=k in inc))
    res = pd.DataFrame(rows)
    res["p"] = 1 - stats.norm.cdf(res.DM_t_QLIKE)
    res["p_holm"] = np.nan; res["p_bh"] = np.nan
    for w, g in res[res.model != REF].groupby("window"):
        p = g.p.values; k = len(p); o = np.argsort(p); holm = np.empty(k); run = 0
        for rank, i in enumerate(o):
            run = max(run, (k - rank) * p[i]); holm[i] = min(1, run)
        bh = np.empty(k); prev = 1
        for rank, i in reversed(list(enumerate(o))):
            prev = min(prev, p[i] * k / (rank + 1)); bh[i] = prev
        res.loc[g.index, "p_holm"] = holm; res.loc[g.index, "p_bh"] = bh
    res = res.sort_values(["window", "QLIKE"])
    res.to_csv("hourly_opt_report.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
