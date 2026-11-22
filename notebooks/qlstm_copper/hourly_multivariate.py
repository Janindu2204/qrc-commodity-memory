"""Hourly univariate vs multivariate screen for HG=F, same walk-forward as hourly_baselines.py.

Exogenous series (Yahoo hourly, data/hourly_exog/, stamps = bar start, ET) are used only
through bars that complete no later than copper bar t does: the exog bar stamped t
(Yahoo stamps = bar start, :00 series only; both complete at t+1h, the forecast origin). Missing exog bars -> return 0 / ffill<=3h variance. Add-one tests against
the univariate baseline (ret: ridge on own lags; vol: HAR), then all-in. Holm over
the add-one family. Preliminary, AI-written, not independently re-derived.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import hourly_baselines as hb

ROOT = Path(__file__).resolve().parents[2]
TICKERS = ["GC_F", "SI_F", "PL_F", "ES_F", "CL_F", "BZ_F", "ZN_F", "DX-Y.NYB",
           "EURUSD_X", "AUDUSD_X", "USDCAD_X", "USDZAR_X"]
OUT = Path(__file__).with_name("hourly_multivariate.csv")


def exog_features(idx):
    ret, var = {}, {}
    for t in TICKERS:
        d = pd.read_csv(ROOT / "data" / "hourly_exog" / f"{t}_1h.csv", parse_dates=["timestamp"]).set_index("timestamp")
        d = d[d.index.minute == 0]
        r = np.log(d.close).diff()
        r[d.index.to_series().diff() > pd.Timedelta("2h")] = np.nan
        lv = np.log((r ** 2).rolling(24, min_periods=19).mean() + 1e-10)
        ret[t] = r.reindex(idx).fillna(0.0).values        # bar stamped t completes at t+1h, same as copper bar t
        ret[t + "_2"] = r.reindex(idx - pd.Timedelta("1h")).fillna(0.0).values
        var[t] = lv.reindex(lv.index.union(idx)).ffill(limit=3).reindex(idx).values
    return (pd.DataFrame(ret, index=idx), pd.DataFrame(var, index=idx))


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj


def main():
    d, r = hb.load(); X, y_vol, R, y_ret, hod = hb.build(r)
    months = pd.date_range(hb.TEST_START, d.index.max(), freq="MS")
    ER, EV = exog_features(d.index)
    shock = (d.index >= hb.SHOCK[0] - pd.Timedelta(hours=hb.H + 1)) & (d.index <= hb.SHOCK[1])
    wins = {"all": np.ones(len(d), bool), "ex_shock": ~shock}
    har_cols = ["lv24", "lv120", "lv480"]
    rows = []
    base_ret = hb.walk_forward(R, y_ret, 10000, months)
    base_vol = hb.walk_forward(X[har_cols], y_vol, 1.0, months)

    def evaluate(task, name, cand, base):
        for w, wm in wins.items():
            if task == "ret":
                m = cand.notna() & base.notna() & y_ret.notna() & (d.index >= hb.TEST_START) & wm
                e_b, e_c = y_ret[m] - base[m], y_ret[m] - cand[m]
                R2 = 1 - float((e_c ** 2).sum() / (y_ret[m] ** 2).sum())
                diff = e_b ** 2 - e_c ** 2 + (base[m] - cand[m]) ** 2   # Clark-West, nested
                t = hb.hac_t(diff, 5)
                rows.append(dict(task=task, window=w, spec=name, n=int(m.sum()), R2_OS_vs_RW=R2, t_vs_base=t))
            else:
                m = cand.notna() & base.notna() & y_vol.notna() & (d.index >= hb.TEST_START) & wm
                yt = y_vol[m]
                rel = float(((yt - cand[m]) ** 2).mean() / ((yt - base[m]) ** 2).mean())
                t = hb.hac_t((yt - base[m]) ** 2 - (yt - cand[m]) ** 2, 2 * hb.H)
                rows.append(dict(task=task, window=w, spec=name, n=int(m.sum()), MSE_rel_HAR=rel, t_vs_base=t))

    for tk in TICKERS:
        Zr = pd.concat([R, ER[[tk, tk + "_2"]]], axis=1)
        evaluate("ret", f"+{tk}", hb.walk_forward(Zr, y_ret, 10000, months), base_ret)
        Zv = pd.concat([X[har_cols], EV[[tk]].rename(columns={tk: "x"})], axis=1)
        evaluate("vol", f"+{tk}", hb.walk_forward(Zv, y_vol, 1.0, months), base_vol)
    evaluate("ret", "ALL", hb.walk_forward(pd.concat([R, ER], axis=1), y_ret, 10000, months), base_ret)
    evaluate("vol", "ALL", hb.walk_forward(pd.concat([X[har_cols], EV.add_prefix("x_")], axis=1), y_vol, 1.0, months), base_vol)
    res = pd.DataFrame(rows)
    # Holm over add-one family per task/window (one-sided: candidate better)
    res["p_one_sided"] = 1 - stats.norm.cdf(res.t_vs_base)
    res["p_holm"] = np.nan
    for (task, w), g in res[res.spec != "ALL"].groupby(["task", "window"]):
        res.loc[g.index, "p_holm"] = holm(g.p_one_sided.values)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
