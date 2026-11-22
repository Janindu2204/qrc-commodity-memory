"""Score every stored per-asset forecast like hourly_opt_report.py, then pool across assets.

Per asset and window (all / ex_shock = excluding 2025-07-07..08-15): QLIKE and log-MSE relative to fixed-origin HAR, DM t on
QLIKE-loss differences (HAC 48), Model Confidence Set (arch, 10%). Pooled: Holm / BH over all asset x model one-sided DM tests per window.
Network cells = mean of seeds (LSTM 10, QLSTMT 5).
"""
from pathlib import Path
import os
import numpy as np, pandas as pd
from scipy import stats
from arch.bootstrap import MCS
import hourly_baselines as hb
import hourly_opt_common as oc
import mc_lib as mc

ROOT = Path(__file__).with_name(os.environ.get("MC_OUT", "mc_preds"))


def load_asset(a):
    out = {}
    for f in sorted((ROOT / a).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "p" in z.files:
            out[f.stem] = (z["p"], str(z["kind"]))
    for cell in ("LSTM_uni", "LSTM_multi", "QLSTMT_uni", "QLSTMT_multi", "LSTM-tuned_uni", "LSTM-tuned_multi", "QLSTMT-tuned_uni", "QLSTMT-tuned_multi"):
        fs = sorted((ROOT / a).glob(f"{cell}_s*.npz"))
        if fs:
            out[cell] = (np.mean([np.load(f)["pred"] for f in fs], 0), "log")
            out[cell + "_nseeds"] = len(fs)
    return out


REFS = ["HAR_fixed", "EWMA_0.97", "EWMA_tuned", "HAR_anch"]


def holm_bh(p):
    k = len(p); o = np.argsort(p); holm = np.empty(k); run = 0
    for rank, i in enumerate(o):
        run = max(run, (k - rank) * p[i]); holm[i] = min(1, run)
    bh = np.empty(k); prev = 1
    for rank, i in reversed(list(enumerate(o))):
        prev = min(prev, p[i] * k / (rank + 1)); bh[i] = prev
    return holm, bh


def main():
    rows = []
    for a in mc.ASSETS:
        if not (ROOT / a).exists():
            continue
        S = mc.prepare(a); y = S["yv"][S["te"]]; s2 = S["s2"]
        preds = load_asset(a)
        nseeds = {k: v for k, v in preds.items() if k.endswith("_nseeds")}
        preds = {k: v for k, v in preds.items() if not k.endswith("_nseeds")}
        lev = {k: oc.level(p, kind, s2) for k, (p, kind) in preds.items()}
        logp = {k: (p if kind == "log" else np.log(p)) for k, (p, kind) in preds.items()}
        for w, wm in S["wins"].items():
            QL = pd.DataFrame({k: oc.qlike_vec(v[wm], np.exp(y[wm])) for k, v in lev.items()})
            try:
                mcs = MCS(QL, size=0.10, reps=500, block_size=48, method="R", seed=0); mcs.compute(); inc = set(mcs.included)
            except Exception:
                inc = set()
            for k in QL.columns:
                row = dict(asset=a, window=w, model=k, n=int(wm.sum()), seeds=nseeds.get(k + "_nseeds", np.nan),
                           QLIKE=QL[k].mean(), in_MCS10=k in inc)
                for ref in REFS:
                    if ref not in QL.columns:
                        continue
                    row[f"rel_{ref}"] = QL[k].mean() / QL[ref].mean()
                    row[f"DM_{ref}"] = hb.hac_t(pd.Series(QL[ref].values - QL[k].values), 48) if k != ref else np.nan
                rows.append(row)
        print(a, "scored", flush=True)
    res = pd.DataFrame(rows)
    for ref in REFS:
        if f"DM_{ref}" not in res:
            continue
        res[f"p_{ref}"] = 1 - stats.norm.cdf(res[f"DM_{ref}"])
        res[f"holm_{ref}"] = np.nan; res[f"bh_{ref}"] = np.nan
        for w, g in res[(res.model != ref) & res[f"DM_{ref}"].notna()].groupby("window"):
            h, b = holm_bh(g[f"p_{ref}"].values)
            res.loc[g.index, f"holm_{ref}"] = h; res.loc[g.index, f"bh_{ref}"] = b
    res.to_csv("mc_report.csv" if os.environ.get("MC_OUT", "mc_preds") == "mc_preds" else f'{os.environ["MC_OUT"]}_report.csv', index=False)
    pd.set_option("display.width", 220, "display.max_rows", 500)
    for ref in REFS:
        if f"DM_{ref}" not in res:
            continue
        for w in ("all", "ex_shock"):
            g = res[(res.window == w) & (res.model != ref) & res[f"DM_{ref}"].notna()]
            summ = g.groupby("model").agg(assets=("asset", "nunique"), mean_rel=(f"rel_{ref}", "mean"), median_rel=(f"rel_{ref}", "median"),
                                          beats=(f"rel_{ref}", lambda s: int((s < 1).sum())), raw_sig=(f"p_{ref}", lambda s: int((s < .05).sum())),
                                          holm_sig=(f"holm_{ref}", lambda s: int((s < .05).sum())), in_MCS=("in_MCS10", "sum")).sort_values("median_rel")
            summ.index = [str(i) for i in summ.index]
            print(f"\n== reference {ref}, window {w}: {len(g)} tests; raw p<.05 {(g[f'p_{ref}'] < .05).sum()}, BH<.05 {(g[f'bh_{ref}'] < .05).sum()}, Holm<.05 {(g[f'holm_{ref}'] < .05).sum()}")
            print(summ.round(3).to_string())
    return res


if __name__ == "__main__":
    main()
