"""Score the v2 improvements (see v2_prereg.json; decision rule: beat EWMA_multi_cal on a majority of assets AND >=1 Holm-significant asset-test).
Calibration: every model's level forecast is scaled by c = mean(y_level/forecast) on validation rows; test rows are scored afterwards."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from arch.bootstrap import MCS
import hourly_baselines as hb
import hourly_opt_common as oc
import mc_lib as mc, mc_report as rep, mc_combos as cb
import v2_core as vc

import os
TAG = os.environ.get("V2_TAG", "")   # suffix for output CSVs (rolling-origin runs)
P = json.load(open("v2_prereg.json"))
ASSETS = mc.ASSETS + ["HG_F"]
FAM = {"econ": ["EWMA", "EWMA_multi", "GARCH11_t", "HARq", "HARq_roll"], "ml": ["HARq_X", "LGBM_q", "LSTM_q"], "qml": ["Ising_q", "QRCangle_q", "QLSTMT_q"],
       "pool": ["HARq_pool", "LGBM_pool", "LSTM_pool", "Ising_pool", "QLSTMT_pool"]}
SEEDED = ["LSTM_q", "QLSTMT_q", "LSTM_pool", "QLSTMT_pool"]
REFS = ["EWMA", "EWMA_multi"]
MEMBERS = [m for ms in FAM.values() for m in ms]
CONTROLS = {"QRCangle_q": ["ESN_q", "RandFeat_q", "QRCnoent_q"], "Ising_q": ["ESN_q", "RandFeat_q"], "Ising_pool": ["ESN_pool"], "QKernel_q": ["QKernelPS_q", "RBF_q"], "QKAN_q": ["KAN_ctrl_q", "MLP_ctrl_q"]}


def load(a):
    out = {}
    for f in (vc.OUT / a).glob("*.npz"):
        n = f.stem
        if "_s" in n and n.rsplit("_s", 1)[0] in SEEDED: continue
        z = np.load(f); out[n] = (z["lv_val"], z["lv_te"])
    for m in SEEDED:
        fs = sorted((vc.OUT / a).glob(f"{m}_s*.npz"))
        if fs: out[m] = (np.nanmean([np.load(f)["lv_val"] for f in fs], 0), np.nanmean([np.load(f)["lv_te"] for f in fs], 0)); out[m + "__n"] = len(fs)
    return out


def holm(p):
    k = len(p); o = np.argsort(p); h = np.empty(k); run = 0
    for rank, i in enumerate(o):
        run = max(run, (k - rank) * p[i]); h[i] = min(1, run)
    return h


def main():
    rows, h2h = [], []
    for a in ASSETS:
        D = vc.prepare_v2(a); y = D["y"]; va, te = D["va"], D["te"]; d = D["d"]
        raw = load(a); nseeds = {k[:-3]: v for k, v in raw.items() if k.endswith("__n")}; raw = {k: v for k, v in raw.items() if not k.endswith("__n")}
        cal, val_lv = {}, {}
        for k, (lv_va, lv_te) in raw.items():
            ok = np.isfinite(lv_va) & (lv_va > 0)
            if ok.sum() < 200 or not np.isfinite(lv_te).all(): continue
            c = vc.calib_scale(np.exp(y[va][ok]), lv_va[ok]); cal[k] = c * lv_te; val_lv[k] = c * np.where(np.isfinite(lv_va) & (lv_va > 0), lv_va, np.nan)
        if not all(r_ in cal for r_ in REFS): print(a, "missing refs"); continue
        fam = {f: [m for m in ms if m in cal] for f, ms in FAM.items()}
        L = dict(cal); fm = {f: np.mean([L[m] for m in ms], 0) for f, ms in fam.items() if ms}
        combos = {"V1_econ_mean": fm["econ"], "V2_ml_mean": fm["ml"], "V3_qml_mean": fm["qml"], **({"V4_pooled_mean": fm["pool"]} if "pool" in fm else {}), "V5_all_equal": np.mean([L[m] for m in cal if m in MEMBERS], 0),
                  "V6_ewma_multi_plus_harq_roll": 0.5 * L["EWMA_multi"] + 0.5 * L["HARq_roll"], **({"V8_econ_plus_pooled": 0.5 * fm["econ"] + 0.5 * fm["pool"]} if "pool" in fm else {})}
        names = [m for m in cal if m in MEMBERS]
        try:
            vm = pd.DataFrame({m: oc.qlike_vec(np.where(np.isfinite(val_lv[m]), val_lv[m], np.nanmedian(val_lv[m])), np.exp(y[va])) for m in names})
            mc_ = MCS(vm, size=0.10, reps=300, block_size=24, method="R", seed=0); mc_.compute(); inc = set(mc_.included) | {"EWMA_multi"}
        except Exception:
            inc = set(names)
        combos["V7_mcs_trimmed"] = np.mean([L[m] for m in inc if m in L], 0)
        pool = {**L, **combos}
        yl = np.exp(y[te]); shock = (d.index[te] >= hb.SHOCK[0] - pd.Timedelta(hours=hb.H + 1)) & (d.index[te] <= hb.SHOCK[1])
        for wname, wm in (("all", np.ones(len(te), bool)), ("ex_shock", ~shock)):
            QL = pd.DataFrame({k: oc.qlike_vec(v[wm], yl[wm]) for k, v in pool.items()})
            try:
                m2 = MCS(QL, size=0.10, reps=300, block_size=48, method="R", seed=0); m2.compute(); inc2 = set(m2.included)
            except Exception: inc2 = set()
            for k in QL.columns:
                row = dict(asset=a, window=wname, model=k, is_combo=k.startswith("V"), n=int(wm.sum()), seeds=nseeds.get(k, np.nan), in_MCS=k in inc2, QLIKE=QL[k].mean())
                for ref in REFS:
                    if k == ref: continue
                    dl = QL[ref].values - QL[k].values
                    row[f"rel_{ref}"] = QL[k].mean() / QL[ref].mean(); row[f"DM_{ref}"] = hb.hac_t(pd.Series(dl), 48)
                    try: row[f"GW_{ref}"] = cb.gw(dl, lags=48) if np.ptp(dl) > 0 else np.nan
                    except Exception: row[f"GW_{ref}"] = np.nan
                rows.append(row)
        yl_all = np.exp(y[te])
        for q, cs in CONTROLS.items():
            for c_ in cs:
                if q in cal and c_ in cal:
                    ql_q, ql_c = oc.qlike_vec(cal[q], yl_all), oc.qlike_vec(cal[c_], yl_all)
                    h2h.append(dict(asset=a, quantum=q, control=c_, ratio=ql_q.mean() / ql_c.mean(), DM_t=hb.hac_t(pd.Series(ql_c - ql_q), 48)))
        print(a, "scored", flush=True)
    r = pd.DataFrame(rows)
    for ref in REFS:
        r[f"p_{ref}"] = 1 - stats.norm.cdf(r[f"DM_{ref}"]); r[f"holm_{ref}"] = np.nan
        sub = r[(r.asset != "HG_F") & r[f"DM_{ref}"].notna()]
        for w, g in sub.groupby("window"): r.loc[g.index, f"holm_{ref}"] = holm(g[f"p_{ref}"].values)
    r.to_csv(f"v2_report{TAG}.csv", index=False)
    hh = pd.DataFrame(h2h, columns=["asset", "quantum", "control", "ratio", "DM_t"]); hh.to_csv(f"v2_controls_h2h{TAG}.csv", index=False)
    print("\n== quantum vs matching classical control (10 commodities; ratio < 1 = quantum better; Z>0 = quantum better)")
    for (q, c_), g in hh[hh.asset != "HG_F"].groupby(["quantum", "control"]):
        print(f"  {q:12s} vs {c_:11s}: median ratio {g.ratio.median():.3f}, quantum better in {int((g.ratio < 1).sum())}/{len(g)} assets, Stouffer Z {g.DM_t.sum() / np.sqrt(len(g)):.2f}")
    pd.set_option("display.width", 200, "display.max_rows", 200)
    for ref in REFS:
        g = r[(r.window == "all") & (r.asset != "HG_F") & r[f"DM_{ref}"].notna()]
        s = g.groupby("model").agg(assets=("asset", "nunique"), med=(f"rel_{ref}", "median"), beats=(f"rel_{ref}", lambda x: int((x < 1).sum())),
                                   Z=(f"DM_{ref}", lambda t: t.sum() / np.sqrt(len(t))), raw=(f"p_{ref}", lambda x: int((x < .05).sum())),
                                   holm=(f"holm_{ref}", lambda x: int((x < .05).sum()))).sort_values("med"); s.index = [str(i) for i in s.index]
        print(f"\n== reference {ref}_cal, 10 commodities, all test rows ({len(g)} tests, Holm<.05: {int((g[f'holm_{ref}'] < .05).sum())})"); print(s.round(3).to_string())
        c = r[(r.window == "all") & (r.asset == "HG_F") & r[f"DM_{ref}"].notna()].set_index("model")[f"rel_{ref}"]
        print("copper alone:", c.sort_values().round(3).head(8).to_dict())


if __name__ == "__main__":
    main()
