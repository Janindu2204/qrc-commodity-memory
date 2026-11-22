"""Hourly LSTM / QLSTMT on the 24-bar log-variance target, univariate vs multivariate.

Usage: python hourly_nn.py MODEL VARIANT SEED   (MODEL: LSTM|QLSTMT, VARIANT: uni|multi)
Target is the residual y - lv24 (HAR also contains lv24); stored preds are total log variance.
Writes hourly_nn_preds/<MODEL>_<VARIANT>_s<SEED>.npz with test-origin predictions (log variance).
Fixed origin: train < 2025-05-01 (minus 30h gap) (interleaved 10-day blocks = early-stopping val), test >= 2025-05-01,
30h gap before test so overlapping 24-bar targets cannot leak. Train stride 2.
Reuses fit()/make_model() from returns_lib (Adam, MSE, early stopping on val only).
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

torch.set_num_threads(1)
import hourly_baselines as hb
import hourly_multivariate as hm
import returns_lib as rl

SEQ = 24
STRIDE = 2
TRAIN_END = pd.Timestamp("2025-05-01")
OUTDIR = Path(__file__).with_name("hourly_nn_preds")


def make(variant):
    d, r = hb.load(); X, y_vol, R, y_ret, hod = hb.build(r)
    F = pd.DataFrame({"ret": r, "lv24": X.lv24, "lv120": X.lv120, "lv480": X.lv480}, index=d.index)
    ER, EV = hm.exog_features(d.index)
    F = pd.concat([F, ER[hm.TICKERS].add_prefix("er_"), EV.add_prefix("ev_")], axis=1)
    F = F.fillna({"ret": 0.0})
    cols = list(F.columns) if variant == "multi" else ["ret", "lv24", "lv120", "lv480"]
    ok = F.notna().all(axis=1).values & y_vol.notna().values   # mask from ALL columns: uni/multi share rows
    F = F[cols]
    pos = np.arange(len(d))
    # window ending at row t needs the previous SEQ-1 rows complete too
    full = pd.Series(ok).rolling(SEQ).min().fillna(0).astype(bool).values
    t_tr = pos[full & (d.index < TRAIN_END - pd.Timedelta(hours=30))]
    t_te = pos[full & (d.index >= hb.TEST_START)]
    # interleaved validation blocks (10 calendar days, every 5th block) so early stopping sees all regimes;
    # train rows within 30h of a val block are purged (24-bar windows/targets overlap)
    days = np.asarray((d.index[t_tr] - d.index[t_tr[0]]).days)
    blk = days // 10
    is_va = blk % 5 == 4
    near = np.zeros(len(t_tr), bool)
    for b in np.unique(blk[is_va]):
        lo = d.index[t_tr[0]] + pd.Timedelta(days=int(b) * 10) - pd.Timedelta(hours=30)
        hi = d.index[t_tr[0]] + pd.Timedelta(days=int(b) * 10 + 10) + pd.Timedelta(hours=30)
        near |= (d.index[t_tr] >= lo) & (d.index[t_tr] < hi)
    t_va = t_tr[is_va][::STRIDE]; t_tr = t_tr[~near][::STRIDE]
    mu, sd = F.values[t_tr].mean(0), F.values[t_tr].std(0) + 1e-9
    Fz = ((F.values - mu) / sd).astype("float32")
    base = X.lv24.values  # residual target: network forecasts y - lv24 (a skip connection to the naive forecast)
    yres = y_vol.values - base
    ym, ys = yres[t_tr].mean(), yres[t_tr].std()
    yz = ((yres - ym) / ys).astype("float32")
    win = lambda ts: torch.tensor(np.stack([Fz[t - SEQ + 1: t + 1] for t in ts]))
    col = lambda ts: torch.tensor(yz[ts]).reshape(-1, 1)
    data = {"Xtr": win(t_tr), "ytr": col(t_tr), "Xva": win(t_va), "yva": col(t_va), "Xte": win(t_te)}
    data["base_te"] = base[t_te]
    data["val_const"] = float(np.mean(yz[t_va] ** 2))
    return data, t_te, ym, ys, d.index


def main():
    model, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data, t_te, ym, ys, idx = make(variant)
    t0 = time.time()
    # optional validation-tuned overrides (hourly_tune_nn.py): HN_HIDDEN, HN_LR, HN_QLAYERS, HN_TAG
    lr = float(os.environ.get("HN_LR", 0.005)); tag = os.environ.get("HN_TAG", "")
    if "HN_HIDDEN" in os.environ:
        rl.HIDDEN = int(os.environ["HN_HIDDEN"])
    kw = {"n_qlayers": int(os.environ["HN_QLAYERS"])} if model == "QLSTMT" and "HN_QLAYERS" in os.environ else None
    m, info = rl.fit(model, data, lr=lr, seed=seed, max_epochs=int(sys.argv[4]) if len(sys.argv) > 4 else 25, model_kwargs=kw)
    pred = rl.predict(m, data["Xte"]).ravel() * ys + ym + data["base_te"]
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"{model}{tag}_{variant}_s{seed}.npz", pred=pred, t_te=t_te,
             seconds=info["seconds"], epochs=info["epochs"], n_params=info["n_params"],
             n_train=len(data["Xtr"]), best_val=info["best_val"])
    print(f"{model}{tag} {variant} s{seed}: ntr={len(data['Xtr'])} nte={len(t_te)} epochs={info['epochs']} "
          f"params={info['n_params']} {info['seconds']:.0f}s val={info['best_val']:.3f} (const {data['val_const']:.3f})")


if __name__ == "__main__":
    main()
