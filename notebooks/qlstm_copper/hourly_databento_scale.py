"""Scaling-curve runs: LSTM / GRU / QLSTMT trained on the most recent N years before 2026, tested on the same 2026 hours.
(see HOURLY_DATABENTO_PROTOCOL_v3.json)

Usage: python hourly_databento_scale.py YEARS MODEL SEED   (YEARS: 1|2|4|8|16, MODEL: LSTM|GRU|QLSTMT)
DRY=1: a few hundred rows, 2 epochs, writes to hourly_databento_preds_v3_dry.
Target: ln realized variance of the next hour. Inputs: last 10 hourly bars x [ret, lrv1m, lvol, hod sin, hod cos].
Validation: every 5th block of 500 windows in the span; train windows within PURGE bars of a validation window are dropped.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

torch.set_num_threads(1)
import returns_lib as rl
from hourly_databento_nn import GRUReg
from hourly_databento_nn2 import match_hidden

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "databento" / "hourly_ext_windows.csv"
DRY = os.environ.get("DRY") == "1"
OUTDIR = Path(__file__).with_name("hourly_databento_preds_v3_dry" if DRY else "hourly_databento_preds_v3")
CH = ["ret", "lrv1m", "lvol", "hs", "hc"]
BLOCK, VAL_EVERY, PURGE, LR = 500, 5, 10, 0.005


def load_ext():
    return pd.read_csv(CSV, parse_dates=["timestamp_utc"])


def split_for_years(df, years):
    """Returns train, val, test frames for the most recent `years` years before 2026-01-01."""
    start = pd.Timestamp(f"{2026 - years}-01-01", tz="UTC")
    pre = df[(df.split == "pre") & (df.timestamp_utc >= start)].sort_values("timestamp_utc").reset_index(drop=True)
    is_val = (np.arange(len(pre)) // BLOCK) % VAL_EVERY == VAL_EVERY - 1
    vt = pre.t.values[is_val]
    j = np.searchsorted(vt, pre.t.values)
    near = np.minimum(np.abs(pre.t.values - vt[np.clip(j - 1, 0, len(vt) - 1)]), np.abs(pre.t.values - vt[np.clip(j, 0, len(vt) - 1)])) <= PURGE
    tr = pre[~is_val & ~near]
    return tr, pre[is_val], df[df.split == "test"]


def arrays(d):
    return np.stack([d[[f"a_{c}_{l + 1}" for c in CH]].values for l in range(10)], axis=1)   # (N, 10, 5)


def main():
    years, model, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    tr, va, te = split_for_years(load_ext(), years)
    if DRY:
        tr, va, te = tr.head(400), va.head(100), te.head(100)
    xs = {"tr": arrays(tr), "va": arrays(va), "te": arrays(te)}
    mu, sd = xs["tr"].reshape(-1, 5).mean(0), xs["tr"].reshape(-1, 5).std(0) + 1e-12
    ym, ys = tr.yb.mean(), tr.yb.std()
    T = lambda k: torch.tensor(((xs[k] - mu) / sd).astype("float32"))
    data = {"Xtr": T("tr"), "ytr": torch.tensor(((tr.yb.values - ym) / ys).astype("float32")),
            "Xva": T("va"), "yva": torch.tensor(((va.yb.values - ym) / ys).astype("float32")), "Xte": T("te")}
    if model == "QLSTMT":
        hidden = rl.HIDDEN
    else:
        qn = sum(p.numel() for p in rl.make_model("QLSTMT", 5).parameters() if p.requires_grad)
        hidden = match_hidden(model, 5, qn)[0]
        rl.HIDDEN = hidden
        if model == "GRU":
            rl.make_model = lambda name, n_features, **kw: GRUReg(n_features, hidden)
    m, info = rl.fit(model, data, lr=LR, seed=seed, max_epochs=2 if DRY else 25)
    pred = rl.predict(m, data["Xte"]).ravel() * ys + ym
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"S{years}_{model}_h{hidden}_s{seed}.npz", pred=pred, y=te.yb.values, t=te.timestamp_utc.astype("int64").values,
             seconds=info["seconds"], epochs=info["epochs"], n_params=info["n_params"], n_train=len(tr), best_val=info["best_val"])
    print(f"S{years} {model} h{hidden} s{seed}: ntr={len(tr)} nva={len(va)} nte={len(pred)} epochs={info['epochs']} "
          f"params={info['n_params']} {info['seconds']:.1f}s val={info['best_val']:.4f}", flush=True)


if __name__ == "__main__":
    main()
