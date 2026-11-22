"""LSTM / GRU / QLSTMT for experiments A, B, C (see HOURLY_DATABENTO_PROTOCOL_v2.json).

Usage: python hourly_databento_nn2.py EXP MODEL SEED [HIDDEN]   (EXP: A|B|C, MODEL: LSTM|GRU|QLSTMT)
Inputs are standardised per channel with train statistics; the target is standardised with train mean/sd.
DRY=1 uses a few hundred rows and 2 epochs and writes to hourly_databento_preds_v2_dry.
Writes hourly_databento_preds_v2/<EXP>_<MODEL>_h<HIDDEN>_s<SEED>.npz with test predictions in raw target units.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

torch.set_num_threads(1)
import returns_lib as rl
from hourly_databento_nn import GRUReg

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "databento" / "hourly_features_windows.csv"
DRY = os.environ.get("DRY") == "1"
OUTDIR = Path(__file__).with_name("hourly_databento_preds_v2_dry" if DRY else "hourly_databento_preds_v2")
CH = ["ret", "lrv1m", "lrv1s", "lvol", "lnsec"]
LR = 0.005


def load(exp):
    df = pd.read_csv(CSV, parse_dates=["timestamp_utc"])
    df = df[df["ok_" + ("B" if exp == "B" else exp)]]
    if exp == "C":
        cols = [[f"c{k + 1}"] for k in range(12)]
    else:
        cols = [[f"a_{c}_{l + 1}" for c in CH] for l in range(10)]
    tgt = "yb" if exp == "B" else "y"
    if DRY:
        df = pd.concat([df[df.split == s].head(n) for s, n in [("train", 400), ("val", 100), ("test", 100)]])
    xs = {s: np.stack([df[df.split == s][c].values for c in cols], axis=1) for s in ["train", "val", "test"]}  # (N, T, F)
    mu = xs["train"].reshape(-1, xs["train"].shape[-1]).mean(0); sd = xs["train"].reshape(-1, xs["train"].shape[-1]).std(0) + 1e-12
    ytr = df[df.split == "train"][tgt]; ym, ys = ytr.mean(), ytr.std()
    T = lambda s: torch.tensor(((xs[s] - mu) / sd).astype("float32"))
    Y = lambda s: torch.tensor(((df[df.split == s][tgt].values - ym) / ys).astype("float32"))
    data = {"Xtr": T("train"), "ytr": Y("train"), "Xva": T("val"), "yva": Y("val"), "Xte": T("test")}
    return data, df[df.split == "test"], tgt, ym, ys


def match_hidden(kind, n_features, target_params):
    best = None
    for h in range(2, 40):
        if kind == "LSTM":
            from Factory import ShallowRegressionLSTM as M
            n = sum(p.numel() for p in M(n_features, h).parameters())
        else:
            n = sum(p.numel() for p in GRUReg(n_features, h).parameters())
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (h, n)
    return best


def main():
    exp, model, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data, te, tgt, ym, ys = load(exp)
    nf = data["Xtr"].shape[-1]
    if model == "QLSTMT":
        hidden = rl.HIDDEN
    else:
        qn = sum(p.numel() for p in rl.make_model("QLSTMT", nf).parameters() if p.requires_grad)
        hidden = int(sys.argv[4]) if len(sys.argv) > 4 else match_hidden(model, nf, qn)[0]
        rl.HIDDEN = hidden
        if model == "GRU":
            rl.make_model = lambda name, n_features, **kw: GRUReg(n_features, hidden)
    m, info = rl.fit(model, data, lr=LR, seed=seed, max_epochs=2 if DRY else 25)
    pred = rl.predict(m, data["Xte"]).ravel() * ys + ym
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"{exp}_{model}_h{hidden}_s{seed}.npz", pred=pred, y=te[tgt].values, t=te.timestamp_utc.astype("int64").values,
             seconds=info["seconds"], epochs=info["epochs"], n_params=info["n_params"], n_train=len(data["Xtr"]), best_val=info["best_val"])
    print(f"{exp} {model} h{hidden} s{seed}: ntr={len(data['Xtr'])} nva={len(data['Xva'])} nte={len(pred)} nf={nf} T={data['Xtr'].shape[1]} "
          f"epochs={info['epochs']} params={info['n_params']} {info['seconds']:.1f}s val={info['best_val']:.4f}", flush=True)


if __name__ == "__main__":
    main()
