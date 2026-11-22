"""LSTM / GRU / QLSTMT on the Databento hourly next-hour-return dataset (see HOURLY_DATABENTO_PROTOCOL.json).

Usage: python hourly_databento_nn.py MODEL SEED [HIDDEN]   (MODEL: LSTM|GRU|QLSTMT)
Univariate: inputs are the last SEQ=10 hourly log returns, standardised with train statistics; target is the standardised next return.
Writes hourly_databento_preds/<MODEL>_h<HIDDEN>_s<SEED>.npz with test predictions (raw log-return units).
Reuses fit()/predict() from returns_lib (Adam, MSE, early stopping on validation only).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

torch.set_num_threads(1)
import returns_lib as rl

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "databento" / "hourly_returns_windows.csv"
OUTDIR = Path(__file__).with_name("hourly_databento_preds")
SEQ = 10
LR = 0.005


class GRUReg(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.linear(hn[0]).flatten()


def load():
    df = pd.read_csv(CSV, parse_dates=["timestamp_utc"])
    xc = [f"x{i + 1}" for i in range(SEQ)]
    tr = df[df.split == "train"]
    mu, sd = tr[xc].values.mean(), tr[xc].values.std()
    ym, ys = tr.y.mean(), tr.y.std()
    T = lambda s: torch.tensor(((df[df.split == s][xc].values - mu) / sd).astype("float32")).unsqueeze(-1)
    Y = lambda s: torch.tensor(((df[df.split == s].y.values - ym) / ys).astype("float32"))
    data = {"Xtr": T("train"), "ytr": Y("train"), "Xva": T("val"), "yva": Y("val"), "Xte": T("test")}
    te = df[df.split == "test"]
    return data, te, ym, ys


def main():
    model, seed = sys.argv[1], int(sys.argv[2])
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else rl.HIDDEN
    rl.HIDDEN = hidden
    if model == "GRU":
        rl.make_model = lambda name, n_features, **kw: GRUReg(n_features, hidden)
    data, te, ym, ys = load()
    m, info = rl.fit(model, data, lr=LR, seed=seed, max_epochs=25)
    pred = rl.predict(m, data["Xte"]).ravel() * ys + ym
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"{model}_h{hidden}_s{seed}.npz", pred=pred, y=te.y.values, t=te.timestamp_utc.astype("int64").values,
             seconds=info["seconds"], epochs=info["epochs"], n_params=info["n_params"], n_train=len(data["Xtr"]), best_val=info["best_val"])
    print(f"{model} h{hidden} s{seed}: ntr={len(data['Xtr'])} nva={len(data['Xva'])} nte={len(pred)} epochs={info['epochs']} "
          f"params={info['n_params']} {info['seconds']:.0f}s val={info['best_val']:.4f}", flush=True)


if __name__ == "__main__":
    main()
