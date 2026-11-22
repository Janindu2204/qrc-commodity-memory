"""Return-target pipeline for the copper LSTM / QLSTM comparison.

Used by QLSTM_copper_returns.ipynb and run_returns.py. Reuses the model classes in
Factory.py unchanged; only the data preparation, training loop and metrics live here.
Same raw data and the same test rows (`size`..end) as QLSTM_copper.ipynb.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Factory import QShallowRegressionLSTM, ShallowRegressionLSTM
from factory_fixed import QShallowRegressionLSTMTrained

HERE = Path(__file__).parent
SEQ_LEN = 10
BATCH = 32
VAL_FRAC = 0.15
MAX_EPOCHS = 25
PATIENCE = 5
HIDDEN = 16
N_QUBITS = 4
TRAIN_FRAC = 0.67  # identical to the original notebook

UNI_FEATURES = ["ret", "ret5", "vol20", "dlogvol"]
MULTI_FEATURES = UNI_FEATURES + ["wti_chg", "eur_ret", "vix_dlog", "vix_lvl"]


def load_frame():
    raw = pd.read_csv(HERE / "dataset_copper_prediction.csv", parse_dates=["Date"])
    al = pd.read_csv(HERE / ".." / ".." / "data" / "aligned_dataset.csv", parse_dates=["date"])
    d = raw[["Date", "Close", "Volume", "Close_lead1"]].merge(
        al[["date", "wti", "eurusd", "vix"]], left_on="Date", right_on="date", how="left"
    )
    assert len(d) == len(raw) and d[["wti", "eurusd", "vix"]].isna().sum().sum() == 0
    # Close_lead1 must be the next row's Close (the original notebook's target)
    assert np.allclose(d.Close_lead1.values[:-1], d.Close.values[1:], atol=1e-5)
    return d.drop(columns="date")


def build_features(d):
    """Stationary, trailing-only features (no centred windows, nothing from the future)."""
    c = np.log(d.Close)
    ret = c.diff()
    f = pd.DataFrame(index=d.index)
    f["ret"] = ret
    f["ret5"] = c.diff(5)
    f["vol20"] = ret.rolling(20).std()
    f["dlogvol"] = np.log1p(d.Volume).diff()
    f["wti_chg"] = d.wti.diff()  # WTI went negative in Apr-2020, so no log
    f["eur_ret"] = np.log(d.eurusd).diff()
    f["vix_dlog"] = np.log(d.vix).diff()
    f["vix_lvl"] = d.vix
    return f


def make_data(variant, seq_len=SEQ_LEN):
    """variant: 'univariate' (copper-only) or 'multivariate' (+ WTI, EUR/USD, VIX)."""
    d = load_frame()
    cols = UNI_FEATURES if variant == "univariate" else MULTI_FEATURES
    feats = build_features(d)[cols]
    n = len(d)
    size = int(n * TRAIN_FRAC)  # same split point as the original notebook
    y_raw = np.log(d.Close_lead1.values / d.Close.values)  # next-day log return

    first = int(np.where(feats.notna().all(axis=1))[0][0])
    train_rows = feats.iloc[first:size]
    lo, hi = train_rows.quantile(0.005), train_rows.quantile(0.995)
    clipped = train_rows.clip(lo, hi, axis=1)
    mu, sd = clipped.mean(), clipped.std()
    fz = ((feats.clip(lo, hi, axis=1) - mu) / sd).values.astype(np.float32)

    all_t = np.arange(first + seq_len - 1, n)
    tr_t, te_t = all_t[all_t < size], all_t[all_t >= size]
    n_val = int(len(tr_t) * VAL_FRAC)
    tr_t, va_t = tr_t[:-n_val], tr_t[-n_val:]

    ylo, yhi = np.quantile(y_raw[tr_t], [0.005, 0.995])
    ymean = float(np.clip(y_raw[tr_t], ylo, yhi).mean())
    ystd = float(np.clip(y_raw[tr_t], ylo, yhi).std())
    yz = ((np.clip(y_raw, ylo, yhi) - ymean) / ystd).astype(np.float32)

    def windows(ts):
        return torch.tensor(np.stack([fz[t - seq_len + 1 : t + 1] for t in ts]))

    return {
        "variant": variant,
        "features": cols,
        "Xtr": windows(tr_t), "ytr": torch.tensor(yz[tr_t]),
        "Xva": windows(va_t), "yva": torch.tensor(yz[va_t]),
        "Xte": windows(te_t),
        "ymean": ymean, "ystd": ystd,
        "drift": float(y_raw[tr_t].mean()),
        "test_t": te_t,
        "test_dates": d.Date.values[te_t],
        "close_t": d.Close.values[te_t],
        "close_next": d.Close_lead1.values[te_t],
        "y_test": y_raw[te_t],
        "n_train": len(tr_t), "n_val": len(va_t), "n_test": len(te_t),
    }


def make_model(name, n_features, n_qubits=N_QUBITS, n_qlayers=1):
    """QLSTM = original Factory model (circuit weights NOT trained, see factory_fixed.py);
    QLSTMT = same model with the circuit weights registered and trained."""
    if name == "LSTM":
        return ShallowRegressionLSTM(num_sensors=n_features, hidden_units=HIDDEN)
    if name == "QLSTM":
        return QShallowRegressionLSTM(num_sensors=n_features, hidden_units=HIDDEN, n_qubits=n_qubits, n_qlayers=n_qlayers)
    if name == "QLSTMT":
        return QShallowRegressionLSTMTrained(num_sensors=n_features, hidden_units=HIDDEN, n_qubits=n_qubits, n_qlayers=n_qlayers)
    raise ValueError(name)


@torch.no_grad()
def predict(model, X, batch=256):
    model.eval()
    return torch.cat([model(X[i : i + batch]) for i in range(0, len(X), batch)]).numpy()


def fit(name, data, lr, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE, model_kwargs=None):
    """Adam + MSE on the standardised return; early stopping on validation loss only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(name, data["Xtr"].shape[-1], **(model_kwargs or {}))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(data["Xtr"], data["ytr"]), batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        tot = 0.0
        for X, y in loader:
            loss = loss_fn(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        tr = tot / len(data["ytr"])
        va = float(loss_fn(torch.tensor(predict(model, data["Xva"])), data["yva"]))
        hist.append((epoch, tr, va))
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {"best_val": best, "epochs": len(hist), "seconds": time.time() - t0,
                   "n_params": n_params, "history": hist}


def rhat_from_model(model, data):
    return predict(model, data["Xte"]) * data["ystd"] + data["ymean"]


def evaluate(rhat, data):
    """All metrics on the same test rows. Price metrics are one-step: Close_t * exp(rhat)."""
    y, c0, c1 = data["y_test"], data["close_t"], data["close_next"]
    price_hat = c0 * np.exp(rhat)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    rmse_naive = rmse(c1 - c0)
    nz = y != 0
    return {
        "price_rmse": rmse(c1 - price_hat),
        "price_mae": float(np.mean(np.abs(c1 - price_hat))),
        "rel_rmse_vs_naive": rmse(c1 - price_hat) / rmse_naive,
        "ret_rmse": rmse(y - rhat),
        "r2_oos_vs_zero": float(1 - np.sum((y - rhat) ** 2) / np.sum(y ** 2)),
        "dir_acc": float(np.mean(np.sign(rhat[nz]) == np.sign(y[nz]))),
        "up_frac": float(np.mean(y[nz] > 0)),
    }
