"""v5 runs (see HOURLY_DATABENTO_PROTOCOL_v5.json). Same model, inputs, target, training and validation scheme as hourly_databento_scale.py (v3).

Usage: python hourly_databento_v5.py seeds 2 MODEL SEED      # 2-year span, 2026 test (as v3), seeds 0-49
       python hourly_databento_v5.py roll YEAR MODEL SEED    # train = the 2 calendar years before YEAR, test = calendar year YEAR
MODEL: LSTM|GRU|QLSTMT. Output: hourly_databento_preds_v5[_dry]/{seeds2|rollYEAR}_{MODEL}_h{hidden}_s{SEED}.npz (never touches v3 files).
DRY=1: a few hundred rows, 2 epochs.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

torch.set_num_threads(1)
import returns_lib as rl
from hourly_databento_nn import GRUReg
from hourly_databento_nn2 import match_hidden
from hourly_databento_scale import load_ext, split_for_years, arrays, BLOCK, VAL_EVERY, PURGE, LR

DRY = os.environ.get("DRY") == "1"
OUTDIR = Path(os.environ.get("V5_OUT") or Path(__file__).with_name("hourly_databento_preds_v5_dry" if DRY else "hourly_databento_preds_v5"))


def split_roll(df, year):
    lo, hi = pd.Timestamp(f"{year - 2}-01-01", tz="UTC"), pd.Timestamp(f"{year}-01-01", tz="UTC")
    te = df[(df.timestamp_utc >= hi) & (df.timestamp_utc < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))].sort_values("timestamp_utc")
    pre = df[(df.timestamp_utc >= lo) & (df.timestamp_utc < hi) & (df.t < te.t.min() - PURGE)].sort_values("timestamp_utc").reset_index(drop=True)
    is_val = (np.arange(len(pre)) // BLOCK) % VAL_EVERY == VAL_EVERY - 1
    vt = pre.t.values[is_val]
    j = np.searchsorted(vt, pre.t.values)
    near = np.minimum(np.abs(pre.t.values - vt[np.clip(j - 1, 0, len(vt) - 1)]), np.abs(pre.t.values - vt[np.clip(j, 0, len(vt) - 1)])) <= PURGE
    return pre[~is_val & ~near], pre[is_val], te


def main():
    mode, arg, model, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    df = load_ext()
    tr, va, te = split_for_years(df, 2) if mode == "seeds" else split_roll(df, arg)
    tag = "seeds2" if mode == "seeds" else f"roll{arg}"
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
    np.savez(OUTDIR / f"{tag}_{model}_h{hidden}_s{seed}.npz", pred=pred, y=te.yb.values, t=te.timestamp_utc.astype("int64").values,
             seconds=info["seconds"], epochs=info["epochs"], n_params=info["n_params"], n_train=len(tr), best_val=info["best_val"])
    print(f"{tag} {model} h{hidden} s{seed}: ntr={len(tr)} nva={len(va)} nte={len(pred)} epochs={info['epochs']} "
          f"params={info['n_params']} {info['seconds']:.1f}s val={info['best_val']:.4f}", flush=True)


if __name__ == "__main__":
    main()
