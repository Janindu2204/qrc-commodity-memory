"""Validation-only tuning for the hourly LSTM / QLSTMT (test rows are never touched).

Usage: python hourly_tune_nn.py MODEL HIDDEN LR QLAYERS SEED  -> appends one line to hourly_tune_nn.log.jsonl
Score = best validation MSE of the standardised residual target on the interleaved validation blocks (lower is better).
"""
import json, sys
import numpy as np, torch
torch.set_num_threads(1)
import hourly_nn as hn
import returns_lib as rl

model, hidden, lr, ql, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rl.HIDDEN = hidden
data, t_te, ym, ys, idx = hn.make("uni")          # tuned on the univariate variant, then reused for multi
kw = {"n_qlayers": ql} if model == "QLSTMT" else None
m, info = rl.fit(model, data, lr=lr, seed=seed, model_kwargs=kw)
rec = dict(model=model, hidden=hidden, lr=lr, qlayers=ql, seed=seed, val=info["best_val"], epochs=info["epochs"],
           params=info["n_params"], val_const=data["val_const"])
open("hourly_tune_nn.log.jsonl", "a").write(json.dumps(rec) + "\n")
print(rec)
