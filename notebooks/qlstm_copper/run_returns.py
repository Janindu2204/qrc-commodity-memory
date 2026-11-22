"""Run the return-target experiments in parallel and append every finished run to a CSV.

  python run_returns.py tune        # lr grid on validation loss, univariate, tuning seed only
  python run_returns.py eval        # N_SEEDS per model x variant at the chosen lr; test metrics
Resumable: runs already in the CSV are skipped. Test rows are only touched by `eval`.
"""
import csv, fcntl, sys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TUNE_CSV, RES_CSV, PRED_DIR = HERE / "returns_tuning.csv", HERE / "returns_results.csv", HERE / "returns_preds"
LR_GRID = [3e-3, 1e-2, 3e-2]
TUNE_SEED = 100
EVAL_SEEDS = list(range(10))
MODELS, VARIANTS = ["LSTM", "QLSTM"], ["univariate", "multivariate"]


def append(path, row):
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)


def done(path, keys):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(map(tuple, df[keys].astype(str).values))


def work(task):
    import torch
    torch.set_num_threads(1)
    import returns_lib as R
    kind, model, variant, seed, lr = task
    data = R.make_data(variant)
    m, info = R.fit(model, data, lr, seed)
    base = {"model": model, "variant": variant, "seed": seed, "lr": lr, "best_val": info["best_val"],
            "epochs": info["epochs"], "seconds": round(info["seconds"], 1), "n_params": info["n_params"]}
    if kind == "tune":
        append(TUNE_CSV, base)
    else:
        rhat = R.rhat_from_model(m, data)
        PRED_DIR.mkdir(exist_ok=True)
        np.save(PRED_DIR / f"{model}_{variant}_{seed}.npy", rhat)
        append(RES_CSV, {**base, **R.evaluate(rhat, data)})
    print("finished", task, round(info["seconds"]), "s", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if stage == "tune":
        have = done(TUNE_CSV, ["model", "variant", "seed", "lr"])
        tasks = [("tune", m, "univariate", TUNE_SEED, lr) for m in MODELS for lr in LR_GRID
                 if (m, "univariate", str(TUNE_SEED), str(lr)) not in have]
    else:
        t = pd.read_csv(TUNE_CSV)
        best_lr = t.loc[t.groupby("model").best_val.idxmin()].set_index("model").lr.to_dict()
        print("chosen lr per model (min validation loss):", best_lr, flush=True)
        have = done(RES_CSV, ["model", "variant", "seed"])
        tasks = [("eval", m, v, s, best_lr[m]) for m in reversed(MODELS) for v in VARIANTS for s in EVAL_SEEDS
                 if (m, v, str(s)) not in have]  # slow QLSTM runs first
    print(len(tasks), "tasks", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap_unordered(work, tasks):
            pass
