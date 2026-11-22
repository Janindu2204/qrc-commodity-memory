"""Validation-only tuning of the hourly QRC / ESN settings (test rows never used).

Score = validation MSE (log-variance units) of the HAR+features ridge readout on the interleaved validation blocks,
averaged over 3 reservoir seeds (seeds 100-102, disjoint from the final-run seeds 0-9). Note ridge() also picks its
lambda on those same blocks, so absolute scores are slightly optimistic; the optimism is the same for every config.
Grids: angle QRC layers x input scale; recurrent Ising tau x J scale; ESN spectral radius x leak.
Writes hourly_tune_qrc.csv and hourly_tune_qrc_best.json.
"""
import json
import numpy as np, pandas as pd
import hourly_qrc as hq
import hourly_qrc_mem as mem

SEEDS = (100, 101, 102)


def prep():
    d, Xf, y, F, tr, va, te = hq.rows()
    yv = y.values; ytr, yva = yv[tr], yv[va]
    cop = F[["ret", "lv24", "lv120", "lv480"]].values
    cop = np.where(np.isfinite(cop), cop, 0.0)
    mu, sd = cop[tr].mean(0), cop[tr].std(0) + 1e-9
    cz = (cop - mu) / sd
    Hc = Xf[["lv24", "lv120", "lv480"]].values
    return cz, Hc, ytr, yva, tr, va


def val_mse(f, Hc, ytr, yva, tr, va):
    # fit on train rows, score on val rows; readout lambda picked on val inside ridge() (same optimism for all configs)
    p = hq.ridge(np.hstack([Hc[tr], f[tr]]), ytr, np.hstack([Hc[va], f[va]]), yva, np.hstack([Hc[va], f[va]]))
    return float(np.mean((yva - p) ** 2))


def esn(inp, seed, radius, leak, units=28):
    rng = np.random.default_rng(5000 + seed)
    W = rng.normal(size=(units, units)); W *= radius / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(size=(units, inp.shape[1])) / np.sqrt(inp.shape[1])
    u = np.arctan(inp); s = np.zeros(units); out = np.empty((len(inp), units))
    for t in range(len(inp)):
        s = (1 - leak) * s + leak * np.tanh(Win @ u[t] + W @ s); out[t] = s
    return out


def main():
    cz, Hc, ytr, yva, tr, va = prep()
    base = hq.ridge(Hc[tr], ytr, Hc[va], yva, Hc[va])
    print("HAR val MSE", float(np.mean((yva - base) ** 2)))
    inp_all = np.hstack([cz, np.vstack([np.zeros((1, 4)), cz[:-1]])])
    rows = [dict(family="HAR", cfg="HAR", val_mse=float(np.mean((yva - base) ** 2)))]
    # angle QRC
    for layers in (1, 2, 3):
        for scale in (0.5, 1.0, 2.0):
            v = []
            for s in SEEDS:
                f = np.zeros((len(cz), 36))
                for idx in (tr, va):
                    f[idx] = hq.reservoir_features(inp_all[idx] * scale, 8, s, True, layers)
                v.append(val_mse(f, Hc, ytr, yva, tr, va))
            rows.append(dict(family="angle_QRC", cfg=f"layers={layers},scale={scale}", layers=layers, scale=scale, val_mse=float(np.mean(v))))
            print(rows[-1], flush=True)
    # recurrent Ising
    for tau in (0.5, 1.0, 2.0):
        for js in (0.5, 1.0, 2.0):
            mem.TAU, mem.JSCALE = tau, js
            v = []
            for s in SEEDS:
                f = mem.run_quantum(cz, mem.unitary(s), True)
                v.append(val_mse(f, Hc, ytr, yva, tr, va))
            rows.append(dict(family="Ising_mem", cfg=f"tau={tau},J={js}", tau=tau, jscale=js, val_mse=float(np.mean(v))))
            print(rows[-1], flush=True)
    # ESN
    for radius in (0.5, 0.9, 1.2):
        for leak in (0.2, 0.5, 1.0):
            v = [val_mse(esn(cz, s, radius, leak), Hc, ytr, yva, tr, va) for s in SEEDS]
            rows.append(dict(family="ESN", cfg=f"radius={radius},leak={leak}", radius=radius, leak=leak, val_mse=float(np.mean(v))))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows); res.to_csv("hourly_tune_qrc.csv", index=False)
    best = {}
    for fam, g in res[res.family != "HAR"].groupby("family"):
        b = g.sort_values("val_mse").iloc[0]
        best[fam] = {k: (None if pd.isna(v) else v) for k, v in b.drop("family").to_dict().items()}
    json.dump(best, open("hourly_tune_qrc_best.json", "w"), indent=1, default=float)
    print(json.dumps(best, indent=1, default=float))


if __name__ == "__main__":
    main()
