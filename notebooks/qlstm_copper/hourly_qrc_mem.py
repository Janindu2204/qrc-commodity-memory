"""Recurrent QRC with memory qubits (Ising reservoir) on the hourly 24-bar log-variance target.

My implementation in the spirit of Li et al. (arXiv 2505.13933, PRR 2026); only the abstract was read, so this is NOT a
reproduction: sizes, field/coupling scales, evolution time and readout are my choices (tau, J scale NOT tuned).
7 qubits = 4 input + 3 memory, exact density-matrix simulation, ideal (no noise, no shots).
Each bar: input qubits are re-prepared from the current features, memory keeps its reduced state from the previous
bar, U = exp(-i H tau) with H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i (fully connected transverse-field Ising), read <Z_i>, <Z_i Z_j> (28
features), then input qubits are traced out. Causal: the state at bar t depends only on inputs up to t.
Specs per random reservoir (10 seeds), each alone and added to HAR (nested -> Clark-West):
  IsingMem   : full model
  IsingNoMem : memory qubits reset every bar (static feature map; tests whether recurrence matters)
  NoCoupling : J = 0 (no entanglement between input and memory; tests whether coupling matters)
  ESN        : classical echo state network, 28 tanh units, spectral radius 0.9 (classical recurrent control)
Inputs: uni = copper [ret, lv24, lv120, lv480]; multi = those + first 4 PCs of the 12 driver log-variances
(two features per input qubit via RY then RZ). Rows, fixed origin, val blocks, scoring: as hourly_qrc.py.
"""
import itertools
import numpy as np, pandas as pd
from scipy.linalg import expm
import hourly_baselines as hb
import hourly_qrc as hq

NIN, NMEM = 4, 3
NQ = NIN + NMEM
DIM = 2 ** NQ
TAU, JSCALE, HFIELD = 1.0, 1.0, 1.0
N_RES = 10
OUT = hq.OUT.with_name("hourly_qrc_mem_results.csv")

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Zm = np.diag([1.0, -1.0])


def op(single, q):
    m = np.array([[1.0 + 0j]])
    for k in range(NQ):
        m = np.kron(m, single if k == q else I2)
    return m


ZOPS = [np.real(np.diag(op(Zm, q))) for q in range(NQ)]
ZFEATS = ZOPS + [ZOPS[i] * ZOPS[j] for i, j in itertools.combinations(range(NQ), 2)]
ZMAT = np.stack(ZFEATS, 1)  # (DIM, 28)


def unitary(seed, coupled=True):
    rng = np.random.default_rng(seed)
    diag = np.zeros(DIM)
    if coupled:
        for i, j in itertools.combinations(range(NQ), 2):
            diag += rng.uniform(-JSCALE, JSCALE) * ZOPS[i] * ZOPS[j]
    else:
        for _ in range(NQ * (NQ - 1) // 2):
            rng.uniform()
    H = np.diag(diag).astype(complex)
    for q in range(NQ):
        H += HFIELD * rng.uniform(0.5, 1.5) * op(X, q)
    return expm(-1j * TAU * H)


def input_states(inp):
    """inp (N, 4) or (N, 8) -> pure input-qubit states (N, 2^NIN), RY(arctan a) then RZ(arctan b)."""
    N = len(inp)
    a = np.arctan(inp[:, :NIN])
    b = np.arctan(inp[:, NIN:2 * NIN]) if inp.shape[1] > NIN else np.zeros((N, NIN))
    st = np.ones((N, 1), complex)
    for q in range(NIN):
        v = np.stack([np.cos(a[:, q] / 2) * np.exp(-1j * b[:, q] / 2), np.sin(a[:, q] / 2) * np.exp(1j * b[:, q] / 2)], 1)
        st = (st[:, :, None] * v[:, None, :]).reshape(N, -1)
    return st


def run_quantum(inp, U, memory=True):
    psi = input_states(inp)
    rho_m = np.zeros((2 ** NMEM, 2 ** NMEM), complex); rho_m[0, 0] = 1
    Ud = U.conj().T
    F = np.empty((len(inp), ZMAT.shape[1]))
    for t in range(len(inp)):
        rho = np.kron(np.outer(psi[t], psi[t].conj()), rho_m)
        rho = U @ rho @ Ud
        F[t] = np.real(np.diag(rho)) @ ZMAT
        if memory:
            rho_m = np.einsum("iaib->ab", rho.reshape(2 ** NIN, 2 ** NMEM, 2 ** NIN, 2 ** NMEM))
            rho_m = rho_m / np.real(np.trace(rho_m))
    return F


def run_esn(inp, seed, units=28, leak=0.5):
    rng = np.random.default_rng(5000 + seed)
    W = rng.normal(size=(units, units)); W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
    Win = rng.normal(size=(units, inp.shape[1])) / np.sqrt(inp.shape[1])
    u = np.arctan(inp)
    s = np.zeros(units); F = np.empty((len(inp), units))
    for t in range(len(inp)):
        s = (1 - leak) * s + leak * np.tanh(Win @ u[t] + W @ s)
        F[t] = s
    return F


def main():
    d, Xf, y, F, tr, va, te = hq.rows()
    assert len(te) == 7667
    yv = y.values; ytr, yva, yte = yv[tr], yv[va], yv[te]
    cop = F[["ret", "lv24", "lv120", "lv480"]].values
    cop = np.where(np.isfinite(cop), cop, 0.0)
    mu, sd = cop[tr].mean(0), cop[tr].std(0) + 1e-9
    cz = (cop - mu) / sd
    ev = F[[c for c in F.columns if c.startswith("ev_")]].values
    ev = np.where(np.isfinite(ev), ev, np.nanmean(ev[tr], 0))
    evz = (ev - ev[tr].mean(0)) / (ev[tr].std(0) + 1e-9)
    _, S, Vt = np.linalg.svd(evz[tr], full_matrices=False)
    pcs = evz @ Vt[:4].T; pcs = pcs / pcs[tr].std(0)
    inputs = {"uni": cz, "multi": np.hstack([cz, pcs])}
    Hc = Xf[["lv24", "lv120", "lv480"]].values
    base_te = hq.ridge(Hc[tr], ytr, Hc[va], yva, Hc[te])
    s2 = float(np.mean((ytr - hq.ridge(Hc[tr], ytr, Hc[va], yva, Hc[tr])) ** 2))
    shock = (d.index >= hb.SHOCK[0] - pd.Timedelta(hours=hb.H + 1)) & (d.index <= hb.SHOCK[1])
    wins = {"all": np.ones(len(te), bool), "ex_shock": ~shock[te]}
    out = []
    for variant, inp in inputs.items():
        ens = {k: [] for k in ["IsingMem", "IsingNoMem", "NoCoupling", "ESN"]}
        ens.update({"HAR+" + k: [] for k in list(ens)})
        for seed in range(N_RES):
            feats = {"IsingMem": run_quantum(inp, unitary(seed), True),
                     "IsingNoMem": run_quantum(inp, unitary(seed), False),
                     "NoCoupling": run_quantum(inp, unitary(seed, coupled=False), True),
                     "ESN": run_esn(inp, seed)}
            for k, f in feats.items():
                ens[k].append(hq.ridge(f[tr], ytr, f[va], yva, f[te]))
                ens["HAR+" + k].append(hq.ridge(np.hstack([Hc[tr], f[tr]]), ytr, np.hstack([Hc[va], f[va]]), yva, np.hstack([Hc[te], f[te]])))
            print(f"{variant} reservoir {seed} done", flush=True)
        for k, v in ens.items():
            p = np.mean(v, 0)
            for w, wm in wins.items():
                yw, pw, bw = yte[wm], p[wm], base_te[wm]
                seedmse = [np.mean((yw - q[wm]) ** 2) / np.mean((yw - bw) ** 2) for q in v]
                out.append(dict(variant=variant, window=w, model=k, n=int(wm.sum()),
                                MSE_rel_HARfixed=np.mean((yw - pw) ** 2) / np.mean((yw - bw) ** 2),
                                QLIKE_rel_HARfixed=hb.qlike(np.exp(pw + .5 * s2), np.exp(yw)) / hb.qlike(np.exp(bw + .5 * s2), np.exp(yw)),
                                CW_t_vs_HAR=hq.cw_hac(yw, bw, pw) if k.startswith("HAR+") else np.nan,
                                DM_t_vs_HAR=hb.hac_t(pd.Series((yw - bw) ** 2 - (yw - pw) ** 2), 48) if not k.startswith("HAR+") else np.nan,
                                res_MSE_rel_min=min(seedmse), res_MSE_rel_max=max(seedmse)))
    res = pd.DataFrame(out); res.to_csv(OUT, index=False)
    pd.set_option("display.width", 230)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
