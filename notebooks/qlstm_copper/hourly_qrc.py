"""Hourly quantum reservoir computing (QRC) on the 24-bar log-variance target, vs matched controls.

Exact numpy statevector simulation, ideal (noiseless); no variational training. Same rows, fixed origin and
interleaved validation blocks as hourly_nn.py (asserted below). 8 qubits, 8 inputs, angle encoding RY(arctan z),
2 layers of fixed random RX/RY/RZ + CZ ring, read <Z_i> and <Z_iZ_j> (36 features), ridge readout (lambda on val).
10 random reservoirs per spec, forecasts averaged (one ensemble test per cell).
  uni  inputs: copper [ret, lv24, lv120, lv480] at t and t-1
  multi inputs: copper [ret, lv24, lv120, lv480] at t + first 4 PCs of the 12 exogenous log-variances (PCA on train rows)
Controls: no-entanglement reservoir (CZ removed), classical random tanh features (36-d).
Each feature set is added to the HAR features (nested: Clark-West vs HAR_fixed) and also used alone.
"""
import itertools
import numpy as np, pandas as pd
import hourly_baselines as hb
import hourly_multivariate as hm
import hourly_nn as hn

N_RES = 10
LAMS = [1e-2, 1e-1, 1, 10, 100, 1000]
OUT = hn.Path(__file__).with_name("hourly_qrc_results.csv")


def ry_product(theta):
    N, n = theta.shape
    st = np.ones((N, 1), complex)
    for q in range(n):
        v = np.stack([np.cos(theta[:, q] / 2), np.sin(theta[:, q] / 2)], 1)
        st = (st[:, :, None] * v[:, None, :]).reshape(N, -1)
    return st


def apply_1q(st, U, q, n):
    N = st.shape[0]
    s = st.reshape(N * 2 ** q, 2, 2 ** (n - q - 1))
    return np.einsum("ab,ibk->iak", U, s).reshape(N, -1)


rx = lambda a: np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
ry = lambda a: np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], complex)
rz = lambda a: np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


def cz_ring_signs(n):
    idx = np.arange(2 ** n)
    bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
    sgn = np.ones(2 ** n)
    for q in range(n):
        sgn *= 1 - 2 * (bits[q] & bits[(q + 1) % n])
    return sgn


def z_features(st, n):
    p = np.abs(st) ** 2
    idx = np.arange(2 ** n)
    z = [1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)]
    return np.stack([p @ zq for zq in z] + [p @ (z[i] * z[j]) for i, j in itertools.combinations(range(n), 2)], 1)


def reservoir_features(inp, n, seed, entangle=True, layers=2):
    rng = np.random.default_rng(seed)
    st = ry_product(np.arctan(inp))
    sgn = cz_ring_signs(n)
    for _ in range(layers):
        for q in range(n):
            for g in (rx, ry, rz):
                st = apply_1q(st, g(rng.uniform(0, 2 * np.pi)), q, n)
        if entangle:
            st = st * sgn
    return z_features(st, n)


def ridge(Ftr, ytr, Fva, yva, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A, B, T = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd
    ym, best, bw = ytr.mean(), np.inf, None
    for lam in LAMS:
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym))
        e = np.mean((yva - (B @ w + ym)) ** 2)
        if e < best:
            best, bw = e, w
    return T @ bw + ym


def rows():
    """Replicates hourly_nn.make row selection (stride 1 for ridge/QRC: cheap, no early stopping needed)."""
    d, r = hb.load(); X, y, R, yr, hod = hb.build(r)
    ER, EV = hm.exog_features(d.index)
    F = pd.concat([pd.DataFrame({"ret": r.fillna(0.0), "lv24": X.lv24, "lv120": X.lv120, "lv480": X.lv480}, index=d.index),
                   ER[hm.TICKERS].add_prefix("er_"), EV.add_prefix("ev_")], axis=1)
    ok = F.notna().all(axis=1).values & y.notna().values
    pos = np.arange(len(d))
    full = pd.Series(ok).rolling(hn.SEQ).min().fillna(0).astype(bool).values
    t_tr = pos[full & (d.index < hn.TRAIN_END - pd.Timedelta(hours=30))]
    t_te = pos[full & (d.index >= hb.TEST_START)]
    days = np.asarray((d.index[t_tr] - d.index[t_tr[0]]).days); blk = days // 10; is_va = blk % 5 == 4
    near = np.zeros(len(t_tr), bool)
    for b in np.unique(blk[is_va]):
        lo = d.index[t_tr[0]] + pd.Timedelta(days=int(b) * 10) - pd.Timedelta(hours=30)
        hi = d.index[t_tr[0]] + pd.Timedelta(days=int(b) * 10 + 10) + pd.Timedelta(hours=30)
        near |= (d.index[t_tr] >= lo) & (d.index[t_tr] < hi)
    return d, X, y, F, t_tr[~near], t_tr[is_va], t_te


def cw_hac(y, pb, pc):
    return hb.hac_t(pd.Series((y - pb) ** 2 - (y - pc) ** 2 + (pb - pc) ** 2), 48)


def main():
    d, X, y, F, tr, va, te = rows()
    assert len(te) == 7667, len(te)
    yv = y.values; ytr, yva, yte = yv[tr], yv[va], yv[te]
    cop = F[["ret", "lv24", "lv120", "lv480"]].values
    mu, sd = cop[tr].mean(0), cop[tr].std(0) + 1e-9
    cz = (cop - mu) / sd
    ev = F[[c for c in F.columns if c.startswith("ev_")]].values
    emu, esd = ev[tr].mean(0), ev[tr].std(0) + 1e-9
    evz = (ev - emu) / esd
    U, S, Vt = np.linalg.svd(evz[tr], full_matrices=False)
    pcs = evz @ Vt[:4].T / S[:4].std() * 1.0
    pcs = pcs / pcs[tr].std(0)
    inputs = {"uni": lambda t: np.hstack([cz[t], cz[t - 1]]), "multi": lambda t: np.hstack([cz[t], pcs[t]])}
    Hc = X[["lv24", "lv120", "lv480"]].values
    base_te = ridge(Hc[tr], ytr, Hc[va], yva, Hc[te])          # HAR_fixed (ridge, lambda by val; matches HAR to ~1e-3)
    shock = (d.index >= hb.SHOCK[0] - pd.Timedelta(hours=hb.H + 1)) & (d.index <= hb.SHOCK[1])
    wins = {"all": np.ones(len(te), bool), "ex_shock": ~shock[te]}
    out = []
    for variant, inp in inputs.items():
        ens = {k: [] for k in ["QRC", "QRC_noent", "RandFeat", "HAR+QRC", "HAR+QRC_noent", "HAR+RandFeat"]}
        for seed in range(N_RES):
            Fs = {}
            for tag, ent in [("QRC", True), ("QRC_noent", False)]:
                Fs[tag] = [reservoir_features(inp(idx), 8, seed, ent) for idx in (tr, va, te)]
            rng = np.random.default_rng(1000 + seed)
            W, b = rng.normal(size=(8, 36)) / np.sqrt(8), rng.uniform(0, 2 * np.pi, 36)
            Fs["RandFeat"] = [np.tanh(np.arctan(inp(idx)) @ W + b) for idx in (tr, va, te)]  # same arctan squashing as the QRC inputs
            for tag, f in Fs.items():
                ens[tag].append(ridge(f[0], ytr, f[1], yva, f[2]))
                ens["HAR+" + tag].append(ridge(np.hstack([Hc[tr], f[0]]), ytr, np.hstack([Hc[va], f[1]]), yva, np.hstack([Hc[te], f[2]])))
            print(f"{variant} reservoir {seed} done", flush=True)
        # QLIKE bias correction from HAR train residuals
        s2 = float(np.mean((ytr - ridge(Hc[tr], ytr, Hc[va], yva, Hc[tr])) ** 2))
        for k, v in ens.items():
            p = np.mean(v, 0)
            for w, wm in wins.items():
                yw, pw, bw = yte[wm], p[wm], base_te[wm]
                mse = np.mean((yw - pw) ** 2) / np.mean((yw - bw) ** 2)
                ql = hb.qlike(np.exp(pw + .5 * s2), np.exp(yw)) / hb.qlike(np.exp(bw + .5 * s2), np.exp(yw))
                seedmse = [np.mean((yw - q[wm]) ** 2) / np.mean((yw - bw) ** 2) for q in v]
                out.append(dict(variant=variant, window=w, model=k, n=int(wm.sum()), MSE_rel_HARfixed=mse, QLIKE_rel_HARfixed=ql,
                                CW_t_vs_HAR=cw_hac(yw, bw, pw) if k.startswith("HAR+") else np.nan,
                                DM_t_vs_HAR=hb.hac_t(pd.Series((yw - bw) ** 2 - (yw - pw) ** 2), 48) if not k.startswith("HAR+") else np.nan,
                                reservoir_MSE_rel_min=min(seedmse), reservoir_MSE_rel_max=max(seedmse)))
    res = pd.DataFrame(out); res.to_csv(OUT, index=False)
    pd.set_option("display.width", 230)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
