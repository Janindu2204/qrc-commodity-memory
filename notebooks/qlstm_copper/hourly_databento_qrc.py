"""Quantum reservoir (exact 8-qubit statevector, ideal) vs matched classical controls on next-hour ln RV, at each training size.
(see HOURLY_DATABENTO_PROTOCOL_v3.json). Circuit code is copied from hourly_qrc.py (8 qubits, angle encoding RY(arctan z),
2 layers of fixed random RX/RY/RZ + CZ ring, readout <Z_i> and <Z_iZ_j> = 36 features, ridge readout with lambda on validation).

Usage: python hourly_databento_qrc.py [YEARS ...]   (default 1 2 4 8 16)    DRY=1 for a few hundred rows, 2 reservoirs.
Models per size (10 random reservoirs each, forecasts averaged):
  QRC (entangled), QRC_noent (CZ removed), RandFeat (36 random tanh features of the same 8 inputs), ESN (36-unit leaky tanh
  reservoir over all 10 steps x 5 channels). Each used alone and as a nested residual add-on to the HAR+hour-of-day benchmark.
Scored: MSE on ln RV as ratio to HAR+hour-of-day (OLS on train), DM (HAC lag 24) vs that benchmark.
"""
import itertools, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

from hourly_databento_scale import load_ext, split_for_years, arrays, CH

HERE = Path(__file__).parent
DRY = os.environ.get("DRY") == "1"
N_RES = 2 if DRY else 10
LAMS = [1e-2, 1e-1, 1, 10, 100, 1000]
CHUNK = 6000
OUT = HERE / ("hourly_databento_qrc_results_dry.csv" if DRY else "hourly_databento_qrc_results.csv")

rx = lambda a: np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
ry = lambda a: np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], complex)
rz = lambda a: np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


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
    gates = [[g(rng.uniform(0, 2 * np.pi)) for q in range(n) for g in (rx, ry, rz)] for _ in range(layers)]   # same draw order as hourly_qrc.py
    sgn = cz_ring_signs(n)
    out = []
    for i in range(0, len(inp), CHUNK):
        st = ry_product(np.arctan(inp[i:i + CHUNK]))
        for L in range(layers):
            k = 0
            for q in range(n):
                for _ in range(3):
                    st = apply_1q(st, gates[L][k], q, n); k += 1
            if entangle:
                st = st * sgn
        out.append(z_features(st, n))
    return np.vstack(out)


def esn_features(seq, seed, units=36, leak=0.5, rho=0.9, scale=0.5):
    rng = np.random.default_rng(2000 + seed)
    Win = rng.uniform(-scale, scale, (seq.shape[-1], units))
    W = rng.normal(size=(units, units)); W *= rho / max(abs(np.linalg.eigvals(W)))
    s = np.zeros((len(seq), units))
    for t in range(seq.shape[1]):
        s = (1 - leak) * s + leak * np.tanh(seq[:, t] @ Win + s @ W)
    return s


def ridge_fit(Ftr, ytr, Fva, yva, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A, B, T = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd
    ym, best, bw = ytr.mean(), np.inf, None
    for lam in LAMS:
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym))
        e = np.mean((yva - (B @ w + ym)) ** 2)
        if e < best:
            best, bw = e, w
    return T @ bw + ym


def har_hod(d):
    hod = pd.get_dummies(d.hod_target).reindex(columns=range(24), fill_value=0).astype(float).values[:, 1:]
    return np.column_stack([np.ones(len(d)), d.har_l1, d[[f"a_lrv1m_{l}" for l in range(1, 11)]].mean(1), d.har_l24, d.har_l120, hod])


def dm_hac(e_a, e_b, lag=24):
    d = e_a - e_b; n = len(d); dc = d - d.mean(); v = dc @ dc / n
    for h in range(1, lag + 1):
        v += 2 * (1 - h / (lag + 1)) * (dc[h:] @ dc[:-h] / n)
    t = d.mean() / np.sqrt(v / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def run(years, df):
    tr, va, te = split_for_years(df, years)
    if DRY:
        tr, va, te = tr.head(600), va.head(150), te.head(150)
    ytr, yva, yte = tr.yb.values, va.yb.values, te.yb.values
    Htr, Hva, Hte = har_hod(tr), har_hod(va), har_hod(te)
    beta = np.linalg.lstsq(Htr, ytr, rcond=None)[0]
    base = {k: H @ beta for k, H in [("tr", Htr), ("va", Hva), ("te", Hte)]}
    e_base = (yte - base["te"]) ** 2
    # QRC / RandFeat inputs: 8 numbers = [ret, lrv1m, lvol, hs, hc] at t and [ret, lrv1m, lvol] at t-1, standardised on train
    def q8(d):
        a = lambda c, l: d[f"a_{c}_{l}"].values
        return np.column_stack([a("ret", 10), a("lrv1m", 10), a("lvol", 10), a("hs", 10), a("hc", 10), a("ret", 9), a("lrv1m", 9), a("lvol", 9)])
    mu, sd = q8(tr).mean(0), q8(tr).std(0) + 1e-9
    Q = {k: (q8(d) - mu) / sd for k, d in [("tr", tr), ("va", va), ("te", te)]}
    S = {k: arrays(d) for k, d in [("tr", tr), ("va", va), ("te", te)]}
    smu, ssd = S["tr"].reshape(-1, 5).mean(0), S["tr"].reshape(-1, 5).std(0) + 1e-9
    S = {k: (v - smu) / ssd for k, v in S.items()}
    preds = {f"{m}{s}": [] for m in ["QRC", "QRC_noent", "RandFeat", "ESN"] for s in ["", "+HAR"]}
    t0 = time.time()
    for seed in range(N_RES):
        rng = np.random.default_rng(1000 + seed)
        W, b = rng.normal(size=(8, 36)) / np.sqrt(8), rng.uniform(0, 2 * np.pi, 36)
        F = {"QRC": {k: reservoir_features(Q[k], 8, seed, True) for k in Q},
             "QRC_noent": {k: reservoir_features(Q[k], 8, seed, False) for k in Q},
             "RandFeat": {k: np.tanh(np.arctan(Q[k]) @ W + b) for k in Q},
             "ESN": {k: esn_features(S[k], seed) for k in S}}
        for m, f in F.items():
            preds[m].append(ridge_fit(f["tr"], ytr, f["va"], yva, f["te"]))
            preds[m + "+HAR"].append(base["te"] + ridge_fit(f["tr"], ytr - base["tr"], f["va"], yva - base["va"], f["te"]))
    secs = time.time() - t0
    rows = [{"years": years, "model": "HAR+hod", "n_train": len(tr), "n_test": len(te), "mse_ratio": 1.0, "dm_t": np.nan, "p": np.nan, "res_min": np.nan, "res_max": np.nan, "seconds": secs}]
    for k, v in preds.items():
        p = np.mean(v, 0); e = (yte - p) ** 2
        t, pv = dm_hac(e, e_base)
        rs = [np.mean((yte - q) ** 2) / e_base.mean() for q in v]
        rows.append({"years": years, "model": k, "n_train": len(tr), "n_test": len(te), "mse_ratio": e.mean() / e_base.mean(), "dm_t": t, "p": pv,
                     "res_min": min(rs), "res_max": max(rs), "seconds": secs})
    r = pd.DataFrame(rows)
    print(f"\n== {years} years: train {len(tr)}, val {len(va)}, test {len(te)}, {secs:.0f}s for {N_RES} reservoirs ==", flush=True)
    print(r.drop(columns=["years", "n_train", "n_test", "seconds"]).round(4).to_string(index=False), flush=True)
    return r


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8, 16]
    df = load_ext()
    allr = pd.concat([run(y, df) for y in sizes])
    allr.to_csv(OUT, index=False)
    print("written", OUT)
