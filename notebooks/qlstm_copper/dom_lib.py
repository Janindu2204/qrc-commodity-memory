"""Reservoir of Domingo, Grande, Carlo, Borondo, Borondo (arXiv:2401.03347; Eng. Appl. Artif. Intell. 182, 2026) for long_prereg_dom.json.
N = 7 qubits, qubit 0 carries the input u in [0,1] as sqrt(1-u)|0> + sqrt(u)|1>; the other 6 qubits keep the memory:
rho(t) = |u_t><u_t| (x) Tr_0[rho(t-1)], then a fixed random circuit U of NG gates from a gate family. Features: <X_i>, <Z_i> of all 7 qubits
after the first half of the gates and after all gates (Nv = 2) = 28 features. Exact (density matrix), no shots, no noise.
The 6-qubit memory state tau is propagated with the two Kraus operators of the channel, which is the same map as the full 7-qubit simulation
(checked by selftest()). Family G3 = {CNOT, H, T}; family G1 = {CNOT, H, X}. The same seed gives the same gate layout in both families (only T <-> X differs).
"""
import numpy as np

N = 7; D = 2 ** N; DM = D // 2; NG = 150; NFEAT = 4 * N
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); _X = np.array([[0, 1], [1, 0]], complex); _Z = np.diag([1.0, -1.0]).astype(complex)
_T = np.diag([1, np.exp(1j * np.pi / 4)]); _I = np.eye(2, dtype=complex)


def _one(g, q):
    m = np.array([[1.0 + 0j]])
    for k in range(N): m = np.kron(m, g if k == q else _I)
    return m


def _cnot(c, t):
    P0, P1 = np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)
    return _one(P0, c) + _one(P1, c) @ _one(_X, t)


def circuit(seed, family):
    """returns (Ua, Ub): unitary of the first NG/2 gates and of the last NG/2 gates"""
    rng = np.random.default_rng(9000 + seed); third = {"G3": _T, "G1": _X}[family]; U = [np.eye(D, dtype=complex), np.eye(D, dtype=complex)]
    for k in range(NG):
        kind = rng.integers(3); q = rng.integers(N); r = (q + 1 + rng.integers(N - 1)) % N        # same draws for both families
        g = _cnot(q, r) if kind == 0 else _one(_H if kind == 1 else third, q)
        h = 0 if k < NG // 2 else 1; U[h] = g @ U[h]
    return U[0], U[1]


def _prep(seed, family):
    Ua, Ub = circuit(seed, family); W = Ub @ Ua
    paulis = [_one(_X, q) for q in range(N)] + [_one(_Z, q) for q in range(N)]
    G = np.empty((3, NFEAT, 2 * DM * DM))
    for h, V in enumerate((Ua, W)):
        for j, P in enumerate(paulis):
            A = V.conj().T @ P @ V; blk = (A[:DM, :DM], A[DM:, DM:], A[:DM, DM:] + A[DM:, :DM])   # coefficients (1-u), u, sqrt(u(1-u))
            for c in range(3):
                B = blk[c].T; G[c, h * 2 * N + j] = np.concatenate([B.real.ravel(), -B.imag.ravel()])   # tr(A tau) = sum(Re B Re tau - Im B Im tau)
    K = np.stack([W[:DM, :DM], W[:DM, DM:], W[DM:, :DM], W[DM:, DM:]])                        # K_k(u) = sqrt(1-u) W[k,0] + sqrt(u) W[k,1]
    return G, K


def features(u, seed, family):
    """u: 1-D array in [0,1] (no NaN). Returns (len(u), 28) features; row t uses inputs up to and including u[t]."""
    G, K = _prep(seed, family); tau = np.eye(DM, dtype=complex) / DM; out = np.empty((len(u), NFEAT)); v = np.empty(2 * DM * DM)
    a = np.sqrt(1 - u); b = np.sqrt(u); c = a * b; GG = np.ascontiguousarray(G.reshape(3 * NFEAT, -1))
    for t in range(len(u)):
        v[:DM * DM] = tau.real.ravel(); v[DM * DM:] = tau.imag.ravel()
        g = GG @ v; out[t] = (1 - u[t]) * g[:NFEAT] + u[t] * g[NFEAT:2 * NFEAT] + c[t] * g[2 * NFEAT:]
        K0 = a[t] * K[0] + b[t] * K[1]; K1 = a[t] * K[2] + b[t] * K[3]
        tau = K0 @ tau @ K0.conj().T + K1 @ tau @ K1.conj().T
        tau = (tau + tau.conj().T) / (2 * np.trace(tau).real)
    return out


def features_bruteforce(u, seed, family):
    """direct 7-qubit density-matrix simulation; used only by selftest()"""
    Ua, Ub = circuit(seed, family); paulis = [_one(_X, q) for q in range(N)] + [_one(_Z, q) for q in range(N)]
    tau = np.eye(DM, dtype=complex) / DM; out = np.empty((len(u), NFEAT))
    for t in range(len(u)):
        psi = np.array([np.sqrt(1 - u[t]), np.sqrt(u[t])]); rho = np.kron(np.outer(psi, psi), tau)
        r1 = Ua @ rho @ Ua.conj().T; r2 = Ub @ r1 @ Ub.conj().T
        out[t] = [np.trace(P @ r).real for r in (r1, r2) for P in paulis]
        tau = np.einsum("iaib->ab", r2.reshape(2, DM, 2, DM))
    return out


def esn1(z, seed, units=NFEAT, leak=0.5, rho=0.9, scale=0.5):
    """univariate leaky echo state network run along the series; z standardised input (no NaN). Row t uses inputs up to z[t]."""
    rng = np.random.default_rng(7000 + seed); Win = rng.uniform(-scale, scale, units); W = rng.normal(size=(units, units)); W *= rho / max(abs(np.linalg.eigvals(W)))
    s = np.zeros(units); out = np.empty((len(z), units))
    for t in range(len(z)):
        s = (1 - leak) * s + leak * np.tanh(Win * z[t] + W @ s); out[t] = s
    return out


def hold(x):
    """replace NaN by the last finite value (first values by the first finite one)"""
    x = np.asarray(x, float).copy(); ok = np.isfinite(x); idx = np.where(ok, np.arange(len(x)), 0); np.maximum.accumulate(idx, out=idx)
    x = x[idx]; x[~np.isfinite(x)] = x[np.isfinite(x)][0]; return x


def selftest():
    rng = np.random.default_rng(1); u = rng.uniform(0, 1, 40)
    for fam in ("G3", "G1"):
        d = np.abs(features(u, 3, fam) - features_bruteforce(u, 3, fam)).max(); assert d < 1e-10, (fam, d)
        Ua, Ub = circuit(3, fam); assert np.allclose(Ua @ Ua.conj().T, np.eye(D)) and np.allclose(Ub @ Ub.conj().T, np.eye(D))
    f3, f1 = features(u, 3, "G3"), features(u, 3, "G1"); assert np.abs(f3 - f1).max() > 1e-3 and np.abs(f3).max() <= 1 + 1e-9
    f = features(np.r_[u, u], 3, "G3"); g = features(np.r_[rng.uniform(0, 1, 40), u], 3, "G3")
    assert np.allclose(hold([np.nan, 1, np.nan, 2]), [1, 1, 1, 2])
    return float(np.abs(f[-1] - g[-1]).max()), float(f3.std(0).min()), float(f3.std(0).max())


if __name__ == "__main__":
    import time
    print("selftest ok; memory difference after 40 shared inputs, min/max feature sd:", selftest())
    u = np.random.default_rng(0).uniform(0, 1, 2000); t0 = time.time(); features(u, 0, "G3"); print(f"{(time.time() - t0) / 2000 * 1e3:.3f} ms per step")
