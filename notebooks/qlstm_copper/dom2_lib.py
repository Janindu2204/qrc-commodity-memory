"""Additions for long_prereg_dom2.json (dom_lib.py is frozen by long_prereg_dom.sha256 and is not edited).
ising_unitaries: the transverse-field Ising reservoir of Domingo et al. (their Eq. 8): H = sum_{i<j} J_ij Z_i Z_j + h sum_i X_i, J_ij ~ U(-Js/2, Js/2), h = 0.1 Js,
evolution time T = 10 (Js = 1), exact matrix exponential; the intermediate readout is at T/2. features_u: the dom_lib propagation for any pair of unitaries.
"""
import numpy as np
from scipy.linalg import expm
import dom_lib as dl

N, D, DM, NFEAT = dl.N, dl.D, dl.DM, dl.NFEAT


def ising_unitaries(seed, Js=1.0, h_over_Js=0.1, T=10.0):
    rng = np.random.default_rng(11000 + seed); H = np.zeros((D, D), complex)
    for i in range(N):
        for j in range(i + 1, N): H += rng.uniform(-Js / 2, Js / 2) * dl._one(dl._Z, i) @ dl._one(dl._Z, j)
        H += h_over_Js * Js * dl._one(dl._X, i)
    U = expm(-1j * H * T / 2); return U, U


def features_u(u, Ua, Ub):
    """same computation as dom_lib.features, for given half-step unitaries"""
    W = Ub @ Ua; paulis = [dl._one(dl._X, q) for q in range(N)] + [dl._one(dl._Z, q) for q in range(N)]; G = np.empty((3, NFEAT, 2 * DM * DM))
    for h, V in enumerate((Ua, W)):
        for j, P in enumerate(paulis):
            A = V.conj().T @ P @ V; blk = (A[:DM, :DM], A[DM:, DM:], A[:DM, DM:] + A[DM:, :DM])
            for c in range(3):
                B = blk[c].T; G[c, h * 2 * N + j] = np.concatenate([B.real.ravel(), -B.imag.ravel()])
    K = np.stack([W[:DM, :DM], W[:DM, DM:], W[DM:, :DM], W[DM:, DM:]]); GG = np.ascontiguousarray(G.reshape(3 * NFEAT, -1))
    tau = np.eye(DM, dtype=complex) / DM; out = np.empty((len(u), NFEAT)); v = np.empty(2 * DM * DM); a = np.sqrt(1 - u); b = np.sqrt(u); c = a * b
    for t in range(len(u)):
        v[:DM * DM] = tau.real.ravel(); v[DM * DM:] = tau.imag.ravel()
        g = GG @ v; out[t] = (1 - u[t]) * g[:NFEAT] + u[t] * g[NFEAT:2 * NFEAT] + c[t] * g[2 * NFEAT:]
        K0 = a[t] * K[0] + b[t] * K[1]; K1 = a[t] * K[2] + b[t] * K[3]
        tau = K0 @ tau @ K0.conj().T + K1 @ tau @ K1.conj().T
        tau = (tau + tau.conj().T) / (2 * np.trace(tau).real)
    return out


def ising_features(u, seed): return features_u(u, *ising_unitaries(seed))


def brute(u, Ua, Ub):
    paulis = [dl._one(dl._X, q) for q in range(N)] + [dl._one(dl._Z, q) for q in range(N)]; tau = np.eye(DM, dtype=complex) / DM; out = np.empty((len(u), NFEAT))
    for t in range(len(u)):
        psi = np.array([np.sqrt(1 - u[t]), np.sqrt(u[t])]); rho = np.kron(np.outer(psi, psi), tau); r1 = Ua @ rho @ Ua.conj().T; r2 = Ub @ r1 @ Ub.conj().T
        out[t] = [np.trace(P @ r).real for r in (r1, r2) for P in paulis]; tau = np.einsum("iaib->ab", r2.reshape(2, DM, 2, DM))
    return out


def selftest():
    u = np.random.default_rng(1).uniform(0, 1, 40)
    assert np.array_equal(features_u(u, *dl.circuit(3, "G3")), dl.features(u, 3, "G3"))          # identical to the frozen code
    Ua, Ub = ising_unitaries(3); assert np.allclose(Ua @ Ua.conj().T, np.eye(D)); assert np.abs(ising_features(u, 3) - brute(u, Ua, Ub)).max() < 1e-10
    uu = np.random.default_rng(0).uniform(0, 1, 400); return [int((ising_features(uu, s)[50:].std(0) > 1e-6).sum()) for s in range(3)]


if __name__ == "__main__":
    print("selftest ok; Ising features that respond to a random input, seeds 0-2:", selftest())
