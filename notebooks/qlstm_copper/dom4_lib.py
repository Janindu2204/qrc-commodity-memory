"""Additions for long_prereg_dom4.json (dom_lib.py, dom2_lib.py, dom3_lib.py are frozen by earlier hashes and are not edited).
Larger classical stand-ins: the two stand-ins of dom3_lib with a 256-number state, read through 28 fixed random linear projections
(the quantum reservoir also has a large hidden state, a 64 x 64 density matrix, read through 28 numbers).
  esn_big : 256-unit leaky echo state network with k inputs, 28 projections.
  sas_big : 256-number state-affine system s_t = (1-leak) s + leak (A(u) s + b(u)), A(u) = (1-u) A0 + u A1 + sqrt(u(1-u)) A2, 28 projections.
"""
import numpy as np

NSTATE = 256; NFEAT = 28


def esn_big(Z, seed, units=NSTATE, leak=0.5, rho=0.9, scale=0.5):
    rng = np.random.default_rng(19000 + seed); Win = rng.uniform(-scale, scale, (units, Z.shape[1])); W = rng.normal(size=(units, units)); W *= rho / max(abs(np.linalg.eigvals(W)))
    P = rng.normal(size=(NFEAT, units)) / np.sqrt(units); s = np.zeros(units); out = np.empty((len(Z), NFEAT)); drive = Z @ Win.T
    for t in range(len(Z)):
        s = (1 - leak) * s + leak * np.tanh(drive[t] + W @ s); out[t] = P @ s
    return out


def sas_big(u, seed, units=NSTATE, leak=0.5, gain=0.5):
    rng = np.random.default_rng(17000 + seed); A = rng.normal(size=(3, units, units)); A *= (gain / np.array([np.linalg.norm(a, 2) for a in A]))[:, None, None]
    b = rng.uniform(-1, 1, (3, units)); P = rng.normal(size=(NFEAT, units)) / np.sqrt(units); AA = np.ascontiguousarray(A.reshape(3 * units, units))
    c = np.sqrt(u * (1 - u)); s = np.zeros(units); out = np.empty((len(u), NFEAT))
    for t in range(len(u)):
        w0, w1, w2 = 1 - u[t], u[t], c[t]; g = AA @ s
        s = (1 - leak) * s + leak * (w0 * (g[:units] + b[0]) + w1 * (g[units:2 * units] + b[1]) + w2 * (g[2 * units:] + b[2])); out[t] = P @ s
    return out


def selftest():
    import dom3_lib as d3
    rng = np.random.default_rng(1); u = rng.uniform(0, 1, 400); Z = rng.normal(size=(400, 2)); f = esn_big(Z, 3); g = sas_big(u, 3, gain=0.65, leak=1.0)
    assert f.shape == g.shape == (400, NFEAT) and np.isfinite(f).all() and np.isfinite(g).all() and (f[50:].std(0) > 1e-6).all() and (g[50:].std(0) > 1e-6).all()
    g2 = sas_big(np.r_[rng.uniform(0, 1, 400), u], 3, gain=0.65, leak=1.0)[400:]; f2 = esn_big(np.r_[rng.normal(size=(400, 2)), Z], 3)[400:]
    assert np.abs(g[-1] - g2[-1]).max() < 1e-6 and np.abs(f[-1] - f2[-1]).max() < 1e-6                     # fading memory
    # with a 28-number state and identity projection the big system is the same computation as dom3_lib.sas (checked on the update rule)
    A = np.random.default_rng(5).normal(size=(3, 4, 4)); s = np.arange(4.0); w = (0.3, 0.7, np.sqrt(0.21)); g = A.reshape(12, 4) @ s
    assert np.allclose(w[0] * g[:4] + w[1] * g[4:8] + w[2] * g[8:], (w[0] * A[0] + w[1] * A[1] + w[2] * A[2]) @ s)
    return "ok"


if __name__ == "__main__":
    import time
    print("selftest", selftest()); u = np.random.default_rng(0).uniform(0, 1, 20000); Z = np.random.default_rng(1).normal(size=(20000, 2))
    t = time.time(); esn_big(Z, 0); print("esn_big", (time.time() - t) * 50, "us/step"); t = time.time(); sas_big(u, 0); print("sas_big", (time.time() - t) * 50, "us/step")
