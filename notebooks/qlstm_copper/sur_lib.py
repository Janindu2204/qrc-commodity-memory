"""Shared pieces for long_prereg_surrogate.json and long_prereg_smalln.json (drafts until frozen). Mirrors qrc_roll.py: 6 inputs, 21 features, QLIKE-trained linear readout."""
import itertools
import numpy as np
import long_core as lc, v2_core as vc, hourly_qrc as hq, hourly_tune_qrc as tq

NQ, UNITS = 6, 21
LAMGRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
PAIRS = list(itertools.combinations(range(NQ), 2))


def load_block(year):
    D = lc.prepare("range_m", year, length=1)
    y, anc, M, tr, va, te = D["y"], D["anchor"], D["M"], D["tr"], D["va"], D["te"]
    X0 = M[:, :3]; G = np.hstack([X0, np.vstack([np.zeros((1, 3)), X0[:-1]])])        # today and previous day: 6 inputs
    return D, y, anc, M, X0, G, tr, va, te


def readout(M, F, y, anc, tr, va):
    X = M if F is None else np.hstack([M, F]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sc, lam, p = vc.best_lambda_linear(X, y, anc, tr, va, grid=LAMGRID)
    return np.exp(vc.predict_linear(p, X, anc))


def put(rows, n, Fr):
    F = np.zeros((n, Fr.shape[1])); F[rows] = Fr; return F


def qrc_feats(G, rows, s, ent, layers, scale=1.0):
    return put(rows, len(G), hq.reservoir_features(G[rows] * scale, NQ, s, ent, layers))


def basis_feats(G, rows, d, scale=1.0):
    """all products of up to d DISTINCT inputs of {cos theta_i, sin theta_i}, theta = arctan(scale*G); no constant column"""
    th = np.arctan(G[rows] * scale); C, S = np.cos(th), np.sin(th); cols = []
    for k in range(1, d + 1):
        for idx in itertools.combinations(range(NQ), k):
            for fs in itertools.product((C, S), repeat=k): cols.append(np.prod([f[:, i] for i, f in zip(idx, fs)], axis=0))
    return put(rows, len(G), np.column_stack(cols))


def randproj_feats(B, tr, rows, s):
    mu, sd = B[tr].mean(0), B[tr].std(0) + 1e-9; Bs = (B[rows] - mu) / sd
    W = np.random.default_rng(3000 + s).normal(size=(B.shape[1], UNITS)) / np.sqrt(B.shape[1])
    return put(rows, len(B), Bs @ W)


def trig_feats(G, rows, s, scale=1.0, freq=1.0):
    rng = np.random.default_rng(2000 + s); phi = rng.uniform(0, 2 * np.pi, UNITS); sg = rng.choice([-1.0, 1.0], len(PAIRS))
    th = np.arctan(G[rows] * scale); Fm = np.zeros((len(rows), UNITS))
    for i in range(NQ): Fm[:, i] = np.cos(freq * th[:, i] + phi[i])
    for j, (a, c) in enumerate(PAIRS): Fm[:, NQ + j] = np.cos(freq * (th[:, a] + sg[j] * th[:, c]) + phi[NQ + j])
    return put(rows, len(G), Fm)


def randfeat_feats(G, rows, s, scale=1.0, wscale=1.0):
    rng = np.random.default_rng(1000 + s); W = wscale * rng.normal(size=(NQ, UNITS)) / np.sqrt(NQ); b = rng.uniform(0, 2 * np.pi, UNITS)
    return put(rows, len(G), np.tanh(np.arctan(G[rows] * scale) @ W + b))


def esn_feats(X0, s, radius, leak):
    return tq.esn(X0, s, radius, leak, UNITS)


def span_r2(F, B):
    """R2 of each column of F regressed on [1, B]"""
    A = np.column_stack([np.ones(len(B)), B]); W = np.linalg.lstsq(A, F, rcond=None)[0]; R = F - A @ W
    return 1 - (R ** 2).sum(0) / ((F - F.mean(0)) ** 2).sum(0)
