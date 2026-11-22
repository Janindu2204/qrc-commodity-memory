"""Part 4 (long_prereg_mech.json): memory and nonlinear capacity of the reservoirs, and the task profile of copper's HAR residual.
Capacity (Dambre et al. 2012 style, on i.i.d. input): u_t ~ U(0,1) i.i.d., v = 2u - 1. A ridge readout (intercept + standardised features)
is fitted on rows [WASH, FIT) and scored by R^2 on rows [FIT, T). Capacity for a target = max(0, R^2).
  linear memory function MF(k): target v_{t-k}, k = 0..KMAX.
  degree-2 capacity: targets L2(v_{t-i}) = (3 v^2 - 1)/2, i = 0..20, and v_{t-i} v_{t-j}, 0 <= i < j <= 10.
Pure functions; no copper data here (mech_copper.py reads copper).
"""
import numpy as np

T, WASH, FIT, KMAX = 20000, 500, 12500, 400
LAM = 1e-6


def inputs(seed=0):
    return np.random.default_rng(31000 + seed).uniform(0, 1, T)


def _ridge_r2(F, Y):
    """F (T, p) features, Y (T, m) targets. Fit rows [WASH, FIT), score rows [FIT, T). Returns R^2 per target."""
    A, B = F[WASH:FIT], F[FIT:]; mu, sd = A.mean(0), A.std(0) + 1e-12; A = (A - mu) / sd; B = (B - mu) / sd
    A = np.column_stack([np.ones(len(A)), A]); B = np.column_stack([np.ones(len(B)), B])
    G = A.T @ A + LAM * len(A) * np.eye(A.shape[1]); G[0, 0] -= LAM * len(A)
    W = np.linalg.solve(G, A.T @ Y[WASH:FIT]); E = Y[FIT:] - B @ W; V = Y[FIT:].var(0)
    return 1 - (E ** 2).mean(0) / V


def targets(u):
    v = 2 * u - 1; lag = lambda k: np.r_[np.full(k, 0.0), v[:len(v) - k]] if k else v
    lin = np.column_stack([lag(k) for k in range(KMAX + 1)])
    sq = np.column_stack([(3 * lag(i) ** 2 - 1) / 2 for i in range(21)])
    cross = np.column_stack([lag(i) * lag(j) for i in range(11) for j in range(i + 1, 11)])
    return lin, sq, cross


def capacity(F, u):
    """returns dict: MF (KMAX+1,), MC (sum), C2_sq (21,), C2_cross (55,), C2 (sum), MC_beyond(k) helpers"""
    lin, sq, cross = targets(u); ok = np.all(np.isfinite(F)) and F[WASH:].std(0).max() > 1e-12
    if not ok: z = np.zeros; return dict(MF=z(KMAX + 1), MC=0.0, C2_sq=z(21), C2_cross=z(55), C2=0.0)
    mf = np.maximum(0, _ridge_r2(F, lin)); s2 = np.maximum(0, _ridge_r2(F, sq)); c2 = np.maximum(0, _ridge_r2(F, cross))
    return dict(MF=mf, MC=float(mf.sum()), C2_sq=s2, C2_cross=c2, C2=float(s2.sum() + c2.sum()))


def horizon(mf, frac=0.9):
    """smallest lag K such that lags 0..K hold `frac` of the total linear memory capacity"""
    c = np.cumsum(mf); return int(np.searchsorted(c, frac * c[-1])) if c[-1] > 0 else 0


def selftest():
    u = inputs(0); lin, sq, cross = targets(u)
    # a delay line holding the last 30 inputs has MF = 1 for k <= 30 and 0 after; degree-2 capacity ~0
    F = lin[:, :31].copy(); c = capacity(F, u)
    assert c["MF"][:31].min() > 0.999 and c["MF"][35:].max() < 0.01 and abs(c["MC"] - 31) < 0.5, c["MC"]
    assert c["C2"] < 1.0, c["C2"]
    G = np.column_stack([F, sq[:, :5]]); c2 = capacity(G, u); assert c2["C2_sq"][:5].min() > 0.999
    assert horizon(c["MF"]) <= 30
    return round(c["MC"], 3), round(c2["C2"], 3)


if __name__ == "__main__":
    print("selftest ok", selftest())
