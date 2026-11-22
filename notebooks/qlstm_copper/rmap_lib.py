"""Regime map (long_prereg_rmap.json): synthetic series in which WE choose how the next value depends on the past,
to find the conditions under which the quantum reservoirs (DOM, ISING) beat matched classical models.

Process (one dataset):  x_{t+1} = b1 x_t + b2 mean(x_{t-9..t}) + b3 mean(x_{t-49..t}) + a * g_t + e_{t+1},  e ~ N(0, 1).
  memory  "long"  b = (0.35, 0.35, 0.25)   (HAR-like);  "short"  b = (0.60, 0, 0).
  g_t     a hidden "teacher" function of x_{<=t}, centred and scaled to unit variance with constants from a pilot run (a = 0).
  a       sqrt(s / (1 - s)), s = intended share of the next-step variance (beyond the linear part) that comes from g.
Teachers ("structure"): QDOM (Domingo G3 circuit, seed 500+r), QISING (Ising reservoir, seed 500+r), CESN (tanh echo state
network, seed 500+r), CSAS (28-number state-affine system, seed 500+r), POLY2 (z_t z_{t-4}), POLY3 (z_t z_{t-1} z_{t-6}), each
with a random Gaussian readout (seed 600+r) for the reservoir teachers. Quantum teachers read u_t = clip((x_t - lo)/(hi - lo)),
lo, hi = 0.1% and 99.9% quantiles of the pilot run, the same kind of min-max scaling the student pipeline uses.
Nothing here reads copper data.
"""
import numpy as np
import dom_lib as dl, dom2_lib as d2, dom3_lib as d3

BURN, T = 500, 7000
B = {"long": (0.35, 0.35, 0.25), "short": (0.60, 0.0, 0.0)}
STRUCT = ["QDOM", "QISING", "CESN", "CSAS", "POLY2", "POLY3"]
SHARES = [0.0, 0.02, 0.05, 0.15]


def q_prep(Ua, Ub):
    """the (G, K) tensors of dom2_lib.features_u for given half-step unitaries"""
    N, DM, NF = dl.N, dl.DM, dl.NFEAT; W = Ub @ Ua; paulis = [dl._one(dl._X, q) for q in range(N)] + [dl._one(dl._Z, q) for q in range(N)]
    G = np.empty((3, NF, 2 * DM * DM))
    for h, V in enumerate((Ua, W)):
        for j, P in enumerate(paulis):
            A = V.conj().T @ P @ V; blk = (A[:DM, :DM], A[DM:, DM:], A[:DM, DM:] + A[DM:, :DM])
            for c in range(3):
                Bk = blk[c].T; G[c, h * 2 * N + j] = np.concatenate([Bk.real.ravel(), -Bk.imag.ravel()])
    K = np.stack([W[:DM, :DM], W[:DM, DM:], W[DM:, :DM], W[DM:, DM:]]); return np.ascontiguousarray(G.reshape(3 * NF, -1)), K


class QTeacher:
    """online version of dom2_lib.features_u: step(u) returns the 28 features for input u given all earlier inputs"""
    def __init__(self, Ua, Ub):
        self.GG, self.K = q_prep(Ua, Ub); self.tau = np.eye(dl.DM, dtype=complex) / dl.DM; self.v = np.empty(2 * dl.DM * dl.DM)

    def step(self, u):
        DM, NF = dl.DM, dl.NFEAT; self.v[:DM * DM] = self.tau.real.ravel(); self.v[DM * DM:] = self.tau.imag.ravel()
        g = self.GG @ self.v; a, b = np.sqrt(1 - u), np.sqrt(u); f = (1 - u) * g[:NF] + u * g[NF:2 * NF] + a * b * g[2 * NF:]
        K0 = a * self.K[0] + b * self.K[1]; K1 = a * self.K[2] + b * self.K[3]
        tau = K0 @ self.tau @ K0.conj().T + K1 @ self.tau @ K1.conj().T; self.tau = (tau + tau.conj().T) / (2 * np.trace(tau).real); return f


class ESNTeacher:
    """online version of dom_lib.esn1 (seed offset 7000 as there), leak 0.5, spectral radius 0.9"""
    def __init__(self, seed, units=28, leak=0.5, rho=0.9, scale=0.5):
        rng = np.random.default_rng(7000 + seed); self.Win = rng.uniform(-scale, scale, units); W = rng.normal(size=(units, units))
        self.W = W * rho / max(abs(np.linalg.eigvals(W))); self.s = np.zeros(units); self.leak = leak

    def step(self, z):
        self.s = (1 - self.leak) * self.s + self.leak * np.tanh(self.Win * z + self.W @ self.s); return self.s.copy()


class SASTeacher:
    """online version of dom3_lib.sas (seed offset 15000 as there), gain 0.5, leak 0.5"""
    def __init__(self, seed, units=28, leak=0.5, gain=0.5):
        rng = np.random.default_rng(15000 + seed); A = rng.normal(size=(3, units, units)); self.A = A * (gain / np.array([np.linalg.norm(x, 2) for x in A]))[:, None, None]
        self.b = rng.uniform(-1, 1, (3, units)); self.s = np.zeros(units); self.leak = leak

    def step(self, u):
        w = (1 - u, u, np.sqrt(u * (1 - u)))
        self.s = (1 - self.leak) * self.s + self.leak * ((w[0] * self.A[0] + w[1] * self.A[1] + w[2] * self.A[2]) @ self.s + w[0] * self.b[0] + w[1] * self.b[1] + w[2] * self.b[2])
        return self.s.copy()


def teacher_seed(r): return 500 + r


def make_teacher(struct, r):
    """returns (kind, object or None, readout weights or None); kind 'u' reads the scaled input, 'z' the standardised input"""
    w = np.random.default_rng(600 + r).normal(size=28)
    if struct == "QDOM": return "u", QTeacher(*dl.circuit(teacher_seed(r), "G3")), w
    if struct == "QISING": return "u", QTeacher(*d2.ising_unitaries(teacher_seed(r))), w
    if struct == "CESN": return "z", ESNTeacher(teacher_seed(r)), w
    if struct == "CSAS": return "u", SASTeacher(teacher_seed(r)), w
    return "z", None, None


def simulate(memory, struct, a, r, gconst=None, scale=None, sd=None, n=T, burn=BURN):
    """one path. gconst = (mean, sd) of the raw teacher output; scale = (lo, hi) for u; sd = sd of x for z. Returns x (n,), g (n,) where
    g[t] is the centred, scaled teacher output computed from x[<=t] (it drives x[t+1])."""
    b1, b2, b3 = B[memory]; rng = np.random.default_rng(1000 * r + 7 + 100000 * STRUCT.index(struct) + (0 if memory == "long" else 50000))
    e = rng.normal(size=n + burn + 1); x = np.zeros(n + burn + 1); graw = np.zeros(n + burn + 1)
    kind, obj, w = make_teacher(struct, r); lo, hi = scale if scale else (-4.0, 4.0); s0 = sd if sd else 1.0; gm, gs = gconst if gconst else (0.0, 1.0)
    for t in range(n + burn):
        z = x[t] / s0
        if struct == "POLY2": raw = z * (x[t - 4] / s0 if t >= 4 else 0.0)
        elif struct == "POLY3": raw = z * (x[t - 1] / s0 if t >= 1 else 0.0) * (x[t - 6] / s0 if t >= 6 else 0.0)
        else:
            inp = float(np.clip((x[t] - lo) / (hi - lo), 0.0, 1.0)) if kind == "u" else z; raw = float(obj.step(inp) @ w)
        graw[t] = raw; g = np.clip((raw - gm) / gs, -4.0, 4.0)
        m10 = x[max(0, t - 9):t + 1].mean(); m50 = x[max(0, t - 49):t + 1].mean()
        x[t + 1] = b1 * x[t] + b2 * m10 + b3 * m50 + a * g + e[t + 1]
    gfin = np.clip((graw - gm) / gs, -4.0, 4.0)
    return x[burn:burn + n], gfin[burn:burn + n], graw[burn:burn + n]


def pilot(memory, struct, r):
    """constants for scaling, from a run with a = 0 (same noise path); they depend only on the linear process and the teacher"""
    x, _, graw = simulate(memory, struct, 0.0, r)
    lo, hi = np.quantile(x, [0.001, 0.999]); sd = x.std()
    x2, _, graw2 = simulate(memory, struct, 0.0, r, scale=(lo, hi), sd=sd)
    return dict(scale=(float(lo), float(hi)), sd=float(sd), gconst=(float(graw2.mean()), float(graw2.std() + 1e-12)))


def dataset(memory, struct, share, r):
    a = float(np.sqrt(share / (1 - share))); c = pilot(memory, struct, r)
    x, g, _ = simulate(memory, struct, a, r, gconst=c["gconst"], scale=c["scale"], sd=c["sd"]); return x, g, a, c


def selftest():
    rng = np.random.default_rng(1); u = rng.uniform(0, 1, 40); Ua, Ub = dl.circuit(3, "G3"); qt = QTeacher(Ua, Ub)
    on = np.array([qt.step(v) for v in u]); assert np.abs(on - d2.features_u(u, Ua, Ub)).max() < 1e-10
    z = rng.normal(size=40); et = ESNTeacher(3); assert np.abs(np.array([et.step(v) for v in z]) - dl.esn1(z, 3)).max() < 1e-12
    st = SASTeacher(3); assert np.abs(np.array([st.step(v) for v in u]) - d3.sas(u, 3)).max() < 1e-12
    out = {}
    for s in STRUCT:
        x, g, a, c = dataset("long", s, 0.15, 0); assert np.isfinite(x).all() and np.isfinite(g).all()
        out[s] = (round(float(g.std()), 2), round(float(np.corrcoef(g[:-1], x[1:])[0, 1]), 3), round(float(x.std()), 2))
    return out


if __name__ == "__main__":
    import time
    t = time.time(); print("selftest ok; (sd of g, corr(g_t, x_t+1), sd x) at share 0.15, long memory:", selftest(), f"{time.time() - t:.0f}s")
