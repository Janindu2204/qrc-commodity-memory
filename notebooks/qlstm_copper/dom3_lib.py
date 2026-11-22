"""Additions for long_prereg_dom3.json (dom_lib.py and dom2_lib.py are frozen by earlier hashes and are not edited).
Classical stand-ins for the quantum reservoirs (mechanism check):
  esn_k : leaky echo state network with k inputs (used with the two input functions of the quantum encoding: u and sqrt(u(1-u))).
  sas   : state-affine system s_t = (1-leak) s_{t-1} + leak (A(u_t) s_{t-1} + b(u_t)), A(u) = (1-u) A0 + u A1 + sqrt(u(1-u)) A2, same form for b.
          This is the algebraic form of the quantum reservoir's memory update (linear in the state, coefficients 1-u, u, sqrt(u(1-u))) with a 28-number state and no quantum simulation.
Classical networks (classical machine learning tier): Net (LSTM or GRU + linear head), fit_seq (squared error, intraday), fit_daily (QLIKE, anchored, daily).
"""
import numpy as np, torch
from torch import nn

torch.set_num_threads(1)
NFEAT = 28
ESN_GRID = [dict(rho=r, leak=k) for r in (0.6, 0.9, 1.2) for k in (0.1, 0.2, 0.5, 1.0)]
SAS_GRID = [dict(gain=g, leak=k) for g in (0.3, 0.5, 0.65) for k in (0.1, 0.2, 0.5, 1.0)]
NN_GRID = [dict(hidden=h, lr=l) for h in (8, 16, 32) for l in (3e-4, 1e-3, 3e-3, 1e-2)]


def esn_k(Z, seed, units=NFEAT, leak=0.5, rho=0.9, scale=0.5):
    """Z (T, k) standardised inputs (no NaN). Row t uses inputs up to Z[t]."""
    rng = np.random.default_rng(13000 + seed); Win = rng.uniform(-scale, scale, (units, Z.shape[1])); W = rng.normal(size=(units, units)); W *= rho / max(abs(np.linalg.eigvals(W)))
    s = np.zeros(units); out = np.empty((len(Z), units)); drive = Z @ Win.T
    for t in range(len(Z)):
        s = (1 - leak) * s + leak * np.tanh(drive[t] + W @ s); out[t] = s
    return out


def sas(u, seed, units=NFEAT, leak=0.5, gain=0.5):
    """u in [0,1]. Each A_i has spectral norm `gain`; the coefficients sum to at most 1.5, so gain <= 0.65 keeps the update a contraction."""
    rng = np.random.default_rng(15000 + seed); A = rng.normal(size=(3, units, units)); A *= (gain / np.array([np.linalg.norm(a, 2) for a in A]))[:, None, None]
    b = rng.uniform(-1, 1, (3, units)); c = np.sqrt(u * (1 - u)); s = np.zeros(units); out = np.empty((len(u), units))
    for t in range(len(u)):
        w = (1 - u[t], u[t], c[t]); s = (1 - leak) * s + leak * ((w[0] * A[0] + w[1] * A[1] + w[2] * A[2]) @ s + w[0] * b[0] + w[1] * b[1] + w[2] * b[2]); out[t] = s
    return out


class Net(nn.Module):
    def __init__(self, kind, n_in, hidden):
        super().__init__(); self.rnn = getattr(nn, kind)(n_in, hidden, batch_first=True); self.head = nn.Linear(hidden, 1)

    def forward(self, x): return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


@torch.no_grad()
def _pred(net, X, chunk=8192): return torch.cat([net(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def fit_seq(kind, cfg, seed, Xtr, rtr, Xva, rva, Xte=None, max_epochs=30, patience=4, batch=512):
    """Xtr (n, SEQ, k), rtr (n,) float32 tensors (standardised target). Adam, squared error, early stopping on validation. Returns (val mse, val pred, test pred) as numpy."""
    torch.manual_seed(seed); np.random.seed(seed); net = Net(kind, Xtr.shape[2], cfg["hidden"]); opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    best, bs, bad = np.inf, None, 0
    for ep in range(max_epochs):
        net.train(); perm = torch.as_tensor(np.random.permutation(len(Xtr)))
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]; loss = ((net(Xtr[j]) - rtr[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); v = float(((_pred(net, Xva) - rva) ** 2).mean())
        if v < best - 1e-7: best, bs, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(bs); net.eval()
    return best, _pred(net, Xva).numpy().astype(float), (None if Xte is None else _pred(net, Xte).numpy().astype(float))


def fit_daily(kind, cfg, seed, M, anc, y, tr, va, seq=5, max_epochs=25, patience=5, batch=32):
    """As v2_nn.train_net: windows of the last `seq` rows of M; eta = anchor + net(window); loss = mean(exp(y - eta) + eta). Returns level forecasts for all rows (NaN before row seq)."""
    torch.manual_seed(seed); np.random.seed(seed); net = Net(kind, M.shape[1], cfg["hidden"]); opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    X = torch.tensor(M.astype("float32")); an = torch.tensor(anc.astype("float32")); yy = torch.tensor(y.astype("float32"))
    win = lambda rows: torch.stack([X[t - seq + 1: t + 1] for t in rows]); trw, vaw = tr[tr >= seq], va[va >= seq]; Xtr, Xva = win(trw), win(vaw)

    def loss_fn(rows, W): eta = an[rows] + net(W); return (torch.exp(yy[rows] - eta) + eta).mean()

    best, bs, bad = np.inf, None, 0
    for ep in range(max_epochs):
        net.train(); perm = np.random.permutation(len(trw))
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]; loss = loss_fn(trw[j], Xtr[j]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = float(loss_fn(vaw, Xva))
        if np.isfinite(v) and v < best - 1e-6: best, bs, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    out = np.full(len(M), np.nan)
    if bs is None: return out
    net.load_state_dict(bs); net.eval(); rows = np.arange(seq, len(M))
    with torch.no_grad(): out[rows] = np.exp((an[rows] + torch.cat([net(win(rows[i:i + 512])) for i in range(0, len(rows), 512)])).numpy())
    return out


def selftest():
    rng = np.random.default_rng(1); u = rng.uniform(0, 1, 400); Z = rng.normal(size=(400, 2))
    f = esn_k(Z, 3); g = sas(u, 3, gain=0.65, leak=1.0); assert f.shape == g.shape == (400, NFEAT) and np.isfinite(f).all() and np.isfinite(g).all()
    g2 = sas(np.r_[rng.uniform(0, 1, 400), u], 3, gain=0.65, leak=1.0)[400:]; f2 = esn_k(np.r_[rng.normal(size=(400, 2)), Z], 3)[400:]      # fading memory: the start is forgotten
    assert np.abs(g[-1] - g2[-1]).max() < 1e-6 and np.abs(f[-1] - f2[-1]).max() < 1e-6
    assert (f[50:].std(0) > 1e-6).all() and (g[50:].std(0) > 1e-6).all()
    torch.manual_seed(0); X = torch.randn(600, 10, 5); r = X[:, -1, 0] * 0.8 + 0.1 * torch.randn(600)                                         # a planted signal is learned
    for kind in ("LSTM", "GRU"):
        v, pv, pt = fit_seq(kind, dict(hidden=8, lr=1e-2), 0, X[:400], r[:400], X[400:500], r[400:500], X[500:]); assert v < 0.3 and pt.shape == (100,), (kind, v)
    return "ok"


if __name__ == "__main__":
    print("selftest", selftest())
