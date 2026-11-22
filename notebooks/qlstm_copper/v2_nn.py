"""v2 per-asset networks with QLIKE loss (see v2_prereg.json): LSTM_q and QLSTMT_q on feature set X, anchored on lvR24.
Usage: python v2_nn.py ASSET MODEL SEED   (MODEL: LSTM|QLSTMT).  Saves v2_preds/<asset>/<MODEL>_q_s<seed>.npz (uncalibrated levels, val+test)."""
import sys, time
import numpy as np, torch
from torch import nn
torch.set_num_threads(1)
import v2_core as vc
import returns_lib as rl

SEQ = 24

def train_net(model_name, Xs, anc, y, tr, va, seed, lr, hidden=8, max_epochs=25, patience=5, batch=32):
    """Xs (N, F) standardised; windows of the last SEQ rows ending at each row; loss = mean(exp(y-eta)+eta), eta = anchor + net(window)."""
    torch.manual_seed(seed); np.random.seed(seed)
    rl.HIDDEN = hidden
    net = rl.make_model(model_name, Xs.shape[1], **({"n_qlayers": 1} if model_name == "QLSTMT" else {}))
    X32 = torch.tensor(Xs.astype("float32")); an = torch.tensor(anc.astype("float32")); yy = torch.tensor(y.astype("float32"))
    win = lambda rows: torch.stack([X32[t - SEQ + 1: t + 1] for t in rows])
    trw, vaw = tr[tr >= SEQ], va[va >= SEQ]
    Xtr, Xva = win(trw), win(vaw)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    def loss_fn(rows, W):
        eta = an[rows] + net(W).squeeze(-1); return (torch.exp(yy[rows] - eta) + eta).mean()
    best, bs, bad = 1e9, None, 0; t0 = time.time()
    for ep in range(max_epochs):
        net.train(); perm = np.random.permutation(len(trw))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]; loss = loss_fn(trw[b], Xtr[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = float(loss_fn(vaw, Xva))
        if v < best - 1e-6: best, bs, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(bs); net.eval()
    @torch.no_grad()
    def predict(rows):
        rows = np.asarray(rows); out = np.full(len(rows), np.nan); ok = rows >= SEQ
        out[ok] = (an[rows[ok]] + torch.cat([net(win(rows[ok][i:i + 256])).squeeze(-1) for i in range(0, ok.sum(), 256)])).numpy()
        return np.exp(out)
    return predict, time.time() - t0

if __name__ == "__main__":
    asset, model, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    D = vc.prepare_v2(asset)
    predict, secs = train_net(model, D["M"]["X"], D["anchor"], D["y"], D["tr"], D["va"], seed, lr=0.02 if model == "LSTM" else 0.005)
    out = vc.OUT / asset; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{model}_q_s{seed}.npz", lv_val=predict(D["va"]), lv_te=predict(D["te"]), seconds=secs)
    print(asset, model, seed, f"{secs:.0f}s", flush=True)
