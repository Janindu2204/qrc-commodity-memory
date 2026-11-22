"""One network fit: python long_nn.py VERSION ORIGIN MODEL SEED   (MODEL: LSTM|QLSTMT, or LSTMx|QLSTMTx = with 3 lagged outside inputs). Reuses v2_nn.train_net (QLIKE loss, anchored on lvR5) with seq 5."""
import sys, numpy as np
import v2_nn as vn
import long_core as lc

version, origin, model, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
vn.SEQ = 5
mvf = model.endswith("x"); net_name = model[:-1] if mvf else model
D = lc.prepare(version, origin, mv=mvf)
predict, secs = vn.train_net(net_name, D["M"], D["anchor"], D["y"], D["tr"], D["va"], seed, lr=0.02 if net_name == "LSTM" else 0.005)
out = lc.OUT / version / str(origin); out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{model}_s{seed}.npz", lv_val=predict(D["va"]), lv_te=predict(D["te"]), seconds=secs)
print(version, origin, model, seed, f"{secs:.0f}s", flush=True)
