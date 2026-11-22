"""Shared pieces for the optimised-run stage: rows, features, baseline HAR, prediction store, evaluation.

All final predictions are stored on the SAME 7,667 test rows (hq.rows()) as hourly_opt_preds/<name>.npz with
fields p (forecast) and kind ('log' = log mean-variance forecast needing the log-normal correction exp(p + s2/2) for
level use; 'level' = already a mean-variance forecast). Tuning uses validation rows only; test rows only in finals.
"""
from pathlib import Path
import numpy as np, pandas as pd
import hourly_baselines as hb
import hourly_qrc as hq

PREDS = Path(__file__).with_name("hourly_opt_preds")
PREDS.mkdir(exist_ok=True)


def setup():
    d, Xf, y, F, tr, va, te = hq.rows()
    assert len(te) == 7667
    yv = y.values
    Hc = Xf[["lv24", "lv120", "lv480"]].values
    base_te = hq.ridge(Hc[tr], yv[tr], Hc[va], yv[va], Hc[te])
    s2 = float(np.mean((yv[tr] - hq.ridge(Hc[tr], yv[tr], Hc[va], yv[va], Hc[tr])) ** 2))
    shock = (d.index >= hb.SHOCK[0] - pd.Timedelta(hours=hb.H + 1)) & (d.index <= hb.SHOCK[1])
    return dict(d=d, X=Xf, y=y, yv=yv, F=F, tr=tr, va=va, te=te, Hc=Hc, base_te=base_te, s2=s2,
                wins={"all": np.ones(len(te), bool), "ex_shock": ~shock[te]})


def save(name, p, kind="log"):
    np.savez(PREDS / f"{name}.npz", p=np.asarray(p, float), kind=kind)


def load_all():
    out = {}
    for f in sorted(PREDS.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        out[f.stem] = (z["p"], str(z["kind"]))
    return out


def level(p, kind, s2):
    return np.exp(p + 0.5 * s2) if kind == "log" else p


def qlike_vec(var_pred, var_true):
    r = var_true / var_pred
    return r - np.log(r) - 1


def zscore_fit(a, rows):
    mu, sd = a[rows].mean(0), a[rows].std(0) + 1e-9
    return (a - mu) / sd
