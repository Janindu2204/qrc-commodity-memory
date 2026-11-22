"""Multi-commodity version of the hourly protocol (same windows, split, target, test-row logic as the copper work).

Target for asset A: log of mean squared hourly return over the next 24 valid bars. Drivers = a fixed list of hourly futures/FX
series minus A itself, same-stamp completed bars (return and 24-bar log variance). Split dates are identical to copper:
train/val before 2025-05-01 minus 30h (interleaved 10-day validation blocks, 30h purge), test from 2025-05-01.
Preliminary, AI-written, not independently re-derived.
"""
from pathlib import Path
import numpy as np, pandas as pd
import hourly_baselines as hb

DATA = Path(__file__).resolve().parents[2] / "data" / "hourly_commod"
ASSETS = ["GC_F", "SI_F", "PL_F", "PA_F", "CL_F", "NG_F", "HO_F", "ZC_F", "ZS_F", "ZW_F"]
DRIVERS = ["HG_F", "GC_F", "SI_F", "PL_F", "CL_F", "ES_F", "ZN_F", "DX-Y.NYB", "EURUSD_X", "AUDUSD_X", "USDCAD_X", "USDZAR_X"]
SEQ, STRIDE = 24, 2
TRAIN_END = pd.Timestamp("2025-05-01")


def read(name):
    d = pd.read_csv(DATA / f"{name}_1h.csv", parse_dates=["timestamp"]).set_index("timestamp")
    d = d[~d.index.duplicated()]
    return d


def returns(d):
    r = np.log(d.close).diff()
    r[d.index.to_series().diff() > pd.Timedelta("2h")] = np.nan
    return r


def driver_frames(idx, asset):
    ret, var = {}, {}
    names = [t for t in DRIVERS if t != asset]
    for t in names:
        e = read(t); r = returns(e)
        lv = np.log((r ** 2).rolling(24, min_periods=19).mean() + 1e-10)
        ret[t] = r.reindex(idx).fillna(0.0).values
        var[t] = lv.reindex(lv.index.union(idx)).ffill(limit=3).reindex(idx).values
    return pd.DataFrame(ret, index=idx), pd.DataFrame(var, index=idx)


def prepare(asset):
    d = read(asset); r = returns(d)
    X, y, R, yr, hod = hb.build(r)
    ER, EV = driver_frames(d.index, asset)
    F = pd.concat([pd.DataFrame({"ret": r.fillna(0.0), "lv24": X.lv24, "lv120": X.lv120, "lv480": X.lv480}, index=d.index),
                   ER.add_prefix("er_"), EV.add_prefix("ev_")], axis=1)
    ok = F.notna().all(axis=1).values & y.notna().values
    pos = np.arange(len(d))
    full = pd.Series(ok).rolling(SEQ).min().fillna(0).astype(bool).values
    t_tr = pos[full & (d.index < TRAIN_END - pd.Timedelta(hours=hb.H + 6))]
    cnt = np.cumsum(r.notna().values)                      # purge by trading bars: the label window ends at the H-th valid return after t
    lab_end = np.minimum(np.searchsorted(cnt, cnt[t_tr] + hb.H), len(d) - 1)
    t_tr = t_tr[d.index[lab_end] < hb.TEST_START]
    te = pos[full & (d.index >= hb.TEST_START)]
    days = np.asarray((d.index[t_tr] - d.index[t_tr[0]]).days); blk = days // 10; is_va = blk % 5 == 4
    near = np.zeros(len(t_tr), bool)
    for b in np.unique(blk[is_va]):
        lo = d.index[t_tr[0]] + pd.Timedelta(days=int(b) * 10) - pd.Timedelta(hours=hb.H + 6)
        hi = d.index[t_tr[0]] + pd.Timedelta(days=int(b) * 10 + 10) + pd.Timedelta(hours=hb.H + 6)
        near |= (d.index[t_tr] >= lo) & (d.index[t_tr] < hi)
    tr, va = t_tr[~near], t_tr[is_va]
    yv = y.values
    Hc = X[["lv24", "lv120", "lv480"]].values
    import hourly_qrc as hq
    base_te = hq.ridge(Hc[tr], yv[tr], Hc[va], yv[va], Hc[te])
    s2 = float(np.mean((yv[tr] - hq.ridge(Hc[tr], yv[tr], Hc[va], yv[va], Hc[tr])) ** 2))
    shock = (d.index >= hb.SHOCK[0] - pd.Timedelta(hours=hb.H + 1)) & (d.index <= hb.SHOCK[1])
    return dict(asset=asset, d=d, r=r, X=X, y=y, yv=yv, F=F, tr=tr, va=va, te=te, Hc=Hc, base_te=base_te, s2=s2,
                wins={"all": np.ones(len(te), bool), "ex_shock": ~shock[te]})
