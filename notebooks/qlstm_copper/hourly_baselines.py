"""Hourly HG=F baselines: random-walk returns test and HAR-style volatility forecast.

Data: data/copper_hgf_1h.csv (Yahoo, ~2.4 years, unadjusted front-month, ET).
No QLSTM here; this is the classical floor any network must beat on hourly bars.

Tasks (forecast origin = every hourly bar t, expanding window, monthly refit):
  ret : next-bar log return. Ridge on lagged returns vs the zero (random-walk) forecast.
  vol : log of mean squared return over the next 24 valid bars. Naive (last 24 bars),
        HAR (24/120/480-bar log variance), and a ridge on HAR + hour-of-day dummies.
Returns across gaps >2h (weekends, holidays) are dropped, not filled.
Test window is run twice: all, and excluding the 2025-07 tariff-shock window.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
TEST_START = pd.Timestamp("2025-05-01")
SHOCK = (pd.Timestamp("2025-07-07"), pd.Timestamp("2025-08-15"))
H = int(os.environ.get("HB_H", 24))   # forecast horizon in bars (default 24; HB_H=120 for the weekly-horizon study)
WINS = (24, 120, 480)
OUT = Path(__file__).with_name("hourly_baselines.csv")


def load():
    d = pd.read_csv(ROOT / "data" / "copper_hgf_1h.csv", parse_dates=["timestamp"]).set_index("timestamp")
    r = np.log(d.close).diff()
    r[d.index.to_series().diff() > pd.Timedelta("2h")] = np.nan
    return d, r


def build(r):
    v = r ** 2
    X = pd.DataFrame(index=r.index)
    for w in WINS:
        X[f"lv{w}"] = np.log(v.rolling(w, min_periods=int(0.8 * w)).mean() + 1e-10)
    fwd = v[::-1].rolling(H, min_periods=int(0.8 * H)).mean()[::-1].shift(-1)
    y_vol = np.log(fwd + 1e-10)
    R = pd.DataFrame(index=r.index)
    for k in range(6):
        R[f"r{k}"] = r.shift(k)
    R["rm24"] = r.rolling(24, min_periods=19).mean()
    R["am24"] = r.abs().rolling(24, min_periods=19).mean()
    y_ret = r.shift(-1)
    hod = pd.get_dummies(r.index.hour, prefix="h", dtype=float)
    hod.index = r.index
    return X, y_vol, R, y_ret, hod


def ridge_fit(Z, y, lam):
    mu, sd = Z.mean(0), Z.std(0).replace(0, 1)
    Zs = (Z - mu) / sd
    A = Zs.values.T @ Zs.values + lam * np.eye(Zs.shape[1])
    b = np.linalg.solve(A, Zs.values.T @ (y - y.mean()))
    return lambda Zn: ((Zn - mu) / sd).values @ b + y.mean()


def walk_forward(Z, y, lam, months):
    pred = pd.Series(np.nan, index=Z.index)
    ok = Z.notna().all(axis=1) & y.notna()
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = ok & (Z.index <= m0 - pd.Timedelta(hours=30))
        te = (Z.index >= m0) & (Z.index < m1) & Z.notna().all(axis=1)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        pred[te] = ridge_fit(Z[tr], y[tr], lam)(Z[te])
    return pred


def hac_t(d, lag):
    res = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(res.tvalues[0])


def qlike(var_pred, var_true):
    ratio = var_true / var_pred
    return float(np.mean(ratio - np.log(ratio) - 1))


def main():
    d, r = load()
    X, y_vol, R, y_ret, hod = build(r)
    months = pd.date_range(TEST_START, d.index.max(), freq="MS")
    rows = []

    shock_origin = (d.index >= SHOCK[0] - pd.Timedelta(hours=H + 1)) & (d.index <= SHOCK[1])
    windows = {"all": np.ones(len(d), bool), "ex_shock": ~shock_origin}

    # ---- returns ----
    Rz = R.copy()
    best = None
    for lam in (10, 100, 1000, 10000):
        p = walk_forward(Rz, y_ret, lam, pd.date_range("2025-01-01", TEST_START - pd.offsets.MonthBegin(1), freq="MS"))
        m = p.notna() & y_ret.notna()
        if m.sum() < 500:
            continue
        mse = float(((y_ret[m] - p[m]) ** 2).mean())
        best = (lam, mse) if best is None or mse < best[1] else best
    lam_ret = best[0] if best else 1000
    p_ret = walk_forward(Rz, y_ret, lam_ret, months)
    for name, w in windows.items():
        m = p_ret.notna() & y_ret.notna() & (d.index >= TEST_START) & w
        e0, e1 = y_ret[m], y_ret[m] - p_ret[m]
        r2 = 1 - float((e1 ** 2).sum() / (e0 ** 2).sum())
        cw = (e0 ** 2 - e1 ** 2 + p_ret[m] ** 2)
        rows.append({"task": "ret_1h", "window": name, "model": f"ridge(lam={lam_ret})",
                     "n": int(m.sum()), "R2_OS_vs_RW": r2, "CW_t": hac_t(cw, 5)})

    # ---- volatility ----
    har_cols = [f"lv{w}" for w in WINS]
    models = {
        "naive_24": None,
        "HAR": walk_forward(X[har_cols], y_vol, 1.0, months),
        "HAR+hod": walk_forward(pd.concat([X[har_cols], hod], axis=1), y_vol, 10.0, months),
    }
    # bias correction for QLIKE: log-normal factor from pre-test HAR residuals
    pre = walk_forward(X[har_cols], y_vol, 1.0, pd.date_range("2025-01-01", TEST_START - pd.offsets.MonthBegin(1), freq="MS"))
    mpre = pre.notna() & y_vol.notna()
    s2 = float(((y_vol[mpre] - pre[mpre]) ** 2).mean())
    models["naive_24"] = X["lv24"]
    for name, w in windows.items():
        mask = (d.index >= TEST_START) & w
        for k, p in models.items():
            mask &= p.notna().values
        mask &= y_vol.notna().values
        idx = d.index[mask]
        yt = y_vol[idx]
        har_p = models["HAR"][idx]
        for k, p in models.items():
            pp = p[idx]
            mse = float(((yt - pp) ** 2).mean())
            mse_har = float(((yt - har_p) ** 2).mean())
            ql = qlike(np.exp(pp + 0.5 * s2), np.exp(yt))
            ql_har = qlike(np.exp(har_p + 0.5 * s2), np.exp(yt))
            dm = hac_t((yt - har_p) ** 2 - (yt - pp) ** 2, 2 * H) if k != "HAR" else np.nan
            rows.append({"task": "vol_24bar", "window": name, "model": k, "n": len(idx),
                         "MSE_logvar_rel_HAR": mse / mse_har, "QLIKE_rel_HAR": ql / ql_har,
                         "DM_t_vs_HAR(pos=better)": dm})
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(f"bars={len(d)} test_start={TEST_START.date()} ridge_lam_ret={lam_ret} resid_var={s2:.3f}")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
