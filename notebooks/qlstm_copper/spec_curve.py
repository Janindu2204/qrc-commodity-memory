"""Specification curve for the copper frequency ladder (spec_curve_list.json). Usage: python spec_curve.py selftest | run
Reads SAVED forecasts only; fits no model. Writes spec_curve.csv, spec_curve_summary.txt, spec_curve.png.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import v2_core as vc, long_core as lc

HERE = Path(__file__).parent; LAGS = {"H": 24, "M": 96, "F": 288, "D": 10}; BAND = 0.03
SRC = {"DOM": "long_preds_dom", "ESN28u": "long_preds_dom", "ISING": "long_preds_dom2", "ESN28u_t": "long_preds_dom2", "ESNq_t": "long_preds_dom3", "SAS28_t": "long_preds_dom3",
       "LSTM": "long_preds_dom3", "GRU": "long_preds_dom3", "ESNb_t": "long_preds_dom4", "SASb_t": "long_preds_dom4", "LSTM1": "long_preds_dom4", "GRU1": "long_preds_dom4"}
OLD = {"QRC": ("QRC", "QRC_angle"), "ESN_s": ("ESN", "ESN"), "RandFeat": ("RandFeat", "RandFeat"), "TrigFeat": ("TrigFeat", "TrigFeat")}     # name -> (intraday key, daily file)
REFS1 = {"matched reservoir or stand-in": ["ESN28u", "ESN28u_t", "ESNq_t", "SAS28_t", "ESNb_t", "SASb_t"], "network, same single input": ["LSTM1", "GRU1"], "network, more inputs": ["LSTM", "GRU"], "benchmark": ["HAR"]}
REFSQ = {"matched reservoir or stand-in": ["ESN_s", "RandFeat", "TrigFeat"], "network, more inputs": ["LSTM", "GRU"], "benchmark": ["HAR"]}
PAIRS = [(q, r, c) for q in ("DOM", "ISING") for c, rs in REFS1.items() for r in rs] + [("QRC", r, c) for c, rs in REFSQ.items() for r in rs]
assert len(PAIRS) == 28


def nw_ratio(ea, eb, lag):
    """ratio of mean losses a/b with a 95% interval from the Newey-West (Bartlett) variance of the mean of eb - ea (no small-sample correction, as statsmodels HAC)"""
    d = eb - ea; n = len(d); md = d.mean(); x = d - md; s = x @ x / n
    for k in range(1, min(lag, n - 1) + 1): s += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    se = np.sqrt(max(s, 0) / n); ref = eb.mean(); r = 1 - md / ref; h = 1.96 * se / ref
    return r, r - h, r + h


def label(lo, hi): return "interval below 0.97" if hi < 1 - BAND else "interval below 1" if hi < 1 else "interval above 1" if lo > 1 else "interval contains 1"


def intraday(g, years):
    E = {}; yy, vol = [], {}
    def add(m, k, v): E.setdefault((m, k), []).append(v)
    for y in years:
        Z = {d: np.load(HERE / d / g / f"{y}.npz") for d in set(SRC.values())}; a = Z["long_preds_dom"]
        b = np.load((HERE / "long_preds_hf" / g / f"{y}.npz") if (g in "HM" and y != 2026) else (HERE / "long_preds_hf2" / "fixed" / g / f"{y}.npz"))
        for c in list(Z.values()) + [b]: assert np.array_equal(a["t"], c["t"]) and np.allclose(a["y"], c["y"]) and np.allclose(a["base"], c["base"]), ("rows differ", g, y)
        P = {m: Z[d]["p_" + m] for m, d in SRC.items()}; P.update({m: b["p_" + k[0]] for m, k in OLD.items()}); P["HAR"] = a["base"]
        for m, p in P.items(): e = a["y"] - p; add(m, "primary", e ** 2); add(m, "second", np.abs(e))
        yy += [y] * len(a["y"]); vol[y] = float(a["vol"])
    return {k: np.concatenate(v) for k, v in E.items()}, np.array(yy), vol


def daily(years, held):
    data = pd.read_csv(lc.ROOT.parent / "daily_databento" / "HG_db_1d_range.csv", parse_dates=["date"]).set_index("date"); E = {}; yy, vol = [], {}
    for u in years:
        D = lc.prepare("range_m", u, length=1, data=data); y_, va, te = D["y"], D["va"], D["te"]; yl = np.exp(y_[te])
        vol[u] = float(np.sqrt(252 * np.nanmean(D["rv"][D["d"].index.year == u].values))); yy += [u] * len(te)
        old = (HERE / "long_preds_dom" / "D_old" / str(u)) if held else (HERE / "long_preds_qrc" / "dbdaily" / "range" / str(u))
        files = {m: HERE / d / "D" / str(u) / f"{m}.npz" for m, d in SRC.items()}; files.update({m: old / f"{k[1]}.npz" for m, k in OLD.items()}); files["HAR"] = old / "HARq.npz"
        for m, f in files.items():
            z = np.load(f); v, t = z["lv_val"], z["lv_te"]; ok = np.isfinite(v) & (v > 0); assert np.isfinite(t).all() and (t > 0).all(), (m, u)
            fc = vc.calib_scale(np.exp(y_[va][ok]), v[ok]) * t
            E.setdefault((m, "primary"), []).append(lc.qlike_vec(fc, yl)); E.setdefault((m, "second"), []).append((np.log(fc) - y_[te]) ** 2)
    return {k: np.concatenate(v) for k, v in E.items()}, np.array(yy), vol


def selftest():
    import statsmodels.api as sm
    rng = np.random.default_rng(0); n = 5000; e = np.convolve(rng.normal(size=n + 20), np.ones(20) / 20, "valid")[:n]; ea = rng.chisquare(1, n) + e; eb = rng.chisquare(1, n) * 1.03 + e
    r = sm.OLS(eb - ea, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 24}); ref = eb.mean(); want = (1 - r.params[0] / ref, 1 - r.params[0] / ref - 1.96 * r.bse[0] / ref, 1 - r.params[0] / ref + 1.96 * r.bse[0] / ref)
    got = nw_ratio(ea, eb, 24); assert np.allclose(got, want, atol=1e-10), (got, want)
    assert label(0.9, 0.96) == "interval below 0.97" and label(0.98, 0.999) == "interval below 1" and label(0.99, 1.01) == "interval contains 1" and label(1.001, 1.02) == "interval above 1"
    print("selftest ok: Newey-West ratio equals statsmodels HAC", [round(x, 6) for x in got])


def reproduce(df):
    """four numbers from the registered reports must be reproduced"""
    checks = [("dom2_report_rolling.csv", "F", "rolling/all", "DOM", "ESN28u_t"), ("dom3_report_rolling.csv", "M", "rolling/all", "ISING", "LSTM"),
              ("dom4_report_rolling.csv", "D", "rolling/all", "DOM", "LSTM1"), ("dom_report_heldout.csv", "H", "heldout/all", "DOM", "ESN28u")]
    for f, g, st, a, b in checks:
        r = pd.read_csv(HERE / f); r = r[(r.grid == g) & (r.stratum == "all") & (r.model == a) & (r.ref == b)].iloc[0]
        s = df[(df.grid == g) & (df.stratum == st) & (df.quantum == a) & (df.reference == b) & (df.loss == "primary")].iloc[0]
        d = max(abs(r.ratio - s.ratio), abs(r.ci_lo - s.ci_lo), abs(r.ci_hi - s.ci_hi)); print(f"reproduce {f} {g} {a} vs {b}: report {r.ratio:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}] here {s.ratio:.4f} [{s.ci_lo:.4f},{s.ci_hi:.4f}] diff {d:.1e}")
        assert d < 2e-4, "reproduction failed"


def run():
    rows = []
    for g in ("F", "M", "H", "D"):
        for what in ("rolling", "heldout"):
            years = ([2026] if g != "D" else [2025, 2026]) if what == "heldout" else (list(range(2013, 2026)) if g != "D" else list(range(2015, 2025)))
            E, yy, vol = intraday(g, years) if g != "D" else daily(years, what == "heldout"); strata = {"all": np.ones(len(yy), bool)}
            if what == "rolling":
                order = sorted(years, key=lambda y: (vol[y], y)); cut = (4, 9) if len(years) == 13 else (3, 7)
                reg = {y: ("calm" if i < cut[0] else "intermediate" if i < cut[1] else "turbulent") for i, y in enumerate(order)}
                for s in ("calm", "intermediate", "turbulent"): strata[s] = np.isin(yy, [y for y in years if reg[y] == s])
            for st, mk in strata.items():
                for q, r, c in PAIRS:
                    for loss in ("primary", "second"):
                        ratio, lo, hi = nw_ratio(E[(q, loss)][mk], E[(r, loss)][mk], LAGS[g])
                        rows.append(dict(grid=g, period=what, stratum=f"{what}/{st}", quantum=q, reference=r, ref_class=c, loss=loss, ratio=ratio, ci_lo=lo, ci_hi=hi, label=label(lo, hi), n=int(mk.sum())))
    df = pd.DataFrame(rows); assert len(df) == 1120, len(df); reproduce(df); df.to_csv(HERE / "spec_curve.csv", index=False)
    L = ["interval below 0.97", "interval below 1", "interval contains 1", "interval above 1"]; out = []; pd.set_option("display.width", 250)
    def tab(by):
        t = df.groupby(by)["label"].value_counts().unstack(fill_value=0).reindex(columns=L, fill_value=0); t["n_spec"] = t.sum(1)
        q = df.groupby(by)["ratio"].quantile([0.05, 0.5, 0.95]).unstack(); q.columns = ["p05", "median", "p95"]; return t.join(q.round(4))
    out.append(f"Specification curve, copper frequency ladder: {len(df)} specifications (descriptive; dependent; raw 95% intervals, no multiplicity correction).\nratio below 1 favours the quantum model.\n")
    t = df["label"].value_counts().reindex(L, fill_value=0); out.append("ALL\n" + pd.DataFrame({"count": t, "share": (t / len(df)).round(3)}).to_string() + f"\nmedian {df.ratio.median():.4f}  p05 {df.ratio.quantile(.05):.4f}  p95 {df.ratio.quantile(.95):.4f}\n")
    for by in ("ref_class", "quantum", "grid", "period", "loss", ["ref_class", "grid"], ["ref_class", "period"]): out.append(f"BY {by}\n" + tab(by).to_string() + "\n")
    cols = ["grid", "stratum", "quantum", "reference", "ref_class", "loss", "ratio", "ci_lo", "ci_hi", "label", "n"]; s = df.sort_values("ratio")
    out.append("EVERY 'interval below 0.97' SPECIFICATION\n" + (s[s.label == L[0]][cols].round(4).to_string(index=False) if (s.label == L[0]).any() else "none") + "\n")
    out.append("LOWEST 15\n" + s.head(15)[cols].round(4).to_string(index=False) + "\n\nHIGHEST 15\n" + s.tail(15)[cols].round(4).to_string(index=False) + "\n")
    (HERE / "spec_curve_summary.txt").write_text("\n".join(out)); print("\n".join(out))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"matched reservoir or stand-in": "#1f77b4", "network, same single input": "#d62728", "network, more inputs": "#ff7f0e", "benchmark": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(9, 4.2)); s = s.reset_index(drop=True); x = np.arange(len(s))
    for c, k in col.items():
        m = (s.ref_class == c).values; ax.vlines(x[m], s.ci_lo[m], s.ci_hi[m], color=k, lw=0.4, alpha=0.5); ax.plot(x[m], s.ratio[m], ".", ms=2.5, color=k, label=c)
    ax.axhline(1, color="k", lw=0.8); ax.axhline(1 - BAND, color="k", lw=0.6, ls="--"); ax.axhline(1 + BAND, color="k", lw=0.6, ls="--"); ax.set_ylim(0.8, 1.25)
    ax.set_xlabel("specification (sorted by ratio)"); ax.set_ylabel("loss ratio, quantum / reference"); ax.legend(frameon=False, fontsize=8, markerscale=4); fig.tight_layout(); fig.savefig(HERE / "spec_curve.png", dpi=200)


if __name__ == "__main__":
    selftest() if sys.argv[1] == "selftest" else run()
