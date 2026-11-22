"""Parts 1-2 (long_prereg_budget.json): manufactured-advantage budget and its decomposition, from SAVED intraday forecasts only.
No model is trained. Rows: grids F, M, H; test years 2014-2025 (2013 is used only to pick the 2014 reference; 2026 is not read).

Weekly sums. For every grid, loss, model and 3-seed subset (plus the full 10-seed mean, index FULL), the loss is summed per calendar week,
so every later step is a sum over weeks. Losses: MSE and MAE on ln RV; QLIKE on the level (y = exp(ln RV), forecast = exp(pred)).

Six factors, each 'control' or 'pitfall':
  SEED   control: quantum model uses the 10-seed mean.  pitfall: the best 3-seed subset of 10 (120 subsets), chosen on the test rows.
  WINDOW control: pooled test years 2014-2025.         pitfall: the best single test year (12).
  GRID   control: geometric mean over F, M, H.         pitfall: the best grid.
  LOSS   control: MSE on ln RV.                        pitfall: the best of MSE, MAE, QLIKE (chosen per grid).
  QMODEL control: DOM.                                 pitfall: the better of DOM and ISING (chosen per grid).
  REF    control: for test year Y, the classical model (of CLASSICAL) with the lowest loss in year Y-1 on that grid and loss (past only).
         pitfall: ESN28u, the untuned 28-unit echo state network of the same size.
Apparent ratio of a protocol = the minimum, over every option the protocol leaves open, of loss(quantum) / loss(reference); per grid, then
the grid summary. Apparent advantage = 1 - ratio. The researcher is assumed to pick the option that looks best (the pitfall).
Null (selection pitfalls only): in a random half of calendar weeks the quantum model's forecasts (every seed) and the reference's
forecasts are swapped, so neither is better in distribution; the same protocol is then run. Week flips are shared across grids.
"""
import itertools, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = {"DOM": "long_preds_dom", "ESN28u": "long_preds_dom", "ISING": "long_preds_dom2", "ESN28u_t": "long_preds_dom2", "ESNq_t": "long_preds_dom3",
       "SAS28_t": "long_preds_dom3", "LSTM": "long_preds_dom3", "GRU": "long_preds_dom3", "ESNb_t": "long_preds_dom4", "SASb_t": "long_preds_dom4",
       "LSTM1": "long_preds_dom4", "GRU1": "long_preds_dom4"}
QUANT = ["DOM", "ISING"]; WEAK = "ESN28u"; CLASSICAL = ["ESN28u_t", "ESNq_t", "SAS28_t", "ESNb_t", "SASb_t", "LSTM1", "GRU1", "LSTM", "GRU"]
MODELS = QUANT + [WEAK] + CLASSICAL; GRIDS = ["F", "M", "H"]; YEARS = list(range(2013, 2026)); TEST = YEARS[1:]; LOSSES = ["MSE", "MAE", "QLIKE"]
SUBSETS = [tuple(c) for c in itertools.combinations(range(10), 3)]; FULL = len(SUBSETS); FACTORS = ["SEED", "WINDOW", "GRID", "LOSS", "QMODEL", "REF"]
WEEK_NS = 7 * 24 * 3600 * 10 ** 9


def loss(kind, y, p):
    if kind == "MSE": return (y - p) ** 2
    if kind == "MAE": return np.abs(y - p)
    r = np.exp(y - p); return r - (y - p) - 1          # QLIKE on levels: y/s2 - ln(y/s2) - 1 with y/s2 = exp(lny - lns2)


def weekly_tables(grid, years=YEARS):
    """returns W[loss][model] (nsub+1, nweeks) weekly loss sums, CNT (nweeks,), WEEKYEAR (nweeks,), WEEKID (nweeks,) calendar week ids"""
    ys, wk, seed = [], [], {m: [] for m in MODELS}
    for Y in years:
        base = None
        for m in MODELS:
            z = np.load(HERE / SRC[m] / grid / f"{Y}.npz")
            if base is None: base = z["y"]; ys.append(z["y"]); wk.append(z["t"] // WEEK_NS)
            else: assert np.array_equal(base, z["y"]), (grid, Y, m)
            seed[m].append(z[f"seedp_{m}"].astype(np.float64))
    y = np.concatenate(ys); w = np.concatenate(wk); S = {m: np.concatenate(v, 1) for m, v in seed.items()}
    ids, inv = np.unique(w, return_inverse=True); nw = len(ids); cnt = np.bincount(inv, minlength=nw).astype(float)
    yr_of = np.concatenate([np.full(len(a), Y) for a, Y in zip(ys, years)]); wy = np.zeros(nw, int); wy[inv] = yr_of   # a week straddling New Year takes the later year
    W = {L: {} for L in LOSSES}
    for m in MODELS:
        full = S[m].mean(0)
        for L in LOSSES:
            T = np.empty((FULL + 1, nw))
            T[FULL] = np.bincount(inv, loss(L, y, full), minlength=nw)
            for k, sub in enumerate(SUBSETS): T[k] = np.bincount(inv, loss(L, y, S[m][list(sub)].mean(0)), minlength=nw)
            W[L][m] = T
    return W, cnt, wy, ids


def ratios_grid(W, wy, flip, ref_of_year, f):
    """per-grid apparent ratio for protocol f (dict factor -> bool pitfall). flip (nweeks,) bool or None. ref_of_year[L][Y] = reference model name."""
    best = np.inf
    for q in (QUANT if f["QMODEL"] else ["DOM"]):
        for L in (LOSSES if f["LOSS"] else ["MSE"]):
            refs = {Y: (WEAK if f["REF"] else ref_of_year[L][Y]) for Y in TEST}
            subs = range(FULL) if f["SEED"] else [FULL]
            # weekly quantum and reference sums under the swap
            Qw = np.empty((len(subs), len(wy))); Rw = np.empty(len(wy))
            for Y in TEST:
                c = wy == Y; r = refs[Y]; fl = flip[c] if flip is not None else np.zeros(c.sum(), bool)
                qa = W[L][q][list(subs)][:, c]; ra = W[L][r][list(subs)][:, c]                # reference's own subsets are used only for swapped weeks
                Qw[:, c] = np.where(fl, ra, qa); Rw[c] = np.where(fl, W[L][q][FULL][c], W[L][r][FULL][c])
            yrs = [[Y] for Y in TEST] if f["WINDOW"] else [TEST]
            for ys_ in yrs:
                c = np.isin(wy, ys_); rr = Qw[:, c].sum(1) / Rw[c].sum(); best = min(best, float(rr.min()))
    return best


def protocol_ratio(tabs, flips, refsel, f):
    per = [ratios_grid(tabs[g][0], tabs[g][2], None if flips is None else flips[g], refsel[g], f) for g in GRIDS]
    return float(min(per)) if f["GRID"] else float(np.exp(np.mean(np.log(per))))


def ref_selection(W, wy):
    """past-only reference: for test year Y and loss L, the classical model with the lowest year Y-1 loss (10-seed mean)"""
    out = {}
    for L in LOSSES:
        out[L] = {}
        for Y in TEST:
            c = wy == Y - 1; out[L][Y] = min(CLASSICAL, key=lambda m: W[L][m][FULL][c].sum())
    return out


def all_protocols():
    return [dict(zip(FACTORS, bits)) for bits in itertools.product([False, True], repeat=len(FACTORS))]


def shapley(values):
    """values: dict tuple(bits) -> log ratio. Returns per-factor Shapley value of the log ratio (negative = adds apparent advantage)."""
    n = len(FACTORS); phi = np.zeros(n); from math import factorial
    for bits, v in values.items():
        for i in range(n):
            if bits[i]:
                off = tuple(b if j != i else False for j, b in enumerate(bits)); k = sum(bits) - 1
                phi[i] += factorial(k) * factorial(n - k - 1) / factorial(n) * (v - values[off])
    return dict(zip(FACTORS, phi))


def selftest():
    # Shapley of an additive function returns its parts; efficiency: sum = v(all) - v(none)
    rng = np.random.default_rng(0); a = rng.normal(size=6); vals = {b: float(np.dot(a, b)) for b in itertools.product([0, 1], repeat=6)}
    vals = {tuple(bool(x) for x in b): v for b, v in vals.items()}; ph = shapley(vals); assert np.allclose(list(ph.values()), a)
    assert abs(loss("QLIKE", np.log(2.0), np.log(2.0))) < 1e-12 and loss("QLIKE", 0.0, 1.0) > 0
    # min over more options can only lower the apparent ratio (monotone in each factor except REF, which is a swap, not a menu)
    return "ok"


if __name__ == "__main__":
    print("selftest", selftest())
