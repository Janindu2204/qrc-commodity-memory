"""Corrected generator for a possible second regime-map registration (NOT registered; written 2026-11-18 after the defect in rmap_lib.py was found).
Defect in rmap_lib.py: the planted signal fed back into x, so in many datasets x drifted to a new level, the teacher's input was clipped at 0 or 1,
and g became a constant (no learnable signal). Fix: no feedback, and the signal is made purely nonlinear.
  input  x_t : the linear process of rmap_lib (a = 0), so its distribution is exactly the pilot's (scaling constants are exact).
  target y_{t+1} = x_{t+1} + a * g_t,  g_t = teacher(x_{<=t}) with its linear projection on [1, x_t, mean10, mean50] removed, scaled to unit variance
                 (constants from the same path; the path is generated once, so they are exact, not estimated on another run).
  share s: a = sqrt(s / (1 - s)), so Var(a g) / Var(residual of the true linear forecast) = s exactly in-sample.
"""
import numpy as np
import rmap_lib as R


def dataset(memory, struct, share, r):
    x, _, graw = R.simulate(memory, struct, 0.0, r); lo, hi = np.quantile(x, [0.001, 0.999]); sd = x.std()
    x2, _, graw = R.simulate(memory, struct, 0.0, r, scale=(lo, hi), sd=sd); assert np.array_equal(x, x2)
    n = len(x); m10 = np.array([x[max(0, t - 9):t + 1].mean() for t in range(n)]); m50 = np.array([x[max(0, t - 49):t + 1].mean() for t in range(n)])
    Hx = np.column_stack([np.ones(n), x, m10, m50]); gp = graw - Hx @ np.linalg.lstsq(Hx, graw, rcond=None)[0]; g = gp / (gp.std() + 1e-12)
    a = float(np.sqrt(share / (1 - share))); y = np.full(n, np.nan); y[1:] = x[1:] + a * g[:-1]
    return x, y, g, a, dict(scale=(float(lo), float(hi)), sd=float(sd))
