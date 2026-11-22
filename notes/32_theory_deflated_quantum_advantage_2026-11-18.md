# Theory note: how much advantage does picking the best option manufacture? (2026-11-18)

Written by the AI assistant before any new computation; the student must check every step. Registered as part of
`long_prereg_deflate.json`.

## Set-up

A researcher compares a quantum model Q with a classical reference C. One evaluation choice (the test window, the asset or
sampling frequency, the random seeds, the loss, the quantum model) has N options. Under option o the result is the log loss ratio

    r_o = log( sum of Q's losses / sum of C's losses )   under option o      (r_o < 0 favours Q)

The honest (sound) result is r_c: the pooled ratio for windows and units, the full seed mean for seeds, or the option fixed in
advance for the loss and the quantum model. A researcher who reports the best option reports min_o r_o. The manufactured gain is

    G = r_c - min_o r_o.

## Rule (the deflation)

Treat the N option results as draws around a common centre with cross-option standard deviation s, the sample standard
deviation of r_1..r_N. For N exchangeable normal draws, E[centre - min] = s* x E[M_N], where E[M_N] is the expected maximum of N
independent standard normals and s* is the per-draw spread of the deviations from the common part.

- **Equicorrelated draws.** If the draws share a common component (correlation rho), the common part cancels in
  centre - min, and the sample s across options already estimates s* = sigma*sqrt(1-rho). So the same formula holds with the
  observed s and the nominal N. This is why no separate "effective N" is needed when s is measured across the options
  themselves.
- **Prediction:**

      G_pred = s x E[M_N].

- **Deflated result:** a reported best-option ratio exp(min r) is deflated to

      exp(min_o r_o + s x E[M_N]),

  which estimates the honest result exp(r_c).

E[M_N] for reference (exact, numerical integration):

| N | 2 | 3 | 4 | 5 | 10 | 12 | 17 | 20 | 50 | 100 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E[M_N] | 0.564 | 0.846 | 1.029 | 1.163 | 1.539 | 1.629 | 1.794 | 1.867 | 2.249 | 2.508 | 2.572 |

Rule of thumb for reviewers: with 12 test years whose yearly log ratios vary with standard deviation 0.03, picking the best
year manufactures about 1.63 x 3% = 4.9% of apparent advantage.

**Known and not claimed as new:** this is the expected-maximum logic behind the deflated Sharpe ratio (Bailey and Lopez de
Prado) and data-snooping corrections (White; Hansen). What is new is applying it to quantum-versus-classical forecast claims,
choice by choice, and testing it on several datasets and quantum model families.

## Hypotheses (rules fixed in `long_prereg_deflate.json`)

- **H1 (deflation recovers the honest result).** In a cell (one dataset, one unit where applicable, one choice with N >= 4
  options), the deflated best-option result is within tolerance of the honest result:
  |(min r + s E[M_N]) - r_c| <= max(0.01, 0.3 x s E[M_N]) in log units.
- **H2 (ranking).** Within a dataset, the ranking of choices by G_pred matches the ranking by the observed G.
- **H1a (accuracy of the rule under exchangeability).** Under a sign-flip null (each option's r_o flipped with probability
  1/2, as swapping Q and C does), the median of G is close to rms(r) x E[M_N]. This is a check of the approximation, reported
  descriptively.

## Why it may fail

- **Heavy tails.** One blown-up option (as the quantum kernels on the commodity panel) makes the minimum far below what s
  predicts, or inflates s.
- **Small N.** With N = 4 or 5, the sample s is very noisy.
- **Asymmetric control.** Where the honest result is not near the centre of the options (for example, a pooled window ratio
  dominated by one long, volatile year), the deflated value aims at the centre, not at r_c.
