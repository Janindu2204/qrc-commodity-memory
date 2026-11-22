# Addendum to long_prereg_deflate.json, 2026-11-18 (written AFTER the run)

Registered verdicts (deflate_report.txt):
- **H1 NOT SUPPORTED.** 16 of 24 out-of-sample cells pass (67%; kernels 77%, adaptive 56%, hourly_roll 50%). In-sample: 16 of
  40 (40%).
- **H2 SUPPORTED.** Spearman between predicted and observed manufactured gain across choices: kernels +1.00, adaptive +1.00,
  hourly_roll +1.00 (3 of 3 >= 0.6). In-sample: copper +0.80; panel 0.00 (blow-up quantum kernels).

Where H1 fails (reading after the run; not registered):
1. **Heavy tails.** One extreme window (e.g. adaptive NG: observed gain 120.6 log points vs predicted 61.2; kernels CL and HO)
   makes the minimum far lower than the spread predicts.
2. **Asymmetric control for seeds.** The honest result (the full 10- or 5-seed mean) is better than every 3-seed subset, so the
   centre of the options is not the honest result (hourly_roll SEED: observed -0.37, predicted +1.06).
3. **Small N.** hourly_roll WINDOW has N = 4.

By choice, out-of-sample: WINDOW 15 of 21 cells pass (71%); UNIT 1 of 2; SEED 0 of 1. In-sample: WINDOW 10 of 14, SEED 6 of 14, UNIT 0 of 1. Out-of-sample best-window gains range from 2.0 to 120.6 log points (median 16.3).

The rule must not be proposed as a general deflation tool on this evidence. Any revised rule (e.g. a robust spread) would be a
new hypothesis and must be registered and tested on data not used here.

## Correction, 2026-11-18 (found during a deliberately harsh gate run)

**H2 is close to circular and is NOT evidence.** The prediction s x E[M_N] and the observed gain are both computed from the SAME
option results. Across choices whose spreads differ by orders of magnitude, they rank the same way almost automatically.

Check: 3000 pure-noise datasets (4 choices, random spreads, option counts like ours) reach Spearman >= 0.6 in 98.5% of cases,
with median rho 1.00. The registered H2 test therefore had almost no power to fail. The "Fisher combined p = 6.1e-4" (evidence
row 57) used the wrong null (random rankings) and is withdrawn.

H2 could only be informative if the spread were estimated from data independent of the option results, for example from an
earlier period, which was not done. What remains valid:
- the descriptive magnitudes;
- the H1 negative result (the normal deflation fails under single extreme options).
