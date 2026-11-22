# Addendum to long_prereg_deflate2.json, 2026-11-18 (written AFTER the run)

Registered verdicts (deflate2_report.txt):
- **H2 REPLICATED.** Spearman across choices: smalln_tuned +0.95, smalln_fixed +1.00, roll_daily +1.00, mc_daily +1.00
  (4 of 4 at or above 0.6).
- **H1 re-test NOT SUPPORTED.** 2 of 6 cells pass.

Together with long_prereg_deflate.json, H2 holds in 7 of 7 out-of-sample datasets (Spearman 0.95 to 1.00). These cover five
quantum model families: angle-encoded QRC (fixed and adaptive), quantum kernels, trained QLSTM, Ising-memory reservoir and
training-size variants of QRC. H1 holds in 18 of 29 out-of-sample non-seed cells (62%).

New magnitudes:
- Choosing the training size manufactures 3.6 log points (tuned arm) and 58.9 (fixed arm).
- Choosing the test year in daily copper manufactures 10.2 to 11.8.
- Choosing the asset in the daily multi-commodity study manufactures 58.6.

Reading: the spread of option results predicts WHICH choice manufactures most, reliably; it does not predict HOW MUCH,
because single extreme options (one crisis year, one blown-up training size) make the minimum far lower than a normal model
allows. The Samuelson bound (min + s (N-1)/sqrt(N) >= mean of options) always holds, but it bounds the centre of the options,
not the honest result: bound minus honest is negative in 19 of 45 cells.

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
