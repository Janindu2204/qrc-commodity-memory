# Results of OSF registration yj6w4 (gold/silver held-out test), 20 November 2026

The registered run finished at 17:52 UTC on 20 Nov 2026; the registered report (gs_report.py, hash unchanged) was written at
17:53 UTC. Deviations: Addendum 1 (download-script fix; foreknowledge corrected) and Addendum 2 (pause, restart, resume
launcher, out-of-memory re-runs). All three registered predictions are SUPPORTED.

## R1. No quantum edge (24 comparisons, rolling test years): SUPPORTED
0 of 24 comparisons show a meaningful edge (rule: refuted if any does).
- vs the single-input LSTM: the quantum reservoirs are detectably worse in 9 of 12 comparisons (ratios 1.0009 to 1.0054) and
  show no detectable difference in 3 (hourly; 0.9994 to 1.0010).
- vs the 200-lag linear model: detectably worse at 15 and 60 minutes in all 8 comparisons (1.0042 to 1.0199); detectable but
  below the 3% band at 5 minutes in all 4 (0.9945 to 0.9963).

## R2. Memory too short (K* > 8 in at least 5 of 6 cells): SUPPORTED
K* = 200 in five cells and 100 in one (silver, hourly); the quantum reservoirs' 90% memory lag is 8 or less.

## R3. No information beyond a same-memory classical model (at least 5 of 6 cells inside the margin): SUPPORTED
6 of 6 cells pass. A/B ratios 0.9988 to 1.0018; every 90% interval inside +/-0.25% (5-minute) or +/-0.5% (15, 60-minute).

## Sensitivity on never-touched years only (gold 2018-2019; silver 2013-2019): reported, not a decision
- R1: still no meaningful edge in any comparison.
- R3: 4 of 6 cells pass; silver 5-minute (1.0029 [1.0025, 1.0034]) and silver hourly (1.0034 [1.0000, 1.0068]) fall
  outside their margins, both in the direction of the quantum combination being WORSE.

Full output: gs_report.txt and long_preds_gs/gs_report.json.

## Held-out 2026 blocks (registered as "also run and reported, not part of the decisions")
Same comparisons and statistics as R1 on the 2026 block only (to 16 Oct 2026), file gs_heldout2026.txt:
- no meaningful quantum edge in any of 24 comparisons;
- 8 "detectably worse" (mainly 15-minute), 16 "no detectable difference" (corrected 2026-11-22; first version said 9 and 15, a miscount of gs_heldout2026.txt);
- the lowest ratio is 0.9809 (silver hourly, DOM vs LIN200, interval up to 1.0006).
