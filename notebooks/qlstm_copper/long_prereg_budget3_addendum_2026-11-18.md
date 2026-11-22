# Addendum to long_prereg_budget3.json, 2026-11-18 (written BEFORE any budget3 result was seen)

The first launch (17:32) was stopped at 17:3x after the asset tables were built and before any protocol, Shapley value, null or
report line was printed (budget3_report.txt from that launch contains only table-building and warning lines; budget_out/budget3.json
was never written). Reason: overflow warnings showed that budget3_lib.py treated the saved arrays as log forecasts, while v2_run.py
saves UNCALIBRATED variance LEVELS. A data-quality check (ranges only, no losses) confirmed this.

Correction (budget3_lib.py; the registered version is kept as budget3_lib_registered_copy.py):
- each forecast is converted as v2_report.py does: scale c = mean(exp(y)/level) on valid validation rows (at least 200), forecast =
  log(c * level). For QLSTMT and LSTM each 3-seed subset mean is calibrated the same way.
- a model is dropped for an asset if its test forecast is non-finite or non-positive, or any test loss is non-finite (recorded).
  Result of this rule: only QKernel on SI_F is dropped.
- QMODEL options and reference candidates are restricted to the models kept for that asset.
Everything else (factors, protocols, statistics, null, decision rules, seeds) is unchanged.

## Results and readings, written AFTER the run (18 Nov 2026)

Registered outputs (budget3_report.txt):
- Part A control: QLSTMT vs validation-best classical 0.9280 (apparent advantage +7.20%).
- Best: SEED +14.94%, WINDOW (quarter) +33.04%, ASSET +68.11%, LOSS +13.84%, QMODEL +12.92%, ALL5 +85.96%, ALL6 +92.51%.
- Shapley shares: ASSET 40.7, REF 29.5, WINDOW 21.8, LOSS 3.4, SEED 3.1, QMODEL 1.4.
- Registered replication rule: **DOES NOT REPLICATE**. ALL5 > 5% holds, but the top-2 factors are ASSET and REF, not LOSS or WINDOW.
- Month-block null: SEED and ASSET beyond chance; WINDOW and LOSS within.
- Part B, copper typical (random option): control -3.49%, ALL5 -2.64%, ALL6 +0.42%. A researcher who does not select on test
  results gains nearly nothing.

Problems found after the run:
1. Two quantum kernel models (QKernel, QKernelPS) give extreme forecasts on some assets. QLIKE stays finite, but their losses are
   astronomically large. This makes the QMODEL 'typical' value (-2.9 million %) and the nulls for QMODEL and ALL5 (null median
   40,031 log points) meaningless: swapping such a forecast into the reference makes the reference's loss explode. **These four
   cells are not interpretable and must not be reported as results.**
2. The control ratio 0.928 is a geometric mean dominated by two assets. On PA_F the validation-selected reference (GARCH11_t)
   collapses on test (QLSTMT/ref 0.319); on PL_F it is 0.674. Post-hoc, descriptive: QLSTMT is worse than its reference on 7 of
   11 assets; QLSTMT/LSTM geometric mean 0.940. Reference selection on a short validation segment (about 870 rows) is itself
   fragile.
