# Addendum to long_prereg_budget.json, 2026-11-18 (written AFTER the run; registration not edited)

Registered verdicts (budget_report.txt): control ratio 1.0349 (DOM 3.5% worse than the past-selected classical model, which was
LSTM or GRU in 29 of 36 grid-years); each single selection pitfall "within what selection alone produces"; all five selection
pitfalls "exceeds what selection alone produces" (observed +19.84% vs null 95th +19.81%; a margin that B = 200 cannot resolve);
all six pitfalls +6.76% apparent advantage; Shapley shares LOSS 55.6%, WINDOW 43.8%, GRID 17.5%, SEED 8.5%, QMODEL 4.4%, REF -29.8%.

Flaw in the registered single-pitfall rule (found after the run): it compares the LEVEL of the apparent advantage with the null,
but on real data DOM is truly 3.5% worse, while the null is centred at no difference (null control ratio median 1.0001). So the
level comparison is "within" almost by construction. The meaningful quantity is the GAIN a pitfall adds over the control.

Post-hoc (not registered), gain over the control in log points, observed vs the week-swap null (B = 200):
SEED 0.11 (null median -0.03, 95th -0.01); WINDOW 3.09 (1.09, 1.86); GRID 1.15 (0.13, 0.32); LOSS 4.80 (0.12, 3.45);
QMODEL 0.18 (0.01, 0.05); all five 25.54 (4.51, 22.17). Every observed gain is above every null replicate.
Reading: the pitfalls exploit REAL variation in relative performance across years, losses and grids, which is larger than the
week-swap null allows (the swap averages out the models' systematic differences within each year). So the null is too narrow to
price these pitfalls; the paper must say this and report the observed swings as the main result, with the null as a lower bound.

REF has a negative Shapley share: with a fixed weak reference (ESN28u) there is less year-to-year variation in the reference for
the other selection pitfalls to exploit than with the past-selected reference, which changes identity across years.
