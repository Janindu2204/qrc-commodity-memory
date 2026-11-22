# Addendum to long_prereg_mech.json, 2026-11-18 (written AFTER the run)

Registered verdicts (mech_report.txt): on all three copper grids K* lies beyond DOM's and ISING's memory (H90 5 and 8 steps;
K* 200 at F and M, 50 at H): "their memory is too short". Short-lag degree-2 structure at F and M, not at H. At least 1% of
HAR's error is predictable from lagged ln RV on every grid (best ratio F 0.983, M 0.934, H 0.957, test 2023-2025).

DOMG1 rule, registered verdict: "DOMG1 has memory on some seed" (max MC = 1.00). Post-hoc clarification (not a registered
reading): the memory function includes lag 0, i.e. the CURRENT input. Split per seed from mech_out/cap_DOMG1.json: 7 seeds
MF(0) = 0 and memory of past inputs 0; 3 seeds MF(0) = 1.0 and memory of past inputs 0.001. So DOMG1 carries no memory of
past inputs on any seed; the registered threshold was wrong to include lag 0. The paper reports both, labelled.

Note for the paper (post-hoc, descriptive): a ridge regression on the last 50-200 values of ln RV reduces HAR's error by 4.3%
(H) and 6.6% (M) on 2023-2025, more than any model in the comparison. HAR with hour-of-day, as specified, leaves lag structure
at about one to two days unused. This affects how the paper describes its benchmark; it is not a quantum result.
