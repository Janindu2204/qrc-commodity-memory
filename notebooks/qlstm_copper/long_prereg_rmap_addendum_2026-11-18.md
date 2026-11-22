# Addendum to long_prereg_rmap.json, 2026-11-18 (written AFTER the run; the registration file is not edited)

Result: the registered pipeline check FAILED, so by the registered rule the map is **W0 inconclusive** and no headline is drawn.
rmap_report.txt: no Q1 cell in any category other than "no detectable difference"; the teacher's own circuit did not beat the best
classical model detectably in any quantum-generated cell.

Cause (found after the run, a defect in rmap_lib.py written by the assistant): the planted signal fed back into x. In many datasets
x drifted to a new level (e.g. QDOM long share 0.15 rep 6: x from 21 to 32, while the scaling quantiles were about -4 to 4), the
teacher's input was clipped at 0 or 1, and g became a constant. The realised learnable signal (oracle vs HAR on test) was near
zero in many datasets (QDOM long share >= 0.15: 7 of 20 datasets with ratio >= 0.999; POLY3 long: all 20). The single pilot
dataset happened to be one where the drift did not occur, so the pilot did not catch it.

Consequence: the map does not answer its question. The copper placement lines ("no detectable difference" in copper-like cells)
are not interpretable, because those cells mostly held no learnable signal. Nothing from this run may be used as evidence about
quantum reservoirs.

A corrected generator (rmap2_lib.py: no feedback, signal made purely nonlinear) was written and checked on the generator only
(realised share 0.09 to 0.21 for a target of 0.15; no student model run). Running it needs a NEW registration and the student's
decision, because the stop rule of long_prereg_rmap.json forbids further runs on this map after results.
