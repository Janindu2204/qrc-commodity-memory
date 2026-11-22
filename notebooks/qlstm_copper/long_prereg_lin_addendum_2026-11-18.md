# Addendum to long_prereg_lin.json, 2026-11-18 (written AFTER the run)

Blocks H 2013 and H 2014 stopped at the registered guard `assert pos.min() >= K - 1`: the hourly series starts in mid-2010 and
those blocks have fewer than 200 earlier bins. Re-running unchanged would fail the same way. The registered report (lin_report.py)
therefore stopped (lin_report.txt: FileNotFoundError). lin_report_a1.py is lin_report.py with ONE change (diff in chat log):
hourly rolling years are 2015-2025 instead of 2013-2025; output lin_report_a1.txt / .csv. F and M are unchanged (2013-2025).

Registered decision lines (lin_report_a1.txt):
- LIN200 vs HAR: meaningful edge at M (0.955 rolling, 0.951 held-out); detectable, below band at F (0.988) and H (0.974).
- No model has a meaningful edge over LIN200 on any rolling grid. Held-out: LSTM at M (0.951) and H (0.947).
- Quantum reservoirs vs LIN200: detectably worse at M (about 1.028) and H rolling (1.015-1.018); at F, DOM no detectable
  difference, ISING detectable below band (0.9986).
- Only the five-channel LSTM/GRU beat LIN200 detectably on M and F.
