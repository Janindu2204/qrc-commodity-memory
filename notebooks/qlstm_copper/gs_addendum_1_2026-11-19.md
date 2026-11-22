# Addendum 1 to OSF registration yj6w4 (gold/silver held-out test), 2026-11-19

Written after the registration (Nov 19, 2026) and BEFORE any gold or silver model was run or any result was seen.

The registered download script `src/data/fetch_databento_metal.py` (sha256 241df2cc...) could not run: its opening docstring
was missing the closing triple quote, so Python stopped with a SyntaxError before any request was sent. Nothing was downloaded
by it and no credit was spent by it.

The data were downloaded with `src/data/fetch_databento_metal_a1.py`, which is the registered file plus ONE added line (the
closing `"""` as line 2). `diff` of the two files shows only that line. The request itself is unchanged: GLBX.MDP3, GC.v.0 and
SI.v.0, ohlcv-1m, 2010-06-07 to 2026-10-17. The registered file is left unchanged, so the hashes in gs_registration.sha256
(which the launcher run_gs.sh checks) still match; the download step is not part of run_gs.sh.

Databento warned that some days have reduced quality (first listed: 2014-06-11 to 2014-06-13, "degraded"). These days are kept,
as for copper; no filter was added.
