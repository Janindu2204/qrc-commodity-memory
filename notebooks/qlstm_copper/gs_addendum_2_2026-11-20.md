# Addendum 2 to OSF registration yj6w4 (gold/silver held-out test), 2026-11-20

Written while the registered run was in progress, before any result was opened.

1. On 20 Nov 2026 at about 02:25 UTC the run was paused at the author's request; the computer (WSL) was later restarted, which
   ended the paused jobs. No result had been opened.
2. The run was resumed at 10:57 UTC with `gs_resume.sh`, a launcher that is NOT in the registration. It checks the registered
   hashes, then runs only the registered jobs (same scripts, same arguments, from gs_jobs.txt) whose output file does not yet
   exist, then the registered mechanism probe and report. Each job writes its output once, at its end, so finished jobs were
   kept and unfinished ones (6 interrupted silver jobs and the out-of-memory jobs) run again unchanged. It runs at most 7 jobs at
   once (10 in run_gs.sh) to reduce out-of-memory kills. Models, settings, seeds and statistics are unchanged.
3. Out-of-memory kills before the pause: GC gs_dom4_run.py F 2022, GC gs_dom_run.py F 2019, SI gs_dom4_run.py F 2020; re-run
   unchanged in the resumed run. GC gs_lin_run.py H 2013 and H 2014 stopped at the 200-lag guard, as for copper; these years are
   outside the registered LIN200 hourly window (2015-2025) and are not re-run.
