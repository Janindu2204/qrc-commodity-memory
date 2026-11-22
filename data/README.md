# Data

No data files are committed. Rebuild them into this folder with the scripts in `src/data/` (run from the repository root).

| Data set | Used for | Source | Script |
|---|---|---|---|
| `data/databento/` minute bars, COMEX copper 2013-2026 | Intraday ladder (5, 15, 60 min) | Databento, GLBX.MDP3, licensed | `src/data/fetch_databento_copper.py` |
| `data/hf_GC/`, `data/hf_SI/` minute bars, gold and silver | Held-out test | Databento, licensed | `src/data/fetch_databento_metal.py` (registered), `fetch_databento_metal_a1.py` (addendum 1) |
| `data/hf/` 5-, 15-, 60-minute grids | All intraday analyses | derived | `notebooks/qlstm_copper/hf_build.py`, `hf2_build.py`, `gs_hf_build.py`, `gs_hf2_build.py` |
| `data/daily_databento/` | Daily Databento target | derived from the minute bars | `notebooks/qlstm_copper/db_build.py` |
| `data/daily_long/` daily copper (HG=F), crude oil, VIX, dollar index | Daily copper tests | Yahoo Finance chart interface | `python src/data/fetch_daily_long.py HG=F CL=F ^VIX DX-Y.NYB`, then `python src/data/flag_copper_long.py v2` |

The Databento scripts read the key from the `DATABENTO_API_KEY` environment variable and refuse to run without it. They take a
cost cap so a download cannot spend more than intended. Never commit a key.
