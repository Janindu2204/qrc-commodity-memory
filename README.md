# Quantum reservoirs and intraday commodity volatility: code, registrations and results

[![DOI](https://zenodo.org/badge/1467702665.svg)](https://doi.org/10.5281/zenodo.24444165)

Archived release v1.0: doi:10.5281/zenodo.24444166 (all versions: doi:10.5281/zenodo.24444165).

Code, pre-registrations, aggregate results and manuscript for the paper:

> S. J. I. Sandanayaka, M. Edirisooriya and A. Aponso, "Why Quantum Reservoirs Do Not Improve Intraday Commodity Volatility
> Forecasts: A Memory Mismatch Confirmed in a Publicly Pre-Registered Held-Out Test" (manuscript, 2026).

The held-out test on gold and silver was registered publicly on OSF before the data were bought:
doi:10.17605/OSF.IO/YJ6W4. Its registration lists SHA-256 hashes of the scripts in this repository.

## Folders

| Folder | What it holds |
|---|---|
| `paper/` | The manuscript (`main.tex`, `main.docx`, `main.pdf`), figures (`figs/`) and the IEEE Access template files |
| `notebooks/qlstm_copper/` | All analysis code, the registrations (`long_prereg_*.json`, `gs_registration`, `HOURLY_DATABENTO_PROTOCOL_v5.json`, `spec_curve_list.json`) with their `.sha256` files, dated addenda (`*_addendum_*.md`) and the aggregate reports used in the paper (`*_report*.txt/.csv`, `budget_out/`, `mech_out/`) |
| `notebooks/qlstm_copper/hw/` | The IBM Quantum hardware test on daily copper: registration, scripts and the measured job results |
| `src/data/` | Scripts that download and rebuild every data set |
| `data/` | No data files; see `data/README.md` |
| `notes/` | The pitfall-prevalence coding sheet (52 papers) and the theory note hashed by the `deflate` registration |

## Which registration produced which result

| Paper section | Registration(s) | Report(s) |
|---|---|---|
| Copper ladder, three tiers (5, 15, 60 min) | `long_prereg_dom`, `dom2`, `dom3`, `dom4` | `dom*_report_*.txt/.csv` |
| 200-lag linear benchmark | `long_prereg_lin` | `lin_report*.txt/.csv` |
| Memory mechanism | `long_prereg_mech` | `mech_report.txt`, `mech_out/` |
| Synthetic positive control | `long_prereg_rmap`, `rmap2` (the first run was invalid) | `rmap*_report.*` |
| Held-out gold and silver | `gs_registration` (public on OSF) | `gs_report.txt`, `long_preds_gs/gs_report.json`, `gs_results_for_osf_2026-11-20.md` |
| Daily copper, angle-encoded reservoir | `long_prereg_qrc`, `surrogate`, `smalln`, `mvtrig`, `long_prereg_v2` (quantum LSTM) | `qrc_report_*.txt`, `sur_report.csv`, `smalln_report.csv`, `qrc_mvtrig_report.csv`, `long_report_*_m.csv` |
| Intraday angle-encoded reservoir | `long_prereg_hf`, `hf2`, `hf_addendum`; daily Databento: `long_prereg_dbdaily` | `hf*_report_*.txt`, `db_report_range.txt` |
| IBM hardware, daily copper 2024 | `hw/hw_prereg_daily2024` (+ addendum 1) | `hw/score_hw_daily2024.py` output |
| Evaluation choices, null, second panel | `long_prereg_budget`, `budget2`, `budget3` (+ a1), `deflate`, `deflate2` | `budget*_report.txt`, `budget_out/`, `deflate*_report.txt` |
| Seed lottery (hourly quantum LSTM, 50 seeds) | `HOURLY_DATABENTO_PROTOCOL_v5` | `hourly_databento_results_v5A*.csv`, `budget_out/seed_lottery.json` |
| Specification curve | `spec_curve_list` | `spec_curve_summary.txt`, `spec_curve.csv`, `spec_curve.png` |

The figures are made by `notebooks/qlstm_copper/make_copper_paper_figures.py` from the saved reports only (its paths were
changed to this repository's layout; it is not a registered file). The specification-curve figure is `spec_curve.png`.

## Check the registrations

    ./verify_registrations.sh

### Known hash exceptions (both disclosed in the dated addenda)

- `long_prereg_budget3.sha256`: `budget3_lib.py` was corrected after registration because it read saved forecasts in the
  wrong units; the registered version is kept byte-identical as `budget3_lib_registered_copy.py` (it matches the hash).
  The correction is described in `long_prereg_budget3_addendum_2026-11-18.md`.
- `long_prereg_budget3_a1.sha256`: the addendum file `long_prereg_budget3_addendum_2026-11-18.md` was edited nine minutes
  after its hash was written (local file times). The hashed version was not kept.
- `HOURLY_DATABENTO_PROTOCOL_v5.sha256` holds a bare hash of the matching `.json`; the script checks it directly.
- `long_prereg_v2.json` has no hash file.

All hashes are local: apart from the OSF registration, no registration has an independent time stamp.

## Reproduce

1. `pip install -r requirements.txt` (Python 3.12).
2. Rebuild the data (`data/README.md`). The intraday data need a paid Databento licence.
3. Each registration names its run and report scripts and job lists; the `run_*.sh` files run them in order. A fresh run
   writes prediction arrays to `notebooks/qlstm_copper/long_preds_*` (ignored by git).

## Data licences

No market data are redistributed. The minute-level COMEX copper, gold and silver bars are licensed from Databento and may not
be shared; prediction arrays derived from them are therefore not included either, only aggregate reports. Daily series come
from the Yahoo Finance chart interface, whose terms do not clearly allow redistribution. The hardware job files in
`notebooks/qlstm_copper/hw/` hold measured expectation values for the 2024 daily test rows.
