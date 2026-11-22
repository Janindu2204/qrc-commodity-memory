# Pitfall prevalence: pilot coding round 2, 2026-11-18

Status: written by an AI assistant (Claude) in one session. **One AI coder. Not checked by the student or the
supervisor.** Nothing here may enter a report until the student has read each primary source and agrees with
the code. This note extends `30d_ideas_meta_research_2026-11-18.md` (the 30d note). The full per-paper sheet,
with the page for each cell, is in `31_prevalence_pilot_coding_2026-11-18.csv` (same folder).

## 1. Headline

- **52 papers coded from full text** (44 from the 30d corpus, 8 new). 6 more were read in full and excluded. 8 are
  not coded because no full text could be obtained.
- Pitfalls you can see in the text are common. Among papers where the item can be coded (present or absent):
  no significance test, or tests without correction (P9) **85%**; one seed or no repeats (P2) **82%**; single
  train/test split (P10) **77%**; no naive, random-walk, HAR or ARIMA benchmark (P8) **76%**; level metric
  with no naive comparison (P11) **63%**.
- Pitfalls that need code to see are mostly "not reported": test-period selection (P1) 34 of 52 not reported;
  full-data scaling (P12) 33 of 52; untuned baseline (P3) 36 of 52.
- Median 4 pitfalls per paper; 43 of 52 have 3 or more. Only 3 papers have none: C26, C27, C48. All three are
  2026 null-result papers written to show good practice.

## 2. Coding sheet and decision rules

One row per paper. One comparison per paper: the headline quantum-vs-classical comparison. Four levels:
**present**, **absent**, **not_reported** (the text does not say enough; never turned into present),
**not_applicable**. The 30d note gave rules for P2, P8 and P11 only. The other nine two-line rules below were
written by this coder **before** round 2, from the pitfall definitions in 30d. They need the student's approval.

| Code | Pitfall | Present if | Absent if (else not_reported; not_applicable as stated) |
|---|---|---|---|
| P1 | Test-period selection of instances, features or hyperparameters | A model, variant, instance, feature set or hyperparameter is picked by its test score, or several are scored on test and the best is named without a validation step. | The paper states that choices were made on a validation set, inner CV or before the test period was opened. A validation split whose use is not stated is not_reported. |
| P2 | One seed, few seeds or unequal seeds (30d rule) | Fewer than 5 seeds per stochastic model, or seed counts differ between arms, or no repeat is reported. CV folds and seed ensembles inside one model are not repeats. | 5 or more independent runs for every stochastic arm. N/A only if every model is deterministic. |
| P3 | Untuned or weaker classical baseline | Classical baselines use fixed or default settings while the quantum model is tuned, or the paper admits unequal tuning effort. | A tuning procedure for the classical baselines is stated with comparable effort. N/A if no classical model is run (then P4 is present). |
| P4 | No classical twin or no size-matched classical model | No classical model at all, or no model that copies the architecture with the quantum block swapped for a classical one, and no parameter-matched model; or the sizes are reported and are far apart. | A classical twin (same pipeline, quantum block replaced) or a parameter-matched classical model is reported. A twin-like model with sizes not given and no clear twin is not_reported. |
| P5 | Unequal inputs | The quantum model gets inputs (features, lags, exogenous series, preprocessing) the classical comparator does not. | The text says all models share the same inputs, or describes one shared input pipeline. |
| P6 | Degenerate encoding | Only for quantum kernel models (QSVM, QSVR, QKRR, GP with a quantum kernel). Present if the fidelity kernel reduces to a classical kernel: product angle encoding with no entangler, or a data-independent post-unitary that cancels. | The feature map has data-dependent entangling phases (e.g. ZZ/IQP), or a kernel-swap control is reported. Feature map not described: not_reported. Non-kernel models: not_applicable. |
| P7 | Best of many specifications, no correction | The headline is the best of several quantum variants, configurations, assets or strategies scored on the same test data, with no held-out selection and no multiplicity correction. | One pre-specified model, or selection on validation with all results shown, or a multiplicity control. |
| P8 | No naive benchmark (30d rule) | For a regression target: none of random walk, persistence, last value, seasonal naive, HAR or ARIMA is run. For classification: no majority-class or chance baseline. For trading: no buy-and-hold. | One of those is run. For cross-sectional return ranking, a linear factor or ridge model counts. GARCH alone does not count (it is not in the 30d list). |
| P9 | No statistical test, or tests without multiple-testing correction | No named test for the headline comparison, or many pairwise tests with no correction. "Significant" with no named test counts as present. | One named pre-specified test, or tests with a correction (Holm, Bonferroni, MCS, family-wise control). |
| P10 | One train/test split instead of rolling origin | One hold-out split (chronological or random), or shuffled/stratified k-fold on time-ordered data. | Rolling-origin, walk-forward, expanding-window or several time-ordered folds. |
| P11 | Level metric that rewards persistence (30d rule) | The headline is R2, MAPE, RMSE or MAE on a price (or other persistent level) and no naive comparison is given. | Returns, volatility with QLIKE, or a level metric with a naive comparison. N/A for classification and trading tasks. |
| P12 | Scaling or other preprocessing fitted on full data | Scaler, PCA, smoothing, decomposition or feature selection is fitted on the full series before the split (stated, or shown by the formula). | The paper says the scaler is fitted on training data only, or the scaling is inside each input window, or cross-sectional within a day. Scaling with stage not stated: not_reported. N/A if no scaling. |

Worked examples (the 30d note asked for two per pitfall; one is given here, more are in the CSV notes):
P1 present: C21 headline R2 0.985 is the best of 1,368 configurations, validation used only on a subset.
P1 absent: C4 picks every model by mean validation AUC. P2 present: C9 reports 10 runs that all use seed 42.
P6 present: C4 Variant A, amplitude encoding plus a fixed post-unitary. P12 present: C36 z-scores with the
mean and standard deviation of the whole series (Eq. 20). P12 present: C44 applies a two-sided
Hodrick-Prescott filter before the split.

**Evidence convention.** Each present or absent cell has a page (pN, PDF page) or section in the CSV notes
column. A not_reported cell means a keyword search plus reading of the methods and results found nothing.
Paraphrase only, no long quotes.

## 3. How this was done

- Full text from: local PDFs in `Docs/papers/Best/` and `Docs/papers/` (`pdftotext`, with and without
  `-layout`); 41 arXiv PDFs downloaded to a scratch folder outside the repo; Europe PMC full text (C35); publisher
  open-access PDFs (C36, C37); Research Square PDF (C38). Open-access status checked with the OpenAlex API.
- Reading: a script pulled every sentence on seeds, splits, validation, tuning, baselines, tests, scaling,
  metrics, parameter counts and kernels, with its page. The coder then read the methods and results around
  them. This is more than the 30d keyword pass for the 23 pilot papers, but it is still not a line-by-line
  read of every page. Some not_reported cells may be wrong.
- C6 (Li et al.) is coded **from the paper text only**, like every other paper. Our own audit
  (`25_li_audit_results_2026-11-13.md`) found pitfalls the text does not show (weaker HAR, full-sample
  min-max, unequal inputs to AR/HAR). These are in the C6 notes but not in the counts. This also avoids
  counting the companion paper twice (see 30a).
- Year = year of the first public version (arXiv year for arXiv papers).
- No personal names or emails from the repo were sent to any service.

## 4. Search and flow (PRISMA-style)

**arXiv listing API**, run 2026-11-18, `https://export.arxiv.org/api/query?search_query=<string>`,
max 200 results each, sorted by submission date. Exact strings:

1. `abs:quantum AND (abs:stock OR abs:"stock market" OR abs:equity) AND (abs:forecast* OR abs:predict*) AND (abs:"neural network" OR abs:circuit OR abs:LSTM OR abs:kernel OR abs:reservoir)` (22 hits)
2. `abs:quantum AND abs:volatility AND (abs:forecast* OR abs:predict*)` (34)
3. `abs:quantum AND (abs:cryptocurrency OR abs:bitcoin) AND (abs:forecast* OR abs:predict*)` (6)
4. `abs:quantum AND (abs:commodity OR abs:"crude oil" OR abs:gold OR abs:copper OR abs:"carbon price") AND (abs:forecast* OR abs:predict*)` (135; "gold" pulls in physics)
5. `abs:quantum AND (abs:"exchange rate" OR abs:forex OR abs:currency) AND (abs:forecast* OR abs:predict*) AND (abs:"machine learning" OR abs:"neural network")` (2)
6. `abs:quantum AND abs:"electricity price" AND (abs:forecast* OR abs:predict*)` (0)
7. `abs:quantum AND (abs:inflation OR abs:macroeconomic OR abs:GDP) AND abs:forecast*` (12)
8. `abs:"quantum machine learning" AND abs:financial AND abs:"time series"` (2)
9. `abs:"quantum reservoir" AND (abs:financial OR abs:stock OR abs:market OR abs:price)` (15)
10. `(abs:QLSTM OR abs:"quantum LSTM" OR abs:"quantum long short-term memory") AND (abs:stock OR abs:financial OR abs:price OR abs:market)` (8)
11. `abs:"quantum kernel" AND (abs:stock OR abs:financial OR abs:returns OR abs:price)` (14)
12. `abs:"variational quantum" AND (abs:stock OR abs:financial OR abs:price) AND (abs:forecast* OR abs:predict*)` (8)
13. `ti:quantum AND (ti:forecasting OR ti:prediction) AND (ti:stock OR ti:financial OR ti:market OR ti:price OR ti:volatility)` (26)
14. `abs:quantum AND abs:"time series" AND abs:forecasting AND (abs:financial OR abs:stock OR abs:energy)` (28)
15. `abs:"quantum neural network" AND (abs:stock OR abs:financial OR abs:price)` (30)
16. `abs:"hybrid quantum" AND (abs:stock OR abs:financial OR abs:price OR abs:market) AND (abs:forecast* OR abs:predict*)` (16)

The low counts suggest the arXiv API handles wildcards and phrases poorly. A full review must also use
Scopus, Web of Science, IEEE Xplore and Google Scholar (not searched here).

**Web search**, 2026-11-18 (6 queries): "quantum LSTM stock price forecasting arXiv 2025 comparison classical
LSTM"; "quantum reservoir computing cryptocurrency OR exchange rate OR commodity price forecasting arXiv 2026";
"variational quantum circuit electricity price forecasting OR gold price forecasting quantum neural network open
access"; "hybrid quantum neural network Bitcoin price prediction open access 2025 OR 2026"; "MDPI quantum machine
learning stock market forecasting variational quantum circuit 2025"; "quantum exchange rate forecasting quantum
neural network OR quantum kernel arXiv 2024 2025 2026".

**Local folder screen**: PDFs in `Docs/papers/` that were not in the 30d corpus.

| Stage | Records | Result |
|---|---|---|
| Starting corpus (30d tables 2.1-2.3) | 58 | 23 pilot + 11 partly coded + 24 to code (30d said "about 56") |
| arXiv API hits | 358 (281 unique) | 33 already in corpus or 30d related list; 75 before 2019 |
| arXiv title screen | 173 | 153 excluded (physics, chemistry, materials, non-finance, fraud/credit/portfolio tasks). One of them (2303.17629) was wrongly excluded and recovered by the web search. |
| arXiv abstract screen | 20 | 15 excluded: not QML (1902.10502), optimisation (1904.05808), option pricing (2011.07319), non-temporal (2104.04049, 2306.12965, 2312.01602), classical only (2408.03594), RL with classical forecaster (2507.12835), non-price data (2508.05036 ETT, 2605.24252 load, 2606.04686 ATM cash), synthetic (2510.25183, 2512.10584, 2607.07978), generative only (2606.27561) |
| arXiv full text | 5 + 1 recovered | 2 excluded (2212.04209 housing, not a time series; 2607.24065 synthetic GP data); **4 included** (N1 2302.03244, N2 2303.17629, N4 2511.09218, N5 2602.16976) |
| Web search, new records | 8 | 4 excluded at abstract (2407.16404 multi-agent RL; 2509.12071 chaotic maps; SHS Web Conf. 2025 quantum-inspired; a 2025 master's thesis); 1 no full text (MDPI Mathematics 13(15) 2505, hidden quantum Markov model, HTTP 403); 3 read in full: **1 included** (N3 2508.04048), 2 excluded (2605.04991 electricity load; Alaminos et al., Physica A 654 (2024) 130140, evaluated on agent-based simulated scenarios with a qubit-neuron model) |
| Local folder | 8 | **3 included** (N6, N7, N8); 5 excluded (four self-described quantum-inspired or "quantum-driven optimisation" papers with no circuit; 2607.09905 Pandey, synthetic task) |
| **Added** | **8** | N1-N8 (limit was 15) |
| Corpus records full-text screened | 58 + 8 = 66 | 8 no full text (C51-C58); 6 excluded after full text (C32, C34, C37, C39, C47, C49); **52 coded** |

Excluded after full text from the 30d corpus: C32 Takaishi (modelling study, no forecast comparison), C34 Aaraba
(synthetic data only), C37 Solikhun (hand-worked perceptron in ket notation, no circuit, no classical model),
C39 Lin QFCQT (no quantum circuit), C47 Abdulrahman (no classical model run), C49 Bakshi (quantum-inspired).
Not coded (no full text): C51 Paquet (ESWA, page blocked), C52 Liu (Inf. Sci., closed), C53 Cao (QMI, closed),
C54 Stuhler (ICAART, captcha), C55 Sahoo (closed), C56 Ceschini (closed), C57 Rosato (closed), C58 Poornima
(SSRN blocked). None of these was coded from its abstract.

Border scope, coded but flagged: C3 and C46 (Exchange-rate is one benchmark among ETT/Weather), C15 (thin demo),
C16 and C41 (RL trading), C26 (test windows are synthetic continuations), N1 and N3 (stock is one of several
tasks). The "core" columns below drop these 8.

## 5. Prevalence

P / (P+A) = share present among papers where the item could be coded. 95% Wilson interval in brackets.
P / applicable = present over present + absent + not_reported (cautious lower bound).

| Pitfall | P | A | NR | NA | P/(P+A), all 52 | P/applicable | P/(P+A), core 44 |
|---|---|---|---|---|---|---|---|
| P1 test-period selection | 7 | 11 | 34 | 0 | 39% [20-61] | 13% | 41% |
| P2 seeds | 42 | 9 | 1 | 0 | 82% [70-90] | 81% | 81% |
| P3 untuned baseline | 2 | 12 | 36 | 2 | 14% [4-40] | 4% | 15% |
| P4 no twin / size match | 14 | 24 | 14 | 0 | 37% [23-53] | 27% | 36% |
| P5 unequal inputs | 1 | 40 | 11 | 0 | 2% [0-13] | 2% | 3% |
| P6 degenerate encoding (kernel papers only) | 2 | 3 | 4 | 43 | 40% [12-77] | 22% | 40% |
| P7 best of many, uncorrected | 13 | 4 | 35 | 0 | 76% [53-90] | 25% | 81% |
| P8 no naive benchmark | 39 | 12 | 1 | 0 | 76% [63-86] | 75% | 79% |
| P9 no test / no correction | 44 | 8 | 0 | 0 | 85% [72-92] | 85% | 84% |
| P10 single split | 34 | 10 | 8 | 0 | 77% [63-87] | 65% | 76% |
| P11 level metric, no naive | 22 | 13 | 4 | 13 | 63% [46-77] | 56% | 60% |
| P12 full-data preprocessing | 4 | 15 | 33 | 0 | 21% [9-43] | 8% | 24% |

Computed with a script from the CSV, not by hand. Pitfalls per paper: median 4, maximum 8 (C1, C35, C36);
43 of 52 have 3 or more. Peer-reviewed papers (21) have median 5; preprints (31) median 4.

How to read it:
- **Robust**: P2, P8, P9, P10 have almost no not_reported cells, so the share is close to the true share in
  this sample. These are the numbers to pair with our measured inflation.
- **Weak**: P1, P3, P7, P12. Most cells are not_reported. The P/(P+A) share rests on 5-19 papers and the
  interval is wide. Do not quote these as prevalence. Quote the count present as a floor.
- **P6** applies to 9 kernel papers only; 2 present (C4, C17). Too few for a share.
- **P5** is rare in the text but our Li audit found it in code. Text coding under-counts it.

### 5.1 Per-paper codes (P / A / NR / NA)

| ID | Paper | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Zhai 2025, Energy | P | P | NR | A | A | NA | P | P | P | P | P | P | 8 |
| C2 | Jones 2024, arXiv | A | A | A | P | A | NA | A | A | P | A | A | NR | 2 |
| C3 | Jura 2025, arXiv (border) | NR | P | NR | NR | A | NA | NR | P | P | P | P | A | 5 |
| C4 | Ahmad 2026, arXiv | A | P | A | A | A | P | P | P | P | A | A | A | 5 |
| C5 | Maheshwari 2026a, arXiv | NR | P | NR | A | A | NA | NR | P | P | P | NR | NR | 4 |
| C6 | Li 2026, PRR (text only) | P | P | A | NR | A | NA | P | A | A | A | A | NR | 3 |
| C7 | Kundu 2024, IC-SIT | NR | P | NR | P | A | NA | NR | P | P | P | P | NR | 6 |
| C8 | Adith 2026, ICICIT | NR | P | NR | P | A | NA | NR | P | P | P | P | A | 6 |
| C9 | Reddy 2026, NQComp | NR | P | A | A | A | A | NR | P | A | P | NA | P | 4 |
| C10 | Singh 2026, CCIC | NR | P | NR | A | NR | NR | P | P | P | P | A | NR | 5 |
| C11 | Srivastava 2023, RASSE | NR | P | NR | A | A | A | P | P | P | NR | NA | NR | 4 |
| C12 | Anand 2024, COMSNETS | NR | P | NR | A | A | NA | NR | P | P | P | P | NR | 5 |
| C13 | Krichen 2026, IWCMC | P | P | NR | A | A | NA | P | P | P | P | NA | NR | 6 |
| C14 | Hachaichi 2025, STA | NR | P | A | A | A | NA | NR | P | P | NR | P | NR | 4 |
| C15 | Matseevich 2026 (border) | NR | P | NR | NR | A | NA | NR | P | P | P | NR | NR | 4 |
| C16 | Dutta 2024, QCE (border) | NR | P | NR | P | NR | NA | NR | A | P | P | NA | NR | 4 |
| C17 | Nair 2026, Front. AI | A | P | P | NR | A | P | NR | P | P | P | P | A | 7 |
| C18 | Emmanoulopoulos 2022 | NR | P | NR | P | A | NA | NR | P | P | P | A | A | 5 |
| C19 | Mahmood 2024 | NR | P | NR | NR | A | NA | NR | P | P | P | NR | NR | 4 |
| C20 | Choudhary 2025, QMI | NR | P | NR | P | A | NA | NR | P | P | A | P | NR | 5 |
| C21 | Waller 2026 | P | A | NR | P | A | NA | P | P | P | P | P | NR | 7 |
| C22 | Otieno 2026 | P | P | NR | A | A | NA | P | P | P | P | P | NR | 7 |
| C23 | Domingo 2023 | A | NR | NA | P | A | NA | NR | P | P | P | P | NR | 5 |
| C24 | Xu 2023 | NR | P | NR | A | A | NA | NR | P | P | P | A | A | 4 |
| C25 | Mourya 2025, Sci. Rep. | NR | P | NA | P | A | NA | NR | P | P | P | A | NR | 5 |
| C26 | Maheshwari 2026b (border) | A | A | A | A | A | NA | A | A | A | A | A | A | 0 |
| C27 | Shen 2026 | A | A | A | A | A | A | A | A | A | A | A | A | 0 |
| C28 | Chen 2025 (QTCNN) | A | P | NR | P | A | NR | P | P | P | P | NA | A | 6 |
| C29 | Weinberg 2025 | A | P | NR | NR | NR | NA | P | A | P | P | NA | NR | 4 |
| C30 | Saeed 2026 | NR | P | NR | P | NR | NA | NR | P | P | P | P | A | 6 |
| C31 | Chen 2026 (QCBM) | NR | P | NR | A | A | NA | NR | P | P | P | A | NR | 4 |
| C33 | Kobayashi 2023, QMI | NR | P | NR | A | A | NA | NR | A | P | A | NA | NR | 2 |
| C35 | Kea 2024, Entropy | P | P | P | NR | A | NA | P | P | P | P | P | NR | 8 |
| C36 | Su 2025, HSSC | NR | P | NR | P | P | NA | NR | P | P | P | P | P | 8 |
| C38 | Polakhare 2026, preprint | NR | A | A | P | NR | NA | P | P | P | NR | NA | A | 4 |
| C40 | Marchisio 2026 | P | P | NR | NR | NR | NA | P | A | P | A | NA | NR | 4 |
| C41 | Chen 2025 (DRL; border) | NR | P | NR | P | NR | NA | NR | A | P | P | NA | NR | 4 |
| C42 | Chen 2025 (DeFi) | NR | P | NR | NR | A | NR | NR | P | P | P | P | NR | 5 |
| C43 | Kamm 2026 | NR | A | NR | A | A | NA | NR | P | P | NR | NA | A | 2 |
| C44 | Deshpande 2025 | NR | P | NR | A | A | NA | NR | P | P | P | P | P | 6 |
| C45 | Amanov 2026 | A | P | A | A | A | NA | NR | P | P | P | P | A | 5 |
| C46 | Chakraborty 2026 (border) | NR | P | NR | NR | A | NA | NR | P | P | NR | P | NR | 4 |
| C48 | Chen 2026 (trained/frozen) | A | A | A | A | A | NA | A | NR | A | A | NA | A | 0 |
| C50 | Kaliutau 2026 | NR | P | NR | A | A | NA | NR | A | A | A | A | NR | 1 |
| N1 | Li Y. 2023, QRNN (border) | NR | P | NR | A | A | NA | NR | P | P | P | P | NR | 5 |
| N2 | Xia 2023, configured QRC | NR | P | NR | NR | NR | NA | NR | P | P | P | P | NR | 5 |
| N3 | Barik 2025, QTFT (border) | NR | P | NR | A | A | NA | NR | P | P | NR | P | NR | 4 |
| N4 | Kar 2025, HPQRC | NR | A | A | A | NR | NA | NR | A | A | P | A | NR | 1 |
| N5 | Nayak 2026, HQFS | A | A | A | NR | A | NA | NR | A | P | P | A | A | 2 |
| N6 | Dwibedi 2025, ICICAT | NR | P | NR | NR | NR | NA | NR | P | P | NR | NR | NR | 3 |
| N7 | Thangavel 2025, ICOSEC | NR | P | NR | NR | NR | NR | NR | P | P | NR | NA | NR | 3 |
| N8 | Ganguly 2025, IC3 | NR | P | NR | A | A | NA | NR | P | A | P | P | NR | 4 |

Other leaks noted (not one of the 12 codes): C16 test window overlaps training; N2 tests GBP/USD inside the
training dates of AUD/USD and NZD/USD; N3 and C20 use same-day open/high/low to predict the close; C10 (94.6%
direction accuracy) and N7 (95.2%) need a leak check.

## 6. Corrections to the 30d pilot table

**Systematic change (17 cells).** 30d coded P6 as U for every paper. Under the rule above, P6 applies only to
kernel models, so 17 non-kernel papers move from U to not_applicable. Other U cells map to not_reported, and
N maps to not_applicable; those are not counted as corrections.

**Cell changes found in the full text (34 cells, 30d value -> new value):**

| Paper | Cell | 30d | New | Reason (page) |
|---|---|---|---|---|
| C3 | P11 | A | P | MSE/MAE on standardised exchange-rate levels with no naive (p8) |
| C4 | P1 | U | A | all models selected by mean validation AUC (p8) |
| C6 | P3 | P | A | text: classical hyperparameters by time-series CV in training window (p10); the weak HAR is an audit finding |
| C6 | P4 | P | NR | classical RC reported (p3-5); size match not stated |
| C6 | P5 | P | A | HARX/LSTMX/RCX get the exogenous inputs (p3-4, p10); unequal inputs are an audit finding |
| C6 | P8 | P | A | AR, HAR, HARX are run (p9) |
| C6 | P9 | P | A | Model Confidence Set used (p13) |
| C6 | P10 | U | A | rolling monthly re-estimation (p9-10) |
| C6 | P12 | P | NR | paper does not describe scaling; full-sample min-max is an audit finding |
| C9 | P3 | U | A | all baselines grid-searched with CV (p1) |
| C9 | P6 | U | A | ZZFeatureMap, not a classical kernel (p1) |
| C10 | P11 | U | A | target is next-day returns (p3); headline is direction accuracy |
| C11 | P6 | U | A | ZZFeatureMap with four entanglement schemes (p4) |
| C11 | P8 | U | P | classification with no majority-class baseline |
| C13 | P9 | U | P | no test found |
| C15 | P2 | U | P | no repeats (p7) |
| C15 | P4 | A | NR | LSTM vs QLSTM sizes not given (p7) |
| C15 | P5 | U | A | same SPY OHLCV inputs (p7) |
| C15 | P8 | U | P | no naive (p7) |
| C15 | P9 | U | P | no test |
| C15 | P10 | U | P | 80/20 split (p7) |
| C17 | P5 | U | A | same RF/PCA features for all models (p4-6) |
| C17 | P6 | U | P | QSVM with angle-embedding quantum kernel (p7) |
| C17 | P11 | U | P | R2/RMSE on index closing price (p4, p7) |
| C18 | P2 | U | P | no repeats reported |
| C18 | P11 | P | A | target is percentage change of close, not price level (p6, p10) |
| C21 | P1 | A | P | validation only "on a subset of QNN models" (p10); headline is best of 1,368 on test (p1, p14) |
| C21 | P7 | U | P | best of 1,368 configurations, no correction (p1-2) |
| C21 | P9 | U | P | no test found |
| C22 | P4 | U | A | classical ESN and MLP baselines (p9); not size-matched |
| C22 | P10 | U | P | 60/40 train/test split (p10) |
| C26 | P12 | U | A | scalers fitted on training window only (p5) |
| C27 | P6 | U | A | IQP-style ZZ feature map plus RBF control (p6) |
| C31 | P9 | U | P | "significantly outperforms" with no test (p1) |

Also note: C26's validation and test windows are synthetic continuations of the revenue series (p5), so C26 is
now flagged border scope. 30d said C29's selection data was not found; it is the validation set (p7), so P1 is
absent. 30d's "U" for C15 was mostly codable once the experiment section (p7) was read.

## 7. Double-coding subsample for Cohen's kappa

Seed **20261118**. Method: Python 3 `random.seed(20261118); random.sample(ids, 12)`, where `ids` is the list of
the 52 coded IDs in CSV row order. 12 of 52 = 23%. The student and the supervisor code these blind to this
note and to each other, then compute kappa per pitfall (target 0.6 or more).

**C2, C3, C5, C11, C14, C15, C21, C22, C25, C28, C33, C46**

Mix: 3 IEEE conference, 2 journal, 7 arXiv; 2 border-scope (C3, C46); includes C21, the paper with the largest
change from 30d. Kappa for P6 will not be estimable (only C11 and C28 are kernel papers here).

## 8. Honest limits

- **One AI coder.** No second coder, no kappa yet. The coder also wrote 9 of the 12 decision rules. Treat every
  number as provisional until the subsample is double-coded.
- **Not a line-by-line read.** A sentence-level search plus reading around the hits. Some not_reported cells
  may be wrong, and a few present cells rest on absence of evidence (P2, P8, P9: "none found").
- **Sample bias.** The corpus started from papers the student collected and from arXiv. 13 of 52 are low-tier
  IEEE conference papers, 31 are preprints. No Scopus, Web of Science, IEEE Xplore or Google Scholar search.
  8 closed-access papers could not be coded; closed journals may differ. The shares describe this sample, not the
  field.
- **Selection of good-practice papers.** C26, C27, C48 were collected partly because they are careful null
  results. They pull the shares down.
- **Text-only coding under-counts code-level pitfalls** (P1, P3, P5, P12). The Li audit shows the gap: C6 has
  3 present cells from text, while 30d (using the audit) coded 9.
- **Scope calls are judgment.** Eight border papers are kept but flagged; the core-44 column shows the shares
  barely change.
- **Arithmetic** was done by a script on the CSV; the counts in sections 1 and 5 come from the same run.

## 9. Next steps for the student

1. Approve or edit the nine new decision rules in section 2 first.
2. Code the 12-paper subsample blind; ask the supervisor to do the same.
3. Try to get the 8 closed papers (C51-C58) through the university library and code them.
4. Run the Scopus / IEEE Xplore string from the 30d note (section D) before any number goes into a report.
