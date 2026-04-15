# Code Availability Handoff

## 1. Security Scan Summary

- Scanned `0_ingredients` for Firebase credentials, service-account JSON fragments, private keys, API keys, and other hard-coded secrets.
- Redacted Firebase web config in [index.html](webapp/index.html).
- Generalized the Firebase Admin SDK path in [download_data.py](data_acquisition/download_data.py).
- No embedded service-account JSON, PEM private key block, or additional API-key-like token was found in the notebooks.

## 2. High-Risk Data Findings

- `surveys_export.csv` contains an `email` column with 747 non-empty values.
- The same `email` field propagates into:
  - `enriched_surveys_data.csv` (747 non-empty)
  - `analysis_data_first_timers.csv` (717 non-empty)
  - `analysis_data_first_timers_mobile.csv` (563 non-empty)
  - `analysis_data_first_timers_web.csv` (154 non-empty)
  - `analysis_data_mobile_age_filtered_20_69.csv` (529 non-empty)
  - `analysis_data_web_age_filtered_20_69.csv` (143 non-empty)
- These CSVs should not be included in a public code-release package without a separate de-identification pass.
- Several CSVs also retain free-text fields such as `suggestions`, `otherStrategy`, `otherAiTool`, and nested `responses`, so they should be treated as participant data, not code artifacts.

## 3. Paper-to-Code Mapping

### A. Web experiment source

- File: [index.html](webapp/index.html)
- Role: Single-file web experiment frontend described in Fig. 1.
- Inputs:
  - Firebase/Firestore project config
  - Stimulus pool and trial logic embedded in the page
  - Participant survey responses entered in-browser
- Outputs:
  - Firestore `responses` collection: trial-level records with fields such as `imageType`, `response`, `isCorrect`, `participantId`, `deviceType`, `trial`, `rt`, `imageId`
  - Firestore `surveys` collection: participant-level survey data including age, gender/sex, AI exposure, confidence, strategy, and summary metrics
- Paper link:
  - Matches the PDF description of device selection, 4 practice trials, 20 main trials, post-task survey, and score report.

### B. Firebase export script

- File: [download_data.py](data_acquisition/download_data.py)
- Role: Pulls raw Firestore collections into CSV.
- Inputs:
  - Firebase Admin SDK JSON path
  - Firestore collections `surveys` and `responses`
- Outputs:
  - `surveys_export.csv`
  - `responses_export.csv`
- Data flow:
  - Firestore -> pandas DataFrame -> UTF-8-SIG CSV export

### C. Core preprocessing and participant-level aggregation

- Original file: `0_ingredients/process_data_v8.ipynb` (archived; now split into `preprocessing/` and `analysis/` scripts)
- Relevant section:
  - Section `(1)` preprocesses and merges raw exports
- Inputs:
  - `responses_export.csv` (69,494 rows)
  - `surveys_export.csv` (1,843 rows)
- Outputs:
  - `filtered_responses_completed.csv` (trial data restricted to survey completers; 44,232 rows)
  - `participant_stats_wide.csv`
  - `participant_stats_long.csv`
  - `enriched_surveys_data.csv`
- Data flow:
  - Raw Firestore exports -> keep participants with completed surveys -> compute per-participant accuracy/RT/model-specific stats -> merge back into survey table
- Paper link:
  - Provides the post-survey-completion base table used for downstream analytic filtering and overall performance summaries.

### D. First-attempt and device split preprocessing

- Original file: `0_ingredients/process_data_v8.ipynb` (archived; now split into `preprocessing/` and `analysis/` scripts)
- Relevant section:
  - Section `(3)` first-time filtering + device split
- Inputs:
  - `enriched_surveys_data.csv`
- Outputs:
  - `analysis_data_first_timers.csv`
  - `analysis_data_first_timers_mobile.csv`
  - `analysis_data_first_timers_web.csv`
  - section artifacts under `outputs/run_.../03_first_time_device_split/`
- Data flow:
  - Enriched participant table -> retain first attempts -> normalize `deviceType` / infer from participant ID suffix -> split into mobile and web cohorts
- Paper link:
  - Corresponds to the PDF's "first attempt only" filter and the device-stratified analyses.

### E. Age filter preprocessing

- Original file: `0_ingredients/process_data_v8.ipynb` (archived; now split into `preprocessing/` and `analysis/` scripts)
- Relevant section:
  - Section `(4)` age filtering and age-accuracy analysis
- Inputs:
  - `analysis_data_first_timers_mobile.csv`
  - `analysis_data_first_timers_web.csv`
- Outputs:
  - `analysis_data_mobile_age_filtered_20_69.csv` (1,330 rows)
  - `analysis_data_web_age_filtered_20_69.csv` (334 rows)
  - age-analysis plots/stats under `outputs/run_.../04_age_accuracy/`
- Data flow:
  - First-time device cohorts -> restrict to ages 20-69 -> save final analytic cohorts -> compute age vs. accuracy summaries
- Paper link:
  - Matches the final analytic sample described in the PDF abstract and Fig. 2.

### F. Main analysis notebook for manuscript figures and stats

- Original file: `0_ingredients/process_data_v8.ipynb` (archived; now split into `preprocessing/` and `analysis/` scripts)
- Role:
  - Primary statistical pipeline.
- Inputs:
  - `enriched_surveys_data.csv`
  - `analysis_data_first_timers_*.csv`
  - `analysis_data_*_age_filtered_20_69.csv`
  - intermediate run tables under `outputs/run_<RUN_TAG>/...`
- Outputs:
  - Section-wise stats tables, raw tables, and plots under `outputs/run_<RUN_TAG>/...`
- Paper link:
  - Early sections cover Fig. 2-5 foundations.
  - Later sections cover sex effects, strategy analyses, mediation, RT, and generator comparison corresponding to later main figures and supplementary figures.

### G. Manuscript-ready figure assembly notebook

- Original file: `0_ingredients/process_data_v8plus.ipynb` (archived; now split into `figures/` scripts)
- Role:
  - Figure-composition notebook for final paper panels.
- Inputs:
  - final analytic CSVs in `0_ingredients`
  - many precomputed tables from `outputs/run_20260119_192624/...`
  - some cells reference absolute local paths such as `C:\\Users\\gdrpa\\Desktop\\Real_vs_AI_data`
- Outputs:
  - manuscript-oriented figures under `plots/run_20260119_192624/...`
- Paper link:
  - Explicitly labeled as Figure 3, Figure 4, Figure 5, Figure 6, Figure 7, Figure 8, and supplementary figure builders.

## 4. Practical Release Notes for Claude Code

- For a code-only public release, keep:
  - `index.html`
  - `download_data.py`
  - notebooks after path cleanup and documentation
- Do not publish raw or derived CSVs as-is; they contain participant data and contact information.
- `process_data_v8plus.ipynb` is not yet portable because it references fixed run tags and absolute local filesystem paths.
- If the goal is reproducible code availability, the next cleanup step should be:
  - remove or externalize all participant-data files
  - parameterize notebook input/output roots
  - split large notebooks into scriptable stages
