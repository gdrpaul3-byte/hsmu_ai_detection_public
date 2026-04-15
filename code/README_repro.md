# Reproducibility Guide

This directory contains the analysis code for the paper:

> **Human Factors in Detecting AI-Generated Portraits: Age, Sex, Device, and Confidence**
> Sunwhi Kim & Sunyul Kim (2026)

The pipeline processes de-identified experimental data and reproduces the statistical
analyses, tables, and figures reported in the manuscript. Raw participant data with
personally identifiable information (PII) is **not** included; only de-identified
inputs are provided.

---

## Repository structure

```
code/
├── config.py                  # Single source of truth for all file paths
├── requirements.txt           # Python dependencies
├── README_repro.md            # This file
├── CITATION.cff               # Citation metadata
├── code_availability_handoff.md  # Security scan & paper-to-code mapping
├── data_acquisition/          # Firebase → CSV export script (reference only)
│   └── download_data.py
├── webapp/                    # Web experiment source code (reference only)
│   └── index.html
├── preprocessing/             # Data cleaning pipeline (run sequentially)
│   ├── 01_merge_and_aggregate.py
│   ├── 02_first_attempt_device_split.py
│   └── 03_age_filter.py
├── analysis/                  # Statistical analyses (run after preprocessing)
│   ├── 04_accuracy_distribution.py
│   ├── 05_sex_gender_effects.py
│   ├── 06_strategy_analysis.py
│   ├── 07_correlation_and_mediation.py
│   ├── 08_reaction_time.py
│   ├── 09_generator_comparison.py
│   └── 10_supplementary_analyses.py
├── figures/                   # Manuscript figure generation
│   ├── fig2_accuracy_overview.py
│   ├── fig3_mobile_heatmap_mediation.py
│   ├── fig4_sex_differences.py
│   ├── fig5_reaction_time.py
│   ├── fig6_human_factors_model.py
│   ├── fig7_rt_age_sex.py
│   ├── fig8_generator_comparison.py
│   └── figS_supplementary.py
├── data/intermediate/         # Generated at runtime (gitignored)
├── outputs/                   # Analysis outputs: stats, tables (gitignored)
├── plots/                     # Generated figures: PNG, SVG (gitignored)
└── 0_ingredients/             # Original working files (archive, not executed)
```

### Directory roles

| Directory | Purpose |
|---|---|
| `data_acquisition/` | Script that originally downloaded raw data from Firebase/Firestore. Provided for transparency; not needed for reproduction since de-identified data is already in `data/exp_data/`. |
| `webapp/` | The single-file web experiment (`index.html`) described in Figure 1 of the paper. Provided for methodological transparency. |
| `preprocessing/` | Three scripts that transform de-identified raw CSVs into the final analytic cohorts. Must be run sequentially. |
| `analysis/` | Statistical analysis scripts corresponding to the paper's results sections. Can be run in any order after preprocessing completes. |
| `figures/` | Scripts that produce the manuscript figures (Figures 2–8 and supplementary). Can be run in any order after analysis completes. |
| `data/intermediate/` | Intermediate CSVs created by the preprocessing pipeline at runtime. Gitignored because they are generated, not source data. |
| `outputs/` | Section-wise statistics tables, CSV summaries, and metadata JSON produced by analysis scripts. Gitignored. |
| `plots/` | PNG and SVG figure files produced by figure scripts. Gitignored. |
| `0_ingredients/` | Archive of the original working directory before reorganization. Not referenced by any script. Do not execute code from this directory. |

---

## Prerequisites

### Python environment

Python 3.10 or later is recommended. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The key packages are: `pandas`, `numpy`, `scipy`, `statsmodels`, `pingouin`,
`scikit-posthocs`, `matplotlib`, `seaborn`, `networkx`, and `graphviz`.

To re-download raw data from Firebase (not required for reproduction), install
`firebase-admin` separately: `pip install firebase-admin`.

**Graphviz system dependency**: The `graphviz` Python package requires the Graphviz
system binary. Install it from https://graphviz.org/download/ and ensure the `dot`
command is on your PATH. This is only needed for path-diagram visualizations
(mediation figures).

### Input data

The de-identified input data is already committed to the repository at
(paths relative to repository root, **not** `code/`):

```
data/exp_data/
├── surveys_export_deidentified.csv     # 1,843 participant survey records
├── responses_export_deidentified.csv   # 69,494 trial-level response records
└── data_dictionary.md                  # Column descriptions
```

No additional data download is required for reproduction. `config.py` resolves
these paths automatically via `REPO_ROOT / "data" / "exp_data" / ...` — scripts
can be run from any working directory.

### How `config.py` works

Every script imports `config.py` to resolve input/output paths. You do not need to
edit `config.py` unless you move the repository or want to change the output
directory structure. Key constants:

| Constant | Resolves to |
|---|---|
| `config.RAW_SURVEYS` | `<repo>/data/exp_data/surveys_export_deidentified.csv` |
| `config.RAW_RESPONSES` | `<repo>/data/exp_data/responses_export_deidentified.csv` |
| `config.INTERMEDIATE_DIR` | `<repo>/code/data/intermediate/` |
| `config.OUTPUTS_DIR` | `<repo>/code/outputs/` |
| `config.PLOTS_DIR` | `<repo>/code/plots/` |
| `config.RUN_TAG` | `"20260119_192624"` (timestamp tag for output subdirectories) |

---

## Firebase placeholders (reference only)

The following files contain placeholder values where Firebase credentials were
removed for security. **You do not need to fill these in to reproduce the paper's
results** — the de-identified data is already provided. These placeholders are
documented here for researchers who wish to deploy their own instance of the
experiment.

### `webapp/index.html` (lines 322–328)

```javascript
apiKey: "YOUR_API_KEY",
authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
projectId: "YOUR_PROJECT_ID",
storageBucket: "YOUR_PROJECT_ID.appspot.com",
messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
appId: "YOUR_APP_ID",
measurementId: "YOUR_MEASUREMENT_ID"
```

Also on line 337:
```javascript
const RESTART_URL = "https://YOUR_PROJECT_ID.web.app/";
```

Replace these with your own Firebase project credentials from the
[Firebase Console](https://console.firebase.google.com/).

### `data_acquisition/download_data.py` (line 5)

```python
SERVICE_ACCOUNT_PATH = "YOUR_FIREBASE_ADMIN_SDK_JSON_PATH"
```

Replace with the local path to your Firebase Admin SDK JSON key file. This file
should never be committed to version control.

---

## Execution order

Run all commands from the `code/` directory.

### Step 1: Preprocessing (sequential)

These three scripts must be run in order. Each reads the output of the previous step.

```bash
python preprocessing/01_merge_and_aggregate.py
python preprocessing/02_first_attempt_device_split.py
python preprocessing/03_age_filter.py
```

| Script | Input | Output |
|---|---|---|
| `01_merge_and_aggregate.py` | `<repo>/data/exp_data/surveys_export_deidentified.csv`, `<repo>/data/exp_data/responses_export_deidentified.csv` | `data/intermediate/enriched_surveys_data.csv`, `data/intermediate/filtered_responses_completed.csv`, `data/intermediate/participant_stats_wide.csv`, `data/intermediate/participant_stats_long.csv` |
| `02_first_attempt_device_split.py` | `data/intermediate/enriched_surveys_data.csv` | `data/intermediate/analysis_data_first_timers.csv`, `data/intermediate/analysis_data_first_timers_mobile.csv`, `data/intermediate/analysis_data_first_timers_web.csv` |
| `03_age_filter.py` | `data/intermediate/analysis_data_first_timers_mobile.csv`, `data/intermediate/analysis_data_first_timers_web.csv` | `data/intermediate/analysis_data_mobile_age_filtered_20_69.csv`, `data/intermediate/analysis_data_web_age_filtered_20_69.csv` |

> **Note on sample size**: The paper reports a final analytic sample of 1,664
> (mobile 1,330 + PC 334). When running this pipeline from the de-identified
> data, the final cohorts are slightly larger (mobile 1,332 + PC 335 = 1,667).
> This difference of 3 participants is attributable to the de-identified export
> having been reconstructed independently from the original analysis snapshot.
> The exact cause of the discrepancy has not been isolated within this
> repository. The statistical conclusions are not materially affected by this
> difference.

### Step 2: Analysis (independent)

After preprocessing, these scripts can be run in any order. Each reads from
`data/intermediate/` and writes statistics/tables to `outputs/`.

```bash
python analysis/04_accuracy_distribution.py
python analysis/05_sex_gender_effects.py
python analysis/06_strategy_analysis.py
python analysis/07_correlation_and_mediation.py
python analysis/08_reaction_time.py
python analysis/09_generator_comparison.py
python analysis/10_supplementary_analyses.py
```

### Step 3: Figures (independent)

After analysis, these scripts can be run in any order. Each reads from
`data/intermediate/` and `outputs/`, and writes PNG/SVG files to `plots/`.

```bash
python figures/fig2_accuracy_overview.py
python figures/fig3_mobile_heatmap_mediation.py
python figures/fig4_sex_differences.py
python figures/fig5_reaction_time.py
python figures/fig6_human_factors_model.py
python figures/fig7_rt_age_sex.py
python figures/fig8_generator_comparison.py
python figures/figS_supplementary.py
```

### Run everything at once (Bash / macOS / Linux)

```bash
# Preprocessing (sequential)
python preprocessing/01_merge_and_aggregate.py && \
python preprocessing/02_first_attempt_device_split.py && \
python preprocessing/03_age_filter.py

# Analysis (can be parallelized)
for f in analysis/0*.py analysis/1*.py; do python "$f"; done

# Figures (can be parallelized)
for f in figures/fig*.py; do python "$f"; done
```

On **Windows PowerShell**, run the step-by-step commands in Steps 1–3 above
instead, or use Git Bash / WSL.

---

## Paper-to-code mapping

### Main figures

| Figure | Script(s) | Description |
|---|---|---|
| Fig. 1 | `webapp/index.html` | Stimulus generation and experimental procedure (schematic; not generated by code) |
| Fig. 2 | `figures/fig2_accuracy_overview.py` | Overall accuracy distribution, age–accuracy relationship, device comparison |
| Fig. 3 | `figures/fig3_mobile_heatmap_mediation.py` | Mobile cohort: heatmap, confidence/exposure scatter, mediation path diagram |
| Fig. 4 | `figures/fig4_sex_differences.py` | Sex differences: age-bin × sex accuracy, forest plots with bootstrap CIs |
| Fig. 5 | `figures/fig6_human_factors_model.py` (first half) | Decision cues (self-reported strategies) associated with accuracy |
| Fig. 6 | `figures/fig6_human_factors_model.py` (second half) | Unified human-factors regression model (standardized beta + incremental R²) |
| Fig. 7 | `figures/fig5_reaction_time.py`, `figures/fig7_rt_age_sex.py` | RT verification cost (5) + age-group × sex RT with condition-specific LMM EMMs (7) |
| Fig. 8 | `figures/fig8_generator_comparison.py` | ChatGPT-4o vs. Imagen 3 accuracy and RT comparison |

> **Note on script naming**: `fig5_reaction_time.py` and `fig6_human_factors_model.py`
> carry internal numbering from an earlier draft. The paper's final figure order is:
> Fig. 5 = decision cues/strategies, Fig. 6 = unified model, Fig. 7 = RT analyses.
> The mapping above reflects the paper's published caption numbering.

### Supplementary figures

| Figure | Script | Description |
|---|---|---|
| Fig. S0 | `figures/figS_supplementary.py` | Age-bin participant counts by device and sex |
| Fig. S1 | `figures/figS_supplementary.py` | PC cohort: correlations, self-report scatters, mediation |
| Fig. S2 | `figures/figS_supplementary.py` | PC cohort: sex differences replication |
| Fig. S3 | `figures/figS_supplementary.py` | PC cohort: strategy effects and endorsement rates |
| Fig. S4 | `figures/figS_supplementary.py` | PC cohort: unified human-factors model |
| Fig. S5 | `figures/figS_supplementary.py` | PC cohort: RT patterns |
| Fig. S6 | `figures/figS_supplementary.py` | PC cohort: generator comparison |

### Analysis scripts → paper sections

| Script | Paper content |
|---|---|
| `04_accuracy_distribution.py` | Overall accuracy, age–accuracy regression (Fig. 2 foundations) |
| `05_sex_gender_effects.py` | Sex/gender effects: ANOVA, t-tests, age × sex interaction |
| `06_strategy_analysis.py` | AI self-reports by sex, strategy effectiveness (OLS + HC3), strategy usage (chi-square) |
| `07_correlation_and_mediation.py` | Correlation matrices, parallel mediation (age → exposure + confidence → accuracy) |
| `08_reaction_time.py` | RT analyses: age/sex trends, correct vs. incorrect RT, verification cost, trial-level LMM |
| `09_generator_comparison.py` | ChatGPT-4o vs. Imagen 3 paired comparisons |
| `10_supplementary_analyses.py` | Confidence × attitude interaction, moderated mediation, sex-stratified networks, SAT |

---

## Important notes

1. **Do not publish `0_ingredients/*.csv`**. These files contain participant PII
   (email addresses, free-text responses). They are gitignored and excluded from
   the public repository. Use only the de-identified files in `data/exp_data/`.

2. **Generated directories are gitignored**. The `data/intermediate/`, `outputs/`,
   and `plots/` directories are created at runtime and are not tracked in version
   control. Delete them and re-run the pipeline to regenerate from scratch.

3. **System Graphviz is required** for mediation path diagrams. If not installed,
   those specific visualizations will fail, but all other analyses and figures will
   complete normally.

4. **Font rendering**. Some figure scripts configure Korean-language fonts for axis
   labels. If Korean fonts are not installed on your system, matplotlib will fall
   back to default fonts with missing glyphs. This affects cosmetic rendering only,
   not statistical results.

5. **Encoding**. All CSV I/O uses `utf-8-sig` encoding to preserve Korean text in
   survey fields. Do not change this encoding without verifying that Korean
   characters are preserved.
