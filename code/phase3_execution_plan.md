# Phase 3 Execution Plan — Code Reorganization for Reproducibility

> **Purpose**: This document is a concrete work-order for the next-stage executor (Codex).
> It specifies exactly which files to move, which notebook cells to extract, and how
> to wire data I/O paths so the pipeline runs end-to-end from a clean checkout.

---

## 0. Prerequisite Reading

| Document | Why |
|---|---|
| `code_availability_handoff.md` | Security scan, PII warnings, paper-to-code map |
| This file (`phase3_execution_plan.md`) | You are here |
| `requirements.txt` | Python dependencies (already created) |

---

## 1. Target Directory Layout

```
code/
├── data_acquisition/          # Step A: Firebase → raw CSV
│   └── download_data.py
├── webapp/                    # Web experiment source (reference only)
│   └── index.html
├── preprocessing/             # Steps B-D: raw CSV → final analytic cohorts
│   ├── 01_merge_and_aggregate.py
│   ├── 02_first_attempt_device_split.py
│   └── 03_age_filter.py
├── analysis/                  # Steps E+: all statistical analyses
│   ├── 04_accuracy_distribution.py
│   ├── 05_sex_gender_effects.py
│   ├── 06_strategy_analysis.py
│   ├── 07_correlation_and_mediation.py
│   ├── 08_reaction_time.py
│   ├── 09_generator_comparison.py
│   └── 10_supplementary_analyses.py
├── figures/                   # Manuscript-ready figure assembly
│   ├── fig2_accuracy_overview.py
│   ├── fig3_mobile_heatmap_mediation.py
│   ├── fig4_sex_differences.py
│   ├── fig5_reaction_time.py
│   ├── fig6_human_factors_model.py
│   ├── fig7_rt_age_sex.py
│   ├── fig8_generator_comparison.py
│   └── figS_supplementary.py
├── data/
│   └── intermediate/          # Pipeline hand-off CSVs (gitignored)
├── config.py                  # Shared path constants & RUN_TAG
├── requirements.txt           # Python dependencies
├── phase3_execution_plan.md   # This file
└── 0_ingredients/             # Original files (archive, not used at runtime)
```

---

## 2. Shared Configuration: `config.py`

Create `code/config.py` as the single source of truth for all paths and the run tag.
Every script in preprocessing/, analysis/, and figures/ must `import config` at the top.

```python
"""
Shared configuration for the HSMU AI Detection analysis pipeline.
All scripts import paths from here — no hard-coded paths elsewhere.
"""
from pathlib import Path

# ── Root directories ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent          # code/
REPO_ROOT    = PROJECT_ROOT.parent                      # hsmu_ai_detection_public/

# De-identified raw data (committed to repo under data/exp_data/)
RAW_DATA_DIR = REPO_ROOT / "data" / "exp_data"

# Intermediate pipeline outputs (generated, gitignored)
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# Analysis outputs (section-wise stats, tables)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Figure outputs
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run tag ───────────────────────────────────────────────────────
# The original notebooks used "20260119_192624". New runs will generate
# their own tag, but for reproducing the paper figures this tag is used
# to locate pre-computed intermediate tables.
RUN_TAG = "20260119_192624"

# ── Convenience paths for raw inputs ─────────────────────────────
RAW_SURVEYS   = RAW_DATA_DIR / "surveys_export_deidentified.csv"
RAW_RESPONSES = RAW_DATA_DIR / "responses_export_deidentified.csv"

# ── Convenience paths for intermediate outputs ───────────────────
ENRICHED_SURVEYS              = INTERMEDIATE_DIR / "enriched_surveys_data.csv"
FILTERED_RESPONSES            = INTERMEDIATE_DIR / "filtered_responses_completed.csv"
PARTICIPANT_STATS_WIDE        = INTERMEDIATE_DIR / "participant_stats_wide.csv"
PARTICIPANT_STATS_LONG        = INTERMEDIATE_DIR / "participant_stats_long.csv"
FIRST_TIMERS_ALL              = INTERMEDIATE_DIR / "analysis_data_first_timers.csv"
FIRST_TIMERS_MOBILE           = INTERMEDIATE_DIR / "analysis_data_first_timers_mobile.csv"
FIRST_TIMERS_WEB              = INTERMEDIATE_DIR / "analysis_data_first_timers_web.csv"
MOBILE_AGE_FILTERED           = INTERMEDIATE_DIR / "analysis_data_mobile_age_filtered_20_69.csv"
WEB_AGE_FILTERED              = INTERMEDIATE_DIR / "analysis_data_web_age_filtered_20_69.csv"
```

---

## 3. File Mapping Table

### 3-A. Direct file moves (no splitting required)

| Source (in `0_ingredients/`) | Destination | Notes |
|---|---|---|
| `download_data.py` | `data_acquisition/download_data.py` | Copy as-is. Already sanitized. |
| `index.html` | `webapp/index.html` | Copy as-is. Firebase config already redacted. |
| `code_availability_handoff.md` | `code/code_availability_handoff.md` (keep at root) | Reference doc. |
| `Sunwhi_Kim_...arXiv7.pdf` | `code/paper.pdf` (or leave in `0_ingredients/`) | Not code; optional move. |

### 3-B. CSV data files — DO NOT copy into new structure

All `.csv` files in `0_ingredients/` contain PII (see handoff §2).
They must **not** be committed to the public repo.
The pipeline will instead:
- **Start from** `data/exp_data/surveys_export_deidentified.csv` and `responses_export_deidentified.csv` (already in repo).
- **Generate** all intermediate CSVs into `code/data/intermediate/` at runtime.

### 3-C. Notebook splitting — `process_data_v8.ipynb` (166 cells)

This is the core work. The notebook must be split into Python scripts.
Below is the cell-to-file mapping.

#### preprocessing/01_merge_and_aggregate.py
- **Source cells**: 0–1 (Section 1)
- **Logic**: Load `RAW_SURVEYS` + `RAW_RESPONSES` → filter to survey completers → compute per-participant accuracy/RT/model-specific stats → merge → save
- **Reads**: `config.RAW_SURVEYS`, `config.RAW_RESPONSES`
- **Writes**:
  - `config.FILTERED_RESPONSES`
  - `config.PARTICIPANT_STATS_WIDE`
  - `config.PARTICIPANT_STATS_LONG`
  - `config.ENRICHED_SURVEYS`
- **Path changes**: Replace all hard-coded `"surveys_export.csv"` / `"responses_export.csv"` with config constants. Replace all `to_csv(...)` calls with config constants.

#### preprocessing/02_first_attempt_device_split.py
- **Source cells**: 5–6 (Section 3)
- **Logic**: Load enriched surveys → keep `repeatCount == 0` (first attempts) → normalize deviceType → split mobile/web → save
- **Reads**: `config.ENRICHED_SURVEYS`
- **Writes**:
  - `config.FIRST_TIMERS_ALL`
  - `config.FIRST_TIMERS_MOBILE`
  - `config.FIRST_TIMERS_WEB`

#### preprocessing/03_age_filter.py
- **Source cells**: 7–8 (Section 4, first half — the filtering part only)
- **Logic**: Load first-timer mobile/web → restrict ages 20–69 → save final analytic cohorts
- **Reads**: `config.FIRST_TIMERS_MOBILE`, `config.FIRST_TIMERS_WEB`
- **Writes**:
  - `config.MOBILE_AGE_FILTERED`
  - `config.WEB_AGE_FILTERED`

#### analysis/04_accuracy_distribution.py
- **Source cells**: 2–4 (Section 2) + cells 7–8 second half (Section 4 age-accuracy analysis) + cells 9–11
- **Logic**: Accuracy distribution histograms, age-accuracy regression summaries
- **Reads**: `config.ENRICHED_SURVEYS`, `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`
- **Writes**: `config.OUTPUTS_DIR / "04_age_accuracy" / ...`

#### analysis/05_sex_gender_effects.py
- **Source cells**: 12–24 (Sections 5–8, 8-extra)
- **Logic**: Sex/gender accuracy comparisons (one-way ANOVA, t-tests, Two-Way ANOVA with Tukey/FDR)
- **Reads**: `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`
- **Writes**: `config.OUTPUTS_DIR / "05_sex_effects" / ...` , `config.OUTPUTS_DIR / "08_age_sex_interaction" / ...`

#### analysis/06_strategy_analysis.py
- **Source cells**: 25–42 (Sections 9–12)
- **Logic**: AI exposure/confidence/attitude by sex, strategy effectiveness (Welch t, OLS+HC3), strategy usage by age×sex (chi-square)
- **Reads**: `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`
- **Writes**: `config.OUTPUTS_DIR / "09_ai_self_reports" / ...` through `"12_strategy_usage" / ...`

#### analysis/07_correlation_and_mediation.py
- **Source cells**: 43–68 (Sections 13–17 v2)
- **Logic**: Correlation matrices, accuracy-focused re-analysis, correlation network visualization, parallel mediation (Age → Exposure+Confidence → Accuracy), path diagrams
- **Reads**: `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`
- **Writes**: `config.OUTPUTS_DIR / "13_correlation" / ...` through `"17_mediation_path" / ...`

#### analysis/08_reaction_time.py
- **Source cells**: 69–118 (Sections 18–26 v1.2)
- **Logic**: AI attitude interaction, MBTI, AI tool usage ANCOVA, learning effects, device effects, RT trend analysis, correct/incorrect RT, RT by image kind, verification cost, trial-level mixed model
- **Reads**: `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`
- **Writes**: `config.OUTPUTS_DIR / "18_attitude_interaction" / ...` through `"26_verification_cost" / ...`

#### analysis/09_generator_comparison.py
- **Source cells**: 137–140 (Section 31)
- **Logic**: ChatGPT vs Gemini accuracy comparison (participant-level paired analysis)
- **Reads**: `config.MOBILE_AGE_FILTERED`
- **Writes**: `config.OUTPUTS_DIR / "31_generator_comparison" / ...`

#### analysis/10_supplementary_analyses.py
- **Source cells**: 119–136, 141–165 (Sections 26 v2–30, 32–36)
- **Logic**: Confidence×Attitude interaction, sex-stratified models, RT by accuracy group, speed-accuracy tradeoff, time trend, moderated mediation, sex-stratified correlation networks, path diagrams
- **Reads**: `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`, various intermediate outputs
- **Writes**: `config.OUTPUTS_DIR / "26v2_confidence_attitude" / ...` through `"36_path_diagrams" / ...`

### 3-D. Notebook splitting — `process_data_v8plus.ipynb` (69 cells)

This notebook assembles manuscript-ready figures. Split into per-figure scripts.

#### figures/fig2_accuracy_overview.py
- **Source cells**: 0–6 (Sections 8_plus §1, Unified, 4-2 ONLY, Slope Bar)
- **Logic**: Participant overview pie chart, accuracy histogram, age-accuracy regplot, device slope comparison
- **Reads**: `config.MOBILE_AGE_FILTERED`, `config.WEB_AGE_FILTERED`, `config.OUTPUTS_DIR / ...`
- **Writes**: `config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "fig2_..." / ...`

#### figures/fig3_mobile_heatmap_mediation.py
- **Source cells**: 7–9 (Figure 3 Option B, 36P pooled path)
- **Logic**: Mobile heatmap + confidence/exposure scatter + mediation path diagram
- **Reads**: `config.MOBILE_AGE_FILTERED`, mediation outputs from analysis step
- **Writes**: `config.PLOTS_DIR / ... / "fig3_..."  `

#### figures/fig4_sex_differences.py
- **Source cells**: 10–15 (Figure 4 Prep, 4A/B/C/D, forest plots)
- **Logic**: Sex distribution pie, age-bin×sex accuracy, forest plots with bootstrap CIs
- **Reads**: `config.MOBILE_AGE_FILTERED`, sex-effects outputs from analysis step
- **Writes**: `config.PLOTS_DIR / ... / "fig4_..."`

#### figures/fig5_reaction_time.py
- **Source cells**: 16–18 (Figure 5, verification cost)
- **Logic**: RT analysis panels, verification cost visualization
- **Reads**: `config.MOBILE_AGE_FILTERED`, RT outputs from analysis step
- **Writes**: `config.PLOTS_DIR / ... / "fig5_..."`

#### figures/fig6_human_factors_model.py
- **Source cells**: 22–36 (Figure 5 strategy story, Figure 6 integrated model)
- **Logic**: Strategy effectiveness visualization, integrated human-factors model diagram
- **Reads**: `config.MOBILE_AGE_FILTERED`, strategy + regression outputs
- **Writes**: `config.PLOTS_DIR / ... / "fig6_..."`

#### figures/fig7_rt_age_sex.py
- **Source cells**: 37 (Figure 7)
- **Logic**: Age-bin × Sex RT interaction with MixedLM EMM
- **Reads**: `config.MOBILE_AGE_FILTERED`, RT outputs
- **Writes**: `config.PLOTS_DIR / ... / "fig7_..."`

#### figures/fig8_generator_comparison.py
- **Source cells**: 38–45 (Figure 8)
- **Logic**: ChatGPT vs Gemini comparison panels
- **Reads**: `config.MOBILE_AGE_FILTERED`, generator comparison outputs
- **Writes**: `config.PLOTS_DIR / ... / "fig8_..."`

#### figures/figS_supplementary.py
- **Source cells**: 47–68 (Figures S0–S6)
- **Logic**: PC-cohort replications of main figures (S1–S6), age-bin participant counts (S0)
- **Reads**: `config.WEB_AGE_FILTERED`, various analysis outputs
- **Writes**: `config.PLOTS_DIR / ... / "figS_..."`

---

## 4. Data Pipeline Flow (with I/O paths)

```
┌─────────────────────────────────────────────────────────────┐
│  data/exp_data/                                             │
│    surveys_export_deidentified.csv                          │
│    responses_export_deidentified.csv                        │
└────────────────────┬────────────────────────────────────────┘
                     │  (config.RAW_SURVEYS, config.RAW_RESPONSES)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  preprocessing/01_merge_and_aggregate.py                    │
│  OUTPUT →  data/intermediate/                               │
│    enriched_surveys_data.csv                                │
│    filtered_responses_completed.csv                         │
│    participant_stats_wide.csv                               │
│    participant_stats_long.csv                               │
└────────────────────┬────────────────────────────────────────┘
                     │  (config.ENRICHED_SURVEYS)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  preprocessing/02_first_attempt_device_split.py             │
│  OUTPUT →  data/intermediate/                               │
│    analysis_data_first_timers.csv                           │
│    analysis_data_first_timers_mobile.csv                    │
│    analysis_data_first_timers_web.csv                       │
└────────────────────┬────────────────────────────────────────┘
                     │  (config.FIRST_TIMERS_MOBILE, config.FIRST_TIMERS_WEB)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  preprocessing/03_age_filter.py                             │
│  OUTPUT →  data/intermediate/                               │
│    analysis_data_mobile_age_filtered_20_69.csv              │
│    analysis_data_web_age_filtered_20_69.csv                 │
└────────────────────┬────────────────────────────────────────┘
                     │  (config.MOBILE_AGE_FILTERED, config.WEB_AGE_FILTERED)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  analysis/04–10_*.py   (can run independently of each other)│
│  OUTPUT →  outputs/run_<RUN_TAG>/section_name/              │
│    stats CSVs, describe tables, metadata JSON               │
└────────────────────┬────────────────────────────────────────┘
                     │  (config.OUTPUTS_DIR / ...)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  figures/fig*.py   (can run independently of each other)    │
│  OUTPUT →  plots/run_<RUN_TAG>/fig_name/                    │
│    PNG + SVG publication figures                             │
└─────────────────────────────────────────────────────────────┘
```

**Execution order**:
```bash
# Step 1: Preprocessing (must run sequentially)
python preprocessing/01_merge_and_aggregate.py
python preprocessing/02_first_attempt_device_split.py
python preprocessing/03_age_filter.py

# Step 2: Analysis (can run in any order after Step 1)
python analysis/04_accuracy_distribution.py
python analysis/05_sex_gender_effects.py
python analysis/06_strategy_analysis.py
python analysis/07_correlation_and_mediation.py
python analysis/08_reaction_time.py
python analysis/09_generator_comparison.py
python analysis/10_supplementary_analyses.py

# Step 3: Figures (can run in any order after Step 2)
python figures/fig2_accuracy_overview.py
python figures/fig3_mobile_heatmap_mediation.py
python figures/fig4_sex_differences.py
python figures/fig5_reaction_time.py
python figures/fig6_human_factors_model.py
python figures/fig7_rt_age_sex.py
python figures/fig8_generator_comparison.py
python figures/figS_supplementary.py
```

---

## 5. Critical Path-Rewriting Rules

When extracting code from notebooks into `.py` files, apply these transformations:

### 5-A. Replace all hard-coded CSV paths

| Original pattern (in notebook) | Replacement |
|---|---|
| `"surveys_export.csv"` or `"responses_export.csv"` | `config.RAW_SURVEYS` / `config.RAW_RESPONSES` |
| `"enriched_surveys_data.csv"` | `config.ENRICHED_SURVEYS` |
| `"filtered_responses_completed.csv"` | `config.FILTERED_RESPONSES` |
| `"analysis_data_first_timers.csv"` | `config.FIRST_TIMERS_ALL` |
| `"analysis_data_first_timers_mobile.csv"` | `config.FIRST_TIMERS_MOBILE` |
| `"analysis_data_first_timers_web.csv"` | `config.FIRST_TIMERS_WEB` |
| `"analysis_data_mobile_age_filtered_20_69.csv"` | `config.MOBILE_AGE_FILTERED` |
| `"analysis_data_web_age_filtered_20_69.csv"` | `config.WEB_AGE_FILTERED` |
| `"participant_stats_wide.csv"` | `config.PARTICIPANT_STATS_WIDE` |
| `"participant_stats_long.csv"` | `config.PARTICIPANT_STATS_LONG` |

### 5-B. Replace all absolute paths

| Original pattern | Replacement |
|---|---|
| `C:\\Users\\gdrpa\\Desktop\\Real_vs_AI_data\\...` | `config.PROJECT_ROOT / ...` (relative) |
| `fr"outputs\run_..."` (backslash Windows paths) | `config.OUTPUTS_DIR / f"run_{config.RUN_TAG}" / ...` |
| `Path("outputs/run_20260119_192624/...")` | `config.OUTPUTS_DIR / f"run_{config.RUN_TAG}" / ...` |
| `Path("plots")/f"run_..."` | `config.PLOTS_DIR / f"run_{config.RUN_TAG}" / ...` |

### 5-C. Replace hard-coded RUN_TAG

| Original pattern | Replacement |
|---|---|
| `"20260119_192624"` (literal string) | `config.RUN_TAG` |
| `RUN_TAG="20260119_192624"` (local variable) | `from config import RUN_TAG` |

### 5-D. Add `import config` and `sys.path` setup

Every script should begin with:
```python
"""<description of this pipeline step>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
```

### 5-E. Wrap executable code in `if __name__ == "__main__":`

All top-level notebook code should be wrapped in a `main()` function called from `if __name__ == "__main__":` so scripts can also be imported as modules if needed.

### 5-F. Encoding

All `pd.read_csv()` calls should use `encoding="utf-8-sig"` (matching the original export).
All `pd.to_csv()` calls should use `encoding="utf-8-sig", index=False`.

---

## 6. `.gitignore` Updates

Add the following to the repo's `.gitignore`:

```
# Pipeline intermediate data (generated at runtime, may contain PII)
code/data/intermediate/
code/outputs/
code/plots/

# Original ingredients (archived, not used at runtime)
code/0_ingredients/*.csv
```

---

## 7. Validation Checklist

After all files are moved and rewritten, verify:

- [ ] `python preprocessing/01_merge_and_aggregate.py` completes and writes 4 CSVs to `data/intermediate/`
- [ ] `python preprocessing/02_first_attempt_device_split.py` reads enriched CSV and writes 3 CSVs
- [ ] `python preprocessing/03_age_filter.py` reads first-timer CSVs and writes 2 final cohort CSVs
- [ ] Each `analysis/*.py` script reads from `config.*` paths and writes to `outputs/`
- [ ] Each `figures/*.py` script reads from `config.*` paths and writes PNG/SVG to `plots/`
- [ ] No file contains hard-coded absolute paths (grep for `C:\\Users` and `Desktop`)
- [ ] No file reads directly from `0_ingredients/`
- [ ] `data/intermediate/` is gitignored
- [ ] `requirements.txt` covers all imports
- [ ] The paper's Figures 2–8 + supplementary figures can all be reproduced from a clean checkout

---

## 8. Notes for Executor

1. **Do not delete `0_ingredients/`** until the full pipeline is verified. Keep it as an archive for diff-checking.
2. **PII reminder**: The de-identified CSVs in `data/exp_data/` do NOT have an `email` column. If any preprocessing code references `email`, that column reference should be removed or made conditional.
3. **Large notebook cells**: Some cells (e.g., Section 8, Sections 32–36) are very long (500+ lines). Extract them as-is into the target file; do not attempt to further subdivide within a single analysis section.
4. **Matplotlib Korean text**: Several cells set Korean font paths. Extract these into a shared helper in `config.py` or a `plotting_utils.py` if the pattern repeats across 3+ files.
5. **The v8plus notebook** references pre-computed tables from `outputs/run_20260119_192624/`. The figure scripts must either (a) run the analysis scripts first to regenerate these tables, or (b) accept the RUN_TAG as a parameter pointing to existing outputs.
