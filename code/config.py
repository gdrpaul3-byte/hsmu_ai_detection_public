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


def apply_plot_style(
    *,
    font_scale: float = 3.0,
    base: float = 10,
    theme: str = "whitegrid",
    legend_scale: float = 0.7,
    title_scale: float = 1.10,
    tick_scale: float = 0.90,
    korean: bool = False,
) -> float:
    """Apply a consistent seaborn/matplotlib style and return the computed base size."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    theme_kwargs = {"style": theme}
    if korean:
        theme_kwargs["font"] = "Malgun Gothic"
    sns.set_theme(**theme_kwargs)

    big = base * font_scale
    plt.rcParams.update({
        "font.size": big,
        "axes.titlesize": big * title_scale,
        "axes.labelsize": big,
        "xtick.labelsize": big * tick_scale,
        "ytick.labelsize": big * tick_scale,
        "legend.fontsize": big * legend_scale,
    })
    if korean:
        plt.rcParams["axes.unicode_minus"] = False
    return big


def apply_korean_plot_style(
    *,
    font_scale: float = 3.0,
    base: float = 10,
    theme: str = "whitegrid",
    legend_scale: float = 0.7,
    title_scale: float = 1.10,
    tick_scale: float = 0.90,
) -> float:
    """Apply the default Korean plot style used across analysis notebooks."""
    return apply_plot_style(
        font_scale=font_scale,
        base=base,
        theme=theme,
        legend_scale=legend_scale,
        title_scale=title_scale,
        tick_scale=tick_scale,
        korean=True,
    )
