"""Supplementary analyses including confidence-attitude, moderated mediation, and path diagrams."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_119():
    # ==============================================================================
    # (26 v2 EN | MOBILE ONLY) Accuracy ~ AI Confidence × AI Attitude
    # - MOBILE only
    # - Main stats: Two-way ANOVA on attitude_group(3) × aiConfidence(5)
    # - Sensitivity: Two-way ANOVA on original aiAttitude(5) × aiConfidence(5)
    # - Extra: Ordinal-coded regression (linear trend + interaction), HC3 robust
    # - Save: plots (png/svg), ANOVA tables (csv), reports (txt),
    #         Prism-friendly long + cell summary tables (csv)
    # - Y-axis fixed: 0..100
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # -----------------------------
    # Config (MOBILE ONLY)
    # -----------------------------
    COHORT_TAG = "mobile"
    FILE_PATH = config.MOBILE_AGE_FILTERED
    OUTDIR = config.OUTPUTS_DIR / "outputs_26_confidence_attitude_v2_en"

    CONF_ORDER = ["very-not-confident", "not-confident", "neutral", "confident", "very-confident"]

    USE_KOREAN_XTICKS = False
    CONF_LABELS_EN = ["Very unconfident", "Unconfident", "Neutral", "Confident", "Very confident"]
    CONF_LABELS_KO = ["매우 자신 없음", "자신 없음", "보통", "자신 있음", "매우 자신 있음"]

    ATT5_ORDER = ["very-positive", "positive", "neutral", "negative", "very-negative"]
    ATT3_ORDER = ["Positive", "Neutral", "Negative"]

    ATTITUDE_MAP = {
        "very-positive": "Positive",
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
        "very-negative": "Negative",
    }

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def eta_squared_from_anova(anova_tbl: pd.DataFrame, effect_row: str) -> float:
        try:
            ss_total = anova_tbl["sum_sq"].sum()
            ss_eff = float(anova_tbl.loc[effect_row, "sum_sq"])
            return ss_eff / ss_total if ss_total > 0 else np.nan
        except Exception:
            return np.nan

    def sig_mark(p: float) -> str:
        return "SIGNIFICANT" if p < 0.05 else "n.s."

    def normalize_token(x):
        """
        데이터에 '_' / 대문자 / 공백 등이 섞여있을 수 있어서 표준화.
        예: "Very-Positive", "very_positive" -> "very-positive"
        """
        s = str(x).strip().lower()
        s = s.replace("_", "-").replace(" ", "-")
        return s

    def pick_accuracy_col(df: pd.DataFrame):
        """
        네 데이터 흐름상 overallAccuracy_y가 자주 등장해서 우선순위로 잡음.
        """
        candidates = ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def make_prism_tables(df: pd.DataFrame, cohort_out: str, tag: str):
        long_cols = ["participantId", "overallAccuracy", "aiConfidence", "aiAttitude", "attitude_group"]
        long_df = df[long_cols].copy()
        long_path = os.path.join(cohort_out, f"26v2-0_long_prism_{tag}.csv")
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

        cell_summary = (
            df.groupby(["aiConfidence", "attitude_group"])["overallAccuracy"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
        cell_path = os.path.join(cohort_out, f"26v2-0_cell_summary_mean_sem_n_{tag}.csv")
        cell_summary.to_csv(cell_path, index=False, encoding="utf-8-sig")

        return long_path, cell_path

    # -----------------------------
    # Run
    # -----------------------------
    print("\n" + "=" * 78)
    print(f"(26 v2 EN | MOBILE ONLY) Accuracy ~ AI Confidence × AI Attitude [{COHORT_TAG.upper()}]")
    print("=" * 78)

    cohort_out = os.path.join(OUTDIR, COHORT_TAG)
    ensure_dir(cohort_out)

    df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
    print(f"✅ Loaded: {FILE_PATH} (rows={len(df)})")

    # required base columns
    if "participantId" not in df.columns:
        raise ValueError("Missing 'participantId' column.")
    if "aiConfidence" not in df.columns:
        raise ValueError("Missing 'aiConfidence' column.")
    if "aiAttitude" not in df.columns:
        raise ValueError("Missing 'aiAttitude' column.")

    acc_col = pick_accuracy_col(df)
    if acc_col is None:
        raise ValueError("Missing accuracy column. Expected one of: overallAccuracy_y / overallAccuracy / overallAccuracy_x")

    d = df.copy()

    # ---- accuracy normalize ----
    d["overallAccuracy"] = pd.to_numeric(d[acc_col], errors="coerce")
    # 만약 0~1 비율이면 %로 변환
    mx = d["overallAccuracy"].max(skipna=True)
    if pd.notna(mx) and mx <= 1.0:
        d["overallAccuracy"] = d["overallAccuracy"] * 100.0
    print(f"✅ Using accuracy col: {acc_col} -> overallAccuracy (%)")

    # ---- normalize confidence/attitude tokens ----
    d["aiConfidence"] = d["aiConfidence"].apply(normalize_token)
    d["aiAttitude"] = d["aiAttitude"].apply(normalize_token)

    # ---- 3-level attitude group ----
    d["attitude_group"] = d["aiAttitude"].map(ATTITUDE_MAP)

    # ---- categorical ordering ----
    d["aiConfidence"] = pd.Categorical(d["aiConfidence"], categories=CONF_ORDER, ordered=True)
    d["aiAttitude"] = pd.Categorical(d["aiAttitude"], categories=ATT5_ORDER, ordered=True)
    d["attitude_group"] = pd.Categorical(d["attitude_group"], categories=ATT3_ORDER, ordered=True)

    # ---- clean ----
    d = d.dropna(subset=["participantId", "overallAccuracy", "aiConfidence", "aiAttitude", "attitude_group"]).copy()
    print(f"✅ N after cleaning: {len(d)}")

    # ---- save Prism ----
    long_path, cell_path = make_prism_tables(d, cohort_out, COHORT_TAG)
    print(f"✅ Saved Prism long: {long_path}")
    print(f"✅ Saved cell summary: {cell_path}")

    # -----------------------------
    # Plot (Y fixed 0..100)
    # -----------------------------
    config.apply_korean_plot_style()

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.pointplot(
        x="aiConfidence",
        y="overallAccuracy",
        hue="attitude_group",
        data=d,
        order=CONF_ORDER,
        hue_order=ATT3_ORDER,
        errorbar="se",
        ax=ax
    )

    ax.set_title("Accuracy by AI Confidence and Attitude (MOBILE)", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("AI Discrimination Confidence", fontsize=12)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)

    xticks = CONF_LABELS_KO if USE_KOREAN_XTICKS else CONF_LABELS_EN
    ax.set_xticklabels(xticks, rotation=15, ha="right")
    ax.legend(title="AI Attitude (3-level)")

    # 고정 y축
    ax.set_ylim(0, 100)

    sns.despine()
    plot_png = os.path.join(cohort_out, f"26v2-1_pointplot_confidence_x_attitude3_{COHORT_TAG}.png")
    plot_svg = os.path.join(cohort_out, f"26v2-1_pointplot_confidence_x_attitude3_{COHORT_TAG}.svg")
    fig.savefig(plot_png, dpi=300, bbox_inches="tight")
    fig.savefig(plot_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"✅ Saved plot: {plot_png} / {plot_svg}")

    # -----------------------------
    # (26v2-2) MAIN ANOVA: confidence(5) × attitude_group(3)
    # -----------------------------
    model3 = ols(
        "overallAccuracy ~ C(aiConfidence) + C(attitude_group) + C(aiConfidence):C(attitude_group)",
        data=d
    ).fit()
    anova3 = sm.stats.anova_lm(model3, typ=2)

    eta_conf = eta_squared_from_anova(anova3, "C(aiConfidence)")
    eta_att3 = eta_squared_from_anova(anova3, "C(attitude_group)")
    eta_int3 = eta_squared_from_anova(anova3, "C(aiConfidence):C(attitude_group)")

    p_conf = float(anova3.loc["C(aiConfidence)", "PR(>F)"])
    p_att3 = float(anova3.loc["C(attitude_group)", "PR(>F)"])
    p_int3 = float(anova3.loc["C(aiConfidence):C(attitude_group)", "PR(>F)"])

    print(f"\n--- (26v2-2) Two-way ANOVA (Attitude=3-level) [MOBILE] ---")
    print(anova3.round(6))
    print("\nInterpretation:")
    print(f"- Confidence main effect: p={p_conf:.6g} ({sig_mark(p_conf)}), eta^2={eta_conf:.4f}")
    print(f"- Attitude(3) main effect: p={p_att3:.6g} ({sig_mark(p_att3)}), eta^2={eta_att3:.4f}")
    print(f"- Interaction: p={p_int3:.6g} ({sig_mark(p_int3)}), eta^2={eta_int3:.4f}")

    anova3_csv = os.path.join(cohort_out, f"26v2-2_anova_attitude3_{COHORT_TAG}.csv")
    anova3.to_csv(anova3_csv, encoding="utf-8-sig")

    rep3_path = os.path.join(cohort_out, f"26v2-2_anova_attitude3_report_{COHORT_TAG}.txt")
    save_text(rep3_path, "\n".join([
        "=" * 72,
        "(26v2-2) Two-way ANOVA (Attitude = 3-level group) [MOBILE]",
        "=" * 72,
        f"N after cleaning = {len(d)}\n",
        "ANOVA table (Type II):",
        anova3.round(8).to_string(),
        "\nEffect sizes (eta^2):",
        f"- Confidence: {eta_conf:.8f}",
        f"- Attitude(3): {eta_att3:.8f}",
        f"- Interaction: {eta_int3:.8f}",
        "\nInterpretation:",
        f"- Confidence: p={p_conf:.6g} ({sig_mark(p_conf)})",
        f"- Attitude(3): p={p_att3:.6g} ({sig_mark(p_att3)})",
        f"- Interaction: p={p_int3:.6g} ({sig_mark(p_int3)})",
    ]))

    print(f"✅ Saved ANOVA(3): {anova3_csv}")
    print(f"✅ Saved report(3): {rep3_path}")

    # -----------------------------
    # (26v2-3) Sensitivity ANOVA: confidence(5) × aiAttitude(5)
    # -----------------------------
    model5 = ols(
        "overallAccuracy ~ C(aiConfidence) + C(aiAttitude) + C(aiConfidence):C(aiAttitude)",
        data=d
    ).fit()
    anova5 = sm.stats.anova_lm(model5, typ=2)

    eta_conf5 = eta_squared_from_anova(anova5, "C(aiConfidence)")
    eta_att5 = eta_squared_from_anova(anova5, "C(aiAttitude)")
    eta_int5 = eta_squared_from_anova(anova5, "C(aiConfidence):C(aiAttitude)")

    p_conf5 = float(anova5.loc["C(aiConfidence)", "PR(>F)"])
    p_att5 = float(anova5.loc["C(aiAttitude)", "PR(>F)"])
    p_int5 = float(anova5.loc["C(aiConfidence):C(aiAttitude)", "PR(>F)"])

    print(f"\n--- (26v2-3) Two-way ANOVA (Attitude=5-level) [MOBILE] ---")
    print(anova5.round(6))
    print("\nInterpretation (sensitivity):")
    print(f"- Confidence: p={p_conf5:.6g} ({sig_mark(p_conf5)}), eta^2={eta_conf5:.4f}")
    print(f"- Attitude(5): p={p_att5:.6g} ({sig_mark(p_att5)}), eta^2={eta_att5:.4f}")
    print(f"- Interaction: p={p_int5:.6g} ({sig_mark(p_int5)}), eta^2={eta_int5:.4f}")

    anova5_csv = os.path.join(cohort_out, f"26v2-3_anova_attitude5_{COHORT_TAG}.csv")
    anova5.to_csv(anova5_csv, encoding="utf-8-sig")

    rep5_path = os.path.join(cohort_out, f"26v2-3_anova_attitude5_report_{COHORT_TAG}.txt")
    save_text(rep5_path, anova5.round(8).to_string())
    print(f"✅ Saved ANOVA(5): {anova5_csv}")
    print(f"✅ Saved report(5): {rep5_path}")

    # -----------------------------
    # (26v2-4) Ordinal-coded regression (HC3)
    # -----------------------------
    conf_code = {k: i + 1 for i, k in enumerate(CONF_ORDER)}
    att_code = {
        "very-negative": 1,
        "negative": 2,
        "neutral": 3,
        "positive": 4,
        "very-positive": 5,
    }

    d2 = d.copy()
    d2["conf_num"] = d2["aiConfidence"].astype(str).map(conf_code)
    d2["atti_num"] = d2["aiAttitude"].astype(str).map(att_code)
    d2 = d2.dropna(subset=["conf_num", "atti_num", "overallAccuracy"]).copy()

    d2["conf_c"] = d2["conf_num"] - d2["conf_num"].mean()
    d2["atti_c"] = d2["atti_num"] - d2["atti_num"].mean()
    d2["inter"] = d2["conf_c"] * d2["atti_c"]

    reg = ols("overallAccuracy ~ conf_c + atti_c + inter", data=d2).fit(cov_type="HC3")

    print(f"\n--- (26v2-4) Ordinal-coded regression (HC3) [MOBILE] ---")
    print(reg.summary().tables[1])

    reg_txt = os.path.join(cohort_out, f"26v2-4_ordinal_regression_HC3_{COHORT_TAG}.txt")
    save_text(reg_txt, reg.summary().as_text())
    print(f"✅ Saved ordinal regression report: {reg_txt}")

    print("\n==================== (26 v2 EN | MOBILE ONLY) DONE ====================\n")


def _run_cell_122():
    # ==============================================================================
    # (27 v2 EN | MOBILE ONLY) Sex-stratified analysis:
    #   Accuracy ~ AI Confidence × AI Attitude
    # - MOBILE ONLY
    # - Run separately for Sex=male vs Sex=female
    # - Plot: confidence(5) × attitude_group(3) pointplot (mean ± SEM)
    # - Stats:
    #     (A) Two-way ANOVA: confidence × attitude_group (3-level)
    #     (B) Sensitivity ANOVA: confidence × aiAttitude (5-level)
    #     (C) Optional ANCOVA/Regression with covariates: age + avgRT (HC3)
    # - Save: plots/tables/reports + Prism-friendly long + cell mean/SEM/N
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # -----------------------------
    # Config
    # -----------------------------
    MAIN_FILE = config.MOBILE_AGE_FILTERED  # MOBILE ONLY
    OUTDIR = config.OUTPUTS_DIR / "outputs_27_sex_confidence_attitude_v2_en_mobile"

    CONF_ORDER = ["very-not-confident", "not-confident", "neutral", "confident", "very-confident"]
    CONF_LABELS_EN = ["Very unconfident", "Unconfident", "Neutral", "Confident", "Very confident"]

    ATT5_ORDER = ["very-positive", "positive", "neutral", "negative", "very-negative"]
    ATT3_ORDER = ["Positive", "Neutral", "Negative"]

    ATTITUDE_MAP = {
        "very-positive": "Positive",
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
        "very-negative": "Negative",
    }

    # y-axis upper bound by quantile (still clamp to 100)
    YLIM_Q = 0.99

    # Minimum N required to run ANOVA meaningfully
    MIN_N = 30

    # Optional covariates
    DO_COVARIATES = True

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def robust_upper(y: pd.Series, q=0.99):
        y = pd.to_numeric(y, errors="coerce").dropna()
        if len(y) == 0:
            return None
        return float(y.quantile(q))

    def eta_squared_from_anova(anova_tbl: pd.DataFrame, effect_row: str) -> float:
        try:
            ss_total = anova_tbl["sum_sq"].sum()
            ss_eff = float(anova_tbl.loc[effect_row, "sum_sq"])
            if ss_total > 0:
                return ss_eff / ss_total
        except Exception:
            pass
        return np.nan

    def sig_mark(p: float) -> str:
        return "✅ SIGNIFICANT" if p < 0.05 else "❌ n.s."

    def pick_rt_column(df: pd.DataFrame):
        for c in ["avgRT", "avgRT_overall", "meanRT", "rt_mean"]:
            if c in df.columns:
                return c
        return None

    def pick_accuracy_column(df: pd.DataFrame):
        # prefer already-prepared overallAccuracy; else fall back like earlier sections
        for c in ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x"]:
            if c in df.columns:
                return c
        return None

    def to_percent_if_needed(acc: pd.Series) -> pd.Series:
        s = pd.to_numeric(acc, errors="coerce")
        mx = float(np.nanmax(s.values)) if np.isfinite(np.nanmax(s.values)) else np.nan
        # if looks like 0~1 proportion, convert to %
        if np.isfinite(mx) and mx <= 1.5:
            return s * 100.0
        return s

    def standardize_sex_column(df: pd.DataFrame):
        # Input might be 'sex' or 'gender'
        if "sex" in df.columns:
            col = "sex"
        elif "gender" in df.columns:
            col = "gender"
        else:
            return None, None

        sex = df[col].astype(str).str.lower().str.strip()
        # keep only male/female
        sex = sex.where(sex.isin(["male", "female"]))
        return col, sex

    def make_prism_tables(df: pd.DataFrame, outdir: str, tag: str):
        long_cols = ["participantId", "sex", "overallAccuracy", "aiConfidence", "aiAttitude", "attitude_group", "age"]
        keep = [c for c in long_cols if c in df.columns]
        long_df = df[keep].copy()

        long_path = os.path.join(outdir, f"27v2-0_long_prism_{tag}.csv")
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

        cell_summary = (
            df.groupby(["aiConfidence", "attitude_group"], observed=False)["overallAccuracy"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
        cell_path = os.path.join(outdir, f"27v2-0_cell_summary_mean_sem_n_{tag}.csv")
        cell_summary.to_csv(cell_path, index=False, encoding="utf-8-sig")

        return long_path, cell_path

    def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # accuracy column detection + % conversion
        acc_col = pick_accuracy_column(d)
        if acc_col is None:
            raise ValueError("Missing accuracy column: expected one of overallAccuracy / overallAccuracy_y / overallAccuracy_x")

        d["overallAccuracy"] = to_percent_if_needed(d[acc_col])

        # sex standardization (input: gender or sex)
        sex_col, sex_series = standardize_sex_column(d)
        if sex_col is None:
            raise ValueError("Missing sex column: expected 'sex' or 'gender'")
        d["sex"] = sex_series

        # 3-level attitude_group
        d["attitude_group"] = d["aiAttitude"].astype(str).map(ATTITUDE_MAP)

        # ordered categoricals
        d["aiConfidence"] = pd.Categorical(d["aiConfidence"], categories=CONF_ORDER, ordered=True)
        d["aiAttitude"] = pd.Categorical(d["aiAttitude"], categories=ATT5_ORDER, ordered=True)
        d["attitude_group"] = pd.Categorical(d["attitude_group"], categories=ATT3_ORDER, ordered=True)

        needed = ["participantId", "sex", "overallAccuracy", "aiConfidence", "aiAttitude", "attitude_group"]
        d = d.dropna(subset=needed).copy()

        return d, acc_col, sex_col

    # -----------------------------
    # Analysis for a subgroup
    # -----------------------------
    def run_for_subgroup(dsub: pd.DataFrame, sex_tag: str, outdir: str):
        tag = f"mobile_{sex_tag}".lower()
        ensure_dir(outdir)

        print("\n" + "-" * 78)
        print(f"[MOBILE] Subgroup: {sex_tag} | N={len(dsub)}")
        print("-" * 78)

        if len(dsub) < MIN_N:
            msg = f"Too few samples (N={len(dsub)}) to run stable ANOVA. Skipping stats."
            print("❌ " + msg)
            save_text(os.path.join(outdir, f"27v2-skip_reason_{tag}.txt"), msg)
            return

        # Prism tables
        long_path, cell_path = make_prism_tables(dsub, outdir, tag)
        print(f"✅ Saved Prism long: {long_path}")
        print(f"✅ Saved cell summary: {cell_path}")

        # -------------------------
        # Plot
        # -------------------------
        config.apply_korean_plot_style()

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.pointplot(
            x="aiConfidence",
            y="overallAccuracy",
            hue="attitude_group",
            data=dsub,
            order=CONF_ORDER,
            hue_order=ATT3_ORDER,
            errorbar="se",
            ax=ax
        )

        ax.set_title(f"Accuracy by AI Confidence × Attitude (3-level) — Sex={sex_tag} [MOBILE]",
                     fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("AI Discrimination Confidence")
        ax.set_ylabel("Mean Accuracy (%)")

        # fix ticklabel warning: set ticks explicitly
        ax.set_xticks(np.arange(len(CONF_ORDER)))
        ax.set_xticklabels(CONF_LABELS_EN, rotation=15, ha="right")

        ax.legend(title="AI Attitude (3-level)")

        upper = robust_upper(dsub["overallAccuracy"], q=YLIM_Q)
        if upper and np.isfinite(upper):
            ax.set_ylim(0, min(100, upper))
        else:
            ax.set_ylim(0, 100)

        sns.despine()

        plot_png = os.path.join(outdir, f"27v2-1_pointplot_conf_x_att3_{tag}.png")
        plot_svg = os.path.join(outdir, f"27v2-1_pointplot_conf_x_att3_{tag}.svg")
        fig.savefig(plot_png, dpi=300, bbox_inches="tight")
        fig.savefig(plot_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print(f"✅ Saved plot: {plot_png} / {plot_svg}")

        # -------------------------
        # (A) Two-way ANOVA: 3-level attitude_group
        # -------------------------
        model3 = ols(
            "overallAccuracy ~ C(aiConfidence) + C(attitude_group) + C(aiConfidence):C(attitude_group)",
            data=dsub
        ).fit()
        anova3 = sm.stats.anova_lm(model3, typ=2)

        p_conf = float(anova3.loc["C(aiConfidence)", "PR(>F)"])
        p_att3 = float(anova3.loc["C(attitude_group)", "PR(>F)"])
        p_int3 = float(anova3.loc["C(aiConfidence):C(attitude_group)", "PR(>F)"])

        eta_conf = eta_squared_from_anova(anova3, "C(aiConfidence)")
        eta_att3 = eta_squared_from_anova(anova3, "C(attitude_group)")
        eta_int3 = eta_squared_from_anova(anova3, "C(aiConfidence):C(attitude_group)")

        print(f"\n[27v2-2] Two-way ANOVA (Attitude=3-level) — Sex={sex_tag} [MOBILE]")
        print(anova3.round(4))
        print("\nKey results (3-level):")
        print(f"- Confidence main effect: p={p_conf:.6g} {sig_mark(p_conf)} | eta^2={eta_conf:.4f}")
        print(f"- Attitude main effect:   p={p_att3:.6g} {sig_mark(p_att3)} | eta^2={eta_att3:.4f}")
        print(f"- Interaction:            p={p_int3:.6g} {sig_mark(p_int3)} | eta^2={eta_int3:.4f}")

        anova3_csv = os.path.join(outdir, f"27v2-2_anova_att3_{tag}.csv")
        anova3.to_csv(anova3_csv, encoding="utf-8-sig")

        rep3_path = os.path.join(outdir, f"27v2-2_anova_att3_report_{tag}.txt")
        save_text(rep3_path, anova3.round(6).to_string())

        # -------------------------
        # (B) Sensitivity: 5-level aiAttitude
        # -------------------------
        model5 = ols(
            "overallAccuracy ~ C(aiConfidence) + C(aiAttitude) + C(aiConfidence):C(aiAttitude)",
            data=dsub
        ).fit()
        anova5 = sm.stats.anova_lm(model5, typ=2)

        p_conf5 = float(anova5.loc["C(aiConfidence)", "PR(>F)"])
        p_att5 = float(anova5.loc["C(aiAttitude)", "PR(>F)"])
        p_int5 = float(anova5.loc["C(aiConfidence):C(aiAttitude)", "PR(>F)"])

        eta_conf5 = eta_squared_from_anova(anova5, "C(aiConfidence)")
        eta_att5 = eta_squared_from_anova(anova5, "C(aiAttitude)")
        eta_int5 = eta_squared_from_anova(anova5, "C(aiConfidence):C(aiAttitude)")

        print(f"\n[27v2-3] Two-way ANOVA (Attitude=5-level) — Sex={sex_tag} [MOBILE]")
        print(anova5.round(4))
        print("\nKey results (5-level sensitivity):")
        print(f"- Confidence main effect: p={p_conf5:.6g} {sig_mark(p_conf5)} | eta^2={eta_conf5:.4f}")
        print(f"- Attitude main effect:   p={p_att5:.6g} {sig_mark(p_att5)} | eta^2={eta_att5:.4f}")
        print(f"- Interaction:            p={p_int5:.6g} {sig_mark(p_int5)} | eta^2={eta_int5:.4f}")

        anova5_csv = os.path.join(outdir, f"27v2-3_anova_att5_{tag}.csv")
        anova5.to_csv(anova5_csv, encoding="utf-8-sig")

        rep5_path = os.path.join(outdir, f"27v2-3_anova_att5_report_{tag}.txt")
        save_text(rep5_path, anova5.round(6).to_string())

        # -------------------------
        # (C) Optional covariates (HC3)
        # -------------------------
        if DO_COVARIATES:
            rt_col = pick_rt_column(dsub)
            cov_terms = ["age"]
            if rt_col is not None:
                cov_terms.append(rt_col)

            # build formula only with existing covariates
            existing = [c for c in cov_terms if c in dsub.columns]
            if len(existing) >= 1:
                cov_formula = (
                    "overallAccuracy ~ C(aiConfidence) + C(attitude_group) + "
                    "C(aiConfidence):C(attitude_group) + " + " + ".join(existing)
                )
                cov_model = ols(cov_formula, data=dsub).fit(cov_type="HC3")
                print(f"\n[27v2-4] Covariate model (HC3) — Sex={sex_tag} [MOBILE]")
                print(f"Formula: {cov_formula}")
                print(cov_model.summary().tables[1])

                cov_txt = os.path.join(outdir, f"27v2-4_covariate_model_HC3_{tag}.txt")
                save_text(cov_txt, cov_model.summary().as_text())
            else:
                print("\n[27v2-4] Covariate model skipped (no usable covariates).")

        print(f"\n==================== DONE — Sex={sex_tag} [MOBILE] ====================")

    # -----------------------------
    # Main
    # -----------------------------
    if __name__ == "__main__":
        ensure_dir(OUTDIR)

        print("\n" + "=" * 86)
        print("(27 v2 EN | MOBILE ONLY) Sex-stratified: Accuracy ~ Confidence × Attitude")
        print("=" * 86)

        if not os.path.exists(MAIN_FILE):
            raise FileNotFoundError(f"Missing file: {MAIN_FILE}")

        df0 = pd.read_csv(MAIN_FILE, encoding="utf-8-sig")
        d, acc_col_used, sex_col_used = clean_and_prepare(df0)

        print(f"✅ Loaded: {MAIN_FILE} (rows={len(df0)})")
        print(f"✅ Using accuracy col: {acc_col_used} -> overallAccuracy (%)")
        print(f"✅ Using sex source col: {sex_col_used} -> standardized to 'sex'")
        print(f"✅ N after cleaning: {len(d)}")

        # Split by sex
        male_df = d[d["sex"] == "male"].copy()
        female_df = d[d["sex"] == "female"].copy()

        # Run
        run_for_subgroup(male_df, "male", os.path.join(OUTDIR, "male"))
        run_for_subgroup(female_df, "female", os.path.join(OUTDIR, "female"))

        print("\n==================== (27 v2 EN | MOBILE ONLY) ALL DONE ====================")


def _run_cell_126():
    # ==============================================================================
    # (28 v2 EN | MOBILE ONLY) RT patterns by Accuracy Group (High vs Low)
    # - Participant-level: analysis_data_first_timers.csv
    # - Trial-level: responses_export.csv
    # - MOBILE only (filter by deviceType if available; otherwise infer from participant file if possible)
    # - Practice trials removed
    # - RT: auto-fix sec->ms if median < 20
    # - Visualization: trial-level boxplot on log RT (faceted by accuracy_group)
    # - Stats (stable): subject-level mean_logRT ~ accuracy_group * ImageType * Response
    # - Sensitivity: mean_logRT ~ overallAccuracy (continuous) * ImageType * Response
    # - Saves Prism-friendly tables + ANOVA tables + plots
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as mticker

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # -----------------------------
    # Config
    # -----------------------------
    OUTDIR = config.OUTPUTS_DIR / "outputs_28_rt_patterns_by_accuracy_group_v2_en_mobile_only"

    PART_FILE = config.FIRST_TIMERS_ALL   # participant-level
    RESP_FILE = config.RAW_RESPONSES             # trial-level

    # Optional: if participant file has device column, restrict to mobile participants too
    PART_DEVICE_CANDIDATES = ["deviceType", "device", "platform", "cohort"]

    AGE_MIN, AGE_MAX = 15, 79

    AUTO_UNIT_FIX = True
    RAW_RT_Q = 0.99  # not used in log plot; kept for potential raw RT plots

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): 
        os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def is_practice_trial(x) -> bool:
        s = str(x).lower()
        return s.startswith("practice")

    def infer_and_fix_rt_unit(rt_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(rt_series, errors="coerce")
        med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else np.nan
        if np.isnan(med):
            return s
        if AUTO_UNIT_FIX and med < 20:  # likely seconds
            return s * 1000.0
        return s

    def log_tick_formatter(val, pos=None):
        # val is log(ms)
        try:
            return f"{int(np.exp(val)):,}"
        except Exception:
            return ""

    def sig_mark(p): 
        return "✅ SIGNIFICANT" if p < 0.05 else "❌ n.s."

    def eta_squared_from_anova(anova_tbl: pd.DataFrame, row: str) -> float:
        try:
            ss_total = anova_tbl["sum_sq"].sum()
            ss_eff = float(anova_tbl.loc[row, "sum_sq"])
            if ss_total > 0:
                return ss_eff / ss_total
        except Exception:
            pass
        return np.nan

    def pick_accuracy_column(df: pd.DataFrame) -> str:
        candidates = [
            "overallAccuracy", "overallAccuracy_y", "overallAccuracy_x",
            "overall_accuracy", "accuracy", "overallAcc", "acc"
        ]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            "No accuracy column found. "
            f"Available columns (first 60): {list(df.columns)[:60]}"
        )

    def pick_age_column(df: pd.DataFrame) -> str:
        candidates = ["age", "Age", "participantAge"]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            "No age column found. "
            f"Available columns (first 60): {list(df.columns)[:60]}"
        )

    def pick_sex_column(df: pd.DataFrame) -> str:
        candidates = ["sex", "Sex", "gender", "Gender"]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def pick_part_device_column(df: pd.DataFrame):
        for c in PART_DEVICE_CANDIDATES:
            if c in df.columns:
                return c
        return None

    def normalize_device(x: str) -> str:
        s = str(x).strip().lower()
        if "mobile" in s or "phone" in s:
            return "mobile"
        if "web" in s or "desktop" in s or "pc" in s:
            return "web"
        return s

    def prep_mobile_data():
        # ---- Load participant-level ----
        df_part = pd.read_csv(PART_FILE, encoding="utf-8-sig")
        age_col = pick_age_column(df_part)
        acc_col = pick_accuracy_column(df_part)
        sex_col = pick_sex_column(df_part)

        # standardize columns
        df_part = df_part.rename(columns={age_col: "age", acc_col: "overallAccuracy"}).copy()
        if sex_col is not None and sex_col != "sex":
            df_part = df_part.rename(columns={sex_col: "sex"}).copy()

        # Age filter
        df_part["age"] = pd.to_numeric(df_part["age"], errors="coerce")
        df_part = df_part[(df_part["age"] >= AGE_MIN) & (df_part["age"] <= AGE_MAX)].copy()

        # Accuracy might be proportion -> convert to %
        df_part["overallAccuracy"] = pd.to_numeric(df_part["overallAccuracy"], errors="coerce")
        if np.nanmax(df_part["overallAccuracy"].values) <= 1.0:
            df_part["overallAccuracy"] = df_part["overallAccuracy"] * 100.0

        # If participant file has device, restrict to mobile participants
        part_dev_col = pick_part_device_column(df_part)
        if part_dev_col is not None:
            df_part[part_dev_col] = df_part[part_dev_col].apply(normalize_device)
            df_part_mobile = df_part[df_part[part_dev_col] == "mobile"].copy()
        else:
            df_part_mobile = df_part.copy()

        # Required columns
        need_p = ["participantId", "age", "overallAccuracy"]
        miss_p = [c for c in need_p if c not in df_part_mobile.columns]
        if miss_p:
            raise ValueError(f"Missing columns in {PART_FILE} after standardization: {miss_p}")

        # ---- Load trial-level ----
        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")

        # filter to participants
        pid_set = set(df_part_mobile["participantId"].dropna().unique())
        rt_df = df_resp[df_resp["participantId"].isin(pid_set)].copy()

        # filter to mobile trials if deviceType exists in responses
        if "deviceType" in rt_df.columns:
            rt_df["deviceType"] = rt_df["deviceType"].apply(normalize_device)
            rt_df = rt_df[rt_df["deviceType"] == "mobile"].copy()

        # remove practice
        if "trial" in rt_df.columns:
            rt_df = rt_df[~rt_df["trial"].apply(is_practice_trial)].copy()

        # needed trial columns
        need_r = ["participantId", "rt", "isCorrect", "imageType"]
        miss_r = [c for c in need_r if c not in rt_df.columns]
        if miss_r:
            raise ValueError(f"Missing columns in {RESP_FILE} after filtering: {miss_r}")

        # fix RT units
        rt_df["rt"] = infer_and_fix_rt_unit(rt_df["rt"])
        rt_df["rt"] = pd.to_numeric(rt_df["rt"], errors="coerce")
        rt_df = rt_df.dropna(subset=["rt"]).copy()

        # merge participant vars
        demo_cols = ["participantId", "age", "overallAccuracy"]
        if "sex" in df_part_mobile.columns:
            demo_cols.append("sex")
        demographics = df_part_mobile[demo_cols].copy()

        plot_df = pd.merge(rt_df, demographics, on="participantId", how="inner")

        # helper columns
        plot_df["log_rt"] = np.log(plot_df["rt"].clip(lower=1e-6))
        plot_df["Response"] = np.where(plot_df["isCorrect"].astype(bool), "Correct", "Incorrect")

        plot_df["Image Type"] = plot_df["imageType"].astype(str).str.lower().apply(
            lambda x: "Real" if x == "real" else ("AI" if "ai" in x else "Other")
        )
        plot_df = plot_df[plot_df["Image Type"].isin(["Real", "AI"])].copy()

        # median split accuracy (participant-level median, to avoid trial weighting)
        pid_acc = plot_df.groupby("participantId")["overallAccuracy"].mean()
        median_acc = float(pid_acc.median())

        plot_df["accuracy_group"] = plot_df["participantId"].map(
            lambda pid: "High Accuracy" if pid_acc.loc[pid] >= median_acc else "Low Accuracy"
        )

        return plot_df, median_acc

    def run_mobile(df_trials: pd.DataFrame, median_acc: float, outdir: str):
        ensure_dir(outdir)
        cohort_name = "mobile"

        # Save trial-level long (Prism)
        long_path = os.path.join(outdir, f"28v2-0_long_trials_{cohort_name}.csv")
        df_trials.to_csv(long_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved trial-level long table: {long_path}")

        # Subject-level condition means
        subj = (
            df_trials
            .groupby(["participantId", "accuracy_group", "Image Type", "Response"])["log_rt"]
            .mean()
            .reset_index()
            .rename(columns={"log_rt": "mean_log_rt"})
        )

        # add overallAccuracy (continuous) per participant
        acc_map = df_trials.groupby("participantId")["overallAccuracy"].mean()
        subj["overallAccuracy"] = subj["participantId"].map(acc_map)

        subj_path = os.path.join(outdir, f"28v2-0_subject_means_{cohort_name}.csv")
        subj.to_csv(subj_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved subject-level means table: {subj_path}")

        # -------------------------
        # Visualization: trial-level boxplot on log RT
        # -------------------------
        config.apply_korean_plot_style()

        g = sns.catplot(
            data=df_trials,
            x="Image Type", y="log_rt", hue="Response",
            col="accuracy_group",
            kind="box",
            height=6, aspect=1
        )

        for ax in g.axes.flat:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.grid(axis="y", linestyle=":", alpha=0.7)
            ax.set_xlabel("Image Type")
            ax.set_ylabel("Reaction Time (ms)")

        g.fig.suptitle(
            f"RT Patterns by Accuracy Group (participant-median split={median_acc:.2f}) — MOBILE",
            fontsize=16, y=1.03
        )
        g.set_titles("Group: {col_name}")

        plot_png = os.path.join(outdir, f"28v2-1_boxplot_logrt_{cohort_name}.png")
        plot_svg = os.path.join(outdir, f"28v2-1_boxplot_logrt_{cohort_name}.svg")
        plt.savefig(plot_png, dpi=300, bbox_inches="tight")
        plt.savefig(plot_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"✅ Saved plot: {plot_png} / {plot_svg}")

        # -------------------------
        # Stats (A): ANOVA on subject means (median split)
        # mean_log_rt ~ accuracy_group * ImageType * Response
        # -------------------------
        print(f"\n--- (28v2-2) Subject-mean ANOVA (median split) — MOBILE ---")
        model_a = ols("mean_log_rt ~ C(accuracy_group) * C(Q('Image Type')) * C(Response)", data=subj).fit()
        anova_a = sm.stats.anova_lm(model_a, typ=2)
        print(anova_a.round(4))

        key_rows = [
            "C(accuracy_group)",
            "C(Q('Image Type'))",
            "C(Response)",
            "C(accuracy_group):C(Q('Image Type'))",
            "C(accuracy_group):C(Response)",
            "C(Q('Image Type')):C(Response)",
            "C(accuracy_group):C(Q('Image Type')):C(Response)",
        ]

        lines = []
        lines.append("="*78)
        lines.append("(28v2-2) Subject-mean ANOVA (median split) — MOBILE")
        lines.append("="*78)
        lines.append(f"Participant-median accuracy used for split: {median_acc:.4f}")
        lines.append(f"N trials: {len(df_trials)} | N participants: {subj['participantId'].nunique()}")
        lines.append("")
        lines.append(anova_a.round(6).to_string())
        lines.append("\nKey effects (p-values + eta^2):")
        for r in key_rows:
            if r in anova_a.index:
                p = float(anova_a.loc[r, "PR(>F)"])
                eta = eta_squared_from_anova(anova_a, r)
                lines.append(f"- {r}: p={p:.6g} {sig_mark(p)} | eta^2={eta:.4f}")

        anova_a_csv = os.path.join(outdir, f"28v2-2_anova_subject_means_mediansplit_{cohort_name}.csv")
        anova_a.to_csv(anova_a_csv, encoding="utf-8-sig")
        anova_a_txt = os.path.join(outdir, f"28v2-2_anova_subject_means_mediansplit_{cohort_name}.txt")
        save_text(anova_a_txt, "\n".join(lines))
        print(f"✅ Saved ANOVA table/report: {anova_a_csv}, {anova_a_txt}")

        # -------------------------
        # Stats (B): Sensitivity with continuous accuracy
        # mean_log_rt ~ overallAccuracy * ImageType * Response
        # -------------------------
        print(f"\n--- (28v2-3) Sensitivity model (continuous accuracy) — MOBILE ---")
        model_b = ols("mean_log_rt ~ overallAccuracy * C(Q('Image Type')) * C(Response)", data=subj).fit()
        anova_b = sm.stats.anova_lm(model_b, typ=2)
        print(anova_b.round(4))

        key_rows_b = [
            "overallAccuracy",
            "overallAccuracy:C(Q('Image Type'))",
            "overallAccuracy:C(Response)",
            "overallAccuracy:C(Q('Image Type')):C(Response)",
        ]

        lines_b = []
        lines_b.append("="*78)
        lines_b.append("(28v2-3) Sensitivity ANOVA (continuous accuracy) — MOBILE")
        lines_b.append("="*78)
        lines_b.append(f"N participants: {subj['participantId'].nunique()}")
        lines_b.append("")
        lines_b.append(anova_b.round(6).to_string())
        lines_b.append("\nKey accuracy-related effects:")
        for r in key_rows_b:
            if r in anova_b.index:
                p = float(anova_b.loc[r, "PR(>F)"])
                eta = eta_squared_from_anova(anova_b, r)
                lines_b.append(f"- {r}: p={p:.6g} {sig_mark(p)} | eta^2={eta:.4f}")

        anova_b_csv = os.path.join(outdir, f"28v2-3_anova_subject_means_contacc_{cohort_name}.csv")
        anova_b.to_csv(anova_b_csv, encoding="utf-8-sig")
        anova_b_txt = os.path.join(outdir, f"28v2-3_anova_subject_means_contacc_{cohort_name}.txt")
        save_text(anova_b_txt, "\n".join(lines_b))
        print(f"✅ Saved sensitivity ANOVA table/report: {anova_b_csv}, {anova_b_txt}")

        print(f"\n==================== (28 v2 EN | MOBILE ONLY) DONE ====================\n")


    # -----------------------------
    # Main
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(28 v2 EN | MOBILE ONLY) RT Patterns by Accuracy Group (High vs Low)")
        print("==============================================================================\n")

        ensure_dir(OUTDIR)

        plot_df, median_acc = prep_mobile_data()
        print(f"✅ Prepared MOBILE data.")
        print(f"✅ Participant-median accuracy = {median_acc:.2f}")
        print(f"✅ Trials (filtered): {len(plot_df)} | Participants: {plot_df['participantId'].nunique()}")

        run_mobile(plot_df, median_acc, os.path.join(OUTDIR, "mobile"))

        print("\n==================== (28 v2 EN | MOBILE ONLY) ALL DONE ====================")


def _run_cell_130():
    # ==============================================================================
    # (29 v2 EN | MOBILE ONLY) Speed–Accuracy Tradeoff (SAT): RT–Accuracy relationship
    # ------------------------------------------------------------------------------
    # Goal (MOBILE ONLY):
    #   1) Participant-level: Does average RT predict overall accuracy?
    #   2) Condition-level: Do RT–Accuracy relationships differ by Image Type (Real vs AI)?
    #
    # Inputs:
    #   - analysis_data_mobile_age_filtered_20_69.csv  (participant-level)
    #   - responses_export.csv                        (trial-level)
    #
    # Outputs:
    #   outputs_29_speed_accuracy_tradeoff_v2_en_mobile_only/mobile/
    #     - 29v2-0_participant_level_table_mobile.csv
    #     - 29v2-1_participant_regression_coeffs_mobile.csv
    #     - 29v2-1_participant_regression_report_mobile.txt
    #     - 29v2-2_condition_level_table_mobile.csv
    #     - 29v2-3_condition_regression_coeffs_mobile.csv
    #     - 29v2-3_condition_regression_report_mobile.txt
    #     - 29v2-4_scatter_accuracy_vs_avgRT_mobile.png/.svg
    #     - 29v2-5_scatter_accuracy_vs_rt_by_imageType_mobile.png/.svg
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # -----------------------------
    # Config
    # -----------------------------
    OUTDIR = config.OUTPUTS_DIR / "outputs_29_speed_accuracy_tradeoff_v2_en_mobile_only"
    COHORT_TAG = "mobile"

    PART_FILE = config.MOBILE_AGE_FILTERED
    RESP_FILE = config.RAW_RESPONSES

    AGE_MIN, AGE_MAX = 15, 79

    AUTO_UNIT_FIX = True
    RT_Q_CLIP = 0.99   # x-limit clipping for plots

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def is_practice_trial(x) -> bool:
        return str(x).lower().startswith("practice")

    def infer_and_fix_rt_unit(rt_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(rt_series, errors="coerce")
        med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else np.nan
        if np.isnan(med):
            return s
        if AUTO_UNIT_FIX and med < 20:  # likely seconds
            return s * 1000.0
        return s

    def sig_mark(p):
        try:
            return "✅ SIGNIFICANT (p<0.05)" if float(p) < 0.05 else "❌ n.s. (p>=0.05)"
        except Exception:
            return "?"

    def pick_first_existing(df, candidates, required=True):
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise ValueError(f"Missing required columns. Tried: {candidates}")
        return None

    def normalize_accuracy_to_percent(x: pd.Series) -> pd.Series:
        s = pd.to_numeric(x, errors="coerce")
        # if looks like 0~1, convert to %
        med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else np.nan
        if np.isfinite(med) and med <= 1.2:
            return s * 100.0
        return s

    # -----------------------------
    # Prep
    # -----------------------------
    def prep_mobile_data():
        # 1) Participant-level
        df_part = pd.read_csv(PART_FILE, encoding="utf-8-sig")

        id_col  = pick_first_existing(df_part, ["participantId", "participant_id", "id"])
        age_col = pick_first_existing(df_part, ["age", "Age"])
        sex_col = pick_first_existing(df_part, ["gender", "sex", "Gender", "Sex"])
        acc_col = pick_first_existing(df_part, ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x"])

        df_part = df_part.rename(columns={id_col: "participantId", age_col: "age", sex_col: "sex", acc_col: "overallAccuracy"})
        df_part["age"] = pd.to_numeric(df_part["age"], errors="coerce")
        df_part = df_part[(df_part["age"] >= AGE_MIN) & (df_part["age"] <= AGE_MAX)].copy()

        df_part["overallAccuracy"] = normalize_accuracy_to_percent(df_part["overallAccuracy"])
        df_part["sex"] = df_part["sex"].astype(str).str.lower()

        # keep male/female only (optional but usually cleaner)
        df_part = df_part[df_part["sex"].isin(["male", "female"])].copy()

        # 2) Trial-level
        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")

        need_r = ["participantId", "rt", "imageType", "isCorrect", "trial"]
        miss_r = [c for c in need_r if c not in df_resp.columns]
        if miss_r:
            raise ValueError(f"Missing columns in {RESP_FILE}: {miss_r}")

        pid_set = set(df_part["participantId"].dropna().unique())

        rt_df = df_resp[df_resp["participantId"].isin(pid_set)].copy()
        rt_df = rt_df[~rt_df["trial"].apply(is_practice_trial)].copy()

        rt_df["rt"] = infer_and_fix_rt_unit(rt_df["rt"])
        rt_df = rt_df.dropna(subset=["rt"])

        # Normalize image type
        rt_df["image_kind"] = rt_df["imageType"].astype(str).str.lower().apply(
            lambda x: "Real" if x == "real" else ("AI" if "ai" in x else "Other")
        )
        rt_df = rt_df[rt_df["image_kind"].isin(["Real", "AI"])].copy()

        # isCorrect -> 0/1
        rt_df["isCorrect"] = rt_df["isCorrect"].astype(int)

        # 3) Participant-level average RT
        avg_rt_overall = rt_df.groupby("participantId")["rt"].mean().reset_index(name="avgRT_overall")
        avg_rt_by_kind = rt_df.groupby(["participantId", "image_kind"])["rt"].mean().unstack().reset_index()
        avg_rt_by_kind = avg_rt_by_kind.rename(columns={"Real": "avgRT_real", "AI": "avgRT_ai"})

        df_p = df_part[["participantId", "age", "sex", "overallAccuracy"]].copy()
        df_p = df_p.merge(avg_rt_overall, on="participantId", how="inner")
        df_p = df_p.merge(avg_rt_by_kind, on="participantId", how="left")

        # 4) Condition-level (participant × image_kind): RT + accuracy within kind
        cond = (
            rt_df.groupby(["participantId", "image_kind"])
            .agg(rt_mean=("rt", "mean"), acc_mean=("isCorrect", "mean"), n_trials=("rt", "size"))
            .reset_index()
        )
        cond["acc_mean"] = cond["acc_mean"] * 100.0
        cond = cond.merge(df_p[["participantId", "age", "sex", "overallAccuracy"]], on="participantId", how="left")

        return df_p, cond

    # -----------------------------
    # Run (MOBILE ONLY)
    # -----------------------------
    def run_mobile(df_p: pd.DataFrame, cond: pd.DataFrame):
        out = os.path.join(OUTDIR, COHORT_TAG)
        ensure_dir(out)

        # Save Prism-ready tables
        p_path = os.path.join(out, f"29v2-0_participant_level_table_{COHORT_TAG}.csv")
        df_p.to_csv(p_path, index=False, encoding="utf-8-sig")

        c_path = os.path.join(out, f"29v2-2_condition_level_table_{COHORT_TAG}.csv")
        cond.to_csv(c_path, index=False, encoding="utf-8-sig")

        print("\n==============================================================================")
        print(f"(29 v2 EN | MOBILE ONLY) Speed–Accuracy Tradeoff (SAT)")
        print("==============================================================================")
        print(f"✅ Saved participant table: {p_path}")
        print(f"✅ Saved condition table:   {c_path}")
        print(f"✅ N participants: {df_p['participantId'].nunique()}")
        print(f"✅ N condition rows: {len(cond)}")

        # -------------------------
        # (29v2-1) Participant-level regression
        # overallAccuracy ~ avgRT_overall + age + sex
        # -------------------------
        df_reg = df_p.dropna(subset=["overallAccuracy", "avgRT_overall", "age", "sex"]).copy()

        model = ols("overallAccuracy ~ avgRT_overall + age + C(sex)", data=df_reg).fit(cov_type="HC3")
        anova = sm.stats.anova_lm(model, typ=2)

        p_rt = float(anova.loc["avgRT_overall", "PR(>F)"]) if "avgRT_overall" in anova.index else np.nan

        print("\n--- (29v2-1) Participant-level: overallAccuracy ~ avgRT_overall + age + sex (HC3) ---")
        print(f"RT effect p-value: {p_rt:.6g}  {sig_mark(p_rt)}")
        print(f"R2={model.rsquared:.4f}, Adj.R2={model.rsquared_adj:.4f}")

        coef = pd.DataFrame({
            "term": model.params.index,
            "coef": model.params.values,
            "se": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
        })
        coef_csv = os.path.join(out, f"29v2-1_participant_regression_coeffs_{COHORT_TAG}.csv")
        coef.to_csv(coef_csv, index=False, encoding="utf-8-sig")

        rep_txt = os.path.join(out, f"29v2-1_participant_regression_report_{COHORT_TAG}.txt")
        lines = []
        lines.append("="*78)
        lines.append(f"(29v2-1) Participant-level regression — {COHORT_TAG.upper()}")
        lines.append("="*78)
        lines.append("Model: overallAccuracy ~ avgRT_overall + age + C(sex)  (HC3 robust SE)")
        lines.append(f"N={len(df_reg)} participants")
        lines.append(f"R2={model.rsquared:.6f}, Adj.R2={model.rsquared_adj:.6f}")
        lines.append("\nANOVA (Type II):")
        lines.append(anova.round(6).to_string())
        lines.append("\nCoefficients:")
        lines.append(coef.round(6).to_string(index=False))
        save_text(rep_txt, "\n".join(lines))
        print(f"✅ Saved regression outputs: {coef_csv}, {rep_txt}")

        # Plot: accuracy vs avgRT_overall
        config.apply_korean_plot_style()

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.regplot(data=df_reg, x="avgRT_overall", y="overallAccuracy", scatter_kws={"alpha": 0.2}, ax=ax)
        ax.set_title(f"(29v2-4) Accuracy vs Avg RT (overall) — MOBILE\nRT effect: p={p_rt:.3g}", fontsize=14)
        ax.set_xlabel("Average RT (ms)")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_ylim(0, 100)

        x_max = float(df_reg["avgRT_overall"].quantile(RT_Q_CLIP))
        if np.isfinite(x_max) and x_max > 0:
            ax.set_xlim(0, x_max)

        out_png = os.path.join(out, f"29v2-4_scatter_accuracy_vs_avgRT_{COHORT_TAG}.png")
        out_svg = os.path.join(out, f"29v2-4_scatter_accuracy_vs_avgRT_{COHORT_TAG}.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved plot: {out_png} / {out_svg}")

        # -------------------------
        # (29v2-3) Condition-level interaction:
        # acc_mean ~ rt_mean * image_kind + age + sex
        # -------------------------
        cond_reg = cond.dropna(subset=["acc_mean", "rt_mean", "image_kind", "age", "sex"]).copy()

        model2 = ols("acc_mean ~ rt_mean * C(image_kind) + age + C(sex)", data=cond_reg).fit(cov_type="HC3")
        anova2 = sm.stats.anova_lm(model2, typ=2)

        key = "rt_mean:C(image_kind)"
        p_int = float(anova2.loc[key, "PR(>F)"]) if key in anova2.index else np.nan

        print("\n--- (29v2-3) Condition-level: acc_mean ~ rt_mean * image_kind + age + sex (HC3) ---")
        print(f"RT×ImageType interaction p-value: {p_int:.6g}  {sig_mark(p_int)}")
        print(f"R2={model2.rsquared:.4f}, Adj.R2={model2.rsquared_adj:.4f}")

        coef2 = pd.DataFrame({
            "term": model2.params.index,
            "coef": model2.params.values,
            "se": model2.bse.values,
            "t": model2.tvalues.values,
            "p": model2.pvalues.values,
        })
        coef2_csv = os.path.join(out, f"29v2-3_condition_regression_coeffs_{COHORT_TAG}.csv")
        coef2.to_csv(coef2_csv, index=False, encoding="utf-8-sig")

        rep2_txt = os.path.join(out, f"29v2-3_condition_regression_report_{COHORT_TAG}.txt")
        lines2 = []
        lines2.append("="*78)
        lines2.append(f"(29v2-3) Condition-level regression — {COHORT_TAG.upper()}")
        lines2.append("="*78)
        lines2.append("Model: acc_mean ~ rt_mean * C(image_kind) + age + C(sex)  (HC3 robust SE)")
        lines2.append(f"N={len(cond_reg)} condition rows")
        lines2.append(f"R2={model2.rsquared:.6f}, Adj.R2={model2.rsquared_adj:.6f}")
        lines2.append("\nANOVA (Type II):")
        lines2.append(anova2.round(6).to_string())
        lines2.append("\nCoefficients:")
        lines2.append(coef2.round(6).to_string(index=False))
        save_text(rep2_txt, "\n".join(lines2))
        print(f"✅ Saved condition regression outputs: {coef2_csv}, {rep2_txt}")

        # Plot: within-kind accuracy vs within-kind RT, separated by image_kind
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=cond_reg, x="rt_mean", y="acc_mean", hue="image_kind", alpha=0.25, ax=ax)

        # regression lines per kind (no scatter)
        for kind in ["Real", "AI"]:
            sub = cond_reg[cond_reg["image_kind"] == kind]
            if len(sub) >= 5:
                sns.regplot(data=sub, x="rt_mean", y="acc_mean", scatter=False, ax=ax)

        ax.set_title(f"(29v2-5) Within-kind Accuracy vs RT by Image Type — MOBILE\nInteraction p={p_int:.3g}", fontsize=14)
        ax.set_xlabel("Mean RT within Image Type (ms)")
        ax.set_ylabel("Accuracy within Image Type (%)")
        ax.set_ylim(0, 100)

        x_max2 = float(cond_reg["rt_mean"].quantile(RT_Q_CLIP))
        if np.isfinite(x_max2) and x_max2 > 0:
            ax.set_xlim(0, x_max2)

        out_png2 = os.path.join(out, f"29v2-5_scatter_accuracy_vs_rt_by_imageType_{COHORT_TAG}.png")
        out_svg2 = os.path.join(out, f"29v2-5_scatter_accuracy_vs_rt_by_imageType_{COHORT_TAG}.svg")
        fig.savefig(out_png2, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg2, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved plot: {out_png2} / {out_svg2}")

        print("\n==================== (29 v2 EN | MOBILE ONLY) DONE ====================\n")

    # -----------------------------
    # Main
    # -----------------------------
    if __name__ == "__main__":
        ensure_dir(OUTDIR)

        if not os.path.exists(PART_FILE):
            raise FileNotFoundError(f"Missing participant file: {PART_FILE}")
        if not os.path.exists(RESP_FILE):
            raise FileNotFoundError(f"Missing responses file: {RESP_FILE}")

        df_p, cond = prep_mobile_data()
        print(f"✅ Prepared MOBILE data. Participants={df_p['participantId'].nunique()}, Condition rows={len(cond)}")
        run_mobile(df_p, cond)


def _run_cell_134():
    # ==============================================================================
    # (30 v2.1 | MOBILE ONLY) Participation Date (time trend) vs Accuracy
    # - Robust timestamp parsing
    # - Auto-detect accuracy column: overallAccuracy / overallAccuracy_y / overallAccuracy_x
    # - OLS with covariates (HC3): date_ordinal + age + sex (+ avgRT if exists)
    # - By age-group Pearson + Holm correction
    # - Save Prism table + results + plots (y-axis fixed 0~100)
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")

    # -----------------------------
    # Config
    # -----------------------------
    IN_FILE = config.MOBILE_AGE_FILTERED
    OUTDIR = config.OUTPUTS_DIR / "outputs_30_date_accuracy_v2_1_mobile_only"

    AGE_BINS = [14, 19, 29, 39, 49, 59, 69, 79]
    AGE_LABELS = ["10s", "20s", "30s", "40s", "50s", "60s", "70s"]

    SCATTER_ALPHA = 0.20
    YLIM = (0, 100)

    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sig_mark(p):
        return "✅ SIGNIFICANT (p<0.05)" if p < 0.05 else "❌ n.s. (p>=0.05)"

    def pick_first_existing(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def parse_timestamp_series(ts: pd.Series) -> pd.Series:
        """
        Robust timestamp parsing:
        - 1st: pandas infer (errors='coerce')
        - 2nd: force utc then drop tz (if NaT 많으면)
        """
        dt = pd.to_datetime(ts, errors="coerce", utc=False)
        if dt.isna().mean() > 0.2:
            dt2 = pd.to_datetime(ts, errors="coerce", utc=True)
            dt2 = dt2.dt.tz_convert(None)
            dt = dt.fillna(dt2)
        return dt

    def prep_df(df: pd.DataFrame) -> pd.DataFrame:
        # --- detect key columns ---
        ts_col = pick_first_existing(df, ["timestamp", "createdAt", "submittedAt", "startTime", "endTime", "dt"])
        acc_col = pick_first_existing(df, ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x"])
        age_col = pick_first_existing(df, ["age"])
        id_col  = pick_first_existing(df, ["participantId", "participantID", "pid"])
        sex_src = pick_first_existing(df, ["sex", "gender"])  # 너는 앞으로 Sex 용어 쓴다고 했지

        missing = []
        if ts_col is None:  missing.append("timestamp")
        if acc_col is None: missing.append("overallAccuracy (or overallAccuracy_y/x)")
        if age_col is None: missing.append("age")
        if missing:
            raise ValueError(f"Missing required columns (auto-detect failed): {missing}")

        out = pd.DataFrame()
        if id_col is not None:
            out["participantId"] = df[id_col]
        out["timestamp"] = df[ts_col]
        out["overallAccuracy"] = pd.to_numeric(df[acc_col], errors="coerce")
        out["age"] = pd.to_numeric(df[age_col], errors="coerce")

        # optional covariates
        if sex_src is not None:
            out["sex"] = df[sex_src].astype(str).str.lower()
            out.loc[~out["sex"].isin(["male", "female"]), "sex"] = np.nan

        rt_col = pick_first_existing(df, ["avgRT_overall", "avgRT"])
        if rt_col is not None:
            out["avgRT"] = pd.to_numeric(df[rt_col], errors="coerce")

        # parse timestamp -> dt
        out["dt"] = parse_timestamp_series(out["timestamp"])

        # drop essentials
        out = out.dropna(subset=["dt", "overallAccuracy", "age"]).copy()

        # date ordinal (days since min date)
        out["date"] = out["dt"].dt.date
        min_date = pd.to_datetime(out["date"]).min()
        out["date_ordinal"] = (pd.to_datetime(out["date"]) - min_date).dt.days

        # age group
        out["age_group"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        out = out.dropna(subset=["age_group"]).copy()

        return out

    def run_mobile(df_date: pd.DataFrame):
        ensure_dir(OUTDIR)

        tag = "mobile"
        outdir = os.path.join(OUTDIR, tag)
        ensure_dir(outdir)

        print("\n" + "="*78)
        print("(30 v2.1 | MOBILE ONLY) Date vs Accuracy")
        print("="*78)
        print(f"✅ N={len(df_date)}")
        print(f"Date range: {df_date['date'].min()} → {df_date['date'].max()}")

        # -------------------------
        # Save Prism table
        # -------------------------
        prism_cols = []
        if "participantId" in df_date.columns: prism_cols.append("participantId")
        prism_cols += ["date", "date_ordinal", "overallAccuracy", "age", "age_group"]
        if "sex" in df_date.columns: prism_cols.append("sex")
        if "avgRT" in df_date.columns: prism_cols.append("avgRT")

        prism = df_date[prism_cols].copy()
        prism_path = os.path.join(outdir, f"30v2-0_prism_table_{tag}.csv")
        prism.to_csv(prism_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved Prism table: {prism_path}")

        # -------------------------
        # (30-1) Overall Pearson
        # -------------------------
        r_all, p_all = pearsonr(df_date["date_ordinal"], df_date["overallAccuracy"])
        print("\n--- (30-1) Overall Pearson correlation ---")
        print(f"r = {r_all:.4f}, p = {p_all:.6g}, N = {len(df_date)}  {sig_mark(p_all)}")

        rep1 = os.path.join(outdir, f"30v2-1_overall_pearson_{tag}.txt")
        save_text(rep1, f"r={r_all:.6f}\np={p_all:.6g}\nN={len(df_date)}\n")

        # -------------------------
        # (30-2) OLS with covariates (HC3)
        # -------------------------
        # base: time + age
        formula_terms = ["date_ordinal", "age"]
        if "sex" in df_date.columns and df_date["sex"].notna().any():
            formula_terms.append("C(sex)")
        if "avgRT" in df_date.columns and df_date["avgRT"].notna().any():
            formula_terms.append("avgRT")

        formula = "overallAccuracy ~ " + " + ".join(formula_terms)

        # safe dropna for used columns
        needed_cols = ["overallAccuracy", "date_ordinal", "age"]
        if "C(sex)" in formula_terms: needed_cols.append("sex")
        if "avgRT" in formula_terms: needed_cols.append("avgRT")

        df_ols = df_date.dropna(subset=needed_cols).copy()

        print("\n--- (30-2) OLS (HC3) controlling covariates ---")
        print(f"Model: {formula}")
        print(f"N used: {len(df_ols)}")

        ols_txt_path  = os.path.join(outdir, f"30v2-2_ols_HC3_{tag}.txt")
        ols_coef_path = os.path.join(outdir, f"30v2-2_ols_coeffs_HC3_{tag}.csv")

        if len(df_ols) >= 30:
            ols_res = smf.ols(formula, data=df_ols).fit(cov_type="HC3")
            b_time = float(ols_res.params.get("date_ordinal", np.nan))
            p_time = float(ols_res.pvalues.get("date_ordinal", np.nan))

            print(f"Time coef(date_ordinal) = {b_time:.6f}, p = {p_time:.6g}  {sig_mark(p_time)}")
            print(f"R2={ols_res.rsquared:.4f}, Adj.R2={ols_res.rsquared_adj:.4f}")

            coef = pd.DataFrame({
                "term": ols_res.params.index,
                "coef": ols_res.params.values,
                "se": ols_res.bse.values,
                "t_or_z": ols_res.tvalues.values,
                "p": ols_res.pvalues.values,
            })
            coef.to_csv(ols_coef_path, index=False, encoding="utf-8-sig")
            save_text(ols_txt_path, ols_res.summary().as_text())
            print(f"✅ Saved OLS outputs: {ols_coef_path}, {ols_txt_path}")
        else:
            save_text(ols_txt_path, "Skipped OLS: too few samples after NA drop.\n")
            print("⚠️ Skipped OLS (too few samples).")

        # -------------------------
        # (30-3) By age-group Pearson + Holm
        # -------------------------
        rows = []
        for ag in AGE_LABELS:
            sub = df_date[df_date["age_group"] == ag].copy()
            if len(sub) < 3:
                continue
            r, p = pearsonr(sub["date_ordinal"], sub["overallAccuracy"])
            rows.append({"age_group": ag, "N": len(sub), "r": r, "p_raw": p})

        res = pd.DataFrame(rows)
        if len(res) > 0:
            rej, p_holm, _, _ = multipletests(res["p_raw"].values, alpha=0.05, method="holm")
            res["p_holm"] = p_holm
            res["sig_holm"] = np.where(res["p_holm"] < 0.05, "✅", "❌")

        print("\n--- (30-3) Pearson by age group (Holm corrected) ---")
        if len(res) == 0:
            print("⚠️ Not enough data per age group.")
        else:
            print(res.sort_values("age_group").round({"r":4, "p_raw":6, "p_holm":6}).to_string(index=False))

        res_csv = os.path.join(outdir, f"30v2-3_agegroup_correlations_holm_{tag}.csv")
        res_txt = os.path.join(outdir, f"30v2-3_agegroup_correlations_holm_{tag}.txt")
        res.to_csv(res_csv, index=False, encoding="utf-8-sig")
        save_text(res_txt, res.to_string(index=False))
        print(f"✅ Saved age-group results: {res_csv}")

        # -------------------------
        # Plots (y fixed 0~100)
        # -------------------------
        config.apply_korean_plot_style()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            data=df_date, x="date_ordinal", y="overallAccuracy",
            scatter_kws={"alpha": SCATTER_ALPHA},
            ax=ax
        )
        ax.set_title(f"(30-4) Accuracy over Time — MOBILE\nPearson r={r_all:.3f}, p={p_all:.3g}", fontsize=14, pad=12)
        ax.set_xlabel(f"Days since start (0 = {df_date['date'].min()})")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(*YLIM)

        out_png = os.path.join(outdir, f"30v2-4_accuracy_by_date_total_{tag}.png")
        out_svg = os.path.join(outdir, f"30v2-4_accuracy_by_date_total_{tag}.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved plot: {out_png} / {out_svg}")

        g = sns.lmplot(
            data=df_date,
            x="date_ordinal", y="overallAccuracy",
            col="age_group", col_wrap=3, height=3.8,
            scatter_kws={"alpha": 0.25},
        )
        g.fig.suptitle("(30-5) Accuracy over Time by Age Group — MOBILE", fontsize=14, y=1.02)
        g.set_axis_labels("Days since start", "Accuracy (%)")
        for ax in g.axes.flatten():
            ax.set_ylim(*YLIM)

        out_png2 = os.path.join(outdir, f"30v2-5_accuracy_by_date_age_group_{tag}.png")
        out_svg2 = os.path.join(outdir, f"30v2-5_accuracy_by_date_age_group_{tag}.svg")
        plt.savefig(out_png2, dpi=300, bbox_inches="tight")
        plt.savefig(out_svg2, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(g.fig)
        print(f"✅ Saved plot: {out_png2} / {out_svg2}")

        print("\n==================== (30 v2.1 | MOBILE ONLY) DONE ====================\n")

    def main():
        ensure_dir(OUTDIR)
        df = pd.read_csv(IN_FILE, encoding="utf-8-sig")
        df_date = prep_df(df)
        run_mobile(df_date)

    if __name__ == "__main__":
        main()


def _run_cell_141():
    # ==============================================================================
    # (32) Moderated Mediation (Sex as Moderator) - MOBILE + WEB  [FINAL + DISPLAY]
    # ------------------------------------------------------------------------------
    # 목적:
    #   Age -> (AI Exposure, AI Confidence) -> Accuracy
    #   간접효과가 Sex(male vs female)에 따라 달라지는지(Moderated mediation) 테스트
    #
    # 모델:
    #   M1: exposure_score   ~ age + sex_female + age:sex_female
    #   M2: confidence_score ~ age + sex_female + age:sex_female
    #   Y : accuracy_pct     ~ age + sex_female + exposure_score + confidence_score
    #                        + exposure_score:sex_female + confidence_score:sex_female
    #
    # Conditional indirect effects:
    #   male (sex=0):   ind1 = a1*b1 ; ind2 = a2*b2
    #   female (sex=1): ind1 = (a1+a1w)*(b1+b1w) ; ind2 = (a2+a2w)*(b2+b2w)
    #   diff = female - male (bootstrap CI)
    #
    # INPUT (핵심!):
    #   - outputs/run_.../16_mediation_parallel/.../16-0_mediation_input_*.csv 를 "가능하면" 사용
    #   - BUT: 그 파일에 participantId 또는 sex/gender가 없으면 섹션32에 못 쓰므로
    #          자동으로 analysis_data_*_age_filtered_20_69.csv 로 fallback
    #
    # OUTPUT:
    #   outputs/run_.../32_moderated_mediation/<cohort>/
    #     32-0_clean_input_<cohort>_<subset>.csv
    #     32-1_effects_<cohort>_<subset>.csv
    #     32-2_model_coeffs_<cohort>_<subset>.csv
    #     32-3_report_<cohort>_<subset>.txt
    #   + cohort_summary.csv, meta.json
    # ==============================================================================

    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    import statsmodels.formula.api as smf

    # (optional) pretty display in notebooks
    try:
        from IPython.display import display
        _HAS_IPY = True
    except Exception:
        _HAS_IPY = False


    # -----------------------------
    # Run/Section 폴더 유틸
    # -----------------------------
    def _get_latest_run_dir(outputs_root=config.OUTPUTS_DIR):
        root = Path(outputs_root)
        if not root.exists():
            return None
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        return runs[-1] if runs else None

    def _make_section_dir(section_name, outputs_root=config.OUTPUTS_DIR, run_tag=None):
        root = Path(outputs_root)
        root.mkdir(parents=True, exist_ok=True)

        if run_tag is not None:
            run_dir = root / f"run_{run_tag}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = _get_latest_run_dir(outputs_root)
            if run_dir is None:
                auto_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = root / f"run_{auto_tag}"
                run_dir.mkdir(parents=True, exist_ok=True)

        section_dir = run_dir / section_name
        section_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, section_dir

    def _write_text(path: Path, title: str, content: str):
        path.write_text(
            "============================================================\n"
            f"{title}\n"
            "============================================================\n\n"
            + content,
            encoding="utf-8"
        )

    def _has_cols(path: Path, required_cols: list[str]) -> bool:
        try:
            df0 = pd.read_csv(path, encoding="utf-8-sig", nrows=5)
            cols = set(df0.columns)
            return all(c in cols for c in required_cols)
        except Exception:
            return False


    # -----------------------------
    # Accuracy 컬럼 선택/스케일
    # -----------------------------
    def resolve_overall_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("overallAccuracy 계열 컬럼(overallAccuracy_y/overallAccuracy/overallAccuracy_x)이 없습니다.")

    def to_percent_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        finite = s_num.dropna()
        if finite.empty:
            return s_num
        mx = float(finite.max())
        if mx <= 1.5:  # 0~1 비율이면 %로 변환
            return s_num * 100.0
        return s_num


    # -----------------------------
    # Sex 정규화 (gender/sex 혼재 처리)
    # -----------------------------
    def normalize_sex(df: pd.DataFrame, in_col_candidates=("sex", "gender"), out_col="sex") -> pd.DataFrame:
        out = df.copy()

        src_col = None
        for c in in_col_candidates:
            if c in out.columns:
                src_col = c
                break

        if src_col is None:
            out[out_col] = np.nan
            return out

        s = out[src_col].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})

        s = s.replace({
            "m": "male",
            "man": "male",
            "male ": "male",
            "f": "female",
            "woman": "female",
            "female ": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer not to": "prefer-not-to-say",
            "prefer-not-to-say ": "prefer-not-to-say",
        })

        out[out_col] = s
        return out


    # -----------------------------
    # Exposure/Confidence 문자열 -> 숫자화
    # -----------------------------
    def map_ai_confidence_to_score(series: pd.Series) -> pd.Series:
        """
        aiConfidence example:
          not-confident / neutral / confident
        -> 1..3
        """
        s = series.astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})

        mapping = {
            "not-confident": 1,
            "not confident": 1,
            "neutral": 2,
            "confident": 3,
        }
        out = s.map(mapping)
        out = out.fillna(pd.to_numeric(series, errors="coerce"))
        return out

    def map_ai_exposure_freq_to_score(series: pd.Series) -> pd.Series:
        """
        aiExposureFrequency example:
          never / rarely / sometimes / weekly / daily ...
        -> 0..5
        """
        s = series.astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})

        mapping = {
            "never": 0,
            "rarely": 1,
            "sometimes": 2,
            "weekly": 3,
            "daily": 4,
            "multiple times a day": 5,
        }
        out = s.map(mapping)
        out = out.fillna(pd.to_numeric(series, errors="coerce"))
        return out

    def ensure_exposure_confidence_scores(d: pd.DataFrame) -> pd.DataFrame:
        """
        표준 컬럼:
          exposure_score, confidence_score
        이미 있으면 그대로, 없으면 aiExposureFrequency/aiConfidence에서 생성
        """
        out = d.copy()

        if "confidence_score" not in out.columns:
            if "aiConfidence" in out.columns:
                out["confidence_score"] = map_ai_confidence_to_score(out["aiConfidence"])
            else:
                raise KeyError("confidence_score가 없고 aiConfidence도 없습니다.")

        if "exposure_score" not in out.columns:
            if "aiExposureFrequency" in out.columns:
                out["exposure_score"] = map_ai_exposure_freq_to_score(out["aiExposureFrequency"])
            else:
                raise KeyError("exposure_score가 없고 aiExposureFrequency도 없습니다.")

        out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce")
        out["exposure_score"] = pd.to_numeric(out["exposure_score"], errors="coerce")
        return out


    # -----------------------------
    # Participant-level 보장 (혹시 중복 row 대비)
    # -----------------------------
    def ensure_participant_level(d: pd.DataFrame) -> pd.DataFrame:
        if "participantId" not in d.columns:
            return d

        n_rows = len(d)
        n_pid = d["participantId"].nunique(dropna=True)

        if n_pid > 0 and n_rows > n_pid * 1.2:
            num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            cat_cols = [c for c in d.columns if c not in num_cols]

            agg = {c: "mean" for c in num_cols}
            for c in cat_cols:
                agg[c] = "first"

            out = d.groupby("participantId", as_index=False).agg(agg)
            return out

        return d


    # -----------------------------
    # Input selector (요청하신 1번: 자동 fallback)
    # -----------------------------
    def choose_input_file(run_tag: str, cohort_tag: str) -> Path:
        """
        Prefer section16 input ONLY if it contains participantId OR sex/gender.
        Otherwise fallback to analysis_data_* (has participantId+gender).
        """

        p16 = Path(f"outputs/run_{run_tag}/16_mediation_parallel/{cohort_tag}/16-0_mediation_input_{cohort_tag}.csv")
        if p16.exists():
            ok = (
                _has_cols(p16, ["participantId"])
                or _has_cols(p16, ["sex"])
                or _has_cols(p16, ["gender"])
            )
            if ok:
                return p16
            else:
                print(f"⚠️ {p16.name} lacks participantId and sex/gender -> fallback to analysis_data")

        pA = Path(f"analysis_data_{cohort_tag}_age_filtered_20_69.csv")
        if pA.exists():
            return pA

        raise FileNotFoundError(
            f"No usable input file found for cohort={cohort_tag} (run_tag={run_tag}). "
            f"Looked for: {p16} and {pA}"
        )


    # -----------------------------
    # Moderated mediation core
    # -----------------------------
    def fit_models(d):
        m1 = smf.ols("exposure_score ~ age + sex_female + age:sex_female", data=d).fit()
        m2 = smf.ols("confidence_score ~ age + sex_female + age:sex_female", data=d).fit()
        y  = smf.ols(
            "accuracy_pct ~ age + sex_female + exposure_score + confidence_score + "
            "exposure_score:sex_female + confidence_score:sex_female",
            data=d
        ).fit()
        return m1, m2, y

    def extract_conditional_effects(m1, m2, y):
        a1  = float(m1.params.get("age", np.nan))
        a1w = float(m1.params.get("age:sex_female", 0.0))

        a2  = float(m2.params.get("age", np.nan))
        a2w = float(m2.params.get("age:sex_female", 0.0))

        b1  = float(y.params.get("exposure_score", np.nan))
        b1w = float(y.params.get("exposure_score:sex_female", 0.0))

        b2  = float(y.params.get("confidence_score", np.nan))
        b2w = float(y.params.get("confidence_score:sex_female", 0.0))

        cprime = float(y.params.get("age", np.nan))

        ind1_male   = a1 * b1
        ind2_male   = a2 * b2
        ind1_female = (a1 + a1w) * (b1 + b1w)
        ind2_female = (a2 + a2w) * (b2 + b2w)

        ind_total_male   = ind1_male + ind2_male
        ind_total_female = ind1_female + ind2_female

        return {
            # moderation terms
            "a1w_ageXsex_on_exposure": a1w,
            "a2w_ageXsex_on_confidence": a2w,
            "b1w_exposureXsex_on_accuracy": b1w,
            "b2w_confidenceXsex_on_accuracy": b2w,

            # conditional indirect effects
            "ind_exposure_male": ind1_male,
            "ind_exposure_female": ind1_female,
            "diff_ind_exposure_female_minus_male": ind1_female - ind1_male,

            "ind_confidence_male": ind2_male,
            "ind_confidence_female": ind2_female,
            "diff_ind_confidence_female_minus_male": ind2_female - ind2_male,

            "ind_total_male": ind_total_male,
            "ind_total_female": ind_total_female,
            "diff_ind_total_female_minus_male": ind_total_female - ind_total_male,

            # direct
            "cprime_direct_age": cprime,
        }

    def bootstrap_effects(d, n_boot=5000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(d)
        rows = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            b = d.iloc[idx].copy()
            m1, m2, y = fit_models(b)
            rows.append(extract_conditional_effects(m1, m2, y))
        return pd.DataFrame(rows)

    def summarize_effects(point: dict, boot_df: pd.DataFrame, alpha=0.05):
        rows = []
        for k, v in point.items():
            lo = float(boot_df[k].quantile(alpha/2))
            hi = float(boot_df[k].quantile(1 - alpha/2))
            sig = not (lo <= 0 <= hi)
            rows.append([k, float(v), lo, hi, sig])
        return pd.DataFrame(rows, columns=["effect", "coef", "CI[2.5%]", "CI[97.5%]", "sig_CI_nonzero"])

    def model_coeff_table(m1, m2, y):
        def tidy(model, name):
            return pd.DataFrame({
                "model": name,
                "term": model.params.index,
                "coef": model.params.values,
                "se": model.bse.values,
                "t": model.tvalues.values,
                "p": model.pvalues.values,
            })
        return pd.concat([tidy(m1,"M1_exposure"), tidy(m2,"M2_confidence"), tidy(y,"Y_accuracy")], ignore_index=True)

    def _print_or_display_df(df: pd.DataFrame, title: str | None = None):
        if title:
            print(title)
        if _HAS_IPY:
            display(df)
        else:
            print(df.to_string(index=False))


    # -----------------------------
    # Cohort runner
    # -----------------------------
    def run_one_cohort(
        cohort_tag: str,
        input_path: Path,
        out_dir: Path,
        subset_tag: str = "all",
        age_min: int | None = None,
        n_boot: int = 5000,
        seed: int = 42,
    ):
        title = f"(32) Moderated Mediation [{cohort_tag.upper()} | {subset_tag}]"

        d = pd.read_csv(input_path, encoding="utf-8-sig")
        d = ensure_participant_level(d)

        # age subset
        if "age" in d.columns:
            d["age"] = pd.to_numeric(d["age"], errors="coerce")
        if age_min is not None and "age" in d.columns:
            d = d[d["age"] >= age_min].copy()

        # accuracy_pct
        if "accuracy_pct" not in d.columns:
            acc_col = resolve_overall_accuracy_column(d)
            d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
            d["accuracy_pct"] = to_percent_series(d[acc_col])
            acc_used = acc_col
        else:
            d["accuracy_pct"] = pd.to_numeric(d["accuracy_pct"], errors="coerce")
            acc_used = "accuracy_pct"

        # exposure/confidence scores
        d = ensure_exposure_confidence_scores(d)

        # sex
        d = normalize_sex(d, in_col_candidates=("sex", "gender"), out_col="sex")
        d["sex_female"] = np.where(d["sex"]=="female", 1,
                            np.where(d["sex"]=="male", 0, np.nan))

        # keep complete (male/female only)
        keep_cols = ["age","exposure_score","confidence_score","accuracy_pct","sex_female"]
        for c in keep_cols:
            if c not in d.columns:
                raise KeyError(f"Missing required column: {c}")

        d = d.dropna(subset=keep_cols).copy()
        if len(d) == 0:
            raise ValueError("After filtering to male/female complete cases, N=0. "
                             "Check sex/gender availability and mappings.")

        d["sex_female"] = d["sex_female"].astype(int)

        # save cleaned input
        clean_path = out_dir / f"32-0_clean_input_{cohort_tag}_{subset_tag}.csv"
        d.to_csv(clean_path, index=False, encoding="utf-8-sig")

        # models
        m1, m2, y = fit_models(d)
        point = extract_conditional_effects(m1, m2, y)

        # bootstrap
        boot_df = bootstrap_effects(d, n_boot=n_boot, seed=seed)
        effects_df = summarize_effects(point, boot_df, alpha=0.05)

        eff_path = out_dir / f"32-1_effects_{cohort_tag}_{subset_tag}.csv"
        effects_df.to_csv(eff_path, index=False, encoding="utf-8-sig")

        coef_df = model_coeff_table(m1, m2, y)
        coef_path = out_dir / f"32-2_model_coeffs_{cohort_tag}_{subset_tag}.csv"
        coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

        # report
        sex_counts = d["sex_female"].value_counts().sort_index()
        lines = []
        lines.append("Overview")
        lines.append("-----------------------------------------")
        lines.append(f"- Cohort: {cohort_tag}")
        lines.append(f"- Input:  {str(input_path)}")
        lines.append(f"- Subset: {subset_tag}")
        lines.append(f"- N used (male/female): {len(d):,}")
        lines.append(f"- male (0): {int(sex_counts.get(0,0)):,}")
        lines.append(f"- female (1): {int(sex_counts.get(1,0)):,}")
        lines.append(f"- Accuracy source: {acc_used}")
        lines.append(f"- Bootstrap: n_boot={n_boot}, seed={seed}")
        lines.append("")
        lines.append("Key effects (coef, 95% CI, sig)")
        lines.append("-----------------------------------------")

        key_order = [
            "ind_total_male","ind_total_female","diff_ind_total_female_minus_male",
            "ind_exposure_male","ind_exposure_female","diff_ind_exposure_female_minus_male",
            "ind_confidence_male","ind_confidence_female","diff_ind_confidence_female_minus_male",
            "cprime_direct_age",
            "a1w_ageXsex_on_exposure","a2w_ageXsex_on_confidence","b1w_exposureXsex_on_accuracy","b2w_confidenceXsex_on_accuracy",
        ]
        eff_map = effects_df.set_index("effect")
        for k in key_order:
            if k not in eff_map.index:
                continue
            r = eff_map.loc[k]
            lines.append(
                f"- {k}: {r['coef']:.6g} [{r['CI[2.5%]']:.6g}, {r['CI[97.5%]']:.6g}]  "
                f"{'SIG' if bool(r['sig_CI_nonzero']) else 'n.s.'}"
            )

        rep_path = out_dir / f"32-3_report_{cohort_tag}_{subset_tag}.txt"
        _write_text(rep_path, title, "\n".join(lines))

        # -----------------------------
        # DISPLAY (요청하신 부분!)
        # -----------------------------
        print("\n-----------------------------------------")
        print(f"[{cohort_tag.upper()} | {subset_tag}] Moderated mediation summary")
        print(f"- N used: {len(d):,}  (male={int(sex_counts.get(0,0)):,}, female={int(sex_counts.get(1,0)):,})")
        print(f"- Saved: {eff_path.name}, {coef_path.name}, {rep_path.name}")
        print("-----------------------------------------")

        show_keys = [
            "ind_total_male","ind_total_female","diff_ind_total_female_minus_male",
            "ind_exposure_male","ind_exposure_female","diff_ind_exposure_female_minus_male",
            "ind_confidence_male","ind_confidence_female","diff_ind_confidence_female_minus_male",
            "cprime_direct_age",
        ]
        disp_df = effects_df[effects_df["effect"].isin(show_keys)].copy()
        disp_df = disp_df.set_index("effect").loc[[k for k in show_keys if k in disp_df["effect"].values]].reset_index()

        _print_or_display_df(disp_df, title="Key effects (coef, 95% CI, sig_CI_nonzero):")
        print("-----------------------------------------\n")

        return {
            "cohort": cohort_tag,
            "subset": subset_tag,
            "input": str(input_path),
            "N_used": int(len(d)),
            "n_male": int(sex_counts.get(0,0)),
            "n_female": int(sex_counts.get(1,0)),
            "bootstrap_n": int(n_boot),
            "seed": int(seed),
            "effects_csv": str(eff_path),
            "coeffs_csv": str(coef_path),
            "report_txt": str(rep_path),
        }


    # -----------------------------
    # MAIN
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(32) Moderated Mediation (Sex as Moderator) - MOBILE + WEB")
        print("------------------------------------------------------------------------------")
        print("Age -> (Exposure, Confidence) -> Accuracy, conditional indirect effects by Sex")
        print("==============================================================================\n")

        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG   # 너 run tag
        N_BOOT = 5000                # 먼저 테스트면 500~1000 추천, 확정이면 5000
        SEED = 42

        run_dir, section_dir = _make_section_dir("32_moderated_mediation", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
        print(f"📁 섹션32 저장 위치: {section_dir}")

        cohorts = ["mobile", "web"]
        summaries = []

        meta = {
            "section": "32_moderated_mediation",
            "run_tag": RUN_TAG,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_boot": N_BOOT,
            "seed": SEED,
            "models": {
                "M1_exposure": "exposure_score ~ age + sex_female + age:sex_female",
                "M2_confidence": "confidence_score ~ age + sex_female + age:sex_female",
                "Y_accuracy": "accuracy_pct ~ age + sex_female + exposure_score + confidence_score + exposure_score:sex_female + confidence_score:sex_female",
            },
            "notes": [
                "Input auto-fallback: if section16 input lacks participantId AND sex/gender, use analysis_data_* instead.",
                "Sex coded male=0, female=1. prefer-not-to-say excluded by dropna (sex_female becomes NaN).",
                "If exposure_score/confidence_score not present, maps from aiExposureFrequency/aiConfidence strings.",
                "Accuracy: uses accuracy_pct if present; else converts overallAccuracy_y/overallAccuracy/overallAccuracy_x to percent.",
            ],
        }

        for cohort_tag in cohorts:
            cohort_out = section_dir / cohort_tag
            cohort_out.mkdir(parents=True, exist_ok=True)

            try:
                input_path = choose_input_file(RUN_TAG, cohort_tag)
                print(f"\n✅ [{cohort_tag}] input selected: {input_path}")

                # (32-1) 전체
                summaries.append(run_one_cohort(
                    cohort_tag=cohort_tag,
                    input_path=input_path,
                    out_dir=cohort_out,
                    subset_tag="all",
                    age_min=None,
                    n_boot=N_BOOT,
                    seed=SEED,
                ))

                # (32-2) 50+만
                summaries.append(run_one_cohort(
                    cohort_tag=cohort_tag,
                    input_path=input_path,
                    out_dir=cohort_out,
                    subset_tag="age50plus",
                    age_min=50,
                    n_boot=N_BOOT,
                    seed=SEED,
                ))

            except Exception as e:
                print(f"❌ [{cohort_tag}] failed: {e}")
                summaries.append({
                    "cohort": cohort_tag,
                    "subset": "all",
                    "status": "failed",
                    "error": str(e),
                })

        # cohort 요약 저장
        summary_df = pd.DataFrame(summaries)
        summary_path = section_dir / "cohort_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ cohort summary saved: {summary_path}")

        # meta 저장
        meta_path = section_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ meta saved: {meta_path}")

        print("\n==================== (32) DONE ====================")


def _run_cell_144():
    # ==============================================================================
    # (32.1) Moderated Mediation Figures from Saved Results (NO re-fitting)
    # ------------------------------------------------------------------------------
    # Reads outputs from Section 32 and draws:
    #   - (32.1-1) Forest plot for key effects (coef + 95% CI)
    #   - (32.1-2) Path diagram with male vs female coefficients (a/b/c')
    #
    # Input files expected (per cohort/subset):
    #   outputs/run_<RUN_TAG>/32_moderated_mediation/<cohort>/
    #     32-1_effects_<cohort>_<subset>.csv
    #     32-2_model_coeffs_<cohort>_<subset>.csv
    #
    # Output:
    #   outputs/run_<RUN_TAG>/32_1_moderated_mediation_figures/<cohort>/
    #     32-1-1_forest_key_effects_<cohort>_<subset>.png/.svg
    #     32-1-2_path_diagram_<cohort>_<subset>.png/.svg
    # ==============================================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    # -----------------------------
    # Run/Section 폴더 유틸 (섹션6 스타일)
    # -----------------------------
    def _get_latest_run_dir(outputs_root=config.OUTPUTS_DIR):
        root = Path(outputs_root)
        if not root.exists():
            return None
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        return runs[-1] if runs else None

    def _make_section_dir(section_name, outputs_root=config.OUTPUTS_DIR, run_tag=None):
        root = Path(outputs_root)
        root.mkdir(parents=True, exist_ok=True)

        if run_tag is not None:
            run_dir = root / f"run_{run_tag}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = _get_latest_run_dir(outputs_root)
            if run_dir is None:
                auto_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = root / f"run_{auto_tag}"
                run_dir.mkdir(parents=True, exist_ok=True)

        section_dir = run_dir / section_name
        section_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, section_dir

    # -----------------------------
    # Helpers: stars + coefficient pick
    # -----------------------------
    def p_to_stars(p):
        try:
            p = float(p)
        except Exception:
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def get_coef_row(coeffs_df, model, term):
        hit = coeffs_df[(coeffs_df["model"] == model) & (coeffs_df["term"] == term)]
        if hit.empty:
            return None
        return hit.iloc[0]

    def fmt_coef_star(row, nd=2):
        if row is None:
            return "NA"
        c = row["coef"]
        p = row["p"]
        return f"{float(c):.{nd}f}{p_to_stars(p)}"

    def safe_float(x, default=np.nan):
        try:
            return float(x)
        except Exception:
            return default

    # -----------------------------
    # (32.1-1) Forest plot from effects CSV
    # -----------------------------
    def plot_forest_key_effects(effects_df, out_base: Path, title: str, effect_order: list[str]):
        df = effects_df.copy()
        df = df[df["effect"].isin(effect_order)].copy()

        # order 유지
        df["order"] = df["effect"].apply(lambda s: effect_order.index(s) if s in effect_order else 999)
        df = df.sort_values("order")

        # 보기 좋게 y축 위->아래
        df = df.iloc[::-1].reset_index(drop=True)

        x = df["coef"].astype(float).values
        lo = df["CI[2.5%]"].astype(float).values
        hi = df["CI[97.5%]"].astype(float).values
        y = np.arange(len(df))

        xerr = np.vstack([x - lo, hi - x])

        plt.figure(figsize=(10, max(4, 0.55 * len(df))))
        plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
        plt.axvline(0, linestyle="--")
        plt.yticks(y, df["effect"].values)
        plt.title(title)
        plt.xlabel("Effect (coef) with 95% CI")
        plt.tight_layout()

        plt.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(str(out_base) + ".svg", bbox_inches="tight")
        plt.show()
        plt.close()

    # -----------------------------
    # (32.1-2) Path diagram from model_coeffs CSV
    # -----------------------------
    def draw_moderated_path_diagram(coeffs_df: pd.DataFrame, out_base: Path, title: str):
        """
        Diagram nodes: Age -> Exposure, Age -> Confidence, Exposure/Confidence -> Accuracy, Age -> Accuracy
        Labels show male vs female coefficients:
          male = base term
          female = base + interaction term
        """
        # Pull rows
        # M1: exposure ~ age + sex + age:sex
        a1_row  = get_coef_row(coeffs_df, "M1_exposure", "age")
        a1w_row = get_coef_row(coeffs_df, "M1_exposure", "age:sex_female")

        # M2: confidence ~ age + sex + age:sex
        a2_row  = get_coef_row(coeffs_df, "M2_confidence", "age")
        a2w_row = get_coef_row(coeffs_df, "M2_confidence", "age:sex_female")

        # Y: accuracy ~ age + sex + exposure + confidence + exposure:sex + confidence:sex
        b1_row  = get_coef_row(coeffs_df, "Y_accuracy", "exposure_score")
        b1w_row = get_coef_row(coeffs_df, "Y_accuracy", "exposure_score:sex_female")

        b2_row  = get_coef_row(coeffs_df, "Y_accuracy", "confidence_score")
        b2w_row = get_coef_row(coeffs_df, "Y_accuracy", "confidence_score:sex_female")

        c_row   = get_coef_row(coeffs_df, "Y_accuracy", "age")  # direct effect

        # Numeric values for male/female
        a1_m = safe_float(a1_row["coef"]) if a1_row is not None else np.nan
        a1_f = a1_m + (safe_float(a1w_row["coef"], 0.0) if a1w_row is not None else 0.0)

        a2_m = safe_float(a2_row["coef"]) if a2_row is not None else np.nan
        a2_f = a2_m + (safe_float(a2w_row["coef"], 0.0) if a2w_row is not None else 0.0)

        b1_m = safe_float(b1_row["coef"]) if b1_row is not None else np.nan
        b1_f = b1_m + (safe_float(b1w_row["coef"], 0.0) if b1w_row is not None else 0.0)

        b2_m = safe_float(b2_row["coef"]) if b2_row is not None else np.nan
        b2_f = b2_m + (safe_float(b2w_row["coef"], 0.0) if b2w_row is not None else 0.0)

        c_m  = safe_float(c_row["coef"]) if c_row is not None else np.nan
        # (Age direct effect is not sex-moderated in current model spec, so same for both)

        # Labels with stars (stars: base term p; female computed doesn't have direct p, so show stars for base + interaction separately)
        a1_lab = f"male {a1_m:.2f}{p_to_stars(a1_row['p']) if a1_row is not None else ''}\n" \
                 f"female {a1_f:.2f} (Δ {safe_float(a1w_row['coef'],0.0):+.2f}{p_to_stars(a1w_row['p']) if a1w_row is not None else ''})"

        a2_lab = f"male {a2_m:.2f}{p_to_stars(a2_row['p']) if a2_row is not None else ''}\n" \
                 f"female {a2_f:.2f} (Δ {safe_float(a2w_row['coef'],0.0):+.2f}{p_to_stars(a2w_row['p']) if a2w_row is not None else ''})"

        b1_lab = f"male {b1_m:.2f}{p_to_stars(b1_row['p']) if b1_row is not None else ''}\n" \
                 f"female {b1_f:.2f} (Δ {safe_float(b1w_row['coef'],0.0):+.2f}{p_to_stars(b1w_row['p']) if b1w_row is not None else ''})"

        b2_lab = f"male {b2_m:.2f}{p_to_stars(b2_row['p']) if b2_row is not None else ''}\n" \
                 f"female {b2_f:.2f} (Δ {safe_float(b2w_row['coef'],0.0):+.2f}{p_to_stars(b2w_row['p']) if b2w_row is not None else ''})"

        c_lab  = f"{c_m:.2f}{p_to_stars(c_row['p']) if c_row is not None else ''}"

        # Draw
        fig, ax = plt.subplots(figsize=(14, 10))

        pos = {
            "Age (X)": (0.0, 0.0),
            "AI Exposure (M1)": (0.55, 0.55),
            "AI Confidence (M2)": (0.55, -0.55),
            "Accuracy (Y)": (1.15, 0.0),
        }

        node_style = dict(boxstyle="round,pad=0.75", fc="skyblue", ec="black", lw=1.6)
        arrow_style = dict(arrowstyle="->,head_width=0.35,head_length=0.75", color="black", lw=2.2)

        for name, (x, y) in pos.items():
            ax.text(x, y, name, ha="center", va="center", fontsize=16, fontweight="bold", bbox=node_style)

        def draw_path(src, dst, label, rad=0.15, yoff=0.08):
            ax.annotate(
                "",
                xy=pos[dst], xytext=pos[src],
                arrowprops={**arrow_style, "connectionstyle": f"arc3,rad={rad}"}
            )
            mx = (pos[src][0] + pos[dst][0]) / 2
            my = (pos[src][1] + pos[dst][1]) / 2 + yoff
            ax.text(
                mx, my, label,
                ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
            )

        draw_path("Age (X)", "AI Exposure (M1)", a1_lab, rad=0.20, yoff=0.12)
        draw_path("Age (X)", "AI Confidence (M2)", a2_lab, rad=-0.20, yoff=0.12)
        draw_path("AI Exposure (M1)", "Accuracy (Y)", b1_lab, rad=0.20, yoff=0.12)
        draw_path("AI Confidence (M2)", "Accuracy (Y)", b2_lab, rad=-0.20, yoff=0.12)
        draw_path("Age (X)", "Accuracy (Y)", c_lab, rad=0.00, yoff=-0.14)

        ax.set_title(title, fontsize=18, fontweight="bold", pad=18)

        legend = (
            "Labels show:\n"
            "  male = base term\n"
            "  female = base + interaction (Δ shown)\n"
            "Stars: * p<.05, ** p<.01, *** p<.001 (for base/interaction terms)"
        )
        ax.text(
            0.58, -1.13, legend,
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E5", ec="gray", lw=1),
        )

        ax.set_xlim(-0.3, 1.45)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        plt.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.savefig(str(out_base) + ".svg", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.show()
        plt.close(fig)

    # -----------------------------
    # Main
    # -----------------------------
    if __name__ == "__main__":
        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG  # 너 run tag

        # Section 32 results folder
        sec32_dir = Path(f"{OUTPUTS_ROOT}/run_{RUN_TAG}/32_moderated_mediation")

        # New figure-only section folder
        run_dir, sec321_dir = _make_section_dir("32_1_moderated_mediation_figures", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)

        cohorts = ["mobile", "web"]
        subsets = ["all", "age50plus"]

        # forest plot에 보여줄 effect 순서(논문용 핵심)
        effect_order = [
            "ind_total_male",
            "ind_total_female",
            "diff_ind_total_female_minus_male",
            "ind_exposure_male",
            "ind_exposure_female",
            "diff_ind_exposure_female_minus_male",
            "ind_confidence_male",
            "ind_confidence_female",
            "diff_ind_confidence_female_minus_male",
            "cprime_direct_age",
        ]

        print("==============================================================================")
        print("(32.1) Draw figures from Section 32 saved CSVs (no re-fitting)")
        print("==============================================================================")
        print(f"- Read from: {sec32_dir}")
        print(f"- Save to :  {sec321_dir}\n")

        for cohort in cohorts:
            in_cohort_dir = sec32_dir / cohort
            out_cohort_dir = sec321_dir / cohort
            out_cohort_dir.mkdir(parents=True, exist_ok=True)

            for subset in subsets:
                effects_path = in_cohort_dir / f"32-1_effects_{cohort}_{subset}.csv"
                coeffs_path  = in_cohort_dir / f"32-2_model_coeffs_{cohort}_{subset}.csv"

                if not effects_path.exists() or not coeffs_path.exists():
                    print(f"⚠️ skip [{cohort} | {subset}] missing files")
                    print(f"   - effects: {effects_path.exists()}  coeffs: {coeffs_path.exists()}")
                    continue

                effects_df = pd.read_csv(effects_path, encoding="utf-8-sig")
                coeffs_df  = pd.read_csv(coeffs_path, encoding="utf-8-sig")

                # (32.1-1) Forest plot
                out_base1 = out_cohort_dir / f"32-1-1_forest_key_effects_{cohort}_{subset}"
                plot_forest_key_effects(
                    effects_df,
                    out_base1,
                    title=f"Key effects (95% CI) [{cohort.upper()} | {subset}]",
                    effect_order=effect_order
                )
                print(f"✅ saved forest: {out_base1}.png/.svg")

                # (32.1-2) Path diagram
                out_base2 = out_cohort_dir / f"32-1-2_path_diagram_{cohort}_{subset}"
                draw_moderated_path_diagram(
                    coeffs_df,
                    out_base2,
                    title=f"Moderated mediation path diagram [{cohort.upper()} | {subset}]"
                )
                print(f"✅ saved diagram: {out_base2}.png/.svg\n")

        print("==================== (32.1) DONE ====================")


def _run_cell_147():
    # ==============================================================================
    # (33) Correlation Network by Sex (Emphasized) - MOBILE + WEB [ENGLISH]
    # ------------------------------------------------------------------------------
    # Based on Section 15, but split into Sex groups (male vs female) per cohort.
    # Saves per cohort/sex:
    #   - raw prepped csv
    #   - network figure png/svg
    #   - corr matrix csv
    #   - p matrix csv
    #   - summary txt
    #
    # Notes:
    # - Uses gender/sex column robustly -> normalized 'sex' column
    # - By default, runs only for male/female (exclude prefer-not-to-say)
    # - Uses complete-case filtering on the network variables
    # ==============================================================================

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import networkx as nx

    # -----------------------------
    # Run / Output config
    # -----------------------------
    RUN_DIR = r"outputs\run_20260119_192624"
    SECTION_DIR = os.path.join(RUN_DIR, "33_corr_network_by_sex")
    ALPHA = 0.05

    # -----------------------------
    # Style (same as Section 15 emphasized)
    # -----------------------------
    NODE_SIZE = 60000
    NODE_COLOR = "skyblue"
    NODE_ALPHA = 0.92
    NODE_LABEL_FONTSIZE = 36

    EDGE_LABEL_FONTSIZE = 30

    SIG_MIN, SIG_MAX = 8, 26
    NS_MIN,  NS_MAX  = 5, 14

    LAYOUT = "circular"
    CIRCULAR_SCALE = 0.45
    AX_MARGINS = 0.22
    SAVE_PAD_INCHES = 0.80

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def choose_accuracy_column(df: pd.DataFrame):
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y", "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy", "overallAccuracy"
        raise KeyError("No accuracy column found: expected overallAccuracy_y or overallAccuracy")

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str):
        s = pd.to_numeric(df[acc_col], errors="coerce")
        s_nonan = s.dropna()
        if len(s_nonan) == 0:
            return s, "accuracy empty"
        if (s_nonan.max() <= 1.5) and (s_nonan.mean() <= 1.0):
            return s * 100.0, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return s, f"{acc_col} treated as percent already"

    def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {
            "very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5
        }
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        def norm_str(s):
            return s.astype(str).str.lower().str.strip()

        if "exposure_score" not in df.columns:
            if "aiExposureFrequency" in df.columns:
                df["exposure_score"] = norm_str(df["aiExposureFrequency"]).map(exposure_map)
            else:
                df["exposure_score"] = np.nan

        if "confidence_score" not in df.columns:
            if "aiConfidence" in df.columns:
                df["confidence_score"] = norm_str(df["aiConfidence"]).map(confidence_map)
            else:
                df["confidence_score"] = np.nan

        if "attitude_score" not in df.columns:
            if "aiAttitude" in df.columns:
                df["attitude_score"] = norm_str(df["aiAttitude"]).map(attitude_map)
            else:
                df["attitude_score"] = np.nan

        return df

    def normalize_sex(df: pd.DataFrame, in_col_candidates=("sex", "gender"), out_col="sex") -> pd.DataFrame:
        out = df.copy()

        src_col = None
        for c in in_col_candidates:
            if c in out.columns:
                src_col = c
                break

        if src_col is None:
            out[out_col] = np.nan
            return out

        s = out[src_col].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})

        s = s.replace({
            "m": "male",
            "man": "male",
            "male ": "male",
            "f": "female",
            "woman": "female",
            "female ": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer not to": "prefer-not-to-say",
            "prefer-not-to-say ": "prefer-not-to-say",
        })

        out[out_col] = s
        return out

    def build_corr_network(df: pd.DataFrame, vars_map: dict, alpha: float = 0.05):
        needed = list(vars_map.values())
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        corr_df = df[needed].rename(columns={v: k for k, v in vars_map.items()}).dropna()
        n_complete = len(corr_df)
        if n_complete < 20:
            return None, None, None, None, n_complete, corr_df

        corr_matrix = corr_df.corr(method="pearson")
        p_matrix = corr_df.corr(method=lambda x, y: pearsonr(x, y)[1])

        nodes = list(corr_matrix.columns)
        G = nx.Graph()
        G.add_nodes_from(nodes)

        edge_labels = {}

        for i in range(len(nodes)):
            for j in range(i):
                n1, n2 = nodes[i], nodes[j]
                r_val = corr_matrix.loc[n1, n2]
                p_val = p_matrix.loc[n1, n2]

                if not np.isfinite(r_val) or not np.isfinite(p_val):
                    continue

                significant = (p_val < alpha)

                G.add_edge(
                    n1, n2,
                    weight=float(abs(r_val)),
                    r=float(r_val),
                    p=float(p_val),
                    significant=bool(significant),
                )

                if significant:
                    stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                    edge_labels[(n1, n2)] = f"{r_val:.2f}{stars}"
                else:
                    edge_labels[(n1, n2)] = f"{r_val:.2f}"

        return G, corr_matrix, p_matrix, edge_labels, n_complete, corr_df

    def make_layout(G: nx.Graph, layout: str):
        if layout == "spring":
            return nx.spring_layout(G, seed=42, k=0.55)
        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        return nx.circular_layout(G, scale=CIRCULAR_SCALE)

    def draw_emphasized_network(G: nx.Graph, edge_labels: dict, title: str, out_base: str, layout: str = "circular"):
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        if len(weights) == 0:
            raise ValueError("No edges to draw (likely NaN correlations / constant columns).")

        w_max = max(weights) if max(weights) > 0 else 1.0
        pos = make_layout(G, layout)

        sig_edges, ns_edges = [], []
        sig_widths, ns_widths = [], []
        sig_colors, ns_colors = [], []

        for u, v in G.edges():
            r_val = G[u][v]["r"]
            w_norm = G[u][v]["weight"] / w_max

            if G[u][v]["significant"]:
                sig_edges.append((u, v))
                sig_widths.append(SIG_MIN + w_norm * (SIG_MAX - SIG_MIN))
                sig_colors.append("#DB4437" if r_val > 0 else "#4285F4")  # red/blue
            else:
                ns_edges.append((u, v))
                ns_widths.append(NS_MIN + w_norm * (NS_MAX - NS_MIN))
                ns_colors.append("#BDBDBD")  # gray

        fig, ax = plt.subplots(figsize=(24, 24))

        if ns_edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=ns_edges, width=ns_widths,
                                   edge_color=ns_colors, alpha=0.55)
        if sig_edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=sig_edges, width=sig_widths,
                                   edge_color=sig_colors, alpha=0.85)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color=NODE_COLOR, alpha=NODE_ALPHA)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=NODE_LABEL_FONTSIZE, font_weight="bold")

        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_labels,
            font_size=EDGE_LABEL_FONTSIZE, font_color="black",
            bbox=dict(facecolor="white", alpha=0.45, edgecolor="none"),
        )

        ax.set_title(title, fontsize=32, fontweight="bold", pad=20)
        ax.set_axis_off()
        ax.margins(AX_MARGINS)

        fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
        fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
        plt.show()
        plt.close(fig)

    # -----------------------------
    # Main
    # -----------------------------
    print("==============================================================================")
    print("(33) Correlation Network by Sex (Emphasized) - MOBILE + WEB [ENGLISH]")
    print("==============================================================================\n")

    ensure_dir(SECTION_DIR)

    corr_vars = {
        "Age": "age",
        "Accuracy": "accuracy_pct",
        "AI Exposure": "exposure_score",
        "AI Confidence": "confidence_score",
        "AI Attitude": "attitude_score",
    }

    cohorts = [
        ("mobile", config.MOBILE_AGE_FILTERED),
        ("web",    config.WEB_AGE_FILTERED),
    ]

    # ✅ 분석할 sex 그룹 (원하면 "prefer-not-to-say" 추가 가능)
    SEX_GROUPS = ["male", "female"]  # or ["male","female","prefer-not-to-say"]

    for cohort_tag, file_path in cohorts:
        print(f"\n==================== [{cohort_tag.upper()}] (33) START ====================")
        cohort_out = os.path.join(SECTION_DIR, cohort_tag)
        ensure_dir(cohort_out)

        # load
        try:
            df0 = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df0)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path}")
            continue

        # normalize sex
        df0 = normalize_sex(df0, in_col_candidates=("sex", "gender"), out_col="sex")
        sex_counts_all = df0["sex"].value_counts(dropna=False).to_dict()
        print(f"✅ [{cohort_tag}] Sex counts (raw): {sex_counts_all}")

        # accuracy + scores (do once, then split)
        try:
            acc_col, acc_note = choose_accuracy_column(df0)
            acc_pct, scale_note = ensure_accuracy_percent(df0, acc_col)
            df0 = df0.copy()
            df0["accuracy_pct"] = acc_pct
            print(f"✅ [{cohort_tag}] Accuracy column used: {acc_note} | {scale_note}")
        except KeyError as e:
            print(f"❌ [{cohort_tag}] {e}")
            continue

        df0 = ensure_score_columns(df0)

        # numeric safety
        for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]:
            if c in df0.columns:
                df0[c] = pd.to_numeric(df0[c], errors="coerce")

        # run per sex group
        for sex_g in SEX_GROUPS:
            df = df0[df0["sex"] == sex_g].copy()
            out_dir = os.path.join(cohort_out, sex_g)
            ensure_dir(out_dir)

            print(f"\n---- [{cohort_tag.upper()} | {sex_g}] ----")
            print(f"rows before dropna: {len(df):,}")

            # save prepped raw (sex-subset)
            raw_out = os.path.join(out_dir, f"33-0_raw_prepped_{cohort_tag}_{sex_g}.csv")
            df.to_csv(raw_out, index=False, encoding="utf-8-sig")
            print(f"✅ Saved raw prepped: {raw_out}")

            # build network
            try:
                G, corr_matrix, p_matrix, edge_labels, n_complete, corr_df = build_corr_network(
                    df=df, vars_map=corr_vars, alpha=ALPHA
                )
            except KeyError as e:
                print(f"❌ Column error: {e}")
                continue

            # constant columns check (after dropna)
            if corr_df is not None:
                nunique = corr_df.nunique()
                const_cols = nunique[nunique <= 1].index.tolist()
                if const_cols:
                    print(f"⚠️ Constant columns after dropna: {const_cols}")

            print(f"✅ Complete-case N for network: {n_complete}")

            if G is None or len(G.edges()) == 0:
                print("⚠️ No drawable edges (NaN correlations / too few complete cases).")
                # still save a small summary
                summary_path = os.path.join(out_dir, f"33-4_network_summary_{cohort_tag}_{sex_g}.txt")
                summary_txt = (
                    f"(33) Network summary [{cohort_tag} | {sex_g}]\n"
                    f"- complete-case N: {n_complete}\n"
                    f"- status: no drawable edges (too few complete cases or NaNs)\n"
                    f"- alpha: {ALPHA}\n"
                )
                save_text(summary_path, summary_txt)
                continue

            sig_n = sum(1 for u, v in G.edges() if G[u][v]["significant"])
            ns_n = len(G.edges()) - sig_n
            print(f"Edges total={len(G.edges())} | significant={sig_n} | non-significant={ns_n}")

            # draw + save
            title = f"(33-1) Correlation Network (ALL edges, alpha={ALPHA}) [{cohort_tag.upper()} | {sex_g}]"
            out_base = os.path.join(out_dir, f"33-1_correlation_network_all_edges_{cohort_tag}_{sex_g}")
            draw_emphasized_network(G, edge_labels, title, out_base, layout=LAYOUT)

            # save matrices
            corr_out = os.path.join(out_dir, f"33-2_corr_matrix_{cohort_tag}_{sex_g}.csv")
            p_out = os.path.join(out_dir, f"33-3_p_matrix_{cohort_tag}_{sex_g}.csv")
            corr_matrix.to_csv(corr_out, encoding="utf-8-sig")
            p_matrix.to_csv(p_out, encoding="utf-8-sig")
            print(f"✅ Saved matrices: {corr_out}, {p_out}")

            # summary
            summary_txt = (
                f"(33) Network summary [{cohort_tag} | {sex_g}]\n"
                f"- complete-case N: {n_complete}\n"
                f"- nodes: {len(G.nodes())}\n"
                f"- edges total: {len(G.edges())}\n"
                f"- significant edges (p<{ALPHA}): {sig_n}\n"
                f"- non-significant edges: {ns_n}\n"
                f"- layout: {LAYOUT} (circular_scale={CIRCULAR_SCALE}, ax_margins={AX_MARGINS}, pad_inches={SAVE_PAD_INCHES})\n"
                f"- accuracy: {acc_col} -> accuracy_pct (% scale)\n"
                f"- sex source: normalized from gender/sex\n"
            )
            summary_path = os.path.join(out_dir, f"33-4_network_summary_{cohort_tag}_{sex_g}.txt")
            save_text(summary_path, summary_txt)
            print(f"✅ Saved summary: {summary_path}")

        print(f"\n==================== [{cohort_tag.upper()}] (33) END ====================")

    print("\n==================== (33) DONE ====================")


def _run_cell_150():
    # ==============================================================================
    # (34) Sex-stratified scatter (fixed colors): Exposure/Confidence/Attitude vs Accuracy
    #      (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # - male = blue (#4285F4), female = red (#DB4437)  ✅ points + regression lines fixed
    # - y-axis fixed to 0~100 (Accuracy %)
    # - Saves png/svg + within-sex Pearson/Spearman report
    # - Output: outputs/run_20260119_192624/34_corr_scatter_by_sex/<cohort>/
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr

    RUN_DIR = os.path.join("outputs", "run_20260119_192624")
    SECTION_DIR = os.path.join(RUN_DIR, "34_corr_scatter_by_sex")

    # ✅ fixed sex colors
    SEX_ORDER = ["male", "female"]
    SEX_PALETTE = {"male": "#4285F4", "female": "#DB4437"}

    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def safe_slug(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = s.replace(" ", "_").replace("/", "_")
        return s

    def find_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise ValueError("No accuracy column found (expected overallAccuracy_y / overallAccuracy / overallAccuracy_x).")

    def convert_accuracy_to_percent_if_needed(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        df = df.copy()
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        finite = df[acc_col].dropna()
        if finite.empty:
            return df, "accuracy empty"
        if (finite.max() <= 1.5) and (finite.mean() <= 1.0):
            df[acc_col] = df[acc_col] * 100.0
            return df, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return df, f"{acc_col} treated as percent already"

    def normalize_sex(df: pd.DataFrame, in_col_candidates=("sex", "gender"), out_col="sex") -> pd.DataFrame:
        out = df.copy()
        src = None
        for c in in_col_candidates:
            if c in out.columns:
                src = c
                break
        if src is None:
            out[out_col] = np.nan
            return out

        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
        s = s.replace({
            "m": "male",
            "man": "male",
            "male ": "male",
            "f": "female",
            "woman": "female",
            "female ": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer-not-to-say": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        def norm_str(s):
            return s.astype(str).str.lower().str.strip()

        if "exposure_score" not in df.columns:
            if "aiExposureFrequency" in df.columns:
                df["exposure_score"] = norm_str(df["aiExposureFrequency"]).map(exposure_map)
            else:
                df["exposure_score"] = np.nan

        if "confidence_score" not in df.columns:
            if "aiConfidence" in df.columns:
                df["confidence_score"] = norm_str(df["aiConfidence"]).map(confidence_map)
            else:
                df["confidence_score"] = np.nan

        if "attitude_score" not in df.columns:
            if "aiAttitude" in df.columns:
                df["attitude_score"] = norm_str(df["aiAttitude"]).map(attitude_map)
            else:
                df["attitude_score"] = np.nan

        return df

    def corr_report(df: pd.DataFrame, x: str, y: str, group: str) -> str:
        d = df[[x, y]].dropna()
        n = len(d)
        if n < 10:
            return f"{group}: N<10 (N={n})"
        rp, pp = pearsonr(d[x], d[y])
        rs, ps = spearmanr(d[x], d[y])
        return (f"{group}: N={n} | Pearson r={rp:.3f}, p={pp:.3g} | "
                f"Spearman rho={rs:.3f}, p={ps:.3g}")

    def plot_by_sex_fixed_colors(
        out_dir: str,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        cohort_tag: str,
    ):
        """
        Scatter colored by sex with fixed palette + fixed-color regression lines.
        Saves png/svg + text report.
        """
        ensure_dir(out_dir)

        d = df[[x_col, y_col, "sex"]].dropna().copy()
        d = d[d["sex"].isin(SEX_ORDER)].copy()

        if len(d) < 30:
            print(f"⚠️ [{cohort_tag}] {title}: skipped (N={len(d)} < 30 after sex filter).")
            return

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        # points (fixed palette)
        sns.scatterplot(
            data=d, x=x_col, y=y_col, hue="sex",
            hue_order=SEX_ORDER, palette=SEX_PALETTE,
            alpha=0.25, s=25, edgecolor="none", ax=ax
        )

        # regression line per sex (fixed line colors)
        for sex_g in SEX_ORDER:
            dg = d[d["sex"] == sex_g]
            if len(dg) >= 20:
                sns.regplot(
                    data=dg, x=x_col, y=y_col,
                    scatter=False,
                    line_kws={"color": SEX_PALETTE[sex_g], "linewidth": 3},
                    ax=ax
                )

        ax.set_ylim(0, 100)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"(34) {title} [{cohort_tag}]", fontsize=15, fontweight="bold")

        # legend clean
        handles, labels = ax.get_legend_handles_labels()
        # seaborn adds a "sex" title entry first sometimes; keep stable order
        ax.legend(title="Sex", loc="best")

        base = f"34_{safe_slug(title)}_{cohort_tag}"
        out_png = os.path.join(out_dir, f"{base}.png")
        out_svg = os.path.join(out_dir, f"{base}.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        rep = []
        rep.append(f"(34) {title} [{cohort_tag}]")
        rep.append("-" * 60)
        rep.append(corr_report(d[d["sex"] == "male"], x_col, y_col, "male"))
        rep.append(corr_report(d[d["sex"] == "female"], x_col, y_col, "female"))
        rep.append("")
        rep_txt = "\n".join(rep)

        rep_path = os.path.join(out_dir, f"{base}_report.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(rep_txt)

        print(f"✅ [{cohort_tag}] Saved plots: {out_png}, {out_svg}")
        print(f"✅ [{cohort_tag}] Saved report: {rep_path}")

    def run_one_cohort(cohort_tag: str, main_path: str):
        print(f"\n==================== [{cohort_tag.upper()}] (34) START ====================")
        out_dir = os.path.join(SECTION_DIR, cohort_tag)
        ensure_dir(out_dir)

        df = pd.read_csv(main_path, encoding="utf-8-sig")

        # accuracy
        acc_col = find_accuracy_column(df)
        df, note = convert_accuracy_to_percent_if_needed(df, acc_col)
        print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} | {note}")

        # sex + scores
        df = normalize_sex(df, in_col_candidates=("sex", "gender"), out_col="sex")
        df = add_score_columns(df)

        # numeric safety
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        for c in ["exposure_score", "confidence_score", "attitude_score"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # save prepped
        prepped_out = os.path.join(out_dir, f"34-0_prepped_{cohort_tag}.csv")
        df.to_csv(prepped_out, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved prepped: {prepped_out}")

        # rename acc to accuracy_pct for plotting
        df_plot = df.rename(columns={acc_col: "accuracy_pct"})

        plot_by_sex_fixed_colors(out_dir, df_plot, "exposure_score",   "accuracy_pct",
                                 "Exposure vs Accuracy (by Sex)", cohort_tag)

        plot_by_sex_fixed_colors(out_dir, df_plot, "confidence_score", "accuracy_pct",
                                 "Confidence vs Accuracy (by Sex)", cohort_tag)

        plot_by_sex_fixed_colors(out_dir, df_plot, "attitude_score",   "accuracy_pct",
                                 "Attitude vs Accuracy (by Sex)", cohort_tag)

        print(f"==================== [{cohort_tag.upper()}] (34) END ====================\n")

    if __name__ == "__main__":
        print("==============================================================================")
        print("(34) Sex-stratified scatter (fixed colors): Exposure/Confidence/Attitude vs Accuracy")
        print("==============================================================================\n")

        ensure_dir(SECTION_DIR)

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web":    config.WEB_AGE_FILTERED,
        }

        for cohort_tag, main_path in cohort_files.items():
            if not os.path.exists(main_path):
                print(f"❌ Missing file: {main_path} (skip {cohort_tag})")
                continue
            run_one_cohort(cohort_tag, main_path)

        print("\n==================== (34) DONE ====================")


def _run_cell_153():
    # ==============================================================================
    # (35) TRUE Parallel Mediation with Attitude + Sex-stratified diagrams
    #      (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Model (parallel mediators in the SAME Y model):
    #   X = Age
    #   M1 = AI Exposure (exposure_score)
    #   M2 = AI Confidence (confidence_score)
    #   M3 = AI Attitude (attitude_score)
    #   Y = Accuracy (%)
    #
    # a-paths:   Mi ~ Age
    # b-paths:   Y ~ Age + M1 + M2 + M3   (TRUE parallel)
    # c-total:   Y ~ Age
    # c'-direct: coefficient of Age in the full model
    #
    # indirect_i = a_i * b_i
    # ind_total  = sum(indirect_i)
    #
    # Bootstrap CI (default 5000) for ALL effects.
    #
    # Outputs:
    #   outputs/run_20260119_192624/35_mediation_parallel_attitude_true/<cohort>/
    #     - 35-0_prepped_input_<cohort>_<subset>.csv
    #     - 35-1_bootstrap_effects_<cohort>_<subset>.csv
    #     - 35-2_path_coeffs_<cohort>_<subset>.csv
    #     - 35-3_path_diagram_<cohort>_<subset>.png/.svg
    #     - 35-4_report_<cohort>_<subset>.txt
    #   + cohort_summary.csv, meta.json
    # ==============================================================================

    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    # -----------------------------
    # Config
    # -----------------------------
    RUN_DIR = config.OUTPUTS_DIR / "run_20260119_192624"
    SECTION_DIR = RUN_DIR / "35_mediation_parallel_attitude_true"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    N_BOOT = 5000
    SEED = 42
    SUBSETS = ["all", "male", "female"]   # sex-stratified

    # -----------------------------
    # Helpers: scoring / cleaning
    # -----------------------------
    def choose_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("No accuracy column found (overallAccuracy_y/overallAccuracy/overallAccuracy_x).")

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        df = df.copy()
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        finite = df[acc_col].dropna()
        if finite.empty:
            return df, "accuracy empty"
        if (finite.max() <= 1.5) and (finite.mean() <= 1.0):
            df[acc_col] = df[acc_col] * 100.0
            return df, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return df, f"{acc_col} treated as % already"

    def normalize_sex(df: pd.DataFrame, in_col_candidates=("sex", "gender"), out_col="sex") -> pd.DataFrame:
        out = df.copy()
        src = None
        for c in in_col_candidates:
            if c in out.columns:
                src = c
                break
        if src is None:
            out[out_col] = np.nan
            return out

        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
        s = s.replace({
            "m": "male", "man": "male", "male ": "male",
            "f": "female", "woman": "female", "female ": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer-not-to-say": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {
            "very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5
        }
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        def norm_str(s):
            return s.astype(str).str.lower().str.strip()

        out = df.copy()

        if "exposure_score" not in out.columns:
            if "aiExposureFrequency" in out.columns:
                out["exposure_score"] = norm_str(out["aiExposureFrequency"]).map(exposure_map)
            else:
                out["exposure_score"] = np.nan

        if "confidence_score" not in out.columns:
            if "aiConfidence" in out.columns:
                out["confidence_score"] = norm_str(out["aiConfidence"]).map(confidence_map)
            else:
                out["confidence_score"] = np.nan

        if "attitude_score" not in out.columns:
            if "aiAttitude" in out.columns:
                out["attitude_score"] = norm_str(out["aiAttitude"]).map(attitude_map)
            else:
                out["attitude_score"] = np.nan

        return out

    def prep_input(df_raw: pd.DataFrame, subset_tag: str) -> tuple[pd.DataFrame, str]:
        df = normalize_sex(df_raw)
        df = add_score_columns(df)

        acc_col = choose_accuracy_column(df)
        df, note = ensure_accuracy_percent(df, acc_col)
        df = df.rename(columns={acc_col: "accuracy_pct"}).copy()

        # subset sex
        if subset_tag in ["male", "female"]:
            df = df[df["sex"] == subset_tag].copy()

        # numeric coercion
        for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # complete-cases needed for the model
        need = ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]
        df = df[need].dropna().copy()

        return df, note

    # -----------------------------
    # OLS via numpy lstsq (fast)
    # -----------------------------
    def ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return beta from least squares. X must already include intercept column if desired.
        """
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    def bootstrap_parallel_mediation(df: pd.DataFrame, n_boot=5000, seed=42):
        """
        TRUE parallel mediation with bootstrap CI.
        Returns:
          effects_df: point + CI for a/b/c/c' and indirects (incl total)
          coeffs_df: point estimates for paths (a1,a2,a3,b1,b2,b3,c,c')
        """
        rng = np.random.default_rng(seed)

        # arrays
        age = df["age"].to_numpy(float)
        y = df["accuracy_pct"].to_numpy(float)
        m1 = df["exposure_score"].to_numpy(float)
        m2 = df["confidence_score"].to_numpy(float)
        m3 = df["attitude_score"].to_numpy(float)

        n = len(df)
        if n < 30:
            raise ValueError(f"Not enough complete cases (N={n})")

        # --- point estimates ---
        # a paths: Mi ~ 1 + age
        Xa = np.column_stack([np.ones(n), age])
        a1 = ols_coef(Xa, m1)[1]
        a2 = ols_coef(Xa, m2)[1]
        a3 = ols_coef(Xa, m3)[1]

        # b paths + c': Y ~ 1 + age + m1 + m2 + m3
        Xy = np.column_stack([np.ones(n), age, m1, m2, m3])
        by = ols_coef(Xy, y)
        cprime = by[1]
        b1, b2, b3 = by[2], by[3], by[4]

        # total c: Y ~ 1 + age
        Xc = np.column_stack([np.ones(n), age])
        c = ols_coef(Xc, y)[1]

        ind1 = a1 * b1
        ind2 = a2 * b2
        ind3 = a3 * b3
        ind_total = ind1 + ind2 + ind3

        # --- bootstrap ---
        boot = {
            "a1": [], "a2": [], "a3": [],
            "b1": [], "b2": [], "b3": [],
            "c": [], "cprime": [],
            "ind1": [], "ind2": [], "ind3": [], "ind_total": []
        }

        idx = np.arange(n)
        for _ in range(n_boot):
            s = rng.choice(idx, size=n, replace=True)

            age_b = age[s]
            y_b = y[s]
            m1_b = m1[s]
            m2_b = m2[s]
            m3_b = m3[s]

            # a paths
            Xa_b = np.column_stack([np.ones(n), age_b])
            a1_b = ols_coef(Xa_b, m1_b)[1]
            a2_b = ols_coef(Xa_b, m2_b)[1]
            a3_b = ols_coef(Xa_b, m3_b)[1]

            # b paths + c'
            Xy_b = np.column_stack([np.ones(n), age_b, m1_b, m2_b, m3_b])
            by_b = ols_coef(Xy_b, y_b)
            cprime_b = by_b[1]
            b1_b, b2_b, b3_b = by_b[2], by_b[3], by_b[4]

            # c total
            Xc_b = np.column_stack([np.ones(n), age_b])
            c_b = ols_coef(Xc_b, y_b)[1]

            ind1_b = a1_b * b1_b
            ind2_b = a2_b * b2_b
            ind3_b = a3_b * b3_b
            indt_b = ind1_b + ind2_b + ind3_b

            boot["a1"].append(a1_b); boot["a2"].append(a2_b); boot["a3"].append(a3_b)
            boot["b1"].append(b1_b); boot["b2"].append(b2_b); boot["b3"].append(b3_b)
            boot["c"].append(c_b); boot["cprime"].append(cprime_b)
            boot["ind1"].append(ind1_b); boot["ind2"].append(ind2_b); boot["ind3"].append(ind3_b); boot["ind_total"].append(indt_b)

        def ci(arr):
            arr = np.asarray(arr, float)
            return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

        point = {
            "a1": a1, "a2": a2, "a3": a3,
            "b1": b1, "b2": b2, "b3": b3,
            "c": c, "cprime": cprime,
            "ind1": ind1, "ind2": ind2, "ind3": ind3, "ind_total": ind_total
        }

        effects_rows = []
        for k in ["a1","a2","a3","b1","b2","b3","c","cprime","ind1","ind2","ind3","ind_total"]:
            lo, hi = ci(boot[k])
            sig = not (lo <= 0 <= hi)
            effects_rows.append({
                "effect": k,
                "coef": float(point[k]),
                "CI[2.5%]": lo,
                "CI[97.5%]": hi,
                "sig_CI_nonzero": bool(sig)
            })

        effects_df = pd.DataFrame(effects_rows)

        coeffs_df = pd.DataFrame([{
            "N": int(n),
            "a_exposure": float(a1),
            "a_confidence": float(a2),
            "a_attitude": float(a3),
            "b_exposure": float(b1),
            "b_confidence": float(b2),
            "b_attitude": float(b3),
            "c_total": float(c),
            "cprime_direct": float(cprime),
            "ind_exposure": float(ind1),
            "ind_confidence": float(ind2),
            "ind_attitude": float(ind3),
            "ind_total": float(ind_total),
        }])

        return effects_df, coeffs_df

    # -----------------------------
    # Label formatting for diagram (CI-based)
    # -----------------------------
    def label_from_effects(effects_df: pd.DataFrame, key: str) -> str:
        r = effects_df[effects_df["effect"] == key]
        if len(r) == 0:
            return "NA"
        r = r.iloc[0]
        coef = float(r["coef"])
        lo = float(r["CI[2.5%]"])
        hi = float(r["CI[97.5%]"])
        sig = bool(r["sig_CI_nonzero"])
        return f"{coef:.2f} ({'sig' if sig else 'n.s.'})"

    # -----------------------------
    # Diagram drawing (3 mediators) - same style as your Section 17
    # -----------------------------
    def draw_diagram_3M(title: str, out_base: Path,
                        a1: str, a2: str, a3: str,
                        b1: str, b2: str, b3: str,
                        cprime: str, ctotal: str,
                        ind1: str, ind2: str, ind3: str, indt: str):
        fig, ax = plt.subplots(figsize=(16, 11))

        pos = {
            "Age (X)": (0.0, 0.0),
            "AI Exposure (M1)": (0.62, 0.72),
            "AI Confidence (M2)": (0.62, 0.00),
            "AI Attitude (M3)": (0.62, -0.72),
            "Accuracy (Y)": (1.35, 0.0),
        }

        node_style = dict(boxstyle="round,pad=0.75", fc="skyblue", ec="black", lw=1.6)
        arrow_style = dict(arrowstyle="->,head_width=0.35,head_length=0.75", color="black", lw=2.2)

        for name, (x, y) in pos.items():
            ax.text(x, y, name, ha="center", va="center", fontsize=16, fontweight="bold", bbox=node_style)

        def draw_path(src, dst, label, rad=0.15, yoff=0.08):
            ax.annotate(
                "",
                xy=pos[dst], xytext=pos[src],
                arrowprops={**arrow_style, "connectionstyle": f"arc3,rad={rad}"}
            )
            mx = (pos[src][0] + pos[dst][0]) / 2
            my = (pos[src][1] + pos[dst][1]) / 2 + yoff
            ax.text(
                mx, my, label,
                ha="center", va="center", fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            )

        # X -> Ms
        draw_path("Age (X)", "AI Exposure (M1)", a1, rad=0.20, yoff=0.12)
        draw_path("Age (X)", "AI Confidence (M2)", a2, rad=0.00, yoff=0.12)
        draw_path("Age (X)", "AI Attitude (M3)", a3, rad=-0.20, yoff=0.12)

        # Ms -> Y
        draw_path("AI Exposure (M1)", "Accuracy (Y)", b1, rad=0.20, yoff=0.12)
        draw_path("AI Confidence (M2)", "Accuracy (Y)", b2, rad=0.00, yoff=0.12)
        draw_path("AI Attitude (M3)", "Accuracy (Y)", b3, rad=-0.20, yoff=0.12)

        # Direct
        draw_path("Age (X)", "Accuracy (Y)", cprime, rad=0.00, yoff=-0.16)

        ax.set_title(title, fontsize=20, fontweight="bold", pad=18)

        summary = (
            f"Total effect (c): {ctotal}\n"
            f"Direct effect (c'): {cprime}\n\n"
            f"Indirect 1 (Age → Exposure → Accuracy): {ind1}\n"
            f"Indirect 2 (Age → Confidence → Accuracy): {ind2}\n"
            f"Indirect 3 (Age → Attitude → Accuracy): {ind3}\n"
            f"Indirect TOTAL (sum): {indt}\n\n"
            f"Labels: coef (sig/n.s.) based on 95% bootstrap CI"
        )

        ax.text(
            0.80, -1.24, summary,
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E5", ec="gray", lw=1),
        )

        ax.set_xlim(-0.35, 1.55)
        ax.set_ylim(-1.50, 1.50)
        ax.axis("off")

        out_base.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.savefig(str(out_base) + ".svg", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.show()

    # -----------------------------
    # Run one cohort/subset
    # -----------------------------
    def run_one(df_raw: pd.DataFrame, cohort_tag: str, subset_tag: str, out_dir: Path,
                n_boot=5000, seed=42):
        d, acc_note = prep_input(df_raw, subset_tag=subset_tag)

        if len(d) < 30:
            raise ValueError(f"[{cohort_tag}|{subset_tag}] Not enough complete cases (N={len(d)})")

        effects_df, coeffs_df = bootstrap_parallel_mediation(d, n_boot=n_boot, seed=seed)

        out_dir.mkdir(parents=True, exist_ok=True)
        in_path = out_dir / f"35-0_prepped_input_{cohort_tag}_{subset_tag}.csv"
        eff_path = out_dir / f"35-1_bootstrap_effects_{cohort_tag}_{subset_tag}.csv"
        coef_path = out_dir / f"35-2_path_coeffs_{cohort_tag}_{subset_tag}.csv"
        rep_path = out_dir / f"35-4_report_{cohort_tag}_{subset_tag}.txt"

        d.to_csv(in_path, index=False, encoding="utf-8-sig")
        effects_df.to_csv(eff_path, index=False, encoding="utf-8-sig")
        coeffs_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

        # labels for diagram
        a1 = label_from_effects(effects_df, "a1")
        a2 = label_from_effects(effects_df, "a2")
        a3 = label_from_effects(effects_df, "a3")
        b1 = label_from_effects(effects_df, "b1")
        b2 = label_from_effects(effects_df, "b2")
        b3 = label_from_effects(effects_df, "b3")
        c  = label_from_effects(effects_df, "c")
        cp = label_from_effects(effects_df, "cprime")
        ind1 = label_from_effects(effects_df, "ind1")
        ind2 = label_from_effects(effects_df, "ind2")
        ind3 = label_from_effects(effects_df, "ind3")
        indt = label_from_effects(effects_df, "ind_total")

        title = f"(35) TRUE Parallel Mediation (Exposure+Confidence+Attitude) [{cohort_tag.upper()} | {subset_tag}]"
        fig_base = out_dir / f"35-3_path_diagram_{cohort_tag}_{subset_tag}"

        draw_diagram_3M(
            title=title,
            out_base=fig_base,
            a1=a1, a2=a2, a3=a3,
            b1=b1, b2=b2, b3=b3,
            cprime=cp, ctotal=c,
            ind1=ind1, ind2=ind2, ind3=ind3, indt=indt
        )

        # report (text)
        rep_lines = []
        rep_lines.append(f"(35) TRUE Parallel Mediation with Attitude [{cohort_tag} | {subset_tag}]")
        rep_lines.append("-" * 80)
        rep_lines.append(f"- N complete: {len(d)}")
        rep_lines.append(f"- Accuracy scaling note: {acc_note}")
        rep_lines.append("")
        rep_lines.append("Key effects (coef, 95% CI, sig_CI_nonzero):")
        rep_lines.append(effects_df.to_string(index=False))
        rep_lines.append("")
        rep_path.write_text("\n".join(rep_lines), encoding="utf-8")

        summary = {
            "cohort": cohort_tag,
            "subset": subset_tag,
            "N_complete": int(len(d)),
            "accuracy_note": acc_note,
            "saved_input": str(in_path),
            "saved_effects": str(eff_path),
            "saved_coeffs": str(coef_path),
            "saved_figure_base": str(fig_base),
            "saved_report": str(rep_path),
        }
        return summary

    # -----------------------------
    # MAIN
    # -----------------------------
    print("==============================================================================")
    print("(35) TRUE Parallel Mediation with Attitude + Sex-stratified diagrams (MOBILE + WEB)")
    print("==============================================================================\n")

    summaries = []

    for cohort_tag, f in COHORT_FILES.items():
        p = Path(f)
        if not p.exists():
            print(f"❌ Missing cohort file: {f} (skip {cohort_tag})")
            continue

        df_raw = pd.read_csv(p, encoding="utf-8-sig")
        cohort_out = SECTION_DIR / cohort_tag
        cohort_out.mkdir(parents=True, exist_ok=True)

        print(f"\n==================== [{cohort_tag.upper()}] (35) START ====================")
        print(f"✅ Loaded: {f} rows={len(df_raw):,}")

        for subset_tag in SUBSETS:
            try:
                s = run_one(df_raw, cohort_tag, subset_tag, cohort_out, n_boot=N_BOOT, seed=SEED)
                summaries.append(s)
                print(f"✅ [{cohort_tag}|{subset_tag}] done. N_complete={s['N_complete']:,}")
            except Exception as e:
                print(f"❌ [{cohort_tag}|{subset_tag}] failed: {e}")

        print(f"==================== [{cohort_tag.upper()}] (35) END ====================\n")

    # save cohort summary + meta
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = SECTION_DIR / "cohort_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ cohort summary saved: {summary_path}")

    meta = {
        "section": "35_mediation_parallel_attitude_true",
        "created_at": datetime.now().isoformat(),
        "n_boot": N_BOOT,
        "seed": SEED,
        "input_files": COHORT_FILES,
        "outputs_dir": str(SECTION_DIR),
        "note": "TRUE parallel mediation: Y model includes all mediators simultaneously; bootstrap CI for paths and indirect effects."
    }
    meta_path = SECTION_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ meta saved: {meta_path}")

    print("\n==================== (35) DONE ====================")


def _run_cell_154():
    # ==============================================================================
    # (35.1) TRUE Parallel Mediation + Sex Difference (Female - Male) + CONSOLE PRINT
    #         (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Same as prior 35.1, but ALSO prints key results to console:
    #   - male/female/diff tables (head or key effects)
    #   - ind_total / ind1/ind2/ind3 / c / cprime quick summary
    # ==============================================================================

    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    # -----------------------------
    # Config
    # -----------------------------
    RUN_DIR = config.OUTPUTS_DIR / "run_20260119_192624"
    SECTION_DIR = RUN_DIR / "35_1_sex_diff_text"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    N_BOOT = 5000
    SEED = 42

    # -----------------------------
    # Helpers: scoring / cleaning
    # -----------------------------
    def choose_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("No accuracy column found (overallAccuracy_y/overallAccuracy/overallAccuracy_x).")

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        df = df.copy()
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        finite = df[acc_col].dropna()
        if finite.empty:
            return df, "accuracy empty"
        if (finite.max() <= 1.5) and (finite.mean() <= 1.0):
            df[acc_col] = df[acc_col] * 100.0
            return df, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return df, f"{acc_col} treated as % already"

    def normalize_sex(df: pd.DataFrame, in_col_candidates=("sex", "gender"), out_col="sex") -> pd.DataFrame:
        out = df.copy()
        src = None
        for c in in_col_candidates:
            if c in out.columns:
                src = c
                break
        if src is None:
            out[out_col] = np.nan
            return out

        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
        s = s.replace({
            "m": "male", "man": "male", "male ": "male",
            "f": "female", "woman": "female", "female ": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer-not-to-say": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {
            "very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5
        }
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        def norm_str(s):
            return s.astype(str).str.lower().str.strip()

        out = df.copy()
        if "exposure_score" not in out.columns:
            out["exposure_score"] = norm_str(out["aiExposureFrequency"]).map(exposure_map) if "aiExposureFrequency" in out.columns else np.nan
        if "confidence_score" not in out.columns:
            out["confidence_score"] = norm_str(out["aiConfidence"]).map(confidence_map) if "aiConfidence" in out.columns else np.nan
        if "attitude_score" not in out.columns:
            out["attitude_score"] = norm_str(out["aiAttitude"]).map(attitude_map) if "aiAttitude" in out.columns else np.nan
        return out

    def prep_sex_subset(df_raw: pd.DataFrame, sex_value: str) -> tuple[pd.DataFrame, str]:
        df = normalize_sex(df_raw)
        df = add_score_columns(df)

        acc_col = choose_accuracy_column(df)
        df, note = ensure_accuracy_percent(df, acc_col)
        df = df.rename(columns={acc_col: "accuracy_pct"}).copy()

        df = df[df["sex"] == sex_value].copy()

        for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        need = ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]
        df = df[need].dropna().copy()
        return df, note

    # -----------------------------
    # OLS via numpy lstsq
    # -----------------------------
    def ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    def point_estimates(df: pd.DataFrame) -> dict:
        age = df["age"].to_numpy(float)
        y   = df["accuracy_pct"].to_numpy(float)
        m1  = df["exposure_score"].to_numpy(float)
        m2  = df["confidence_score"].to_numpy(float)
        m3  = df["attitude_score"].to_numpy(float)
        n = len(df)

        Xa = np.column_stack([np.ones(n), age])
        a1 = ols_coef(Xa, m1)[1]
        a2 = ols_coef(Xa, m2)[1]
        a3 = ols_coef(Xa, m3)[1]

        Xy = np.column_stack([np.ones(n), age, m1, m2, m3])
        by = ols_coef(Xy, y)
        cprime = by[1]
        b1, b2, b3 = by[2], by[3], by[4]

        Xc = np.column_stack([np.ones(n), age])
        c = ols_coef(Xc, y)[1]

        ind1 = a1*b1
        ind2 = a2*b2
        ind3 = a3*b3
        ind_total = ind1+ind2+ind3

        return {
            "N": int(n),
            "a1": float(a1), "a2": float(a2), "a3": float(a3),
            "b1": float(b1), "b2": float(b2), "b3": float(b3),
            "c": float(c), "cprime": float(cprime),
            "ind1": float(ind1), "ind2": float(ind2), "ind3": float(ind3),
            "ind_total": float(ind_total),
        }

    def bootstrap_effects(df: pd.DataFrame, n_boot=5000, seed=42) -> tuple[dict, dict]:
        rng = np.random.default_rng(seed)
        point = point_estimates(df)

        age = df["age"].to_numpy(float)
        y   = df["accuracy_pct"].to_numpy(float)
        m1  = df["exposure_score"].to_numpy(float)
        m2  = df["confidence_score"].to_numpy(float)
        m3  = df["attitude_score"].to_numpy(float)
        n = len(df)
        idx = np.arange(n)

        keys = ["a1","a2","a3","b1","b2","b3","c","cprime","ind1","ind2","ind3","ind_total"]
        boot = {k: [] for k in keys}

        for _ in range(n_boot):
            s = rng.choice(idx, size=n, replace=True)

            age_b = age[s]; y_b = y[s]
            m1_b = m1[s]; m2_b = m2[s]; m3_b = m3[s]

            Xa = np.column_stack([np.ones(n), age_b])
            a1 = ols_coef(Xa, m1_b)[1]
            a2 = ols_coef(Xa, m2_b)[1]
            a3 = ols_coef(Xa, m3_b)[1]

            Xy = np.column_stack([np.ones(n), age_b, m1_b, m2_b, m3_b])
            by = ols_coef(Xy, y_b)
            cprime = by[1]
            b1, b2, b3 = by[2], by[3], by[4]

            Xc = np.column_stack([np.ones(n), age_b])
            c = ols_coef(Xc, y_b)[1]

            ind1 = a1*b1
            ind2 = a2*b2
            ind3 = a3*b3
            indt = ind1+ind2+ind3

            boot["a1"].append(a1); boot["a2"].append(a2); boot["a3"].append(a3)
            boot["b1"].append(b1); boot["b2"].append(b2); boot["b3"].append(b3)
            boot["c"].append(c); boot["cprime"].append(cprime)
            boot["ind1"].append(ind1); boot["ind2"].append(ind2); boot["ind3"].append(ind3); boot["ind_total"].append(indt)

        for k in boot:
            boot[k] = np.asarray(boot[k], float)

        return point, boot

    def effects_table(point: dict, boot: dict, label: str) -> pd.DataFrame:
        rows = []
        for k, arr in boot.items():
            lo = float(np.quantile(arr, 0.025))
            hi = float(np.quantile(arr, 0.975))
            coef = float(point[k])
            sig = not (lo <= 0 <= hi)
            rows.append({
                "group": label,
                "effect": k,
                "coef": coef,
                "CI[2.5%]": lo,
                "CI[97.5%]": hi,
                "sig_CI_nonzero": bool(sig),
            })
        return pd.DataFrame(rows)

    def diff_table(boot_f: dict, boot_m: dict, label="female_minus_male") -> pd.DataFrame:
        rows = []
        for k in boot_f.keys():
            diff = boot_f[k] - boot_m[k]
            lo = float(np.quantile(diff, 0.025))
            hi = float(np.quantile(diff, 0.975))
            coef = float(np.mean(diff))  # bootstrap mean for diff
            sig = not (lo <= 0 <= hi)
            rows.append({
                "group": label,
                "effect": k,
                "coef": coef,
                "CI[2.5%]": lo,
                "CI[97.5%]": hi,
                "sig_CI_nonzero": bool(sig),
            })
        return pd.DataFrame(rows)

    def print_key_block(title: str, df: pd.DataFrame, effects=("ind_total","ind1","ind2","ind3","c","cprime")):
        sub = df[df["effect"].isin(effects)].copy()
        # keep a nice order
        cat = pd.Categorical(sub["effect"], categories=list(effects), ordered=True)
        sub = sub.assign(effect=cat).sort_values("effect")
        print(title)
        print(sub[["effect","coef","CI[2.5%]","CI[97.5%]","sig_CI_nonzero"]].to_string(index=False))
        print("")

    def write_report(path: Path, cohort: str, n_m: int, n_f: int,
                     note_m: str, note_f: str,
                     male_df: pd.DataFrame, female_df: pd.DataFrame, diff_df: pd.DataFrame):
        lines = []
        lines.append("==============================================================================")
        lines.append(f"(35.1) TRUE Parallel Mediation + Sex Difference Report [{cohort.upper()}]")
        lines.append("==============================================================================")
        lines.append("")
        lines.append(f"- N (male complete):   {n_m:,}")
        lines.append(f"- N (female complete): {n_f:,}")
        lines.append(f"- Accuracy note (male):   {note_m}")
        lines.append(f"- Accuracy note (female): {note_f}")
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("[MALE] effects (coef, 95% bootstrap CI, sig_CI_nonzero)")
        lines.append("------------------------------------------------------------")
        lines.append(male_df.to_string(index=False))
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("[FEMALE] effects (coef, 95% bootstrap CI, sig_CI_nonzero)")
        lines.append("------------------------------------------------------------")
        lines.append(female_df.to_string(index=False))
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("[DIFF] FEMALE - MALE (bootstrap distribution) ✅")
        lines.append("------------------------------------------------------------")
        lines.append(diff_df.to_string(index=False))
        lines.append("")
        lines.append("Interpretation tips:")
        lines.append("- For DIFF: if CI excludes 0 => sex difference evidence for that path/effect.")
        lines.append("- ind_total: sum of indirect effects through Exposure+Confidence+Attitude.")
        lines.append("- ind1/ind2/ind3 correspond to Exposure/Confidence/Attitude respectively.")
        path.write_text("\n".join(lines), encoding="utf-8")

    # -----------------------------
    # MAIN
    # -----------------------------
    print("==============================================================================")
    print("(35.1) TRUE Parallel Mediation + Sex Difference (Female - Male) - MOBILE + WEB")
    print("==============================================================================\n")

    summaries = []

    for cohort_tag, f in COHORT_FILES.items():
        p = Path(f)
        if not p.exists():
            print(f"❌ Missing cohort file: {f} (skip {cohort_tag})")
            continue

        df_raw = pd.read_csv(p, encoding="utf-8-sig")
        out_dir = SECTION_DIR / cohort_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==================== [{cohort_tag.upper()}] (35.1) START ====================")
        print(f"✅ Loaded: {f} rows={len(df_raw):,}")

        df_m, note_m = prep_sex_subset(df_raw, "male")
        df_f, note_f = prep_sex_subset(df_raw, "female")
        n_m = len(df_m); n_f = len(df_f)

        print(f"- complete-case male:   {n_m:,}")
        print(f"- complete-case female: {n_f:,}")

        if n_m < 30 or n_f < 30:
            print(f"❌ [{cohort_tag}] Not enough complete cases for sex-diff bootstrap (male={n_m}, female={n_f}).")
            continue

        # bootstrap
        point_m, boot_m = bootstrap_effects(df_m, n_boot=N_BOOT, seed=SEED + 1)
        point_f, boot_f = bootstrap_effects(df_f, n_boot=N_BOOT, seed=SEED + 2)

        male_df = effects_table(point_m, boot_m, label="male")
        female_df = effects_table(point_f, boot_f, label="female")
        diff_df = diff_table(boot_f, boot_m, label="female_minus_male")

        # -----------------------------
        # ✅ CONSOLE PRINT (key summaries + full tables optional)
        # -----------------------------
        print("\n--------------------------------------------")
        print(f"[{cohort_tag.upper()}] KEY SUMMARY (MALE)")
        print("--------------------------------------------")
        print_key_block("", male_df)

        print("--------------------------------------------")
        print(f"[{cohort_tag.upper()}] KEY SUMMARY (FEMALE)")
        print("--------------------------------------------")
        print_key_block("", female_df)

        print("--------------------------------------------")
        print(f"[{cohort_tag.upper()}] KEY SUMMARY (DIFF = FEMALE - MALE)")
        print("--------------------------------------------")
        print_key_block("", diff_df)

        # If you want FULL tables in console, uncomment:
        # print("[MALE full]\n", male_df.to_string(index=False), "\n")
        # print("[FEMALE full]\n", female_df.to_string(index=False), "\n")
        # print("[DIFF full]\n", diff_df.to_string(index=False), "\n")

        # save csvs
        male_path = out_dir / f"35.1-1_effects_male_{cohort_tag}.csv"
        female_path = out_dir / f"35.1-2_effects_female_{cohort_tag}.csv"
        diff_path = out_dir / f"35.1-3_effects_diff_female_minus_male_{cohort_tag}.csv"
        male_df.to_csv(male_path, index=False, encoding="utf-8-sig")
        female_df.to_csv(female_path, index=False, encoding="utf-8-sig")
        diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")

        rep_path = out_dir / f"35.1-4_report_{cohort_tag}.txt"
        write_report(rep_path, cohort_tag, n_m, n_f, note_m, note_f, male_df, female_df, diff_df)

        print(f"✅ Saved: {male_path.name}, {female_path.name}, {diff_path.name}")
        print(f"✅ Report: {rep_path}")

        # summary row (quick)
        indt = diff_df[diff_df["effect"] == "ind_total"].iloc[0].to_dict()
        summaries.append({
            "cohort": cohort_tag,
            "N_male": int(n_m),
            "N_female": int(n_f),
            "diff_ind_total_coef": float(indt["coef"]),
            "diff_ind_total_CI2.5": float(indt["CI[2.5%]"]),
            "diff_ind_total_CI97.5": float(indt["CI[97.5%]"]),
            "diff_ind_total_sig": bool(indt["sig_CI_nonzero"]),
            "report_path": str(rep_path),
        })

        print(f"==================== [{cohort_tag.upper()}] (35.1) END ====================\n")

    # save cohort summary + meta
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = SECTION_DIR / "cohort_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ cohort summary saved: {summary_path}")

    meta = {
        "section": "35.1_sex_diff_text_console",
        "created_at": datetime.now().isoformat(),
        "n_boot": N_BOOT,
        "seed": SEED,
        "input_files": COHORT_FILES,
        "outputs_dir": str(SECTION_DIR),
        "note": "Same as 35.1 but prints key effect tables to console."
    }
    meta_path = SECTION_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ meta saved: {meta_path}")

    print("\n==================== (35.1) DONE ====================")


def _run_cell_157():
    # ==============================================================================
    # (35.2) Slope Equality Test (Female vs Male) for a- and b-paths
    #         Parallel Mediation with Exposure/Confidence/Attitude (MOBILE + WEB)
    # ------------------------------------------------------------------------------
    # Tests whether specific path coefficients differ by Sex:
    #   a-paths: Mi ~ Age
    #     - a1: Exposure ~ Age
    #     - a2: Confidence ~ Age
    #     - a3: Attitude ~ Age
    #   b-paths: Accuracy ~ Age + Exposure + Confidence + Attitude
    #     - b1: Exposure -> Accuracy
    #     - b2: Confidence -> Accuracy
    #     - b3: Attitude -> Accuracy
    #
    # For each path:
    #   Z = (b_female - b_male) / sqrt(SE_female^2 + SE_male^2)
    #   p = 2*(1 - Phi(|Z|))
    #
    # Saves:
    #   35.2-1_path_coeffs_by_sex_<cohort>.csv   (coef + SE by sex)
    #   35.2-2_slope_equality_tests_<cohort>.csv (Z + p + sig)
    #   35.2-3_report_<cohort>.txt
    #   cohort_summary.csv, meta.json
    # ==============================================================================

    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    import statsmodels.api as sm
    from scipy.stats import norm

    # -----------------------------
    # Config
    # -----------------------------
    RUN_DIR = config.OUTPUTS_DIR / "run_20260119_192624"
    SECTION_DIR = RUN_DIR / "35_2_slope_equality_test"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    ALPHA = 0.05

    # -----------------------------
    # Helpers (same mappings as earlier)
    # -----------------------------
    def choose_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("No accuracy column found (overallAccuracy_y/overallAccuracy/overallAccuracy_x).")

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        df = df.copy()
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        finite = df[acc_col].dropna()
        if finite.empty:
            return df, "accuracy empty"
        if (finite.max() <= 1.5) and (finite.mean() <= 1.0):
            df[acc_col] = df[acc_col] * 100.0
            return df, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return df, f"{acc_col} treated as % already"

    def normalize_sex(df: pd.DataFrame, in_col_candidates=("sex", "gender"), out_col="sex") -> pd.DataFrame:
        out = df.copy()
        src = None
        for c in in_col_candidates:
            if c in out.columns:
                src = c
                break
        if src is None:
            out[out_col] = np.nan
            return out
        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
        s = s.replace({
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer-not-to-say": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        def norm_str(s):
            return s.astype(str).str.lower().str.strip()

        out = df.copy()
        if "exposure_score" not in out.columns:
            out["exposure_score"] = norm_str(out["aiExposureFrequency"]).map(exposure_map) if "aiExposureFrequency" in out.columns else np.nan
        if "confidence_score" not in out.columns:
            out["confidence_score"] = norm_str(out["aiConfidence"]).map(confidence_map) if "aiConfidence" in out.columns else np.nan
        if "attitude_score" not in out.columns:
            out["attitude_score"] = norm_str(out["aiAttitude"]).map(attitude_map) if "aiAttitude" in out.columns else np.nan
        return out

    def prep_df(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        df = normalize_sex(df_raw)
        df = add_score_columns(df)

        acc_col = choose_accuracy_column(df)
        df, note = ensure_accuracy_percent(df, acc_col)
        df = df.rename(columns={acc_col: "accuracy_pct"}).copy()

        for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # only male/female
        df = df[df["sex"].isin(["male", "female"])].copy()

        # complete cases for all models used below
        need = ["sex", "age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]
        df = df[need].dropna().copy()
        return df, note

    # -----------------------------
    # Model fitting + extraction
    # -----------------------------
    def fit_a_model(df: pd.DataFrame, dv: str):
        # dv ~ age
        X = sm.add_constant(df[["age"]], has_constant="add")
        y = df[dv]
        model = sm.OLS(y, X).fit()
        coef = float(model.params["age"])
        se = float(model.bse["age"])
        return coef, se, model

    def fit_b_model(df: pd.DataFrame):
        # accuracy ~ age + exposure + confidence + attitude
        X = sm.add_constant(df[["age", "exposure_score", "confidence_score", "attitude_score"]], has_constant="add")
        y = df["accuracy_pct"]
        model = sm.OLS(y, X).fit()
        out = {}
        for k in ["exposure_score", "confidence_score", "attitude_score"]:
            out[k] = (float(model.params[k]), float(model.bse[k]))
        # also keep age direct coef if needed
        out["age"] = (float(model.params["age"]), float(model.bse["age"]))
        return out, model

    def z_test_diff(coef_f, se_f, coef_m, se_m):
        denom = np.sqrt(se_f**2 + se_m**2)
        if denom <= 0 or not np.isfinite(denom):
            return np.nan, np.nan
        z = (coef_f - coef_m) / denom
        p = 2 * (1 - norm.cdf(abs(z)))
        return float(z), float(p)

    # -----------------------------
    # MAIN
    # -----------------------------
    print("==============================================================================")
    print("(35.2) Slope Equality Test (Female vs Male) - MOBILE + WEB")
    print("==============================================================================\n")

    summaries = []

    for cohort, f in COHORT_FILES.items():
        p = Path(f)
        if not p.exists():
            print(f"❌ Missing file: {f} (skip {cohort})")
            continue

        out_dir = SECTION_DIR / cohort
        out_dir.mkdir(parents=True, exist_ok=True)

        df_raw = pd.read_csv(p, encoding="utf-8-sig")
        df, note = prep_df(df_raw)

        n_m = int((df["sex"] == "male").sum())
        n_f = int((df["sex"] == "female").sum())

        print(f"\n==================== [{cohort.upper()}] (35.2) START ====================")
        print(f"✅ Loaded: {f} rows={len(df_raw):,} -> complete male/female N={len(df):,}")
        print(f"- male:   {n_m:,}")
        print(f"- female: {n_f:,}")
        print(f"- accuracy note: {note}")

        if n_m < 30 or n_f < 30:
            print(f"❌ Not enough data for stable slope tests (male={n_m}, female={n_f})")
            continue

        df_m = df[df["sex"] == "male"].copy()
        df_f = df[df["sex"] == "female"].copy()

        # ---- a-paths ----
        a_paths = {
            "a1_age_to_exposure": "exposure_score",
            "a2_age_to_confidence": "confidence_score",
            "a3_age_to_attitude": "attitude_score",
        }

        rows_coeffs = []
        rows_tests = []

        for path_name, dv in a_paths.items():
            coef_m, se_m, _ = fit_a_model(df_m, dv)
            coef_f, se_f, _ = fit_a_model(df_f, dv)

            rows_coeffs += [
                {"path": path_name, "sex": "male", "coef": coef_m, "se": se_m},
                {"path": path_name, "sex": "female", "coef": coef_f, "se": se_f},
            ]

            z, pval = z_test_diff(coef_f, se_f, coef_m, se_m)
            rows_tests.append({
                "path": path_name,
                "coef_male": coef_m, "se_male": se_m,
                "coef_female": coef_f, "se_female": se_f,
                "diff_female_minus_male": coef_f - coef_m,
                "Z": z,
                "p": pval,
                "sig_p_lt_0.05": bool(pval < ALPHA) if np.isfinite(pval) else False,
            })

        # ---- b-paths ----
        b_m, _ = fit_b_model(df_m)
        b_f, _ = fit_b_model(df_f)

        b_map = {
            "b1_exposure_to_accuracy": "exposure_score",
            "b2_confidence_to_accuracy": "confidence_score",
            "b3_attitude_to_accuracy": "attitude_score",
        }

        for path_name, k in b_map.items():
            coef_m, se_m = b_m[k]
            coef_f, se_f = b_f[k]

            rows_coeffs += [
                {"path": path_name, "sex": "male", "coef": coef_m, "se": se_m},
                {"path": path_name, "sex": "female", "coef": coef_f, "se": se_f},
            ]

            z, pval = z_test_diff(coef_f, se_f, coef_m, se_m)
            rows_tests.append({
                "path": path_name,
                "coef_male": coef_m, "se_male": se_m,
                "coef_female": coef_f, "se_female": se_f,
                "diff_female_minus_male": coef_f - coef_m,
                "Z": z,
                "p": pval,
                "sig_p_lt_0.05": bool(pval < ALPHA) if np.isfinite(pval) else False,
            })

        coeffs_df = pd.DataFrame(rows_coeffs)
        tests_df = pd.DataFrame(rows_tests)

        # save
        coeffs_path = out_dir / f"35.2-1_path_coeffs_by_sex_{cohort}.csv"
        tests_path = out_dir / f"35.2-2_slope_equality_tests_{cohort}.csv"
        coeffs_df.to_csv(coeffs_path, index=False, encoding="utf-8-sig")
        tests_df.to_csv(tests_path, index=False, encoding="utf-8-sig")

        # console print (핵심만)
        print("\n--- Slope equality tests (female - male) ---")
        print(tests_df[["path","diff_female_minus_male","Z","p","sig_p_lt_0.05"]].to_string(index=False))

        # text report
        rep_path = out_dir / f"35.2-3_report_{cohort}.txt"
        report = []
        report.append("==============================================================================")
        report.append(f"(35.2) Slope Equality Test Report [{cohort.upper()}]")
        report.append("==============================================================================")
        report.append("")
        report.append(f"- N used (complete cases): {len(df):,}")
        report.append(f"- male: {n_m:,} | female: {n_f:,}")
        report.append(f"- accuracy note: {note}")
        report.append("")
        report.append("Interpretation:")
        report.append("- Z test compares coefficients directly between female and male.")
        report.append("- If |Z| > 1.96 (p<.05), slopes differ (approx. Wald test).")
        report.append("")
        report.append("Results table:")
        report.append(tests_df.to_string(index=False))
        rep_path.write_text("\n".join(report), encoding="utf-8")

        print(f"\n✅ Saved: {coeffs_path.name}, {tests_path.name}, {rep_path.name}")

        # summary row for quick view
        # focus on b paths (what you want to decompose ind differences into a vs b)
        b2 = tests_df[tests_df["path"] == "b2_confidence_to_accuracy"].iloc[0]
        summaries.append({
            "cohort": cohort,
            "N_used": int(len(df)),
            "N_male": n_m,
            "N_female": n_f,
            "b2_diff_female_minus_male": float(b2["diff_female_minus_male"]),
            "b2_Z": float(b2["Z"]),
            "b2_p": float(b2["p"]),
            "b2_sig": bool(b2["sig_p_lt_0.05"]),
            "report_path": str(rep_path),
        })

        print(f"==================== [{cohort.upper()}] (35.2) END ====================\n")

    # cohort summary + meta
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = SECTION_DIR / "cohort_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ cohort summary saved: {summary_path}")

    meta = {
        "section": "35.2_slope_equality_test",
        "created_at": datetime.now().isoformat(),
        "alpha": ALPHA,
        "input_files": COHORT_FILES,
        "outputs_dir": str(SECTION_DIR),
        "note": "Z-test compares female vs male path coefficients (a and b paths) using OLS SEs. Run per cohort."
    }
    meta_path = SECTION_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ meta saved: {meta_path}")

    print("\n==================== (35.2) DONE ====================")


def _run_cell_160():
    # ==============================================================================
    # (36) Final Path Diagram from 35.1 outputs (AUTO-DISCOVERY, Graphviz) - FIXED
    # ------------------------------------------------------------------------------
    # Fixes:
    # 1) No use of g_svg.source setter (read-only). Render PNG then SVG by switching g.format.
    # 2) Avoid non-ASCII arrows to reduce pango width warnings: use "->" instead of "→"
    # 3) Force a common Windows font ("Arial") for better stability.
    # ==============================================================================

    import json
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    RUN_DIR = config.OUTPUTS_DIR / "run_20260119_192624"
    OUT_DIR = RUN_DIR / "36_final_path_diagram"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    COHORTS = ["mobile", "web"]

    ARCS = [
        ("Age", "Exposure",   "a1",     "Age -> Exposure (a1)"),
        ("Age", "Confidence", "a2",     "Age -> Confidence (a2)"),
        ("Age", "Attitude",   "a3",     "Age -> Attitude (a3)"),
        ("Exposure",   "Accuracy", "b1","Exposure -> Accuracy (b1)"),
        ("Confidence", "Accuracy", "b2","Confidence -> Accuracy (b2)"),
        ("Attitude",   "Accuracy", "b3","Attitude -> Accuracy (b3)"),
        ("Age", "Accuracy",   "cprime", "Age -> Accuracy (c')"),
    ]

    def safe_graphviz_import():
        try:
            import graphviz  # noqa
            return True
        except Exception:
            return False

    def find_35_1_files(run_dir: Path, cohort: str):
        patterns = {
            "male":   f"35.1-1_effects_male_{cohort}.csv",
            "female": f"35.1-2_effects_female_{cohort}.csv",
            "diff":   f"35.1-3_effects_diff_female_minus_male_{cohort}.csv",
        }
        found = {}
        for k, fn in patterns.items():
            hits = list(run_dir.rglob(fn))
            if len(hits) == 0:
                found[k] = None
            else:
                hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
                found[k] = hits[0]
        return found

    def get_row(df: pd.DataFrame, effect: str):
        hit = df[df["effect"] == effect]
        if hit.empty:
            return None
        return hit.iloc[0]

    def fmt_coef(row):
        if row is None:
            return "NA", False
        coef = float(row["coef"])
        sig = bool(row.get("sig_CI_nonzero", False))
        return f"{coef:.2f}{'*' if sig else ''}", sig

    def diff_sig(row):
        if row is None:
            return False
        return bool(row.get("sig_CI_nonzero", False))

    def make_diagram(cohort: str):
        if not safe_graphviz_import():
            raise RuntimeError("graphviz python package not available.")

        import graphviz

        paths = find_35_1_files(RUN_DIR, cohort)
        if any(paths[k] is None for k in ["male", "female", "diff"]):
            raise FileNotFoundError(f"Could not locate 35.1 csv files for {cohort}. Found: {paths}")

        male_df = pd.read_csv(paths["male"], encoding="utf-8-sig")
        female_df = pd.read_csv(paths["female"], encoding="utf-8-sig")
        diff_df = pd.read_csv(paths["diff"], encoding="utf-8-sig")

        # ind2 summary
        ind2_m = get_row(male_df, "ind2")
        ind2_f = get_row(female_df, "ind2")
        ind2_d = get_row(diff_df, "ind2")

        ind2_m_txt, _ = fmt_coef(ind2_m)
        ind2_f_txt, _ = fmt_coef(ind2_f)
        ind2_diff_sig = diff_sig(ind2_d)

        if ind2_d is not None and ("CI[2.5%]" in ind2_d.index) and ("CI[97.5%]" in ind2_d.index):
            dcoef = float(ind2_d["coef"])
            dlo = float(ind2_d["CI[2.5%]"]); dhi = float(ind2_d["CI[97.5%]"])
            ind2_diff_txt = f"{dcoef:.2f} [{dlo:.2f}, {dhi:.2f}]{' *' if ind2_diff_sig else ''}"
        else:
            ind2_diff_txt = "NA"

        g = graphviz.Digraph(name=f"final_path_{cohort}")
        g.attr(rankdir="LR", bgcolor="white")

        # Use a stable Windows font
        g.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue",
               fontname="Arial", fontsize="14")
        g.attr("edge", fontname="Arial", fontsize="12", color="black")

        # nodes
        g.node("Age", "Age (X)")
        g.node("Exposure", "AI Exposure (M1)")
        g.node("Confidence", "AI Confidence (M2)")
        g.node("Attitude", "AI Attitude (M3)")
        g.node("Accuracy", "Accuracy (Y)")

        with g.subgraph() as s:
            s.attr(rank="same")
            s.node("Exposure")
            s.node("Confidence")
            s.node("Attitude")

        for src, dst, key, human_label in ARCS:
            rm = get_row(male_df, key)
            rf = get_row(female_df, key)
            rd = get_row(diff_df, key)

            male_txt, male_sig = fmt_coef(rm)
            fem_txt, fem_sig = fmt_coef(rf)
            d_sig = diff_sig(rd)

            linestyle = "solid" if (male_sig or fem_sig) else "dashed"

            penwidth = "2.0"
            edgecolor = "black"
            if key in ["a2", "b2"]:
                penwidth = "3.2"
                edgecolor = "#6A1B9A"
                if d_sig:
                    penwidth = "4.6"
                    edgecolor = "#C62828"

            delta = "  Delta*" if d_sig else ""
            edge_label = f"{human_label}\\nB: {male_txt} / {fem_txt}{delta}"

            g.edge(src, dst, label=edge_label, style=linestyle, color=edgecolor, penwidth=penwidth)

        # summary note
        g.attr("node", shape="note", style="filled", fillcolor="#FFF9E5", fontsize="12", fontname="Arial")
        box_txt = (
            "Key moderated mediation summary\\n"
            "ind2 (Age->Confidence->Accuracy)\\n"
            f"male:   {ind2_m_txt}\\n"
            f"female: {ind2_f_txt}\\n"
            f"diff (F-M): {ind2_diff_txt}\\n"
            "(* = 95% CI excludes 0)"
        )
        g.node("summary", box_txt)

        out_base = OUT_DIR / f"36-1_final_path_diagram_{cohort}"

        # Render PNG
        g.format = "png"
        png_path = g.render(str(out_base), cleanup=True)

        # Render SVG
        g.format = "svg"
        svg_path = g.render(str(out_base), cleanup=True)

        print(f"✅ [{cohort}] using files:")
        print(f"   - male:   {paths['male']}")
        print(f"   - female: {paths['female']}")
        print(f"   - diff:   {paths['diff']}")
        print(f"✅ saved: {png_path}")
        print(f"✅ saved: {svg_path}")

        return {
            "cohort": cohort,
            "male_csv": str(paths["male"]),
            "female_csv": str(paths["female"]),
            "diff_csv": str(paths["diff"]),
            "png": str(Path(png_path)),
            "svg": str(Path(svg_path)),
        }

    print("==============================================================================")
    print("(36) Final Path Diagram from 35.1 outputs (AUTO-DISCOVERY) - FIXED")
    print("==============================================================================")

    results = []
    for c in COHORTS:
        try:
            results.append(make_diagram(c))
        except Exception as e:
            print(f"❌ [{c}] failed:", e)

    meta = {
        "section": "36_final_path_diagram_graphviz_auto_discovery_fixed",
        "created_at": datetime.now().isoformat(),
        "run_dir": str(RUN_DIR),
        "out_dir": str(OUT_DIR),
        "results": results,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ meta saved: {OUT_DIR / 'meta.json'}")


def _run_cell_163():
    # ==============================================================================
    # (35.1β) Parallel Mediation (Exposure, Confidence, Attitude) - Standardized Betas
    # ------------------------------------------------------------------------------
    # Standardize within each cohort (mobile/web) using ref = male+female only.
    # Then run:
    #   - all (male+female)
    #   - male only
    #   - female only
    # and bootstrap female-minus-male difference (true stratified bootstrap).
    # ==============================================================================

    import os
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    import statsmodels.api as sm

    # -----------------------------
    # Config
    # -----------------------------
    RUN_DIR = config.OUTPUTS_DIR / "run_20260119_192624"
    SECTION_DIR = RUN_DIR / "35_1b_beta_text"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    N_BOOT = 5000
    SEED = 42


    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_accuracy_col(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("No overallAccuracy column found (overallAccuracy_y/overallAccuracy/overallAccuracy_x).")

    def to_accuracy_pct(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        finite = s.dropna()
        if finite.empty:
            return s
        mx = float(finite.max())
        if mx <= 1.5:
            return s * 100.0
        return s

    def normalize_sex(df: pd.DataFrame, out_col="sex") -> pd.DataFrame:
        out = df.copy()
        src = None
        for c in ["sex", "gender"]:
            if c in out.columns:
                src = c
                break
        if src is None:
            out[out_col] = np.nan
            return out

        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
        s = s.replace({
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer not to": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    def ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        def norm(s): return s.astype(str).str.lower().str.strip()

        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {
            "very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5
        }
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        if "exposure_score" not in out.columns:
            out["exposure_score"] = norm(out["aiExposureFrequency"]).map(exposure_map) if "aiExposureFrequency" in out.columns else np.nan
        if "confidence_score" not in out.columns:
            out["confidence_score"] = norm(out["aiConfidence"]).map(confidence_map) if "aiConfidence" in out.columns else np.nan
        if "attitude_score" not in out.columns:
            out["attitude_score"] = norm(out["aiAttitude"]).map(attitude_map) if "aiAttitude" in out.columns else np.nan
        return out

    def zscore_with_reference(df: pd.DataFrame, cols: list[str], ref_df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            xref = pd.to_numeric(ref_df[c], errors="coerce")
            mu = xref.mean()
            sd = xref.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                out[c + "_z"] = np.nan
            else:
                out[c + "_z"] = (pd.to_numeric(out[c], errors="coerce") - mu) / sd
        return out

    def ols_fit(y: pd.Series, X_df: pd.DataFrame):
        X = sm.add_constant(X_df, has_constant="add")
        return sm.OLS(y, X, missing="drop").fit()

    def point_only_effects(dd: pd.DataFrame) -> dict:
        need = ["age_z","exposure_z","confidence_z","attitude_z","acc_z"]
        s = dd.dropna(subset=need).copy()
        if len(s) < 30:
            raise ValueError(f"N too small after complete-case filtering: N={len(s)}")

        m1 = ols_fit(s["exposure_z"], s[["age_z"]])
        m2 = ols_fit(s["confidence_z"], s[["age_z"]])
        m3 = ols_fit(s["attitude_z"], s[["age_z"]])
        y  = ols_fit(s["acc_z"], s[["age_z","exposure_z","confidence_z","attitude_z"]])

        a1 = m1.params.get("age_z", np.nan)
        a2 = m2.params.get("age_z", np.nan)
        a3 = m3.params.get("age_z", np.nan)
        b1 = y.params.get("exposure_z", np.nan)
        b2 = y.params.get("confidence_z", np.nan)
        b3 = y.params.get("attitude_z", np.nan)
        cprime = y.params.get("age_z", np.nan)

        ind1 = a1*b1
        ind2 = a2*b2
        ind3 = a3*b3
        ind_total = ind1 + ind2 + ind3

        return dict(
            a1=a1,a2=a2,a3=a3,
            b1=b1,b2=b2,b3=b3,
            cprime=cprime,
            ind1=ind1,ind2=ind2,ind3=ind3,
            ind_total=ind_total,
            N=len(s),
            _ymod=y
        )

    def bootstrap_with_ci(dd: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          effects_df: coef + CI for effects
          ycoef_df:   coefficient table for final Y model (β scale)
        """
        pt = point_only_effects(dd)
        ymod = pt.pop("_ymod")
        n = pt["N"]

        keys = ["a1","a2","a3","b1","b2","b3","cprime","ind1","ind2","ind3","ind_total"]
        boot = {k: [] for k in keys}

        s = dd.dropna(subset=["age_z","exposure_z","confidence_z","attitude_z","acc_z"]).copy()
        idx = np.arange(len(s))

        for _ in range(N_BOOT):
            samp = s.iloc[rng.choice(idx, size=len(s), replace=True)]
            try:
                bpt = point_only_effects(samp)
            except Exception:
                continue
            for k in keys:
                boot[k].append(bpt[k])

        def ci(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 200:
                return (np.nan, np.nan)
            return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

        rows = []
        for k in keys:
            lo, hi = ci(boot[k])
            sig = (np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi))
            rows.append({
                "effect": k,
                "coef": float(pt[k]),
                "CI[2.5%]": float(lo) if np.isfinite(lo) else np.nan,
                "CI[97.5%]": float(hi) if np.isfinite(hi) else np.nan,
                "sig_CI_nonzero": bool(sig),
                "N_used": int(n),
            })
        effects_df = pd.DataFrame(rows)

        ycoef_df = pd.DataFrame({
            "term": ymod.params.index,
            "coef_beta": ymod.params.values,
            "se": ymod.bse.values,
            "t": ymod.tvalues.values,
            "p": ymod.pvalues.values,
            "N_used": int(n),
        })

        return effects_df, ycoef_df

    def bootstrap_diff_female_minus_male(d_mf: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, int, int]:
        dd = d_mf.dropna(subset=["sex","age_z","exposure_z","confidence_z","attitude_z","acc_z"]).copy()
        dd = dd[dd["sex"].isin(["male","female"])].copy()
        dm = dd[dd["sex"]=="male"].copy()
        df = dd[dd["sex"]=="female"].copy()
        if len(dm) < 30 or len(df) < 30:
            raise ValueError(f"Not enough rows for diff bootstrap (male={len(dm)}, female={len(df)})")

        keys = ["a1","a2","a3","b1","b2","b3","cprime","ind1","ind2","ind3","ind_total"]

        # point diff
        pm = point_only_effects(dm)
        pf = point_only_effects(df)
        point_diff = {k: (pf[k]-pm[k]) for k in keys}

        boot = {k: [] for k in keys}
        idx_m = np.arange(len(dm))
        idx_f = np.arange(len(df))

        for _ in range(N_BOOT):
            sm = dm.iloc[rng.choice(idx_m, size=len(dm), replace=True)]
            sf = df.iloc[rng.choice(idx_f, size=len(df), replace=True)]
            try:
                bm = point_only_effects(sm)
                bf = point_only_effects(sf)
            except Exception:
                continue
            for k in keys:
                boot[k].append(bf[k]-bm[k])

        def ci(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 200:
                return (np.nan, np.nan)
            return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

        rows = []
        for k in keys:
            lo, hi = ci(boot[k])
            sig = (np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi))
            rows.append({
                "effect": k,
                "coef": float(point_diff[k]),
                "CI[2.5%]": float(lo) if np.isfinite(lo) else np.nan,
                "CI[97.5%]": float(hi) if np.isfinite(hi) else np.nan,
                "sig_CI_nonzero": bool(sig),
                "male_N": int(len(dm)),
                "female_N": int(len(df)),
            })
        return pd.DataFrame(rows), int(len(dm)), int(len(df))


    # -----------------------------
    # Run
    # -----------------------------
    print("==============================================================================")
    print("(35.1β) Parallel Mediation with Standardized Coefficients (β) - MOBILE + WEB")
    print("==============================================================================")
    print(f"- N_BOOT={N_BOOT}, SEED={SEED}")
    print(f"📁 output: {SECTION_DIR}\n")

    meta = {
        "section": "35.1β_parallel_mediation_standardized",
        "created_at": datetime.now().isoformat(),
        "N_BOOT": N_BOOT,
        "SEED": SEED,
        "standardization": "z-score on male+female within each device cohort; then stratified",
    }

    rng = np.random.default_rng(SEED)

    for cohort_tag, csv_path in COHORT_FILES.items():
        print(f"\n==================== [{cohort_tag.upper()}] (35.1β) START ====================")
        out_dir = SECTION_DIR / cohort_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {csv_path} rows={len(df):,}")

        df = normalize_sex(df, out_col="sex")
        df = ensure_scores(df)

        acc_col = resolve_accuracy_col(df)
        df["accuracy_pct"] = to_accuracy_pct(df[acc_col])

        # numeric coercion
        for c in ["age","accuracy_pct","exposure_score","confidence_score","attitude_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # ---- Standardize (β): reference = male+female only
        must_have = ["age","accuracy_pct","exposure_score","confidence_score","attitude_score","sex"]
        missing = [c for c in must_have if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns before z-score: {missing}")

        ref = df[df["sex"].isin(["male","female"])].copy()
        z_cols = ["age","exposure_score","confidence_score","attitude_score","accuracy_pct"]
        d_z = zscore_with_reference(df, cols=z_cols, ref_df=ref)

        # map z columns
        mapping = {
            "age_z": "age_z",
            "exposure_z": "exposure_score_z",
            "confidence_z": "confidence_score_z",
            "attitude_z": "attitude_score_z",
            "acc_z": "accuracy_pct_z",
        }
        gen_missing = [src for src in mapping.values() if src not in d_z.columns]
        if gen_missing:
            print("⚠️ z-score columns not found:", gen_missing)
            print("Available columns:", list(d_z.columns))
            raise KeyError(f"Z-score columns missing: {gen_missing}")

        for dst, src in mapping.items():
            d_z[dst] = d_z[src]

        d_mf = d_z[d_z["sex"].isin(["male","female"])].copy()

        # run subsets
        for subset_tag in ["all","male","female"]:
            try:
                if subset_tag == "all":
                    dd = d_mf.copy()
                else:
                    dd = d_mf[d_mf["sex"]==subset_tag].copy()

                eff_df, ycoef_df = bootstrap_with_ci(dd, rng=rng)

                eff_out = out_dir / f"35.1b-1_effects_{subset_tag}_{cohort_tag}.csv"
                coef_out = out_dir / f"35.1b-2_model_coeffs_{subset_tag}_{cohort_tag}.csv"
                eff_df.to_csv(eff_out, index=False, encoding="utf-8-sig")
                ycoef_df.to_csv(coef_out, index=False, encoding="utf-8-sig")

                key = eff_df.set_index("effect").loc[
                    ["ind_total","ind1","ind2","ind3","cprime","a2","b2"],
                    ["coef","CI[2.5%]","CI[97.5%]","sig_CI_nonzero","N_used"]
                ]
                print(f"\n--- [{cohort_tag} | {subset_tag}] β summary (key effects) ---")
                print(key.to_string())

                print(f"✅ saved: {eff_out.name}, {coef_out.name}")

            except Exception as e:
                print(f"❌ [{cohort_tag}|{subset_tag}] failed:", e)

        # diff
        try:
            diff_df, n_m, n_f = bootstrap_diff_female_minus_male(d_mf, rng=rng)
            diff_out = out_dir / f"35.1b-3_effects_diff_female_minus_male_{cohort_tag}.csv"
            diff_df.to_csv(diff_out, index=False, encoding="utf-8-sig")

            diff_key = diff_df.set_index("effect").loc[
                ["ind_total","ind2","a2","b2","cprime"],
                ["coef","CI[2.5%]","CI[97.5%]","sig_CI_nonzero","male_N","female_N"]
            ]
            print(f"\n--- [{cohort_tag}] DIFF (female - male) β (key effects) ---")
            print(diff_key.to_string())
            print(f"✅ saved: {diff_out.name}")

        except Exception as e:
            print(f"❌ [{cohort_tag}|diff] failed:", e)

        print(f"==================== [{cohort_tag.upper()}] (35.1β) END ====================\n")

    meta_out = SECTION_DIR / "meta.json"
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ meta saved: {meta_out}")
    print("\n==================== (35.1β) DONE ====================")


def _run_cell_164():
    # ==============================================================================
    # (36β) Final Path Diagram (Graphviz) from 35.1β outputs
    # ------------------------------------------------------------------------------
    # Reads:
    #   outputs/run_20260119_192624/35_1b_beta_text/<cohort>/
    #     - 35.1b-1_effects_male_<cohort>.csv
    #     - 35.1b-1_effects_female_<cohort>.csv
    #     - 35.1b-3_effects_diff_female_minus_male_<cohort>.csv
    # Builds Graphviz diagram:
    #   - Arrow labels: "male β / female β" + stars
    #   - Solid if each sex effect sig, dashed if n.s.
    #   - If DIFF (female-male) sig => thicker + emphasized
    # Saves:
    #   outputs/run_20260119_192624/36b_final_path_diagram/
    #     - 36b-1_final_path_diagram_<cohort>.png
    #     - 36b-1_final_path_diagram_<cohort>.svg
    # Also prints key coefficients to console + displays image in notebook.
    # ==============================================================================

    import os
    import json
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    from graphviz import Digraph

    # notebook inline display (optional but recommended)
    try:
        from IPython.display import Image, display
        HAVE_IPY = True
    except Exception:
        HAVE_IPY = False

    RUN_DIR = config.OUTPUTS_DIR / "run_20260119_192624"
    IN_BASE = RUN_DIR / "35_1b_beta_text"
    OUT_DIR = RUN_DIR / "36b_final_path_diagram"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    COHORTS = ["mobile", "web"]

    # -----------------------------
    # Utilities
    # -----------------------------
    def load_effects_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")
        req = {"effect","coef","CI[2.5%]","CI[97.5%]","sig_CI_nonzero"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing cols in {path.name}: {missing}")
        df = df.copy()
        df["effect"] = df["effect"].astype(str)
        return df

    def find_file_by_pattern(folder: Path, pattern_contains: str):
        hits = [p for p in folder.glob("*.csv") if pattern_contains in p.name]
        return hits[0] if hits else None

    def get_effect_row(df: pd.DataFrame, effect: str):
        row = df[df["effect"] == effect]
        if row.empty:
            return None
        return row.iloc[0]

    def stars(sig: bool) -> str:
        return "*" if sig else ""

    def fmt_beta(row) -> str:
        if row is None:
            return "NA"
        try:
            return f"{float(row['coef']):.2f}{stars(bool(row['sig_CI_nonzero']))}"
        except Exception:
            return "NA"

    def is_sig(row) -> bool:
        if row is None:
            return False
        try:
            return bool(row["sig_CI_nonzero"])
        except Exception:
            return False

    def diff_sig(diff_df: pd.DataFrame, effect: str) -> bool:
        r = get_effect_row(diff_df, effect)
        return is_sig(r)

    # -----------------------------
    # Diagram builder
    # -----------------------------
    def build_diagram(cohort_tag: str, male_df: pd.DataFrame, female_df: pd.DataFrame, diff_df: pd.DataFrame, out_png: Path, out_svg: Path):
        # Effects we need
        # a paths: Age -> M
        # b paths: M -> Accuracy
        # cprime: Age -> Accuracy (direct)
        want = ["a1","a2","a3","b1","b2","b3","cprime","ind1","ind2","ind3","ind_total"]
        male = {k: get_effect_row(male_df, k) for k in want}
        fem  = {k: get_effect_row(female_df, k) for k in want}

        # console summary
        print(f"\n--- [{cohort_tag.upper()}] β key paths ---")
        for k in ["a1","a2","a3","b1","b2","b3","cprime","ind1","ind2","ind3","ind_total"]:
            m = fmt_beta(male.get(k))
            f = fmt_beta(fem.get(k))
            d_sig = diff_sig(diff_df, k)
            dmark = " (DIFF*) " if d_sig else ""
            print(f"{k:8s}: male {m:>8s} | female {f:>8s}{dmark}")

        # Graphviz
        dot = Digraph(name=f"FinalPath_{cohort_tag}_beta", format="png")
        dot.attr(rankdir="LR", splines="spline", nodesep="0.55", ranksep="0.85")
        dot.attr("graph", bgcolor="white", fontname="Helvetica")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="#E9F3FF", color="#1f1f1f", penwidth="1.6", fontname="Helvetica", fontsize="14")
        dot.attr("edge", color="#1f1f1f", fontname="Helvetica", fontsize="12")

        # Nodes
        dot.node("Age", "Age (X)")
        dot.node("Exp", "AI Exposure (M1)")
        dot.node("Conf", "AI Confidence (M2)")
        dot.node("Att", "AI Attitude (M3)")
        dot.node("Acc", "Accuracy (Y)")

        # Layout: vertical mediators
        with dot.subgraph() as s:
            s.attr(rank="same")
            s.node("Exp")
            s.node("Conf")
            s.node("Att")

        def edge_style(effect_key: str, base_color="#1f1f1f"):
            """
            - solid if BOTH male & female sig? (or you can choose OR)
            - dashed if both n.s.
            - thicker + colored if DIFF sig (female-male)
            """
            m_sig = is_sig(male.get(effect_key))
            f_sig = is_sig(fem.get(effect_key))
            d_sig = diff_sig(diff_df, effect_key)

            # significance rule for line style (너 취향에 맞게 바꿔도 됨)
            line = "solid" if (m_sig or f_sig) else "dashed"

            penwidth = "2.2"
            color = base_color

            if d_sig:
                penwidth = "4.0"
                # diff significant => emphasize
                color = "#C62828"  # deep red

            return line, penwidth, color, d_sig

        def add_edge(src, dst, key, label_prefix="β"):
            mtxt = fmt_beta(male.get(key))
            ftxt = fmt_beta(fem.get(key))
            line, penwidth, color, d_sig = edge_style(key)
            # label: male / female
            lab = f"{label_prefix}: {mtxt} / {ftxt}"
            if d_sig:
                lab += "  (Δ*)"
            dot.edge(src, dst, label=lab, style=line, penwidth=penwidth, color=color)

        # a paths
        add_edge("Age", "Exp",  "a1")
        add_edge("Age", "Conf", "a2")
        add_edge("Age", "Att",  "a3")
        # b paths
        add_edge("Exp",  "Acc", "b1")
        add_edge("Conf", "Acc", "b2")
        add_edge("Att",  "Acc", "b3")
        # direct
        add_edge("Age",  "Acc", "cprime", label_prefix="β (c')")

        # Add a legend note (graph label)
        # keep it simple (Graphviz label supports basic text)
        ind2_diff_sig = diff_sig(diff_df, "ind2")
        extra = f"Key moderated mediation: Δ(ind2) {'sig' if ind2_diff_sig else 'n.s.'}"
        dot.attr(label=f"(36β) Final Path Diagram [{cohort_tag.upper()}]\nStandardized Coefficients (β). Solid=significant (male or female), Dashed=n.s. Thick red=Sex diff significant.\n{extra}",
                 labelloc="t", fontsize="12")

        # Save PNG + SVG
        # graphviz python wrapper writes to .render() with base path (no extension)
        out_base = out_png.with_suffix("")  # remove .png
        dot.render(str(out_base), cleanup=True)  # outputs out_base + .png by default
        # also svg
        dot.format = "svg"
        dot.render(str(out_svg.with_suffix("")), cleanup=True)

        print(f"✅ saved: {out_png}")
        print(f"✅ saved: {out_svg}")

        # notebook display
        if HAVE_IPY and out_png.exists():
            display(Image(filename=str(out_png)))

    # -----------------------------
    # Main
    # -----------------------------
    print("==============================================================================")
    print("(36β) Final Path Diagram from 35.1β outputs (AUTO-DISCOVERY, Graphviz)")
    print("==============================================================================")

    meta = {
        "section": "36β_final_path_diagram_graphviz",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(IN_BASE),
        "output_dir": str(OUT_DIR),
    }

    for cohort_tag in COHORTS:
        try:
            cohort_dir = IN_BASE / cohort_tag
            if not cohort_dir.exists():
                raise FileNotFoundError(f"Missing cohort folder: {cohort_dir}")

            # auto-discover files
            male_path   = find_file_by_pattern(cohort_dir, f"effects_male_{cohort_tag}")
            female_path = find_file_by_pattern(cohort_dir, f"effects_female_{cohort_tag}")
            diff_path   = find_file_by_pattern(cohort_dir, f"diff_female_minus_male_{cohort_tag}")

            if not male_path or not female_path or not diff_path:
                raise FileNotFoundError(
                    f"Missing required csvs in {cohort_dir}.\n"
                    f"Found male={male_path}, female={female_path}, diff={diff_path}"
                )

            print(f"\n✅ [{cohort_tag}] using files:")
            print(f"   - male:   {male_path}")
            print(f"   - female: {female_path}")
            print(f"   - diff:   {diff_path}")

            male_df = load_effects_csv(male_path)
            fem_df  = load_effects_csv(female_path)
            diff_df = load_effects_csv(diff_path)

            out_png = OUT_DIR / f"36b-1_final_path_diagram_{cohort_tag}.png"
            out_svg = OUT_DIR / f"36b-1_final_path_diagram_{cohort_tag}.svg"

            build_diagram(cohort_tag, male_df, fem_df, diff_df, out_png, out_svg)

            meta[cohort_tag] = {
                "male_csv": str(male_path),
                "female_csv": str(female_path),
                "diff_csv": str(diff_path),
                "png": str(out_png),
                "svg": str(out_svg),
            }

        except Exception as e:
            print(f"❌ [{cohort_tag}] failed: {e}")
            meta[cohort_tag] = {"error": str(e)}

    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ meta saved: {meta_path}")
    print("\n==================== (36β) DONE ====================")


def main():
    _run_cell_119()
    _run_cell_122()
    _run_cell_126()
    _run_cell_130()
    _run_cell_134()
    _run_cell_141()
    _run_cell_144()
    _run_cell_147()
    _run_cell_150()
    _run_cell_153()
    _run_cell_154()
    _run_cell_157()
    _run_cell_160()
    _run_cell_163()
    _run_cell_164()


if __name__ == "__main__":
    main()
