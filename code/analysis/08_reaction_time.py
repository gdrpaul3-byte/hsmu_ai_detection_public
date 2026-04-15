"""Analysis sections for reaction time, verification cost, and trial-level mixed models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_071():
    # ==============================================================================
    # (18 v2.2) AI Attitude Interaction: Age Group × Sex (from gender/sex) (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Fixes:
    # - Robust sex column detection: gender/sex/Gender/Sex ...
    # - Robust sex value normalization: male/female/M/F/0/1 etc.
    # - Seaborn pointplot compatibility:
    #     - new seaborn: errorbar="se"
    #     - old seaborn: fallback to ci=68 (≈ 1 SE)
    # - Print ANOVA stats to console (df, F, p, eta^2)
    # - Save plot + tables + report per cohort folder
    # - Silence pandas observed warning by passing observed=False explicitly
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
    OUTPUT_DIR = "outputs_18_attitude_v2"
    ALPHA = 0.05

    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    ATTITUDE_MAP = {
        "very-negative": -2,
        "negative": -1,
        "neutral": 0,
        "positive": 1,
        "very-positive": 2,
    }

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fmt_p(p: float) -> str:
        if not np.isfinite(p):
            return "nan"
        if p < 1e-4:
            return f"{p:.2e}"
        return f"{p:.4f}"

    def compute_eta_squared(anova_tbl: pd.DataFrame) -> pd.Series:
        ss_total = anova_tbl["sum_sq"].sum()
        if ss_total <= 0:
            return pd.Series([np.nan] * len(anova_tbl), index=anova_tbl.index, name="eta_sq")
        return (anova_tbl["sum_sq"] / ss_total).rename("eta_sq")

    def print_anova_console(cohort_tag: str, anova_tbl: pd.DataFrame):
        disp = anova_tbl.copy()
        disp["eta_sq"] = compute_eta_squared(disp)
        cols = [c for c in ["df", "sum_sq", "F", "PR(>F)", "eta_sq"] if c in disp.columns]
        disp = disp[cols]

        print("\n[ANOVA Table: attitude_score ~ age_group * sex] "
              f"[{cohort_tag.upper()}]\n")
        print(disp.round(6).to_string())
        print("\n[Key effects summary]")

        for term in ["C(age_group)", "C(sex)", "C(age_group):C(sex)"]:
            if term not in anova_tbl.index:
                continue
            row = anova_tbl.loc[term]
            p = float(row["PR(>F)"]) if "PR(>F)" in anova_tbl.columns else np.nan
            F = float(row["F"]) if "F" in anova_tbl.columns else np.nan
            df1 = float(row["df"]) if "df" in anova_tbl.columns else np.nan
            eta = float(compute_eta_squared(anova_tbl).loc[term])
            sig = "✅" if (np.isfinite(p) and p < ALPHA) else "❌"
            print(f"- {term}: F={F:.3f}, df={df1:.0f}, p={fmt_p(p)}, eta²={eta:.4f} {sig}")

    def pick_sex_column(df: pd.DataFrame):
        candidates = ["sex", "Sex", "gender", "Gender"]
        for c in candidates:
            if c in df.columns:
                return c
        # fallback: any column that contains sex/gender
        for c in df.columns:
            lc = str(c).lower()
            if "sex" in lc or "gender" in lc:
                return c
        return None

    def normalize_sex(series: pd.Series) -> pd.Series:
        """
        Normalize to 'male'/'female' (string).
        Handles: Male/Female, M/F, 0/1, man/woman, etc.
        Unknown -> NaN
        """
        s = series.astype(str).str.strip().str.lower()

        mapping = {
            "male": "male", "m": "male", "man": "male", "남": "male",
            "0": "male", "1": "female",  # sometimes coded
            "female": "female", "f": "female", "woman": "female", "여": "female",
        }

        out = s.map(mapping)
        # try more robust pattern-based
        out = out.where(out.notna(), np.where(s.str.contains(r"^m(ale)?$"), "male", np.nan))
        out = pd.Series(out, index=series.index)
        out = out.where(out.notna(), np.where(s.str.contains(r"^f(emale)?$"), "female", np.nan))
        out = pd.Series(out, index=series.index)

        return out

    def seaborn_pointplot_se(df: pd.DataFrame, ax):
        """
        Compatible pointplot:
        - new seaborn: errorbar="se"
        - old seaborn: ci=68 (≈ 1 SE)
        """
        try:
            sns.pointplot(
                data=df,
                x="age_group", y="attitude_score", hue="sex",
                order=AGE_LABELS, hue_order=["male", "female"],
                errorbar="se", markers=["o", "s"], ax=ax
            )
            return "errorbar=se"
        except TypeError:
            # old seaborn fallback
            sns.pointplot(
                data=df,
                x="age_group", y="attitude_score", hue="sex",
                order=AGE_LABELS, hue_order=["male", "female"],
                ci=68, markers=["o", "s"], ax=ax
            )
            return "ci=68 (≈1SE)"

    def run_cohort(cohort_tag: str, path: str):
        print(f"\n==================== [{cohort_tag.upper()}] (18 v2.2) START ====================")

        out_dir = os.path.join(OUTPUT_DIR, cohort_tag)
        ensure_dir(out_dir)

        # Load
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"✅ Loaded: {path} [{cohort_tag}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {path}")
            return

        df = df.copy()

        sex_col = pick_sex_column(df)
        if sex_col is None:
            print(f"⚠️ [{cohort_tag}] Missing sex/gender column. Skipping.")
            return
        if "age" not in df.columns or "aiAttitude" not in df.columns:
            print(f"⚠️ [{cohort_tag}] Missing required columns among: age, aiAttitude. Skipping.")
            return

        # normalize sex
        df["sex"] = normalize_sex(df[sex_col])
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        # attitude score
        df["attitude_score"] = df["aiAttitude"].astype(str).str.lower().str.strip().map(ATTITUDE_MAP)

        # age group + dropna
        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        df = df.dropna(subset=["age_group", "attitude_score", "sex"]).copy()

        # keep only male/female
        df = df[df["sex"].isin(["male", "female"])].copy()

        print(f"✅ [{cohort_tag}] Sex source column used: {sex_col}")
        print(f"✅ [{cohort_tag}] N after cleaning: {len(df)}")

        # Descriptives (mean/count/std/se)
        desc = (
            df.groupby(["age_group", "sex"], observed=False)["attitude_score"]
              .agg(["mean", "count", "std"])
              .reset_index()
        )
        desc["se"] = desc["std"] / np.sqrt(desc["count"].clip(lower=1))
        desc_path = os.path.join(out_dir, f"18v2-0_attitude_descriptives_{cohort_tag}.csv")
        desc.to_csv(desc_path, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved descriptives: {desc_path}")

        # Plot
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        used_mode = seaborn_pointplot_se(df, ax)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1.5)

        ax.set_title(f"(18v2-1) Mean AI Attitude by Age Group and Sex [{cohort_tag.upper()}]",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Mean attitude score (-2 to +2)")
        ax.set_ylim(-2.05, 2.05)
        ax.legend(title="Sex", loc="best")
        ax.text(0.99, 0.01, f"Plot CI mode: {used_mode}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, alpha=0.8)

        plot_base = os.path.join(out_dir, f"18v2-1_attitude_age_sex_pointplot_{cohort_tag}")
        fig.savefig(plot_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(plot_base + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ [{cohort_tag}] Saved plot: {plot_base}.png/.svg")

        # Two-way ANOVA
        model = ols("attitude_score ~ C(age_group) + C(sex) + C(age_group):C(sex)", data=df).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)

        anova_out = os.path.join(out_dir, f"18v2-2_anova_table_{cohort_tag}.csv")
        anova_tbl.to_csv(anova_out, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved ANOVA table: {anova_out}")

        # Print to console
        print_anova_console(cohort_tag, anova_tbl)

        # Report (with eta^2)
        anova_with_eta = anova_tbl.copy()
        anova_with_eta["eta_sq"] = compute_eta_squared(anova_tbl)

        p_age = float(anova_tbl.loc["C(age_group)", "PR(>F)"]) if "C(age_group)" in anova_tbl.index else np.nan
        p_sex = float(anova_tbl.loc["C(sex)", "PR(>F)"]) if "C(sex)" in anova_tbl.index else np.nan
        p_inter = float(anova_tbl.loc["C(age_group):C(sex)", "PR(>F)"]) if "C(age_group):C(sex)" in anova_tbl.index else np.nan

        summary_lines = []
        summary_lines.append("Summary")
        summary_lines.append("-----------------------------------------")
        summary_lines.append(f"Age group main effect:    p={fmt_p(p_age)}  -> {'SIGNIFICANT' if (np.isfinite(p_age) and p_age < ALPHA) else 'n.s.'}")
        summary_lines.append(f"Sex main effect:          p={fmt_p(p_sex)}  -> {'SIGNIFICANT' if (np.isfinite(p_sex) and p_sex < ALPHA) else 'n.s.'}")
        summary_lines.append(f"Interaction (Age×Sex):    p={fmt_p(p_inter)} -> {'SIGNIFICANT' if (np.isfinite(p_inter) and p_inter < ALPHA) else 'n.s.'}")
        summary_lines.append("")

        report = []
        report.append(f"(18 v2.2) AI Attitude Interaction: Age Group × Sex [{cohort_tag.upper()}]")
        report.append("="*80)
        report.append(f"N used: {len(df)}")
        report.append(f"Sex source column: {sex_col}")
        report.append("")
        report.append("\n".join(summary_lines))
        report.append("ANOVA table (Type II):")
        report.append("-----------------------------------------")
        report.append(anova_with_eta.round(6).to_string())
        report.append("")
        report.append("Notes:")
        report.append("- eta_sq is eta-squared (SS_effect / SS_total).")
        report.append(f"- alpha = {ALPHA}")
        report.append("- If pointplot used ci=68, it approximates +/- 1 SE under normality.")

        report_path = os.path.join(out_dir, f"18v2-2_attitude_anova_report_{cohort_tag}.txt")
        save_text(report_path, "\n".join(report))
        print(f"✅ [{cohort_tag}] Saved report: {report_path}")

        # OLS summary
        ols_path = os.path.join(out_dir, f"18v2-2_ols_summary_{cohort_tag}.txt")
        save_text(ols_path, str(model.summary()))
        print(f"✅ [{cohort_tag}] Saved OLS summary: {ols_path}")

        print(f"==================== [{cohort_tag.upper()}] (18 v2.2) END ====================")


    # -----------------------------
    # Main
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(18 v2.2) AI Attitude Interaction: Age Group × Sex (MOBILE + WEB) [ENGLISH]")
        print("==============================================================================\n")

        ensure_dir(OUTPUT_DIR)

        for cohort_tag, path in COHORT_FILES.items():
            run_cohort(cohort_tag, path)

        print("\n==================== (18) DONE ====================")


def _run_cell_074():
    # ==============================================================================
    # (19 v2.1) Prior experience & personality (MBTI) effects on Accuracy
    # - MOBILE + WEB
    # - Robust fixes:
    #   * auto-pick accuracy column (overallAccuracy / overallAccuracy_y / overallAccuracy_x)
    #   * convert proportion(0-1) -> percent(0-100) automatically
    #   * fix MBTI regex: [EI][SN][TF][JP]
    # - Print key statistics to console + save CSV/TXT/PNG/SVG
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_19_experience_mbti_v2"
    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }
    ALPHA = 0.05

    sns.set_theme(style="whitegrid")

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def safe_slug(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = s.replace(" ", "_").replace("/", "_")
        return s

    def pick_accuracy_column(df: pd.DataFrame) -> str:
        # Prefer already-cleaned column if present
        candidates = ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x", "accuracy", "Accuracy"]
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError("No accuracy column found among: " + ", ".join(candidates))

    def ensure_accuracy_percent(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        # If it looks like a proportion (0~1), convert to %
        # Use robust check: ignore NaN, look at max
        mx = s.dropna().max() if s.notna().any() else np.nan
        if np.isfinite(mx) and mx <= 1.5:  # allow tiny overs
            return s * 100.0
        return s

    def _welch_df(x, y):
        nx, ny = len(x), len(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        num = (vx/nx + vy/ny)**2
        den = (vx*vx)/((nx*nx)*(nx-1)) + (vy*vy)/((ny*ny)*(ny-1))
        return num/den if den > 0 else np.nan

    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
        sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)) if (nx+ny-2) > 0 else np.nan
        return (np.mean(x) - np.mean(y)) / sp if sp and np.isfinite(sp) and sp > 0 else np.nan

    def mean_diff_ci_welch(x, y, alpha=0.05):
        nx, ny = len(x), len(y)
        mx, my = np.mean(x), np.mean(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        se = np.sqrt(vx/nx + vy/ny)
        df = _welch_df(x, y)
        if not np.isfinite(df) or not np.isfinite(se) or se <= 0:
            return (np.nan, np.nan, np.nan)
        tcrit = stats.t.ppf(1 - alpha/2, df)
        diff = mx - my
        return (diff, diff - tcrit*se, diff + tcrit*se)

    def describe_series(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return {
            "n": int(len(s)),
            "mean": float(s.mean()) if len(s) else np.nan,
            "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "se": float(s.sem(ddof=1)) if len(s) > 1 else np.nan,
        }

    def barplot_means_with_se(ax, labels, means, ses, title, ylab="Accuracy (%)"):
        ax.bar(labels, means, yerr=ses, capsize=6)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 100)

    # -----------------------------
    # Main per cohort
    # -----------------------------
    print("==============================================================================")
    print("(19 v2.1) Prior experience & MBTI effects on Accuracy (MOBILE + WEB) [ENGLISH]")
    print("==============================================================================\n")

    ensure_dir(OUTPUT_DIR)

    for cohort, fp in COHORT_FILES.items():
        print(f"\n==================== [{cohort.upper()}] (19 v2.1) START ====================")
        out_dir = os.path.join(OUTPUT_DIR, cohort)
        ensure_dir(out_dir)

        try:
            df = pd.read_csv(fp, encoding="utf-8-sig")
            print(f"✅ Loaded: {fp} [{cohort}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {fp}")
            continue

        # -----------------------------
        # Accuracy column robust handling
        # -----------------------------
        try:
            acc_col = pick_accuracy_column(df)
        except KeyError as e:
            print(f"❌ {e}")
            continue

        df["overallAccuracy"] = ensure_accuracy_percent(df[acc_col])
        print(f"✅ [{cohort}] Accuracy column used: {acc_col} -> stored as overallAccuracy (%)")

        # keep within 0-100 for plotting sanity (optional clip)
        df["overallAccuracy"] = df["overallAccuracy"].clip(lower=0, upper=100)

        # ============================================================
        # (19v2-1) Midjourney usage vs Accuracy (Welch t-test)
        # ============================================================
        print("\n--- (19v2-1) Midjourney usage vs Accuracy ---")

        if "usedAiTools" not in df.columns:
            print("⚠️ usedAiTools column missing. Skipping (19v2-1).")
        else:
            d1 = df[["usedAiTools", "overallAccuracy"]].copy()
            d1 = d1.dropna(subset=["overallAccuracy"])
            d1["usedAiTools"] = d1["usedAiTools"].astype(str)

            # Contains "midjourney" anywhere (case-insensitive)
            d1["used_midjourney"] = d1["usedAiTools"].str.contains("midjourney", case=False, na=False)

            mj = d1.loc[d1["used_midjourney"], "overallAccuracy"].dropna()
            non = d1.loc[~d1["used_midjourney"], "overallAccuracy"].dropna()

            if len(mj) < 10 or len(non) < 10:
                print(f"⚠️ Not enough data for t-test (MJ N={len(mj)}, non-MJ N={len(non)}).")
            else:
                t_stat, p_val = stats.ttest_ind(mj, non, equal_var=False, nan_policy="omit")
                df_welch = _welch_df(mj.values, non.values)
                d = cohen_d(mj.values, non.values)
                diff, ci_lo, ci_hi = mean_diff_ci_welch(mj.values, non.values, alpha=0.05)

                desc_mj = describe_series(mj)
                desc_non = describe_series(non)

                print(f"N (MJ)={desc_mj['n']} | mean={desc_mj['mean']:.3f} | sd={desc_mj['sd']:.3f}")
                print(f"N (non)={desc_non['n']} | mean={desc_non['mean']:.3f} | sd={desc_non['sd']:.3f}")
                print(f"Welch t-test: t={t_stat:.4f}, df≈{df_welch:.2f}, p={p_val:.6g}, Cohen's d={d:.3f}")
                print(f"Mean diff (MJ - non) = {diff:.3f}  | 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

                mj_desc_df = pd.DataFrame([
                    {"group": "Midjourney", **desc_mj},
                    {"group": "Non-Midjourney", **desc_non},
                ])
                mj_desc_path = os.path.join(out_dir, f"19v2-1_midjourney_descriptives_{cohort}.csv")
                mj_desc_df.to_csv(mj_desc_path, index=False, encoding="utf-8-sig")
                print(f"✅ Saved descriptives: {mj_desc_path}")

                fig, ax = plt.subplots(figsize=(8, 6))
                labels = ["Midjourney", "Non-Midjourney"]
                means = [desc_mj["mean"], desc_non["mean"]]
                ses = [desc_mj["se"], desc_non["se"]]
                barplot_means_with_se(ax, labels, means, ses, f"(19v2-1) Accuracy by Midjourney usage [{cohort}]")
                base = os.path.join(out_dir, f"19v2-1_accuracy_by_midjourney_{cohort}")
                fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
                fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                print(f"✅ Saved plot: {base}.png/.svg")

                report = (
                    f"(19v2-1) Midjourney usage vs Accuracy [{cohort}]\n"
                    f"--------------------------------------------------\n"
                    f"Accuracy column: {acc_col} (converted to % if needed)\n\n"
                    f"N (MJ)={desc_mj['n']}, mean={desc_mj['mean']:.4f}, sd={desc_mj['sd']:.4f}\n"
                    f"N (non)={desc_non['n']}, mean={desc_non['mean']:.4f}, sd={desc_non['sd']:.4f}\n\n"
                    f"Welch t-test: t={t_stat:.6f}, df≈{df_welch:.3f}, p={p_val:.8g}\n"
                    f"Cohen's d={d:.4f}\n"
                    f"Mean diff (MJ - non)={diff:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]\n"
                )
                report_path = os.path.join(out_dir, f"19v2-1_midjourney_ttest_report_{cohort}.txt")
                save_text(report_path, report)
                print(f"✅ Saved report: {report_path}")

        # ============================================================
        # (19v2-2) MBTI 16-type descriptives + ranking plot
        # ============================================================
        print("\n--- (19v2-2) MBTI 16-type descriptives ---")

        if "mbti" not in df.columns:
            print("⚠️ mbti column missing. Skipping (19v2-2)/(19v2-3).")
            print(f"==================== [{cohort.upper()}] (19 v2.1) END ====================\n")
            continue

        d2 = df[["mbti", "overallAccuracy"]].copy()
        d2 = d2.dropna(subset=["overallAccuracy"])
        d2["mbti"] = d2["mbti"].astype(str).str.upper().str.strip()

        # remove non-answers
        d2 = d2[~d2["mbti"].isin(["IDK", "PREFER-NOT-TO-SAY", "PREFER_NOT_TO_SAY", "NONE", "NAN", ""])]
        # ✅ correct MBTI regex (4 letters)
        d2 = d2[d2["mbti"].str.fullmatch(r"[EI][SN][TF][JP]", na=False)]

        if len(d2) < 20:
            print(f"⚠️ Not enough MBTI rows after filtering (N={len(d2)}).")
        else:
            mbti_stats = (
                d2.groupby("mbti")["overallAccuracy"]
                  .agg(n="count", mean="mean", sd=lambda x: x.std(ddof=1))
                  .sort_values("mean", ascending=False)
            )
            mbti_path = os.path.join(out_dir, f"19v2-2_mbti_16type_descriptives_{cohort}.csv")
            mbti_stats.to_csv(mbti_path, encoding="utf-8-sig")
            print(f"✅ Saved MBTI descriptives: {mbti_path}")

            print("\n[Top 8 MBTI types by mean accuracy]")
            print(mbti_stats.head(8).round(3).to_string())
            print("\n[Bottom 8 MBTI types by mean accuracy]")
            print(mbti_stats.tail(8).round(3).to_string())

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(mbti_stats.index[::-1], mbti_stats["mean"][::-1])
            ax.set_xlabel("Mean Accuracy (%)")
            ax.set_ylabel("MBTI type")
            ax.set_xlim(0, 100)
            ax.set_title(f"(19v2-2) Accuracy by MBTI type (mean) [{cohort}]", fontsize=14, fontweight="bold", pad=12)
            base = os.path.join(out_dir, f"19v2-2_accuracy_by_mbti_rank_{cohort}")
            fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
            fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            print(f"✅ Saved plot: {base}.png/.svg")

            eligible = mbti_stats[mbti_stats["n"] >= 10].index.tolist()
            if len(eligible) >= 3:
                groups = [d2.loc[d2["mbti"] == t, "overallAccuracy"].values for t in eligible]
                F, p = stats.f_oneway(*groups)
                print(f"\n[Exploratory ANOVA across MBTI types (n>=10)]")
                print(f"F={F:.4f}, p={p:.6g}, k={len(eligible)} groups")

                anova_report = (
                    f"(19v2-2) Exploratory one-way ANOVA across MBTI types (n>=10) [{cohort}]\n"
                    f"Accuracy column: {acc_col} (converted to % if needed)\n"
                    f"Eligible groups (k={len(eligible)}): {', '.join(eligible)}\n"
                    f"F={F:.6f}, p={p:.8g}\n"
                    f"Note: exploratory; MBTI groups can be imbalanced.\n"
                )
                save_text(os.path.join(out_dir, f"19v2-2_mbti_anova_exploratory_{cohort}.txt"), anova_report)

        # ============================================================
        # (19v2-3) MBTI N vs S (2nd letter) vs Accuracy (Welch t-test)
        # ============================================================
        print("\n--- (19v2-3) MBTI N vs S vs Accuracy ---")

        if len(d2) < 20:
            print("⚠️ Skipping (19v2-3) due to insufficient MBTI rows.")
        else:
            d3 = d2.copy()
            d3["NS"] = d3["mbti"].str[1]  # 2nd letter: S/N
            d3 = d3[d3["NS"].isin(["N", "S"])]

            n_group = d3.loc[d3["NS"] == "N", "overallAccuracy"].dropna()
            s_group = d3.loc[d3["NS"] == "S", "overallAccuracy"].dropna()

            if len(n_group) < 10 or len(s_group) < 10:
                print(f"⚠️ Not enough data for N vs S t-test (N={len(n_group)}, S={len(s_group)}).")
            else:
                t_stat, p_val = stats.ttest_ind(n_group, s_group, equal_var=False, nan_policy="omit")
                df_welch = _welch_df(n_group.values, s_group.values)
                d = cohen_d(n_group.values, s_group.values)
                diff, ci_lo, ci_hi = mean_diff_ci_welch(n_group.values, s_group.values, alpha=0.05)

                desc_n = describe_series(n_group)
                desc_s = describe_series(s_group)

                print(f"N-group: N={desc_n['n']} | mean={desc_n['mean']:.3f} | sd={desc_n['sd']:.3f}")
                print(f"S-group: N={desc_s['n']} | mean={desc_s['mean']:.3f} | sd={desc_s['sd']:.3f}")
                print(f"Welch t-test: t={t_stat:.4f}, df≈{df_welch:.2f}, p={p_val:.6g}, Cohen's d={d:.3f}")
                print(f"Mean diff (N - S) = {diff:.3f}  | 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

                fig, ax = plt.subplots(figsize=(7, 6))
                means = [desc_n["mean"], desc_s["mean"]]
                ses = [desc_n["se"], desc_s["se"]]
                barplot_means_with_se(ax, ["N", "S"], means, ses, f"(19v2-3) Accuracy: MBTI N vs S [{cohort}]")
                base = os.path.join(out_dir, f"19v2-3_accuracy_by_mbti_NS_{cohort}")
                fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
                fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                print(f"✅ Saved plot: {base}.png/.svg")

                report = (
                    f"(19v2-3) MBTI N vs S vs Accuracy [{cohort}]\n"
                    f"--------------------------------------------------\n"
                    f"Accuracy column: {acc_col} (converted to % if needed)\n\n"
                    f"N-group: N={desc_n['n']}, mean={desc_n['mean']:.4f}, sd={desc_n['sd']:.4f}\n"
                    f"S-group: N={desc_s['n']}, mean={desc_s['mean']:.4f}, sd={desc_s['sd']:.4f}\n\n"
                    f"Welch t-test: t={t_stat:.6f}, df≈{df_welch:.3f}, p={p_val:.8g}\n"
                    f"Cohen's d={d:.4f}\n"
                    f"Mean diff (N - S)={diff:.4f} | 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]\n"
                )
                report_path = os.path.join(out_dir, f"19v2-3_mbti_NS_ttest_report_{cohort}.txt")
                save_text(report_path, report)
                print(f"✅ Saved report: {report_path}")

        print(f"==================== [{cohort.upper()}] (19 v2.1) END ====================\n")

    print("\n==================== (19 v2.1) DONE ====================")


def _run_cell_077():
    # ==============================================================================
    # (19A v2.1) MBTI Disclosure/Distribution + Accuracy Differences (MOBILE + WEB)
    # ------------------------------------------------------------------------------
    # Fixes:
    #  - Robust accuracy column detection: overallAccuracy or overallAccuracy_y
    #  - If accuracy is 0~1 proportion -> convert to %
    #  - Safer MBTI disclosure pct (avoid divide-by-zero)
    #  - Save all tables/plots + KW test + epsilon^2
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_19A_mbti_v2"
    MIN_N_PER_TYPE = 8    # types with N < this are excluded from KW test

    # Preferred column names (we'll auto-detect if missing)
    ACC_COL = "overallAccuracy"
    ACC_FALLBACK_COLS = ["overallAccuracy_y", "overallAccuracy", "accuracy", "Accuracy"]

    MBTI_COL = "mbti"

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def find_accuracy_col(df: pd.DataFrame) -> str | None:
        for c in [ACC_COL] + ACC_FALLBACK_COLS:
            if c in df.columns:
                return c
        return None

    def normalize_accuracy_to_percent(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Make df[ACC_COL] as numeric percent (0~100).
        If values look like proportion (<= 1.2), multiply by 100.
        """
        out = df.copy()
        s = pd.to_numeric(out[col], errors="coerce")

        # detect proportion: max <= 1.2 (a little buffer)
        smax = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else np.nan
        if np.isfinite(smax) and smax <= 1.2:
            s = s * 100.0

        out[ACC_COL] = s
        return out

    def is_valid_16type(x: str) -> bool:
        if x is None:
            return False
        x = str(x).strip().upper()
        return bool(re.fullmatch(r"[EI][SN][TF][JP]", x))

    def normalize_mbti(x: str) -> str:
        if x is None:
            return ""
        return str(x).strip().upper()

    def classify_mbti_status(x: str) -> str:
        x = normalize_mbti(x)

        if x in ["", "NAN", "NONE"]:
            return "Missing/Other"
        if x == "IDK":
            return "IDK"
        if x in ["PREFER-NOT-TO-SAY", "PREFER_NOT_TO_SAY", "PREFER NOT TO SAY"]:
            return "Prefer-not-to-say"
        if is_valid_16type(x):
            return "Valid MBTI (16-type)"
        return "Missing/Other"

    def eps_squared_kw(H: float, k: int, n: int) -> float:
        """
        epsilon^2 for Kruskal–Wallis:
          eps^2 = (H - k + 1) / (n - k)
        clipped at >= 0 for interpretability.
        """
        if (n - k) <= 0:
            return np.nan
        e = (H - k + 1) / (n - k)
        return float(max(0.0, e))

    # -----------------------------
    # Plot / Analysis blocks
    # -----------------------------
    def plot_mbti_disclosure_pie(df: pd.DataFrame, cohort: str, out_dir: str):
        print("\n--- (19A-1) MBTI disclosure status (pie) ---")

        mbti_raw = df[MBTI_COL].copy() if MBTI_COL in df.columns else pd.Series([], dtype=str)
        mbti_status = mbti_raw.map(classify_mbti_status)

        order = ["Valid MBTI (16-type)", "IDK", "Prefer-not-to-say", "Missing/Other"]
        counts = mbti_status.value_counts().reindex(order, fill_value=0)

        total = int(counts.sum())
        if total == 0:
            print("⚠️ No MBTI responses at all (total=0). Skipping pie plot.")
            disclosure_df = pd.DataFrame({"status": order, "count": counts.values, "pct": [0.0]*len(order)})
            disclosure_path = os.path.join(out_dir, f"19A-1_mbti_disclosure_table_{cohort}.csv")
            disclosure_df.to_csv(disclosure_path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved table: {disclosure_path}")
            return

        pct = (counts / total * 100).round(2)

        print("MBTI disclosure breakdown (count / %):")
        for k in order:
            print(f"  {k}: {int(counts[k])} ({pct[k]}%)")

        # table save
        disclosure_df = pd.DataFrame({"status": order, "count": counts.values, "pct": pct.values})
        disclosure_path = os.path.join(out_dir, f"19A-1_mbti_disclosure_table_{cohort}.csv")
        disclosure_df.to_csv(disclosure_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved table: {disclosure_path}")

        # pie plot save
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"(19A-1) MBTI disclosure status [{cohort}]", fontsize=14, fontweight="bold", pad=12)

        base = os.path.join(out_dir, f"19A-1_mbti_disclosure_pie_{cohort}")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved plot: {base}.png/.svg")

    def plot_mbti_16type_distribution(df_valid: pd.DataFrame, cohort: str, out_dir: str):
        print("\n--- (19A-2) MBTI 16-type distribution (bar) ---")

        if df_valid.empty:
            print("⚠️ No valid 16-type MBTI responses. Skipping distribution plot.")
            return

        counts = df_valid["mbti"].value_counts().sort_index()
        total = int(counts.sum())
        pct = (counts / total * 100).round(3) if total > 0 else (counts * 0.0)

        # table save
        dist_df = pd.DataFrame({"mbti": counts.index, "count": counts.values, "pct": pct.values})
        dist_path = os.path.join(out_dir, f"19A-2_mbti_16type_distribution_{cohort}.csv")
        dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved table: {dist_path}")

        # bar plot save
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(counts.index, counts.values)
        ax.set_title(f"(19A-2) MBTI 16-type distribution [{cohort}]", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("MBTI type")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        base = os.path.join(out_dir, f"19A-2_mbti_16type_distribution_bar_{cohort}")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved plot: {base}.png/.svg")

    def run_kw_test_mbti_accuracy(df_valid: pd.DataFrame, cohort: str, out_dir: str):
        print("\n--- (19A-3) MBTI type effect on Accuracy (Kruskal–Wallis) ---")

        if df_valid.empty:
            print("⚠️ No valid 16-type MBTI data. Skipping test.")
            return

        df_valid = df_valid.copy()
        df_valid[ACC_COL] = pd.to_numeric(df_valid[ACC_COL], errors="coerce")
        df_valid = df_valid.dropna(subset=[ACC_COL, "mbti"])

        counts = df_valid["mbti"].value_counts()
        eligible = counts[counts >= MIN_N_PER_TYPE].index.tolist()

        if len(eligible) < 3:
            print(f"⚠️ Not enough MBTI types with N >= {MIN_N_PER_TYPE} (k={len(eligible)}). Skipping test.")
            report = (
                f"(19A-3) Kruskal–Wallis test skipped [{cohort}]\n"
                f"Reason: Not enough MBTI types with N >= {MIN_N_PER_TYPE}.\n\n"
                f"Counts (all valid types):\n{counts.to_string()}\n"
            )
            save_text(os.path.join(out_dir, f"19A-3_kw_test_skipped_{cohort}.txt"), report)
            return

        groups = [df_valid.loc[df_valid["mbti"] == t, ACC_COL].values for t in eligible]
        H, p = stats.kruskal(*groups)

        n_total = sum(len(g) for g in groups)
        k = len(groups)
        eps2 = eps_squared_kw(H, k, n_total)

        print(f"Kruskal–Wallis: H={H:.4f}, p={p:.6g}, k={k}, n={n_total}, eps^2={eps2:.4f}")

        # Summary per type
        summary_rows = []
        for t in eligible:
            vals = df_valid.loc[df_valid["mbti"] == t, ACC_COL]
            summary_rows.append({
                "mbti": t,
                "n": int(len(vals)),
                "mean_accuracy": float(vals.mean()),
                "std_accuracy": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "median_accuracy": float(vals.median()),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("mean_accuracy", ascending=False)
        summary_path = os.path.join(out_dir, f"19A-3_mbti_accuracy_summary_{cohort}.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved table: {summary_path}")

        # Also save eligibility counts
        elig_df = pd.DataFrame({"mbti": counts.index, "count": counts.values})
        elig_path = os.path.join(out_dir, f"19A-3_mbti_counts_valid_{cohort}.csv")
        elig_df.to_csv(elig_path, index=False, encoding="utf-8-sig")

        report = (
            f"(19A-3) MBTI type effect on Accuracy (Kruskal–Wallis) [{cohort}]\n"
            f"------------------------------------------------------------\n"
            f"Eligibility: MBTI types with N >= {MIN_N_PER_TYPE}\n"
            f"Included types (k={k}): {', '.join(eligible)}\n"
            f"Total N={n_total}\n\n"
            f"H={H:.6f}\n"
            f"p={p:.10g}\n"
            f"Effect size (epsilon-squared) eps^2={eps2:.6f}\n\n"
            f"Notes:\n"
            f"- Nonparametric omnibus test across MBTI types.\n"
            f"- MBTI groups are often imbalanced; small-N types excluded.\n"
            f"- If you want post-hoc: Dunn test + FDR (scikit-posthocs).\n"
        )
        report_path = os.path.join(out_dir, f"19A-3_kw_test_report_{cohort}.txt")
        save_text(report_path, report)
        print(f"✅ Saved report: {report_path}")

    # -----------------------------
    # Main
    # -----------------------------
    def run_section_19A(cohort: str, file_path: str):
        print(f"\n==================== [{cohort.upper()}] (19A) START ====================")

        out_dir = os.path.join(OUTPUT_DIR, cohort)
        ensure_dir(out_dir)

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path}")
            return

        # Column checks
        if MBTI_COL not in df.columns:
            print(f"❌ Missing required column: {MBTI_COL}")
            return

        acc_src = find_accuracy_col(df)
        if acc_src is None:
            print(f"❌ Missing accuracy column. Tried: {ACC_COL} + {ACC_FALLBACK_COLS}")
            return

        # Normalize accuracy to df[ACC_COL] in percent
        df = normalize_accuracy_to_percent(df, acc_src)
        print(f"✅ [{cohort}] Accuracy column used: {acc_src} -> stored as {ACC_COL} (%)")

        # (19A-1) disclosure status (includes missing/IDK/etc.)
        plot_mbti_disclosure_pie(df, cohort, out_dir)

        # Valid 16-type subset for distribution + accuracy tests
        df_valid = df[[MBTI_COL, ACC_COL]].copy()
        df_valid["mbti"] = df_valid[MBTI_COL].map(normalize_mbti)
        df_valid = df_valid[df_valid["mbti"].map(is_valid_16type)]
        df_valid[ACC_COL] = pd.to_numeric(df_valid[ACC_COL], errors="coerce")
        df_valid = df_valid.dropna(subset=[ACC_COL, "mbti"])

        # (19A-2) 16-type distribution
        plot_mbti_16type_distribution(df_valid, cohort, out_dir)

        # (19A-3) KW test + eps^2 effect size
        run_kw_test_mbti_accuracy(df_valid, cohort, out_dir)

        print(f"==================== [{cohort.upper()}] (19A) END ====================\n")

    if __name__ == "__main__":
        print("==============================================================================")
        print("(19A v2.1) MBTI Disclosure/Distribution + Accuracy Differences (MOBILE + WEB)")
        print("==============================================================================\n")

        ensure_dir(OUTPUT_DIR)

        for cohort, fp in COHORT_FILES.items():
            run_section_19A(cohort, fp)

        print("\n==================== (19A) DONE ====================")


def _run_cell_081():
    # ==============================================================================
    # (19A-4 v2.2) MBTI Accuracy Ranking: Post-hoc Pairwise Tests (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Fixes:
    #  - Auto-pick accuracy column: overallAccuracy / overallAccuracy_y / overallAccuracy_x ...
    #  - Convert proportion (0~1) -> percent (0~100) when needed
    #  - Always compute effect size (rank-biserial r) via MWU
    #  - If scikit-posthocs available: use Dunn + FDR for p_fdr (more "posthoc-consistent" with KW)
    #    but still attach MWU effect sizes in the tidy table
    #  - Else: use pairwise MWU p-values + BH-FDR
    #  - Save ranking table + pairwise table + p-matrix + heatmap + report
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats

    OUTPUT_DIR = "outputs_19A_mbti_v2"   # same as 19A
    MIN_N_PER_TYPE = 8                  # keep consistent with 19A
    MBTI_COL = "mbti"
    ALPHA = 0.05
    BOOT_N = 5000
    SEED = 42

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def normalize_mbti(x: str) -> str:
        if x is None:
            return ""
        return str(x).strip().upper()

    def is_valid_16type(x: str) -> bool:
        if x is None:
            return False
        x = str(x).strip().upper()
        return bool(re.fullmatch(r"[EI][SN][TF][JP]", x))

    def bh_fdr(pvals: np.ndarray) -> np.ndarray:
        """Benjamini–Hochberg FDR adjusted p-values."""
        pvals = np.asarray(pvals, dtype=float)
        n = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj_ranked = np.empty(n, dtype=float)
        prev = 1.0
        # step-up in reverse ensures monotonicity
        for i in range(n - 1, -1, -1):
            rank = i + 1
            val = ranked[i] * n / rank
            prev = min(prev, val)
            adj_ranked[i] = prev
        out = np.empty(n, dtype=float)
        out[order] = np.clip(adj_ranked, 0, 1)
        return out

    def bootstrap_mean_ci(x: np.ndarray, n_boot=5000, seed=42, ci=95):
        """Bootstrap mean CI (percentile). Deterministic per type."""
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan, np.nan, np.nan
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        lo = np.percentile(boots, (100 - ci) / 2)
        hi = np.percentile(boots, 100 - (100 - ci) / 2)
        return float(np.mean(x)), float(lo), float(hi)

    def rank_biserial_from_u(u_stat, n1, n2):
        """
        Rank-biserial correlation (RBC) from MWU U for group1.
        rbc = (2U)/(n1*n2) - 1
        rbc > 0 => group1 tends to be larger than group2.
        """
        denom = n1 * n2
        if denom <= 0:
            return np.nan
        return float((2.0 * u_stat) / denom - 1.0)

    def choose_accuracy_column(df: pd.DataFrame):
        """
        Choose accuracy column robustly.
        Priority:
          1) overallAccuracy
          2) overallAccuracy_y
          3) overallAccuracy_x
          4) any column containing 'overallAccuracy'
        """
        candidates = []
        for c in ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x"]:
            if c in df.columns:
                candidates.append(c)
        if not candidates:
            # fallback: any containing overallAccuracy
            hits = [c for c in df.columns if "overallaccuracy" in str(c).lower()]
            candidates.extend(hits)

        if not candidates:
            return None

        # pick the first valid candidate with enough numeric
        best = None
        best_n = -1
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            n = int(s.notna().sum())
            if n > best_n:
                best_n = n
                best = c
        return best

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str):
        """
        Return a Series 'overallAccuracy' in percent (0~100).
        If values look like proportions (<=1.2), convert *100.
        """
        s = pd.to_numeric(df[acc_col], errors="coerce")
        # heuristic: if most finite values are <= 1.2 => proportion
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            return s, f"{acc_col} (empty)"
        q95 = np.nanpercentile(finite, 95)
        if q95 <= 1.2:
            return s * 100.0, f"{acc_col} treated as proportion (0~1) -> converted to %"
        return s, f"{acc_col} treated as percent (0~100)"

    def build_valid_mbti_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build df with columns: mbti, overallAccuracy (percent)
        """
        if MBTI_COL not in df.columns:
            return pd.DataFrame(columns=["mbti", "overallAccuracy"])

        acc_col = choose_accuracy_column(df)
        if acc_col is None:
            return pd.DataFrame(columns=["mbti", "overallAccuracy"])

        acc_pct, note = ensure_accuracy_percent(df, acc_col)
        print(f"✅ Accuracy column used: {acc_col} -> stored as overallAccuracy (%)")
        print(f"   - {note}")

        out = pd.DataFrame({
            "mbti": df[MBTI_COL].map(normalize_mbti),
            "overallAccuracy": acc_pct
        })
        out["overallAccuracy"] = pd.to_numeric(out["overallAccuracy"], errors="coerce")
        out = out.dropna(subset=["mbti", "overallAccuracy"])
        out = out[out["mbti"].map(is_valid_16type)]
        return out


    # -----------------------------
    # (19A-4) Core
    # -----------------------------
    def posthoc_mbti_accuracy(cohort: str, file_path: str):
        print(f"\n==================== [{cohort.upper()}] (19A-4) START ====================")

        out_dir = os.path.join(OUTPUT_DIR, cohort)
        ensure_dir(out_dir)

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path}")
            return

        if MBTI_COL not in df.columns:
            print(f"❌ Missing required column: {MBTI_COL}")
            return

        df_valid = build_valid_mbti_df(df)
        if df_valid.empty:
            print("⚠️ No valid 16-type MBTI rows after filtering. Skip.")
            return

        counts_all = df_valid["mbti"].value_counts()
        eligible = counts_all[counts_all >= MIN_N_PER_TYPE].index.tolist()

        if len(eligible) < 3:
            msg = f"⚠️ Not enough eligible MBTI types (N >= {MIN_N_PER_TYPE}) for post-hoc. k={len(eligible)}"
            print(msg)
            save_text(
                os.path.join(out_dir, f"19A-4_posthoc_skipped_{cohort}.txt"),
                msg + "\n\nCounts:\n" + counts_all.to_string()
            )
            return

        df_elig = df_valid[df_valid["mbti"].isin(eligible)].copy()

        # --------
        # (A) Ranking with bootstrap CI
        # --------
        ranking_rows = []
        for t in eligible:
            vals = df_elig.loc[df_elig["mbti"] == t, "overallAccuracy"].values
            mean_, lo, hi = bootstrap_mean_ci(vals, n_boot=BOOT_N, seed=SEED, ci=95)
            ranking_rows.append({
                "mbti": t,
                "n": int(len(vals)),
                "mean_accuracy": mean_,
                "ci95_low": lo,
                "ci95_high": hi,
            })

        ranking_df = pd.DataFrame(ranking_rows).sort_values("mean_accuracy", ascending=False)
        ranking_path = os.path.join(out_dir, f"19A-4a_mbti_accuracy_ranking_bootstrap_{cohort}.csv")
        ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved ranking table: {ranking_path}")

        # --------
        # (B) Pairwise MWU: always compute effect sizes + p_raw
        # --------
        types = sorted(eligible)
        rows = []
        p_raw_list = []

        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                t1, t2 = types[i], types[j]
                x1 = df_elig.loc[df_elig["mbti"] == t1, "overallAccuracy"].values
                x2 = df_elig.loc[df_elig["mbti"] == t2, "overallAccuracy"].values

                u, p_raw = stats.mannwhitneyu(x1, x2, alternative="two-sided")
                rbc = rank_biserial_from_u(u, len(x1), len(x2))

                rows.append({
                    "mbti_1": t1,
                    "mbti_2": t2,
                    "n1": int(len(x1)),
                    "n2": int(len(x2)),
                    "mean1": float(np.mean(x1)),
                    "mean2": float(np.mean(x2)),
                    "mean_diff_1_minus_2": float(np.mean(x1) - np.mean(x2)),
                    "U": float(u),
                    "p_raw": float(p_raw),
                    "rank_biserial_r": float(rbc),
                })
                p_raw_list.append(float(p_raw))

        pairwise_df = pd.DataFrame(rows)

        # --------
        # (C) p_fdr: prefer Dunn-FDR if available (more aligned with KW)
        #     else: BH-FDR on MWU p-values
        # --------
        used_method = None
        pmatrix = None

        try:
            import scikit_posthocs as sp  # pip install scikit-posthocs
            used_method = "Dunn test (scikit-posthocs) + BH-FDR (p_fdr) + MWU effect sizes"

            dunn_fdr = sp.posthoc_dunn(df_elig, val_col="overallAccuracy", group_col="mbti", p_adjust="fdr_bh")

            # save matrix
            dunn_path = os.path.join(out_dir, f"19A-4b_posthoc_pmatrix_dunn_fdr_{cohort}.csv")
            dunn_fdr.to_csv(dunn_path, encoding="utf-8-sig")
            print(f"✅ Saved Dunn FDR p-matrix: {dunn_path}")

            pmatrix = dunn_fdr.copy()

            # map p_fdr into tidy table
            def _lookup_p(a, b):
                if a in dunn_fdr.index and b in dunn_fdr.columns:
                    return float(dunn_fdr.loc[a, b])
                if b in dunn_fdr.index and a in dunn_fdr.columns:
                    return float(dunn_fdr.loc[b, a])
                return np.nan

            pairwise_df["p_fdr"] = pairwise_df.apply(lambda r: _lookup_p(r["mbti_1"], r["mbti_2"]), axis=1)

        except Exception:
            used_method = "Pairwise MWU + BH-FDR (fallback)"
            p_raw = np.array(p_raw_list, dtype=float)
            p_fdr = bh_fdr(p_raw)
            pairwise_df["p_fdr"] = p_fdr

            # build p-matrix from BH-FDR results
            pmatrix = pd.DataFrame(np.ones((len(types), len(types))), index=types, columns=types)
            k = 0
            for i in range(len(types)):
                for j in range(i + 1, len(types)):
                    pmatrix.iloc[i, j] = p_fdr[k]
                    pmatrix.iloc[j, i] = p_fdr[k]
                    k += 1

            pmatrix_path = os.path.join(out_dir, f"19A-4b_posthoc_pmatrix_mwu_fdr_{cohort}.csv")
            pmatrix.to_csv(pmatrix_path, encoding="utf-8-sig")
            print(f"✅ Saved MWU BH-FDR p-matrix: {pmatrix_path}")

        # save tidy table
        pairwise_df = pairwise_df.sort_values("p_fdr")
        pairwise_path = os.path.join(out_dir, f"19A-4c_posthoc_pairwise_table_{cohort}.csv")
        pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved pairwise table: {pairwise_path}")
        print(f"✅ Method used: {used_method}")

        # --------
        # (D) Heatmap (adjusted p-values)
        # --------
        heatmap_png = os.path.join(out_dir, f"19A-4d_posthoc_pmatrix_heatmap_{cohort}.png")
        heatmap_svg = os.path.join(out_dir, f"19A-4d_posthoc_pmatrix_heatmap_{cohort}.svg")

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            # show only 0..ALPHA range (values above alpha saturate)
            im = ax.imshow(pmatrix.values, vmin=0, vmax=ALPHA, aspect="auto")

            ax.set_xticks(np.arange(len(pmatrix.columns)))
            ax.set_yticks(np.arange(len(pmatrix.index)))
            ax.set_xticklabels(pmatrix.columns, rotation=45, ha="right")
            ax.set_yticklabels(pmatrix.index)

            ax.set_title(
                f"(19A-4) Post-hoc adjusted p-values (<= {ALPHA}) [{cohort.upper()}]\n{used_method}",
                fontsize=12, fontweight="bold", pad=10
            )

            # annotate
            for i in range(pmatrix.shape[0]):
                for j in range(pmatrix.shape[1]):
                    val = pmatrix.iloc[i, j]
                    txt = f"{val:.3f}" if np.isfinite(val) else "NA"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="FDR-adjusted p")
            fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
            fig.savefig(heatmap_svg, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            print(f"✅ Saved heatmap: {heatmap_png}/.svg")
        except Exception as e:
            print(f"⚠️ Heatmap skipped: {e}")

        # --------
        # (E) Text report: significant pairs
        # --------
        sig_pairs = pairwise_df[pairwise_df["p_fdr"] < ALPHA].copy()

        report = []
        report.append(f"(19A-4) MBTI Accuracy Ranking + Post-hoc Pairwise Tests [{cohort.upper()}]")
        report.append("------------------------------------------------------------")
        report.append(f"Eligible types (N >= {MIN_N_PER_TYPE}): {len(eligible)}")
        report.append(f"Method: {used_method}")
        report.append(f"Alpha: {ALPHA}")
        report.append("")
        report.append("[Top 10 ranking by mean accuracy (bootstrap CI)]")
        report.append(ranking_df.head(10).to_string(index=False))
        report.append("")
        report.append("[Significant pairwise differences (FDR < alpha)]")
        if sig_pairs.empty:
            report.append("None.")
        else:
            cols_show = [c for c in [
                "mbti_1","mbti_2","n1","n2","mean1","mean2","mean_diff_1_minus_2","p_raw","p_fdr","rank_biserial_r"
            ] if c in sig_pairs.columns]
            report.append(sig_pairs[cols_show].head(40).to_string(index=False))
            if len(sig_pairs) > 40:
                report.append(f"... ({len(sig_pairs) - 40} more rows omitted)")

        report_path = os.path.join(out_dir, f"19A-4e_posthoc_report_{cohort}.txt")
        save_text(report_path, "\n".join(report))
        print(f"✅ Saved report: {report_path}")

        print(f"==================== [{cohort.upper()}] (19A-4) END ====================\n")


    # -----------------------------
    # Run
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(19A-4) MBTI Accuracy Ranking: Post-hoc Pairwise Tests (MOBILE + WEB) [ENGLISH]")
        print("==============================================================================\n")

        ensure_dir(OUTPUT_DIR)

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        for cohort, fp in cohort_files.items():
            posthoc_mbti_accuracy(cohort, fp)

        print("\n==================== (19A-4) DONE ====================")


def _run_cell_083():

    # ==============================================================================
    # (19B v2.1.1) MBTI & Accuracy: Statistical meaning of MBTI accuracy ranking
    # ------------------------------------------------------------------------------
    # MOBILE + WEB
    # Fixes:
    #  - Auto-pick accuracy column (overallAccuracy / overallAccuracy_y / overallAccuracy_x / contains)
    #  - Convert proportion (0~1) -> percent (0~100) when needed
    #  - Kruskal: remove nan_policy arg for SciPy compatibility (we already drop NaNs)
    #  - Keep "no duplicate mbti columns" fix
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import kruskal, ttest_ind
    import pingouin as pg

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_19B_mbti_accuracy"
    ALPHA = 0.05

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web": config.WEB_AGE_FILTERED,
    }

    MBTI_EXCLUDE = {"idk", "prefer-not-to-say", "prefer_not_to_say", "nan", "none", ""}  # normalize to lower
    MBTI_REGEX = re.compile(r"^[EI][NS][TF][JP]$")

    sns.set_theme(style="whitegrid")

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def choose_accuracy_column(df: pd.DataFrame):
        """
        Choose accuracy column robustly.
        Priority:
          1) overallAccuracy
          2) overallAccuracy_y
          3) overallAccuracy_x
          4) any column containing 'overallAccuracy'
        Pick the one with most numeric non-NA.
        """
        candidates = []
        for c in ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x"]:
            if c in df.columns:
                candidates.append(c)
        if not candidates:
            hits = [c for c in df.columns if "overallaccuracy" in str(c).lower()]
            candidates.extend(hits)
        if not candidates:
            return None

        best, best_n = None, -1
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            n = int(s.notna().sum())
            if n > best_n:
                best, best_n = c, n
        return best

    def ensure_accuracy_percent(series: pd.Series):
        """
        Convert to numeric. If it looks like proportion (95th percentile <= 1.2), multiply by 100.
        Returns (acc_pct_series, note)
        """
        s = pd.to_numeric(series, errors="coerce")
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            return s, "empty"
        q95 = np.nanpercentile(finite, 95)
        if q95 <= 1.2:
            return s * 100.0, "proportion (0~1) -> converted to %"
        return s, "already percent (0~100)"

    def cohen_d_welch(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        if not np.isfinite(sp) or sp == 0:
            return np.nan
        return (np.mean(x) - np.mean(y)) / sp

    def holm_correction(pvals: pd.Series) -> pd.Series:
        """
        Holm step-down adjusted p-values.
        """
        pvals = pd.Series(pvals).astype(float)
        m = len(pvals)
        order = np.argsort(pvals.values)  # ascending
        adj = np.empty(m, dtype=float)

        # Holm: adj_i = max_{j<=i} ( (m-j+1)*p_(j) )
        running_max = 0.0
        for k, idx in enumerate(order):
            raw = pvals.iloc[idx]
            val = (m - k) * raw
            running_max = max(running_max, val)
            adj[idx] = min(running_max, 1.0)

        return pd.Series(adj, index=pvals.index)

    def clean_mbti_series(s: pd.Series) -> pd.Series:
        s2 = s.astype(str).str.strip().str.upper()

        # normalize typical missing tokens
        s2 = s2.replace({"NAN": "", "NONE": ""})

        # remove excluded tokens (case-insensitive)
        lower = s2.str.lower()
        mask_excl = lower.isin(MBTI_EXCLUDE)
        s2 = s2.mask(mask_excl, "")

        # keep only valid 4-letter types
        valid = s2.apply(lambda x: bool(MBTI_REGEX.match(x)))
        s2 = s2.where(valid, np.nan)
        return s2

    def extract_dimensions(mbti: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=mbti.index)
        out["EI"] = mbti.str[0]
        out["NS"] = mbti.str[1]
        out["TF"] = mbti.str[2]
        out["JP"] = mbti.str[3]
        return out

    def plot_pie_disclosure(cohort_out: str, cohort_tag: str, disclosed_n: int, total_n: int):
        labels = ["Disclosed MBTI", "Not disclosed / invalid"]
        sizes = [disclosed_n, total_n - disclosed_n]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"(19B-0) MBTI disclosure rate [{cohort_tag.upper()}] (N={total_n})", pad=12)
        fn = os.path.join(cohort_out, f"19B-0_mbti_disclosure_pie_{cohort_tag}")
        fig.savefig(f"{fn}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{fn}.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return f"{fn}.png/.svg"

    def plot_mbti_distribution(cohort_out: str, cohort_tag: str, mbti_counts: pd.Series):
        # Donut
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.pie(
            mbti_counts.values,
            labels=mbti_counts.index.tolist(),
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.78,
            labeldistance=1.05,
        )
        centre_circle = plt.Circle((0, 0), 0.52, fc="white")
        fig.gca().add_artist(centre_circle)
        ax.set_title(f"(19B-0) MBTI type distribution (valid only) [{cohort_tag.upper()}]", pad=12)
        fn1 = os.path.join(cohort_out, f"19B-0_mbti_type_distribution_donut_{cohort_tag}")
        fig.savefig(f"{fn1}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{fn1}.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Bar
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=mbti_counts.values, y=mbti_counts.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("MBTI")
        ax.set_title(f"(19B-0) MBTI distribution (bar) [{cohort_tag.upper()}]", pad=12)
        fn2 = os.path.join(cohort_out, f"19B-0_mbti_type_distribution_bar_{cohort_tag}")
        fig.savefig(f"{fn2}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{fn2}.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return f"{fn1}.png/.svg", f"{fn2}.png/.svg"

    def plot_mbti_accuracy_ranking(cohort_out: str, cohort_tag: str, df_valid: pd.DataFrame):
        stats_tbl = (
            df_valid.groupby("mbti")["overallAccuracy"]
            .agg(mean="mean", sd="std", n="count")
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        stats_tbl["se"] = stats_tbl["sd"] / np.sqrt(stats_tbl["n"].clip(lower=1))

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=stats_tbl, x="mean", y="mbti", ax=ax, orient="h")
        ax.errorbar(stats_tbl["mean"], np.arange(len(stats_tbl)), xerr=stats_tbl["se"], fmt="none", capsize=3)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Mean Accuracy (%)")
        ax.set_ylabel("MBTI")
        ax.set_title(f"(19B-0) MBTI accuracy ranking (descriptive) [{cohort_tag.upper()}]", pad=12)

        fn = os.path.join(cohort_out, f"19B-0_mbti_accuracy_ranking_{cohort_tag}")
        fig.savefig(f"{fn}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{fn}.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

        csv_path = os.path.join(cohort_out, f"19B-0_mbti_accuracy_descriptives_{cohort_tag}.csv")
        stats_tbl.to_csv(csv_path, index=False, encoding="utf-8-sig")

        return f"{fn}.png/.svg", csv_path, stats_tbl

    def run_omnibus_tests(df_valid: pd.DataFrame):
        # groups per MBTI (already 1D and numeric)
        groups = [g["overallAccuracy"].dropna().values for _, g in df_valid.groupby("mbti")]
        group_sizes = [len(x) for x in groups]

        welch_tbl = pg.welch_anova(dv="overallAccuracy", between="mbti", data=df_valid)
        welch_p = float(welch_tbl["p-unc"].iloc[0])

        # Kruskal (safe, NaNs already removed)
        if sum(n >= 2 for n in group_sizes) >= 2:
            groups2 = [x for x in groups if len(x) >= 2]
            H, p_kw = kruskal(*groups2)   # <- removed nan_policy for compatibility
            H = float(H); p_kw = float(p_kw)
            N = sum(len(x) for x in groups2)
            k = len(groups2)
            eps2 = (H - k + 1) / (N - k) if (N - k) > 0 else np.nan
        else:
            H, p_kw, eps2 = np.nan, np.nan, np.nan

        return {
            "welch_table": welch_tbl,
            "welch_p": welch_p,
            "kw_H": H,
            "kw_p": p_kw,
            "kw_eps2": eps2,
            "welch_sig": welch_p < ALPHA,
            "kw_sig": (p_kw < ALPHA) if np.isfinite(p_kw) else False,
        }

    def run_posthoc_gameshowell(df_valid: pd.DataFrame):
        return pg.pairwise_gameshowell(dv="overallAccuracy", between="mbti", data=df_valid)

    def run_dimension_tests(df_valid: pd.DataFrame):
        dims = extract_dimensions(df_valid["mbti"])
        out_rows = []

        for dim, (g1, g2) in {
            "EI": ("E", "I"),
            "NS": ("N", "S"),
            "TF": ("T", "F"),
            "JP": ("J", "P"),
        }.items():
            df_dim = df_valid.copy()
            df_dim[dim] = dims[dim]
            a = df_dim.loc[df_dim[dim] == g1, "overallAccuracy"].dropna()
            b = df_dim.loc[df_dim[dim] == g2, "overallAccuracy"].dropna()

            if len(a) < 10 or len(b) < 10:
                out_rows.append({
                    "dimension": dim,
                    "group1": g1, "n1": len(a), "mean1": float(a.mean()) if len(a) else np.nan,
                    "group2": g2, "n2": len(b), "mean2": float(b.mean()) if len(b) else np.nan,
                    "t": np.nan, "p": np.nan, "cohen_d": np.nan,
                    "note": "Skipped (too small N in at least one group)"
                })
                continue

            t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            d = cohen_d_welch(a.values, b.values)

            out_rows.append({
                "dimension": dim,
                "group1": g1, "n1": len(a), "mean1": float(a.mean()),
                "group2": g2, "n2": len(b), "mean2": float(b.mean()),
                "t": float(t), "p": float(p), "cohen_d": float(d) if np.isfinite(d) else np.nan,
                "note": ""
            })

        out_df = pd.DataFrame(out_rows)
        mask = np.isfinite(out_df["p"].values)
        out_df["p_holm"] = np.nan
        if mask.sum() > 0:
            out_df.loc[mask, "p_holm"] = holm_correction(out_df.loc[mask, "p"])
        out_df["significant_holm"] = out_df["p_holm"] < ALPHA
        return out_df


    def analyze_cohort(cohort_tag: str, file_path: str):
        print(f"\n==================== [{cohort_tag.upper()}] (19B) START ====================")

        cohort_out = os.path.join(OUTPUT_DIR, cohort_tag)
        ensure_dir(cohort_out)

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")

        if "mbti" not in df.columns:
            raise KeyError("Missing required column: mbti")

        acc_col = choose_accuracy_column(df)
        if acc_col is None:
            raise KeyError("Missing accuracy column (overallAccuracy / overallAccuracy_y / overallAccuracy_x / ... )")

        acc_pct, note = ensure_accuracy_percent(df[acc_col])
        print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} -> stored as overallAccuracy (%)")
        print(f"   - {note}")

        df2 = df[["mbti"]].copy()
        df2["overallAccuracy"] = acc_pct

        # Clean numeric
        df2["overallAccuracy"] = pd.to_numeric(df2["overallAccuracy"], errors="coerce")

        # Clean MBTI
        df2["mbti_clean"] = clean_mbti_series(df2["mbti"])

        total_n = len(df2)
        disclosed_n = int(df2["mbti_clean"].notna().sum())

        # --- (19B-0) disclosure pie ---
        pie1 = plot_pie_disclosure(cohort_out, cohort_tag, disclosed_n, total_n)
        print(f"✅ [{cohort_tag}] Saved disclosure pie: {pie1}")

        # ✅ build df_valid with ONLY mbti_clean + overallAccuracy (avoid duplicate 'mbti')
        df_valid = df2.loc[
            df2["mbti_clean"].notna() & df2["overallAccuracy"].notna(),
            ["mbti_clean", "overallAccuracy"]
        ].copy()
        df_valid = df_valid.rename(columns={"mbti_clean": "mbti"})  # single 1D mbti column

        if len(df_valid) < 30:
            print(f"⚠️ [{cohort_tag}] Too few valid MBTI rows for inferential stats (N={len(df_valid)}).")
            mbti_counts = df2["mbti_clean"].value_counts(dropna=True).sort_index()
            if len(mbti_counts) > 0:
                donut, bar = plot_mbti_distribution(cohort_out, cohort_tag, mbti_counts)
                print(f"✅ [{cohort_tag}] Saved MBTI distribution: {donut} and {bar}")
            return

        mbti_counts = df_valid["mbti"].value_counts().sort_index()
        donut, bar = plot_mbti_distribution(cohort_out, cohort_tag, mbti_counts)
        print(f"✅ [{cohort_tag}] Saved MBTI distribution: {donut} and {bar}")

        rank_plot, desc_csv, _ = plot_mbti_accuracy_ranking(cohort_out, cohort_tag, df_valid)
        print(f"✅ [{cohort_tag}] Saved MBTI accuracy ranking plot: {rank_plot}")
        print(f"✅ [{cohort_tag}] Saved MBTI accuracy descriptives: {desc_csv}")

        # --- (19B-1) Omnibus tests ---
        group_sizes = df_valid.groupby("mbti")["overallAccuracy"].size()
        valid_groups = group_sizes[group_sizes >= 5].index.tolist()
        df_omni = df_valid[df_valid["mbti"].isin(valid_groups)].copy()

        print(f"\n--- (19B-1) Omnibus tests (Welch ANOVA + Kruskal-Wallis) [{cohort_tag}] ---")
        print(f"Valid MBTI groups (n>=5): {len(valid_groups)} / {df_valid['mbti'].nunique()} types")
        print(f"N used for omnibus (after group-size filter): {len(df_omni)}")

        omnibus = run_omnibus_tests(df_omni)
        welch_tbl = omnibus["welch_table"].copy()

        welch_tbl_path = os.path.join(cohort_out, f"19B-1_welch_anova_{cohort_tag}.csv")
        welch_tbl.to_csv(welch_tbl_path, index=False, encoding="utf-8-sig")

        welch_F = float(welch_tbl["F"].iloc[0])
        welch_p = float(welch_tbl["p-unc"].iloc[0])
        welch_df1 = float(welch_tbl["ddof1"].iloc[0])
        welch_df2 = float(welch_tbl["ddof2"].iloc[0])
        welch_np2 = float(welch_tbl["np2"].iloc[0]) if "np2" in welch_tbl.columns else np.nan

        print(f"Welch ANOVA: F({welch_df1:.0f}, {welch_df2:.2f}) = {welch_F:.3f}, p = {welch_p:.6g}, np2 = {welch_np2:.4f}")
        if np.isfinite(omnibus["kw_p"]):
            print(f"Kruskal-Wallis: H = {omnibus['kw_H']:.3f}, p = {omnibus['kw_p']:.6g}, epsilon^2 = {omnibus['kw_eps2']:.4f}")
        else:
            print("Kruskal-Wallis: skipped (insufficient group sizes).")

        report = []
        report.append(f"(19B-1) Omnibus MBTI test on Accuracy [{cohort_tag}]\n")
        report.append(f"- Total rows: {total_n}\n")
        report.append(f"- Valid MBTI rows: {len(df_valid)}\n")
        report.append(f"- Valid groups (n>=5): {len(valid_groups)} types\n")
        report.append(f"- N used (after group-size filter): {len(df_omni)}\n\n")
        report.append("[Welch ANOVA]\n")
        report.append(welch_tbl.to_string(index=False))
        report.append("\n\n")
        report.append(f"Key: F({welch_df1:.0f}, {welch_df2:.2f})={welch_F:.3f}, p={welch_p:.6g}, np2={welch_np2:.4f}\n\n")
        report.append("[Kruskal-Wallis]\n")
        report.append(f"H={omnibus['kw_H']:.3f}, p={omnibus['kw_p']:.6g}, epsilon^2={omnibus['kw_eps2']:.4f}\n\n")

        gh_path = None
        if omnibus["welch_sig"]:
            print(f"✅ [{cohort_tag}] Welch omnibus is significant (p<{ALPHA}). Running Games-Howell post-hoc...")
            gh = run_posthoc_gameshowell(df_omni)
            gh_path = os.path.join(cohort_out, f"19B-1_posthoc_gameshowell_{cohort_tag}.csv")
            gh.to_csv(gh_path, index=False, encoding="utf-8-sig")
            gh_sorted = gh.sort_values("pval").head(10)
            print("Top post-hoc pairs (smallest p):")
            print(gh_sorted[["A", "B", "diff", "pval", "hedges"]].to_string(index=False))
            report.append("[Post-hoc: Games-Howell]\n")
            report.append(f"Saved: {gh_path}\n")
            report.append("Top 10 pairs by p-value:\n")
            report.append(gh_sorted[["A", "B", "diff", "pval", "hedges"]].to_string(index=False))
            report.append("\n\n")
        else:
            print(f"⚠️ [{cohort_tag}] Welch omnibus is NOT significant (p>={ALPHA}). Post-hoc skipped.")
            report.append("[Post-hoc]\nSkipped because omnibus test was not significant.\n\n")

        omni_report_path = os.path.join(cohort_out, f"19B-1_omnibus_report_{cohort_tag}.txt")
        save_text(omni_report_path, "".join(report))
        print(f"✅ [{cohort_tag}] Saved Welch table: {welch_tbl_path}")
        print(f"✅ [{cohort_tag}] Saved omnibus report: {omni_report_path}")
        if gh_path:
            print(f"✅ [{cohort_tag}] Saved post-hoc: {gh_path}")

        # --- (19B-2) Dimension tests ---
        print(f"\n--- (19B-2) MBTI dimension tests (Welch t-test + Holm) [{cohort_tag}] ---")
        dim_df = run_dimension_tests(df_valid)

        dim_csv = os.path.join(cohort_out, f"19B-2_mbti_dimension_tests_{cohort_tag}.csv")
        dim_df.to_csv(dim_csv, index=False, encoding="utf-8-sig")

        show_cols = ["dimension", "group1", "n1", "mean1", "group2", "n2", "mean2", "t", "p", "p_holm", "cohen_d", "significant_holm", "note"]
        print(dim_df[show_cols].round(4).to_string(index=False))

        dim_report = []
        dim_report.append(f"(19B-2) MBTI dimension tests on Accuracy [{cohort_tag}]\n")
        dim_report.append("- Tests: EI, NS, TF, JP (Welch t-test)\n")
        dim_report.append("- Multiple-comparison correction: Holm (across 4 tests)\n\n")
        dim_report.append(dim_df[show_cols].to_string(index=False))
        dim_report.append("\n")
        dim_report_path = os.path.join(cohort_out, f"19B-2_dimension_report_{cohort_tag}.txt")
        save_text(dim_report_path, "".join(dim_report))

        print(f"✅ [{cohort_tag}] Saved dimension tests: {dim_csv}")
        print(f"✅ [{cohort_tag}] Saved dimension report: {dim_report_path}")

        print(f"==================== [{cohort_tag.upper()}] (19B) END ====================\n")


    # -----------------------------
    # Run
    # -----------------------------
    print("==============================================================================")
    print("(19B) MBTI & Accuracy: Statistical meaning (MOBILE + WEB) [ENGLISH]")
    print("==============================================================================\n")

    ensure_dir(OUTPUT_DIR)

    for cohort_tag, path in COHORT_FILES.items():
        try:
            analyze_cohort(cohort_tag, path)
        except Exception as e:
            print(f"❌ [{cohort_tag}] Failed: {e}")

    print("\n==================== (19B) DONE ====================")


def _run_cell_087():
    # ==============================================================================
    # (19B v2.3) MBTI & Accuracy with Covariates (ANCOVA/Regression) - MOBILE + WEB
    # ------------------------------------------------------------------------------
    # FIXES:
    #  - Robustly detect Accuracy column (overallAccuracy / overallAccuracy_y / etc.)
    #  - Convert accuracy proportion (0~1) -> percent (0~100) automatically
    #  - Robustly detect Age column (age / Age / etc.)  [kept minimal]
    #  - Robust parameter extraction for dimension models (HC3)
    #  - Cleaner adjusted-dimension plotting (no axis clear hack)
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_19B_mbti_accuracy_v2"
    SHOW_FIGURES = True
    MIN_GROUP_N = 5
    ALPHA = 0.05

    RT_COL_CANDIDATES = ["avgRT", "mean_rt", "MeanRT", "meanRT"]
    ACC_COL_CANDIDATES = ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x", "accuracy", "Accuracy"]
    AGE_COL_CANDIDATES = ["age", "Age", "age_years", "AgeYears"]

    # -----------------------------
    # Small utilities
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _safe_show_close(fig):
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)

    def get_first_col(df: pd.DataFrame, colname: str):
        """Handle duplicated colnames: returns first as Series."""
        if colname not in df.columns:
            return None
        sub = df.loc[:, colname]
        if isinstance(sub, pd.DataFrame):
            return sub.iloc[:, 0]
        return sub

    def find_first_existing_col(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def normalize_mbti(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in {"IDK", "PREFER-NOT-TO-SAY", "PREFER_NOT_TO_SAY", "PREFER NOT TO SAY",
                 "NA", "N/A", "NONE", "NAN", ""}:
            return np.nan
        s = re.sub(r"[^A-Z]", "", s)
        if len(s) != 4:
            return np.nan
        if s[0] not in {"E", "I"}: return np.nan
        if s[1] not in {"N", "S"}: return np.nan
        if s[2] not in {"T", "F"}: return np.nan
        if s[3] not in {"J", "P"}: return np.nan
        return s

    def find_rt_column(df: pd.DataFrame):
        return find_first_existing_col(df, RT_COL_CANDIDATES)

    def find_age_column(df: pd.DataFrame):
        return find_first_existing_col(df, AGE_COL_CANDIDATES)

    def find_acc_column(df: pd.DataFrame):
        return find_first_existing_col(df, ACC_COL_CANDIDATES)

    def to_percent_if_needed(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        # If looks like proportion, convert to %
        mx = s.max(skipna=True)
        if pd.notna(mx) and mx <= 1.5:
            return s * 100.0
        return s

    def star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    # -----------------------------
    # Plot helpers
    # -----------------------------
    def plot_disclosure_pie(cohort_out, cohort_tag, n_disclosed, n_total):
        fig, ax = plt.subplots(figsize=(7, 7))
        labels = ["MBTI disclosed", "Not disclosed"]
        sizes = [n_disclosed, n_total - n_disclosed]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"(19Bv2-0) MBTI disclosure [{cohort_tag.upper()}]", fontsize=14, fontweight="bold")
        out_base = os.path.join(cohort_out, f"19Bv2-0_mbti_disclosure_pie_{cohort_tag}")
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved disclosure pie: {out_base}.png/.svg")
        _safe_show_close(fig)

    def plot_mbti_distribution(cohort_out, cohort_tag, mbti_series):
        counts = mbti_series.value_counts().sort_values(ascending=False)
        dist_df = counts.reset_index()
        dist_df.columns = ["mbti", "count"]
        dist_df["pct"] = dist_df["count"] / dist_df["count"].sum() * 100

        # donut
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.pie(
            dist_df["count"].values,
            labels=dist_df["mbti"].values,
            autopct="%1.1f%%",
            startangle=90
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        ax.add_artist(centre_circle)
        ax.set_title(f"(19Bv2-0) MBTI type distribution (donut) [{cohort_tag.upper()}]",
                     fontsize=14, fontweight="bold")
        out_base = os.path.join(cohort_out, f"19Bv2-0_mbti_type_distribution_donut_{cohort_tag}")
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved MBTI distribution donut: {out_base}.png/.svg")
        _safe_show_close(fig)

        # bar (%)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(dist_df["mbti"], dist_df["pct"])
        ax.set_ylabel("Percent (%)")
        ax.set_xlabel("MBTI")
        ax.set_title(f"(19Bv2-0) MBTI type distribution (bar) [{cohort_tag.upper()}]",
                     fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        out_base = os.path.join(cohort_out, f"19Bv2-0_mbti_type_distribution_bar_{cohort_tag}")
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved MBTI distribution bar: {out_base}.png/.svg")
        _safe_show_close(fig)

    def compute_raw_descriptives(df_mbti, acc_col, cohort_out, cohort_tag):
        desc = (
            df_mbti.groupby("mbti")[acc_col]
            .agg(mean="mean", std="std", n="count")
            .sort_values("mean", ascending=False)
        )
        desc["se"] = desc["std"] / np.sqrt(desc["n"].clip(lower=1))
        desc["ci95"] = 1.96 * desc["se"]
        desc["ci_low"] = desc["mean"] - desc["ci95"]
        desc["ci_high"] = desc["mean"] + desc["ci95"]

        out_csv = os.path.join(cohort_out, f"19Bv2-0_mbti_accuracy_descriptives_raw_{cohort_tag}.csv")
        desc.reset_index().to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw descriptives: {out_csv}")
        return desc

    def plot_raw_ranking(desc, cohort_out, cohort_tag):
        plot_df = desc.reset_index().sort_values("mean", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(plot_df["mbti"], plot_df["mean"])
        ax.errorbar(
            plot_df["mean"], np.arange(len(plot_df)),
            xerr=(plot_df["mean"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["mean"]),
            fmt="none", capsize=3
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Mean Accuracy (%)")
        ax.set_title(f"(19Bv2-0) Raw MBTI accuracy ranking (supplementary) [{cohort_tag.upper()}]",
                     fontsize=14, fontweight="bold")
        out_base = os.path.join(cohort_out, f"19Bv2-0_mbti_accuracy_ranking_raw_{cohort_tag}")
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved raw ranking plot: {out_base}.png/.svg")
        _safe_show_close(fig)

    def robust_wald_term(robust, term_name="C(mbti)"):
        wt = robust.wald_test_terms(skip_single=False)
        # In some versions, wt.table index might be a MultiIndex-like or include spaces.
        idxs = list(wt.table.index)
        if term_name not in idxs:
            # fallback partial match
            hit = None
            for i in idxs:
                if str(i).strip() == term_name:
                    hit = i
                    break
            if hit is None:
                for i in idxs:
                    if term_name in str(i):
                        hit = i
                        break
            if hit is None:
                raise KeyError(f"Term '{term_name}' not found in wald_test_terms(). Index={idxs[:10]}")
            term_name = hit

        row = wt.table.loc[term_name]
        return {
            "stat": float(row["statistic"]),
            "df": int(row["df_constraint"]),
            "p": float(row["pvalue"]),
        }

    def adjusted_means_by_level(model, factor_col, levels, covar_fixed: dict):
        rows = []
        for lv in levels:
            row = {factor_col: lv}
            row.update(covar_fixed)
            pred_df = pd.DataFrame([row])
            sf = model.get_prediction(pred_df).summary_frame(alpha=0.05)
            rows.append({
                factor_col: lv,
                "adj_mean": float(sf["mean"].iloc[0]),
                "ci_low": float(sf["mean_ci_lower"].iloc[0]),
                "ci_high": float(sf["mean_ci_upper"].iloc[0]),
            })
        return pd.DataFrame(rows)

    def plot_adjusted_ranking(adj_df, cohort_out, cohort_tag, mbti_main_p):
        plot_df = adj_df.sort_values("adj_mean", ascending=True).copy()
        sig_mark = star(mbti_main_p)
        title = f"(19Bv2-2) Adjusted MBTI means (Age/RT-controlled) [{cohort_tag.upper()}]  p={mbti_main_p:.4g}{sig_mark}"

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(plot_df["mbti"], plot_df["adj_mean"])
        ax.errorbar(
            plot_df["adj_mean"], np.arange(len(plot_df)),
            xerr=(plot_df["adj_mean"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["adj_mean"]),
            fmt="none", capsize=3
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Adjusted Accuracy (%)")
        ax.set_title(title, fontsize=12.5, fontweight="bold")

        out_base = os.path.join(cohort_out, f"19Bv2-2_adjusted_mbti_means_ranking_{cohort_tag}" +
                                ("_SIG" if mbti_main_p < ALPHA else ""))
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved adjusted ranking plot: {out_base}.png/.svg")
        _safe_show_close(fig)

    def plot_dimension_bars_raw(df_dim, dim_name, cohort_out, cohort_tag, acc_col="overallAccuracy"):
        means = df_dim.groupby("dim")[acc_col].mean()
        ses = df_dim.groupby("dim")[acc_col].sem()
        order = list(means.index)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(order, [means[o] for o in order], yerr=[ses[o] for o in order], capsize=4)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"(19Bv2-3) {dim_name} raw means [{cohort_tag.upper()}]", fontsize=12, fontweight="bold")

        out_base = os.path.join(cohort_out, f"19Bv2-3_{dim_name}_raw_bar_{cohort_tag}")
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        _safe_show_close(fig)

    def plot_dimension_bars_adjusted(adj2_df, dim_name, cohort_out, cohort_tag, p_holm):
        plot_df = adj2_df.copy()
        order = plot_df["dim"].tolist()

        sig_mark = star(p_holm)
        title = f"(19Bv2-3) {dim_name} adjusted means (Age/RT-controlled) [{cohort_tag.upper()}]  p(Holm)={p_holm:.4g}{sig_mark}"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(order, plot_df["adj_mean"].values)
        yerr = np.vstack([
            plot_df["adj_mean"].values - plot_df["ci_low"].values,
            plot_df["ci_high"].values - plot_df["adj_mean"].values
        ])
        ax.errorbar(order, plot_df["adj_mean"].values, yerr=yerr, fmt="none", capsize=4)

        ax.set_ylim(0, 100)
        ax.set_ylabel("Adjusted Accuracy (%)")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        if p_holm < ALPHA:
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)

        out_base = os.path.join(cohort_out, f"19Bv2-3_{dim_name}_adjusted_bar_{cohort_tag}" +
                                ("_SIG" if p_holm < ALPHA else ""))
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight")
        _safe_show_close(fig)

    # -----------------------------
    # Main runner per cohort
    # -----------------------------
    def run_19B_v2_for_cohort(cohort_tag, file_path):
        cohort_out = os.path.join(OUTPUT_DIR, cohort_tag)
        ensure_dir(cohort_out)

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")

        rt_col = find_rt_column(df)
        if rt_col is None:
            raise KeyError(f"No RT column found among {RT_COL_CANDIDATES}.")
        age_col = find_age_column(df)
        if age_col is None:
            raise KeyError(f"No age column found among {AGE_COL_CANDIDATES}.")
        acc_col = find_acc_column(df)
        if acc_col is None:
            raise KeyError(f"No accuracy column found among {ACC_COL_CANDIDATES}.")

        print(f"✅ [{cohort_tag}] Using Accuracy column: {acc_col}")
        print(f"✅ [{cohort_tag}] Using Age column: {age_col}")
        print(f"✅ [{cohort_tag}] Using RT column: {rt_col}")

        mbti_raw = get_first_col(df, "mbti")
        if mbti_raw is None:
            raise KeyError("Missing column: mbti")

        # Build analysis df (keep only needed columns)
        d = pd.DataFrame({
            "mbti_raw": mbti_raw,
            "overallAccuracy": to_percent_if_needed(df[acc_col]),
            "age": pd.to_numeric(df[age_col], errors="coerce"),
            rt_col: pd.to_numeric(df[rt_col], errors="coerce"),
        })

        # Basic cleaning: acc/age/rt present
        d = d.dropna(subset=["overallAccuracy", "age", rt_col]).copy()
        print(f"✅ [{cohort_tag}] N after basic cleaning (acc/age/rt present): {len(d)}")

        # Normalize MBTI
        d["mbti"] = d["mbti_raw"].apply(normalize_mbti)

        n_total = len(d)
        n_disclosed = int(d["mbti"].notna().sum())
        print(f"✅ [{cohort_tag}] N with MBTI disclosed: {n_disclosed} / {n_total}")

        # (0) disclosure pie
        plot_disclosure_pie(cohort_out, cohort_tag, n_disclosed, n_total)

        # keep disclosed only
        df_mbti = d.dropna(subset=["mbti"]).copy()

        # (0b) distribution plots
        plot_mbti_distribution(cohort_out, cohort_tag, df_mbti["mbti"])

        # raw descriptives + plot
        raw_desc = compute_raw_descriptives(df_mbti, "overallAccuracy", cohort_out, cohort_tag)
        plot_raw_ranking(raw_desc, cohort_out, cohort_tag)

        # group-size filter
        counts = df_mbti["mbti"].value_counts()
        valid_types = counts[counts >= MIN_GROUP_N].index.tolist()
        print(f"✅ [{cohort_tag}] Valid MBTI groups (n>={MIN_GROUP_N}): {len(valid_types)} / 16")

        df_mbti_valid = df_mbti[df_mbti["mbti"].isin(valid_types)].copy()
        print(f"✅ [{cohort_tag}] N used for 16-type models: {len(df_mbti_valid)}")

        # -----------------------------
        # (19Bv2-1) ANCOVA/Regression (HC3 robust)
        # -----------------------------
        print(f"\n--- (19Bv2-1) ANCOVA/Regression with covariates [{cohort_tag}] ---")
        formula = f"overallAccuracy ~ C(mbti) + age + {rt_col}"
        model = smf.ols(formula, data=df_mbti_valid).fit()
        robust = model.get_robustcov_results(cov_type="HC3")

        wald = robust_wald_term(robust, term_name="C(mbti)")
        mbti_p = wald["p"]

        print(f"[19Bv2-1] Model: {formula} (HC3)")
        print(f"  - Robust Wald (MBTI): stat={wald['stat']:.3f}, df={wald['df']}, p={mbti_p:.6g}")
        print(f"  - R2={model.rsquared:.4f}, Adj.R2={model.rsquared_adj:.4f}")

        # save coefficients (HC3)
        coef_df = pd.DataFrame({
            "term": robust.model.exog_names,
            "coef": np.asarray(robust.params, dtype=float),
            "se_HC3": np.asarray(robust.bse, dtype=float),
            "t": np.asarray(robust.tvalues, dtype=float),
            "p": np.asarray(robust.pvalues, dtype=float),
        })
        coef_path = os.path.join(cohort_out, f"19Bv2-1_coefficients_HC3_{cohort_tag}.csv")
        coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

        report = []
        report.append(f"(19B v2.3) ANCOVA/Regression report [{cohort_tag}]\n")
        report.append(f"Model: {formula}\n")
        report.append(f"Robust covariance: HC3\n")
        report.append(f"MBTI main effect (robust Wald): stat={wald['stat']:.4f}, df={wald['df']}, p={mbti_p:.6g}\n")
        report.append(f"R2={model.rsquared:.4f}, Adj.R2={model.rsquared_adj:.4f}\n\n")
        report.append("Coefficients (HC3):\n")
        report.append(coef_df.to_string(index=False))
        report_path = os.path.join(cohort_out, f"19Bv2-1_ancova_report_{cohort_tag}.txt")
        save_text(report_path, "\n".join(report))
        print(f"✅ [{cohort_tag}] Saved ANCOVA report + coeffs: {report_path}, {coef_path}")

        # -----------------------------
        # (19Bv2-2) Adjusted means (LS-means) + plot
        # -----------------------------
        print(f"\n--- (19Bv2-2) Adjusted MBTI means + ranking plot [{cohort_tag}] ---")
        mean_age = float(df_mbti_valid["age"].mean())
        mean_rt = float(df_mbti_valid[rt_col].mean())

        adj_df = adjusted_means_by_level(
            model=model,
            factor_col="mbti",
            levels=valid_types,
            covar_fixed={"age": mean_age, rt_col: mean_rt}
        )
        adj_df = adj_df.sort_values("adj_mean", ascending=False)

        adj_path = os.path.join(cohort_out, f"19Bv2-2_adjusted_means_{cohort_tag}.csv")
        adj_df.to_csv(adj_path, index=False, encoding="utf-8-sig")

        print("[Adjusted means TOP 5]")
        print(adj_df.head(5).to_string(index=False))
        print("[Adjusted means BOTTOM 5]")
        print(adj_df.tail(5).to_string(index=False))
        print(f"✅ [{cohort_tag}] Saved adjusted means table: {adj_path}")

        plot_adjusted_ranking(adj_df, cohort_out, cohort_tag, mbti_main_p=mbti_p)

        # -----------------------------
        # (19Bv2-3) Dimension tests + Holm + plots
        # -----------------------------
        print(f"\n--- (19Bv2-3) MBTI dimension tests + bar graphs [{cohort_tag}] ---")

        df_dim = df_mbti_valid.copy()
        df_dim["EI"] = df_dim["mbti"].str[0]
        df_dim["NS"] = df_dim["mbti"].str[1]
        df_dim["TF"] = df_dim["mbti"].str[2]
        df_dim["JP"] = df_dim["mbti"].str[3]

        # Standard contrasts (group1 vs group2)
        dim_specs = {
            "EI": ("E", "I"),
            "NS": ("N", "S"),
            "TF": ("T", "F"),
            "JP": ("J", "P"),
        }

        p_list = []
        dim_order = []
        dim_models = {}
        raw_dim_means_rows = []

        for dim_name, (g1, g2) in dim_specs.items():
            dd = df_dim[df_dim[dim_name].isin([g1, g2])].copy()
            dd = dd.rename(columns={dim_name: "dim"})
            dd["dim"] = pd.Categorical(dd["dim"], categories=[g1, g2], ordered=True)

            # raw summary
            for g in [g1, g2]:
                vals = dd.loc[dd["dim"] == g, "overallAccuracy"].values
                raw_dim_means_rows.append({
                    "dimension": dim_name, "group": g,
                    "n": int(np.sum(dd["dim"] == g)),
                    "mean": float(np.nanmean(vals)) if len(vals) else np.nan
                })

            f_dim = f"overallAccuracy ~ C(dim) + age + {rt_col}"
            m_dim = smf.ols(f_dim, data=dd).fit()
            r_dim = m_dim.get_robustcov_results(cov_type="HC3")

            wt = r_dim.wald_test_terms(skip_single=False)
            row = wt.table.loc["C(dim)"]
            pval = float(row["pvalue"])

            p_list.append(pval)
            dim_order.append(dim_name)
            dim_models[dim_name] = (dd, m_dim, r_dim, f_dim, g1, g2)

        reject, p_holm, _, _ = multipletests(p_list, alpha=ALPHA, method="holm")
        holm_map = {dim: float(p_holm[i]) for i, dim in enumerate(dim_order)}
        rej_map = {dim: bool(reject[i]) for i, dim in enumerate(dim_order)}

        dim_rows = []
        for dim_name in dim_order:
            dd, m_dim, r_dim, f_dim, g1, g2 = dim_models[dim_name]
            ph = holm_map[dim_name]
            sig = rej_map[dim_name]

            # beta = (g2 - g1), because g1 is reference category
            coef_name = f"C(dim)[T.{g2}]"
            exog_names = list(r_dim.model.exog_names)
            beta = np.nan
            if coef_name in exog_names:
                beta = float(np.asarray(r_dim.params)[exog_names.index(coef_name)])

            wt = r_dim.wald_test_terms(skip_single=False)
            row = wt.table.loc["C(dim)"]

            dim_rows.append({
                "dimension": dim_name,
                "contrast": f"{g2} - {g1} (ref={g1})",
                "wald_stat": float(row["statistic"]),
                "wald_df": int(row["df_constraint"]),
                "p_value": float(row["pvalue"]),
                "p_holm": ph,
                "significant_holm": sig,
                "beta_(g2_minus_g1)": beta,
                "R2": float(m_dim.rsquared),
                "model": f_dim
            })

            # plots
            plot_dimension_bars_raw(dd, dim_name, cohort_out, cohort_tag, acc_col="overallAccuracy")

            mean_age2 = float(dd["age"].mean())
            mean_rt2 = float(dd[rt_col].mean())
            adj2 = adjusted_means_by_level(
                model=m_dim,
                factor_col="dim",
                levels=[g1, g2],
                covar_fixed={"age": mean_age2, rt_col: mean_rt2}
            )
            # keep order g1,g2
            adj2 = adj2.rename(columns={"dim": "dim"})
            plot_dimension_bars_adjusted(adj2, dim_name, cohort_out, cohort_tag, p_holm=ph)

        dim_df = pd.DataFrame(dim_rows)
        dim_path = os.path.join(cohort_out, f"19Bv2-3_dimension_results_{cohort_tag}.csv")
        dim_df.to_csv(dim_path, index=False, encoding="utf-8-sig")

        raw_dim_means_df = pd.DataFrame(raw_dim_means_rows)
        raw_dim_means_path = os.path.join(cohort_out, f"19Bv2-3_dimension_raw_means_{cohort_tag}.csv")
        raw_dim_means_df.to_csv(raw_dim_means_path, index=False, encoding="utf-8-sig")

        print("\n[19Bv2-3] Dimension tests (HC3) + Holm")
        print(dim_df[["dimension","contrast","wald_stat","wald_df","p_value","p_holm","significant_holm","beta_(g2_minus_g1)","R2"]]
              .to_string(index=False))
        print(f"✅ [{cohort_tag}] Saved dimension results: {dim_path}")
        print(f"✅ [{cohort_tag}] Saved dimension raw means: {raw_dim_means_path}")

    # -----------------------------
    # Run
    # -----------------------------
    print("==============================================================================")
    print("(19B v2.3) MBTI & Accuracy with Covariates (ANCOVA/Regression) - MOBILE + WEB")
    print("==============================================================================\n")

    ensure_dir(OUTPUT_DIR)

    cohorts = [
        ("mobile", config.MOBILE_AGE_FILTERED),
        ("web",    config.WEB_AGE_FILTERED),
    ]

    for cohort_tag, file_path in cohorts:
        print(f"\n==================== [{cohort_tag.upper()}] (19B v2.3) START ====================")
        if not os.path.exists(file_path):
            print(f"❌ Missing file for {cohort_tag}: {file_path}")
            continue
        try:
            run_19B_v2_for_cohort(cohort_tag, file_path)
        except Exception as e:
            print(f"❌ [{cohort_tag}] Failed: {e}")
        print(f"==================== [{cohort_tag.upper()}] (19B v2.3) END ====================\n")

    print("==================== (19B v2.3) DONE ====================")


def _run_cell_090():
    # ==============================================================================
    # (20 v2 FIXED) AI Tool Usage & Accuracy with Covariates (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Fixes:
    #  - Auto-detect accuracy column: overallAccuracy or overallAccuracy_y
    #  - Convert accuracy to percent if looks like proportion (0~1)
    #  - Seaborn barplot errorbar compatibility: manual mean+SE bars (no errorbar=)
    #  - Avoid overwriting formula variable in report (formula_201 vs formula_202)
    #  - Robust scalar extraction for statsmodels results
    #  - Avoid SettingWithCopyWarning
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import pearsonr, spearmanr
    from statsmodels.formula.api import ols

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_20_tool_usage_v2"
    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }
    ALPHA = 0.05

    IMAGE_GEN_TOOLS = [
        "midjourney", "dall-e", "dalle", "stable-diffusion", "stablediffusion", "sd",
        "firefly", "adobe firefly", "leonardo", "leonardo.ai", "ideogram", "runway",
        "flux", "krea", "kandinsky", "dreamstudio"
    ]

    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()


    # -----------------------------
    # Utilities
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def safe_num(s):
        return pd.to_numeric(s, errors="coerce")

    def as_scalar(x):
        a = np.asarray(x)
        return float(a.item()) if a.size == 1 else float(a.ravel()[0])

    def console_sig(p, alpha=ALPHA):
        return "✅" if (np.isfinite(p) and p < alpha) else "❌"

    def get_rt_col(df):
        for c in ["avgRT", "mean_rt", "MeanRT", "meanRT"]:
            if c in df.columns:
                return c
        return None

    def get_acc_col(df):
        # prefer overallAccuracy_y if exists (your pipeline often uses it)
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        return None

    def normalize_tool_string(s: str) -> str:
        if pd.isna(s):
            return "none"
        s = str(s).strip()
        if s == "" or s.lower() in ["none", "nan", "null"]:
            return "none"
        return s

    def parse_tools(tool_string: str):
        tool_string = normalize_tool_string(tool_string)
        if tool_string == "none":
            return []
        parts = re.split(r"[,;/|]+", tool_string)
        tools = []
        for p in parts:
            t = p.strip()
            if t == "" or t.lower() == "none":
                continue
            tools.append(t)
        return tools

    def tool_count(tool_string: str) -> int:
        return len(parse_tools(tool_string))

    def classify_user_group(tool_string: str) -> str:
        tools = [t.lower() for t in parse_tools(tool_string)]
        if len(tools) == 0:
            return "AI non-user"
        for t in tools:
            if any(key in t for key in IMAGE_GEN_TOOLS):
                return "Image-gen AI user"
        return "Other AI user"

    def holm_adjust(pvals):
        pvals = np.array(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj = np.empty(m, dtype=float)
        for i, p in enumerate(ranked):
            adj[i] = (m - i) * p
        adj = np.maximum.accumulate(adj)
        adj = np.clip(adj, 0, 1)
        out = np.empty(m, dtype=float)
        out[order] = adj
        return out


    # -----------------------------
    # Core analyses per cohort
    # -----------------------------
    def run_20v2_for_cohort(cohort_tag: str, file_path: str):
        print(f"\n==================== [{cohort_tag.upper()}] (20 v2) START ====================")

        cohort_out = os.path.join(OUTPUT_DIR, cohort_tag)
        ensure_dir(cohort_out)

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")

        # detect columns
        acc_col = get_acc_col(df)
        if acc_col is None:
            raise KeyError("No accuracy column found. Expected overallAccuracy or overallAccuracy_y.")
        rt_col = get_rt_col(df)
        if rt_col is None:
            print(f"⚠️ [{cohort_tag}] No RT column found (expected avgRT). ANCOVA parts will be skipped.")
        else:
            print(f"✅ [{cohort_tag}] Using RT column: {rt_col}")
        if "usedAiTools" not in df.columns:
            raise KeyError("Missing column: usedAiTools")
        if "age" not in df.columns:
            raise KeyError("Missing column: age")

        # build working df
        d = df.copy()

        # numeric cleaning
        d["overallAccuracy"] = safe_num(d[acc_col])
        d["age"] = safe_num(d["age"])
        if rt_col:
            d[rt_col] = safe_num(d[rt_col])

        # convert accuracy to percent if looks like proportion (0~1)
        # heuristic: if max <= 1.2, treat as proportion
        mx = d["overallAccuracy"].max(skipna=True)
        if np.isfinite(mx) and mx <= 1.2:
            d["overallAccuracy"] = d["overallAccuracy"] * 100.0
            print(f"✅ [{cohort_tag}] Accuracy looks like proportion (max={mx:.3f}) → converted to %")

        # tool parsing
        d["usedAiTools"] = d["usedAiTools"].apply(normalize_tool_string)
        d["tool_count"] = d["usedAiTools"].apply(tool_count)
        d["user_group"] = d["usedAiTools"].apply(classify_user_group)

        base_cols = ["overallAccuracy", "age", "tool_count", "user_group", "usedAiTools"]
        if rt_col:
            base_cols.append(rt_col)

        d_base = d[base_cols].dropna(subset=["overallAccuracy", "age"]).copy()
        if rt_col:
            d_base = d_base.dropna(subset=[rt_col]).copy()

        print(f"✅ [{cohort_tag}] N after basic cleaning (acc/age{'/rt' if rt_col else ''} present): {len(d_base)}")

        base_data_path = os.path.join(cohort_out, f"20v2-0_base_data_{cohort_tag}.csv")
        d_base.to_csv(base_data_path, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved base data (Prism-ready): {base_data_path}")

        # =========================================================================
        # (20v2-0) Tool count distribution (manual bar; no seaborn version dependency)
        # =========================================================================
        print(f"\n--- (20v2-0) Tool count distribution [{cohort_tag}] ---")
        vc = d_base["tool_count"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(vc.index.astype(int).astype(str), vc.values)
        ax.set_title(f"(20v2-0) Tool count distribution [{cohort_tag.upper()}]", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Number of AI tools used")
        ax.set_ylabel("Count")
        fig.tight_layout()

        fn = os.path.join(cohort_out, f"20v2-0_tool_count_distribution_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # =========================================================================
        # (20v2-1) tool_count vs accuracy (corr + regression with covariates)
        # =========================================================================
        print(f"\n--- (20v2-1) Tool count vs Accuracy [{cohort_tag}] ---")
        df_201 = d_base.dropna(subset=["tool_count", "overallAccuracy"]).copy()
        n_201 = len(df_201)

        scatter_path = None
        coef_path_201 = None
        report_txt_201 = None

        if n_201 < 20:
            print(f"⚠️ [{cohort_tag}] Not enough rows for (20v2-1) (N={n_201} < 20). Skipping.")
        else:
            # correlations (guard constant input)
            try:
                r_p, p_p = pearsonr(df_201["tool_count"], df_201["overallAccuracy"])
            except Exception:
                r_p, p_p = np.nan, np.nan
            try:
                r_s, p_s = spearmanr(df_201["tool_count"], df_201["overallAccuracy"])
            except Exception:
                r_s, p_s = np.nan, np.nan

            print(f"[20v2-1][Correlation] Pearson r={r_p:.3f}, p={p_p:.4g} {console_sig(p_p)}")
            print(f"[20v2-1][Correlation] Spearman rho={r_s:.3f}, p={p_s:.4g} {console_sig(p_s)}")

            # scatter + regression line (raw)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(
                data=df_201, x="tool_count", y="overallAccuracy",
                x_jitter=0.25,
                scatter_kws={"alpha": 0.20, "edgecolor": "none"},
                line_kws={"linewidth": 2},
                ax=ax
            )
            ax.set_title(f"(20v2-1) Tool count vs Accuracy (raw) [{cohort_tag.upper()}]", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Tool count")
            ax.set_ylabel("Accuracy (%)")
            fig.tight_layout()

            fn = os.path.join(cohort_out, f"20v2-1_tool_count_vs_accuracy_raw_{cohort_tag}")
            fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
            fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            # save prism-ready points
            scatter_path = os.path.join(cohort_out, f"20v2-1_scatter_points_{cohort_tag}.csv")
            keep_cols = ["tool_count", "overallAccuracy", "age"] + ([rt_col] if rt_col else [])
            df_201[keep_cols].to_csv(scatter_path, index=False, encoding="utf-8-sig")
            print(f"✅ [{cohort_tag}] Saved scatter raw points: {scatter_path}")

            # regression with covariates
            if rt_col:
                formula_201 = f"overallAccuracy ~ tool_count + age + {rt_col}"
            else:
                formula_201 = "overallAccuracy ~ tool_count + age"

            model_201 = ols(formula_201, data=df_201).fit()
            robust_201 = model_201.get_robustcov_results(cov_type="HC3")

            terms = model_201.model.exog_names
            coef = np.asarray(robust_201.params).reshape(-1)
            se = np.asarray(robust_201.bse).reshape(-1)
            tvals = np.asarray(robust_201.tvalues).reshape(-1)
            pvals = np.asarray(robust_201.pvalues).reshape(-1)

            term_df = pd.DataFrame({"term": terms, "coef": coef, "se_HC3": se, "t_HC3": tvals, "p_HC3": pvals})
            p_tool = float(term_df.loc[term_df["term"] == "tool_count", "p_HC3"].iloc[0]) if "tool_count" in terms else np.nan

            print(f"[20v2-1][Regression HC3] {formula_201}")
            print(f"  - tool_count effect p={p_tool:.4g} {console_sig(p_tool)}")
            print(f"  - R2={model_201.rsquared:.4f}, Adj.R2={model_201.rsquared_adj:.4f}")

            coef_path_201 = os.path.join(cohort_out, f"20v2-1_regression_coeffs_HC3_{cohort_tag}.csv")
            term_df.to_csv(coef_path_201, index=False, encoding="utf-8-sig")

            report = []
            report.append(f"(20v2-1) Tool count vs Accuracy [{cohort_tag}]\n")
            report.append(f"[N] {n_201}\n\n")
            report.append("[Correlations]\n")
            report.append(f"- Pearson r={r_p:.4f}, p={p_p:.6g}\n")
            report.append(f"- Spearman rho={r_s:.4f}, p={p_s:.6g}\n\n")
            report.append("[Regression (HC3 robust SE)]\n")
            report.append(f"- formula: {formula_201}\n")
            report.append(f"- R2={model_201.rsquared:.4f}, Adj.R2={model_201.rsquared_adj:.4f}\n\n")
            report.append(term_df.to_string(index=False))

            report_txt_201 = os.path.join(cohort_out, f"20v2-1_tool_count_report_{cohort_tag}.txt")
            save_text(report_txt_201, "".join(report))

            print(f"✅ [{cohort_tag}] Saved regression coeffs: {coef_path_201}")
            print(f"✅ [{cohort_tag}] Saved report: {report_txt_201}")

        # =========================================================================
        # (20v2-2) tool_type vs accuracy (raw + ANCOVA + adjusted + pairwise)
        # =========================================================================
        print(f"\n--- (20v2-2) Tool type group vs Accuracy [{cohort_tag}] ---")

        df_202 = d_base.dropna(subset=["user_group", "overallAccuracy"]).copy()
        order = ["Image-gen AI user", "Other AI user", "AI non-user"]
        df_202["user_group"] = pd.Categorical(df_202["user_group"], categories=order, ordered=True)

        raw_points_path = os.path.join(cohort_out, f"20v2-2_raw_points_tool_type_{cohort_tag}.csv")
        keep_cols = ["user_group", "overallAccuracy", "age"] + ([rt_col] if rt_col else [])
        df_202[keep_cols].to_csv(raw_points_path, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw points for tool_type plot: {raw_points_path}")

        # raw plot: manual mean+SE bars + points (no seaborn errorbar dependency)
        g = df_202.groupby("user_group", observed=True)["overallAccuracy"]
        g_mean = g.mean().reindex(order)
        g_se = g.sem().reindex(order)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(order, g_mean.values, yerr=g_se.values, capsize=6)
        sns.stripplot(
            data=df_202, x="user_group", y="overallAccuracy",
            order=order, ax=ax, alpha=0.18, jitter=0.25, size=3
        )
        ax.set_title(f"(20v2-2) Accuracy by tool type (raw) [{cohort_tag.upper()}]", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("User group")
        ax.set_ylabel("Accuracy (%)")
        fig.tight_layout()

        fn = os.path.join(cohort_out, f"20v2-2_accuracy_by_tool_type_raw_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        desc = df_202.groupby("user_group", observed=True)["overallAccuracy"].agg(["mean", "std", "count"]).reset_index()
        desc_path = os.path.join(cohort_out, f"20v2-2_raw_descriptives_{cohort_tag}.csv")
        desc.to_csv(desc_path, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw descriptives: {desc_path}")

        if not rt_col:
            print(f"⚠️ [{cohort_tag}] RT column missing → skipping ANCOVA/adjusted plots for (20v2-2).")
            print(f"==================== [{cohort_tag.upper()}] (20 v2) END ====================\n")
            return

        # ANCOVA (HC3)
        formula_202 = f"overallAccuracy ~ C(user_group) + age + {rt_col}"
        model_202 = ols(formula_202, data=df_202).fit()
        robust_202 = model_202.get_robustcov_results(cov_type="HC3")

        exog_names = model_202.model.exog_names
        group_terms = [t for t in exog_names if t.startswith("C(user_group)[T.")]
        if len(group_terms) == 0:
            wald_stat, wald_df, wald_p = np.nan, 0, np.nan
            print(f"⚠️ [{cohort_tag}] Could not find group dummy terms for Wald test.")
        else:
            R = np.zeros((len(group_terms), len(exog_names)))
            for i, term in enumerate(group_terms):
                R[i, exog_names.index(term)] = 1.0
            w = robust_202.wald_test(R, scalar=True)
            wald_stat = as_scalar(w.statistic)
            wald_df = len(group_terms)
            wald_p = as_scalar(w.pvalue)

        print(f"[20v2-2][ANCOVA HC3] {formula_202}")
        print(f"  - Robust Wald group main effect: stat={wald_stat:.3f}, df={wald_df}, p={wald_p:.4g} {console_sig(wald_p)}")
        print(f"  - R2={model_202.rsquared:.4f}, Adj.R2={model_202.rsquared_adj:.4f}")

        coef = np.asarray(robust_202.params).reshape(-1)
        se = np.asarray(robust_202.bse).reshape(-1)
        tvals = np.asarray(robust_202.tvalues).reshape(-1)
        pvals = np.asarray(robust_202.pvalues).reshape(-1)
        coef_df = pd.DataFrame({"term": exog_names, "coef": coef, "se_HC3": se, "t_HC3": tvals, "p_HC3": pvals})
        coef_path_202 = os.path.join(cohort_out, f"20v2-2_ancova_coeffs_HC3_{cohort_tag}.csv")
        coef_df.to_csv(coef_path_202, index=False, encoding="utf-8-sig")

        # adjusted means at mean covariates
        age_mean = float(df_202["age"].mean())
        rt_mean = float(df_202[rt_col].mean())

        pred_df = pd.DataFrame([{"user_group": g, "age": age_mean, rt_col: rt_mean} for g in order])
        pred = model_202.get_prediction(pred_df).summary_frame(alpha=0.05)

        adj = pred_df.copy()
        adj["adj_mean"] = pred["mean"].values
        adj["ci_low"] = pred["mean_ci_lower"].values
        adj["ci_high"] = pred["mean_ci_upper"].values

        adj_path = os.path.join(cohort_out, f"20v2-2_adjusted_means_{cohort_tag}.csv")
        adj.to_csv(adj_path, index=False, encoding="utf-8-sig")

        print(f"[20v2-2][Adjusted means] covariates at mean: age={age_mean:.2f}, {rt_col}={rt_mean:.2f}")
        print(adj[["user_group", "adj_mean", "ci_low", "ci_high"]].to_string(index=False))

        # adjusted plot
        fig, ax = plt.subplots(figsize=(10, 6))
        yerr = np.vstack([adj["adj_mean"] - adj["ci_low"], adj["ci_high"] - adj["adj_mean"]])
        ax.bar(adj["user_group"], adj["adj_mean"], yerr=yerr, capsize=6)
        ax.set_title(f"(20v2-2) Accuracy by tool type (adjusted) [{cohort_tag.upper()}]\n(covariates at mean)", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("User group")
        ax.set_ylabel("Adjusted Accuracy (%)")
        ax.set_ylim(0, 100)
        fig.tight_layout()

        fn = os.path.join(cohort_out, f"20v2-2_accuracy_by_tool_type_adjusted_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # pairwise contrasts (HC3 + Holm)
        idx = {t: i for i, t in enumerate(exog_names)}
        baseline = order[0]  # first category as baseline in patsy with ordered categorical

        def contrast_vec(gA, gB):
            v = np.zeros(len(exog_names))
            def add_group(level, sign):
                if level == baseline:
                    return
                term = f"C(user_group)[T.{level}]"
                if term in idx:
                    v[idx[term]] += sign
            add_group(gB, +1.0)
            add_group(gA, -1.0)
            return v

        pairs = []
        pvals_pairs = []
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                A, B = order[i], order[j]
                v = contrast_vec(A, B)
                tt = robust_202.t_test(v)
                diff = as_scalar(tt.effect)
                p = as_scalar(tt.pvalue)
                pairs.append((A, B, diff, p))
                pvals_pairs.append(p)

        p_holm = holm_adjust(pvals_pairs)
        pair_df = pd.DataFrame(pairs, columns=["A", "B", "diff_(B-A)", "p_value"])
        pair_df["p_holm"] = p_holm
        pair_df["significant_holm"] = pair_df["p_holm"] < ALPHA

        pair_path = os.path.join(cohort_out, f"20v2-2_pairwise_contrasts_HC3_{cohort_tag}.csv")
        pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")

        print("\n[20v2-2][Pairwise contrasts on adjusted means] (HC3 + Holm)")
        print(pair_df.to_string(index=False))

        # combined report
        report = []
        report.append(f"(20 v2) Tool usage & Accuracy [{cohort_tag}]\n")
        report.append("============================================================\n\n")
        report.append("[Files saved]\n")
        report.append(f"- base data: {base_data_path}\n")
        report.append(f"- scatter points (20v2-1): {scatter_path if n_201 >= 20 else 'SKIPPED'}\n")
        report.append(f"- tool_type raw points (20v2-2): {raw_points_path}\n")
        report.append(f"- raw descriptives: {desc_path}\n")
        report.append(f"- ANCOVA coeffs (20v2-2): {coef_path_202}\n")
        report.append(f"- adjusted means: {adj_path}\n")
        report.append(f"- pairwise contrasts: {pair_path}\n\n")

        report.append("[20v2-1] tool_count vs accuracy\n")
        if n_201 >= 20:
            report.append(f"- regression formula: {('overallAccuracy ~ tool_count + age + ' + rt_col) if rt_col else 'overallAccuracy ~ tool_count + age'}\n")
        else:
            report.append("- skipped (N<20)\n")

        report.append("\n[20v2-2] tool_type group vs accuracy (ANCOVA HC3)\n")
        report.append(f"- formula: {formula_202}\n")
        report.append(f"- Robust Wald group main effect: stat={wald_stat:.4f}, df={wald_df}, p={wald_p:.6g}\n")
        report.append(f"- R2={model_202.rsquared:.4f}, Adj.R2={model_202.rsquared_adj:.4f}\n\n")
        report.append("[Adjusted means]\n")
        report.append(adj.to_string(index=False))
        report.append("\n\n[Pairwise contrasts (HC3 + Holm)]\n")
        report.append(pair_df.to_string(index=False))
        report.append("\n")

        report_txt = os.path.join(cohort_out, f"20v2_report_{cohort_tag}.txt")
        save_text(report_txt, "".join(report))
        print(f"✅ [{cohort_tag}] Saved report: {report_txt}")

        print(f"==================== [{cohort_tag.upper()}] (20 v2) END ====================\n")


    # -----------------------------
    # Run all cohorts
    # -----------------------------
    print("==============================================================================")
    print("(20 v2) AI Tool Usage & Accuracy with Covariates (MOBILE + WEB) [ENGLISH]")
    print("==============================================================================\n")

    ensure_dir(OUTPUT_DIR)

    for cohort_tag, fp in COHORT_FILES.items():
        if not os.path.exists(fp):
            print(f"❌ Missing file for {cohort_tag}: {fp}")
            continue
        run_20v2_for_cohort(cohort_tag, fp)

    print("\n==================== (20 v2) DONE ====================")


def _run_cell_093():
    # ==============================================================================
    # (21 v2.2) Learning effect: Practice vs Main (PAIRED by participantId)
    # ------------------------------------------------------------------------------
    # MOBILE + WEB
    # - Auto-detect MAIN accuracy column (overallAccuracy / overallAccuracy_y / ...)
    # - Pair within-subject using participantId (1 participant = 1 pair)
    # - Scale BOTH practice & main to percent if they look like proportions (<=1)
    # - All plots enforce axes 0~100
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import ttest_rel, ttest_1samp, wilcoxon, t
    from math import sqrt

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_21_learning_v2_2"
    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    ID_COL = "participantId"
    PRACTICE_COL = "practice_accuracy"

    # ✅ MAIN accuracy column candidates (auto-pick first found)
    MAIN_COL_CANDIDATES = [
        "overallAccuracy", "overallAccuracy_y", "overallAccuracy_x",
        "main_accuracy", "mainAccuracy", "accuracy_main"
    ]

    ALPHA = 0.05

    # If participantId duplicates exist:
    #   "last": last row per participant (good if rows are already in time order)
    #   "mean": average per participant
    #   "max" : best per participant
    AGG_METHOD = "last"  # change to "mean" or "max" if you want

    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()


    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def to_numeric(s):
        return pd.to_numeric(s, errors="coerce")

    def maybe_scale_to_percent(series: pd.Series) -> pd.Series:
        """
        If values look like proportion (<=1), scale to percent.
        """
        x = series.copy()
        mx = x.max(skipna=True)
        if pd.notna(mx) and mx <= 1.0:
            return x * 100.0
        return x

    def cohens_dz(diff: np.ndarray) -> float:
        diff = np.asarray(diff, dtype=float)
        sd = diff.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        return diff.mean() / sd

    def mean_diff_ci(diff: np.ndarray, alpha=0.05):
        diff = np.asarray(diff, dtype=float)
        n = len(diff)
        m = diff.mean()
        sd = diff.std(ddof=1)
        if n < 2 or sd == 0 or not np.isfinite(sd):
            return (m, np.nan, np.nan)
        se = sd / sqrt(n)
        crit = t.ppf(1 - alpha/2, df=n-1)
        lo = m - crit * se
        hi = m + crit * se
        return (m, lo, hi)

    def sig_mark(p, alpha=ALPHA):
        return "✅" if (p < alpha) else "❌"

    def pick_main_col(df: pd.DataFrame):
        """
        Return the first available MAIN accuracy column name from candidates.
        """
        for c in MAIN_COL_CANDIDATES:
            if c in df.columns:
                return c
        return None


    # -----------------------------
    # Core
    # -----------------------------
    def run_21v2_2_for_cohort(cohort_tag: str, file_path: str):
        print(f"\n==================== [{cohort_tag.upper()}] (21 v2.2) START ====================")

        out_dir = os.path.join(OUTPUT_DIR, cohort_tag)
        ensure_dir(out_dir)

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")

        main_col = pick_main_col(df)
        if main_col is None:
            print(f"❌ [{cohort_tag}] Missing MAIN accuracy column. Tried: {MAIN_COL_CANDIDATES}")
            print(f"   Available columns (first 30): {list(df.columns)[:30]}")
            return
        print(f"✅ [{cohort_tag}] Using MAIN accuracy column: {main_col}")

        # check required columns
        need = [ID_COL, PRACTICE_COL, main_col]
        missing = [c for c in need if c not in df.columns]
        if missing:
            print(f"❌ [{cohort_tag}] Missing required columns: {missing}")
            return

        d = df[[ID_COL, PRACTICE_COL, main_col]].copy()
        d[PRACTICE_COL] = to_numeric(d[PRACTICE_COL])
        d[main_col] = to_numeric(d[main_col])

        # scale BOTH if needed
        d[PRACTICE_COL] = maybe_scale_to_percent(d[PRACTICE_COL])
        d[main_col] = maybe_scale_to_percent(d[main_col])

        # clamp
        d[PRACTICE_COL] = d[PRACTICE_COL].clip(0, 100)
        d[main_col] = d[main_col].clip(0, 100)

        # duplicates info
        dup_rows = int(d[ID_COL].duplicated().sum())
        n_unique = int(d[ID_COL].nunique())
        print(f"✅ [{cohort_tag}] Unique participantId: {n_unique} | duplicated rows: {dup_rows}")

        # Pair by participantId (aggregate)
        if AGG_METHOD not in {"last", "mean", "max"}:
            raise ValueError("AGG_METHOD must be one of: 'last', 'mean', 'max'")

        if AGG_METHOD == "last":
            paired = d.groupby(ID_COL, as_index=False).agg({PRACTICE_COL: "last", main_col: "last"})
        elif AGG_METHOD == "mean":
            paired = d.groupby(ID_COL, as_index=False).agg({PRACTICE_COL: "mean", main_col: "mean"})
        else:
            paired = d.groupby(ID_COL, as_index=False).agg({PRACTICE_COL: "max", main_col: "max"})

        paired = paired.dropna(subset=[PRACTICE_COL, main_col]).copy()
        n_paired = len(paired)
        if n_paired < 20:
            print(f"⚠️ [{cohort_tag}] Not enough paired data (N={n_paired} < 20). Skipping.")
            return

        practice = paired[PRACTICE_COL].to_numpy()
        main = paired[main_col].to_numpy()
        diff = main - practice

        improved = int((diff > 0).sum())
        declined = int((diff < 0).sum())
        same = int((diff == 0).sum())

        # paired t-test
        t_stat_2s, p_2s = ttest_rel(main, practice, nan_policy="omit")

        # one-sided: mean(diff) > 0
        t_stat_diff, p_two = ttest_1samp(diff, popmean=0.0, nan_policy="omit")
        p_1s = p_two / 2 if t_stat_diff > 0 else 1 - (p_two / 2)

        dz = cohens_dz(diff)
        m_diff, ci_lo, ci_hi = mean_diff_ci(diff, alpha=0.05)

        # Wilcoxon two-sided + greater
        try:
            w2, p_w2 = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
        except Exception as e:
            w2, p_w2 = np.nan, np.nan
            print(f"⚠️ [{cohort_tag}] Wilcoxon two-sided failed: {e}")

        try:
            wg, p_wg = wilcoxon(diff, zero_method="wilcox", alternative="greater", mode="auto")
        except Exception as e:
            wg, p_wg = np.nan, np.nan
            print(f"⚠️ [{cohort_tag}] Wilcoxon one-sided failed: {e}")

        # Console summary
        print(f"✅ [{cohort_tag}] N paired used: {n_paired} (AGG_METHOD={AGG_METHOD})")
        print(f"[21v2.2] Means: practice={practice.mean():.2f}%, main={main.mean():.2f}%")
        print(f"[21v2.2] Mean diff (main-practice) = {m_diff:.2f} pp | 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"[21v2.2] Paired t-test (two-sided): t={t_stat_2s:.3f}, p={p_2s:.6g} {sig_mark(p_2s)}")
        print(f"[21v2.2] One-sided (main>practice): t={t_stat_diff:.3f}, p={p_1s:.6g} {sig_mark(p_1s)}")
        print(f"[21v2.2] Effect size: Cohen's dz = {dz:.3f}")
        if np.isfinite(p_w2):
            print(f"[21v2.2] Wilcoxon (two-sided): W={w2:.3f}, p={p_w2:.6g} {sig_mark(p_w2)}")
        if np.isfinite(p_wg):
            print(f"[21v2.2] Wilcoxon (greater):   W={wg:.3f}, p={p_wg:.6g} {sig_mark(p_wg)}")

        print(f"[21v2.2] Change counts: improved={improved} ({improved/n_paired:.1%}), "
              f"declined={declined} ({declined/n_paired:.1%}), same={same} ({same/n_paired:.1%})")

        # Save Prism tables
        paired_out = paired.rename(columns={main_col: "main_accuracy"}).copy()
        paired_out["diff_main_minus_practice"] = diff

        long_path = os.path.join(out_dir, f"21v2-0_paired_by_id_{cohort_tag}.csv")
        paired_out.to_csv(long_path, index=False, encoding="utf-8-sig")

        diff_path = os.path.join(out_dir, f"21v2-0_diff_only_{cohort_tag}.csv")
        pd.DataFrame({"diff_main_minus_practice": diff}).to_csv(diff_path, index=False, encoding="utf-8-sig")

        print(f"✅ [{cohort_tag}] Saved Prism tables: {long_path}, {diff_path}")

        # -----------------------------
        # Plots (axes 0~100)
        # -----------------------------
        # Scatter practice vs main
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(practice, main, alpha=0.25)
        ax.plot([0, 100], [0, 100], linestyle="--", linewidth=2, label="y=x (no change)")
        ax.set_title(f"(21v2.2-1) Practice vs Main Accuracy [{cohort_tag.upper()}]", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Practice accuracy (%)")
        ax.set_ylabel("Main accuracy (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right")
        fig.tight_layout()

        fn = os.path.join(out_dir, f"21v2-1_scatter_practice_vs_main_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # Paired lines + mean±SE
        fig, ax = plt.subplots(figsize=(8, 7))
        for i in range(n_paired):
            ax.plot(["Practice", "Main"], [practice[i], main[i]], alpha=0.08, linewidth=1)

        means = [practice.mean(), main.mean()]
        ses = [practice.std(ddof=1)/np.sqrt(n_paired), main.std(ddof=1)/np.sqrt(n_paired)]
        ax.errorbar(["Practice", "Main"], means, yerr=ses, capsize=6, linewidth=3)

        ax.set_title(f"(21v2.2-2) Paired change (Practice → Main) [{cohort_tag.upper()}]", fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        fig.tight_layout()

        fn = os.path.join(out_dir, f"21v2-2_paired_lines_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # Diff histogram
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(diff, bins=30, alpha=0.9)
        ax.axvline(0, linestyle="--", linewidth=2)
        ax.set_title(f"(21v2.2-3) Improvement distribution (Main - Practice) [{cohort_tag.upper()}]", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Improvement (percentage points)")
        ax.set_ylabel("Count")
        fig.tight_layout()

        fn = os.path.join(out_dir, f"21v2-3_diff_hist_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # Report
        report = []
        report.append("==============================================================================\n")
        report.append(f"(21 v2.2) Learning effect: Practice vs Main [{cohort_tag}]\n")
        report.append("==============================================================================\n\n")
        report.append(f"[Columns]\n- ID: {ID_COL}\n- Practice: {PRACTICE_COL}\n- Main used: {main_col}\n\n")
        report.append(f"[Pairing]\n- Aggregation per ID: {AGG_METHOD}\n- Unique IDs: {n_unique}\n- Duplicated rows: {dup_rows}\n- N paired used: {n_paired}\n\n")
        report.append("[Descriptives]\n")
        report.append(f"- Practice mean = {practice.mean():.6f}%\n")
        report.append(f"- Main mean     = {main.mean():.6f}%\n")
        report.append(f"- Mean diff (Main-Practice) = {m_diff:.6f} pp\n")
        report.append(f"- 95% CI for mean diff = [{ci_lo:.6f}, {ci_hi:.6f}]\n\n")
        report.append("[Paired t-test]\n")
        report.append(f"- Two-sided: t = {t_stat_2s:.6f}, p = {p_2s:.6g}\n")
        report.append(f"- One-sided (Main>Practice): t = {t_stat_diff:.6f}, p = {p_1s:.6g}\n")
        report.append(f"- Cohen's dz = {dz:.6f}\n\n")
        report.append("[Wilcoxon signed-rank]\n")
        report.append(f"- Two-sided: W = {w2}, p = {p_w2}\n")
        report.append(f"- Greater (Main>Practice): W = {wg}, p = {p_wg}\n\n")
        report.append("[Change counts]\n")
        report.append(f"- improved: {improved} ({improved/n_paired:.2%})\n")
        report.append(f"- declined: {declined} ({declined/n_paired:.2%})\n")
        report.append(f"- same    : {same} ({same/n_paired:.2%})\n\n")
        report.append("[Saved files]\n")
        report.append(f"- Prism paired by ID: {long_path}\n")
        report.append(f"- Prism diff only   : {diff_path}\n")
        report.append(f"- Plots: 21v2-1, 21v2-2, 21v2-3 (png/svg)\n")

        report_path = os.path.join(out_dir, f"21v2-0_learning_report_{cohort_tag}.txt")
        save_text(report_path, "".join(report))
        print(f"✅ [{cohort_tag}] Saved report: {report_path}")

        print(f"==================== [{cohort_tag.upper()}] (21 v2.2) END ====================\n")


    # -----------------------------
    # Run
    # -----------------------------
    print("==============================================================================")
    print("(21 v2.2) Learning effect: Practice vs Main (PAIRED by participantId) (MOBILE + WEB)")
    print("==============================================================================\n")

    ensure_dir(OUTPUT_DIR)

    for cohort_tag, fp in COHORT_FILES.items():
        if not os.path.exists(fp):
            print(f"❌ Missing file for {cohort_tag}: {fp}")
            continue
        run_21v2_2_for_cohort(cohort_tag, fp)

    print("\n==================== (21 v2.2) DONE ====================")


def _run_cell_096():
    # ==============================================================================
    # (22 v2.1) Device effect (Mobile vs Web): Accuracy & AvgRT (Welch + Regression)
    # ------------------------------------------------------------------------------
    # Fixes:
    #  - Auto-detect accuracy column: overallAccuracy / overallAccuracy_y / overallAccuracy_x
    #  - If accuracy looks like proportion (max<=1), convert to %
    #  - Accuracy plots y-lim fixed to 0~100
    #  - Robust (HC3) regression models
    #  - Prism-ready raw table + summary tables + plots (png/svg)
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import ttest_ind
    import statsmodels.formula.api as smf

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_22_device_v2_1"
    FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    ACC_COL_CANDIDATES = ["overallAccuracy", "overallAccuracy_y", "overallAccuracy_x"]
    RT_COL  = "avgRT"
    AGE_COL = "age"
    ID_COL_CANDIDATES = ["participantId", "ParticipantId", "participant_id", "id"]

    ALPHA = 0.05

    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()


    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): 
        os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def to_num(s): 
        return pd.to_numeric(s, errors="coerce")

    def find_first_existing(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def find_id_col(df):
        return find_first_existing(df, ID_COL_CANDIDATES)

    def maybe_accuracy_to_percent(series: pd.Series) -> pd.Series:
        """
        If accuracy looks like proportion (max<=1), convert to %.
        """
        s = series.copy()
        mx = s.max(skipna=True)
        if pd.notna(mx) and mx <= 1.0:
            print(f"✅ Accuracy looks like proportion (max={mx:.3f}) → converted to %")
            s = s * 100.0
        return s

    def hedges_g(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        if sp2 <= 0 or not np.isfinite(sp2):
            return np.nan
        d = (x.mean() - y.mean()) / np.sqrt(sp2)
        df = nx + ny - 2
        J = 1 - (3 / (4 * df - 1))
        return J * d

    def fmt_p(p):
        if p < 0.001: return f"{p:.2e}"
        return f"{p:.6f}"

    def sig_mark(p, alpha=ALPHA):
        return "✅" if (p < alpha) else "❌"

    def describe_group(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        return {
            "n": int(n),
            "mean": float(np.mean(x)) if n else np.nan,
            "sd": float(np.std(x, ddof=1)) if n > 1 else np.nan,
            "sem": float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        }


    # -----------------------------
    # Load & concat
    # -----------------------------
    def load_cohorts():
        frames = []
        meta = {}
        for device, fp in FILES.items():
            if not os.path.exists(fp):
                print(f"❌ Missing file: {fp}")
                continue
            df = pd.read_csv(fp, encoding="utf-8-sig").copy()
            df["deviceType"] = device

            acc_col = find_first_existing(df, ACC_COL_CANDIDATES)
            if acc_col is None:
                raise KeyError(f"[{device}] No accuracy column found among {ACC_COL_CANDIDATES}")

            # numeric coercion
            df[acc_col] = to_num(df[acc_col])
            df[AGE_COL] = to_num(df[AGE_COL]) if AGE_COL in df.columns else np.nan
            df[RT_COL]  = to_num(df[RT_COL])  if RT_COL in df.columns else np.nan

            # unify column name to "accuracy"
            df["accuracy"] = df[acc_col]
            df["accuracy"] = maybe_accuracy_to_percent(df["accuracy"])
            df["accuracy"] = df["accuracy"].clip(0, 100)

            frames.append(df)
            meta[device] = {"file": fp, "acc_col_used": acc_col, "rows": len(df)}
            print(f"✅ Loaded: {fp} [{device}] rows={len(df)} | accuracy_col={acc_col}")

        if not frames:
            raise FileNotFoundError("No cohort files loaded.")
        return pd.concat(frames, ignore_index=True), meta


    # -----------------------------
    # Main analysis
    # -----------------------------
    def run_22v21():
        ensure_dir(OUTPUT_DIR)

        df, meta = load_cohorts()
        id_col = find_id_col(df)

        keep_cols = ["deviceType", "accuracy", RT_COL, AGE_COL]
        if id_col:
            keep_cols = [id_col] + keep_cols

        df0 = df[keep_cols].copy()

        # Save Prism-friendly raw table
        raw_path = os.path.join(OUTPUT_DIR, "22v2-0_raw_table_device_accuracy_rt.csv")
        df0.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved raw table (Prism-friendly): {raw_path}")

        # -------------------------
        # (22v2-1) Welch t-tests
        # -------------------------
        print("\n--- (22v2-1) Welch t-tests (Mobile vs Web) ---")

        # Accuracy
        df_acc = df0.dropna(subset=["deviceType", "accuracy"])
        mob_acc = df_acc[df_acc["deviceType"] == "mobile"]["accuracy"].values
        web_acc = df_acc[df_acc["deviceType"] == "web"]["accuracy"].values
        t_acc, p_acc = ttest_ind(mob_acc, web_acc, equal_var=False, nan_policy="omit")
        g_acc = hedges_g(mob_acc, web_acc)

        # RT
        df_rt = df0.dropna(subset=["deviceType", RT_COL])
        mob_rt = df_rt[df_rt["deviceType"] == "mobile"][RT_COL].values
        web_rt = df_rt[df_rt["deviceType"] == "web"][RT_COL].values
        t_rt, p_rt = ttest_ind(mob_rt, web_rt, equal_var=False, nan_policy="omit")
        g_rt = hedges_g(mob_rt, web_rt)

        desc_acc_m = describe_group(mob_acc)
        desc_acc_w = describe_group(web_acc)
        desc_rt_m  = describe_group(mob_rt)
        desc_rt_w  = describe_group(web_rt)

        print(f"[Accuracy] mobile n={desc_acc_m['n']} mean={desc_acc_m['mean']:.3f} sd={desc_acc_m['sd']:.3f} | "
              f"web n={desc_acc_w['n']} mean={desc_acc_w['mean']:.3f} sd={desc_acc_w['sd']:.3f}")
        print(f"  Welch t={t_acc:.3f}, p={fmt_p(p_acc)} {sig_mark(p_acc)} | Hedges g={g_acc:.3f}")

        print(f"[AvgRT]    mobile n={desc_rt_m['n']} mean={desc_rt_m['mean']:.3f} sd={desc_rt_m['sd']:.3f} | "
              f"web n={desc_rt_w['n']} mean={desc_rt_w['mean']:.3f} sd={desc_rt_w['sd']:.3f}")
        print(f"  Welch t={t_rt:.3f}, p={fmt_p(p_rt)} {sig_mark(p_rt)} | Hedges g={g_rt:.3f}")

        test_df = pd.DataFrame([
            {"outcome": "Accuracy", "t_welch": t_acc, "p": p_acc, "sig_p<0.05": p_acc < ALPHA, "hedges_g": g_acc,
             "mobile_n": desc_acc_m["n"], "mobile_mean": desc_acc_m["mean"], "web_n": desc_acc_w["n"], "web_mean": desc_acc_w["mean"]},
            {"outcome": "AvgRT", "t_welch": t_rt, "p": p_rt, "sig_p<0.05": p_rt < ALPHA, "hedges_g": g_rt,
             "mobile_n": desc_rt_m["n"], "mobile_mean": desc_rt_m["mean"], "web_n": desc_rt_w["n"], "web_mean": desc_rt_w["mean"]},
        ])
        test_path = os.path.join(OUTPUT_DIR, "22v2-1_welch_tests.csv")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved Welch test table: {test_path}")

        report = []
        report.append("==============================================================================\n")
        report.append("(22 v2.1) Device effect (Mobile vs Web): Accuracy & AvgRT\n")
        report.append("==============================================================================\n\n")
        report.append("[Files]\n")
        for dev, info in meta.items():
            report.append(f"- {dev}: {info['file']} | acc_col={info['acc_col_used']} | rows={info['rows']}\n")
        report.append("\n[Welch t-tests]\n")
        report.append(f"- Accuracy: t={t_acc:.6f}, p={fmt_p(p_acc)}, Hedges g={g_acc:.6f}\n")
        report.append(f"  mobile (n={desc_acc_m['n']}): mean={desc_acc_m['mean']:.3f}, sd={desc_acc_m['sd']:.3f}\n")
        report.append(f"  web    (n={desc_acc_w['n']}): mean={desc_acc_w['mean']:.3f}, sd={desc_acc_w['sd']:.3f}\n\n")
        report.append(f"- AvgRT: t={t_rt:.6f}, p={fmt_p(p_rt)}, Hedges g={g_rt:.6f}\n")
        report.append(f"  mobile (n={desc_rt_m['n']}): mean={desc_rt_m['mean']:.3f}, sd={desc_rt_m['sd']:.3f}\n")
        report.append(f"  web    (n={desc_rt_w['n']}): mean={desc_rt_w['mean']:.3f}, sd={desc_rt_w['sd']:.3f}\n\n")
        report_path = os.path.join(OUTPUT_DIR, "22v2-1_device_welch_report.txt")
        save_text(report_path, "".join(report))
        print(f"✅ Saved report: {report_path}")

        # -------------------------
        # (22v2-2) Regression (HC3)
        # -------------------------
        print("\n--- (22v2-2) Regression (HC3) ---")

        # Accuracy model A: accuracy ~ device + age
        dfA = df0.dropna(subset=["deviceType", "accuracy", AGE_COL]).copy()
        if len(dfA) >= 50:
            mA = smf.ols(f"accuracy ~ C(deviceType) + {AGE_COL}", data=dfA).fit(cov_type="HC3")
            p_dev_A = mA.pvalues.get("C(deviceType)[T.web]", np.nan)
            print(f"[Accuracy model A] accuracy ~ device + age (HC3)")
            print(f"  device(web vs mobile) p={p_dev_A:.6g} {sig_mark(p_dev_A)}")
            print(f"  R2={mA.rsquared:.4f}, Adj.R2={mA.rsquared_adj:.4f}")

            coefA = pd.DataFrame({
                "term": mA.params.index,
                "coef": mA.params.values,
                "se_HC3": mA.bse.values,
                "t": mA.tvalues.values,
                "p": mA.pvalues.values
            })
            coefA_path = os.path.join(OUTPUT_DIR, "22v2-2_accuracy_modelA_coeffs_HC3.csv")
            coefA.to_csv(coefA_path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved model A coeffs: {coefA_path}")
        else:
            mA = None
            print("⚠️ Not enough rows for Accuracy model A.")

        # Accuracy model B: accuracy ~ device + age + avgRT
        dfB = df0.dropna(subset=["deviceType", "accuracy", AGE_COL, RT_COL]).copy()
        if len(dfB) >= 50:
            mB = smf.ols(f"accuracy ~ C(deviceType) + {AGE_COL} + {RT_COL}", data=dfB).fit(cov_type="HC3")
            p_dev_B = mB.pvalues.get("C(deviceType)[T.web]", np.nan)
            print(f"[Accuracy model B] accuracy ~ device + age + avgRT (HC3)")
            print(f"  device(web vs mobile) p={p_dev_B:.6g} {sig_mark(p_dev_B)}")
            print(f"  R2={mB.rsquared:.4f}, Adj.R2={mB.rsquared_adj:.4f}")

            coefB = pd.DataFrame({
                "term": mB.params.index,
                "coef": mB.params.values,
                "se_HC3": mB.bse.values,
                "t": mB.tvalues.values,
                "p": mB.pvalues.values
            })
            coefB_path = os.path.join(OUTPUT_DIR, "22v2-2_accuracy_modelB_coeffs_HC3.csv")
            coefB.to_csv(coefB_path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved model B coeffs: {coefB_path}")
        else:
            mB = None
            print("⚠️ Not enough rows for Accuracy model B.")

        # RT model: avgRT ~ device + age
        dfRT = df0.dropna(subset=["deviceType", RT_COL, AGE_COL]).copy()
        if len(dfRT) >= 50:
            mRT = smf.ols(f"{RT_COL} ~ C(deviceType) + {AGE_COL}", data=dfRT).fit(cov_type="HC3")
            p_dev_RT = mRT.pvalues.get("C(deviceType)[T.web]", np.nan)
            print(f"[RT model] avgRT ~ device + age (HC3)")
            print(f"  device(web vs mobile) p={p_dev_RT:.6g} {sig_mark(p_dev_RT)}")
            print(f"  R2={mRT.rsquared:.4f}, Adj.R2={mRT.rsquared_adj:.4f}")

            coefRT = pd.DataFrame({
                "term": mRT.params.index,
                "coef": mRT.params.values,
                "se_HC3": mRT.bse.values,
                "t": mRT.tvalues.values,
                "p": mRT.pvalues.values
            })
            coefRT_path = os.path.join(OUTPUT_DIR, "22v2-2_rt_model_coeffs_HC3.csv")
            coefRT.to_csv(coefRT_path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved RT model coeffs: {coefRT_path}")
        else:
            mRT = None
            print("⚠️ Not enough rows for RT model.")

        # save regression report text
        reg_report = []
        reg_report.append("==============================================================================\n")
        reg_report.append("(22 v2.1) Regression/ANCOVA-style models (HC3)\n")
        reg_report.append("==============================================================================\n\n")

        def add_model_block(name, model):
            reg_report.append(f"[{name}]\n")
            if model is None:
                reg_report.append("  (skipped)\n\n")
                return
            reg_report.append(f"  R2={model.rsquared:.6f}, Adj.R2={model.rsquared_adj:.6f}\n")
            tmp = pd.DataFrame({
                "coef": model.params,
                "se_HC3": model.bse,
                "t": model.tvalues,
                "p": model.pvalues
            })
            reg_report.append(tmp.to_string() + "\n\n")

        add_model_block("Accuracy model A: accuracy ~ device + age", mA)
        add_model_block("Accuracy model B: accuracy ~ device + age + avgRT", mB)
        add_model_block("RT model: avgRT ~ device + age", mRT)

        reg_report_path = os.path.join(OUTPUT_DIR, "22v2-2_regression_report.txt")
        save_text(reg_report_path, "".join(reg_report))
        print(f"✅ Saved regression report: {reg_report_path}")

        # -------------------------
        # (22v2-3) Plots
        # -------------------------
        print("\n--- (22v2-3) Plots (mean+SEM + points) ---")

        # Accuracy plot (0~100 fixed)
        df_plot_acc = df0.dropna(subset=["deviceType", "accuracy"]).copy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df_plot_acc, x="deviceType", y="accuracy", order=["mobile", "web"],
                    errorbar="se", capsize=0.12, ax=ax)
        sns.stripplot(data=df_plot_acc, x="deviceType", y="accuracy", order=["mobile", "web"],
                      alpha=0.25, jitter=0.25, ax=ax)

        ax.set_title(f"(22v2-3A) Accuracy by Device (Welch p={fmt_p(p_acc)})", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Device")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.text(0.5, 0.98, f"{sig_mark(p_acc)}", transform=ax.transAxes, ha="center", va="top", fontsize=16)

        fn = os.path.join(OUTPUT_DIR, "22v2-3A_accuracy_by_device")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved: {fn}.png/.svg")

        # RT plot (use 95% quantile upper limit for readability)
        df_plot_rt = df0.dropna(subset=["deviceType", RT_COL]).copy()
        rt_upper = df_plot_rt[RT_COL].quantile(0.95)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df_plot_rt, x="deviceType", y=RT_COL, order=["mobile", "web"],
                    errorbar="se", capsize=0.12, ax=ax)
        sns.stripplot(data=df_plot_rt, x="deviceType", y=RT_COL, order=["mobile", "web"],
                      alpha=0.25, jitter=0.25, ax=ax)

        ax.set_title(f"(22v2-3B) AvgRT by Device (Welch p={fmt_p(p_rt)})", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Device")
        ax.set_ylabel("AvgRT (ms)")
        ax.set_ylim(0, rt_upper)
        ax.text(0.5, 0.98, f"{sig_mark(p_rt)}", transform=ax.transAxes, ha="center", va="top", fontsize=16)

        fn = os.path.join(OUTPUT_DIR, "22v2-3B_avgRT_by_device")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved: {fn}.png/.svg")

        print("\n==================== (22 v2.1) DONE ====================")


    if __name__ == "__main__":
        print("==============================================================================")
        print("(22 v2.1) Device effect (Mobile vs Web): Accuracy & AvgRT [ENGLISH]")
        print("==============================================================================\n")
        run_22v21()


def _run_cell_099():
    # ==============================================================================
    # (23 v2.1) RT Analysis: Overall trend + Condition-specific detail (MOBILE + WEB)
    # - Sex/Gender column auto-detect (standardize to 'sex')
    # - Safe condition column names (Correct_AI, ...)
    # - Prism-friendly raw + summary(mean/sem) tables saved
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_23_rt_v2_1"

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    RESPONSES_FILE = config.RAW_RESPONSES  # optional

    RT_COL_CANDIDATES = ["avgRT", "mean_rt", "avg_rt", "MeanRT", "meanRT"]
    AGE_COL = "age"

    # auto-detect sex/gender col, then standardize to 'sex'
    SEX_COL_CANDIDATES = ["sex", "Sex", "gender", "Gender", "GENDER", "SEX"]

    AGE_BINS   = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    ALPHA = 0.05

    # plots
    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()


    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def to_num(s): return pd.to_numeric(s, errors="coerce")

    def find_id_col(df: pd.DataFrame):
        for c in ["participantId", "ParticipantId", "participant_id", "id", "ID"]:
            if c in df.columns:
                return c
        return None

    def pick_rt_col(df: pd.DataFrame):
        for c in RT_COL_CANDIDATES:
            if c in df.columns:
                return c
        return None

    def pick_sex_col(df: pd.DataFrame):
        for c in SEX_COL_CANDIDATES:
            if c in df.columns:
                return c
        return None

    def normalize_sex_value(x):
        """
        Normalize various encodings to 'male'/'female' or NaN.
        """
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        # common variants
        if s in ["m", "male", "man", "men", "남", "남자", "남성"]:
            return "male"
        if s in ["f", "female", "woman", "women", "여", "여자", "여성"]:
            return "female"
        # sometimes '0/1'
        if s in ["0", "1"]:
            # can't reliably map without metadata; drop
            return np.nan
        return np.nan

    def add_age_group(df: pd.DataFrame):
        df = df.copy()
        df["age_group"] = pd.cut(df[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        return df

    def save_group_summary(df, value_col, out_path, group_cols=("age_group", "sex")):
        """
        Save mean/sd/sem/n summary for Prism (no raw needed).
        """
        g = df.groupby(list(group_cols), observed=True)[value_col].agg(
            n="count",
            mean="mean",
            sd="std"
        ).reset_index()
        g["sem"] = g["sd"] / np.sqrt(g["n"])
        g.to_csv(out_path, index=False, encoding="utf-8-sig")

    def run_two_way_anova(df, y_col, out_dir, tag, title_for_report, do_posthoc=True):
        """
        Two-way ANOVA (Type II): y ~ C(age_group) + C(sex) + interaction
        Tukey: if interaction sig -> on combined groups; elif age sig -> on age_group
        """
        formula = f"{y_col} ~ C(age_group) + C(sex) + C(age_group):C(sex)"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        p_age = float(anova_table.loc["C(age_group)", "PR(>F)"]) if "C(age_group)" in anova_table.index else np.nan
        p_sex = float(anova_table.loc["C(sex)", "PR(>F)"]) if "C(sex)" in anova_table.index else np.nan
        p_inter = float(anova_table.loc["C(age_group):C(sex)", "PR(>F)"]) if "C(age_group):C(sex)" in anova_table.index else np.nan

        def mark(p):
            return "✅" if (pd.notna(p) and p < ALPHA) else "❌"

        # ---- Console display ----
        print(f"\n[{tag}] {title_for_report}")
        print(f"  N={len(df)} | DV={y_col}")
        print(f"  Age group:   p={p_age:.6g} {mark(p_age)}")
        print(f"  Sex:         p={p_sex:.6g} {mark(p_sex)}")
        print(f"  Interaction: p={p_inter:.6g} {mark(p_inter)}")

        # ---- Report text ----

        summary = []
        summary.append(f"{title_for_report}\n")
        summary.append("------------------------------------------------------------\n")
        summary.append(f"N used (complete cases): {len(df)}\n\n")
        summary.append("ANOVA summary (Type II):\n")
        summary.append(f"- Age group main effect: p={p_age:.6g} {'(sig)' if p_age < ALPHA else '(n.s.)'}\n")
        summary.append(f"- Sex main effect:       p={p_sex:.6g} {'(sig)' if p_sex < ALPHA else '(n.s.)'}\n")
        summary.append(f"- Interaction:           p={p_inter:.6g} {'(sig)' if p_inter < ALPHA else '(n.s.)'}\n\n")
        summary.append("ANOVA table:\n")
        summary.append(anova_table.round(6).to_string())
        summary.append("\n\n")

        posthoc_txt = ""
        if do_posthoc:
            try:
                if (p_inter < ALPHA):
                    tmp = df.copy()
                    tmp["group"] = tmp["age_group"].astype(str) + "_" + tmp["sex"].astype(str)
                    tuk = pairwise_tukeyhsd(endog=tmp[y_col], groups=tmp["group"], alpha=ALPHA)
                    posthoc_txt += "Post-hoc (Tukey HSD) on Interaction groups (age_group × sex):\n"
                    posthoc_txt += str(tuk) + "\n\n"
                elif (p_age < ALPHA):
                    tuk = pairwise_tukeyhsd(endog=df[y_col], groups=df["age_group"], alpha=ALPHA)
                    posthoc_txt += "Post-hoc (Tukey HSD) on Age groups:\n"
                    posthoc_txt += str(tuk) + "\n\n"
            except Exception as e:
                posthoc_txt += f"(Post-hoc skipped due to error: {e})\n\n"

        # save
        anova_path = os.path.join(out_dir, f"{tag}_anova_table.csv")
        anova_table.round(8).to_csv(anova_path, encoding="utf-8-sig")
        report_path = os.path.join(out_dir, f"{tag}_anova_report.txt")
        save_text(report_path, "".join(summary) + posthoc_txt)

        print(f"✅ Saved ANOVA table: {anova_path}")
        print(f"✅ Saved report: {report_path}")

        return anova_table, (p_age, p_sex, p_inter)

    def pointplot_age_sex(df, y_col, out_dir, tag, title, y_label, ylim_upper_q=0.95):
        """
        Pointplot with SE. For readability, cap y-axis at quantile (optional).
        Raw data still saved separately.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.pointplot(
            data=df, x="age_group", y=y_col, hue="sex",
            order=AGE_LABELS, hue_order=["male", "female"],
            markers=["o", "s"], errorbar="se", ax=ax
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Age group")
        ax.set_ylabel(y_label)

        # cap for readability (optional)
        if ylim_upper_q is not None and len(df[y_col].dropna()) > 10:
            upper = float(df[y_col].quantile(ylim_upper_q))
            if np.isfinite(upper) and upper > 0:
                ax.set_ylim(0, upper)

        fn = os.path.join(out_dir, tag)
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved plot: {fn}.png/.svg")


    # -----------------------------
    # Optional: condition RT from responses_export.csv
    # -----------------------------
    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def compute_condition_rt_from_responses(responses_df: pd.DataFrame, participant_ids: set):
        """
        Returns wide table indexed by participantId:
          columns: Correct_AI, Correct_Real, Incorrect_AI, Incorrect_Real
        """
        df = responses_df.copy()

        # detect columns (robust)
        pid_col = _pick_col(df, ["participantId", "ParticipantId", "participant_id", "id", "ID"])
        trial_col = _pick_col(df, ["trial", "Trial", "trialName", "trial_id"])
        rt_col = _pick_col(df, ["rt", "RT", "responseTime", "response_time", "rt_ms"])
        corr_col = _pick_col(df, ["isCorrect", "IsCorrect", "correct", "is_correct"])
        type_col = _pick_col(df, ["imageType", "ImageType", "stimType", "stim_type", "sourceType", "source_type"])

        missing = [("participantId", pid_col), ("trial", trial_col), ("rt", rt_col), ("isCorrect", corr_col), ("imageType", type_col)]
        missing = [name for name, col in missing if col is None]
        if missing:
            raise KeyError(f"responses_export.csv missing required columns (auto-detect failed): {missing}")

        df = df[df[pid_col].isin(participant_ids)].copy()

        # exclude practice
        df["trial_str"] = df[trial_col].astype(str)
        df = df[~df["trial_str"].str.lower().str.startswith("practice", na=False)].copy()

        # numeric RT
        df["rt"] = to_num(df[rt_col])
        df = df.dropna(subset=["rt"])

        # correctness
        # allow 0/1, True/False, "true"/"false"
        def to_bool(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (bool, np.bool_)): return bool(x)
            s = str(x).strip().lower()
            if s in ["1", "true", "t", "yes", "y"]: return True
            if s in ["0", "false", "f", "no", "n"]: return False
            return np.nan

        df["is_correct_bool"] = df[corr_col].apply(to_bool)
        df = df.dropna(subset=["is_correct_bool"])

        # AI vs Real
        def is_ai(x):
            s = str(x).lower()
            return ("ai" in s)

        df["cond_correct"] = np.where(df["is_correct_bool"].astype(bool), "Correct", "Incorrect")
        df["cond_type"] = np.where(df[type_col].apply(is_ai), "AI", "Real")
        df["condition"] = df["cond_correct"] + "_" + df["cond_type"]  # safe

        # per participant mean
        wide = df.groupby([pid_col, "condition"])["rt"].mean().unstack()

        # ensure all columns exist
        for c in ["Correct_AI", "Correct_Real", "Incorrect_AI", "Incorrect_Real"]:
            if c not in wide.columns:
                wide[c] = np.nan

        wide = wide[["Correct_AI", "Correct_Real", "Incorrect_AI", "Incorrect_Real"]]
        wide.index.name = "participantId"
        return wide


    # -----------------------------
    # Main
    # -----------------------------
    print("==============================================================================")
    print("(23 v2.1) RT Analysis: Overall trend + Condition-specific detail (MOBILE + WEB)")
    print("==============================================================================\n")

    ensure_dir(OUTPUT_DIR)

    # Try load responses (optional)
    responses_df = None
    if os.path.exists(RESPONSES_FILE):
        try:
            responses_df = pd.read_csv(RESPONSES_FILE, encoding="utf-8-sig")
            print(f"✅ Loaded optional responses file: {RESPONSES_FILE} (rows={len(responses_df)})")
        except Exception as e:
            print(f"⚠️ Failed to load responses_export.csv. Condition RT will be skipped. Error: {e}")
            responses_df = None
    else:
        print("ℹ️ responses_export.csv not found. Condition RT will be skipped.")

    for cohort, fp in COHORT_FILES.items():
        print(f"\n==================== [{cohort.upper()}] (23 v2.1) START ====================")
        cohort_out = os.path.join(OUTPUT_DIR, cohort)
        ensure_dir(cohort_out)

        if not os.path.exists(fp):
            print(f"❌ Missing cohort file: {fp}")
            continue

        df = pd.read_csv(fp, encoding="utf-8-sig")
        print(f"✅ Loaded: {fp} [{cohort}] (rows={len(df)})")

        id_col = find_id_col(df)
        rt_col = pick_rt_col(df)
        sex_col = pick_sex_col(df)

        if id_col is None:
            print(f"⚠️ [{cohort}] participantId column not found in cohort file. Condition RT merge may be skipped.")
        else:
            print(f"✅ [{cohort}] Using participant id column: {id_col}")

        if rt_col is None:
            print(f"⚠️ [{cohort}] No RT column found in cohort file. Will try responses_export.csv for overall RT if available.")
        else:
            print(f"✅ [{cohort}] Using RT column: {rt_col}")

        if sex_col is None:
            print(f"❌ [{cohort}] No Sex/Gender column found. (looked for {SEX_COL_CANDIDATES}) -> skip cohort")
            continue
        else:
            print(f"✅ [{cohort}] Using Sex/Gender column: {sex_col} -> standardized to 'sex'")

        # basic cleaning
        df = df.copy()
        df[AGE_COL] = to_num(df[AGE_COL])
        df["sex"] = df[sex_col].apply(normalize_sex_value)

        # keep male/female only
        df = df[df["sex"].isin(["male", "female"])].copy()

        # age_group
        df = add_age_group(df)

        # overall RT 확보
        if rt_col is not None:
            df["avgRT_overall"] = to_num(df[rt_col])
        else:
            df["avgRT_overall"] = np.nan

        # If overall RT missing, try responses_export
        if df["avgRT_overall"].isna().all() and (responses_df is not None) and (id_col is not None):
            print(f"✅ [{cohort}] Computing overall RT from responses_export.csv ...")
            tmp = responses_df.copy()

            pid_col_r = _pick_col(tmp, ["participantId", "ParticipantId", "participant_id", "id", "ID"])
            trial_col_r = _pick_col(tmp, ["trial", "Trial", "trialName", "trial_id"])
            rt_col_r = _pick_col(tmp, ["rt", "RT", "responseTime", "response_time", "rt_ms"])
            if pid_col_r and trial_col_r and rt_col_r:
                part_ids = set(df[id_col].dropna().unique())
                tmp = tmp[tmp[pid_col_r].isin(part_ids)].copy()
                tmp["trial_str"] = tmp[trial_col_r].astype(str)
                tmp = tmp[~tmp["trial_str"].str.lower().str.startswith("practice", na=False)].copy()
                tmp["rt"] = to_num(tmp[rt_col_r])
                tmp = tmp.dropna(subset=["rt"])
                overall_rt = tmp.groupby(pid_col_r)["rt"].mean().rename("avgRT_overall").reset_index()

                # merge
                df = df.merge(overall_rt, left_on=id_col, right_on=pid_col_r, how="left", suffixes=("", "_y"))
                if "avgRT_overall_y" in df.columns:
                    df["avgRT_overall"] = df["avgRT_overall_y"]
                    df.drop(columns=["avgRT_overall_y"], inplace=True)
                if pid_col_r in df.columns and pid_col_r != id_col:
                    # keep both or drop? keep for debugging; but avoid confusion
                    pass

                print(f"✅ [{cohort}] Overall RT merged from responses. non-missing={df['avgRT_overall'].notna().sum()}")
            else:
                print(f"⚠️ [{cohort}] responses_export.csv missing trial/rt/participant columns -> cannot compute overall RT.")
        else:
            print(f"✅ [{cohort}] Overall RT available from cohort file: non-missing={df['avgRT_overall'].notna().sum()}")

        # Save Prism raw table (participant-level)
        prism_cols = []
        if id_col is not None and id_col in df.columns:
            prism_cols.append(id_col)
        prism_cols += [AGE_COL, "age_group", "sex", "avgRT_overall"]
        df_prism = df[prism_cols].copy()
        prism_path = os.path.join(cohort_out, f"23v2-0_prism_table_overall_rt_{cohort}.csv")
        df_prism.to_csv(prism_path, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort}] Saved Prism raw table (overall RT): {prism_path}")

        # Summary table (mean/sem)
        df_overall = df.dropna(subset=["age_group", "sex", "avgRT_overall"]).copy()
        summary_path = os.path.join(cohort_out, f"23v2-0_summary_overall_rt_mean_sem_{cohort}.csv")
        save_group_summary(df_overall, "avgRT_overall", summary_path)
        print(f"✅ [{cohort}] Saved summary (mean/sem): {summary_path}")

        # Overall RT plot + ANOVA
        n_overall = len(df_overall)
        print(f"✅ [{cohort}] N for overall RT (complete cases): {n_overall}")
        if n_overall >= 30:
            pointplot_age_sex(
                df_overall, "avgRT_overall", cohort_out,
                tag=f"23v2-1_overall_rt_age_sex_pointplot_{cohort}",
                title=f"(23v2-1) Overall AvgRT by Age group × Sex [{cohort.upper()}]",
                y_label="AvgRT (ms)",
                ylim_upper_q=0.95
            )
            run_two_way_anova(
                df_overall, "avgRT_overall", cohort_out,
                tag=f"23v2-2_overall_rt_{cohort}",
                title_for_report=f"(23v2-2) Two-way ANOVA: overall AvgRT ~ Age group × Sex [{cohort.upper()}]"
            )
        else:
            print(f"⚠️ [{cohort}] Not enough complete cases for overall RT. Skipping overall ANOVA/plot.")

        # -------------------------
        # Condition-specific RT
        # -------------------------
        if (responses_df is None) or (id_col is None) or (id_col not in df.columns):
            print(f"ℹ️ [{cohort}] Condition RT skipped (need responses_export.csv + participantId column in cohort file).")
            print(f"==================== [{cohort.upper()}] (23 v2.1) END ====================")
            continue

        try:
            part_ids = set(df[id_col].dropna().unique())
            wide = compute_condition_rt_from_responses(responses_df, part_ids)

            wide_path = os.path.join(cohort_out, f"23v2-3_condition_rt_wide_{cohort}.csv")
            wide.reset_index().to_csv(wide_path, index=False, encoding="utf-8-sig")
            print(f"✅ [{cohort}] Saved condition RT wide table: {wide_path}")
        except Exception as e:
            print(f"⚠️ [{cohort}] Failed to compute condition RT. Skipping condition analyses. Error: {e}")
            print(f"==================== [{cohort.upper()}] (23 v2.1) END ====================")
            continue

        # merge condition RT into df
        df_cond = df.merge(wide, left_on=id_col, right_index=True, how="left")

        cond_cols = ["Correct_AI", "Correct_Real", "Incorrect_AI", "Incorrect_Real"]
        for k, cond in enumerate(cond_cols, start=3):
            d = df_cond.dropna(subset=["age_group", "sex", cond]).copy()
            n_used = len(d)
            print(f"✅ [{cohort}] Condition '{cond}' N complete: {n_used}")

            if n_used < 30:
                print(f"⚠️ [{cohort}] Not enough data for condition '{cond}'. Skipping.")
                continue

            # Prism raw table
            prism_cols2 = []
            if id_col in d.columns:
                prism_cols2.append(id_col)
            prism_cols2 += [AGE_COL, "age_group", "sex", cond]
            prism_path2 = os.path.join(cohort_out, f"23v2-{k}_prism_table_rt_{cond.lower()}_{cohort}.csv")
            d[prism_cols2].to_csv(prism_path2, index=False, encoding="utf-8-sig")
            print(f"✅ [{cohort}] Saved Prism raw table: {prism_path2}")

            # Summary mean/sem table
            summary_path2 = os.path.join(cohort_out, f"23v2-{k}_summary_rt_mean_sem_{cond.lower()}_{cohort}.csv")
            save_group_summary(d, cond, summary_path2)
            print(f"✅ [{cohort}] Saved summary (mean/sem): {summary_path2}")

            # Plot
            pointplot_age_sex(
                d, cond, cohort_out,
                tag=f"23v2-{k}_rt_{cond.lower()}_pointplot_{cohort}",
                title=f"(23v2-{k}) RT in condition: {cond.replace('_',' ')} (Age group × Sex) [{cohort.upper()}]",
                y_label="RT (ms)",
                ylim_upper_q=0.95
            )

            # ANOVA
            run_two_way_anova(
                d, cond, cohort_out,
                tag=f"23v2-{k}_anova_rt_{cond.lower()}_{cohort}",
                title_for_report=f"(23v2-{k}) Two-way ANOVA: RT({cond.replace('_',' ')}) ~ Age group × Sex [{cohort.upper()}]"
            )

        print(f"==================== [{cohort.upper()}] (23 v2.1) END ====================")

    print("\n==================== (23 v2.1) DONE ====================")


def _run_cell_102():
    # ==============================================================================
    # (24 v2.3) Correct vs Incorrect RT by Age Group
    #   - MOBILE + WEB separated
    #   - Participant-level paired RT (Incorrect - Correct)
    #   - Console prints: overall + interaction(diff~age_group HC3) + age-group paired + Holm
    #   - Saves Prism files (wide/long) + summary(mean/SE) + plots per cohort
    #   - Robust parsing: rt numeric, isCorrect boolean-ish, Sex/Gender -> sex
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import ttest_rel
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multitest import multipletests

    # -----------------------------
    # Config
    # -----------------------------
    BASE_OUTDIR = "outputs_24_rt_correct_incorrect_v2_3"

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web": config.WEB_AGE_FILTERED,
    }
    RESP_FILE = config.RAW_RESPONSES

    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    # Columns
    PID_CANDS = ["participantId", "ParticipantId", "participant_id", "id"]
    AGE_COL = "age"
    SEX_CANDS = ["sex", "Sex", "gender", "Gender", "GENDER"]

    # Responses columns
    RESP_PID = "participantId"
    RESP_RT = "rt"
    RESP_TRIAL = "trial"
    RESP_ISCORRECT = "isCorrect"

    # Plot
    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()


    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sem(x):
        x = pd.Series(x).dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def robust_ylim(series, lower_q=0.02, upper_q=0.98, pad=0.10):
        s = pd.Series(series).dropna()
        if s.empty:
            return (0, 1)
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)
        if np.isclose(lo, hi):
            lo, hi = s.min(), s.max()
        rng = hi - lo if hi > lo else (s.max() - s.min())
        lo2 = max(0, lo - pad * rng)
        hi2 = hi + pad * rng
        return (lo2, hi2)

    def fmt_p(p):
        if pd.isna(p):
            return "nan"
        if p < 0.001:
            return f"{p:.2e}"
        return f"{p:.4f}"

    def find_first_col(df, cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    def make_age_group(df, age_col=AGE_COL):
        df = df.copy()
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        df["age_group"] = pd.cut(df[age_col], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        return df

    def standardize_sex(df, sex_col):
        df = df.copy()
        if sex_col is None:
            df["sex"] = np.nan
            return df
        df["sex"] = df[sex_col].astype(str).str.lower().str.strip()
        # keep only male/female if present
        df.loc[~df["sex"].isin(["male", "female"]), "sex"] = np.nan
        return df

    def coerce_isCorrect_to_boolish(s):
        """
        Accepts True/False, 1/0, 'true'/'false', '1'/'0', 'TRUE'/'FALSE', etc.
        Returns: boolean or NaN
        """
        if pd.isna(s):
            return np.nan
        if isinstance(s, (bool, np.bool_)):
            return bool(s)
        # numeric-like
        try:
            if str(s).strip() in ["0", "1"]:
                return True if str(s).strip() == "1" else False
        except Exception:
            pass
        ss = str(s).strip().lower()
        if ss in ["true", "t", "yes", "y"]:
            return True
        if ss in ["false", "f", "no", "n"]:
            return False
        return np.nan

    def map_unstack_columns(subj_unstacked):
        """
        subj_unstacked columns could be True/False or 1/0 or 'true'/'false'
        rename them to Correct_RT / Incorrect_RT
        """
        col_map = {}
        for c in subj_unstacked.columns:
            sc = str(c).strip().lower()
            if (c is True) or (sc == "true") or (sc == "1"):
                col_map[c] = "Correct_RT"
            if (c is False) or (sc == "false") or (sc == "0"):
                col_map[c] = "Incorrect_RT"
        return subj_unstacked.rename(columns=col_map)

    def print_df(df, title):
        print("\n" + title)
        print(df.to_string(index=False))


    # -----------------------------
    # Core per-cohort runner
    # -----------------------------
    def run_section24_for_cohort(cohort_tag, cohort_file):
        outdir = os.path.join(BASE_OUTDIR, cohort_tag)
        ensure_dir(outdir)

        print("\n==============================================================================")
        print(f"(24 v2.3) {cohort_tag.upper()} | Correct vs Incorrect RT by Age Group")
        print("==============================================================================")

        if not os.path.exists(cohort_file):
            print(f"❌ Missing cohort file: {cohort_file}")
            return
        if not os.path.exists(RESP_FILE):
            print(f"❌ Missing responses file: {RESP_FILE}")
            return

        df_main = pd.read_csv(cohort_file, encoding="utf-8-sig")
        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")

        pid_col = find_first_col(df_main, PID_CANDS)
        if pid_col is None:
            print(f"❌ No participant id column in cohort file. Tried: {PID_CANDS}")
            return
        if AGE_COL not in df_main.columns:
            print(f"❌ Cohort file missing '{AGE_COL}'")
            return

        sex_col = find_first_col(df_main, SEX_CANDS)

        df_main = make_age_group(df_main, AGE_COL)
        df_main = standardize_sex(df_main, sex_col)

        # participant set (cohort membership)
        pid_set = set(df_main[pid_col].dropna().unique())

        # responses: exclude practice + keep cohort participants
        if RESP_PID not in df_resp.columns:
            print(f"❌ responses file missing '{RESP_PID}'")
            return
        if RESP_TRIAL not in df_resp.columns:
            print(f"❌ responses file missing '{RESP_TRIAL}'")
            return
        if RESP_RT not in df_resp.columns:
            print(f"❌ responses file missing '{RESP_RT}'")
            return
        if RESP_ISCORRECT not in df_resp.columns:
            print(f"❌ responses file missing '{RESP_ISCORRECT}'")
            return

        df_rt = df_resp[df_resp[RESP_PID].isin(pid_set)].copy()
        df_rt = df_rt[~df_rt[RESP_TRIAL].astype(str).str.startswith("Practice", na=False)].copy()

        # coerce types
        df_rt[RESP_RT] = pd.to_numeric(df_rt[RESP_RT], errors="coerce")
        df_rt["isCorrect_bool"] = df_rt[RESP_ISCORRECT].apply(coerce_isCorrect_to_boolish)

        df_rt = df_rt.dropna(subset=[RESP_PID, RESP_RT, "isCorrect_bool"]).copy()

        # Participant-level mean RT by correctness
        subj = df_rt.groupby([RESP_PID, "isCorrect_bool"])[RESP_RT].mean().unstack()
        subj = map_unstack_columns(subj)

        # ensure both columns exist
        if "Correct_RT" not in subj.columns:
            subj["Correct_RT"] = np.nan
        if "Incorrect_RT" not in subj.columns:
            subj["Incorrect_RT"] = np.nan

        df_subj = df_main[[pid_col, AGE_COL, "age_group", "sex"]].copy()
        df_subj = df_subj.merge(subj.reset_index().rename(columns={RESP_PID: pid_col}),
                                on=pid_col, how="left")

        # paired-ready
        df_pair = df_subj.dropna(subset=["Correct_RT", "Incorrect_RT", "age_group"]).copy()
        df_pair["age_group"] = df_pair["age_group"].astype(str)

        # add diff
        df_pair["diff_inc_minus_cor"] = df_pair["Incorrect_RT"] - df_pair["Correct_RT"]

        # -----------------------------
        # Save Prism tables (wide/long)
        # -----------------------------
        wide_path = os.path.join(outdir, f"24v2-0_subject_wide_correct_incorrect_{cohort_tag}.csv")
        df_pair.to_csv(wide_path, index=False, encoding="utf-8-sig")

        df_long = df_pair.melt(
            id_vars=[pid_col, AGE_COL, "age_group", "sex"],
            value_vars=["Correct_RT", "Incorrect_RT"],
            var_name="condition",
            value_name="rt",
        )
        df_long["condition"] = df_long["condition"].replace({"Correct_RT": "Correct", "Incorrect_RT": "Incorrect"})

        long_path = os.path.join(outdir, f"24v2-0_long_rt_correct_incorrect_{cohort_tag}.csv")
        df_long.to_csv(long_path, index=False, encoding="utf-8-sig")

        print(f"✅ Saved Prism tables: {wide_path}")
        print(f"✅ Saved Prism tables: {long_path}")

        # -----------------------------
        # (24v2-0) Summary mean/SE table (Prism friendly)
        # -----------------------------
        sum_rows = []
        for ag in AGE_LABELS:
            sub = df_pair[df_pair["age_group"] == ag]
            if sub.empty:
                continue
            sum_rows.append({"age_group": ag, "condition": "Correct",
                             "mean": sub["Correct_RT"].mean(), "se": sem(sub["Correct_RT"]), "n": sub["Correct_RT"].notna().sum()})
            sum_rows.append({"age_group": ag, "condition": "Incorrect",
                             "mean": sub["Incorrect_RT"].mean(), "se": sem(sub["Incorrect_RT"]), "n": sub["Incorrect_RT"].notna().sum()})
            sum_rows.append({"age_group": ag, "condition": "Diff(Inc-Cor)",
                             "mean": sub["diff_inc_minus_cor"].mean(), "se": sem(sub["diff_inc_minus_cor"]), "n": sub["diff_inc_minus_cor"].notna().sum()})
        summary_df = pd.DataFrame(sum_rows)
        summary_path = os.path.join(outdir, f"24v2-0_summary_mean_se_{cohort_tag}.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved summary (mean/SE): {summary_path}")

        # -----------------------------
        # (24v2-1) Overall paired t-test
        # -----------------------------
        t_all = ttest_rel(df_pair["Incorrect_RT"], df_pair["Correct_RT"])
        overall_summary = pd.DataFrame([{
            "Cohort": cohort_tag,
            "N(paired)": len(df_pair),
            "Mean Correct": df_pair["Correct_RT"].mean(),
            "Mean Incorrect": df_pair["Incorrect_RT"].mean(),
            "Mean Diff(Inc-Cor)": df_pair["diff_inc_minus_cor"].mean(),
            "t": float(t_all.statistic),
            "p": float(t_all.pvalue)
        }])

        show = overall_summary.copy()
        show["Mean Correct"] = show["Mean Correct"].map(lambda x: f"{x:.2f}")
        show["Mean Incorrect"] = show["Mean Incorrect"].map(lambda x: f"{x:.2f}")
        show["Mean Diff(Inc-Cor)"] = show["Mean Diff(Inc-Cor)"].map(lambda x: f"{x:.2f}")
        show["t"] = show["t"].map(lambda x: f"{x:.3f}")
        show["p"] = show["p"].map(fmt_p)

        print_df(show, "--- (24v2-1) Overall paired t-test ---")

        rep1_path = os.path.join(outdir, f"24v2-1_overall_paired_ttest_{cohort_tag}.txt")
        save_text(rep1_path, overall_summary.to_string(index=False))

        # -----------------------------
        # (24v2-2) Interaction via diff score: diff ~ age_group (HC3)
        # -----------------------------
        model_diff = ols("diff_inc_minus_cor ~ C(age_group)", data=df_pair).fit(cov_type="HC3")

        # Wald test: all age_group dummies = 0 (vs baseline 20s)
        constraints = [f"C(age_group)[T.{lv}] = 0" for lv in AGE_LABELS[1:]]
        wald = model_diff.wald_test(" , ".join(constraints), scalar=True)

        # eta^2 from SS (plain OLS)
        model_plain = ols("diff_inc_minus_cor ~ C(age_group)", data=df_pair).fit()
        anova_tbl = sm.stats.anova_lm(model_plain, typ=2)
        ss_age = float(anova_tbl.loc["C(age_group)", "sum_sq"])
        ss_res = float(anova_tbl.loc["Residual", "sum_sq"])
        eta2 = ss_age / (ss_age + ss_res) if (ss_age + ss_res) > 0 else np.nan

        interaction_row = pd.DataFrame([{
            "Cohort": cohort_tag,
            "Test": "Wald(age_group) on diff",
            "stat": float(wald.statistic),
            "df": len(AGE_LABELS) - 1,
            "p": float(wald.pvalue),
            "eta2": float(eta2)
        }])

        show2 = interaction_row.copy()
        show2["stat"] = show2["stat"].map(lambda x: f"{x:.3f}")
        show2["p"] = show2["p"].map(fmt_p)
        show2["eta2"] = show2["eta2"].map(lambda x: f"{x:.4f}")

        print_df(show2, "--- (24v2-2) Interaction test (diff ~ age_group, HC3) ---")

        rep2_path = os.path.join(outdir, f"24v2-2_interaction_diff_agegroup_report_{cohort_tag}.txt")
        save_text(rep2_path, interaction_row.to_string(index=False) + "\n\n" + str(anova_tbl))

        # -----------------------------
        # (24v2-3) Age-group paired tests + Holm
        # -----------------------------
        rows = []
        for ag in AGE_LABELS:
            dfa = df_pair[df_pair["age_group"] == ag].copy()
            if len(dfa) < 5:
                continue
            tt = ttest_rel(dfa["Incorrect_RT"], dfa["Correct_RT"])
            diff = dfa["Incorrect_RT"] - dfa["Correct_RT"]
            rows.append({
                "age_group": ag,
                "N": len(dfa),
                "mean_correct": dfa["Correct_RT"].mean(),
                "mean_incorrect": dfa["Incorrect_RT"].mean(),
                "mean_diff": diff.mean(),
                "t": float(tt.statistic),
                "p": float(tt.pvalue),
            })

        post_df = pd.DataFrame(rows)
        if not post_df.empty:
            rej, p_holm, _, _ = multipletests(post_df["p"].values, alpha=0.05, method="holm")
            post_df["p_holm"] = p_holm
            post_df["sig_holm"] = rej
        else:
            post_df["p_holm"] = []
            post_df["sig_holm"] = []

        print("\n--- (24v2-3) Age-group paired t-tests + Holm ---")
        if post_df.empty:
            print("No age-group has N>=5 for paired test.")
        else:
            show3 = post_df.copy()
            show3["mean_correct"] = show3["mean_correct"].map(lambda x: f"{x:.2f}")
            show3["mean_incorrect"] = show3["mean_incorrect"].map(lambda x: f"{x:.2f}")
            show3["mean_diff"] = show3["mean_diff"].map(lambda x: f"{x:.2f}")
            show3["t"] = show3["t"].map(lambda x: f"{x:.3f}")
            show3["p"] = show3["p"].map(fmt_p)
            show3["p_holm"] = show3["p_holm"].map(fmt_p)
            show3["sig(Holm)"] = show3["sig_holm"].map(lambda b: "✅" if b else "❌")
            print(show3[["age_group","N","mean_correct","mean_incorrect","mean_diff","t","p","p_holm","sig(Holm)"]].to_string(index=False))

        post_csv = os.path.join(outdir, f"24v2-3_agegroup_paired_tests_holm_{cohort_tag}.csv")
        post_df.to_csv(post_csv, index=False, encoding="utf-8-sig")

        # -----------------------------
        # (24v2-4) Plot (Mean ± SE) with robust y-limits
        # -----------------------------
        # robust ylim from raw RTs (long)
        ylo, yhi = robust_ylim(df_long["rt"], 0.02, 0.98, pad=0.10)

        plot_rows = []
        for ag in AGE_LABELS:
            dfa = df_pair[df_pair["age_group"] == ag]
            if dfa.empty:
                continue
            plot_rows.append({"age_group": ag, "condition": "Correct",
                              "mean": dfa["Correct_RT"].mean(), "se": sem(dfa["Correct_RT"])})
            plot_rows.append({"age_group": ag, "condition": "Incorrect",
                              "mean": dfa["Incorrect_RT"].mean(), "se": sem(dfa["Incorrect_RT"])})
        plot_sum = pd.DataFrame(plot_rows)

        plt.figure(figsize=(11, 6))
        for cond in ["Correct", "Incorrect"]:
            sub = plot_sum[plot_sum["condition"] == cond].set_index("age_group").reindex(AGE_LABELS)
            x = np.arange(len(AGE_LABELS))
            y = sub["mean"].values
            yerr = sub["se"].values
            plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4, label=cond)

        plt.xticks(np.arange(len(AGE_LABELS)), AGE_LABELS)
        plt.ylabel("Mean RT (ms)")
        plt.xlabel("Age group")
        plt.title(f"(24v2-4) {cohort_tag.upper()} | Correct vs Incorrect RT (Mean ± SE)")
        plt.ylim(ylo, yhi)
        plt.legend()

        fig_png = os.path.join(outdir, f"24v2-4_correct_vs_incorrect_lineplot_{cohort_tag}.png")
        fig_svg = os.path.join(outdir, f"24v2-4_correct_vs_incorrect_lineplot_{cohort_tag}.svg")
        plt.savefig(fig_png, dpi=300, bbox_inches="tight")
        plt.savefig(fig_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"\n✅ Saved plot: {fig_png} / {fig_svg}")

        plot_table_path = os.path.join(outdir, f"24v2-4_plot_table_mean_se_{cohort_tag}.csv")
        plot_sum.to_csv(plot_table_path, index=False, encoding="utf-8-sig")

        # Save overall summary CSV too
        overall_csv = os.path.join(outdir, f"24v2-summary_overall_{cohort_tag}.csv")
        overall_summary.to_csv(overall_csv, index=False, encoding="utf-8-sig")

        print(f"✅ Saved outputs under: {outdir}")
        print("==================== DONE ====================")


    # -----------------------------
    # Entrypoint
    # -----------------------------
    if __name__ == "__main__":
        ensure_dir(BASE_OUTDIR)
        for cohort, fpath in COHORT_FILES.items():
            run_section24_for_cohort(cohort, fpath)


def _run_cell_106():
    # ==============================================================================
    # (25 v2.3) RT by Image Kind (Real vs. AI) across Age Groups
    # - MOBILE + WEB analyzed separately
    # - Participant-level mean RT for Real vs AI (paired)
    # - Paired t-test overall + by age group, Holm correction across age groups
    # - Line-only plot (mean ± SEM), robust y-limit via quantile
    # - Save wide/long/summary + ttest tables for Prism
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_rel

    # -----------------------------
    # Config
    # -----------------------------
    RESP_FILE = config.RAW_RESPONSES

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web": config.WEB_AGE_FILTERED,
    }

    OUTDIR = config.OUTPUTS_DIR / "outputs_25_rt_real_vs_ai_v2_3"

    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    YLIM_Q = 0.95
    AUTO_UNIT_FIX = True
    MIN_N_PER_AGEGROUP = 5

    ID_COL_CANDIDATES = ["participantId", "ParticipantId", "participant_id", "id"]
    AGE_COL_CANDIDATES = ["age", "Age", "participantAge"]

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p: str):
        os.makedirs(p, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def fmt_p(p):
        if pd.isna(p):
            return ""
        p = float(p)
        if p < 0.001:
            return f"{p:.2e}"
        return f"{p:.6f}"

    def is_practice_trial(x) -> bool:
        return str(x).lower().startswith("practice")

    def infer_and_fix_rt_unit(rt_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(rt_series, errors="coerce")
        med = np.nanmedian(s.values) if len(s) else np.nan
        if np.isnan(med):
            return s
        if AUTO_UNIT_FIX and med < 20:
            return s * 1000.0
        return s

    def sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def cohens_dz(ai: pd.Series, real: pd.Series) -> float:
        d = (pd.to_numeric(ai, errors="coerce") - pd.to_numeric(real, errors="coerce")).dropna()
        if len(d) <= 1:
            return np.nan
        sd = d.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        return d.mean() / sd

    def pstar(p: float) -> str:
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    def holm_adjust(pvals, alpha=0.05):
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]

        adj_sorted = np.empty(m, dtype=float)
        for i, p in enumerate(ranked):
            adj_sorted[i] = min(1.0, (m - i) * p)

        # enforce monotone non-decreasing
        for i in range(1, m):
            adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])

        out = np.empty(m, dtype=float)
        out[order] = adj_sorted
        sig = out < alpha
        return out, sig

    def find_first_existing(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def infer_image_kind_from_imagetype(x: str) -> str:
        """
        Robust mapping:
        - AI if contains 'ai' or 'synthetic' or 'generated'
        - Real otherwise
        """
        s = str(x).lower()
        if ("ai" in s) or ("synthetic" in s) or ("generated" in s):
            return "AI"
        return "Real"

    # -----------------------------
    # Core per-cohort runner
    # -----------------------------
    def run_section_25_for_cohort(cohort_tag: str, main_file: str):
        print("\n" + "=" * 78)
        print(f"(25 v2.3) RT by Image Kind (Real vs. AI) across Age Groups [{cohort_tag.upper()}]")
        print("=" * 78)

        cohort_out = os.path.join(OUTDIR, cohort_tag)
        ensure_dir(cohort_out)

        # 1) Load
        df_main = pd.read_csv(main_file, encoding="utf-8-sig")
        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")

        id_col = find_first_existing(df_main, ID_COL_CANDIDATES)
        age_col = find_first_existing(df_main, AGE_COL_CANDIDATES)

        if id_col is None:
            raise KeyError(f"[{cohort_tag}] No participant id column found in main file. Tried {ID_COL_CANDIDATES}")
        if age_col is None:
            raise KeyError(f"[{cohort_tag}] No age column found in main file. Tried {AGE_COL_CANDIDATES}")

        required_resp_cols = ["participantId", "trial", "rt", "imageType"]
        missing_resp = [c for c in required_resp_cols if c not in df_resp.columns]
        if missing_resp:
            raise KeyError(f"responses_export.csv missing columns: {missing_resp}")

        print(f"✅ Loaded: {main_file} (rows={len(df_main)}) | id_col={id_col}, age_col={age_col}")
        print(f"✅ Loaded: {RESP_FILE} (rows={len(df_resp)})")

        # 2) Filter responses to cohort participants + non-practice
        pid_set = set(df_main[id_col].dropna().unique())

        rt_df = df_resp[df_resp["participantId"].isin(pid_set)].copy()
        rt_df = rt_df[~rt_df["trial"].apply(is_practice_trial)].copy()

        # 3) RT numeric + unit fix
        rt_df["rt"] = infer_and_fix_rt_unit(rt_df["rt"])
        rt_df = rt_df.dropna(subset=["rt", "imageType", "participantId"])

        # 4) Image kind label (robust)
        rt_df["image_kind"] = rt_df["imageType"].apply(infer_image_kind_from_imagetype)

        # 5) Participant-level mean RT by kind (wide)
        rt_wide = (
            rt_df.groupby(["participantId", "image_kind"])["rt"]
            .mean()
            .unstack()
            .rename(columns={"Real": "Real_RT", "AI": "AI_RT"})
            .reset_index()
        )

        # Ensure columns exist
        if "Real_RT" not in rt_wide.columns:
            rt_wide["Real_RT"] = np.nan
        if "AI_RT" not in rt_wide.columns:
            rt_wide["AI_RT"] = np.nan

        # 6) Merge with main: age + age_group
        df = pd.merge(
            df_main[[id_col, age_col]].rename(columns={id_col: "participantId", age_col: "age"}),
            rt_wide,
            on="participantId",
            how="inner"
        )
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

        # Keep complete pairs only
        df_pair = df.dropna(subset=["age_group", "Real_RT", "AI_RT"]).copy()

        print(f"✅ N with Real+AI paired RTs: {len(df_pair)}")

        # ---- Save Prism-friendly raw tables ----
        wide_path = os.path.join(cohort_out, f"25v2-0_participant_wide_real_ai_rt_{cohort_tag}.csv")
        df_pair.to_csv(wide_path, index=False, encoding="utf-8-sig")

        long_df = df_pair.melt(
            id_vars=["participantId", "age", "age_group"],
            value_vars=["Real_RT", "AI_RT"],
            var_name="image_kind",
            value_name="rt_ms",
        )
        long_df["image_kind"] = long_df["image_kind"].replace({"Real_RT": "Real", "AI_RT": "AI"})
        long_path = os.path.join(cohort_out, f"25v2-0_participant_long_real_ai_rt_{cohort_tag}.csv")
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

        print(f"✅ Saved wide (Prism): {wide_path}")
        print(f"✅ Saved long (Prism): {long_path}")

        # ---- (25v2-1) Paired t-tests: overall + by age group ----
        results = []

        # Overall paired test (AI vs Real)
        t_all, p_all = ttest_rel(df_pair["AI_RT"], df_pair["Real_RT"], nan_policy="omit")
        dz_all = cohens_dz(df_pair["AI_RT"], df_pair["Real_RT"])

        results.append({
            "age_group": "ALL",
            "N": len(df_pair),
            "Real_mean_ms": df_pair["Real_RT"].mean(),
            "AI_mean_ms": df_pair["AI_RT"].mean(),
            "AI_minus_Real_mean_ms": (df_pair["AI_RT"] - df_pair["Real_RT"]).mean(),
            "t": t_all,
            "p_raw": p_all,
            "sig_raw": pstar(p_all),
            "p_holm": np.nan,
            "sig_holm": np.nan,
            "cohens_dz": dz_all,
            "note": "Overall (no correction)"
        })

        # By age group (collect p's for Holm)
        age_rows = []
        pvals = []
        for ag in AGE_LABELS:
            sub = df_pair[df_pair["age_group"] == ag].copy()
            if len(sub) < MIN_N_PER_AGEGROUP:
                continue
            t_stat, p_val = ttest_rel(sub["AI_RT"], sub["Real_RT"], nan_policy="omit")
            dz = cohens_dz(sub["AI_RT"], sub["Real_RT"])
            age_rows.append({
                "age_group": ag,
                "N": len(sub),
                "Real_mean_ms": sub["Real_RT"].mean(),
                "AI_mean_ms": sub["AI_RT"].mean(),
                "AI_minus_Real_mean_ms": (sub["AI_RT"] - sub["Real_RT"]).mean(),
                "t": t_stat,
                "p_raw": p_val,
                "sig_raw": pstar(p_val),
                "cohens_dz": dz,
                "note": "Age-group test (Holm across age groups)"
            })
            pvals.append(p_val)

        if len(pvals) > 0:
            p_holm, sig_holm = holm_adjust(pvals, alpha=0.05)
            for row, ph, sh in zip(age_rows, p_holm, sig_holm):
                row["p_holm"] = ph
                row["sig_holm"] = bool(sh)
        results.extend(age_rows)

        res_df = pd.DataFrame(results)

        def sig_holm_mark(x):
            if pd.isna(x):
                return ""
            return "✅" if bool(x) else "❌"

        res_df["sig(Holm)"] = res_df.get("sig_holm", np.nan).apply(sig_holm_mark)

        # Console output (human-readable p formatting)
        print(f"\n--- (25v2-1) Paired t-tests: AI vs Real RT [{cohort_tag}] ---")
        show_df = res_df.copy()
        show_df["Real_mean_ms"] = show_df["Real_mean_ms"].map(lambda x: f"{x:.2f}")
        show_df["AI_mean_ms"] = show_df["AI_mean_ms"].map(lambda x: f"{x:.2f}")
        show_df["AI_minus_Real_mean_ms"] = show_df["AI_minus_Real_mean_ms"].map(lambda x: f"{x:.2f}")
        show_df["t"] = show_df["t"].map(lambda x: f"{x:.3f}")
        show_df["p_raw"] = show_df["p_raw"].map(fmt_p)
        show_df["p_holm"] = show_df["p_holm"].map(fmt_p)
        show_df["cohens_dz"] = show_df["cohens_dz"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

        show_cols = [
            "age_group", "N",
            "Real_mean_ms", "AI_mean_ms", "AI_minus_Real_mean_ms",
            "t", "p_raw", "sig_raw", "p_holm", "sig(Holm)", "cohens_dz", "note"
        ]
        print(show_df[show_cols].to_string(index=False))

        # Save results
        ttest_csv = os.path.join(cohort_out, f"25v2-1_paired_ttests_by_age_group_holm_{cohort_tag}.csv")
        res_df.to_csv(ttest_csv, index=False, encoding="utf-8-sig")

        ttest_txt = os.path.join(cohort_out, f"25v2-1_paired_ttests_by_age_group_holm_{cohort_tag}.txt")
        save_text(ttest_txt, show_df[show_cols].to_string(index=False))

        print(f"\n✅ Saved t-test results (csv): {ttest_csv}")
        print(f"✅ Saved t-test results (txt): {ttest_txt}")

        # ---- (25v2-2) Summary stats for plotting (mean ± SEM) ----
        summary = (
            long_df.groupby(["age_group", "image_kind"])["rt_ms"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )

        # enforce order
        summary["age_group"] = pd.Categorical(summary["age_group"], categories=AGE_LABELS, ordered=True)
        summary["image_kind"] = pd.Categorical(summary["image_kind"], categories=["Real", "AI"], ordered=True)
        summary = summary.sort_values(["age_group", "image_kind"])

        summary_path = os.path.join(cohort_out, f"25v2-2_summary_mean_sem_{cohort_tag}.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved summary (mean/sem): {summary_path}")

        # ---- (25v2-3) Line-only plot (mean ± SEM), robust ylim ----
        sns.set_theme(style="whitegrid")
        config.apply_korean_plot_style()

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(AGE_LABELS))

        for kind, marker in [("Real", "o"), ("AI", "s")]:
            s = summary[summary["image_kind"] == kind].copy()
            s = s.set_index("age_group").reindex(AGE_LABELS)

            y = s["mean"].values.astype(float)
            yerr = s["sem"].values.astype(float)

            # if all missing, skip
            if np.all(~np.isfinite(y)):
                continue

            ax.errorbar(
                x, y, yerr=yerr,
                marker=marker, linestyle="-", linewidth=2,
                capsize=5, label=kind
            )

        ax.set_title(f"(25v2-3) Mean RT by Image Kind across Age Groups (Real vs AI) [{cohort_tag.upper()}]", fontsize=16, pad=12)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Mean RT (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS)
        ax.legend(title="Image kind")

        upper = float(long_df["rt_ms"].quantile(YLIM_Q)) if len(long_df) else np.nan
        if np.isfinite(upper) and upper > 0:
            ax.set_ylim(0, upper)

        out_png = os.path.join(cohort_out, f"25v2-3_rt_real_vs_ai_line_{cohort_tag}.png")
        out_svg = os.path.join(cohort_out, f"25v2-3_rt_real_vs_ai_line_{cohort_tag}.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot: {out_png} / {out_svg}")

        plt.show()
        plt.close(fig)

        print(f"==================== (25 v2.3) DONE [{cohort_tag.upper()}] ====================\n")


    # -----------------------------
    # Main
    # -----------------------------
    if __name__ == "__main__":
        ensure_dir(OUTDIR)

        if not os.path.exists(RESP_FILE):
            raise FileNotFoundError(f"Missing responses file: {RESP_FILE}")

        for cohort_tag, main_file in COHORT_FILES.items():
            if not os.path.exists(main_file):
                print(f"❌ Missing cohort file for {cohort_tag}: {main_file}")
                continue
            run_section_25_for_cohort(cohort_tag, main_file)


def _run_cell_109():
    # ==============================================================================
    # (26 v1) "Verification cost" test:
    #   4-condition RT (Correct/Incorrect × Real/AI), within-subject 2×2 interaction
    #   - MOBILE + WEB separately
    #   - Key hypothesis: Real requires more verification -> (Incorrect-Correct) gap larger for Real than AI
    # Inputs:
    #   - analysis_data_mobile_age_filtered_20_69.csv
    #   - analysis_data_web_age_filtered_20_69.csv
    #   - responses_export.csv
    # Outputs (per cohort):
    #   - Prism-ready wide/long tables
    #   - Summary mean/SEM tables
    #   - 2×2 repeated-measures ANOVA (complete-case) + key paired tests + Holm correction
    #   - Plots (no scatter): interaction plot + delta plot (png/svg)
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import ttest_rel, ttest_1samp
    from statsmodels.stats.anova import AnovaRM

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_26_verification_cost_v1"

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }
    RESP_FILE = config.RAW_RESPONSES

    AGE_BINS   = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    # RT unit auto-fix: if median < 20 -> assume seconds -> ms
    AUTO_UNIT_FIX = True

    # robust y-limit
    YLIM_Q = 0.98

    # plotting
    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()

    ALPHA = 0.05


    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): os.makedirs(p, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def is_practice_trial(x) -> bool:
        return str(x).lower().startswith("practice")

    def infer_and_fix_rt_unit(rt_series: pd.Series) -> pd.Series:
        s = to_num(rt_series)
        med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else np.nan
        if np.isnan(med):
            return s
        if AUTO_UNIT_FIX and med < 20:
            return s * 1000.0
        return s

    def sem(x: pd.Series) -> float:
        x = to_num(x).dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def cohens_dz_paired(x: pd.Series, y: pd.Series) -> float:
        d = (to_num(x) - to_num(y)).dropna()
        if len(d) <= 1:
            return np.nan
        sd = d.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        return d.mean() / sd

    def fmt_p(p):
        if p < 1e-3:
            return f"{p:.2e}"
        return f"{p:.6f}"

    def holm_adjust(pvals, alpha=0.05):
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj = np.empty(m, dtype=float)

        for i, p in enumerate(ranked):
            adj[i] = min(1.0, (m - i) * p)

        for i in range(1, m):
            adj[i] = max(adj[i], adj[i - 1])

        out = np.empty(m, dtype=float)
        out[order] = adj
        sig = out < alpha
        return out, sig

    def robust_ylim(values, q=0.98, pad=0.08):
        s = to_num(pd.Series(values)).dropna()
        if s.empty:
            return (0, 1)
        hi = float(s.quantile(q))
        lo = float(s.quantile(1 - q)) if q > 0.5 else float(s.min())
        if not np.isfinite(hi) or hi <= 0:
            hi = float(s.max())
        if not np.isfinite(lo):
            lo = float(s.min())
        rng = hi - lo
        lo2 = max(0, lo - pad * rng)
        hi2 = hi + pad * rng
        return (lo2, hi2)

    def label_image_kind(image_type):
        s = str(image_type).lower()
        return "AI" if "ai" in s else "Real"

    def normalize_isCorrect(x):
        # handle bool, 0/1, "true"/"false"
        if isinstance(x, bool):
            return x
        sx = str(x).strip().lower()
        if sx in ["true", "1", "t", "yes"]:
            return True
        if sx in ["false", "0", "f", "no"]:
            return False
        # fallback numeric
        try:
            return bool(int(float(sx)))
        except:
            return np.nan


    # -----------------------------
    # Core computation: 4-condition wide table
    # -----------------------------
    def compute_4cond_rt_wide(df_resp: pd.DataFrame, pid_set: set):
        """
        returns wide table indexed by participantId:
          Correct_Real, Incorrect_Real, Correct_AI, Incorrect_AI
        """
        d = df_resp[df_resp["participantId"].isin(pid_set)].copy()
        d = d[~d["trial"].apply(is_practice_trial)].copy()

        d["rt"] = infer_and_fix_rt_unit(d["rt"])
        d["isCorrect_bool"] = d["isCorrect"].apply(normalize_isCorrect)
        d = d.dropna(subset=["participantId", "rt", "imageType", "isCorrect_bool"])

        d["kind"] = d["imageType"].apply(label_image_kind)
        d["corr"] = np.where(d["isCorrect_bool"].astype(bool), "Correct", "Incorrect")
        d["cond"] = d["corr"] + "_" + d["kind"]  # Correct_Real etc.

        wide = d.groupby(["participantId", "cond"])["rt"].mean().unstack()

        # ensure all 4 exist
        for c in ["Correct_Real", "Incorrect_Real", "Correct_AI", "Incorrect_AI"]:
            if c not in wide.columns:
                wide[c] = np.nan

        return wide[["Correct_Real", "Incorrect_Real", "Correct_AI", "Incorrect_AI"]]


    # -----------------------------
    # Main per cohort
    # -----------------------------
    def run_for_cohort(cohort_tag: str, main_fp: str, resp_fp: str):
        print("\n" + "="*78)
        print(f"(26 v1) Verification cost test (4-condition RT) [{cohort_tag.upper()}]")
        print("="*78)

        outdir = os.path.join(OUTPUT_DIR, cohort_tag)
        ensure_dir(outdir)

        # load
        df_main = pd.read_csv(main_fp, encoding="utf-8-sig")
        df_resp = pd.read_csv(resp_fp, encoding="utf-8-sig")

        # sanity
        if "participantId" not in df_main.columns:
            raise KeyError(f"[{cohort_tag}] main file missing participantId: {main_fp}")
        if "age" not in df_main.columns:
            raise KeyError(f"[{cohort_tag}] main file missing age: {main_fp}")

        df_main = df_main[["participantId", "age"]].copy()
        df_main["age"] = to_num(df_main["age"])
        df_main["age_group"] = pd.cut(df_main["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

        pid_set = set(df_main["participantId"].dropna().unique())

        print(f"✅ Loaded main: {main_fp} (rows={len(df_main)})")
        print(f"✅ Loaded responses: {resp_fp} (rows={len(df_resp)})")
        print(f"✅ Unique participantId in cohort: {len(pid_set)}")

        # 4-cond wide RT
        wide_rt = compute_4cond_rt_wide(df_resp, pid_set).reset_index()

        # merge
        df = df_main.merge(wide_rt, on="participantId", how="left")

        # save "available-case" wide table for Prism (even if some cells missing)
        wide_all_path = os.path.join(outdir, f"26v1-0_wide_4cond_rt_available_{cohort_tag}.csv")
        df.to_csv(wide_all_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved Prism wide (available-case): {wide_all_path}")

        # complete-case for RM ANOVA + clean within-subject comparisons
        df_cc = df.dropna(subset=["Correct_Real","Incorrect_Real","Correct_AI","Incorrect_AI"]).copy()
        print(f"✅ N complete-case (all 4 cells present): {len(df_cc)}")

        wide_cc_path = os.path.join(outdir, f"26v1-0_wide_4cond_rt_completecase_{cohort_tag}.csv")
        df_cc.to_csv(wide_cc_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved Prism wide (complete-case): {wide_cc_path}")

        # long table (complete-case)
        long_rows = []
        for _, r in df_cc.iterrows():
            pid = r["participantId"]
            for cond in ["Correct_Real","Incorrect_Real","Correct_AI","Incorrect_AI"]:
                corr, kind = cond.split("_")  # Correct/Incorrect, Real/AI
                long_rows.append({
                    "participantId": pid,
                    "age": r["age"],
                    "age_group": r["age_group"],
                    "correctness": corr,
                    "kind": kind,
                    "rt_ms": r[cond]
                })
        df_long = pd.DataFrame(long_rows)

        long_path = os.path.join(outdir, f"26v1-0_long_4cond_rt_completecase_{cohort_tag}.csv")
        df_long.to_csv(long_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved Prism long (complete-case): {long_path}")

        # summary mean/SEM
        summary = (
            df_long.groupby(["correctness","kind"])["rt_ms"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
            .sort_values(["correctness","kind"])
        )
        summary_path = os.path.join(outdir, f"26v1-1_summary_mean_sem_4cond_{cohort_tag}.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved summary (mean/sem): {summary_path}")

        # -----------------------------
        # (26-2) Key "verification cost" test:
        #   ΔReal = Inc_Real - Cor_Real
        #   ΔAI   = Inc_AI   - Cor_AI
        #   Test ΔReal > ΔAI  (one-sided) + two-sided
        # -----------------------------
        df_cc["delta_real"] = df_cc["Incorrect_Real"] - df_cc["Correct_Real"]
        df_cc["delta_ai"]   = df_cc["Incorrect_AI"]   - df_cc["Correct_AI"]
        df_cc["delta_delta"] = df_cc["delta_real"] - df_cc["delta_ai"]

        # paired test (ΔReal vs ΔAI)
        t_pair = ttest_rel(df_cc["delta_real"], df_cc["delta_ai"])
        # one-sample on delta_delta vs 0
        t_dd = ttest_1samp(df_cc["delta_delta"], popmean=0.0)

        # convert to one-sided p for H1: delta_delta > 0
        if np.isfinite(t_dd.statistic):
            p_dd_1s = (t_dd.pvalue / 2) if (t_dd.statistic > 0) else (1 - t_dd.pvalue / 2)
        else:
            p_dd_1s = np.nan

        dz_dd = cohens_dz_paired(df_cc["delta_real"], df_cc["delta_ai"])

        print("\n--- (26v1-2) Verification cost 핵심검정 ---")
        print(f"N complete-case = {len(df_cc)}")
        print(f"ΔReal (Inc- Cor) mean = {df_cc['delta_real'].mean():.2f} ms")
        print(f"ΔAI   (Inc- Cor) mean = {df_cc['delta_ai'].mean():.2f} ms")
        print(f"ΔΔ = ΔReal-ΔAI mean   = {df_cc['delta_delta'].mean():.2f} ms")
        print(f"[Paired t] ΔReal vs ΔAI: t={t_pair.statistic:.3f}, p(two)={fmt_p(t_pair.pvalue)}")
        print(f"[One-sample t] ΔΔ vs 0: t={t_dd.statistic:.3f}, p(two)={fmt_p(t_dd.pvalue)}, p(one, ΔΔ>0)={fmt_p(p_dd_1s)}")
        print(f"Cohen's dz (paired, ΔReal-ΔAI) = {dz_dd:.3f}")

        key_df = pd.DataFrame([{
            "cohort": cohort_tag,
            "N_complete": len(df_cc),
            "delta_real_mean": df_cc["delta_real"].mean(),
            "delta_ai_mean": df_cc["delta_ai"].mean(),
            "delta_delta_mean": df_cc["delta_delta"].mean(),
            "paired_t_deltaReal_vs_deltaAI": t_pair.statistic,
            "paired_p_two": t_pair.pvalue,
            "onesample_t_deltaDelta": t_dd.statistic,
            "onesample_p_two": t_dd.pvalue,
            "onesample_p_one_deltaDelta_gt0": p_dd_1s,
            "cohens_dz": dz_dd
        }])
        key_path = os.path.join(outdir, f"26v1-2_key_verification_cost_tests_{cohort_tag}.csv")
        key_df.to_csv(key_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved key test table: {key_path}")

        # -----------------------------
        # (26-3) 2×2 repeated-measures ANOVA (complete-case)
        #   within: correctness (Correct/Incorrect), kind (Real/AI)
        # -----------------------------
        print("\n--- (26v1-3) 2×2 Repeated-measures ANOVA (complete-case) ---")
        try:
            aov = AnovaRM(
                data=df_long,
                depvar="rt_ms",
                subject="participantId",
                within=["correctness","kind"]
            ).fit()
            print(aov.summary())
            aov_txt = os.path.join(outdir, f"26v1-3_rm_anova_summary_{cohort_tag}.txt")
            save_text(aov_txt, str(aov.summary()))
            print(f"✅ Saved RM-ANOVA summary: {aov_txt}")
        except Exception as e:
            print(f"⚠️ RM-ANOVA failed: {e}")

        # -----------------------------
        # (26-4) 4 paired contrasts (Holm)
        #   1) Real: Incorrect vs Correct
        #   2) AI:   Incorrect vs Correct
        #   3) Correct: Real vs AI
        #   4) Incorrect: Real vs AI
        # -----------------------------
        print("\n--- (26v1-4) Paired contrasts (Holm across 4 tests) ---")

        # define series
        CR = df_cc["Correct_Real"]
        IR = df_cc["Incorrect_Real"]
        CA = df_cc["Correct_AI"]
        IA = df_cc["Incorrect_AI"]

        tests = []
        # 1) Real: Inc vs Cor
        t1 = ttest_rel(IR, CR)
        tests.append(("Real: Incorrect - Correct", t1.statistic, t1.pvalue, cohens_dz_paired(IR, CR), (IR-CR).mean()))
        # 2) AI: Inc vs Cor
        t2 = ttest_rel(IA, CA)
        tests.append(("AI: Incorrect - Correct", t2.statistic, t2.pvalue, cohens_dz_paired(IA, CA), (IA-CA).mean()))
        # 3) Correct: Real vs AI (Real - AI)
        t3 = ttest_rel(CR, CA)
        tests.append(("Correct: Real - AI", t3.statistic, t3.pvalue, cohens_dz_paired(CR, CA), (CR-CA).mean()))
        # 4) Incorrect: Real vs AI (Real - AI)
        t4 = ttest_rel(IR, IA)
        tests.append(("Incorrect: Real - AI", t4.statistic, t4.pvalue, cohens_dz_paired(IR, IA), (IR-IA).mean()))

        p_raw = [x[2] for x in tests]
        p_holm, sig_holm = holm_adjust(p_raw, alpha=0.05)

        contrast_rows = []
        for (name, tstat, praw, dz, meandiff), ph, sh in zip(tests, p_holm, sig_holm):
            contrast_rows.append({
                "contrast": name,
                "mean_diff_ms": meandiff,
                "t": tstat,
                "p_raw": praw,
                "p_holm": ph,
                "sig_holm": bool(sh),
                "cohens_dz": dz
            })

        contrast_df = pd.DataFrame(contrast_rows)
        print(
            contrast_df.assign(
                mean_diff_ms=contrast_df["mean_diff_ms"].map(lambda x: f"{x:.2f}"),
                t=contrast_df["t"].map(lambda x: f"{x:.3f}"),
                p_raw=contrast_df["p_raw"].map(fmt_p),
                p_holm=contrast_df["p_holm"].map(fmt_p),
                sig_holm=contrast_df["sig_holm"].map(lambda b: "SIG" if b else "NS"),
                cohens_dz=contrast_df["cohens_dz"].map(lambda x: f"{x:.3f}")
            ).to_string(index=False)
        )

        contrast_path = os.path.join(outdir, f"26v1-4_paired_contrasts_holm_{cohort_tag}.csv")
        contrast_df.to_csv(contrast_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved contrasts table: {contrast_path}")

        # -----------------------------
        # (26-5) Plots (no scatter)
        #   A) Interaction plot: x=correctness, line=kind, y=mean±SEM
        #   B) Delta plot: mean±SEM of delta_real vs delta_ai
        # -----------------------------
        print("\n--- (26v1-5) Plots (no scatter) ---")

        # A) interaction plot
        # build plot table in correct order
        plot_tbl = (
            df_long.groupby(["correctness","kind"])["rt_ms"]
            .agg(mean="mean", sem=sem)
            .reset_index()
        )
        # enforce order
        corr_order = ["Correct", "Incorrect"]
        kind_order = ["Real", "AI"]

        fig, ax = plt.subplots(figsize=(9.5, 6))
        x = np.arange(len(corr_order))

        for kind in kind_order:
            sub = plot_tbl[plot_tbl["kind"] == kind].set_index("correctness").reindex(corr_order)
            y = sub["mean"].values
            yerr = sub["sem"].values
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=5, label=kind)

        ax.set_xticks(x)
        ax.set_xticklabels(corr_order)
        ax.set_xlabel("Correctness")
        ax.set_ylabel("RT (ms)")
        ax.set_title(f"(26v1) RT interaction: Correctness × Image kind [{cohort_tag.upper()}]")
        ax.legend(title="Kind")

        ylo, yhi = robust_ylim(df_long["rt_ms"], q=YLIM_Q, pad=0.08)
        ax.set_ylim(ylo, yhi)

        fn = os.path.join(outdir, f"26v1-5A_interaction_correctness_kind_{cohort_tag}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved: {fn}.png/.svg")

        # B) delta plot (mean±SEM)
        deltas = pd.DataFrame({
            "delta_kind": ["Real"]*len(df_cc) + ["AI"]*len(df_cc),
            "delta_ms": pd.concat([df_cc["delta_real"], df_cc["delta_ai"]], ignore_index=True)
        })
        delta_sum = deltas.groupby("delta_kind")["delta_ms"].agg(mean="mean", sem=sem, n="count").reset_index()
        delta_sum_path = os.path.join(outdir, f"26v1-5B_delta_summary_mean_sem_{cohort_tag}.csv")
        delta_sum.to_csv(delta_sum_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved delta summary: {delta_sum_path}")

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        x2 = np.arange(2)
        y2 = delta_sum.set_index("delta_kind").reindex(["Real","AI"])["mean"].values
        e2 = delta_sum.set_index("delta_kind").reindex(["Real","AI"])["sem"].values
        ax.errorbar(x2, y2, yerr=e2, marker="o", linewidth=2, capsize=6)

        ax.set_xticks(x2)
        ax.set_xticklabels(["ΔReal (Inc-Cor)", "ΔAI (Inc-Cor)"])
        ax.set_ylabel("Delta RT (ms)")
        ax.set_title(f"(26v1) Verification cost deltas [{cohort_tag.upper()}]")
        # y-lim from delta distribution
        ylo2, yhi2 = robust_ylim(deltas["delta_ms"], q=0.98, pad=0.10)
        ax.set_ylim(ylo2, yhi2)

        # annotate one-sided p
        ax.text(0.5, 0.98, f"p(one, ΔΔ>0)={fmt_p(p_dd_1s)}",
                transform=ax.transAxes, ha="center", va="top")

        fn2 = os.path.join(outdir, f"26v1-5B_delta_real_vs_ai_{cohort_tag}")
        fig.savefig(fn2 + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn2 + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"✅ Saved: {fn2}.png/.svg")

        # Final report txt (single file)
        rep = []
        rep.append("==============================================================================\n")
        rep.append(f"(26 v1) Verification cost test [{cohort_tag.upper()}]\n")
        rep.append("==============================================================================\n\n")
        rep.append(f"N cohort participants (from main): {len(pid_set)}\n")
        rep.append(f"N complete-case (all 4 RT cells): {len(df_cc)}\n\n")
        rep.append("[Key test: ΔΔ = (Inc- Cor)_Real - (Inc- Cor)_AI]\n")
        rep.append(f"- ΔReal mean = {df_cc['delta_real'].mean():.3f} ms\n")
        rep.append(f"- ΔAI mean   = {df_cc['delta_ai'].mean():.3f} ms\n")
        rep.append(f"- ΔΔ mean    = {df_cc['delta_delta'].mean():.3f} ms\n")
        rep.append(f"- One-sample t(ΔΔ vs 0): t={t_dd.statistic:.6f}, p(two)={fmt_p(t_dd.pvalue)}, p(one, ΔΔ>0)={fmt_p(p_dd_1s)}\n")
        rep.append(f"- Paired t(ΔReal vs ΔAI): t={t_pair.statistic:.6f}, p(two)={fmt_p(t_pair.pvalue)}\n")
        rep.append(f"- Cohen's dz (paired) = {dz_dd:.6f}\n\n")
        rep.append("[Paired contrasts (Holm across 4)]\n")
        rep.append(contrast_df.to_string(index=False) + "\n\n")
        rep.append("[Saved files]\n")
        rep.append(f"- Wide available-case: {wide_all_path}\n")
        rep.append(f"- Wide complete-case : {wide_cc_path}\n")
        rep.append(f"- Long complete-case : {long_path}\n")
        rep.append(f"- Summary mean/sem   : {summary_path}\n")
        rep.append(f"- Key test table     : {key_path}\n")
        rep.append(f"- Contrast table     : {contrast_path}\n")
        rep.append(f"- Plots              : 26v1-5A, 26v1-5B (png/svg)\n")

        rep_path = os.path.join(outdir, f"26v1_report_{cohort_tag}.txt")
        save_text(rep_path, "".join(rep))
        print(f"\n✅ Saved report: {rep_path}")
        print(f"==================== (26 v1) DONE [{cohort_tag.upper()}] ====================\n")


    # -----------------------------
    # Run
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(26 v1) Verification cost test: 4-condition RT (Correct/Incorrect × Real/AI)")
        print("==============================================================================\n")

        ensure_dir(OUTPUT_DIR)

        if not os.path.exists(RESP_FILE):
            raise FileNotFoundError(f"Missing responses file: {RESP_FILE}")

        for cohort_tag, fp in COHORT_FILES.items():
            if not os.path.exists(fp):
                print(f"❌ Missing cohort file for {cohort_tag}: {fp}")
                continue
            run_for_cohort(cohort_tag, fp, RESP_FILE)

        print("\n==================== (26 v1) ALL DONE ====================")


def _run_cell_112():
    # ==============================================================================
    # (26 v1.1) Verification cost deep-dive:
    #   A) Complete-case selection bias check
    #   B) Trial-level RT distribution (Correct/Incorrect × Real/AI)
    #   C) Age-group pattern of deltas: ΔReal, ΔAI, ΔΔ (=ΔReal-ΔAI)
    #
    # Inputs:
    #   - analysis_data_mobile_age_filtered_20_69.csv
    #   - analysis_data_web_age_filtered_20_69.csv
    #   - responses_export.csv
    #
    # Outputs (per cohort):
    #   - Prism tables: wide available/complete, long, delta tables, bias tables
    #   - Plots (PNG/SVG): 4-cell interaction, delta by age, ΔΔ by age, trial-level RT distro
    #   - Console prints: key stats + bias check summary
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # -----------------------------
    # Config
    # -----------------------------
    OUTPUT_DIR = "outputs_26_verification_cost_v1_1"

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }
    RESP_FILE = config.RAW_RESPONSES

    ID_CANDS = ["participantId", "ParticipantId", "participant_id", "id"]
    SEX_CANDS = ["sex", "Sex", "gender", "Gender"]
    AGE_COL = "age"
    ACC_CANDS = ["overallAccuracy", "overallAccuracy_y", "overall_accuracy", "accuracy", "Accuracy"]
    MAIN_RT_CANDS = ["avgRT", "mean_rt", "avg_rt", "MeanRT", "meanRT"]

    AGE_BINS  = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    ALPHA = 0.05

    # Plot style
    sns.set_theme(style="whitegrid")
    config.apply_korean_plot_style()

    # Robust ylim quantile for RT plots
    RT_UPPER_Q = 0.97

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): os.makedirs(p, exist_ok=True)

    def to_num(s): return pd.to_numeric(s, errors="coerce")

    def find_first_col(df, cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    def standardize_sex(series):
        s = series.astype(str).str.lower().str.strip()
        # common variants
        s = s.replace({"m":"male", "f":"female"})
        return s

    def add_age_group(df):
        df = df.copy()
        df[AGE_COL] = to_num(df[AGE_COL])
        df["age_group"] = pd.cut(df[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        return df

    def infer_and_fix_rt_unit(rt_series: pd.Series) -> pd.Series:
        """
        If median RT < 20, assume seconds -> convert to ms.
        """
        s = to_num(rt_series)
        med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else np.nan
        if np.isnan(med):
            return s
        if med < 20:
            return s * 1000.0
        return s

    def robust_upper(series, q=RT_UPPER_Q):
        s = pd.Series(series).dropna()
        if s.empty:
            return 1
        return float(s.quantile(q))

    def sem(x):
        x = pd.Series(x).dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def cohens_dz(diff):
        diff = np.asarray(diff, dtype=float)
        sd = diff.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        return diff.mean() / sd

    def fmt_p(p):
        if p < 0.001:
            return f"{p:.2e}"
        return f"{p:.4f}"

    def pmark(p):
        return "SIG" if p < ALPHA else "NS"

    # -----------------------------
    # Build 4-condition table from responses
    # -----------------------------
    def build_4cond_rt_from_responses(df_resp, pid_set):
        """
        Returns wide table indexed by participantId:
          Correct_Real, Correct_AI, Incorrect_Real, Incorrect_AI
        Excludes Practice trials.
        """
        df = df_resp.copy()
        df = df[df["participantId"].isin(pid_set)].copy()

        # exclude practice
        df["trial_str"] = df["trial"].astype(str)
        df = df[~df["trial_str"].str.lower().str.startswith("practice")].copy()

        # RT numeric + unit fix
        df["rt"] = infer_and_fix_rt_unit(df["rt"])
        df = df.dropna(subset=["participantId", "rt", "isCorrect", "imageType"])

        # kind
        df["kind"] = df["imageType"].astype(str).str.lower().apply(lambda x: "AI" if "ai" in x else "Real")
        df["correctness"] = df["isCorrect"].astype(bool).map({True:"Correct", False:"Incorrect"})
        df["cell"] = df["correctness"] + "_" + df["kind"]

        wide = df.groupby(["participantId", "cell"])["rt"].mean().unstack()
        # ensure all columns
        for c in ["Correct_Real","Correct_AI","Incorrect_Real","Incorrect_AI"]:
            if c not in wide.columns:
                wide[c] = np.nan
        wide = wide[["Correct_Real","Correct_AI","Incorrect_Real","Incorrect_AI"]]
        return wide

    # -----------------------------
    # Core per-cohort analysis
    # -----------------------------
    def run_26v11_for_cohort(cohort, main_fp, df_resp):
        print("\n" + "="*78)
        print(f"(26 v1.1) Verification cost deep-dive [{cohort.upper()}]")
        print("="*78)

        outdir = os.path.join(OUTPUT_DIR, cohort)
        ensure_dir(outdir)

        df_main = pd.read_csv(main_fp, encoding="utf-8-sig")
        print(f"✅ Loaded main: {main_fp} (rows={len(df_main)})")

        # columns
        id_col = find_first_col(df_main, ID_CANDS)
        sex_col_raw = find_first_col(df_main, SEX_CANDS)
        acc_col = find_first_col(df_main, ACC_CANDS)
        mainrt_col = find_first_col(df_main, MAIN_RT_CANDS)

        if id_col is None:
            raise KeyError(f"[{cohort}] participantId column not found in main file.")
        print(f"✅ [{cohort}] id_col={id_col} | acc_col={acc_col} | mainrt_col={mainrt_col} | sex_col={sex_col_raw}")

        df_main = df_main.copy()
        df_main[id_col] = df_main[id_col]
        df_main = add_age_group(df_main)

        # standardize sex (optional)
        if sex_col_raw is not None:
            df_main["sex"] = standardize_sex(df_main[sex_col_raw])
            df_main = df_main[df_main["sex"].isin(["male","female"])].copy()
        else:
            df_main["sex"] = np.nan

        # standardize accuracy if present
        if acc_col is not None:
            df_main["accuracy"] = to_num(df_main[acc_col])
            # if looks like proportion -> %
            mx = df_main["accuracy"].max(skipna=True)
            if pd.notna(mx) and mx <= 1.0:
                df_main["accuracy"] = df_main["accuracy"] * 100.0
            df_main["accuracy"] = df_main["accuracy"].clip(0, 100)
        else:
            df_main["accuracy"] = np.nan

        # main avgRT if present
        if mainrt_col is not None:
            df_main["avgRT_main"] = infer_and_fix_rt_unit(df_main[mainrt_col])
        else:
            df_main["avgRT_main"] = np.nan

        # participant set
        pid_set = set(df_main[id_col].dropna().unique())

        # 4-cond RT from responses
        wide = build_4cond_rt_from_responses(df_resp, pid_set)
        wide_avail = wide.reset_index().rename(columns={"participantId":"participantId"})
        wide_avail_path = os.path.join(outdir, f"26v11-0_wide_4cond_available_{cohort}.csv")
        wide_avail.to_csv(wide_avail_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved wide (available-case): {wide_avail_path}")

        # complete-case definition
        complete_mask = wide.notna().all(axis=1)
        wide_cc = wide[complete_mask].copy()
        print(f"✅ N complete-case (all 4 cells present): {len(wide_cc)}")

        wide_cc_path = os.path.join(outdir, f"26v11-0_wide_4cond_completecase_{cohort}.csv")
        wide_cc.reset_index().to_csv(wide_cc_path, index=False, encoding="utf-8-sig")

        # merge metadata
        meta_cols = [id_col, AGE_COL, "age_group", "sex", "accuracy", "avgRT_main"]
        meta_cols = [c for c in meta_cols if c in df_main.columns]
        meta = df_main[meta_cols].drop_duplicates(subset=[id_col]).copy()

        df_cc = meta.merge(wide_cc.reset_index(), left_on=id_col, right_on="participantId", how="inner")
        # keep age_group
        df_cc = df_cc.dropna(subset=["age_group"])

        # long for prism
        long = df_cc.melt(
            id_vars=[id_col, AGE_COL, "age_group", "sex", "accuracy", "avgRT_main"],
            value_vars=["Correct_Real","Correct_AI","Incorrect_Real","Incorrect_AI"],
            var_name="cell",
            value_name="rt_ms"
        )
        long_path = os.path.join(outdir, f"26v11-0_long_4cond_completecase_{cohort}.csv")
        long.to_csv(long_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved wide (complete-case): {wide_cc_path}")
        print(f"✅ Saved long (complete-case): {long_path}")

        # =========================
        # (A) Selection bias check
        # =========================
        # label participants
        cc_ids = set(wide_cc.index.astype(str))
        # make ID comparable as string
        df_main_tmp = df_main.copy()
        df_main_tmp["__pid__"] = df_main_tmp[id_col].astype(str)
        df_main_tmp["is_completecase"] = df_main_tmp["__pid__"].isin(cc_ids)

        # bias table
        bias_rows = []
        for var in ["age", "accuracy", "avgRT_main"]:
            if var not in df_main_tmp.columns:
                continue
            a = to_num(df_main_tmp.loc[df_main_tmp["is_completecase"], var]).dropna()
            b = to_num(df_main_tmp.loc[~df_main_tmp["is_completecase"], var]).dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            tstat, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            # Mann-Whitney (robust)
            try:
                u, p_u = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                u, p_u = np.nan, np.nan
            bias_rows.append({
                "variable": var,
                "complete_n": len(a),
                "noncomplete_n": len(b),
                "complete_mean": float(a.mean()),
                "noncomplete_mean": float(b.mean()),
                "diff_complete_minus_non": float(a.mean() - b.mean()),
                "welch_t": float(tstat),
                "p_welch": float(p),
                "mw_u": float(u) if np.isfinite(u) else np.nan,
                "p_mw": float(p_u) if np.isfinite(p_u) else np.nan
            })

        bias_df = pd.DataFrame(bias_rows)
        bias_path = os.path.join(outdir, f"26v11-1_selection_bias_continuous_{cohort}.csv")
        bias_df.to_csv(bias_path, index=False, encoding="utf-8-sig")

        # sex composition
        sex_comp = None
        if "sex" in df_main_tmp.columns:
            sex_comp = (
                df_main_tmp.dropna(subset=["sex"])
                .groupby(["is_completecase","sex"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            sex_path = os.path.join(outdir, f"26v11-1_selection_bias_sex_comp_{cohort}.csv")
            sex_comp.to_csv(sex_path, index=False, encoding="utf-8-sig")

        print("\n--- (26v1.1-A) Selection bias check (complete-case vs others) ---")
        if bias_df.empty:
            print("No continuous variables available for bias check.")
        else:
            show = bias_df.copy()
            for c in ["complete_mean","noncomplete_mean","diff_complete_minus_non"]:
                show[c] = show[c].map(lambda x: f"{x:.2f}")
            show["p_welch"] = show["p_welch"].map(fmt_p)
            show["p_mw"] = show["p_mw"].map(lambda x: fmt_p(x) if np.isfinite(x) else "")
            print(show[["variable","complete_n","noncomplete_n","complete_mean","noncomplete_mean",
                        "diff_complete_minus_non","p_welch","p_mw"]].to_string(index=False))
        if sex_comp is not None:
            print("\nSex composition (counts):")
            print(sex_comp.to_string(index=False))

        # =========================
        # (B) Key verification-cost test + RM-style interaction plot
        # =========================
        df_cc["dReal"] = df_cc["Incorrect_Real"] - df_cc["Correct_Real"]
        df_cc["dAI"]   = df_cc["Incorrect_AI"]   - df_cc["Correct_AI"]
        df_cc["dDiff"] = df_cc["dReal"] - df_cc["dAI"]  # ΔΔ

        # paired test dReal vs dAI (same as interaction contrast)
        t_da = ttest_rel(df_cc["dReal"], df_cc["dAI"])
        dz_da = cohens_dz((df_cc["dReal"] - df_cc["dAI"]).values)

        print("\n--- (26v1.1-B) Key test: dReal vs dAI (complete-case) ---")
        print(f"N={len(df_cc)} | mean dReal={df_cc['dReal'].mean():.2f} | mean dAI={df_cc['dAI'].mean():.2f} | mean(dReal-dAI)={df_cc['dDiff'].mean():.2f}")
        print(f"Paired t: t={t_da.statistic:.3f}, p={fmt_p(t_da.pvalue)} | dz={dz_da:.3f} | direction: {'dReal>dAI' if df_cc['dDiff'].mean()>0 else 'dReal<dAI'}")

        # Save key deltas (Prism)
        deltas_path = os.path.join(outdir, f"26v11-2_deltas_completecase_{cohort}.csv")
        df_cc[[id_col, AGE_COL, "age_group", "sex", "accuracy", "avgRT_main", "dReal","dAI","dDiff"]].to_csv(
            deltas_path, index=False, encoding="utf-8-sig"
        )

        # Interaction plot (mean±SEM, no scatter)
        # Compute means/sem per cell
        cell_order = ["Correct_Real","Correct_AI","Incorrect_Real","Incorrect_AI"]
        summ_rows = []
        for cell in cell_order:
            summ_rows.append({
                "cell": cell,
                "mean": float(df_cc[cell].mean()),
                "sem": float(sem(df_cc[cell]))
            })
        summ = pd.DataFrame(summ_rows)
        summ_path = os.path.join(outdir, f"26v11-2_summary_mean_sem_4cell_{cohort}.csv")
        summ.to_csv(summ_path, index=False, encoding="utf-8-sig")

        # Plot interaction: correctness x kind
        # Prepare aggregated
        tmp = long.copy()
        tmp["correctness"] = tmp["cell"].str.split("_").str[0]
        tmp["kind"] = tmp["cell"].str.split("_").str[1]
        inter = (
            tmp.groupby(["correctness","kind"])["rt_ms"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )

        rt_upper = robust_upper(tmp["rt_ms"], RT_UPPER_Q)

        fig, ax = plt.subplots(figsize=(9, 6))
        # line plot (no scatter): Correct/Incorrect as x, separate lines for kind
        x_order = ["Correct","Incorrect"]
        for kind, marker in [("Real","o"), ("AI","s")]:
            s = inter[inter["kind"] == kind].set_index("correctness").reindex(x_order)
            x = np.arange(len(x_order))
            ax.errorbar(x, s["mean"].values, yerr=s["sem"].values, marker=marker, linewidth=2, capsize=5, label=kind)

        ax.set_xticks(np.arange(len(x_order)))
        ax.set_xticklabels(x_order)
        ax.set_ylabel("Mean RT (ms)")
        ax.set_title(f"(26v1.1) Interaction: Correctness × Kind [{cohort.upper()}]\n(dReal vs dAI p={fmt_p(t_da.pvalue)})")
        ax.set_ylim(0, rt_upper)
        ax.legend(title="Kind")

        fn = os.path.join(outdir, f"26v11-3_interaction_correctness_kind_{cohort}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # =========================
        # (C) Age-group pattern: dReal, dAI, dDiff + interaction test
        # =========================
        # summaries by age_group
        age_sum = (
            df_cc.groupby("age_group")[["dReal","dAI","dDiff"]]
            .agg(["mean", sem, "count"])
        )
        age_sum_path = os.path.join(outdir, f"26v11-4_agegroup_delta_summary_{cohort}.csv")
        age_sum.to_csv(age_sum_path, encoding="utf-8-sig")

        # interaction test: dDiff ~ C(age_group) with HC3
        df_cc2 = df_cc.dropna(subset=["age_group","dDiff"]).copy()
        df_cc2["age_group"] = df_cc2["age_group"].astype(str)
        if df_cc2["age_group"].nunique() >= 2 and len(df_cc2) >= 50:
            m = smf.ols("dDiff ~ C(age_group)", data=df_cc2).fit(cov_type="HC3")
            # Wald test: all non-reference age dummies == 0
            levels = [lv for lv in AGE_LABELS if lv in set(df_cc2["age_group"].unique())]
            if len(levels) >= 2:
                constraints = " , ".join([f"C(age_group)[T.{lv}] = 0" for lv in levels[1:]])
                wald = m.wald_test(constraints, scalar=True)
                p_wald = float(wald.pvalue)
                stat_wald = float(wald.statistic)
            else:
                p_wald, stat_wald = np.nan, np.nan
        else:
            p_wald, stat_wald = np.nan, np.nan

        print("\n--- (26v1.1-C) Age-group pattern on dDiff (=dReal-dAI) ---")
        if np.isfinite(p_wald):
            print(f"Wald test: stat={stat_wald:.3f}, p={fmt_p(p_wald)} ({pmark(p_wald)})")
        else:
            print("Wald test skipped (not enough groups/rows).")

        # Paired dReal vs dAI within each age group (optional descriptive)
        rows = []
        for ag in AGE_LABELS:
            sub = df_cc[df_cc["age_group"].astype(str) == ag].dropna(subset=["dReal","dAI"])
            if len(sub) < 10:
                continue
            tt = ttest_rel(sub["dReal"], sub["dAI"])
            rows.append({
                "age_group": ag,
                "N": len(sub),
                "mean_dReal": float(sub["dReal"].mean()),
                "mean_dAI": float(sub["dAI"].mean()),
                "mean_dDiff": float((sub["dReal"]-sub["dAI"]).mean()),
                "t(dReal_vs_dAI)": float(tt.statistic),
                "p": float(tt.pvalue)
            })
        ag_test = pd.DataFrame(rows)
        ag_test_path = os.path.join(outdir, f"26v11-4_agegroup_paired_dReal_vs_dAI_{cohort}.csv")
        ag_test.to_csv(ag_test_path, index=False, encoding="utf-8-sig")

        if not ag_test.empty:
            show = ag_test.copy()
            show["mean_dReal"] = show["mean_dReal"].map(lambda x: f"{x:.1f}")
            show["mean_dAI"] = show["mean_dAI"].map(lambda x: f"{x:.1f}")
            show["mean_dDiff"] = show["mean_dDiff"].map(lambda x: f"{x:.1f}")
            show["p"] = show["p"].map(fmt_p)
            print(show[["age_group","N","mean_dReal","mean_dAI","mean_dDiff","p"]].to_string(index=False))

        # Plot: dReal vs dAI by age group (mean±SEM)
        # build plot table
        plot_rows = []
        for ag in AGE_LABELS:
            sub = df_cc[df_cc["age_group"].astype(str) == ag]
            if sub.empty:
                continue
            plot_rows.append({"age_group": ag, "delta": "dReal", "mean": float(sub["dReal"].mean()), "sem": float(sem(sub["dReal"]))})
            plot_rows.append({"age_group": ag, "delta": "dAI",   "mean": float(sub["dAI"].mean()),   "sem": float(sem(sub["dAI"]))})
            plot_rows.append({"age_group": ag, "delta": "dDiff", "mean": float(sub["dDiff"].mean()), "sem": float(sem(sub["dDiff"]))})
        plot_tbl = pd.DataFrame(plot_rows)
        plot_tbl_path = os.path.join(outdir, f"26v11-4_plot_table_deltas_mean_sem_{cohort}.csv")
        plot_tbl.to_csv(plot_tbl_path, index=False, encoding="utf-8-sig")

        # dReal & dAI plot
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(AGE_LABELS))
        for delta, marker in [("dReal","o"), ("dAI","s")]:
            s = plot_tbl[plot_tbl["delta"] == delta].set_index("age_group").reindex(AGE_LABELS)
            ax.errorbar(x, s["mean"].values, yerr=s["sem"].values, marker=marker, linewidth=2, capsize=4, label=delta)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS)
        ax.set_ylabel("Delta RT (ms) : (Incorrect - Correct)")
        ax.set_title(f"(26v1.1) dReal vs dAI by age group [{cohort.upper()}]")
        ax.legend()
        fn = os.path.join(outdir, f"26v11-5A_dReal_vs_dAI_by_age_{cohort}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # dDiff plot
        fig, ax = plt.subplots(figsize=(11, 6))
        s = plot_tbl[plot_tbl["delta"] == "dDiff"].set_index("age_group").reindex(AGE_LABELS)
        ax.errorbar(x, s["mean"].values, yerr=s["sem"].values, marker="o", linewidth=2, capsize=4)
        ax.axhline(0, linestyle="--", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS)
        ax.set_ylabel("dDiff (ms) = dReal - dAI")
        ax.set_title(f"(26v1.1) dDiff by age group [{cohort.upper()}] | Wald p={fmt_p(p_wald) if np.isfinite(p_wald) else 'NA'}")
        fn = os.path.join(outdir, f"26v11-5B_dDiff_by_age_{cohort}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # =========================
        # (B2) Trial-level distribution check (no scatter)
        # =========================
        # Make trial-level table for cohort pids
        df_t = df_resp[df_resp["participantId"].isin(pid_set)].copy()
        df_t["trial_str"] = df_t["trial"].astype(str)
        df_t = df_t[~df_t["trial_str"].str.lower().str.startswith("practice")].copy()
        df_t["rt"] = infer_and_fix_rt_unit(df_t["rt"])
        df_t = df_t.dropna(subset=["rt","isCorrect","imageType","participantId"])

        df_t["kind"] = df_t["imageType"].astype(str).str.lower().apply(lambda x: "AI" if "ai" in x else "Real")
        df_t["correctness"] = df_t["isCorrect"].astype(bool).map({True:"Correct", False:"Incorrect"})
        df_t["cell"] = df_t["correctness"] + "_" + df_t["kind"]

        # cap extreme outliers for visualization only (quantile cap)
        cap = robust_upper(df_t["rt"], RT_UPPER_Q)
        df_plot = df_t[df_t["rt"] <= cap].copy()

        # plot: histogram lines per cell (density)
        fig, ax = plt.subplots(figsize=(11, 6))
        for cell in ["Correct_Real","Correct_AI","Incorrect_Real","Incorrect_AI"]:
            s = df_plot[df_plot["cell"] == cell]["rt"]
            if len(s) < 50:
                continue
            ax.hist(s, bins=60, density=True, histtype="step", linewidth=2, label=cell)
        ax.set_xlabel("RT (ms) [capped for plot]")
        ax.set_ylabel("Density")
        ax.set_title(f"(26v1.1) Trial-level RT distribution by cell [{cohort.upper()}] (cap @ q={RT_UPPER_Q})")
        ax.legend()
        fn = os.path.join(outdir, f"26v11-6_triallevel_hist_density_by_cell_{cohort}")
        fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fn + ".svg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # Save trial-level summary stats
        trial_sum = (
            df_t.groupby("cell")["rt"]
            .agg(n="count", mean="mean", median="median",
                 q75=lambda x: float(pd.Series(x).quantile(0.75)),
                 q90=lambda x: float(pd.Series(x).quantile(0.90)),
                 q95=lambda x: float(pd.Series(x).quantile(0.95)))
            .reset_index()
        )
        trial_sum_path = os.path.join(outdir, f"26v11-6_triallevel_summary_by_cell_{cohort}.csv")
        trial_sum.to_csv(trial_sum_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ Saved outputs under: {outdir}")
        print("=== DONE ===")


    # -----------------------------
    # Entrypoint
    # -----------------------------
    if __name__ == "__main__":
        ensure_dir(OUTPUT_DIR)

        if not os.path.exists(RESP_FILE):
            raise FileNotFoundError(f"Missing {RESP_FILE}")

        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")
        print(f"✅ Loaded responses: {RESP_FILE} (rows={len(df_resp)})")

        for cohort, fp in COHORT_FILES.items():
            if not os.path.exists(fp):
                print(f"❌ Missing main file: {fp}")
                continue
            run_26v11_for_cohort(cohort, fp, df_resp)

        print("\n==================== (26 v1.1) ALL DONE ====================")


def _run_cell_115():
    # ==============================================================================
    # (26 v1.2) Trial-level Mixed Model: RT ~ Correctness × Kind (+ Age + Sex)
    #   - MOBILE + WEB separated
    #   - Trial-level (responses_export.csv) mixed model with participant random intercept
    #   - Recommended DV: logRT (log of RT in ms)
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    OUTDIR = config.OUTPUTS_DIR / "outputs_26_verification_cost_v1_2"
    RESP_FILE = config.RAW_RESPONSES
    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web": config.WEB_AGE_FILTERED,
    }

    AGE_COL = "age"
    SEX_COL_CANDS = ["sex", "gender"]
    ID_COL = "participantId"

    def ensure_dir(p): os.makedirs(p, exist_ok=True)

    def to_num(s): return pd.to_numeric(s, errors="coerce")

    def is_practice(trial):
        return str(trial).lower().startswith("practice")

    def infer_rt_ms(rt_series: pd.Series) -> pd.Series:
        s = to_num(rt_series)
        med = np.nanmedian(s.values)
        if np.isfinite(med) and med < 20:   # seconds로 보이면
            return s * 1000.0
        return s

    def pick_sex_col(df):
        for c in SEX_COL_CANDS:
            if c in df.columns:
                return c
        return None

    def kind_from_imageType(x):
        s = str(x).lower()
        return "AI" if "ai" in s else "Real"

    def run_for_cohort(tag, main_fp):
        print("\n" + "="*78)
        print(f"(26 v1.2) Trial-level Mixed Model [{tag.upper()}]")
        print("="*78)

        out = os.path.join(OUTDIR, tag)
        ensure_dir(out)

        df_main = pd.read_csv(main_fp, encoding="utf-8-sig")
        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")

        # cohort participant set + covariates
        sex_col = pick_sex_col(df_main)
        base_cols = [ID_COL, AGE_COL]
        if sex_col: base_cols.append(sex_col)

        cov = df_main[base_cols].copy()
        cov[AGE_COL] = to_num(cov[AGE_COL])
        if sex_col:
            cov["sex"] = cov[sex_col].astype(str).str.lower().str.strip()
            cov = cov[cov["sex"].isin(["male","female"])].copy()
        else:
            cov["sex"] = np.nan

        pid_set = set(cov[ID_COL].dropna().unique())

        # responses: filter + practice 제거
        d = df_resp[df_resp[ID_COL].isin(pid_set)].copy()
        d = d[~d["trial"].apply(is_practice)].copy()

        # RT + unit fix
        d["rt_ms"] = infer_rt_ms(d["rt"])
        d = d.dropna(subset=["rt_ms", "isCorrect", "imageType", ID_COL]).copy()

        # factors
        d["Correctness"] = np.where(d["isCorrect"].astype(bool), "Correct", "Incorrect")
        d["Kind"] = d["imageType"].apply(kind_from_imageType)

        # merge covariates
        d = d.merge(cov[[ID_COL, AGE_COL, "sex"]], on=ID_COL, how="left")
        d = d.dropna(subset=[AGE_COL])  # age는 필수로 두자

        # DV transform
        d["logRT"] = np.log(d["rt_ms"])

        # Save trial-level Prism-ish table
        keep = [ID_COL, "rt_ms", "logRT", "Correctness", "Kind", AGE_COL, "sex"]
        d[keep].to_csv(os.path.join(out, "26v12-0_trial_level_table.csv"), index=False, encoding="utf-8-sig")
        print(f"✅ Saved trial-level table: {os.path.join(out, '26v12-0_trial_level_table.csv')}")

        # Mixed model: random intercept by participant
        # Fixed: Correctness * Kind + Age + Sex
        # Note: statsmodels MixedLM에서 C() 사용 가능
        formula = "logRT ~ C(Correctness) * C(Kind) + age + C(sex)"
        md = smf.mixedlm(formula, d, groups=d[ID_COL])
        m = md.fit(method="lbfgs", reml=False)

        # Save summary
        summ_txt = m.summary().as_text()
        with open(os.path.join(out, "26v12-1_mixedlm_summary.txt"), "w", encoding="utf-8") as f:
            f.write(summ_txt)
        print(f"✅ Saved mixedlm summary: {os.path.join(out, '26v12-1_mixedlm_summary.txt')}")

        # Coef table
        coef = pd.DataFrame({
            "term": m.params.index,
            "coef": m.params.values,
            "se": m.bse.values,
            "z": m.tvalues.values,
            "p": m.pvalues.values
        })
        coef.to_csv(os.path.join(out, "26v12-1_mixedlm_coeffs.csv"), index=False, encoding="utf-8-sig")
        print(f"✅ Saved coeffs: {os.path.join(out, '26v12-1_mixedlm_coeffs.csv')}")

        # 핵심: interaction term p-value 표시
        # 보통 term 이름이 'C(Correctness)[T.Incorrect]:C(Kind)[T.Real]' 같은 형태
        inter_terms = [t for t in coef["term"] if ":" in t and "Correctness" in t and "Kind" in t]
        print("\n--- Key terms ---")
        if inter_terms:
            for t in inter_terms:
                row = coef[coef["term"] == t].iloc[0]
                print(f"[Interaction] {t} | coef={row['coef']:.4f}, p={row['p']:.3g}")
        else:
            print("⚠️ Interaction term not found (check factor labels).")

    if __name__ == "__main__":
        ensure_dir(OUTDIR)
        assert os.path.exists(RESP_FILE), f"Missing {RESP_FILE}"
        for tag, fp in COHORT_FILES.items():
            if not os.path.exists(fp):
                print(f"❌ Missing cohort file: {fp}")
                continue
            run_for_cohort(tag, fp)


def main():
    _run_cell_071()
    _run_cell_074()
    _run_cell_077()
    _run_cell_081()
    _run_cell_083()
    _run_cell_087()
    _run_cell_090()
    _run_cell_093()
    _run_cell_096()
    _run_cell_099()
    _run_cell_102()
    _run_cell_106()
    _run_cell_109()
    _run_cell_112()
    _run_cell_115()


if __name__ == "__main__":
    main()
