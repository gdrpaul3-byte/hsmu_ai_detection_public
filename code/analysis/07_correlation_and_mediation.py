"""Analysis sections for correlation structures and mediation models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_043():
    # ==============================================================================
    # (13) Correlation analysis among key variables (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # - Merge mean RT from responses_export.csv (exclude Practice trials)
    # - Map AI exposure/confidence/attitude to numeric scores
    # - Correlation heatmap (Pearson r) + p-values
    # - Pairwise scatter + Pearson & Spearman reports
    # - ✅ Accuracy(%)가 y축인 scatter plot은 y=0~100으로 고정
    # - ✅ 그래프 색/스타일 유지 (regression line color = #e91e63, heatmap = RdBu_r)
    # - Save all outputs under: outputs/<run_dir>/13_correlation/
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import pearsonr, spearmanr


    # -----------------------------
    # Utilities
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


    def safe_slug(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = s.replace(" ", "_").replace("/", "_")
        return s


    def save_text_report(out_dir: str, section_number: str, base_filename: str, title: str, content: str):
        ensure_dir(out_dir)
        fn = os.path.join(out_dir, f"{section_number}_{base_filename}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f"{title}\n")
            f.write("============================================================\n\n")
            f.write(str(content))
        print(f"✅ Saved report: {fn}")


    def find_accuracy_column(df: pd.DataFrame) -> str:
        """Prefer overallAccuracy_y if exists, else overallAccuracy."""
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        raise ValueError("No accuracy column found (expected overallAccuracy_y or overallAccuracy).")


    def convert_accuracy_to_percent_if_needed(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        """
        Ensure accuracy is numeric and in percent scale (0-100).
        Heuristic:
          - if median <= 1.0 and max <= 1.0 -> treat as proportion (0-1), convert to %
          - else assume already in % scale
        Returns (df, scaling_note)
        """
        df = df.copy()
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")

        med = df[acc_col].median(skipna=True)
        mx = df[acc_col].max(skipna=True)

        if pd.notna(mx) and mx <= 1.0 and pd.notna(med) and med <= 1.0:
            df[acc_col] = df[acc_col] * 100.0
            note = f"{acc_col} treated as proportion (0-1) -> converted to %"
        else:
            note = f"{acc_col} treated as already in % scale (no conversion)"
        return df, note


    # -----------------------------
    # RT merge
    # -----------------------------
    def compute_mean_rt(responses_path: str) -> pd.DataFrame:
        """Compute mean RT per participantId excluding Practice trials."""
        resp = pd.read_csv(responses_path, encoding="utf-8-sig")

        needed = {"participantId", "trial", "rt"}
        if not needed.issubset(resp.columns):
            raise ValueError("responses_export.csv must contain participantId, trial, rt columns.")

        resp = resp.copy()
        resp["trial"] = resp["trial"].astype(str)
        resp = resp[~resp["trial"].str.startswith("Practice")].copy()

        resp["rt"] = pd.to_numeric(resp["rt"], errors="coerce")
        resp = resp.dropna(subset=["participantId", "rt"]).copy()

        mean_rt = resp.groupby("participantId")["rt"].mean().reset_index(name="mean_rt")
        return mean_rt


    def merge_mean_rt(df_main: pd.DataFrame, mean_rt: pd.DataFrame, cohort_tag: str) -> pd.DataFrame:
        """
        Merge mean_rt into df_main.
        1) direct merge on participantId
        2) if >90% missing, try suffix key: df_main.participantId + _{cohort_tag} matches mean_rt.participantId
        """
        df_main = df_main.copy()
        mean_rt = mean_rt.copy()

        if "participantId" not in df_main.columns:
            raise ValueError("Main cohort file must contain participantId.")

        df_merged = pd.merge(df_main, mean_rt, on="participantId", how="left")

        miss_rate = df_merged["mean_rt"].isna().mean()
        if miss_rate > 0.90:
            print(f"⚠️ [{cohort_tag}] High missing mean_rt after direct merge ({miss_rate:.1%}). Trying suffix match.")
            df_main["participantId_rtkey"] = df_main["participantId"].astype(str).str.strip() + f"_{cohort_tag}"
            mean_rt["participantId_rtkey"] = mean_rt["participantId"].astype(str).str.strip()

            df_merged = pd.merge(
                df_main,
                mean_rt[["participantId_rtkey", "mean_rt"]],
                on="participantId_rtkey",
                how="left"
            )
            df_merged.drop(columns=["participantId_rtkey"], inplace=True)

        return df_merged


    # -----------------------------
    # Plotting + Correlation reports
    # -----------------------------
    def plot_single_correlation(
        out_dir: str,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        title: str,
        section_number: str,
        cohort_tag: str,
        y_limits: tuple | None = None,   # ✅ Accuracy plot에만 (0,100)
    ):
        df_plot = df[[x_col, y_col]].dropna()
        n = len(df_plot)
        if n < 20:
            print(f"⚠️ [{cohort_tag}] {title}: skipped (N={n} < 20).")
            return

        r_p, p_p = pearsonr(df_plot[x_col], df_plot[y_col])
        r_s, p_s = spearmanr(df_plot[x_col], df_plot[y_col])

        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=df_plot, x=x_col, y=y_col,
            line_kws={"color": "#e91e63"},  # ✅ 그대로 유지
            scatter_kws={"alpha": 0.2, "edgecolor": "none"}
        )
        plt.suptitle(f"({section_number}) {title} [{cohort_tag}]", fontsize=16, fontweight="bold")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if y_limits is not None:
            plt.ylim(y_limits)

        base_fn = f"{section_number}_{safe_slug(title)}_{cohort_tag}"
        out_png = os.path.join(out_dir, f"{base_fn}.png")
        out_svg = os.path.join(out_dir, f"{base_fn}.svg")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_svg, dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved plot: {out_png} / {out_svg}")
        plt.show()

        sig = (p_p < 0.05)
        concl_icon = "✅" if sig else "❌"
        concl_text = "significant" if sig else "not significant"

        report_title = f"({section_number}) Correlation report: {title} [{cohort_tag}]"
        report_content = (
            f"[Overview]\n"
            f"- X: {x_label}\n"
            f"- Y: {y_label}\n"
            f"- N: {n}\n\n"
            f"[Results]\n"
            f"- Pearson r = {r_p:.3f}, p = {p_p:.6g}\n"
            f"- Spearman rho = {r_s:.3f}, p = {p_s:.6g}\n\n"
            f"[Conclusion]\n"
            f"{concl_icon} Pearson correlation is {concl_text} at alpha=0.05.\n"
        )
        print(report_content)
        save_text_report(out_dir, section_number, f"{safe_slug(title)}_report_{cohort_tag}", report_title, report_content)


    def pearson_pvalue_matrix(df_numeric: pd.DataFrame) -> pd.DataFrame:
        """Return a matrix of Pearson p-values aligned with df_numeric columns."""
        cols = df_numeric.columns
        pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    pmat.iloc[i, j] = 0.0
                elif i < j:
                    x = df_numeric[c1].values
                    y = df_numeric[c2].values
                    # dropna pairwise
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() < 3:
                        p = np.nan
                    else:
                        _, p = pearsonr(x[m], y[m])
                    pmat.iloc[i, j] = p
                    pmat.iloc[j, i] = p
        return pmat


    # -----------------------------
    # Main analysis per cohort
    # -----------------------------
    def analyze_cohort(cohort_tag: str, main_path: str, responses_path: str, run_dir: str):
        print(f"\n==================== [{cohort_tag.upper()}] (13) START ====================")

        out_dir = os.path.join(run_dir, "13_correlation")
        ensure_dir(out_dir)

        sns.set_theme(style="whitegrid")

        # Load cohort
        df_main = pd.read_csv(main_path, encoding="utf-8-sig")
        if "participantId" not in df_main.columns:
            raise ValueError(f"{main_path} must contain participantId column.")

        acc_col = find_accuracy_column(df_main)
        df_main, scaling_note = convert_accuracy_to_percent_if_needed(df_main, acc_col)
        print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} | {scaling_note}")

        # Load + compute mean RT
        mean_rt = compute_mean_rt(responses_path)

        # Merge
        df_merged = merge_mean_rt(df_main, mean_rt, cohort_tag=cohort_tag)
        mean_rt_miss = df_merged["mean_rt"].isna().mean()
        print(f"✅ [{cohort_tag}] Loaded+merged. N={len(df_merged)}. mean_rt missing rate = {mean_rt_miss:.1%}")

        # Map AI scores
        score_maps = {
            "exposure_score": {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5},
            "confidence_score": {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5},
            "attitude_score": {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2},
        }
        col_map = {
            "exposure_score": "aiExposureFrequency",
            "confidence_score": "aiConfidence",
            "attitude_score": "aiAttitude",
        }

        for new_col, mapping in score_maps.items():
            orig = col_map[new_col]
            if orig in df_merged.columns:
                df_merged[new_col] = (
                    df_merged[orig].astype(str).str.lower().str.strip().map(mapping)
                )
            else:
                df_merged[new_col] = np.nan

        # Ensure numeric for key vars
        for c in ["age", acc_col, "mean_rt", "exposure_score", "confidence_score", "attitude_score"]:
            if c in df_merged.columns:
                df_merged[c] = pd.to_numeric(df_merged[c], errors="coerce")

        # Save merged raw
        raw_out = os.path.join(out_dir, f"13-0_raw_merged_{cohort_tag}.csv")
        df_merged.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw merged: {raw_out}")

        # (13-1) Correlation heatmap
        print(f"\n--- (13-1) Correlation heatmap [{cohort_tag}] ---")
        corr_vars = {
            "Age": "age",
            "Accuracy": acc_col,
            "MeanRT": "mean_rt",
            "AIExposure": "exposure_score",
            "AIConfidence": "confidence_score",
            "AIAttitude": "attitude_score",
        }

        corr_df = (
            df_merged[list(corr_vars.values())]
            .rename(columns={v: k for k, v in corr_vars.items()})
            .dropna()
        )

        if len(corr_df) < 20:
            print(f"⚠️ [{cohort_tag}] Not enough complete rows for correlation matrix (N={len(corr_df)}). Skipping heatmap.")
            return

        corr_matrix = corr_df.corr(method="pearson")
        p_matrix = pearson_pvalue_matrix(corr_df)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="RdBu_r",   # ✅ 그대로 유지
            center=0,
            vmin=-0.5,
            vmax=0.5,
            ax=ax,
            linewidths=0.5
        )

        # annotate r + stars
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                r = corr_matrix.iloc[i, j]
                p = p_matrix.iloc[i, j]
                stars = "***" if (pd.notna(p) and p < 0.001) else ("**" if (pd.notna(p) and p < 0.01) else ("*" if (pd.notna(p) and p < 0.05) else ""))
                txt = f"{r:.2f}{stars}"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    color="black" if abs(r) < 0.3 else "white",
                    fontsize=10
                )

        plt.title(
            f"(13-1) Correlation matrix (* p<.05, ** p<.01, *** p<.001) [{cohort_tag}]",
            fontsize=16, fontweight="bold", pad=20
        )

        base_fn_13_1 = os.path.join(out_dir, f"13-1_correlation_matrix_heatmap_{cohort_tag}")
        plt.savefig(f"{base_fn_13_1}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{base_fn_13_1}.svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved heatmap: {base_fn_13_1}.png/.svg")
        plt.show()

        report_title_13_1 = f"(13-1) Pearson correlation matrix report [{cohort_tag}]"
        report_content_13_1 = (
            f"[Cohort] {cohort_tag}\n"
            f"[N complete rows] {len(corr_df)}\n"
            f"[Accuracy column] {acc_col}\n"
            f"[Scaling note] {scaling_note}\n\n"
            f"Pearson r matrix:\n{corr_matrix.round(3).to_string()}\n\n"
            f"P-value matrix:\n{p_matrix.round(6).to_string()}\n"
        )
        save_text_report(out_dir, "13-1", f"correlation_matrix_report_{cohort_tag}", report_title_13_1, report_content_13_1)

        # Pairwise scatter analyses
        print(f"\n--- Pairwise scatter analyses [{cohort_tag}] ---")

        # Accuracy(%)가 y축인 plot => y=0~100 고정
        plot_single_correlation(out_dir, df_merged, "age", acc_col, "Age", "Accuracy (%)",
                                "Age vs Accuracy", "13-2", cohort_tag, y_limits=(0, 100))
        plot_single_correlation(out_dir, df_merged, "mean_rt", acc_col, "Mean RT (ms)", "Accuracy (%)",
                                "RT vs Accuracy", "13-7", cohort_tag, y_limits=(0, 100))
        plot_single_correlation(out_dir, df_merged, "exposure_score", acc_col, "AI Exposure Score", "Accuracy (%)",
                                "Exposure vs Accuracy", "13-8", cohort_tag, y_limits=(0, 100))
        plot_single_correlation(out_dir, df_merged, "confidence_score", acc_col, "AI Confidence Score", "Accuracy (%)",
                                "Confidence vs Accuracy", "13-9", cohort_tag, y_limits=(0, 100))
        plot_single_correlation(out_dir, df_merged, "attitude_score", acc_col, "AI Attitude Score", "Accuracy (%)",
                                "Attitude vs Accuracy", "13-10", cohort_tag, y_limits=(0, 100))

        # 나머지는 y축 고정 없음
        plot_single_correlation(out_dir, df_merged, "age", "mean_rt", "Age", "Mean RT (ms)",
                                "Age vs Reaction Time", "13-3", cohort_tag)
        plot_single_correlation(out_dir, df_merged, "age", "exposure_score", "Age", "AI Exposure Score",
                                "Age vs AI Exposure", "13-4", cohort_tag)
        plot_single_correlation(out_dir, df_merged, "age", "confidence_score", "Age", "AI Confidence Score",
                                "Age vs AI Confidence", "13-5", cohort_tag)
        plot_single_correlation(out_dir, df_merged, "age", "attitude_score", "Age", "AI Attitude Score",
                                "Age vs AI Attitude", "13-6", cohort_tag)

        print(f"==================== [{cohort_tag.upper()}] (13) END ====================\n")


    # -----------------------------
    # Run
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(13) Correlation analysis among key variables (MOBILE + WEB) [ENGLISH]")
        print("==============================================================================\n")

        # ✅ run_dir (원하는 경로로 바꿔도 됨)
        run_dir = os.path.join("outputs", "run_20260119_192624")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }
        responses_path = config.RAW_RESPONSES

        for cohort_tag, main_path in cohort_files.items():
            analyze_cohort(cohort_tag, main_path, responses_path, run_dir)


def _run_cell_046():
    # ==============================================================================
    # (14) Accuracy-focused correlation re-analysis (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Goals:
    # - For each cohort (mobile/web):
    #   - Choose Accuracy column robustly (overallAccuracy_y preferred)
    #   - Convert accuracy to % if needed (0-1 -> 0-100)
    #   - Normalize Sex (optional cleaning only, NO FILTERING)
    #   - Map AI exposure/confidence/attitude to numeric scores
    #   - Correlation heatmap centered on Accuracy (RT excluded)
    #   - Scatter plots: Age/Confidence/Exposure/Attitude vs Accuracy
    #   - Accuracy scatter plots: y-axis fixed to 0..100
    # - Save all outputs separately per cohort (keep visual style consistent)
    # ==============================================================================

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr


    # -----------------------------
    # Global plot style (keep consistent)
    # -----------------------------
    sns.set_theme(style="whitegrid")


    def safe_slug(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = s.replace(" ", "_").replace("/", "_")
        return s


    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


    def save_text_report(section_dir: str, section_number: str, base_filename: str, title: str, content: str):
        fn = os.path.join(section_dir, f"{section_number}_{base_filename}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f"{title}\n")
            f.write("============================================================\n\n")
            f.write(str(content))
        print(f"✅ Saved report: {fn}")


    def normalize_sex_from_any(df: pd.DataFrame):
        """
        Normalize sex column from 'sex' or 'gender'. Output column name: 'sex'.
        This is for data cleaning only. No rows are dropped here.
        Returns: (df, sex_source_str)
        """
        sex_source = "missing"
        s = None
    
        if "sex" in df.columns:
            sex_source = "sex"
            s = df["sex"]
        elif "gender" in df.columns:
            sex_source = "gender"
            s = df["gender"]
    
        if s is not None:
            s = s.astype(str).str.lower().str.strip()
            # Map common variations to standard terms
            s = s.replace({
                "nan": np.nan, "none": np.nan, "": np.nan,
                "m": "male", "man": "male",
                "f": "female", "woman": "female",
                "prefer not to say": "prefer_not_to_say",
                "prefer not to": "prefer_not_to_say",
            })
            df["sex"] = s
        else:
            df["sex"] = np.nan
        
        return df, sex_source


    def choose_accuracy_column(df: pd.DataFrame):
        """
        Prefer 'overallAccuracy_y' then 'overallAccuracy' then any column containing 'accuracy'.
        """
        candidates = []
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns:
                candidates.append(c)
        if not candidates:
            for c in df.columns:
                if "accuracy" in c.lower():
                    candidates.append(c)
                    break
        if not candidates:
            raise KeyError("No accuracy-like column found (expected overallAccuracy_y/overallAccuracy).")
        return candidates[0]


    def ensure_accuracy_pct(df: pd.DataFrame, acc_col: str):
        """
        Ensure accuracy is numeric and in percentage scale (0..100).
        Returns (df, used_colname, scaling_note)
        """
        d = df.copy()
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")

        # Heuristic: if most non-NaN values are between 0 and 1.2 -> treat as proportion
        vals = d[acc_col].dropna().values
        if len(vals) == 0:
            # If column exists but all NaN, just return
            return d, "all_nan"
        
        prop_like = np.mean((vals >= 0) & (vals <= 1.2)) > 0.80

        if prop_like:
            d[acc_col] = d[acc_col] * 100.0
            note = f"{acc_col} treated as proportion (0-1) -> converted to %"
        else:
            note = f"{acc_col} treated as % already"

        return d, note


    def plot_single_correlation(
        section_dir: str,
        df: pd.DataFrame,
        x_col: str, y_col: str,
        x_label: str, y_label: str,
        title: str,
        section_number: str,
        cohort_tag: str,
        fix_y_0_100: bool = False
    ):
        df_plot = df[[x_col, y_col]].dropna()
        n = len(df_plot)
        if n < 10:  # Allow slightly smaller N just in case, but warn
            print(f"⚠️ [{cohort_tag}] {title}: skipped (N={n} too small).")
            return

        r_p, p_p = pearsonr(df_plot[x_col], df_plot[y_col])
        r_s, p_s = spearmanr(df_plot[x_col], df_plot[y_col])

        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=df_plot, x=x_col, y=y_col,
            line_kws={"color": "#e91e63"},      # keep style
            scatter_kws={"alpha": 0.3, "edgecolor": "none"}  # keep style
        )

        if fix_y_0_100:
            plt.ylim(0, 105)   # Slightly over 100 for visual clearance

        plt.suptitle(f"({section_number}) {title} [{cohort_tag}]", fontsize=16, fontweight="bold")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        base_fn = f"{section_number}_{safe_slug(title)}_{cohort_tag}"
        out_png = os.path.join(section_dir, f"{base_fn}.png")
        out_svg = os.path.join(section_dir, f"{base_fn}.svg")

        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_svg, dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved plot: {out_png} / {out_svg}")
        plt.show()

        sig = (p_p < 0.05)
        concl_icon = "✅" if sig else "❌"
        concl_text = "significant" if sig else "not significant"

        report_title = f"({section_number}) Correlation report: {title} [{cohort_tag}]"
        report_content = (
            f"[Overview]\n"
            f"- X: {x_label}\n"
            f"- Y: {y_label}\n"
            f"- N: {n}\n\n"
            f"[Results]\n"
            f"- Pearson r = {r_p:.3f}, p = {p_p:.6g}\n"
            f"- Spearman rho = {r_s:.3f}, p = {p_s:.6g}\n\n"
            f"[Conclusion]\n"
            f"{concl_icon} Pearson correlation is {concl_text} at alpha=0.05.\n"
        )
        print(report_content)

        save_text_report(
            section_dir=section_dir,
            section_number=section_number,
            base_filename=f"{safe_slug(title)}_report_{cohort_tag}",
            title=report_title,
            content=report_content
        )


    def analyze_accuracy_focused(cohort_tag: str, main_path: str, run_dir: str):
        print(f"\n==================== [{cohort_tag.upper()}] (14) START ====================")

        section_dir = os.path.join(run_dir, "14_accuracy_focused")
        ensure_dir(section_dir)
        print(f"📁 section14 dir: {section_dir}")

        df = pd.read_csv(main_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {main_path} [{cohort_tag}] (rows={len(df)})")

        # 1. Clean Sex column (for CSV export), but DO NOT FILTER ROWS
        df, sex_source = normalize_sex_from_any(df)
        # ❌ REMOVED FILTERING: df = df[df["sex"].isin(["male", "female"])].copy() 
        print(f"ℹ️ [{cohort_tag}] Sex normalized from '{sex_source}' (No filtering applied).")

        # 2. Choose accuracy column + convert to %
        acc_col = choose_accuracy_column(df)
        df, scale_note = ensure_accuracy_pct(df, acc_col)
        print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} | {scale_note}")
    
        df = df.rename(columns={acc_col: "overallAccuracy"}).copy()

        # 3. Map scores (ordinal)
        score_maps = {
            "exposure_score": {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5},
            "confidence_score": {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5},
            "attitude_score": {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2},
        }
        col_map = {
            "exposure_score": "aiExposureFrequency",
            "confidence_score": "aiConfidence",
            "attitude_score": "aiAttitude",
        }

        for new_col, mapping in score_maps.items():
            orig = col_map[new_col]
            if orig in df.columns:
                df[new_col] = df[orig].astype(str).str.lower().str.strip().map(mapping)
            else:
                df[new_col] = np.nan

        # 4. Ensure numeric
        for c in ["overallAccuracy", "age", "confidence_score", "exposure_score", "attitude_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Save raw prepped (All participants included)
        raw_out = os.path.join(section_dir, f"14-0_raw_prepped_{cohort_tag}.csv")
        df.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw prepped: {raw_out}")

        # --- (14-1) Heatmap (RT excluded) ---
        print(f"\n--- (14-1) Accuracy-focused heatmap [{cohort_tag}] ---")

        corr_vars = {
            "Accuracy": "overallAccuracy",
            "Age": "age",
            "AI Confidence": "confidence_score",
            "AI Exposure": "exposure_score",
            "AI Attitude": "attitude_score",
        }

        # DropNA only for the variables involved in correlation
        corr_df = df[list(corr_vars.values())].rename(columns={v: k for k, v in corr_vars.items()}).dropna()
    
        if len(corr_df) < 10:
            print(f"⚠️ [{cohort_tag}] Not enough complete rows for heatmap (N={len(corr_df)}). Skipping.")
            return

        corr_matrix = corr_df.corr(method="pearson")
        p_matrix = corr_df.corr(method=lambda x, y: pearsonr(x, y)[1])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="RdBu_r",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            ax=ax,
            linewidths=0.5
        )

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                r = corr_matrix.iloc[i, j]
                p = p_matrix.iloc[i, j]
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                ax.text(
                    j + 0.5, i + 0.5, f"{r:.2f}{stars}",
                    ha="center", va="center",
                    color="black" if abs(r) < 0.3 else "white",
                    fontsize=10
                )

        plt.title(
            f"(14-1) Accuracy-focused correlation matrix (* p<.05, ** p<.01, *** p<.001) [{cohort_tag}]",
            fontsize=16, fontweight="bold", pad=20
        )

        base_fn_14_1 = os.path.join(section_dir, f"14-1_accuracy_focused_heatmap_{cohort_tag}")
        plt.savefig(f"{base_fn_14_1}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{base_fn_14_1}.svg", dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved heatmap: {base_fn_14_1}.png/.svg")
        plt.show()

        report_title_14_1 = f"(14-1) Accuracy-focused Pearson correlation matrix report [{cohort_tag}]"
        report_content_14_1 = (
            f"[Cohort] {cohort_tag}\n"
            f"[N complete rows] {len(corr_df)}\n"
            f"[Accuracy col] {acc_col}\n"
            f"[Scaling note] {scale_note}\n"
            f"[Sex source] {sex_source} (Not filtered)\n\n"
            f"Pearson r matrix:\n{corr_matrix.round(3).to_string()}\n\n"
            f"P-value matrix:\n{p_matrix.round(6).to_string()}\n"
        )
        save_text_report(
            section_dir=section_dir,
            section_number="14-1",
            base_filename=f"accuracy_focused_correlation_matrix_report_{cohort_tag}",
            title=report_title_14_1,
            content=report_content_14_1
        )

        # --- (14-2 ~ 14-5) Scatter analyses (Accuracy-related only) ---
        print(f"\n--- (14-2 ~ 14-5) Accuracy-related scatter analyses [{cohort_tag}] ---")

        # ✅ all accuracy-on-y plots fixed to 0..100
        plot_single_correlation(
            section_dir, df,
            "age", "overallAccuracy",
            "Age", "Accuracy (%)",
            "Age vs Accuracy",
            "14-2", cohort_tag,
            fix_y_0_100=True
        )
        plot_single_correlation(
            section_dir, df,
            "confidence_score", "overallAccuracy",
            "AI Confidence Score", "Accuracy (%)",
            "Confidence vs Accuracy",
            "14-3", cohort_tag,
            fix_y_0_100=True
        )
        plot_single_correlation(
            section_dir, df,
            "exposure_score", "overallAccuracy",
            "AI Exposure Score", "Accuracy (%)",
            "Exposure vs Accuracy",
            "14-4", cohort_tag,
            fix_y_0_100=True
        )
        plot_single_correlation(
            section_dir, df,
            "attitude_score", "overallAccuracy",
            "AI Attitude Score", "Accuracy (%)",
            "Attitude vs Accuracy",
            "14-5", cohort_tag,
            fix_y_0_100=True
        )

        print(f"==================== [{cohort_tag.upper()}] (14) END ====================\n")


    if __name__ == "__main__":
        print("==============================================================================")
        print("(14) Accuracy-focused correlation re-analysis (MOBILE + WEB) [ENGLISH]")
        print("==============================================================================\n")

        # ✅ 경로 수정
        run_dir = r"outputs\run_20260119_192624"

        cohort_files = {
            "mobile": rconfig.MOBILE_AGE_FILTERED,
            "web": rconfig.WEB_AGE_FILTERED,
        }

        for cohort_tag, main_path in cohort_files.items():
            if os.path.exists(main_path):
                analyze_accuracy_focused(cohort_tag, main_path, run_dir)
            else:
                print(f"❌ File not found: {main_path}")


def _run_cell_051():
    # ==============================================================================
    # (14 v2) Accuracy-focused correlation re-analysis (MOBILE + WEB) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Fix v2:
    # - Use existing avgRT column as Mean RT (mean_rt)
    # - Heatmap includes Mean RT (placed last in ordering)
    # - Add scatter: RT vs Accuracy
    # - Scatter y-axis (Accuracy) fixed to 0..100 where applicable
    # - Auto-detect accuracy column (overallAccuracy_y preferred), convert 0-1 -> %
    # - Save outputs per cohort
    # ==============================================================================

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    from scipy.stats import pearsonr, spearmanr


    # -----------------------------
    # Utilities
    # -----------------------------
    def safe_slug(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = s.replace(" ", "_").replace("/", "_")
        return s


    def save_text_report(out_dir, section_number, base_filename, title, content):
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, f"{section_number}_{base_filename}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f"{title}\n")
            f.write("============================================================\n\n")
            f.write(str(content))
        print(f"✅ Saved report: {fn}")


    def choose_accuracy_column(df: pd.DataFrame) -> str:
        """
        Prefer overallAccuracy_y if present (your pipeline convention),
        else fall back to overallAccuracy.
        """
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        raise KeyError("No accuracy column found: expected overallAccuracy_y or overallAccuracy.")


    def ensure_accuracy_percent(series: pd.Series):
        """
        If accuracy looks like proportion (0..1.x), convert to percent.
        Returns (converted_series, scaling_note)
        """
        s = pd.to_numeric(series, errors="coerce")
        s_nonan = s.dropna()
        if len(s_nonan) == 0:
            return s, "accuracy all NaN (no scaling applied)"
        # Heuristic: if max <= 1.2 and min >= -0.2 -> treat as proportion
        if (s_nonan.max() <= 1.2) and (s_nonan.min() >= -0.2):
            return s * 100.0, "treated as proportion (0-1) -> converted to %"
        return s, "treated as percent scale (no conversion)"


    def plot_single_correlation(
        out_dir,
        df,
        x_col, y_col,
        x_label, y_label,
        title, section_number,
        cohort_tag,
        y_is_accuracy=False,
        acc_ylim=(0, 100)
    ):
        df_plot = df[[x_col, y_col]].dropna()
        n = len(df_plot)
        if n < 20:
            print(f"⚠️ [{cohort_tag}] {title}: skipped (N={n} < 20).")
            return

        r_p, p_p = pearsonr(df_plot[x_col], df_plot[y_col])
        r_s, p_s = spearmanr(df_plot[x_col], df_plot[y_col])

        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=df_plot, x=x_col, y=y_col,
            line_kws={"color": "#e91e63"},
            scatter_kws={"alpha": 0.2, "edgecolor": "none"}
        )
        plt.suptitle(f"({section_number} v2) {title} [{cohort_tag}]", fontsize=16, fontweight="bold")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # ✅ Critical: accuracy y-axis fixed
        if y_is_accuracy:
            plt.ylim(acc_ylim[0], acc_ylim[1])

        base_fn = f"{section_number}_v2_{safe_slug(title)}_{cohort_tag}"
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"{base_fn}.png")
        svg_path = os.path.join(out_dir, f"{base_fn}.svg")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(svg_path, dpi=300, bbox_inches="tight")
        print(f"✅ [{cohort_tag}] Saved plot: {png_path} / {svg_path}")
        plt.show()

        sig = (p_p < 0.05)
        concl_icon = "✅" if sig else "❌"
        concl_text = "significant" if sig else "not significant"

        report_title = f"({section_number} v2) Correlation report: {title} [{cohort_tag}]"
        report_content = (
            f"[Overview]\n"
            f"- X: {x_label}\n"
            f"- Y: {y_label}\n"
            f"- N: {n}\n\n"
            f"[Results]\n"
            f"- Pearson r = {r_p:.3f}, p = {p_p:.6g}\n"
            f"- Spearman rho = {r_s:.3f}, p = {p_s:.6g}\n\n"
            f"[Conclusion]\n"
            f"{concl_icon} Pearson correlation is {concl_text} at alpha=0.05.\n"
        )
        print(report_content)
        save_text_report(out_dir, f"{section_number}-report", f"v2_{safe_slug(title)}_{cohort_tag}", report_title, report_content)


    # -----------------------------
    # Main analysis
    # -----------------------------
    def analyze_accuracy_focused_v2(cohort_tag: str, main_path: str, run_dir: str):
        out_dir = os.path.join(run_dir, "14_accuracy_focused_v2")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n==================== [{cohort_tag.upper()}] (14 v2) START ====================")
        print(f"📁 section14(v2) dir: {out_dir}")

        df = pd.read_csv(main_path, encoding="utf-8-sig")
        print(f"✅ Loaded: {main_path} [{cohort_tag}] (rows={len(df)})")

        # --- Accuracy column: choose + convert scale if needed ---
        acc_col = choose_accuracy_column(df)
        df = df.copy()
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        df["accuracy_pct"], acc_note = ensure_accuracy_percent(df[acc_col])
        print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} | {acc_note}")

        # --- Mean RT: use avgRT if present ---
        if "avgRT" in df.columns:
            df["mean_rt"] = pd.to_numeric(df["avgRT"], errors="coerce")
            print(f"✅ [{cohort_tag}] Using avgRT as mean_rt.")
        elif "mean_rt" in df.columns:
            df["mean_rt"] = pd.to_numeric(df["mean_rt"], errors="coerce")
            print(f"✅ [{cohort_tag}] Using existing mean_rt.")
        else:
            df["mean_rt"] = np.nan
            print(f"⚠️ [{cohort_tag}] No avgRT/mean_rt column found. mean_rt will be NaN.")

        # Map scores (ordinal)
        score_maps = {
            "exposure_score": {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5},
            "confidence_score": {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5},
            "attitude_score": {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2},
        }
        col_map = {
            "exposure_score": "aiExposureFrequency",
            "confidence_score": "aiConfidence",
            "attitude_score": "aiAttitude",
        }

        for new_col, mapping in score_maps.items():
            orig = col_map[new_col]
            if orig in df.columns:
                df[new_col] = df[orig].astype(str).str.lower().str.strip().map(mapping)
            else:
                df[new_col] = np.nan

        # Ensure numeric for key vars
        for c in ["age", "confidence_score", "exposure_score", "attitude_score", "mean_rt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Save raw prepped (useful for debugging/Prism)
        raw_out = os.path.join(out_dir, f"14-0_v2_raw_prepped_{cohort_tag}.csv")
        df.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw prepped: {raw_out}")

        # --- (14-1 v2) Heatmap: Accuracy + Age + AI scores + Mean RT (last) ---
        print(f"\n--- (14-1 v2) Accuracy-focused heatmap + Mean RT [{cohort_tag}] ---")

        corr_vars = {
            "Accuracy": "accuracy_pct",
            "Age": "age",
            "AI Confidence": "confidence_score",
            "AI Exposure": "exposure_score",
            "AI Attitude": "attitude_score",
            "Mean RT": "mean_rt",
        }

        corr_df = df[list(corr_vars.values())].rename(columns={v: k for k, v in corr_vars.items()}).dropna()
        if len(corr_df) < 20:
            print(f"⚠️ [{cohort_tag}] Not enough complete rows for heatmap (N={len(corr_df)}). Skipping heatmap.")
        else:
            corr_matrix = corr_df.corr(method="pearson")
            p_matrix = corr_df.corr(method=lambda x, y: pearsonr(x, y)[1])

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, annot=False, cmap="RdBu_r", center=0,
                vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5
            )

            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    r = corr_matrix.iloc[i, j]
                    p = p_matrix.iloc[i, j]
                    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                    ax.text(
                        j + 0.5, i + 0.5, f"{r:.2f}{stars}",
                        ha="center", va="center",
                        color="black" if abs(r) < 0.3 else "white"
                    )

            plt.title(
                f"(14-1 v2) Correlation Matrix Centered on Accuracy [{cohort_tag}]",
                fontsize=16, fontweight="bold", pad=20
            )

            base_fn_14_1 = os.path.join(out_dir, f"14-1_v2_accuracy_focused_heatmap_{cohort_tag}")
            plt.savefig(f"{base_fn_14_1}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{base_fn_14_1}.svg", dpi=300, bbox_inches="tight")
            print(f"✅ [{cohort_tag}] Saved heatmap: {base_fn_14_1}.png/.svg")
            plt.show()

            report_title_14_1 = f"(14-1 v2) Accuracy-focused Pearson correlation matrix report [{cohort_tag}]"
            report_content_14_1 = (
                f"[Cohort] {cohort_tag}\n"
                f"[N complete rows] {len(corr_df)}\n"
                f"[Accuracy col] {acc_col}\n"
                f"[Accuracy scaling] {acc_note}\n\n"
                f"Pearson r matrix:\n{corr_matrix.round(3).to_string()}\n\n"
                f"P-value matrix:\n{p_matrix.round(6).to_string()}\n"
            )
            save_text_report(out_dir, "14-1_v2", f"accuracy_focused_matrix_{cohort_tag}", report_title_14_1, report_content_14_1)

        # --- Scatter analyses ---
        print(f"\n--- (14-2 v2 ~ 14-6 v2) Accuracy-related scatter analyses [{cohort_tag}] ---")
        plot_single_correlation(out_dir, df, "age", "accuracy_pct", "Age", "Accuracy (%)",
                                "Age vs Accuracy", "14-2", cohort_tag, y_is_accuracy=True)
        plot_single_correlation(out_dir, df, "confidence_score", "accuracy_pct", "AI Confidence Score", "Accuracy (%)",
                                "Confidence vs Accuracy", "14-3", cohort_tag, y_is_accuracy=True)
        plot_single_correlation(out_dir, df, "exposure_score", "accuracy_pct", "AI Exposure Score", "Accuracy (%)",
                                "Exposure vs Accuracy", "14-4", cohort_tag, y_is_accuracy=True)
        plot_single_correlation(out_dir, df, "attitude_score", "accuracy_pct", "AI Attitude Score", "Accuracy (%)",
                                "Attitude vs Accuracy", "14-5", cohort_tag, y_is_accuracy=True)
        plot_single_correlation(out_dir, df, "mean_rt", "accuracy_pct", "Mean RT", "Accuracy (%)",
                                "RT vs Accuracy", "14-6", cohort_tag, y_is_accuracy=True)

        print(f"==================== [{cohort_tag.upper()}] (14 v2) END ====================\n")


    if __name__ == "__main__":
        print("==============================================================================")
        print("(14 v2) Accuracy-focused correlation re-analysis (MOBILE + WEB) [ENGLISH]")
        print("==============================================================================\n")

        RUN_DIR = r"outputs\run_20260119_192624"

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        for cohort_tag, main_path in cohort_files.items():
            analyze_accuracy_focused_v2(cohort_tag, main_path, RUN_DIR)


def _run_cell_055():
    # ==============================================================================
    # (15) Correlation Network Visualization (Emphasized) - MOBILE + WEB [ENGLISH]
    # ------------------------------------------------------------------------------
    # - Use run_dir consistent with prior sections
    # - Robust handling:
    #   * choose accuracy column (overallAccuracy_y preferred)
    #   * convert accuracy to % if it looks like 0~1
    #   * create exposure/confidence/attitude scores from raw string columns
    # - Show ALL edges:
    #   * significant: red/blue + stars
    #   * non-significant: gray + r only
    # - Prevent node clipping: circular scale + margins + pad_inches
    # - Save png/svg + matrices + summary per cohort folder
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
    RUN_DIR = r"outputs\run_20260119_192624"    # ✅ keep consistent with your project
    SECTION_DIR = os.path.join(RUN_DIR, "15_corr_network")
    ALPHA = 0.05

    # -----------------------------
    # Style (big like your older emphasized version)
    # -----------------------------
    NODE_SIZE = 60000
    NODE_COLOR = "skyblue"
    NODE_ALPHA = 0.92
    NODE_LABEL_FONTSIZE = 36

    EDGE_LABEL_FONTSIZE = 30

    # Edge thickness ranges (make non-sig visible)
    SIG_MIN, SIG_MAX = 8, 26
    NS_MIN,  NS_MAX  = 5, 14

    # Layout / clipping fixes
    LAYOUT = "circular"     # "circular" | "spring" | "kamada_kawai"
    CIRCULAR_SCALE = 0.45   # smaller => nodes inward (helps prevent clipping)
    AX_MARGINS = 0.22       # extra padding around axis limits
    SAVE_PAD_INCHES = 0.80  # extra whitespace around tight bbox

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def choose_accuracy_column(df: pd.DataFrame):
        """
        Prefer overallAccuracy_y if exists; else overallAccuracy.
        Return (col, note).
        """
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y", "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy", "overallAccuracy"
        raise KeyError("No accuracy column found: expected overallAccuracy_y or overallAccuracy")

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str):
        """
        Convert accuracy to percent if it looks like proportion (0~1).
        Returns (series, note).
        """
        s = pd.to_numeric(df[acc_col], errors="coerce")
        s_nonan = s.dropna()
        if len(s_nonan) == 0:
            return s, "accuracy empty"
        # heuristic: if max <= 1.5 and mean <= 1.0 -> treat as proportion
        if (s_nonan.max() <= 1.5) and (s_nonan.mean() <= 1.0):
            return s * 100.0, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return s, f"{acc_col} treated as percent already"

    def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create numeric score columns from raw columns if needed:
          - exposure_score from aiExposureFrequency
          - confidence_score from aiConfidence
          - attitude_score from aiAttitude
        Robust: lower/strip before map.
        """
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

    def build_corr_network(df: pd.DataFrame, vars_map: dict, alpha: float = 0.05):
        """
        Build a network including ALL edges (sig + non-sig).
        Skips edges where r or p is NaN (e.g., constant columns).
        Returns:
          G, corr_matrix, p_matrix, edge_labels, n_complete, corr_df
        """
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

    def draw_emphasized_network(
        G: nx.Graph,
        edge_labels: dict,
        title: str,
        out_base: str,
        layout: str = "circular",
    ):
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
    print("(15) Correlation Network Visualization (Emphasized) - MOBILE + WEB [ENGLISH]")
    print("==============================================================================\n")

    ensure_dir(SECTION_DIR)

    # Variables in the network (labels -> internal column names)
    corr_vars = {
        "Age": "age",
        "Accuracy": "accuracy_pct",          # <- we will create this
        "AI Exposure": "exposure_score",
        "AI Confidence": "confidence_score",
        "AI Attitude": "attitude_score",
    }

    cohorts = [
        ("mobile", config.MOBILE_AGE_FILTERED),
        ("web",    config.WEB_AGE_FILTERED),
    ]

    for cohort_tag, file_path in cohorts:
        print(f"\n==================== [{cohort_tag.upper()}] (15) START ====================")
        cohort_out = os.path.join(SECTION_DIR, cohort_tag)
        ensure_dir(cohort_out)

        # load
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path}")
            continue

        # accuracy column selection + scaling
        try:
            acc_col, acc_note = choose_accuracy_column(df)
            acc_pct, scale_note = ensure_accuracy_percent(df, acc_col)
            df = df.copy()
            df["accuracy_pct"] = acc_pct
            print(f"✅ [{cohort_tag}] Accuracy column used: {acc_note} | {scale_note}")
        except KeyError as e:
            print(f"❌ [{cohort_tag}] {e}")
            continue

        # scores
        df = ensure_score_columns(df)

        # numeric safety
        for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # save prepped raw
        raw_out = os.path.join(cohort_out, f"15-0_raw_prepped_{cohort_tag}.csv")
        df.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"✅ [{cohort_tag}] Saved raw prepped: {raw_out}")

        # build network
        try:
            G, corr_matrix, p_matrix, edge_labels, n_complete, corr_df = build_corr_network(
                df=df, vars_map=corr_vars, alpha=ALPHA
            )
        except KeyError as e:
            print(f"❌ Column error: {e}")
            continue

        # diagnostics: constant columns
        if corr_df is not None:
            nunique = corr_df.nunique()
            const_cols = nunique[nunique <= 1].index.tolist()
            if const_cols:
                print(f"⚠️ [{cohort_tag}] Constant columns after dropna (correlation becomes NaN): {const_cols}")

        print(f"✅ [{cohort_tag}] Complete-case N for network: {n_complete}")

        if G is None or len(G.edges()) == 0:
            print(f"⚠️ [{cohort_tag}] No drawable edges (NaN correlations / too few complete cases).")
            continue

        sig_n = sum(1 for u, v in G.edges() if G[u][v]["significant"])
        ns_n = len(G.edges()) - sig_n
        print(f"Edges total={len(G.edges())} | significant={sig_n} | non-significant={ns_n}")

        # draw + save
        title = f"(15-1) Correlation Network (ALL edges, alpha={ALPHA}) [{cohort_tag.upper()}]"
        out_base = os.path.join(cohort_out, f"15-1_correlation_network_all_edges_{cohort_tag}")
        draw_emphasized_network(G, edge_labels, title, out_base, layout=LAYOUT)

        # save matrices
        corr_out = os.path.join(cohort_out, f"15-2_corr_matrix_{cohort_tag}.csv")
        p_out = os.path.join(cohort_out, f"15-3_p_matrix_{cohort_tag}.csv")
        corr_matrix.to_csv(corr_out, encoding="utf-8-sig")
        p_matrix.to_csv(p_out, encoding="utf-8-sig")
        print(f"✅ Saved matrices: {corr_out}, {p_out}")

        # save summary
        summary_txt = (
            f"(15) Network summary [{cohort_tag}]\n"
            f"- complete-case N: {n_complete}\n"
            f"- nodes: {len(G.nodes())}\n"
            f"- edges total: {len(G.edges())}\n"
            f"- significant edges (p<{ALPHA}): {sig_n}\n"
            f"- non-significant edges: {ns_n}\n"
            f"- layout: {LAYOUT} (circular_scale={CIRCULAR_SCALE}, ax_margins={AX_MARGINS}, pad_inches={SAVE_PAD_INCHES})\n"
            f"- accuracy: {acc_col} -> accuracy_pct (% scale)\n"
        )
        summary_path = os.path.join(cohort_out, f"15-4_network_summary_{cohort_tag}.txt")
        save_text(summary_path, summary_txt)
        print(f"✅ Saved summary: {summary_path}")

    print("\n==================== (15) DONE ====================")


def _run_cell_057():

    # ==============================================================================
    # (15 v2) Correlation Network Visualization (Emphasized + Label Offset) - MOBILE + WEB [ENGLISH]
    # ------------------------------------------------------------------------------
    # - Based on Section 14 v2 variables (+ Mean RT via avgRT)
    # - Show ALL edges:
    #     significant -> red/blue + stars
    #     non-significant -> gray + r only
    # - Fix: prevent node clipping (shrink circular radius + margins + pad_inches + explicit xlim/ylim padding)
    # - Fix: reduce edge-label overlaps (perpendicular offset labeling)
    # - Fix: choose correct accuracy column + percent conversion (overallAccuracy_y preferred)
    # - Save png/svg + matrices + summary (per cohort folder under run_dir)
    # ==============================================================================

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import networkx as nx

    # -----------------------------
    # Config
    # -----------------------------
    RUN_DIR = r"outputs\run_20260119_192624"  # ✅ keep consistent
    SECTION_DIR = os.path.join(RUN_DIR, "15_corr_network_v2")

    ALPHA = 0.05

    # Node/Label style
    NODE_SIZE = 60000
    NODE_COLOR = "skyblue"
    NODE_ALPHA = 0.92
    NODE_LABEL_FONTSIZE = 36

    # Edge label font (non-sig is slightly smaller automatically)
    EDGE_LABEL_FONTSIZE = 30

    # Edge thickness ranges
    SIG_MIN, SIG_MAX = 8, 26
    NS_MIN,  NS_MAX  = 5, 14

    # Layout / clipping fixes
    LAYOUT = "circular"        # "circular" | "spring" | "kamada_kawai"
    CIRCULAR_SCALE = 0.45      # smaller => nodes move inward (prevents clipping)
    AX_MARGINS = 0.22          # adds padding around axis limits
    SAVE_PAD_INCHES = 0.85     # extra whitespace around tight bbox
    POS_PAD_RATIO = 0.10       # extra padding using pos range (prevents edge label cutoff)

    # Label overlap reduction (perpendicular offset)
    LABEL_BASE_OFF = 0.030
    LABEL_EXTRA_CENTER_1 = 0.070   # when close to center
    LABEL_EXTRA_CENTER_2 = 0.040   # mid-close
    CENTER_THRESH_1 = 0.16
    CENTER_THRESH_2 = 0.28
    LABEL_JITTER = 0.012           # small jitter to avoid stacking


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
        Prefer overallAccuracy_y if exists; else overallAccuracy.
        """
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        raise KeyError("No accuracy column found: expected overallAccuracy_y or overallAccuracy")

    def ensure_accuracy_percent(series: pd.Series):
        """
        Convert accuracy to percent if it looks like 0~1.
        """
        s = pd.to_numeric(series, errors="coerce")
        s_nonan = s.dropna()
        if len(s_nonan) == 0:
            return s, "accuracy empty"
        if (s_nonan.max() <= 1.5) and (s_nonan.mean() <= 1.0):
            return s * 100.0, "treated as proportion (0-1) -> converted to %"
        return s, "treated as percent already"

    def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric score columns if missing."""
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        if "exposure_score" not in df.columns:
            if "aiExposureFrequency" in df.columns:
                df["exposure_score"] = df["aiExposureFrequency"].astype(str).str.lower().str.strip().map(exposure_map)
            else:
                df["exposure_score"] = np.nan

        if "confidence_score" not in df.columns:
            if "aiConfidence" in df.columns:
                df["confidence_score"] = df["aiConfidence"].astype(str).str.lower().str.strip().map(confidence_map)
            else:
                df["confidence_score"] = np.nan

        if "attitude_score" not in df.columns:
            if "aiAttitude" in df.columns:
                df["attitude_score"] = df["aiAttitude"].astype(str).str.lower().str.strip().map(attitude_map)
            else:
                df["attitude_score"] = np.nan

        return df

    def safe_pearson_p(x: pd.Series, y: pd.Series) -> float:
        """Pearson p-value with constant-column safety."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 3:
            return np.nan
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan
        try:
            return float(pearsonr(x, y)[1])
        except Exception:
            return np.nan

    def build_corr_network(df: pd.DataFrame, vars_map: dict, alpha: float = 0.05):
        """
        Build a network including ALL edges (sig + non-sig).
        Robust to missing columns: will drop missing vars and proceed.
        Skips edges where r or p is NaN (e.g., constant columns).
        """
        existing = {name: col for name, col in vars_map.items() if col in df.columns}
        missing = [name for name, col in vars_map.items() if col not in df.columns]
        if missing:
            print(f"⚠️ Missing vars (dropped from network): {missing}")

        if len(existing) < 3:
            return None, None, None, None, 0, None, existing, missing

        needed_cols = list(existing.values())
        corr_df = df[needed_cols].rename(columns={v: k for k, v in existing.items()}).dropna()
        n_complete = len(corr_df)
        if n_complete < 20:
            return None, None, None, None, n_complete, corr_df, existing, missing

        corr_matrix = corr_df.corr(method="pearson")

        cols = list(corr_df.columns)
        p_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i == j:
                    p_matrix.iloc[i, j] = 0.0
                elif j < i:
                    p = safe_pearson_p(corr_df.iloc[:, i], corr_df.iloc[:, j])
                    p_matrix.iloc[i, j] = p
                    p_matrix.iloc[j, i] = p

        G = nx.Graph()
        G.add_nodes_from(cols)

        edge_labels = {}
        for i in range(len(cols)):
            for j in range(i):
                n1, n2 = cols[i], cols[j]
                r_val = float(corr_matrix.loc[n1, n2])
                p_val = float(p_matrix.loc[n1, n2])

                if not np.isfinite(r_val) or not np.isfinite(p_val):
                    continue

                significant = (p_val < alpha)
                G.add_edge(
                    n1, n2,
                    weight=float(abs(r_val)),
                    r=r_val,
                    p=p_val,
                    significant=bool(significant),
                )

                if significant:
                    stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                    edge_labels[(n1, n2)] = f"{r_val:.2f}{stars}"
                else:
                    edge_labels[(n1, n2)] = f"{r_val:.2f}"

        return G, corr_matrix, p_matrix, edge_labels, n_complete, corr_df, existing, missing

    def make_layout(G: nx.Graph, layout: str):
        if layout == "spring":
            return nx.spring_layout(G, seed=42, k=0.55)
        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        return nx.circular_layout(G, scale=CIRCULAR_SCALE)

    def stable_side(u: str, v: str) -> int:
        """Deterministic side selection (no Python hash salt issues)."""
        key = f"{min(u,v)}|{max(u,v)}"
        s = sum(ord(c) for c in key)
        return 1 if (s % 2 == 0) else -1

    def draw_edge_labels_offset(ax, G, pos, edge_labels, base_font=30):
        """
        Draw edge labels with perpendicular offsets to reduce overlaps near the center.
        Non-significant labels are slightly smaller.
        """
        items = sorted(edge_labels.items(), key=lambda kv: (min(kv[0]), max(kv[0])))

        for k, ((u, v), lab) in enumerate(items):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            dx, dy = (x2 - x1), (y2 - y1)
            L = float(np.hypot(dx, dy))
            if L == 0:
                continue

            px, py = (-dy / L), (dx / L)

            center_dist = float(np.hypot(xm, ym))
            extra = 0.0
            if center_dist < CENTER_THRESH_1:
                extra = LABEL_EXTRA_CENTER_1
            elif center_dist < CENTER_THRESH_2:
                extra = LABEL_EXTRA_CENTER_2

            side = stable_side(u, v)
            jitter = (k % 3) * LABEL_JITTER
            off = LABEL_BASE_OFF + extra + jitter

            sig = bool(G[u][v].get("significant", False))
            fs = base_font if sig else int(round(base_font * 0.85))
            bbox = dict(facecolor="white", alpha=(0.45 if sig else 0.30), edgecolor="none")

            ax.text(
                xm + side * off * px,
                ym + side * off * py,
                lab,
                ha="center", va="center",
                fontsize=fs,
                rotation=0,
                bbox=bbox,
            )

    def draw_emphasized_network(G, edge_labels, title, out_base, layout="circular"):
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        if not weights:
            raise ValueError("No edges to draw.")

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
                sig_colors.append("#DB4437" if r_val > 0 else "#4285F4")
            else:
                ns_edges.append((u, v))
                ns_widths.append(NS_MIN + w_norm * (NS_MAX - NS_MIN))
                ns_colors.append("#BDBDBD")

        fig, ax = plt.subplots(figsize=(24, 24))

        if ns_edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=ns_edges, width=ns_widths,
                                   edge_color=ns_colors, alpha=0.55)
        if sig_edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=sig_edges, width=sig_widths,
                                   edge_color=sig_colors, alpha=0.85)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color=NODE_COLOR, alpha=NODE_ALPHA)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=NODE_LABEL_FONTSIZE, font_weight="bold")

        draw_edge_labels_offset(ax, G, pos, edge_labels, base_font=EDGE_LABEL_FONTSIZE)

        ax.set_title(title, fontsize=32, fontweight="bold", pad=20)
        ax.set_axis_off()

        ax.margins(AX_MARGINS)
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        xr = xs.max() - xs.min()
        yr = ys.max() - ys.min()
        pad = POS_PAD_RATIO * max(xr, yr)
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

        fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
        fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
        plt.show()
        plt.close(fig)


    # -----------------------------
    # Main
    # -----------------------------
    print("==============================================================================")
    print("(15 v2) Correlation Network Visualization (Emphasized + Label Offset) - MOBILE + WEB [ENGLISH]")
    print("==============================================================================\n")

    ensure_dir(SECTION_DIR)

    # include Mean RT via avgRT (if present)
    # Accuracy will be created as accuracy_pct from overallAccuracy_y or overallAccuracy
    corr_vars = {
        "Age": "age",
        "Accuracy": "accuracy_pct",
        "AI Exposure": "exposure_score",
        "AI Confidence": "confidence_score",
        "AI Attitude": "attitude_score",
        "Mean RT": "avgRT",
    }

    cohorts = [
        ("mobile", config.MOBILE_AGE_FILTERED),
        ("web",    config.WEB_AGE_FILTERED),
    ]

    for cohort_tag, file_path in cohorts:
        print(f"\n==================== [{cohort_tag.upper()}] (15 v2) START ====================")
        cohort_out = os.path.join(SECTION_DIR, cohort_tag)
        ensure_dir(cohort_out)

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path}")
            continue

        df = ensure_score_columns(df).copy()

        # --- Accuracy handling (consistent with your Section 14/13 logic) ---
        try:
            acc_col = choose_accuracy_column(df)
            acc_pct, acc_note = ensure_accuracy_percent(df[acc_col])
            df["accuracy_pct"] = acc_pct
            print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} | {acc_note}")
        except KeyError as e:
            print(f"❌ [{cohort_tag}] {e}")
            continue

        # numeric safety
        for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score", "avgRT"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # save raw prepped
        raw_out = os.path.join(cohort_out, f"15v2-0_raw_prepped_{cohort_tag}.csv")
        df.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"✅ Saved raw: {raw_out}")

        # build network
        G, corr_matrix, p_matrix, edge_labels, n_complete, corr_df, existing, missing = build_corr_network(
            df=df, vars_map=corr_vars, alpha=ALPHA
        )

        print(f"✅ [{cohort_tag}] Complete-case N for network: {n_complete}")

        if G is None or len(G.edges()) == 0:
            print(f"⚠️ [{cohort_tag}] No drawable edges (NaN correlations / too few complete cases).")
            continue

        sig_n = sum(1 for u, v in G.edges() if G[u][v]["significant"])
        ns_n = len(G.edges()) - sig_n
        print(f"Edges total={len(G.edges())} | significant={sig_n} | non-significant={ns_n}")

        # draw + save
        title = f"(15-1 v2) Correlation Network (ALL edges, alpha={ALPHA}) [{cohort_tag.upper()}]"
        out_base = os.path.join(cohort_out, f"15v2-1_correlation_network_all_edges_{cohort_tag}")
        draw_emphasized_network(G, edge_labels, title, out_base, layout=LAYOUT)

        # save matrices
        corr_out = os.path.join(cohort_out, f"15v2-2_corr_matrix_{cohort_tag}.csv")
        p_out = os.path.join(cohort_out, f"15v2-3_p_matrix_{cohort_tag}.csv")
        corr_matrix.to_csv(corr_out, encoding="utf-8-sig")
        p_matrix.to_csv(p_out, encoding="utf-8-sig")
        print(f"✅ Saved matrices: {corr_out}, {p_out}")

        # save summary
        used_vars = list(existing.keys())
        summary_txt = (
            f"(15 v2) Network summary [{cohort_tag}]\n"
            f"- complete-case N: {n_complete}\n"
            f"- nodes: {len(G.nodes())}\n"
            f"- edges total: {len(G.edges())}\n"
            f"- significant edges (p<{ALPHA}): {sig_n}\n"
            f"- non-significant edges: {ns_n}\n"
            f"- layout: {LAYOUT} (circular_scale={CIRCULAR_SCALE}, ax_margins={AX_MARGINS}, pad_inches={SAVE_PAD_INCHES}, pos_pad_ratio={POS_PAD_RATIO})\n"
            f"- label_offset: perpendicular + center-aware extra + jitter\n"
            f"- accuracy: {acc_col} -> accuracy_pct (%)\n"
            f"- vars_used: {used_vars}\n"
            f"- vars_dropped_missing: {missing}\n"
        )
        summary_path = os.path.join(cohort_out, f"15v2-4_network_summary_{cohort_tag}.txt")
        save_text(summary_path, summary_txt)
        print(f"✅ Saved summary: {summary_path}")

    print("\n==================== (15 v2) DONE ====================")


def _run_cell_060():
    # ==============================================================================
    # (16) Parallel Mediation: Age -> (AI Exposure, AI Confidence) -> Accuracy
    # ------------------------------------------------------------------------------
    # Fix:
    # - Choose correct accuracy column (prefer overallAccuracy_y) + convert 0-1 to %
    # - Map exposure/confidence robustly (lower/strip)
    # - Save under run_dir/16_mediation (consistent with your pipeline)
    # - Print/display key outputs (not silent)
    # Output: CSV results + English report per cohort (MOBILE/WEB).
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd

    # -----------------------------
    # Config
    # -----------------------------
    RUN_DIR = r"outputs\run_20260119_192624"
    SECTION_DIR = os.path.join(RUN_DIR, "16_mediation_parallel")
    N_BOOT = 5000
    SEED = 42

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def choose_accuracy_column(df: pd.DataFrame):
        """Prefer overallAccuracy_y if exists, else overallAccuracy."""
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        raise KeyError("No accuracy column found: expected overallAccuracy_y or overallAccuracy")

    def ensure_accuracy_percent(series: pd.Series):
        """Convert accuracy to percent if it looks like 0~1."""
        s = pd.to_numeric(series, errors="coerce")
        s_nonan = s.dropna()
        if len(s_nonan) == 0:
            return s, "accuracy empty"
        if (s_nonan.max() <= 1.5) and (s_nonan.mean() <= 1.0):
            return s * 100.0, "treated as proportion (0-1) -> converted to %"
        return s, "treated as percent already"

    def ensure_scores_and_accuracy(df: pd.DataFrame, cohort_tag: str) -> pd.DataFrame:
        """
        - Create exposure_score / confidence_score (ordinal)
        - Create accuracy_pct from correct accuracy column (overallAccuracy_y preferred)
        """
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}

        df = df.copy()

        # exposure_score
        if "exposure_score" not in df.columns:
            if "aiExposureFrequency" in df.columns:
                df["exposure_score"] = (
                    df["aiExposureFrequency"].astype(str).str.lower().str.strip().map(exposure_map)
                )
            else:
                df["exposure_score"] = np.nan

        # confidence_score
        if "confidence_score" not in df.columns:
            if "aiConfidence" in df.columns:
                df["confidence_score"] = (
                    df["aiConfidence"].astype(str).str.lower().str.strip().map(confidence_map)
                )
            else:
                df["confidence_score"] = np.nan

        # accuracy_pct
        acc_col = choose_accuracy_column(df)
        df["accuracy_pct"], acc_note = ensure_accuracy_percent(df[acc_col])
        print(f"✅ [{cohort_tag}] Accuracy column used: {acc_col} | {acc_note}")

        # numeric safety
        for c in ["age", "exposure_score", "confidence_score", "accuracy_pct"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def pick_indirect_rows(res: pd.DataFrame, mediator_name: str) -> pd.DataFrame:
        """
        Pingouin 'path' labels vary by version.
        Try to robustly locate the indirect effect row for a given mediator.
        """
        if "path" not in res.columns:
            return pd.DataFrame()

        m = mediator_name.lower()

        # Most common: path contains "Indirect" and mediator name
        path_lower = res["path"].astype(str).str.lower()
        mask = path_lower.str.contains("indirect") & path_lower.str.contains(m)
        out = res[mask].copy()
        if not out.empty:
            return out

        # Fallback: sometimes path is "Indirect" and mediator in another column
        if "mediator" in res.columns:
            med_lower = res["mediator"].astype(str).str.lower()
            mask2 = path_lower.str.contains("indirect") & (med_lower == m)
            out2 = res[mask2].copy()
            return out2

        return pd.DataFrame()

    def interpret_indirect(df_row: pd.DataFrame) -> str:
        """Significant if 0 is NOT inside 95% bootstrap CI."""
        if df_row.empty:
            return "Indirect effect row not found in the output table."

        r = df_row.iloc[0]

        if ("CI[2.5%]" not in df_row.columns) or ("CI[97.5%]" not in df_row.columns):
            return "CI columns not found in the output table (cannot interpret significance)."

        lo = float(r["CI[2.5%]"])
        hi = float(r["CI[97.5%]"])
        sig = not (lo <= 0 <= hi)

        return f"95% CI = [{lo:.4f}, {hi:.4f}] -> {'SIGNIFICANT (does not include 0)' if sig else 'NOT significant (includes 0)'}"


    # -----------------------------
    # Main
    # -----------------------------
    print("==============================================================================")
    print("(16) Parallel Mediation Analysis (MOBILE + WEB) [ENGLISH]")
    print("==============================================================================\n")

    try:
        import pingouin as pg
    except ImportError:
        raise ImportError(
            "pingouin is not installed. Install it first:\n"
            "  pip install pingouin\n"
            "or (conda):\n"
            "  conda install -c conda-forge pingouin"
        )

    ensure_dir(SECTION_DIR)

    cohorts = [
        ("mobile", config.MOBILE_AGE_FILTERED),
        ("web",    config.WEB_AGE_FILTERED),
    ]

    for cohort_tag, file_path in cohorts:
        print(f"\n==================== [{cohort_tag.upper()}] (16) START ====================")

        cohort_out = os.path.join(SECTION_DIR, cohort_tag)
        ensure_dir(cohort_out)

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path}")
            continue

        df = ensure_scores_and_accuracy(df, cohort_tag)

        needed_cols = ["age", "exposure_score", "confidence_score", "accuracy_pct"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            print(f"❌ Missing required columns in {cohort_tag}: {missing}")
            continue

        mediation_df = df[needed_cols].dropna().copy()
        print(f"✅ N after dropna: {len(mediation_df)} [{cohort_tag}]")

        if len(mediation_df) < 50:
            print(f"⚠️ Too few complete cases for stable bootstrap mediation (N={len(mediation_df)}). Skipping.")
            continue

        # Save raw used data (for reproducibility)
        raw_path = os.path.join(cohort_out, f"16-0_mediation_input_{cohort_tag}.csv")
        mediation_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved input (complete-case): {raw_path}")

        # --- (16-1) Parallel mediation ---
        print(f"\n--- (16-1) Parallel Mediation (bootstrap={N_BOOT}) [{cohort_tag}] ---")

        res = pg.mediation_analysis(
            data=mediation_df,
            x="age",
            m=["exposure_score", "confidence_score"],
            y="accuracy_pct",
            n_boot=N_BOOT,
            seed=SEED,
        )

        # Display table in notebook/console
        print("\n[Pingouin output table]")
        print(res.round(6).to_string(index=False))

        # Save raw table
        csv_path = os.path.join(cohort_out, f"16-1_mediation_results_{cohort_tag}.csv")
        res.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved table: {csv_path}")

        # Interpret indirect effects
        ind_exp = pick_indirect_rows(res, "exposure_score")
        ind_conf = pick_indirect_rows(res, "confidence_score")

        exp_line = interpret_indirect(ind_exp)
        conf_line = interpret_indirect(ind_conf)

        # Build English report
        report = []
        report.append("============================================================\n")
        report.append(f"(16-1) Parallel Mediation Report [{cohort_tag.upper()}]\n")
        report.append("============================================================\n\n")
        report.append("Model:\n")
        report.append("  X (Predictor): Age\n")
        report.append("  Y (Outcome): Accuracy (%) [accuracy_pct]\n")
        report.append("  Mediators (parallel):\n")
        report.append("    M1: AI Exposure (exposure_score)\n")
        report.append("    M2: AI Confidence (confidence_score)\n\n")
        report.append("Method:\n")
        report.append(f"  - Bootstrap mediation (n_boot={N_BOOT})\n")
        report.append("  - Significance judged by whether 95% CI excludes 0\n\n")
        report.append(f"Complete-case N: {len(mediation_df)}\n\n")
        report.append("Key Indirect Effects:\n")
        report.append("------------------------------------------------------------\n")
        report.append(f"1) Indirect via AI Exposure: {exp_line}\n")
        report.append(f"2) Indirect via AI Confidence: {conf_line}\n\n")
        report.append("Full Result Table:\n")
        report.append("------------------------------------------------------------\n")
        report.append(res.round(6).to_string(index=False))
        report.append("\n")

        report_text = "".join(report)
        txt_path = os.path.join(cohort_out, f"16-1_mediation_report_{cohort_tag}.txt")
        save_text(txt_path, report_text)
        print(f"✅ Saved report: {txt_path}")

    print("\n==================== (16) DONE ====================")


def _run_cell_063():
    # ==============================================================================
    # (17) Mediation Path Diagram Visualization (Parallel Mediation) [ENGLISH]
    # ------------------------------------------------------------------------------
    # Reads the mediation results CSV saved in (16), then draws a path diagram.
    # Works for BOTH cohorts: MOBILE + WEB.
    # Robust to minor variations in 'path' naming (uses contains-based matching).
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------------
    # Helpers
    # -----------------------------
    def load_mediation_table(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "path" not in df.columns:
            raise ValueError(f"'path' column not found in: {csv_path}")
        return df

    def find_row(df: pd.DataFrame, must_contain: list[str]) -> pd.Series | None:
        """Return the first row whose 'path' contains all keywords (case-insensitive)."""
        path = df["path"].astype(str).str.lower()
        mask = np.ones(len(df), dtype=bool)
        for kw in must_contain:
            mask &= path.str.contains(kw.lower())
        hits = df.loc[mask]
        if hits.empty:
            return None
        return hits.iloc[0]

    def coef_stars(row: pd.Series | None) -> str:
        """Format coef + significance stars if pval exists; if missing row -> 'NA'."""
        if row is None:
            return "NA"
        coef = row["coef"] if "coef" in row.index else np.nan
        pval = row["pval"] if "pval" in row.index else np.nan

        if not np.isfinite(coef):
            return "NA"

        stars = ""
        if np.isfinite(pval):
            stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        return f"{float(coef):.2f}{stars}"

    def indirect_sig_by_ci(row: pd.Series | None) -> str:
        """For indirect effects, prefer CI-based significance if CI columns exist."""
        if row is None:
            return "NA"
        coef = row["coef"] if "coef" in row.index else np.nan
        if not np.isfinite(coef):
            return "NA"

        # Pingouin typically has CI columns for indirect effects
        if "CI[2.5%]" in row.index and "CI[97.5%]" in row.index:
            lo = float(row["CI[2.5%]"])
            hi = float(row["CI[97.5%]"])
            sig = not (lo <= 0 <= hi)
            mark = " (sig)" if sig else " (n.s.)"
            return f"{float(coef):.2f}{mark}"

        # fallback to pval stars
        return coef_stars(row)

    def draw_diagram(
        title: str,
        out_base: str,
        a1: str, a2: str, b1: str, b2: str,
        cprime: str, ctotal: str,
        ind1: str, ind2: str
    ):
        fig, ax = plt.subplots(figsize=(14, 10))

        # Node positions
        pos = {
            "Age (X)": (0.0, 0.0),
            "AI Exposure (M1)": (0.55, 0.55),
            "AI Confidence (M2)": (0.55, -0.55),
            "Accuracy (Y)": (1.15, 0.0),
        }

        node_style = dict(boxstyle="round,pad=0.75", fc="skyblue", ec="black", lw=1.6)
        arrow_style = dict(arrowstyle="->,head_width=0.35,head_length=0.75", color="black", lw=2.2)

        # Draw nodes
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

        # Paths
        draw_path("Age (X)", "AI Exposure (M1)", a1, rad=0.20, yoff=0.10)
        draw_path("Age (X)", "AI Confidence (M2)", a2, rad=-0.20, yoff=0.10)
        draw_path("AI Exposure (M1)", "Accuracy (Y)", b1, rad=0.20, yoff=0.10)
        draw_path("AI Confidence (M2)", "Accuracy (Y)", b2, rad=-0.20, yoff=0.10)

        # Direct effect: straight arrow, put label slightly below
        draw_path("Age (X)", "Accuracy (Y)", cprime, rad=0.00, yoff=-0.12)

        ax.set_title(title, fontsize=20, fontweight="bold", pad=18)

        summary = (
            f"Total effect (c): {ctotal}\n"
            f"Direct effect (c'): {cprime}\n\n"
            f"Indirect 1 (Age → Exposure → Accuracy): {ind1}\n"
            f"Indirect 2 (Age → Confidence → Accuracy): {ind2}\n\n"
            f"Stars (for a/b/c paths): * p<.05, ** p<.01, *** p<.001\n"
            f"Indirect effects: prefer CI-based (sig/n.s.) when available"
        )

        ax.text(
            0.58, -1.05, summary,
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E5", ec="gray", lw=1),
        )

        ax.set_xlim(-0.3, 1.45)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        # IMPORTANT: add pad_inches so nothing gets clipped
        plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.savefig(f"{out_base}.svg", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.show()

    # -----------------------------
    # Main
    # -----------------------------
    print("==============================================================================")
    print("(17) Mediation Path Diagram (Parallel Mediation) - MOBILE + WEB [ENGLISH]")
    print("==============================================================================\n")

    BASE_DIR = r"outputs\run_20260119_192624\16_mediation_parallel"
    cohorts = [
        ("mobile", os.path.join(BASE_DIR, "mobile", "16-1_mediation_results_mobile.csv")),
        ("web",    os.path.join(BASE_DIR, "web",    "16-1_mediation_results_web.csv")),
    ]

    for cohort_tag, csv_path in cohorts:
        print(f"\n==================== [{cohort_tag.upper()}] (17) START ====================")

        if not os.path.exists(csv_path):
            print(f"❌ Missing mediation results CSV: {csv_path}")
            print("   -> Run section (16) first (the version that saves outputs_16_mediation/...).")
            continue

        med = load_mediation_table(csv_path)

        # Robust matching for paths:
        # a paths
        a1_row = find_row(med, ["exposure_score", "~", "x"])          # exposure_score ~ X
        a2_row = find_row(med, ["confidence_score", "~", "x"])        # confidence_score ~ X

        # b paths (often shown as "Y ~ exposure_score" etc.)
        b1_row = find_row(med, ["y", "~", "exposure_score"])
        b2_row = find_row(med, ["y", "~", "confidence_score"])

        # direct/total
        cprime_row = find_row(med, ["direct"])
        ctotal_row = find_row(med, ["total"])

        # indirect effects (CI-based best)
        ind1_row = find_row(med, ["indirect", "exposure_score"])
        ind2_row = find_row(med, ["indirect", "confidence_score"])

        a1 = coef_stars(a1_row)
        a2 = coef_stars(a2_row)
        b1 = coef_stars(b1_row)
        b2 = coef_stars(b2_row)
        cprime = coef_stars(cprime_row)
        ctotal = coef_stars(ctotal_row)
        ind1 = indirect_sig_by_ci(ind1_row)
        ind2 = indirect_sig_by_ci(ind2_row)

        print("Extracted labels:")
        print(f"  a1 (Age→Exposure): {a1}")
        print(f"  a2 (Age→Confidence): {a2}")
        print(f"  b1 (Exposure→Accuracy): {b1}")
        print(f"  b2 (Confidence→Accuracy): {b2}")
        print(f"  c' (Direct): {cprime}")
        print(f"  c (Total): {ctotal}")
        print(f"  Indirect via Exposure: {ind1}")
        print(f"  Indirect via Confidence: {ind2}")

        out_base = os.path.join(BASE_DIR, cohort_tag, f"17-1_mediation_path_diagram_{cohort_tag}")
        title = f"(17-1) Parallel Mediation Path Diagram [{cohort_tag.upper()}]"

        draw_diagram(
            title=title,
            out_base=out_base,
            a1=a1, a2=a2, b1=b1, b2=b2,
            cprime=cprime, ctotal=ctotal,
            ind1=ind1, ind2=ind2
        )

    print("\n==================== (17) DONE ====================")


def _run_cell_067():
    # ==============================================================================
    # (17 v2) Mediation Path Diagram (Parallel + Mean RT) - MOBILE + WEB [ENGLISH]
    # ------------------------------------------------------------------------------
    # Fixes:
    # - Robust accuracy column selection: overallAccuracy_y -> overallAccuracy -> accuracy
    # - If accuracy looks like proportion (<=1.0), convert to %
    # - Robust mean RT selection (avgRT etc.)
    # - Safer label extraction: "contains" matching instead of exact path equality
    # - Saves: mediation table + diagram (png/svg) per cohort
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    BOOT = 5000
    SEED = 42
    OUTROOT = "outputs_17_diagram_v2"

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def ci_is_sig(ci_lo: float, ci_hi: float) -> bool:
        return not (ci_lo <= 0 <= ci_hi)

    def star(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    def pick_accuracy_column(df: pd.DataFrame):
        """
        Choose accuracy column robustly.
        Prefer: overallAccuracy_y -> overallAccuracy -> accuracy -> Accuracy
        """
        candidates = ["overallAccuracy_y", "overallAccuracy", "accuracy", "Accuracy"]
        for c in candidates:
            if c in df.columns:
                return c
        # fallback: any column containing 'accuracy'
        for c in df.columns:
            if "accuracy" in str(c).lower():
                return c
        return None

    def convert_accuracy_to_percent_if_needed(s: pd.Series) -> pd.Series:
        """
        If accuracy seems like proportion (0~1), convert to percent.
        Heuristic: if 95th percentile <= 1.0 and max <= 1.2
        """
        x = pd.to_numeric(s, errors="coerce")
        x_valid = x.dropna()
        if len(x_valid) == 0:
            return x
        q95 = float(np.nanpercentile(x_valid, 95))
        mx = float(np.nanmax(x_valid))
        if q95 <= 1.0 and mx <= 1.2:
            return x * 100.0
        return x

    def ensure_scores_and_meanrt_and_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}

        df = df.copy()

        # exposure / confidence scores
        if "exposure_score" not in df.columns:
            if "aiExposureFrequency" in df.columns:
                df["exposure_score"] = (
                    df["aiExposureFrequency"].astype(str).str.lower().str.strip().map(exposure_map)
                )
            else:
                df["exposure_score"] = np.nan

        if "confidence_score" not in df.columns:
            if "aiConfidence" in df.columns:
                df["confidence_score"] = (
                    df["aiConfidence"].astype(str).str.lower().str.strip().map(confidence_map)
                )
            else:
                df["confidence_score"] = np.nan

        # mean_rt (prefer avgRT)
        rt_candidates = ["avgRT", "meanRT", "MeanRT", "mean_rt", "avg_rt", "rt_mean", "RT", "rt"]
        found_rt = None
        for c in rt_candidates:
            if c in df.columns:
                found_rt = c
                break
        if found_rt is not None:
            df["mean_rt"] = pd.to_numeric(df[found_rt], errors="coerce")
        else:
            df["mean_rt"] = np.nan

        # accuracy: copy to overallAccuracy (unified name)
        acc_col = pick_accuracy_column(df)
        if acc_col is None:
            df["overallAccuracy"] = np.nan
        else:
            df["overallAccuracy"] = convert_accuracy_to_percent_if_needed(df[acc_col])

        # numeric safety
        for c in ["age", "overallAccuracy", "exposure_score", "confidence_score", "mean_rt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def get_row_contains(tbl: pd.DataFrame, must_contain: list[str]):
        """Find first row whose 'path' contains all tokens (case-insensitive)."""
        if "path" not in tbl.columns:
            return None
        path = tbl["path"].astype(str).str.lower()
        mask = np.ones(len(tbl), dtype=bool)
        for kw in must_contain:
            mask &= path.str.contains(kw.lower())
        hits = tbl.loc[mask]
        if hits.empty:
            return None
        return hits.iloc[0]

    def coef_stars_from_row(row) -> str:
        if row is None:
            return "NA"
        coef = row.get("coef", np.nan)
        pval = row.get("pval", np.nan)
        if not np.isfinite(coef):
            return "NA"
        st = ""
        if np.isfinite(pval):
            st = star(float(pval))
        return f"{float(coef):.2f}{st}"

    def indirect_sig_by_ci(row) -> tuple[str, bool]:
        if row is None:
            return ("NA", False)
        coef = row.get("coef", np.nan)
        lo = row.get("CI[2.5%]", np.nan)
        hi = row.get("CI[97.5%]", np.nan)
        if not (np.isfinite(coef) and np.isfinite(lo) and np.isfinite(hi)):
            return ("NA", False)
        sig = ci_is_sig(float(lo), float(hi))
        return (f"{float(coef):.2f}", sig)

    def draw_path(ax, p0, p1, text, rad=0.15, text_offset=(0, 0.06)):
        ax.annotate(
            "",
            xy=p1, xytext=p0,
            arrowprops=dict(
                arrowstyle="->",
                lw=2.2,
                color="black",
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        xm = (p0[0] + p1[0]) / 2.0 + text_offset[0]
        ym = (p0[1] + p1[1]) / 2.0 + text_offset[1]
        ax.text(
            xm, ym, text,
            ha="center", va="center",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
        )

    print("==============================================================================")
    print("(17 v2) Mediation Path Diagram (Parallel + Mean RT) - MOBILE + WEB [ENGLISH]")
    print("==============================================================================\n")

    # pingouin import with friendly error
    try:
        import pingouin as pg
    except ImportError as e:
        raise ImportError(
            "pingouin is not installed. Install it first:\n"
            "  pip install pingouin\n"
            "or (conda):\n"
            "  conda install -c conda-forge pingouin"
        ) from e

    ensure_dir(OUTROOT)

    for cohort, path in COHORT_FILES.items():
        print(f"\n==================== [{cohort.upper()}] (17 v2) START ====================")
        outdir = os.path.join(OUTROOT, cohort)
        ensure_dir(outdir)

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"✅ Loaded: {path} (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {path}")
            continue

        df = ensure_scores_and_meanrt_and_accuracy(df)

        needed = ["age", "exposure_score", "confidence_score", "mean_rt", "overallAccuracy"]
        med_df = df[needed].dropna()
        print(f"✅ Complete-case N: {len(med_df)}")

        if len(med_df) < 50:
            print(f"⚠️ Too few complete cases (N={len(med_df)}). Skipping.")
            continue

        # Mediation analysis (parallel with 3 mediators)
        res = pg.mediation_analysis(
            data=med_df,
            x="age",
            m=["exposure_score", "confidence_score", "mean_rt"],
            y="overallAccuracy",
            n_boot=BOOT,
            seed=SEED,
        )

        # Save raw table
        table_path = os.path.join(outdir, f"17v2-0_mediation_table_{cohort}.csv")
        res.to_csv(table_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved mediation table: {table_path}")

        # Robust path extraction (contains-based)
        a1_row = get_row_contains(res, ["exposure_score", "~", "x"])
        a2_row = get_row_contains(res, ["confidence_score", "~", "x"])
        a3_row = get_row_contains(res, ["mean_rt", "~", "x"])

        b1_row = get_row_contains(res, ["y", "~", "exposure_score"])
        b2_row = get_row_contains(res, ["y", "~", "confidence_score"])
        b3_row = get_row_contains(res, ["y", "~", "mean_rt"])

        cprime_row = get_row_contains(res, ["direct"])
        ctotal_row = get_row_contains(res, ["total"])

        ind1_row = get_row_contains(res, ["indirect", "exposure_score"])
        ind2_row = get_row_contains(res, ["indirect", "confidence_score"])
        ind3_row = get_row_contains(res, ["indirect", "mean_rt"])

        a1 = coef_stars_from_row(a1_row)
        a2 = coef_stars_from_row(a2_row)
        a3 = coef_stars_from_row(a3_row)

        b1 = coef_stars_from_row(b1_row)
        b2 = coef_stars_from_row(b2_row)
        b3 = coef_stars_from_row(b3_row)

        c_prime = coef_stars_from_row(cprime_row)
        c_total = coef_stars_from_row(ctotal_row)

        ind1, sig1 = indirect_sig_by_ci(ind1_row)
        ind2, sig2 = indirect_sig_by_ci(ind2_row)
        ind3, sig3 = indirect_sig_by_ci(ind3_row)

        print("Extracted labels:")
        print(f"  a1 (Age→Exposure): {a1}")
        print(f"  a2 (Age→Confidence): {a2}")
        print(f"  a3 (Age→Mean RT): {a3}")
        print(f"  b1 (Exposure→Accuracy): {b1}")
        print(f"  b2 (Confidence→Accuracy): {b2}")
        print(f"  b3 (Mean RT→Accuracy): {b3}")
        print(f"  c' (Direct): {c_prime}")
        print(f"  c (Total): {c_total}")
        print(f"  Indirect via Exposure: {ind1} ({'sig' if sig1 else 'n.s.'})")
        print(f"  Indirect via Confidence: {ind2} ({'sig' if sig2 else 'n.s.'})")
        print(f"  Indirect via Mean RT: {ind3} ({'sig' if sig3 else 'n.s.'})")

        # ---- draw diagram ----
        fig, ax = plt.subplots(figsize=(16, 10))

        pos = {
            "Age (X)": (0.0, 0.0),
            "AI Exposure (M1)": (0.55, 0.55),
            "AI Confidence (M2)": (0.55, 0.0),
            "Mean RT (M3)": (0.55, -0.55),
            "Accuracy (Y)": (1.15, 0.0),
        }

        node_style = dict(boxstyle="round,pad=0.8", fc="skyblue", ec="black", lw=1.6)

        for name, p in pos.items():
            ax.text(p[0], p[1], name, ha="center", va="center",
                    fontsize=16, fontweight="bold", bbox=node_style)

        draw_path(ax, pos["Age (X)"], pos["AI Exposure (M1)"], a1, rad=0.22, text_offset=(0, 0.07))
        draw_path(ax, pos["Age (X)"], pos["AI Confidence (M2)"], a2, rad=0.0, text_offset=(0, 0.08))
        draw_path(ax, pos["Age (X)"], pos["Mean RT (M3)"], a3, rad=-0.22, text_offset=(0, 0.07))

        draw_path(ax, pos["AI Exposure (M1)"], pos["Accuracy (Y)"], b1, rad=0.18, text_offset=(0, 0.07))
        draw_path(ax, pos["AI Confidence (M2)"], pos["Accuracy (Y)"], b2, rad=0.0, text_offset=(0, 0.08))
        draw_path(ax, pos["Mean RT (M3)"], pos["Accuracy (Y)"], b3, rad=-0.18, text_offset=(0, 0.07))

        draw_path(ax, pos["Age (X)"], pos["Accuracy (Y)"], c_prime, rad=0.0, text_offset=(0, -0.10))

        summary = (
            f"Total (c): {c_total}\n"
            f"Direct (c'): {c_prime}\n\n"
            f"Indirect via Exposure (a1*b1): {ind1} {'(sig)' if sig1 else '(n.s.)'}\n"
            f"Indirect via Confidence (a2*b2): {ind2} {'(sig)' if sig2 else '(n.s.)'}\n"
            f"Indirect via Mean RT (a3*b3): {ind3} {'(sig)' if sig3 else '(n.s.)'}\n\n"
            f"* p < .05, ** p < .01, *** p < .001"
        )
        ax.text(
            0.58, -1.05, summary,
            ha="center", va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E5", ec="gray", lw=1.0)
        )

        ax.set_title(f"(17 v2) Parallel Mediation Path Diagram (+ Mean RT) [{cohort.upper()}]",
                     fontsize=18, fontweight="bold", pad=20)
        ax.set_xlim(-0.25, 1.45)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        out_base = os.path.join(outdir, f"17v2-1_mediation_path_diagram_{cohort}")
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.35)
        fig.savefig(out_base + ".svg", dpi=300, bbox_inches="tight", pad_inches=0.35)
        plt.show()
        plt.close(fig)

        print(f"✅ Saved: {out_base}.png/.svg")

    print("\n==================== (17 v2) DONE ====================")


def main():
    _run_cell_043()
    _run_cell_046()
    _run_cell_051()
    _run_cell_055()
    _run_cell_057()
    _run_cell_060()
    _run_cell_063()
    _run_cell_067()


if __name__ == "__main__":
    main()
