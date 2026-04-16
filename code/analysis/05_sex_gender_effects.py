"""Analysis sections for sex/gender effects, ANOVA, t-tests, and age-by-sex interaction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_012():
    # ==============================================================================
    # (5) 최종 데이터 정확도 분포 시각화 (MOBILE + WEB) - Version: 섹션폴더 + 재현용 CSV + bins=50
    # ------------------------------------------------------------------------------
    # 입력: (4)에서 생성된 최종 데이터
    #   - analysis_data_mobile_age_filtered_20_69.csv
    #   - analysis_data_web_age_filtered_20_69.csv
    #
    # 저장:
    #   - 섹션 폴더: outputs/run_.../05_final_accuracy_distribution/
    #       * 5-1, 5-2 각각: __raw.csv / __describe.csv / __bins.csv / .png / .svg / __meta.json
    #       * cohort_summary.csv (mean/median/std/N 등)
    # ------------------------------------------------------------------------------
    # 정확도 컬럼:
    #   - overallAccuracy_y(0~1) 우선, 없으면 overallAccuracy / overallAccuracy_x
    #   - 그래프는 %로 표시
    # ==============================================================================

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from datetime import datetime
    import json

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
        if mx <= 1.5:
            return s_num * 100.0
        return s_num

    # -----------------------------
    # 분포 저장 함수 (섹션2 스타일)
    # -----------------------------
    def save_accuracy_distribution(
        df: pd.DataFrame,
        cohort_tag: str,
        section_dir: Path,
        section_number: str,
        title_suffix: str,
        bins: int = 50,
        use_kde: bool = True
    ):
        acc_col = resolve_overall_accuracy_column(df)

        d = df.copy()
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
        d = d.dropna(subset=[acc_col]).copy()
        d["accuracy_pct"] = to_percent_series(d[acc_col])

        safe_suffix = title_suffix.lower().replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "")
        base = f"{section_number}_accuracy_histogram_final_{safe_suffix}"

        # raw 저장 (Prism/Excel 재현용)
        raw_out = section_dir / f"{base}__raw.csv"
        keep_cols = [c for c in ["participantId", "deviceType", "age", acc_col, "accuracy_pct"] if c in d.columns]
        d[keep_cols].to_csv(raw_out, index=False, encoding="utf-8-sig")

        # describe 저장
        desc = d["accuracy_pct"].describe()
        describe_out = section_dir / f"{base}__describe.csv"
        desc.round(6).to_frame("value").to_csv(describe_out, encoding="utf-8-sig")

        # bins 저장 (edges 고정으로 완전 재현)
        finite_vals = d["accuracy_pct"].dropna().astype(float).values
        lo = min(0.0, float(np.min(finite_vals))) if finite_vals.size else 0.0
        hi = max(100.0, float(np.max(finite_vals))) if finite_vals.size else 100.0
        bin_edges = np.linspace(lo, hi, bins + 1)
        counts, edges = np.histogram(finite_vals, bins=bin_edges)

        bins_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
        bins_out = section_dir / f"{base}__bins.csv"
        bins_df.to_csv(bins_out, index=False, encoding="utf-8-sig")

        # plot
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(14, 7))
        ax = sns.histplot(
            data=d,
            x="accuracy_pct",
            bins=bin_edges,
            kde=use_kde,
            alpha=0.6,
            edgecolor="black"
        )

        mean_val = float(np.nanmean(finite_vals)) if finite_vals.size else np.nan
        median_val = float(np.nanmedian(finite_vals)) if finite_vals.size else np.nan
        std_val = float(np.nanstd(finite_vals, ddof=1)) if finite_vals.size > 1 else np.nan

        plt.axvline(mean_val, linestyle="--", linewidth=2.5, label=f"Mean: {mean_val:.2f}")
        plt.axvline(median_val, linestyle=":", linewidth=2.5, label=f"Median: {median_val:.2f}")

        plt.suptitle(f"({section_number}) Accuracy Distribution ({title_suffix})", fontsize=20, fontweight="bold")
        plt.xlabel("Accuracy (%)", fontsize=12)
        plt.ylabel("Number of Participants", fontsize=12)

        text_str = f"N: {int(desc['count'])}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
        ax.text(
            0.05, 0.95, text_str,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        sns.despine(left=True)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

        png_out = section_dir / f"{base}.png"
        svg_out = section_dir / f"{base}.svg"
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.savefig(svg_out, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # meta 저장
        meta = {
            "section": "05_final_accuracy_distribution",
            "section_number": section_number,
            "cohort": cohort_tag,
            "accuracy_column_used": acc_col,
            "plot_scale": "percent",
            "bins": int(bins),
            "bin_edges": edges.tolist(),
            "use_kde": bool(use_kde),
            "outputs": {
                "raw_csv": str(raw_out),
                "describe_csv": str(describe_out),
                "bins_csv": str(bins_out),
                "png": str(png_out),
                "svg": str(svg_out),
            },
            "summary": {
                "N": int(desc["count"]),
                "mean_pct": float(mean_val),
                "median_pct": float(median_val),
                "std_pct": float(std_val),
                "min_pct": float(desc["min"]),
                "max_pct": float(desc["max"]),
            }
        }
        meta_out = section_dir / f"{base}__meta.json"
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        return meta["summary"]

    # -----------------------------
    # MAIN
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(5) 최종 데이터 정확도 분포 시각화 (MOBILE + WEB)")
        print("------------------------------------------------------------------------------")
        print("목적: (4)에서 생성된 최종 데이터(20-69세)로 mobile/web 정확도 분포를 시각화합니다.")
        print("==============================================================================\n")

        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG  # 같은 run을 쓰고 싶으면 config.RUN_TAG 같은 값 넣기
        BINS = 50       # ✅ 2% bin

        run_dir, section_dir = _make_section_dir("05_final_accuracy_distribution", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
        print(f"📁 섹션5 저장 위치: {section_dir}")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }
        section_map = {"mobile": "5-1", "web": "5-2"}

        summaries = []
        for cohort_tag, file_path in cohort_files.items():
            p = Path(file_path)
            if not p.exists():
                print(f"❌ '{file_path}' 파일이 없습니다. ({cohort_tag}) 스킵")
                continue

            df = pd.read_csv(p, encoding="utf-8-sig")
            print(f"✅ '{p.name}' 로드 성공 ({cohort_tag}), rows: {len(df):,}")

            summary = save_accuracy_distribution(
                df,
                cohort_tag=cohort_tag,
                section_dir=section_dir,
                section_number=section_map.get(cohort_tag, "5-x"),
                title_suffix=f"Final Participants (20-69 y/o) [{cohort_tag}]",
                bins=BINS,
                use_kde=True
            )
            summary["cohort"] = cohort_tag
            summaries.append(summary)

        # cohort summary 저장
        if summaries:
            summary_df = pd.DataFrame(summaries)[["cohort","N","mean_pct","median_pct","std_pct","min_pct","max_pct"]]
            summary_out = section_dir / "cohort_summary.csv"
            summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ 섹션5 요약 저장: {summary_out}")


def _run_cell_014():
    # ==============================================================================
    # (6) Sex에 따른 정확도 차이 분석 (MOBILE + WEB)  [Palette Unified Version]
    # ------------------------------------------------------------------------------
    # 목적: (4)에서 생성된 최종 필터링 데이터(mobile/web, 20-69세) 각각에 대해
    #       1) Sex 분포(countplot/pie) 저장 + counts.csv
    #       2) Sex별 정확도 분포(violin/box + strip) 저장 + Prism용 raw.csv
    #       3) One-way ANOVA + (가능하면) Tukey HSD 수행, 결과 csv/txt 저장
    # ------------------------------------------------------------------------------
    # 저장 규칙:
    #   - 섹션 폴더: outputs/run_.../06_sex_accuracy/
    #       * 6-1,6-2,6-3,6-4: png/svg + raw/counts/stats/meta
    #       * cohort_summary.csv
    # ------------------------------------------------------------------------------
    # 정확도 처리:
    #   - overallAccuracy_y(0~1) 우선 사용, 없으면 overallAccuracy / overallAccuracy_x
    #   - 분석/그림은 accuracy_pct(0~100)로 통일
    # ------------------------------------------------------------------------------
    # ✨ 변경점:
    #   - Pie chart에 사용된 색 팔레트를 countplot/violin/box/strip에 동일 적용
    # ==============================================================================

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from pathlib import Path
    from datetime import datetime
    import json

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
    # Sex 정규화 (컬럼명이 gender든 sex든 모두 처리)
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

        # 흔한 변형 매핑
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
    # 저장 유틸
    # -----------------------------
    def _write_text(path: Path, title: str, content: str):
        path.write_text(
            "============================================================\n"
            f"{title}\n"
            "============================================================\n\n"
            + content,
            encoding="utf-8"
        )

    # -----------------------------
    # 핵심 분석 함수
    # -----------------------------
    def run_sex_accuracy_analysis(df: pd.DataFrame, cohort_tag: str, section_dir: Path):
        """
        cohort_tag: 'mobile' or 'web'
        df: age-filtered 최종 데이터
        """
        # 표준 순서/팔레트 (✨ 모든 그래프에서 동일 사용)
        order = ["male", "female", "prefer-not-to-say"]
        palette = {
            "male": "#4285F4",
            "female": "#DB4437",
            "prefer-not-to-say": "#F4B400",
        }

        # 1) accuracy 준비
        acc_col = resolve_overall_accuracy_column(df)
        d = df.copy()
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
        d = d.dropna(subset=[acc_col]).copy()
        d["accuracy_pct"] = to_percent_series(d[acc_col])

        # 2) sex 준비
        d = normalize_sex(d, in_col_candidates=("sex", "gender"), out_col="sex")

        # sex 결측은 분석 제외 (필요하면 unknown으로 포함 가능)
        d_valid = d.dropna(subset=["sex"]).copy()

        print(f"\n==================== [{cohort_tag.upper()}] (6) 분석 시작 ====================")
        print(f"- rows (accuracy valid): {len(d):,}")
        print(f"- rows (sex valid):      {len(d_valid):,}")
        print(f"✅ accuracy column used: {acc_col} (analysis/plots in %)")

        # Prism/Excel 재현용 raw 저장 (long)
        raw_cols = [c for c in ["participantId", "deviceType", "age", "sex", acc_col, "accuracy_pct"] if c in d_valid.columns]
        raw_out = section_dir / f"6-0_sex_accuracy_raw_{cohort_tag}.csv"
        d_valid[raw_cols].to_csv(raw_out, index=False, encoding="utf-8-sig")

        # =========================================================
        # (6-1) Sex 분포 Count Plot + counts.csv
        # =========================================================
        counts = d_valid["sex"].value_counts().reindex(order).fillna(0).astype(int)
        counts_df = counts.rename("count").reset_index().rename(columns={"index": "sex"})
        counts_out = section_dir / f"6-1_sex_distribution_counts_{cohort_tag}.csv"
        counts_df.to_csv(counts_out, index=False, encoding="utf-8-sig")

        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(8, 5))
        ax = sns.countplot(x="sex", data=d_valid, order=order, palette=palette)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="center", fontsize=11, color="gray",
                xytext=(0, 5), textcoords="offset points",
            )
        plt.suptitle(f"(6-1) Sex Distribution (Count) [{cohort_tag}]", fontsize=16, fontweight="bold")
        plt.xlabel("Sex"); plt.ylabel("Number of Participants")
        base_6_1 = f"6-1_sex_distribution_countplot_{cohort_tag}"
        plt.savefig(section_dir / f"{base_6_1}.png", dpi=300, bbox_inches="tight")
        plt.savefig(section_dir / f"{base_6_1}.svg", bbox_inches="tight")
        plt.show(); plt.close(fig)

        # =========================================================
        # (6-2) Sex 분포 Pie Chart (✨ palette 동일 적용)
        # =========================================================
        if counts.sum() > 0:
            fig = plt.figure(figsize=(8, 8))
            pie_colors = [palette.get(cat, "#999999") for cat in counts.index]
            plt.pie(
                counts.values,
                labels=None,
                autopct="%1.1f%%",
                startangle=90,
                colors=pie_colors,  # ✅ 동일 palette
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            plt.legend(
                [f"{lab} (n={cnt})" for lab, cnt in zip(counts.index, counts.values)],
                title="Sex",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
            )
            plt.suptitle(f"(6-2) Sex Distribution (Pie) [{cohort_tag}]", fontsize=16, fontweight="bold")
            base_6_2 = f"6-2_sex_distribution_piechart_{cohort_tag}"
            plt.savefig(section_dir / f"{base_6_2}.png", dpi=300, bbox_inches="tight")
            plt.savefig(section_dir / f"{base_6_2}.svg", bbox_inches="tight")
            plt.show(); plt.close(fig)

        # =========================================================
        # (6-3) Accuracy by Sex: violin + box + strip (✨ palette 동일 적용)
        # =========================================================
        sns.set_theme(style="ticks")
        fig = plt.figure(figsize=(10, 7))

        plot_df = d_valid[d_valid["sex"].isin(order)].copy()

        sns.violinplot(
            x="sex", y="accuracy_pct",
            data=plot_df, order=order,
            palette=palette, inner=None, cut=0
        )
        sns.boxplot(
            x="sex", y="accuracy_pct",
            data=plot_df, order=order,
            palette=palette, width=0.25,
            showcaps=True, boxprops={"alpha": 0.6},
            showfliers=False
        )
        sns.stripplot(
            x="sex", y="accuracy_pct",
            data=plot_df, order=order,
            palette=palette, alpha=0.20,
            size=3, jitter=0.25
        )

        plt.suptitle(f"(6-3) Accuracy by Sex [{cohort_tag}]", fontsize=16, fontweight="bold")
        plt.xlabel("Sex"); plt.ylabel("Accuracy (%)")
        sns.despine()

        base_6_3 = f"6-3_accuracy_by_sex_violinbox_{cohort_tag}"
        plt.savefig(section_dir / f"{base_6_3}.png", dpi=300, bbox_inches="tight")
        plt.savefig(section_dir / f"{base_6_3}.svg", bbox_inches="tight")
        plt.show(); plt.close(fig)

        # Sex별 기술통계 저장
        desc = (plot_df.groupby("sex")["accuracy_pct"]
                .agg(n="count", mean="mean", std="std", median="median", min="min", max="max")
                .reindex(order))
        desc_out = section_dir / f"6-3_accuracy_by_sex_descriptives_{cohort_tag}.csv"
        desc.to_csv(desc_out, encoding="utf-8-sig")

        # =========================================================
        # (6-4) ANOVA + Tukey HSD
        # =========================================================
        groups = [plot_df.loc[plot_df["sex"] == g, "accuracy_pct"].dropna() for g in order]
        non_empty = [g for g in groups if len(g) > 0]

        report_title = f"(6-4) Accuracy by Sex: ANOVA & Tukey HSD [{cohort_tag}]"
        report_lines = []
        report_lines.append("분석 개요")
        report_lines.append("-----------------------------------------")
        report_lines.append(f"- Cohort: {cohort_tag}")
        report_lines.append("- DV: accuracy_pct (Accuracy %)")
        report_lines.append("- IV: sex")
        report_lines.append("- Test: One-way ANOVA + (if applicable) Tukey HSD")
        report_lines.append("")
        report_lines.append("데이터 요약")
        report_lines.append("-----------------------------------------")
        report_lines.append(f"- N (sex valid): {len(plot_df)}")
        report_lines.append("")
        report_lines.append(desc.reset_index().to_string(index=False))
        report_lines.append("")

        anova_out = section_dir / f"6-4_anova_results_{cohort_tag}.csv"
        tukey_out = section_dir / f"6-4_tukey_results_{cohort_tag}.csv"
        txt_out = section_dir / f"6-4_anova_tukey_report_{cohort_tag}.txt"

        if len(non_empty) < 2:
            report_lines.append("⚠️ 유효한 sex 그룹이 2개 미만이라 ANOVA를 수행할 수 없습니다.")
            _write_text(txt_out, report_title, "\n".join(report_lines))
            pd.DataFrame([{"cohort": cohort_tag, "status": "not_enough_groups"}]).to_csv(anova_out, index=False, encoding="utf-8-sig")
            print("\n".join(report_lines))
            print(f"✅ 리포트 저장: {txt_out}")
            print(f"==================== [{cohort_tag.upper()}] (6) 분석 종료 ====================\n")
            return {
                "cohort": cohort_tag,
                "N_sex_valid": int(len(plot_df)),
                "anova_F": np.nan,
                "anova_p": np.nan,
            }

        # ANOVA
        F, p = stats.f_oneway(*non_empty)

        # eta^2 (효과크기)
        y = plot_df["accuracy_pct"].values
        grand_mean = np.nanmean(y)
        ss_total = np.nansum((y - grand_mean) ** 2)

        ss_between = 0.0
        for g in order:
            vals = plot_df.loc[plot_df["sex"] == g, "accuracy_pct"].dropna().values
            if len(vals) == 0:
                continue
            ss_between += len(vals) * (np.mean(vals) - grand_mean) ** 2

        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

        anova_df = pd.DataFrame([{
            "cohort": cohort_tag,
            "N": int(len(plot_df)),
            "F": float(F),
            "p": float(p),
            "eta_squared": float(eta2) if np.isfinite(eta2) else np.nan,
            "groups_included": ", ".join([order[i] for i, g in enumerate(groups) if len(g) > 0])
        }])
        anova_df.to_csv(anova_out, index=False, encoding="utf-8-sig")

        report_lines.append("ANOVA 결과")
        report_lines.append("-----------------------------------------")
        report_lines.append(f"- F = {F:.4f}")
        report_lines.append(f"- p = {p:.6g}")
        report_lines.append(f"- eta^2 = {eta2:.4f}")
        report_lines.append("")

        # Tukey
        df_tukey = plot_df[["sex", "accuracy_pct"]].dropna().copy()

        if df_tukey["sex"].nunique() >= 2:
            try:
                tuk = pairwise_tukeyhsd(endog=df_tukey["accuracy_pct"], groups=df_tukey["sex"], alpha=0.05)
                tuk_table = pd.DataFrame(data=tuk.summary().data[1:], columns=tuk.summary().data[0])
                tuk_table.to_csv(tukey_out, index=False, encoding="utf-8-sig")

                report_lines.append("Tukey HSD 결과")
                report_lines.append("-----------------------------------------")
                report_lines.append(tuk.summary().as_text())

            except Exception as e:
                pd.DataFrame([{"cohort": cohort_tag, "status": "tukey_failed", "error": str(e)}]).to_csv(
                    tukey_out, index=False, encoding="utf-8-sig"
                )
                report_lines.append("⚠️ Tukey 수행 중 오류:")
                report_lines.append(str(e))
        else:
            pd.DataFrame([{"cohort": cohort_tag, "status": "tukey_not_applicable", "reason": "n_unique_sex < 2"}]).to_csv(
                tukey_out, index=False, encoding="utf-8-sig"
            )
            report_lines.append("⚠️ 유효한 sex 그룹이 2개 미만이라 Tukey를 수행하지 않았습니다.")

        _write_text(txt_out, report_title, "\n".join(report_lines))

        # 콘솔 출력 (원래처럼 유지)
        print("\n".join(report_lines))
        print(f"✅ 저장: {anova_out}")
        print(f"✅ 저장: {tukey_out}")
        print(f"✅ 저장: {txt_out}")
        print(f"✅ Prism raw: {raw_out}")
        print(f"✅ Sex counts: {counts_out}")
        print(f"✅ Descriptives: {desc_out}")
        print(f"==================== [{cohort_tag.upper()}] (6) 분석 종료 ====================\n")

        return {
            "cohort": cohort_tag,
            "N_sex_valid": int(len(plot_df)),
            "anova_F": float(F),
            "anova_p": float(p),
            "eta_squared": float(eta2) if np.isfinite(eta2) else np.nan,
            "mean_male": float(desc.loc["male", "mean"]) if "male" in desc.index and pd.notna(desc.loc["male", "mean"]) else np.nan,
            "mean_female": float(desc.loc["female", "mean"]) if "female" in desc.index and pd.notna(desc.loc["female", "mean"]) else np.nan,
            "mean_prefer_not": float(desc.loc["prefer-not-to-say", "mean"]) if "prefer-not-to-say" in desc.index and pd.notna(desc.loc["prefer-not-to-say", "mean"]) else np.nan,
        }

    # -----------------------------
    # MAIN
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(6) Sex에 따른 정확도 차이 분석 (MOBILE + WEB)")
        print("------------------------------------------------------------------------------")
        print("목적: mobile/web 각각 Sex 분포 및 Sex 그룹 간 정확도 차이를 분석합니다.")
        print("==============================================================================\n")

        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG  # 같은 run 폴더 쓰기
        run_dir, section_dir = _make_section_dir("06_sex_accuracy", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
        print(f"📁 섹션6 저장 위치: {section_dir}")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        summaries = []
        for cohort_tag, file_path in cohort_files.items():
            p = Path(file_path)
            if not p.exists():
                print(f"❌ '{file_path}' 파일이 없습니다. ({cohort_tag}) 스킵")
                continue

            df = pd.read_csv(p, encoding="utf-8-sig")
            print(f"✅ '{p.name}' 로드 성공 ({cohort_tag}), rows: {len(df):,}")
            summaries.append(run_sex_accuracy_analysis(df, cohort_tag=cohort_tag, section_dir=section_dir))

        # cohort 요약 저장
        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_out = section_dir / "cohort_summary.csv"
            summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ 섹션6 요약 저장: {summary_out}")


def _run_cell_017():
    # ==============================================================================
    # (7) Sex(male vs female) 정확도 차이 심층 분석 (T-test) (MOBILE + WEB)
    # ------------------------------------------------------------------------------
    # 목적: 최종 데이터(mobile/web 각각)에서 'prefer-not-to-say' 등을 제외한
    #       male vs female 두 그룹 간 정확도 차이를
    #       - Student t-test (equal var)
    #       - Welch t-test (unequal var)
    #       - Mann-Whitney U
    #       로 검정하고, 결과를 섹션 폴더에 CSV/TXT로 저장합니다.
    #
    # 저장:
    #   outputs/run_.../07_sex_binary_ttests/
    #     - 7-0_raw_male_female_{cohort}.csv          (Prism/Excel 재현용 long)
    #     - 7-1_ttest_results_{cohort}.csv           (요약 테이블)
    #     - 7-1_ttest_report_{cohort}.txt            (설명 리포트)
    #     - cohort_summary.csv                       (mobile/web 합본)
    #
    # NOTE: 데이터 컬럼명이 gender여도 sex로 정규화하여 사용합니다.
    # ==============================================================================
    import pandas as pd
    import numpy as np
    from scipy import stats
    from pathlib import Path
    from datetime import datetime

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
        return s_num * 100.0 if mx <= 1.5 else s_num

    # -----------------------------
    # Sex 정규화 (gender/sex 모두 지원)
    # -----------------------------
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
            "prefer not to": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    # -----------------------------
    # 효과크기
    # -----------------------------
    def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        sx2 = np.var(x, ddof=1)
        sy2 = np.var(y, ddof=1)
        sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
        if sp == 0:
            return np.nan
        return (np.mean(x) - np.mean(y)) / sp  # (male - female)

    def hedges_g(d: float, nx: int, ny: int) -> float:
        if not np.isfinite(d) or nx + ny < 3:
            return np.nan
        J = 1 - (3 / (4 * (nx + ny) - 9))
        return d * J

    # -----------------------------
    # 리포트 저장
    # -----------------------------
    def save_text(path: Path, title: str, content: str):
        path.write_text(
            "============================================================\n"
            f"{title}\n"
            "============================================================\n\n"
            + content,
            encoding="utf-8"
        )

    # -----------------------------
    # 코호트별 실행
    # -----------------------------
    def run_binary_sex_tests(df: pd.DataFrame, cohort_tag: str, section_dir: Path):
        print(f"\n==================== [{cohort_tag.upper()}] (7) 분석 시작 ====================")

        d = df.copy()

        # sex 정규화
        d = normalize_sex(d, in_col_candidates=("sex", "gender"), out_col="sex")

        # accuracy 준비
        acc_col = resolve_overall_accuracy_column(d)
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
        d = d.dropna(subset=[acc_col]).copy()
        d["accuracy_pct"] = to_percent_series(d[acc_col])

        # male/female만
        d_bin = d[d["sex"].isin(["male", "female"])].dropna(subset=["sex", "accuracy_pct"]).copy()

        male = d_bin.loc[d_bin["sex"] == "male", "accuracy_pct"].astype(float).values
        female = d_bin.loc[d_bin["sex"] == "female", "accuracy_pct"].astype(float).values

        print(f"✅ accuracy column used: {acc_col} (analysis/plots in %)")
        print(f"- male N = {len(male):,}")
        print(f"- female N = {len(female):,}")

        # raw 저장 (Prism/Excel)
        raw_cols = [c for c in ["participantId", "deviceType", "age", "sex", acc_col, "accuracy_pct"] if c in d_bin.columns]
        raw_out = section_dir / f"7-0_raw_male_female_{cohort_tag}.csv"
        d_bin[raw_cols].to_csv(raw_out, index=False, encoding="utf-8-sig")

        title = f"(7-1) Sex(male vs female) Accuracy Tests [{cohort_tag}]"
        report_out = section_dir / f"7-1_ttest_report_{cohort_tag}.txt"
        results_out = section_dir / f"7-1_ttest_results_{cohort_tag}.csv"

        # 표본 부족
        if len(male) < 2 or len(female) < 2:
            content = (
                "표본 수가 부족하여 검정을 수행하지 않았습니다.\n\n"
                f"- male N={len(male)}, female N={len(female)}\n"
            )
            print(content)
            save_text(report_out, title, content)
            pd.DataFrame([{
                "cohort": cohort_tag,
                "status": "insufficient_sample",
                "N_male": len(male),
                "N_female": len(female),
            }]).to_csv(results_out, index=False, encoding="utf-8-sig")
            print(f"✅ 저장: {report_out}")
            print(f"✅ 저장: {results_out}")
            print(f"==================== [{cohort_tag.upper()}] (7) 분석 종료 ====================\n")
            return {
                "cohort": cohort_tag,
                "N_male": int(len(male)),
                "N_female": int(len(female)),
                "welch_p": np.nan,
            }

        # t-tests
        t_std, p_std = stats.ttest_ind(male, female, equal_var=True, nan_policy="omit")
        t_welch, p_welch = stats.ttest_ind(male, female, equal_var=False, nan_policy="omit")

        # Mann-Whitney (two-sided)
        # scipy 버전에 따라 method/alternative 옵션이 달라질 수 있어 기본 형태로
        u_stat, p_mw = stats.mannwhitneyu(male, female)

        # 효과크기
        d_cohen = cohen_d(male, female)         # (male - female)
        g_hedges = hedges_g(d_cohen, len(male), len(female))

        # 요약 통계
        mean_m, mean_f = float(np.mean(male)), float(np.mean(female))
        std_m, std_f = float(np.std(male, ddof=1)), float(np.std(female, ddof=1))

        # 결과 테이블 저장(논문 표 바로 사용 가능)
        res = pd.DataFrame([{
            "cohort": cohort_tag,
            "accuracy_column_used": acc_col,
            "N_male": int(len(male)),
            "N_female": int(len(female)),
            "mean_male_pct": mean_m,
            "mean_female_pct": mean_f,
            "std_male_pct": std_m,
            "std_female_pct": std_f,
            "student_t": float(t_std),
            "student_p": float(p_std),
            "welch_t": float(t_welch),
            "welch_p": float(p_welch),
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(p_mw),
            "cohen_d_male_minus_female": float(d_cohen) if np.isfinite(d_cohen) else np.nan,
            "hedges_g_male_minus_female": float(g_hedges) if np.isfinite(g_hedges) else np.nan,
        }])
        res.to_csv(results_out, index=False, encoding="utf-8-sig")

        # 리포트 텍스트
        content = (
            "male vs female 두 그룹 간 정확도 차이를 다음 검정으로 평가했습니다.\n\n"
            "분석 개요\n"
            "-----------------------------------------\n"
            f"- Cohort: {cohort_tag}\n"
            f"- DV: accuracy_pct (Accuracy %)\n"
            f"- Accuracy column used: {acc_col}\n"
            f"- Groups: male (N={len(male)}) vs female (N={len(female)})\n\n"
            "요약 통계\n"
            "-----------------------------------------\n"
            f"- male:   mean={mean_m:.3f}, sd={std_m:.3f}\n"
            f"- female: mean={mean_f:.3f}, sd={std_f:.3f}\n"
            f"- mean difference (male - female) = {(mean_m-mean_f):.3f}\n\n"
            "검정 결과\n"
            "-----------------------------------------\n"
            f"1) Student t-test (equal var): t={t_std:.4f}, p={p_std:.6g}\n"
            f"2) Welch t-test   (unequal):   t={t_welch:.4f}, p={p_welch:.6g}\n"
            f"3) Mann-Whitney U (nonparam):  U={u_stat:.1f}, p={p_mw:.6g}\n\n"
            "효과크기\n"
            "-----------------------------------------\n"
            f"- Cohen's d (male - female) = {d_cohen:.4f}\n"
            f"- Hedges' g (male - female) = {g_hedges:.4f}\n\n"
            "해석 가이드\n"
            "-----------------------------------------\n"
            "- 분산이 다를 가능성이 있으면 Welch 결과를 우선 참고하는 것이 안전합니다.\n"
            "- 분포/정규성 가정이 걱정되면 Mann-Whitney 결과를 보조적으로 확인하세요.\n"
        )

        # 자동 결론
        if (p_welch < 0.05) or (p_mw < 0.05) or (p_std < 0.05):
            content += (
                "\n종합 결론\n"
                "-----------------------------------------\n"
                "적어도 한 가지 검정에서 p < .05가 관찰되어, male vs female 정확도 차이가 존재할 가능성이 있습니다.\n"
                "(일반적으로 Welch/MW를 우선 참고 권장)\n"
            )
        else:
            content += (
                "\n종합 결론\n"
                "-----------------------------------------\n"
                "세 가지 검정 모두에서 p >= .05로 나타나, 현재 데이터에서는 male vs female 정확도 차이가\n"
                "통계적으로 유의하다고 보기 어렵습니다.\n"
            )

        print("\n" + content)
        save_text(report_out, title, content)

        print(f"✅ 저장: {raw_out}")
        print(f"✅ 저장: {results_out}")
        print(f"✅ 저장: {report_out}")
        print(f"==================== [{cohort_tag.upper()}] (7) 분석 종료 ====================\n")

        return {
            "cohort": cohort_tag,
            "N_male": int(len(male)),
            "N_female": int(len(female)),
            "welch_p": float(p_welch),
            "mean_diff_male_minus_female": float(mean_m - mean_f),
            "cohen_d": float(d_cohen) if np.isfinite(d_cohen) else np.nan,
        }

    # -----------------------------
    # MAIN
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(7) Sex(male vs female) 정확도 차이 심층 분석 (T-test) (MOBILE + WEB)")
        print("------------------------------------------------------------------------------")
        print("목적: mobile/web 각각에서 male vs female 정확도 차이를 다양한 검정으로 평가합니다.")
        print("==============================================================================\n")

        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG  # ✅ 현재 run 고정 (원하면 None)
        run_dir, section_dir = _make_section_dir("07_sex_binary_ttests", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
        print(f"📁 섹션7 저장 위치: {section_dir}")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        summaries = []
        for cohort_tag, file_path in cohort_files.items():
            p = Path(file_path)
            if not p.exists():
                print(f"❌ '{file_path}' 파일이 없습니다. ({cohort_tag}) 스킵")
                continue

            df = pd.read_csv(p, encoding="utf-8-sig")
            print(f"✅ '{p.name}' 로드 성공 ({cohort_tag}), rows: {len(df):,}")
            summaries.append(run_binary_sex_tests(df, cohort_tag=cohort_tag, section_dir=section_dir))

        if summaries:
            summary_df = pd.DataFrame(summaries)
            out = section_dir / "cohort_summary.csv"
            summary_df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n✅ 섹션7 요약 저장: {out}")


def _run_cell_020():
    # ==============================================================================
    # (8) Age group x Sex interaction (Two-Way ANOVA) + Tukey(FWER) + FDR(BH) + HC3
    # ------------------------------------------------------------------------------
    # FIX:
    #   - 입력 파일에 'sex'가 없으면 'gender'를 자동으로 사용해서 sex로 매핑
    #   - 결과를 콘솔에 "반드시" 출력
    #   - statsmodels/pandas 버전차로 생기는 KeyError/df_denom 문제 회피(안전 파서)
    # ==============================================================================
    import os
    import re
    import json
    import warnings
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import levene
    from scipy import stats

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests

    # -----------------------------
    # Notebook display(선택)
    # -----------------------------
    try:
        from IPython.display import display
    except Exception:
        display = None

    # -----------------------------
    # 출력/표시 스위치
    # -----------------------------
    SHOW_PLOTS = True
    PRINT_REPORT = True
    DISPLAY_TABLES = True

    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


    # ==============================================================================
    # 공통 유틸: 폴더/저장
    # ==============================================================================
    def _safe(s: str) -> str:
        s = str(s)
        s = re.sub(r"[^\w\s\-\.\(\)\[\]]", "_", s)
        s = re.sub(r"\s+", "_", s).strip("_")
        return s


    def _make_section_dir(section_slug: str, outputs_root: str = "outputs", run_tag: str | None = None):
        if run_tag is None:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(outputs_root, f"run_{run_tag}")
        section_dir = os.path.join(run_dir, section_slug)
        os.makedirs(section_dir, exist_ok=True)
        return run_tag, run_dir, section_dir


    def _write_json(path: str, obj: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


    def save_detailed_analysis(section_dir: str, filename_base: str, title: str, content: str):
        path = os.path.join(section_dir, f"{filename_base}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f" {title}\n")
            f.write("============================================================\n\n")
            f.write(str(content))
        if PRINT_REPORT:
            print(f"✅ 저장: {path}")
        return path


    # ==============================================================================
    # 데이터/컬럼 정규화
    # ==============================================================================
    def normalize_sex(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower().str.strip()
        s = s.replace({
            "nan": np.nan, "none": np.nan, "": np.nan,
            "m": "male", "man": "male", "male ": "male",
            "f": "female", "woman": "female", "female ": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer not to": "prefer-not-to-say",
        })
        return s


    def ensure_sex_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """
        sex 컬럼이 없으면 gender를 사용해서 sex로 만들어줌.
        반환: (df_out, source_col_used)
        """
        d = df.copy()
        if "sex" in d.columns:
            d["sex"] = normalize_sex(d["sex"])
            return d, "sex"
        if "gender" in d.columns:
            d["sex"] = normalize_sex(d["gender"])
            return d, "gender"
        raise KeyError("missing required column: sex (and no fallback gender found)")


    def choose_accuracy_column(df: pd.DataFrame) -> tuple[str, str]:
        for c in ["overallAccuracy_y", "overallAccuracy_x", "overallAccuracy"]:
            if c in df.columns:
                acc_col = c
                x = pd.to_numeric(df[acc_col], errors="coerce").dropna()
                if len(x) == 0:
                    return acc_col, "unknown scale"
                if x.max() <= 1.5:
                    return acc_col, "fraction (0-1) -> will convert to %"
                return acc_col, "percent-like (0-100) -> will keep as %"
        raise KeyError("No accuracy column found among: overallAccuracy_y / overallAccuracy_x / overallAccuracy")


    def ensure_accuracy_pct(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        d = df.copy()
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
        x = d[acc_col].dropna()
        if len(x) == 0:
            d["accuracy_pct"] = np.nan
            return d, "unknown scale"
        if x.max() <= 1.5:
            d["accuracy_pct"] = d[acc_col] * 100.0
            return d, "fraction->pct"
        d["accuracy_pct"] = d[acc_col]
        return d, "pct"


    # ==============================================================================
    # FDR(BH) pairwise Welch
    # ==============================================================================
    def pairwise_fdr_welch(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> pd.DataFrame:
        tmp = df[[group_col, value_col]].dropna().copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna(subset=[value_col]).copy()

        groups = sorted(tmp[group_col].unique().tolist())
        if len(groups) < 2:
            return pd.DataFrame()

        rows = []
        pvals = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                x1 = tmp.loc[tmp[group_col] == g1, value_col].values
                x2 = tmp.loc[tmp[group_col] == g2, value_col].values
                if len(x1) < 2 or len(x2) < 2:
                    continue

                t_stat, p_raw = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
                pvals.append(p_raw)
                rows.append({
                    "group1": g1, "group2": g2,
                    "n1": len(x1), "n2": len(x2),
                    "mean1": float(np.mean(x1)), "mean2": float(np.mean(x2)),
                    "diff_mean1_minus_mean2": float(np.mean(x1) - np.mean(x2)),
                    "t_stat_welch": float(t_stat),
                    "p_raw": float(p_raw),
                })

        if not rows:
            return pd.DataFrame()

        reject, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for k in range(len(rows)):
            rows[k]["p_fdr_bh"] = float(p_fdr[k])
            rows[k]["reject_fdr_bh"] = bool(reject[k])

        return pd.DataFrame(rows).sort_values("p_fdr_bh")


    # ==============================================================================
    # HC3 wald_test_terms 테이블 파서(버전 안전)
    # ==============================================================================
    def _to_scalar(x):
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                pass
        x = np.asarray(x)
        return float(x.ravel()[0])


    def _pick(row: pd.Series, *cands, default=np.nan):
        for c in cands:
            if c in row.index:
                return row[c]
        return default


    # ==============================================================================
    # (8) 메인 함수
    # ==============================================================================
    def run_two_way_anova_agegroup_sex(
        df: pd.DataFrame,
        cohort_tag: str,
        section_dir: str,
        age_min: int = 20,
        age_max: int = 69,
        bins: list[int] = [19, 29, 39, 49, 59, 69],
        labels: list[str] = ["20s", "30s", "40s", "50s", "60s"],
        palette_sex: dict | None = None,
    ):
        if palette_sex is None:
            palette_sex = {"male": "#4285F4", "female": "#DB4437"}  # 섹션6 pie와 통일

        if "age" not in df.columns:
            raise KeyError(f"[{cohort_tag}] missing required column: age")

        # sex 컬럼 확보 (sex 없으면 gender로 대체)
        d, sex_source = ensure_sex_column(df)

        # accuracy 컬럼 선택/정리
        acc_col, scale_note = choose_accuracy_column(d)
        d, scale_conv = ensure_accuracy_pct(d, acc_col)

        # 숫자 변환 + 결측 제거
        d["age"] = pd.to_numeric(d["age"], errors="coerce")
        d = d.dropna(subset=["age", "sex", "accuracy_pct"]).copy()

        # age range
        d = d[(d["age"] >= age_min) & (d["age"] <= age_max)].copy()

        # male/female만(섹션8)
        d = d[d["sex"].isin(["male", "female"])].copy()

        # age_group binning
        d["age_group"] = pd.cut(d["age"], bins=bins, labels=labels, right=True)
        d = d.dropna(subset=["age_group"]).copy()
        d["age_group"] = d["age_group"].astype(str)

        # interaction group label
        d["group"] = d["age_group"] + "_" + d["sex"]

        # -----------------------------
        # 저장: raw / counts / means
        # -----------------------------
        raw_path = os.path.join(section_dir, f"8-0_raw_agegroup_sex_{cohort_tag}.csv")
        d.to_csv(raw_path, index=False, encoding="utf-8-sig")

        counts = pd.crosstab(d["age_group"], d["sex"])
        counts_path = os.path.join(section_dir, f"8-0_counts_agegroup_by_sex_{cohort_tag}.csv")
        counts.to_csv(counts_path, encoding="utf-8-sig")

        means = d.groupby(["age_group", "sex"], observed=False)["accuracy_pct"].mean().reset_index()
        means_path = os.path.join(section_dir, f"8-0_means_agegroup_by_sex_{cohort_tag}.csv")
        means.to_csv(means_path, index=False, encoding="utf-8-sig")

        means_wide = d.pivot_table(values="accuracy_pct", index="age_group", columns="sex",
                                   aggfunc="mean", observed=False)
        means_wide_path = os.path.join(section_dir, f"8-0_means_wide_{cohort_tag}.csv")
        means_wide.to_csv(means_wide_path, encoding="utf-8-sig")

        if PRINT_REPORT:
            print(f"\n==================== [{cohort_tag.upper()}] (8) 분석 시작 ====================")
            print(f"✅ accuracy column used: {acc_col} ({scale_note}) | analysis in %")
            print(f"✅ sex column source: {sex_source} -> normalized to 'sex'")
            print(f"- N_final (male/female & age_group valid): {len(d)}")
            print(f"✅ 저장: {raw_path}")
            print(f"✅ 저장: {counts_path}")
            print(f"✅ 저장: {means_path}")
            print(f"✅ 저장: {means_wide_path}")

        if DISPLAY_TABLES and display is not None:
            print("\n[Counts (age_group x sex)]")
            display(counts)
            print("\n[Means wide (%)]")
            display(means_wide.round(2))

        # -----------------------------
        # (8-1) Count plot
        # -----------------------------
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(
            x="age_group", hue="sex",
            data=d, order=labels,
            palette=palette_sex
        )
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width()/2.0, p.get_height()),
                ha="center", va="center", color="gray",
                xytext=(0, 5), textcoords="offset points"
            )
        plt.suptitle(f"(8-1) Number of Participants by Age Group and Sex [{cohort_tag}]",
                     fontsize=16, fontweight="bold")
        plt.xlabel("Age Group")
        plt.ylabel("Number of Participants")
        plt.legend(title="Sex", bbox_to_anchor=(1.02, 1), loc="upper left")

        fig1_png = os.path.join(section_dir, f"8-1_participant_count_by_agegroup_sex_{cohort_tag}.png")
        fig1_svg = os.path.join(section_dir, f"8-1_participant_count_by_agegroup_sex_{cohort_tag}.svg")
        plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
        plt.savefig(fig1_svg, bbox_inches="tight")
        if PRINT_REPORT:
            print(f"✅ 저장: {fig1_png}")
            print(f"✅ 저장: {fig1_svg}")
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # -----------------------------
        # (8-2) Interaction plot (mean ± SE)
        # -----------------------------
        sns.set_theme(style="ticks")
        plt.figure(figsize=(12, 7))
        sns.pointplot(
            x="age_group", y="accuracy_pct", hue="sex",
            data=d, order=labels,
            palette=palette_sex,
            markers=["o", "s"],
            errorbar="se"
        )
        plt.suptitle(f"(8-2) Mean Accuracy by Age Group and Sex [{cohort_tag}]",
                     fontsize=16, fontweight="bold")
        plt.xlabel("Age Group")
        plt.ylabel("Mean Accuracy (%)")
        plt.legend(title="Sex", bbox_to_anchor=(1.02, 1), loc="upper left")
        sns.despine()

        fig2_png = os.path.join(section_dir, f"8-2_accuracy_by_agegroup_sex_interaction_{cohort_tag}.png")
        fig2_svg = os.path.join(section_dir, f"8-2_accuracy_by_agegroup_sex_interaction_{cohort_tag}.svg")
        plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
        plt.savefig(fig2_svg, bbox_inches="tight")
        if PRINT_REPORT:
            print(f"✅ 저장: {fig2_png}")
            print(f"✅ 저장: {fig2_svg}")
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # -----------------------------
        # (8-3) Two-way ANOVA (typ=2)
        # -----------------------------
        model = ols("accuracy_pct ~ C(age_group) + C(sex) + C(age_group):C(sex)", data=d).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        anova_path = os.path.join(section_dir, f"8-3_anova_table_{cohort_tag}.csv")
        anova_table.to_csv(anova_path, encoding="utf-8-sig")

        p_age = float(anova_table.loc["C(age_group)", "PR(>F)"])
        p_sex = float(anova_table.loc["C(sex)", "PR(>F)"])
        p_int = float(anova_table.loc["C(age_group):C(sex)", "PR(>F)"])

        # eta^2
        ss_effects = anova_table["sum_sq"].drop(labels=["Residual"], errors="ignore").sum()
        ss_resid = anova_table.loc["Residual", "sum_sq"]
        ss_total = ss_effects + ss_resid
        eta_age = float(anova_table.loc["C(age_group)", "sum_sq"] / ss_total)
        eta_sex = float(anova_table.loc["C(sex)", "sum_sq"] / ss_total)
        eta_int = float(anova_table.loc["C(age_group):C(sex)", "sum_sq"] / ss_total)

        if PRINT_REPORT:
            print("\n분석 결과 요약:")
            print("-----------------------------------------")
            print(f"{'✅' if p_age < 0.05 else '❌'} 연령대 주 효과: p={p_age:.6g}")
            print(f"{'✅' if p_sex < 0.05 else '❌'} Sex 주 효과: p={p_sex:.6g}")
            print(f"{'✅' if p_int < 0.05 else '❌'} 상호작용 효과: p={p_int:.6g}")

            print("\n1. Two-Way ANOVA (typ=2):")
            print("-----------------------------------------")
            print(anova_table)
            print(f"✅ 저장: {anova_path}")

            print("\n효과 크기 (Eta-squared, η²):")
            print("-----------------------------------------")
            print(f"- 연령대 주 효과 (η²): {eta_age:.4f}")
            print(f"- Sex 주 효과 (η²): {eta_sex:.4f}")
            print(f"- 상호작용 효과 (η²): {eta_int:.4f}")

        # -----------------------------
        # Levene test (group=age_group×sex)
        # -----------------------------
        group_vals = [vals.values for _, vals in d.groupby("group")["accuracy_pct"]]
        lev_stat = np.nan
        lev_p = np.nan
        if len(group_vals) >= 2:
            lev_stat, lev_p = levene(*group_vals)
            if PRINT_REPORT:
                print("\n등분산성 검정 (Levene's Test; groups=age_group×sex):")
                print("-----------------------------------------")
                print(f"- Levene stat: {lev_stat:.4f}")
                print(f"- p: {lev_p:.6g}")
                print("경고: p < .05 -> 등분산성 가정 위배 가능" if lev_p < 0.05 else "결론: p >= .05 -> 등분산성 가정 만족")

        # -----------------------------
        # HC3 robust Wald tests (version-safe)
        # -----------------------------
        model_hc3 = ols("accuracy_pct ~ C(age_group) * C(sex)", data=d).fit(cov_type="HC3")
        wt = model_hc3.wald_test_terms(scalar=False).table

        terms = ["C(age_group)", "C(sex)", "C(age_group):C(sex)"]
        hc3_rows = []
        for t in terms:
            if t not in wt.index:
                raise KeyError(f"HC3 wald_test_terms missing term: {t}. Available: {list(wt.index)}")

            r = wt.loc[t]
            stat = _to_scalar(_pick(r, "statistic", "stat"))
            pval = _to_scalar(_pick(r, "pvalue", "p", "P>F"))

            # 버전마다 df 관련 컬럼명이 다름 -> 있으면 저장, 없으면 NaN
            df_num = _pick(r, "df_num", "df_numr", "df_num1", default=np.nan)
            df_den = _pick(r, "df_denom", "df_denomr", "df_denom1", default=np.nan)
            df_one = _pick(r, "df", default=np.nan)

            hc3_rows.append({
                "term": t,
                "wald_stat": float(stat),
                "p_value_hc3": float(pval),
                "df_num": float(_to_scalar(df_num)) if pd.notna(df_num) else np.nan,
                "df_denom": float(_to_scalar(df_den)) if pd.notna(df_den) else np.nan,
                "df": float(_to_scalar(df_one)) if pd.notna(df_one) else np.nan,
            })

        hc3_table = pd.DataFrame(hc3_rows)
        hc3_path = os.path.join(section_dir, f"8-3_hc3_wald_tests_{cohort_tag}.csv")
        hc3_table.to_csv(hc3_path, index=False, encoding="utf-8-sig")

        if PRINT_REPORT:
            print("\nHC3 robust Wald tests:")
            print("-----------------------------------------")
            print(hc3_table.to_string(index=False))
            print(f"✅ 저장: {hc3_path}")

        # -----------------------------
        # Posthoc: Tukey + FDR(Welch)
        # -----------------------------
        tukey_df = pd.DataFrame()
        fdr_df = pd.DataFrame()
        tukey_text = ""
        fdr_path = ""

        if p_int < 0.05:
            posthoc_mode = "interaction(group=age_group×sex)"
            tukey = pairwise_tukeyhsd(endog=d["accuracy_pct"], groups=d["group"], alpha=0.05)
            tukey_text = str(tukey)
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])

            fdr_df = pairwise_fdr_welch(d, "accuracy_pct", "group", alpha=0.05)

        elif p_age < 0.05:
            posthoc_mode = "main(age_group)"
            tukey = pairwise_tukeyhsd(endog=d["accuracy_pct"], groups=d["age_group"], alpha=0.05)
            tukey_text = str(tukey)
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])

            fdr_df = pairwise_fdr_welch(d, "accuracy_pct", "age_group", alpha=0.05)
        else:
            posthoc_mode = "none"

        tukey_out = os.path.join(section_dir, f"8-3_tukey_{cohort_tag}.csv")
        fdr_path = os.path.join(section_dir, f"8-3_fdr_welch_{cohort_tag}.csv")

        if tukey_text:
            tukey_df.to_csv(tukey_out, index=False, encoding="utf-8-sig")
            if PRINT_REPORT:
                print(f"\n사후 분석 1 (Tukey HSD, FWER=0.05 / mode={posthoc_mode}):")
                print("-----------------------------------------")
                print(tukey_text)
                print(f"✅ 저장: {tukey_out}")

        # fdr는 비교쌍 없으면 빈 df일 수 있음(그래도 파일은 저장해서 재현성 유지)
        fdr_df.to_csv(fdr_path, index=False, encoding="utf-8-sig")
        if PRINT_REPORT:
            print(f"\n사후 분석 2 (Pairwise Welch t-test + FDR(BH), q=0.05 / mode={posthoc_mode}):")
            print("-----------------------------------------")
            if len(fdr_df) > 0:
                print("(상위 30개 출력, 전체는 CSV 참고)\n")
                print(fdr_df.head(30).to_string(index=False))
            else:
                print("비교 가능한 그룹쌍이 부족하거나(각 그룹 표본<2), 그룹 수가 적어 빈 결과입니다.")
            print(f"✅ 저장: {fdr_path}")

        # -----------------------------
        # 진단표 출력
        # -----------------------------
        if PRINT_REPORT:
            print("\n진단용 데이터 요약:")
            print("-----------------------------------------")
            print("참여자 수(연령대 x Sex):")
            print(counts)
            print("\n그룹별 평균 정확도(%):")
            print(means_wide.round(2))

        # -----------------------------
        # Report txt 저장
        # -----------------------------
        report_lines = []
        report_lines.append("분석 결과 요약:\n-----------------------------------------")
        report_lines.append(f"{'✅' if p_age < 0.05 else '❌'} 연령대 주 효과: p={p_age:.6g}")
        report_lines.append(f"{'✅' if p_sex < 0.05 else '❌'} Sex 주 효과: p={p_sex:.6g}")
        report_lines.append(f"{'✅' if p_int < 0.05 else '❌'} 상호작용 효과: p={p_int:.6g}\n")

        report_lines.append("1. Two-Way ANOVA (typ=2):\n-----------------------------------------")
        report_lines.append(str(anova_table) + "\n")

        report_lines.append("2. Effect size (Eta-squared, η²):\n-----------------------------------------")
        report_lines.append(f"- 연령대 주 효과 (η²): {eta_age:.4f}")
        report_lines.append(f"- Sex 주 효과 (η²): {eta_sex:.4f}")
        report_lines.append(f"- 상호작용 효과 (η²): {eta_int:.4f}\n")

        report_lines.append("3. Levene's Test (groups=age_group×sex):\n-----------------------------------------")
        if not np.isnan(lev_p):
            report_lines.append(f"- Levene stat: {lev_stat:.4f}")
            report_lines.append(f"- p: {lev_p:.6g}")
            report_lines.append("경고: p < .05 -> 등분산성 가정 위배 가능\n" if lev_p < 0.05 else "결론: p >= .05 -> 등분산성 가정 만족\n")
        else:
            report_lines.append("⚠️ 그룹 수 부족으로 Levene test 불가\n")

        report_lines.append("4. HC3 robust Wald tests:\n-----------------------------------------")
        report_lines.append(hc3_table.to_string(index=False) + "\n")

        if tukey_text:
            report_lines.append(f"5. Tukey HSD (FWER=0.05 / mode={posthoc_mode}):\n-----------------------------------------")
            report_lines.append(tukey_text + "\n")

        report_lines.append(f"6. Pairwise Welch + FDR(BH) (mode={posthoc_mode}):\n-----------------------------------------")
        if len(fdr_df) > 0:
            report_lines.append("(상위 30개)\n" + fdr_df.head(30).to_string(index=False) + "\n")
        else:
            report_lines.append("빈 결과(비교쌍 부족/표본 부족)\n")
        report_lines.append(f"전체 결과 CSV: {fdr_path}\n")

        report_lines.append("진단용 데이터 요약:\n-----------------------------------------")
        report_lines.append("참여자 수(연령대 x Sex):\n" + str(counts) + "\n")
        report_lines.append("그룹별 평균 정확도(%):\n" + str(means_wide.round(2)) + "\n")

        report_text = "\n".join(report_lines)
        report_path = save_detailed_analysis(
            section_dir=section_dir,
            filename_base=f"8-3_agegroup_sex_anova_report_{cohort_tag}",
            title=f"(8-3) Age group x Sex ANOVA report [{cohort_tag}]",
            content=report_text
        )

        # meta 저장
        meta = {
            "cohort": cohort_tag,
            "accuracy_column_used": acc_col,
            "accuracy_scale_note": scale_note,
            "accuracy_scale_conversion": scale_conv,
            "sex_source_column": sex_source,
            "N_final": int(len(d)),
            "p_age_group": p_age,
            "p_sex": p_sex,
            "p_interaction": p_int,
            "eta_sq_age_group": eta_age,
            "eta_sq_sex": eta_sex,
            "eta_sq_interaction": eta_int,
            "levene_stat": float(lev_stat) if not np.isnan(lev_stat) else None,
            "levene_p": float(lev_p) if not np.isnan(lev_p) else None,
            "files": {
                "raw": raw_path,
                "counts": counts_path,
                "means_long": means_path,
                "means_wide": means_wide_path,
                "fig_count_png": fig1_png,
                "fig_interaction_png": fig2_png,
                "anova_csv": anova_path,
                "hc3_csv": hc3_path,
                "tukey_csv": tukey_out if tukey_text else None,
                "fdr_csv": fdr_path,
                "report_txt": report_path,
            }
        }
        meta_path = os.path.join(section_dir, f"8-9_meta_{cohort_tag}.json")
        _write_json(meta_path, meta)
        if PRINT_REPORT:
            print(f"✅ meta 저장: {meta_path}")
            print(f"==================== [{cohort_tag.upper()}] (8) 분석 종료 ====================\n")

        return {
            "cohort": cohort_tag,
            "N": int(len(d)),
            "p_age_group": p_age,
            "p_sex": p_sex,
            "p_interaction": p_int,
            "eta_age": eta_age,
            "eta_sex": eta_sex,
            "eta_interaction": eta_int,
            "levene_p": float(lev_p) if not np.isnan(lev_p) else np.nan,
        }


    # ==============================================================================
    # 실행부
    # ==============================================================================
    if __name__ == "__main__":
        print("==============================================================================")
        print("(8) Age group x Sex interaction (Two-Way ANOVA) + Tukey(FWER) + FDR(BH) + HC3")
        print("==============================================================================\n")

        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG  # 너 run 폴더에 맞춰 고정. 새로 만들려면 None
        _, _, section_dir = _make_section_dir("08_agegroup_sex_anova", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
        print(f"📁 섹션8 저장 위치: {section_dir}")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        # 섹션6 pie/그래프와 통일
        PALETTE_SEX = {"male": "#4285F4", "female": "#DB4437"}

        summary_rows = []
        for cohort_tag, file_path in cohort_files.items():
            try:
                df_in = pd.read_csv(file_path, encoding="utf-8-sig")
                if PRINT_REPORT:
                    print(f"✅ '{file_path}' 로드 성공 ({cohort_tag}), rows: {len(df_in)}")
            except FileNotFoundError:
                print(f"❌ '{file_path}' 파일을 찾을 수 없습니다. ({cohort_tag}) 스킵합니다.")
                continue

            out = run_two_way_anova_agegroup_sex(
                df=df_in,
                cohort_tag=cohort_tag,
                section_dir=section_dir,
                age_min=20,
                age_max=69,
                bins=[19, 29, 39, 49, 59, 69],
                labels=["20s", "30s", "40s", "50s", "60s"],
                palette_sex=PALETTE_SEX,
            )
            summary_rows.append(out)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(section_dir, "cohort_summary.csv")
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"✅ 섹션8 요약 저장: {summary_path}")
            if PRINT_REPORT:
                print(summary_df.to_string(index=False))


def _run_cell_023():
    # ==============================================================================
    # (8-extra) Prism 재현용 RAW / SUMMARY 테이블 저장
    #  - Raw (long): age_group, sex, accuracy_pct (1 row per participant)
    #  - Raw (wide): group별 raw를 column으로 (Prism-friendly)
    #  - Summary: N, mean, SD, SEM
    # ==============================================================================
    import os
    import numpy as np
    import pandas as pd

    RUN_TAG = config.RUN_TAG
    SECTION_DIR = os.path.join("outputs", f"run_{RUN_TAG}", "08_agegroup_sex_anova")
    os.makedirs(SECTION_DIR, exist_ok=True)

    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    def normalize_sex(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower().str.strip()
        s = s.replace({
            "nan": np.nan, "none": np.nan, "": np.nan,
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer not to": "prefer-not-to-say",
        })
        return s

    def ensure_sex(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "sex" in d.columns:
            d["sex"] = normalize_sex(d["sex"])
            return d
        if "gender" in d.columns:
            d["sex"] = normalize_sex(d["gender"])
            return d
        raise KeyError("missing 'sex' and no fallback 'gender'")

    def choose_accuracy_column(df: pd.DataFrame) -> str:
        # 네 파이프라인 기준: overallAccuracy_y를 %로 사용해 왔음
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        raise KeyError("missing accuracy column (overallAccuracy_y/x/overallAccuracy)")

    def to_accuracy_pct(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        # 휴리스틱: 0~1이면 소수로 보고 100 곱, 그 외는 이미 %로 간주
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            return s
        if finite.max() <= 1.0:
            return s * 100.0
        return s

    def make_age_group(age_series: pd.Series) -> pd.Series:
        age = pd.to_numeric(age_series, errors="coerce")
        return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=True)

    def prism_wide_from_long(df_long: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]) -> pd.DataFrame:
        """
        Prism-friendly wide raw table:
          - columns = groups (e.g., '20s_male', '20s_female', ...)
          - each column contains raw values stacked downward (ragged columns allowed via NaN padding)
        """
        cols = {}
        max_len = 0
        for g in group_order:
            vals = df_long.loc[df_long[group_col] == g, value_col].dropna().values.tolist()
            cols[g] = vals
            max_len = max(max_len, len(vals))

        # pad with NaN
        for g in cols:
            if len(cols[g]) < max_len:
                cols[g] = cols[g] + [np.nan] * (max_len - len(cols[g]))

        return pd.DataFrame(cols)

    def add_sem(summary_df: pd.DataFrame) -> pd.DataFrame:
        out = summary_df.copy()
        out["sem"] = out["sd"] / np.sqrt(out["n"])
        return out

    def export_prism_tables(df_in: pd.DataFrame, cohort_tag: str):
        d = df_in.copy()

        # 1) sex, age, accuracy 준비
        d = ensure_sex(d)
        acc_col = choose_accuracy_column(d)
        d["accuracy_pct"] = to_accuracy_pct(d[acc_col])
        d["age_group"] = make_age_group(d["age"])

        # 2) 필터: age_group 유효 + male/female만 (섹션8과 동일하게)
        d = d.dropna(subset=["age_group", "sex", "accuracy_pct"]).copy()
        d = d[d["sex"].isin(["male", "female"])].copy()

        # 3) interaction group
        d["group"] = d["age_group"].astype(str) + "_" + d["sex"].astype(str)

        # group order (Prism 보기 좋게)
        group_order = []
        for ag in AGE_LABELS:
            group_order.append(f"{ag}_male")
            group_order.append(f"{ag}_female")

        # 4) (A) Raw long
        raw_long = d[["age_group", "sex", "group", "accuracy_pct"]].copy()
        raw_long_path = os.path.join(SECTION_DIR, f"8-P_raw_long_agegroup_sex_{cohort_tag}.csv")
        raw_long.to_csv(raw_long_path, index=False, encoding="utf-8-sig")

        # 5) (B) Summary (N, mean, SD, SEM)
        summary = (
            raw_long.groupby("group")["accuracy_pct"]
            .agg(n="count", mean="mean", sd="std")
            .reset_index()
        )
        summary = add_sem(summary)
        # 정렬
        summary["group"] = pd.Categorical(summary["group"], categories=group_order, ordered=True)
        summary = summary.sort_values("group")
        summary_path = os.path.join(SECTION_DIR, f"8-P_summary_mean_sd_sem_agegroup_sex_{cohort_tag}.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

        # 6) (C) Raw wide (Prism-friendly)
        raw_wide = prism_wide_from_long(raw_long, group_col="group", value_col="accuracy_pct", group_order=group_order)
        raw_wide_path = os.path.join(SECTION_DIR, f"8-P_raw_wide_groups_{cohort_tag}.csv")
        raw_wide.to_csv(raw_wide_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ [Prism export 완료] cohort={cohort_tag}")
        print(f" - raw long : {raw_long_path}")
        print(f" - summary  : {summary_path}")
        print(f" - raw wide : {raw_wide_path}")

    # -----------------------------
    # 실행부 (섹션8에서 쓰는 입력 파일 그대로)
    # -----------------------------
    if __name__ == "__main__":
        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        for cohort_tag, file_path in cohort_files.items():
            try:
                df_in = pd.read_csv(file_path, encoding="utf-8-sig")
            except FileNotFoundError:
                print(f"❌ 파일 없음: {file_path} ({cohort_tag})")
                continue
            export_prism_tables(df_in, cohort_tag)


def main():
    _run_cell_012()
    _run_cell_014()
    _run_cell_017()
    _run_cell_020()
    _run_cell_023()


if __name__ == "__main__":
    main()
