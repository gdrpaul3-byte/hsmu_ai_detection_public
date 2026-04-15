"""Analysis sections for AI self-reports, strategy effectiveness, and strategy usage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_025():
    # ==============================================================================
    # (9) AI 경험/자신감/태도와 Sex 간 관계 분석 (MOBILE + WEB)
    # + 연령대별 Welch t-test 요약 + FDR(BH)
    # + (그래프) 연령대별 평균(±SE) + (50-60대) 분포 + 카이제곱
    #
    # FIX / POLICY:
    #   - 앞으로 Gender 대신 Sex 용어 사용
    #   - 입력 데이터에 'sex'가 없으면 'gender'에서 자동 생성
    #   - outputs/run_20260119_192624/09_ai_psych_by_sex 에 저장
    #   - 섹션6 팔레트와 통일: male #4285F4, female #DB4437
    # ==============================================================================
    import os
    import re
    from datetime import datetime

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import chi2_contingency, ttest_ind
    from statsmodels.stats.multitest import multipletests

    # -----------------------------
    # Switches
    # -----------------------------
    SHOW_PLOTS = True
    PRINT_REPORT = True

    # -----------------------------
    # Palette (섹션6와 통일)
    # -----------------------------
    PALETTE_SEX = {"male": "#4285F4", "female": "#DB4437"}

    # -----------------------------
    # Output dir (run tag 고정)
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    OUTPUTS_ROOT = config.OUTPUTS_DIR
    SECTION_SLUG = "09_ai_psych_by_sex"
    SECTION_DIR = os.path.join(OUTPUTS_ROOT, f"run_{RUN_TAG}", SECTION_SLUG)
    os.makedirs(SECTION_DIR, exist_ok=True)


    def _safe(s: str) -> str:
        s = str(s)
        s = re.sub(r"[^\w\s\-\.\(\)\[\]]", "_", s)
        s = re.sub(r"\s+", "_", s).strip("_")
        return s


    def save_detailed_analysis(section_dir: str, section_number: str, base_filename: str, title: str, content: str):
        """
        section_number: '9-5' 등
        base_filename: 저장명 베이스(확장자 제외)
        """
        text_filename = os.path.join(section_dir, f"{section_number}_{base_filename}.txt")
        try:
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write("============================================================\n")
                f.write(f" {title}\n")
                f.write("============================================================\n\n")
                f.write(str(content))
            if PRINT_REPORT:
                print(f"✅ 저장: {text_filename}")
        except Exception as e:
            print(f"❌ 텍스트 파일 저장 중 오류 발생: {e}")
        return text_filename


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
        입력 df에 sex 없으면 gender로부터 sex 생성.
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


    def create_summary_table_with_fdr(
        dataframe: pd.DataFrame,
        age_groups: list[str],
        variable_col: str,
        score_map: dict,
        min_agegroup_n: int = 20,
        min_sex_n: int = 10,
    ) -> pd.DataFrame:
        """
        연령대별 male vs female 평균 비교 (Welch t-test) + FDR(BH)
        """
        rows = []
        pvals = []
        row_idx_for_p = []

        for age_group in age_groups:
            df_subset = dataframe[dataframe["age_group"] == age_group].copy()
            if len(df_subset) < min_agegroup_n:
                continue

            df_subset["score"] = df_subset[variable_col].map(score_map)
            male_scores = df_subset.loc[df_subset["sex"] == "male", "score"].dropna()
            female_scores = df_subset.loc[df_subset["sex"] == "female", "score"].dropna()

            if len(male_scores) < min_sex_n or len(female_scores) < min_sex_n:
                continue

            male_mean, female_mean = male_scores.mean(), female_scores.mean()
            diff = male_mean - female_mean
            higher = "동일" if abs(diff) < 0.01 else ("male ⬆️" if diff > 0 else "female ⬆️")

            t_stat, p_value = ttest_ind(male_scores, female_scores, equal_var=False, nan_policy="omit")  # Welch
            pvals.append(p_value)

            rows.append({
                "age_group": age_group,
                "male_mean": round(male_mean, 3),
                "female_mean": round(female_mean, 3),
                "diff(male-female)": round(diff, 3),
                "higher": higher,
                "t_stat_welch": round(float(t_stat), 3),
                "p_raw": float(p_value),
                "male_n": int(len(male_scores)),
                "female_n": int(len(female_scores)),
            })
            row_idx_for_p.append(len(rows) - 1)

        if len(pvals) == 0:
            return pd.DataFrame()

        reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for k, ridx in enumerate(row_idx_for_p):
            rows[ridx]["p_fdr_bh"] = float(p_fdr[k])
            rows[ridx]["reject_fdr_bh(q<0.05)"] = bool(reject[k])

            p = rows[ridx]["p_fdr_bh"]
            if p < 0.001:
                sig = "✅ (q<.001)"
            elif p < 0.01:
                sig = "✅ (q<.01)"
            elif p < 0.05:
                sig = "✅ (q<.05)"
            else:
                sig = "❌"
            rows[ridx]["sig(FDR)"] = sig

        return pd.DataFrame(rows)


    def plot_age_sex_mean_se(
        df: pd.DataFrame,
        title: str,
        base_filename: str,
        cohort_tag: str,
        order_age: list[str],
        ylim: tuple[float, float] | None = None,
    ):
        """
        age_group별 sex 평균±SE pointplot 저장
        """
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")

        # df에 score가 있어야 함
        sns.pointplot(
            data=df,
            x="age_group",
            y="score",
            hue="sex",
            order=order_age,
            errorbar="se",
            markers=["o", "s"],
            linestyles=["-", "-"],
            palette=PALETTE_SEX,
        )
        plt.suptitle(f"{title} [{cohort_tag}]", fontsize=16, fontweight="bold")
        plt.xlabel("Age Group")
        plt.ylabel("Mean score (±SE)")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend(title="Sex", bbox_to_anchor=(1.02, 1), loc="upper left")
        sns.despine()

        fn = os.path.join(SECTION_DIR, f"{base_filename}_{cohort_tag}")
        plt.savefig(f"{fn}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fn}.svg", bbox_inches="tight")
        if PRINT_REPORT:
            print(f"✅ 그래프 저장: {fn}.png/.svg")
        if SHOW_PLOTS:
            plt.show()
        plt.close()


    def plot_50s60s_distribution(
        df_50s_60s: pd.DataFrame,
        var_col: str,
        order_keys: list[str],
        xlabels_en: list[str],
        title: str,
        base_filename: str,
        cohort_tag: str,
    ):
        """
        50-60대 분포(성별 내 퍼센트) barplot 저장
        """
        # normalize=True -> sex별 내부 비율
        proportions = (
            df_50s_60s.groupby("sex")[var_col]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percentage")
            .reset_index()
        )

        # order_keys에 없는 값은 제외 (혹시 예상 밖 값 있을 때 안전)
        proportions = proportions[proportions[var_col].isin(order_keys)].copy()

        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            data=proportions,
            x=var_col,
            y="percentage",
            hue="sex",
            order=order_keys,
            palette=PALETTE_SEX,
        )
        plt.suptitle(f"{title} [{cohort_tag}]", fontsize=16, fontweight="bold")

        # tick/label 확실히 매칭(경고/깨짐 방지)
        ax.set_xticks(range(len(order_keys)))
        ax.set_xticklabels(xlabels_en, rotation=30, ha="right")

        plt.xlabel(var_col)
        plt.ylabel("Percentage within Sex (%)")
        plt.legend(title="Sex", bbox_to_anchor=(1.02, 1), loc="upper left")
        sns.despine()

        fn = os.path.join(SECTION_DIR, f"{base_filename}_{cohort_tag}")
        plt.savefig(f"{fn}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fn}.svg", bbox_inches="tight")
        if PRINT_REPORT:
            print(f"✅ 그래프 저장: {fn}.png/.svg")
        if SHOW_PLOTS:
            plt.show()
        plt.close()


    def chi2_report(df_subset: pd.DataFrame, var_col: str, cohort_tag: str, section_number: str, base_filename: str):
        """
        50-60대 subset에서 sex x category chi-square
        """
        contingency = pd.crosstab(df_subset["sex"], df_subset[var_col])
        chi2, p_val, dof, expected = chi2_contingency(contingency)

        title = f"({section_number}) {var_col} vs Sex (Chi-square) [{cohort_tag}]"
        content = (
            f"Contingency table (Sex x {var_col}):\n{contingency.to_string()}\n\n"
            f"Chi-square: {chi2:.4f}\n"
            f"dof: {dof}\n"
            f"p-value: {p_val:.6g}\n\n"
            f"Conclusion: p {'<' if p_val < 0.05 else '>='} 0.05 -> "
            f"{'sex-dependent distribution (significant)' if p_val < 0.05 else 'no significant sex difference'}"
        )

        if PRINT_REPORT:
            print(content)

        save_detailed_analysis(
            section_dir=SECTION_DIR,
            section_number=section_number,
            base_filename=f"{base_filename}_{cohort_tag}",
            title=title,
            content=content,
        )

        # csv도 같이 저장
        csv_path = os.path.join(SECTION_DIR, f"{section_number}_{base_filename}_{cohort_tag}.csv")
        contingency.to_csv(csv_path, encoding="utf-8-sig")
        if PRINT_REPORT:
            print(f"✅ 저장: {csv_path}")


    def run_section9_for_cohort(df: pd.DataFrame, cohort_tag: str):
        print(f"\n==================== [{cohort_tag.upper()}] (9) 분석 시작 ====================")

        # sex column 확보
        df, sex_source = ensure_sex_column(df)

        # 필수 컬럼 체크 (sex는 이미 만들어짐)
        needed_cols = ["age", "sex", "aiConfidence", "aiExposureFrequency", "aiAttitude"]
        missing = [c for c in needed_cols if c not in df.columns]
        if len(missing) > 0:
            print(f"❌ [{cohort_tag}] 필요한 컬럼이 없습니다: {missing} -> 스킵합니다.")
            return

        df = df.copy()
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        bins = [19, 29, 39, 49, 59, 69]
        labels = ["20s", "30s", "40s", "50s", "60s"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

        # male/female + 유효 age_group만
        df_binary = df[df["sex"].isin(["male", "female"])].dropna(subset=["age_group"]).copy()
        print(f"✅ [{cohort_tag}] 분석 준비 완료: N={len(df_binary)} | sex source={sex_source}")

        # 저장: raw
        raw_path = os.path.join(SECTION_DIR, f"9-0_raw_ai_psych_by_sex_{cohort_tag}.csv")
        df_binary.to_csv(raw_path, index=False, encoding="utf-8-sig")
        if PRINT_REPORT:
            print(f"✅ 저장: {raw_path}")

        # 척도 매핑
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        attitude_map = {"very-negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very-positive": 5}

        # 순서/라벨
        exposure_order = ["never", "rarely", "sometimes", "weekly", "daily"]
        exposure_labels_en = ["Almost Never", "1-2 times/month", "1-2 times/week", "3-4 times/week", "Daily"]

        confidence_order = ["very-not-confident", "not-confident", "neutral", "confident", "very-confident"]
        confidence_labels_en = ["Very Unconfident", "Unconfident", "Neutral", "Confident", "Very Confident"]

        attitude_order = ["very-negative", "negative", "neutral", "positive", "very-positive"]
        attitude_labels_en = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

        # ============================================================================
        # 분석 1: 연령대별 요약 테이블 (+FDR)
        # ============================================================================
        print("\n--- 분석 1: 연령대별 요약 (Welch t-test + FDR) ---")

        # (9-1) Confidence
        print(f"\n(9-1) AI Confidence summary by age_group×sex [{cohort_tag}]")
        conf_summary = create_summary_table_with_fdr(df_binary, labels, "aiConfidence", confidence_map)
        out_csv_9_1 = os.path.join(SECTION_DIR, f"9-1_summary_confidence_by_agegroup_sex_{cohort_tag}.csv")
        if len(conf_summary) == 0:
            print("⚠️ confidence 요약 테이블 생성 실패(조건 미충족).")
        else:
            print(conf_summary.to_string(index=False))
            conf_summary.to_csv(out_csv_9_1, index=False, encoding="utf-8-sig")
            print(f"✅ 저장: {out_csv_9_1}")

        # (9-2) Exposure
        print(f"\n(9-2) AI Exposure summary by age_group×sex [{cohort_tag}]")
        exp_summary = create_summary_table_with_fdr(df_binary, labels, "aiExposureFrequency", exposure_map)
        out_csv_9_2 = os.path.join(SECTION_DIR, f"9-2_summary_exposure_by_agegroup_sex_{cohort_tag}.csv")
        if len(exp_summary) == 0:
            print("⚠️ exposure 요약 테이블 생성 실패(조건 미충족).")
        else:
            print(exp_summary.to_string(index=False))
            exp_summary.to_csv(out_csv_9_2, index=False, encoding="utf-8-sig")
            print(f"✅ 저장: {out_csv_9_2}")

        # (9-3) Attitude
        print(f"\n(9-3) AI Attitude summary by age_group×sex [{cohort_tag}]")
        att_summary = create_summary_table_with_fdr(df_binary, labels, "aiAttitude", attitude_map)
        out_csv_9_3 = os.path.join(SECTION_DIR, f"9-3_summary_attitude_by_agegroup_sex_{cohort_tag}.csv")
        if len(att_summary) == 0:
            print("⚠️ attitude 요약 테이블 생성 실패(조건 미충족).")
        else:
            print(att_summary.to_string(index=False))
            att_summary.to_csv(out_csv_9_3, index=False, encoding="utf-8-sig")
            print(f"✅ 저장: {out_csv_9_3}")

        # 요약 리포트 txt
        report_title = f"(9-1~9-3) Welch t-tests by age_group (male vs female) + FDR [{cohort_tag}]"
        report_content = (
            f"sex source column: {sex_source}\n"
            f"N (male/female & age_group valid): {len(df_binary)}\n\n"
            f"[9-1 Confidence]\n{conf_summary.to_string(index=False) if len(conf_summary) else 'EMPTY'}\n\n"
            f"[9-2 Exposure]\n{exp_summary.to_string(index=False) if len(exp_summary) else 'EMPTY'}\n\n"
            f"[9-3 Attitude]\n{att_summary.to_string(index=False) if len(att_summary) else 'EMPTY'}\n"
        )
        save_detailed_analysis(SECTION_DIR, "9-0", f"summary_tables_report_{cohort_tag}", report_title, report_content)

        # ============================================================================
        # 분석 1.5: 전 연령대 평균(±SE) 그래프 3개
        # ============================================================================
        print("\n--- 분석 1.5: 전 연령대 Mean±SE 그래프 ---")

        # confidence plot
        df_conf = df_binary.copy()
        df_conf["score"] = df_conf["aiConfidence"].map(confidence_map)
        df_conf = df_conf.dropna(subset=["score"]).copy()
        plot_age_sex_mean_se(
            df_conf,
            title="(9-A) Mean AI Confidence by Age Group and Sex",
            base_filename="9-A_mean_confidence_by_agegroup_sex",
            cohort_tag=cohort_tag,
            order_age=labels,
            ylim=(1, 5),
        )

        # exposure plot
        df_exp = df_binary.copy()
        df_exp["score"] = df_exp["aiExposureFrequency"].map(exposure_map)
        df_exp = df_exp.dropna(subset=["score"]).copy()
        plot_age_sex_mean_se(
            df_exp,
            title="(9-B) Mean AI Exposure Frequency by Age Group and Sex",
            base_filename="9-B_mean_exposure_by_agegroup_sex",
            cohort_tag=cohort_tag,
            order_age=labels,
            ylim=(1, 5),
        )

        # attitude plot
        df_att = df_binary.copy()
        df_att["score"] = df_att["aiAttitude"].map(attitude_map)
        df_att = df_att.dropna(subset=["score"]).copy()
        plot_age_sex_mean_se(
            df_att,
            title="(9-C) Mean AI Attitude by Age Group and Sex",
            base_filename="9-C_mean_attitude_by_agegroup_sex",
            cohort_tag=cohort_tag,
            order_age=labels,
            ylim=(1, 5),
        )

        # ============================================================================
        # 분석 2: 50-60대 심층 (분포 + 카이제곱) 3개 변수
        # ============================================================================
        print("\n--- 분석 2: 50-60대 심층 분석 (분포 + Chi-square) ---")
        df_50_60 = df_binary[df_binary["age_group"].isin(["50s", "60s"])].copy()

        # 표본 체크
        if len(df_50_60) < 20:
            print(f"⚠️ [{cohort_tag}] 50-60대 표본이 적어 심층 분석 스킵 (n={len(df_50_60)})")
            print(f"==================== [{cohort_tag.upper()}] (9) 분석 종료 ====================\n")
            return

        # (9-4) Exposure distribution + chi2
        print(f"\n(9-4) Exposure distribution in 50s-60s [{cohort_tag}]")
        plot_50s60s_distribution(
            df_50_60, "aiExposureFrequency",
            order_keys=exposure_order,
            xlabels_en=exposure_labels_en,
            title="(9-4) AI Exposure Frequency Distribution (50s-60s)",
            base_filename="9-4_exposure_dist_50s_60s",
            cohort_tag=cohort_tag,
        )
        print(f"(9-5) Exposure Chi-square in 50s-60s [{cohort_tag}]")
        chi2_report(df_50_60, "aiExposureFrequency", cohort_tag, "9-5", "exposure_chi2_50s_60s")

        # (9-6) Confidence distribution + chi2
        print(f"\n(9-6) Confidence distribution in 50s-60s [{cohort_tag}]")
        plot_50s60s_distribution(
            df_50_60, "aiConfidence",
            order_keys=confidence_order,
            xlabels_en=confidence_labels_en,
            title="(9-6) AI Confidence Distribution (50s-60s)",
            base_filename="9-6_confidence_dist_50s_60s",
            cohort_tag=cohort_tag,
        )
        print(f"(9-7) Confidence Chi-square in 50s-60s [{cohort_tag}]")
        chi2_report(df_50_60, "aiConfidence", cohort_tag, "9-7", "confidence_chi2_50s_60s")

        # (9-8) Attitude distribution + chi2
        print(f"\n(9-8) Attitude distribution in 50s-60s [{cohort_tag}]")
        plot_50s60s_distribution(
            df_50_60, "aiAttitude",
            order_keys=attitude_order,
            xlabels_en=attitude_labels_en,
            title="(9-8) AI Attitude Distribution (50s-60s)",
            base_filename="9-8_attitude_dist_50s_60s",
            cohort_tag=cohort_tag,
        )
        print(f"(9-9) Attitude Chi-square in 50s-60s [{cohort_tag}]")
        chi2_report(df_50_60, "aiAttitude", cohort_tag, "9-9", "attitude_chi2_50s_60s")

        print(f"==================== [{cohort_tag.upper()}] (9) 분석 종료 ====================\n")


    if __name__ == "__main__":
        print("==============================================================================")
        print("(9) AI 경험/자신감/태도와 Sex 관계 (MOBILE + WEB) + FDR + 그래프")
        print("==============================================================================\n")
        print(f"📁 섹션9 저장 위치: {SECTION_DIR}")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        for cohort_tag, file_path in cohort_files.items():
            try:
                df_in = pd.read_csv(file_path, encoding="utf-8-sig")
                print(f"✅ '{file_path}' 로드 성공 ({cohort_tag}), rows: {len(df_in)}")
            except FileNotFoundError:
                print(f"❌ '{file_path}' 파일을 찾을 수 없습니다. ({cohort_tag}) 스킵합니다.")
                continue

            run_section9_for_cohort(df_in, cohort_tag)


def _run_cell_028():
    # ==============================================================================
    # (9-extra) Prism 재현용 RAW / SUMMARY / CONTINGENCY 테이블 저장
    #  - Variables: aiConfidence, aiExposureFrequency, aiAttitude
    #  - Raw long: age_group, sex, score (1 row per participant)
    #  - Raw wide: group(age_group×sex)별 raw score를 column으로
    #  - Summary : N, mean, SD, SEM
    #  - 50-60s  : sex x category counts + within-sex proportions(%)
    # ==============================================================================
    import os
    import numpy as np
    import pandas as pd

    RUN_TAG = config.RUN_TAG
    SECTION_DIR = os.path.join("outputs", f"run_{RUN_TAG}", "09_ai_by_sex_agegroup")
    os.makedirs(SECTION_DIR, exist_ok=True)

    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    PALETTE_SEX = {"male": "#4285F4", "female": "#DB4437"}  # 그래프 팔레트(참고용)

    # ---- 섹션9에서 쓰는 매핑(동일하게 유지) ----
    CONF_MAP = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
    EXPO_MAP = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
    ATT_MAP  = {"very-negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very-positive": 5}

    VAR_SPECS = [
        {
            "tag": "confidence",
            "col": "aiConfidence",
            "map": CONF_MAP,
            "order": ["very-not-confident", "not-confident", "neutral", "confident", "very-confident"],
        },
        {
            "tag": "exposure",
            "col": "aiExposureFrequency",
            "map": EXPO_MAP,
            "order": ["never", "rarely", "sometimes", "weekly", "daily"],
        },
        {
            "tag": "attitude",
            "col": "aiAttitude",
            "map": ATT_MAP,
            "order": ["very-negative", "negative", "neutral", "positive", "very-positive"],
        },
    ]

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

    def make_age_group(age_series: pd.Series) -> pd.Series:
        age = pd.to_numeric(age_series, errors="coerce")
        return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=True)

    def prism_wide_from_long(df_long: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]) -> pd.DataFrame:
        cols = {}
        max_len = 0
        for g in group_order:
            vals = df_long.loc[df_long[group_col] == g, value_col].dropna().values.tolist()
            cols[g] = vals
            max_len = max(max_len, len(vals))
        for g in cols:
            if len(cols[g]) < max_len:
                cols[g] = cols[g] + [np.nan] * (max_len - len(cols[g]))
        return pd.DataFrame(cols)

    def add_sem(summary_df: pd.DataFrame) -> pd.DataFrame:
        out = summary_df.copy()
        out["sem"] = out["sd"] / np.sqrt(out["n"])
        return out

    def export_section9_prism_tables(df_in: pd.DataFrame, cohort_tag: str):
        d = df_in.copy()

        # ---- sex/age_group 준비 ----
        d = ensure_sex(d)
        d["age_group"] = make_age_group(d["age"])

        # male/female + age_group 유효만
        d = d.dropna(subset=["age_group", "sex"]).copy()
        d = d[d["sex"].isin(["male", "female"])].copy()

        # interaction group order: 20s_male, 20s_female, ...
        group_order = []
        for ag in AGE_LABELS:
            group_order.append(f"{ag}_male")
            group_order.append(f"{ag}_female")

        # ---- 변수별 export ----
        for spec in VAR_SPECS:
            var_tag = spec["tag"]
            col = spec["col"]
            score_map = spec["map"]
            cat_order = spec["order"]

            if col not in d.columns:
                print(f"⚠️ [{cohort_tag}] missing col: {col} -> skip {var_tag}")
                continue

            dd = d[["age_group", "sex", col]].copy()
            dd[col] = dd[col].astype(str).str.lower().str.strip()
            dd["score"] = dd[col].map(score_map)
            dd = dd.dropna(subset=["score"]).copy()

            dd["group"] = dd["age_group"].astype(str) + "_" + dd["sex"].astype(str)

            # (A) Raw long (score)
            raw_long = dd[["age_group", "sex", "group", col, "score"]].copy()
            raw_long_path = os.path.join(SECTION_DIR, f"9-P_raw_long_{var_tag}_{cohort_tag}.csv")
            raw_long.to_csv(raw_long_path, index=False, encoding="utf-8-sig")

            # (B) Summary (mean/SD/SEM/N) by age_group×sex
            summary = (
                raw_long.groupby("group")["score"]
                .agg(n="count", mean="mean", sd="std")
                .reset_index()
            )
            summary = add_sem(summary)
            summary["group"] = pd.Categorical(summary["group"], categories=group_order, ordered=True)
            summary = summary.sort_values("group")
            summary_path = os.path.join(SECTION_DIR, f"9-P_summary_mean_sd_sem_{var_tag}_{cohort_tag}.csv")
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

            # (C) Raw wide (Prism-friendly)
            raw_wide = prism_wide_from_long(raw_long, group_col="group", value_col="score", group_order=group_order)
            raw_wide_path = os.path.join(SECTION_DIR, f"9-P_raw_wide_groups_{var_tag}_{cohort_tag}.csv")
            raw_wide.to_csv(raw_wide_path, index=False, encoding="utf-8-sig")

            # (D) 50-60s contingency counts (sex × category)
            dd_50_60 = dd[dd["age_group"].isin(["50s", "60s"])].copy()
            if len(dd_50_60) > 0:
                # counts
                ct = pd.crosstab(dd_50_60["sex"], dd_50_60[col])
                # 컬럼 순서 정리(가능하면)
                ct = ct.reindex(index=["male", "female"])
                ct = ct.reindex(columns=[c for c in cat_order if c in ct.columns] + [c for c in ct.columns if c not in cat_order])
                ct_path = os.path.join(SECTION_DIR, f"9-P_contingency_50s60s_{var_tag}_{cohort_tag}.csv")
                ct.to_csv(ct_path, encoding="utf-8-sig")

                # within-sex proportions (%)
                prop = ct.div(ct.sum(axis=1), axis=0) * 100.0
                prop_path = os.path.join(SECTION_DIR, f"9-P_proportion_50s60s_{var_tag}_{cohort_tag}.csv")
                prop.to_csv(prop_path, encoding="utf-8-sig")

            print(f"\n✅ [Section9 Prism export] cohort={cohort_tag}, var={var_tag}")
            print(f" - raw long : {raw_long_path}")
            print(f" - summary  : {summary_path}")
            print(f" - raw wide : {raw_wide_path}")
            if len(dd_50_60) > 0:
                print(f" - 50-60 ct : {ct_path}")
                print(f" - 50-60 %  : {prop_path}")

    # -----------------------------
    # 실행부 (섹션9에서 쓰는 입력 파일 그대로)
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

            export_section9_prism_tables(df_in, cohort_tag)


def _run_cell_030():
    # ==============================================================================
    # (10) AI Confidence/Exposure/Attitude 점수의 Age_group × Sex 상호작용 분석 (MOBILE + WEB)
    # ------------------------------------------------------------------------------
    # - Two-Way ANOVA (age_group x sex) for each score
    # - Post-hoc: Tukey HSD (FWER) + Pairwise Welch t-test + FDR(BH)
    # - Plots + reports + tables 저장
    # - (Prism 재현용 extra) RAW wide/long + mean/sd/sem/n 저장
    #   => Prism에서 point plot(mean±SEM) / column scatter / box/violin 재현 가능
    # ==============================================================================

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests

    # -----------------------------
    # CONFIG
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    SECTION_DIR = os.path.join("outputs", f"run_{RUN_TAG}", "10_ai_scores_anova")
    os.makedirs(SECTION_DIR, exist_ok=True)

    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]
    PALETTE_SEX = {"male": "#4285F4", "female": "#DB4437"}  # 섹션9/8과 동일 톤

    # 섹션9과 동일 매핑
    CONF_MAP = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
    EXPO_MAP = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
    ATT_MAP  = {"very-negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very-positive": 5}

    SCORES = [
        {"prefix": "10-A", "score_col": "confidence_score", "name": "AI Confidence", "raw_col": "aiConfidence", "map": CONF_MAP},
        {"prefix": "10-B", "score_col": "exposure_score",   "name": "AI Exposure",   "raw_col": "aiExposureFrequency", "map": EXPO_MAP},
        {"prefix": "10-C", "score_col": "attitude_score",   "name": "AI Attitude",   "raw_col": "aiAttitude", "map": ATT_MAP},
    ]

    PRINT_REPORT = True  # 실행 중 콘솔에 리포트 출력 여부


    # -----------------------------
    # IO helpers
    # -----------------------------
    def save_text(path: str, title: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f" {title}\n")
            f.write("============================================================\n\n")
            f.write(content)

    def save_df(df: pd.DataFrame, path: str):
        df.to_csv(path, index=False, encoding="utf-8-sig")


    # -----------------------------
    # Normalize / preprocess
    # -----------------------------
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

    def ensure_sex(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """
        - sex 컬럼이 있으면 sex 사용
        - 없으면 gender에서 sex 생성 (너의 파이프라인 호환)
        """
        d = df.copy()
        if "sex" in d.columns:
            d["sex"] = normalize_sex(d["sex"])
            return d, "sex"
        if "gender" in d.columns:
            d["sex"] = normalize_sex(d["gender"])
            return d, "gender"
        raise KeyError("missing required column: 'sex' (and no fallback 'gender')")

    def make_age_group(age_series: pd.Series) -> pd.Series:
        age = pd.to_numeric(age_series, errors="coerce")
        return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=True)

    def pairwise_fdr_welch(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> pd.DataFrame:
        """
        모든 그룹쌍에 대해 Welch t-test 수행 후 BH(FDR) 보정.
        """
        tmp = df[[group_col, value_col]].dropna().copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna(subset=[value_col]).copy()

        groups = sorted(tmp[group_col].unique().tolist())
        if len(groups) < 2:
            return pd.DataFrame()

        rows, pvals = [], []
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
                    "n1": int(len(x1)), "n2": int(len(x2)),
                    "mean1": float(np.mean(x1)), "mean2": float(np.mean(x2)),
                    "diff_mean1_minus_mean2": float(np.mean(x1) - np.mean(x2)),
                    "t_stat_welch": float(t_stat),
                    "p_raw": float(p_raw),
                })

        if len(rows) == 0:
            return pd.DataFrame()

        reject, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for k in range(len(rows)):
            rows[k]["p_fdr_bh"] = float(p_fdr[k])
            rows[k]["reject_fdr_bh(q<0.05)"] = bool(reject[k])

        return pd.DataFrame(rows).sort_values("p_fdr_bh")


    # -----------------------------
    # Prism export helpers
    # -----------------------------
    def prism_wide_from_long(df_long: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]) -> pd.DataFrame:
        cols = {}
        max_len = 0
        for g in group_order:
            vals = df_long.loc[df_long[group_col] == g, value_col].dropna().tolist()
            cols[g] = vals
            max_len = max(max_len, len(vals))
        for g in cols:
            if len(cols[g]) < max_len:
                cols[g] = cols[g] + [np.nan] * (max_len - len(cols[g]))
        return pd.DataFrame(cols)

    def export_prism_tables(df_analysis: pd.DataFrame, score_col: str, raw_col: str, prefix: str, score_name: str, cohort_tag: str):
        """
        Prism 재현용:
        - RAW long (age_group, sex, group, raw_category, score)
        - RAW wide (columns=20s_male,20s_female,...)
        - summary (n/mean/sd/sem)
        """
        group_order = []
        for ag in AGE_LABELS:
            group_order.append(f"{ag}_male")
            group_order.append(f"{ag}_female")

        # long
        raw_long = df_analysis[["age_group", "sex", "group", raw_col, score_col]].copy()
        raw_long = raw_long.rename(columns={score_col: "score"})
        p2 = os.path.join(SECTION_DIR, f"{prefix}-P2_prism_raw_long_{score_name.lower().replace(' ','_')}_{cohort_tag}.csv")
        save_df(raw_long, p2)

        # wide
        raw_wide = prism_wide_from_long(raw_long, group_col="group", value_col="score", group_order=group_order)
        p1 = os.path.join(SECTION_DIR, f"{prefix}-P1_prism_raw_wide_{score_name.lower().replace(' ','_')}_{cohort_tag}.csv")
        raw_wide.to_csv(p1, index=False, encoding="utf-8-sig")

        # summary
        summ = (
            raw_long.groupby("group")["score"]
            .agg(n="count", mean="mean", sd="std")
            .reset_index()
        )
        summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
        summ["group"] = pd.Categorical(summ["group"], categories=group_order, ordered=True)
        summ = summ.sort_values("group")
        p3 = os.path.join(SECTION_DIR, f"{prefix}-P3_prism_summary_mean_sd_sem_{score_name.lower().replace(' ','_')}_{cohort_tag}.csv")
        save_df(summ, p3)

        print(f"✅ Prism tables saved ({cohort_tag} | {score_name})")
        print(f" - {os.path.basename(p1)}")
        print(f" - {os.path.basename(p2)}")
        print(f" - {os.path.basename(p3)}")


    # -----------------------------
    # Main analysis per score
    # -----------------------------
    def analyze_interaction_anova(df_analysis: pd.DataFrame, score_col: str, score_name: str, prefix: str, cohort_tag: str):
        """
        Two-way ANOVA + Tukey + FDR, plots, and save everything into SECTION_DIR.
        """
        age_order = AGE_LABELS

        # (prefix-1) plot
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        sns.pointplot(
            x="age_group", y=score_col, hue="sex", data=df_analysis,
            order=age_order,
            palette=PALETTE_SEX,
            markers=["o", "s"], errorbar="se"
        )
        plt.suptitle(f"({prefix}-1) Mean {score_name} Score by Age Group and Sex [{cohort_tag}]",
                     fontsize=16, fontweight="bold")
        plt.xlabel("Age Group")
        plt.ylabel(f"Mean {score_name} Score (1-5)")

        base_fn = os.path.join(SECTION_DIR, f"{prefix}-1_{score_name.lower().replace(' ', '_')}_interaction_{cohort_tag}")
        plt.savefig(f"{base_fn}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{base_fn}.svg", dpi=300, bbox_inches="tight")
        print(f"✅ ({prefix}-1) plot saved: {os.path.basename(base_fn)}.png/.svg")
        plt.show()

        # (prefix-2) ANOVA
        model = ols(f"{score_col} ~ C(age_group) + C(sex) + C(age_group):C(sex)", data=df_analysis).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        p_age = float(anova_table.loc["C(age_group)", "PR(>F)"])
        p_sex = float(anova_table.loc["C(sex)", "PR(>F)"])
        p_int = float(anova_table.loc["C(age_group):C(sex)", "PR(>F)"])

        summary = "분석 결과 요약:\n-----------------------------------------\n"
        summary += f"{'✅' if p_age < 0.05 else '❌'} 연령대 주 효과: p={p_age:.6g}\n"
        summary += f"{'✅' if p_sex < 0.05 else '❌'} Sex 주 효과: p={p_sex:.6g}\n"
        summary += f"{'✅' if p_int < 0.05 else '❌'} 상호작용 효과: p={p_int:.6g}\n\n"

        post_hoc_report = ""
        fdr_report = ""

        # Tukey 기준: interaction 유의면 group(연령xsex), 아니면 age_group
        if p_int < 0.05 and df_analysis["group"].nunique() >= 2:
            tukey = pairwise_tukeyhsd(endog=df_analysis[score_col], groups=df_analysis["group"], alpha=0.05)
            post_hoc_report += (
                "사후 분석 1 (Tukey HSD, FWER=0.05 / groups=age_group×sex):\n"
                "-----------------------------------------\n"
                f"{tukey}\n\n"
            )

            fdr_df = pairwise_fdr_welch(df_analysis, score_col, "group", alpha=0.05)
            if len(fdr_df) > 0:
                fdr_path = os.path.join(
                    SECTION_DIR,
                    f"{prefix}-2_fdr_pairwise_{score_name.lower().replace(' ', '_')}_interaction_{cohort_tag}.csv"
                )
                save_df(fdr_df, fdr_path)
                fdr_report += (
                    "사후 분석 2 (Pairwise Welch t-test + FDR(BH), q=0.05 / groups=age_group×sex):\n"
                    "-----------------------------------------\n"
                    f"- 저장 파일: {os.path.basename(fdr_path)}\n"
                    f"{fdr_df.head(30).to_string(index=False)}\n"
                    "\n(표가 길 수 있어 상위 30개만 출력; 전체는 CSV 참고)\n\n"
                )
            else:
                fdr_report += "사후 분석 2 (FDR) 스킵: 비교 가능한 그룹쌍이 부족합니다.\n\n"

        elif p_age < 0.05 and df_analysis["age_group"].nunique() >= 2:
            tukey = pairwise_tukeyhsd(endog=df_analysis[score_col], groups=df_analysis["age_group"], alpha=0.05)
            post_hoc_report += (
                "사후 분석 1 (Tukey HSD, FWER=0.05 / main effect=age_group):\n"
                "-----------------------------------------\n"
                f"{tukey}\n\n"
            )

            fdr_df = pairwise_fdr_welch(df_analysis, score_col, "age_group", alpha=0.05)
            if len(fdr_df) > 0:
                fdr_path = os.path.join(
                    SECTION_DIR,
                    f"{prefix}-2_fdr_pairwise_{score_name.lower().replace(' ', '_')}_agegroup_{cohort_tag}.csv"
                )
                save_df(fdr_df, fdr_path)
                fdr_report += (
                    "사후 분석 2 (Pairwise Welch t-test + FDR(BH), q=0.05 / main effect=age_group):\n"
                    "-----------------------------------------\n"
                    f"- 저장 파일: {os.path.basename(fdr_path)}\n"
                    f"{fdr_df.to_string(index=False)}\n\n"
                )
            else:
                fdr_report += "사후 분석 2 (FDR) 스킵: 비교 가능한 그룹쌍이 부족합니다.\n\n"

        report = (
            summary
            + "1. Two-Way ANOVA (typ=2):\n-----------------------------------------\n"
            + str(anova_table)
            + "\n\n"
            + post_hoc_report
            + fdr_report
        )

        if PRINT_REPORT:
            print(report)

        rep_path = os.path.join(SECTION_DIR, f"{prefix}-2_{score_name.lower().replace(' ','_')}_anova_report_{cohort_tag}.txt")
        save_text(rep_path, f"({prefix}-2) {score_name} ANOVA report [{cohort_tag}]", report)
        print(f"✅ report saved: {os.path.basename(rep_path)}")


    # -----------------------------
    # Cohort runner
    # -----------------------------
    def run_section10_for_cohort(df_in: pd.DataFrame, cohort_tag: str):
        print(f"\n==================== [{cohort_tag.upper()}] (10) START ====================")

        # sex 준비
        df, sex_source = ensure_sex(df_in)

        needed = ["age", "sex", "aiConfidence", "aiExposureFrequency", "aiAttitude"]
        missing = [c for c in needed if c not in df.columns]
        if len(missing) > 0:
            print(f"❌ [{cohort_tag}] missing required cols: {missing} -> skip")
            return None

        df = df.copy()
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["sex"] = normalize_sex(df["sex"])
        df = df[df["sex"].isin(["male", "female"])].copy()

        df["age_group"] = make_age_group(df["age"])

        # 점수 매핑
        df["confidence_score"] = df["aiConfidence"].astype(str).str.lower().str.strip().map(CONF_MAP)
        df["exposure_score"] = df["aiExposureFrequency"].astype(str).str.lower().str.strip().map(EXPO_MAP)
        df["attitude_score"] = df["aiAttitude"].astype(str).str.lower().str.strip().map(ATT_MAP)

        # 유효값만
        df = df.dropna(subset=["age_group", "confidence_score", "exposure_score", "attitude_score"]).copy()
        df["group"] = df["age_group"].astype(str) + "_" + df["sex"].astype(str)

        # 저장: raw master(섹션10)
        raw_path = os.path.join(SECTION_DIR, f"10-0_raw_scores_agegroup_sex_{cohort_tag}.csv")
        save_df(df, raw_path)

        print(f"✅ [{cohort_tag}] final N={len(df)} | sex source={sex_source}")
        print(f"✅ saved raw: {os.path.basename(raw_path)}")

        # 점수별 분석 + Prism export
        for spec in SCORES:
            prefix = spec["prefix"]
            score_col = spec["score_col"]
            score_name = spec["name"]
            raw_col = spec["raw_col"]

            analyze_interaction_anova(df, score_col=score_col, score_name=score_name, prefix=prefix, cohort_tag=cohort_tag)
            export_prism_tables(df, score_col=score_col, raw_col=raw_col, prefix=prefix, score_name=score_name, cohort_tag=cohort_tag)

        # 요약 row
        return {
            "cohort": cohort_tag,
            "N": int(len(df)),
            "sex_source": sex_source,
        }


    if __name__ == "__main__":
        print("==============================================================================")
        print("(10) AI Confidence/Exposure/Attitude: Age_group × Sex interaction (ANOVA + Tukey + FDR)")
        print(f"📁 section10 dir: {SECTION_DIR}")
        print("==============================================================================\n")

        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        summary_rows = []
        for cohort_tag, file_path in cohort_files.items():
            try:
                df_in = pd.read_csv(file_path, encoding="utf-8-sig")
                print(f"✅ loaded: {file_path} ({cohort_tag}), rows={len(df_in)}")
            except FileNotFoundError:
                print(f"❌ missing file: {file_path} ({cohort_tag})")
                continue

            out = run_section10_for_cohort(df_in, cohort_tag)
            if out is not None:
                summary_rows.append(out)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summ_path = os.path.join(SECTION_DIR, "cohort_summary.csv")
            save_df(summary_df, summ_path)
            print(f"\n✅ cohort summary saved: {os.path.basename(summ_path)}")


def _run_cell_033():
    # ==============================================================================
    # (11) Strategy effectiveness analysis (MOBILE + WEB) + FDR(BH)  [ENGLISH OUTPUT]
    # ------------------------------------------------------------------------------
    # - Strategy used vs not-used: Welch t-test
    # - Multiple testing correction: FDR(BH)
    # - Robust strategy parsing (NO substring bug: "text" vs "texture")
    # - Auto accuracy column selection + auto scaling to %
    # - Save cohort-specific tables/plots/reports + Prism raw tables (wide/long/summary)
    # ==============================================================================

    import os
    import re
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    from matplotlib.patches import Patch

    # -----------------------------
    # Global config
    # -----------------------------
    RUN_TAG = "run_20260119_192624"  # 필요하면 여기만 바꿔
    SECTION_DIR = os.path.join("outputs", RUN_TAG, "11_strategy_effectiveness")

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web": config.WEB_AGE_FILTERED,
    }

    # Prism/plot palette (통일)
    PALETTE_SIG_POS = "#DB4437"  # significant improvement
    PALETTE_SIG_NEG = "#4285F4"  # significant decrease
    PALETTE_NS = "#BDBDBD"       # not significant
    PALETTE_SEX = {"male": "#4285F4", "female": "#DB4437"}

    sns.set_theme(style="whitegrid")


    # -----------------------------
    # Utils
    # -----------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


    def save_text_report(out_dir: str, section_number: str, base_filename: str, title: str, content: str):
        fn = os.path.join(out_dir, f"{section_number}_{base_filename}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f"{title}\n")
            f.write("============================================================\n\n")
            f.write(str(content))
        print(f"✅ Saved report: {fn}")


    def choose_accuracy_column(df: pd.DataFrame):
        """
        우선순위로 정확도 컬럼 선택.
        - overallAccuracy_y (섹션7에서 쓴 컬럼)
        - overallAccuracy
        - accuracy
        - accuracy_pct
        """
        candidates = ["overallAccuracy_y", "overallAccuracy", "accuracy_pct", "accuracy"]
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError("No usable accuracy column found among: overallAccuracy_y / overallAccuracy / accuracy_pct / accuracy")


    def ensure_accuracy_pct(df: pd.DataFrame, acc_col: str):
        """
        acc_col 값이 0~1 스케일이면 0~100%로 변환해 accuracy_pct 생성.
        이미 0~100 스케일이면 그대로 사용.
        """
        x = pd.to_numeric(df[acc_col], errors="coerce")
        x = x.dropna()
        if len(x) == 0:
            raise ValueError("Accuracy column has no numeric values after coercion.")

        # heuristic: if max <= 1.2 => treat as proportion
        mx = float(x.max())
        if mx <= 1.2:
            df["accuracy_pct"] = pd.to_numeric(df[acc_col], errors="coerce") * 100.0
            scale_note = f"{acc_col} treated as proportion (0-1) -> converted to %"
        else:
            df["accuracy_pct"] = pd.to_numeric(df[acc_col], errors="coerce")
            scale_note = f"{acc_col} treated as % scale (kept as-is)"
        return df, scale_note


    def normalize_sex_from_gender(df: pd.DataFrame):
        """
        - 분석에서는 Sex 용어 사용.
        - df에 sex가 없으면 gender에서 생성.
        """
        if "sex" in df.columns:
            src = "sex"
            series = df["sex"]
        elif "gender" in df.columns:
            src = "gender"
            series = df["gender"]
        else:
            raise KeyError("Missing both 'sex' and 'gender' columns.")

        s = series.astype(str).str.lower().str.strip()
        s = s.replace({
            "nan": np.nan, "none": np.nan, "": np.nan,
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "male ": "male", "female ": "female",
            "prefer not to say": np.nan,
            "prefer_not_to_say": np.nan,
            "prefer not to": np.nan,
        })
        df["sex"] = s
        return df, src


    def tokenize_strategy_cell(cell):
        """
        strategy cell을 안전하게 토큰화해서 set으로 반환.
        - substring 매칭 금지! (text vs texture 같은 버그 방지)
        - 구분자: comma / semicolon / pipe / slash
        """
        if pd.isna(cell):
            return set()

        txt = str(cell).strip().lower()
        if txt == "" or txt == "nan" or txt == "none":
            return set()

        # "on"이면 전체 선택은 바깥에서 처리
        if txt == "on":
            return {"__ALL__"}

        # 구분자 통일
        txt = re.sub(r"[;/|]+", ",", txt)
        txt = txt.replace("\\", ",").replace("/", ",")

        # 공백/연속 콤마 정리
        parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
        return set(parts)


    def build_strategy_flags(df: pd.DataFrame, all_strategies):
        """
        strategy string -> tokens -> 정확히 일치하는 멤버십으로 strat_{s} boolean 만들기.
        """
        token_sets = df["strategy"].apply(tokenize_strategy_cell)

        # "on" 처리: __ALL__ 포함이면 전체 strategies를 선택했다고 간주
        def expand_all(tokens: set):
            if "__ALL__" in tokens:
                return set(all_strategies)
            return tokens

        token_sets = token_sets.apply(expand_all)

        for s in all_strategies:
            df[f"strat_{s}"] = token_sets.apply(lambda t: s in t)

        # raw tokens 저장(디버깅/리포팅용)
        df["strategy_tokens"] = token_sets.apply(lambda t: ",".join(sorted(list(t))))
        return df


    def hedges_g(x, y):
        """
        Hedges' g (bias-corrected Cohen's d). x=used, y=not used
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan

        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        sp = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)) if (nx + ny - 2) > 0 else np.nan
        if sp == 0 or np.isnan(sp):
            return 0.0

        d = (np.mean(x) - np.mean(y)) / sp
        # correction
        J = 1 - (3 / (4 * (nx + ny) - 9)) if (nx + ny) > 2 else 1.0
        return float(J * d)


    def make_prism_raw_wide(results_rows, out_dir: str, cohort_tag: str):
        """
        Prism raw wide table:
        columns: {Strategy}_used, {Strategy}_not_used
        각 컬럼 길이가 다를 수 있으니 NaN padding
        """
        cols = {}
        max_len = 0
        for r in results_rows:
            s = r["strategy"]
            used_vals = r["used_vals"]
            not_vals = r["not_used_vals"]
            cols[f"{s}_used"] = used_vals
            cols[f"{s}_not_used"] = not_vals
            max_len = max(max_len, len(used_vals), len(not_vals))

        # pad
        for k, arr in cols.items():
            if len(arr) < max_len:
                cols[k] = list(arr) + [np.nan] * (max_len - len(arr))

        wide = pd.DataFrame(cols)
        fn = os.path.join(out_dir, f"11-P1_prism_raw_wide_strategy_effectiveness_{cohort_tag}.csv")
        wide.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"✅ Prism raw wide saved: {fn}")
        return fn


    def make_prism_raw_long(results_rows, out_dir: str, cohort_tag: str):
        """
        Prism raw long table:
        columns: Strategy, UsedFlag, Accuracy_pct
        """
        rows = []
        for r in results_rows:
            s = r["strategy"]
            for v in r["used_vals"]:
                rows.append({"Strategy": s, "UsedFlag": "Used", "Accuracy_pct": v})
            for v in r["not_used_vals"]:
                rows.append({"Strategy": s, "UsedFlag": "Not used", "Accuracy_pct": v})

        long_df = pd.DataFrame(rows)
        fn = os.path.join(out_dir, f"11-P2_prism_raw_long_strategy_effectiveness_{cohort_tag}.csv")
        long_df.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"✅ Prism raw long saved: {fn}")
        return fn


    def make_prism_summary(results_df: pd.DataFrame, out_dir: str, cohort_tag: str):
        """
        Prism summary table: mean/sd/sem/n for Used and Not used
        """
        # results_df에는 mean_used/mean_not_used/n_used/n_not_used/sd_used/sd_not_used/sem_used/sem_not_used 포함하도록 구성
        summ = results_df[[
            "strategy_label",
            "n_used", "mean_used", "sd_used", "sem_used",
            "n_not_used", "mean_not_used", "sd_not_used", "sem_not_used",
            "p_raw", "q_fdr_bh", "reject_fdr_bh(q<0.05)"
        ]].copy()

        summ.columns = [
            "Strategy",
            "N_used", "Mean_used", "SD_used", "SEM_used",
            "N_not_used", "Mean_not_used", "SD_not_used", "SEM_not_used",
            "p_raw", "q_fdr_bh", "FDR_reject"
        ]

        fn = os.path.join(out_dir, f"11-P3_prism_summary_mean_sd_sem_strategy_effectiveness_{cohort_tag}.csv")
        summ.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"✅ Prism summary saved: {fn}")
        return fn


    # -----------------------------
    # Main analysis
    # -----------------------------
    def run_section11_for_cohort(file_path: str, cohort_tag: str, out_dir: str):
        print(f"\n==================== [{cohort_tag.upper()}] (11) START ====================")
        ensure_dir(out_dir)

        # Load
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path} [{cohort_tag}] -> skip")
            return None

        # Required: strategy + accuracy
        if "strategy" not in df.columns:
            print(f"❌ Missing required column: strategy [{cohort_tag}] -> skip")
            return None

        try:
            acc_col = choose_accuracy_column(df)
        except KeyError as e:
            print(f"❌ {e} [{cohort_tag}] -> skip")
            return None

        df = df.copy()
        df, sex_src = normalize_sex_from_gender(df)
        df, scale_note = ensure_accuracy_pct(df, acc_col)

        df = df.dropna(subset=["strategy", "accuracy_pct"]).copy()
        df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"], errors="coerce")
        df = df.dropna(subset=["accuracy_pct"]).copy()

        print(f"✅ Accuracy column used: {acc_col} | {scale_note}")
        print(f"✅ Sex source: {sex_src}")
        print(f"✅ Analysis N (valid strategy+accuracy): {len(df)} [{cohort_tag}]")

        # Strategy list
        all_strategies = [
            "hands", "eyes", "background", "texture", "painting-like", "lighting",
            "beauty", "symmetry", "text", "feeling", "dont-know", "random", "other"
        ]
        strategy_names_en = {
            "hands": "Hands",
            "eyes": "Eyes",
            "background": "Background",
            "texture": "Texture",
            "painting-like": "Painting-like",
            "lighting": "Lighting",
            "beauty": "Aesthetics/Beauty",
            "symmetry": "Symmetry",
            "text": "Text",
            "feeling": "Feeling/Intuition",
            "dont-know": "Don't know",
            "random": "Random guess",
            "other": "Other"
        }

        # Build robust flags (NO substring bug)
        df = build_strategy_flags(df, all_strategies)

        # Save raw
        raw_path = os.path.join(out_dir, f"11-0_raw_strategy_accuracy_{cohort_tag}.csv")
        keep_cols = ["accuracy_pct", "sex", "strategy", "strategy_tokens"] + [f"strat_{s}" for s in all_strategies]
        df[keep_cols].to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved raw: {raw_path}")

        # Welch t-test per strategy
        results = []
        prism_rows = []  # store raw values for prism wide/long

        MIN_N_USED = 10
        MIN_N_NOT = 10

        for s in all_strategies:
            used = df.loc[df[f"strat_{s}"], "accuracy_pct"].astype(float).dropna().values
            not_used = df.loc[~df[f"strat_{s}"], "accuracy_pct"].astype(float).dropna().values

            if len(used) < MIN_N_USED or len(not_used) < MIN_N_NOT:
                continue

            t_stat, p_raw = stats.ttest_ind(used, not_used, equal_var=False, nan_policy="omit")

            mean_used = float(np.mean(used))
            mean_not = float(np.mean(not_used))
            diff_pp = mean_used - mean_not

            sd_used = float(np.std(used, ddof=1)) if len(used) >= 2 else np.nan
            sd_not = float(np.std(not_used, ddof=1)) if len(not_used) >= 2 else np.nan
            sem_used = float(sd_used / math.sqrt(len(used))) if len(used) >= 2 else np.nan
            sem_not = float(sd_not / math.sqrt(len(not_used))) if len(not_used) >= 2 else np.nan

            g = hedges_g(used, not_used)

            results.append({
                "strategy": s,
                "strategy_label": strategy_names_en.get(s, s),
                "mean_used": mean_used,
                "sd_used": sd_used,
                "sem_used": sem_used,
                "mean_not_used": mean_not,
                "sd_not_used": sd_not,
                "sem_not_used": sem_not,
                "mean_diff_pp": float(diff_pp),
                "hedges_g": float(g),
                "t_stat_welch": float(t_stat),
                "p_raw": float(p_raw),
                "n_used": int(len(used)),
                "n_not_used": int(len(not_used)),
            })

            prism_rows.append({
                "strategy": s,
                "used_vals": list(used),
                "not_used_vals": list(not_used),
            })

        results_df = pd.DataFrame(results)
        if results_df.empty:
            print(f"⚠️ No analyzable strategies (sample size constraints) [{cohort_tag}]")
            return None

        # FDR(BH)
        reject, qvals, _, _ = multipletests(results_df["p_raw"].values, alpha=0.05, method="fdr_bh")
        results_df["q_fdr_bh"] = qvals
        results_df["reject_fdr_bh(q<0.05)"] = reject

        def fdr_sig_label(q):
            if q < 0.001:
                return "YES (q<.001)"
            elif q < 0.01:
                return "YES (q<.01)"
            elif q < 0.05:
                return "YES (q<.05)"
            return "NO"

        results_df["significant_fdr"] = results_df["q_fdr_bh"].apply(fdr_sig_label)

        # (11-1) Save & print table
        print(f"\n--- (11-1) Strategy effectiveness table (FDR) [{cohort_tag}] ---")
        table = results_df[
            ["strategy_label",
             "mean_used", "sem_used",
             "mean_not_used", "sem_not_used",
             "mean_diff_pp", "hedges_g",
             "t_stat_welch", "p_raw", "q_fdr_bh",
             "reject_fdr_bh(q<0.05)", "n_used", "n_not_used"]
        ].copy()

        table.columns = [
            "Strategy",
            "Accuracy (Used)", "SEM (Used)",
            "Accuracy (Not used)", "SEM (Not used)",
            "Diff (pp)", "Hedges g",
            "t (Welch)", "p_raw", "q_fdr_bh",
            "FDR_reject", "N_used", "N_not_used"
        ]
        table = table.sort_values("q_fdr_bh")
        print(table.round(3).to_string(index=False))

        out_csv_11_1 = os.path.join(out_dir, f"11-1_strategy_effectiveness_table_{cohort_tag}.csv")
        table.to_csv(out_csv_11_1, index=False, encoding="utf-8-sig")
        print(f"✅ Saved: {out_csv_11_1}")

        # (11-2) Plot effect sizes (color by FDR)
        print(f"\n--- (11-2) Effect size plot (colored by FDR) [{cohort_tag}] ---")
        plot_data = results_df.sort_values("mean_diff_pp", ascending=True).copy()

        colors = []
        for _, r in plot_data.iterrows():
            if bool(r["reject_fdr_bh(q<0.05)"]):
                colors.append(PALETTE_SIG_POS if r["mean_diff_pp"] > 0 else PALETTE_SIG_NEG)
            else:
                colors.append(PALETTE_NS)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(plot_data["strategy_label"], plot_data["mean_diff_pp"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Accuracy difference (Used - Not used, percentage points)")
        ax.set_title(f"(11-2) Strategy effect sizes & significance (FDR) [{cohort_tag}]")

        legend_elements = [
            Patch(facecolor=PALETTE_SIG_POS, label="Significant improvement (FDR q<0.05)"),
            Patch(facecolor=PALETTE_SIG_NEG, label="Significant decrease (FDR q<0.05)"),
            Patch(facecolor=PALETTE_NS, label="Not significant (FDR)")
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        base_fn_11_2 = os.path.join(out_dir, f"11-2_strategy_effect_sizes_{cohort_tag}")
        plt.savefig(f"{base_fn_11_2}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{base_fn_11_2}.svg", dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot: {base_fn_11_2}.png/.svg")
        plt.show()

        # (11-3) Key findings (FDR)
        print(f"\n--- (11-3) Key findings (FDR) [{cohort_tag}] ---")
        sig_pos = results_df[(results_df["reject_fdr_bh(q<0.05)"]) & (results_df["mean_diff_pp"] > 0)]
        sig_neg = results_df[(results_df["reject_fdr_bh(q<0.05)"]) & (results_df["mean_diff_pp"] < 0)]

        summary_lines = []
        summary_lines.append(f"[{cohort_tag}] Key findings (FDR-based)\n" + "=" * 55 + "\n")
        summary_lines.append(f"Accuracy column: {acc_col}\n")
        summary_lines.append(f"Scaling note: {scale_note}\n")
        summary_lines.append(f"Sex source: {sex_src}\n")
        summary_lines.append(f"Analysis N: {len(df)}\n\n")

        if not sig_pos.empty:
            summary_lines.append("Significant improvements (Used > Not used):\n")
            for _, r in sig_pos.sort_values("q_fdr_bh").iterrows():
                summary_lines.append(
                    f" - {r['strategy_label']}: +{r['mean_diff_pp']:.2f} pp "
                    f"(q={r['q_fdr_bh']:.4f}, g={r['hedges_g']:.3f}, N_used={r['n_used']})\n"
                )
        else:
            summary_lines.append("Significant improvements: none (FDR q<0.05)\n")

        if not sig_neg.empty:
            summary_lines.append("\nSignificant decreases (Used < Not used):\n")
            for _, r in sig_neg.sort_values("q_fdr_bh").iterrows():
                summary_lines.append(
                    f" - {r['strategy_label']}: {r['mean_diff_pp']:.2f} pp "
                    f"(q={r['q_fdr_bh']:.4f}, g={r['hedges_g']:.3f}, N_used={r['n_used']})\n"
                )
        else:
            summary_lines.append("\nSignificant decreases: none (FDR q<0.05)\n")

        summary_text = "".join(summary_lines)
        print(summary_text)
        save_text_report(out_dir, "11-3", f"strategy_key_findings_{cohort_tag}", f"(11-3) Strategy key findings [{cohort_tag}]", summary_text)

        # Prism tables
        print(f"\n--- Prism tables [{cohort_tag}] ---")
        make_prism_raw_wide(prism_rows, out_dir, cohort_tag)
        make_prism_raw_long(prism_rows, out_dir, cohort_tag)
        make_prism_summary(results_df.sort_values("q_fdr_bh"), out_dir, cohort_tag)

        # Return for cohort summary
        out = {
            "cohort": cohort_tag,
            "N": int(len(df)),
            "accuracy_col": acc_col,
            "sex_source": sex_src,
            "n_strategies_tested": int(len(results_df)),
            "n_sig_fdr": int(results_df["reject_fdr_bh(q<0.05)"].sum()),
            "top_sig_pos": (sig_pos.sort_values("q_fdr_bh").iloc[0]["strategy_label"] if len(sig_pos) > 0 else ""),
            "top_sig_neg": (sig_neg.sort_values("q_fdr_bh").iloc[0]["strategy_label"] if len(sig_neg) > 0 else ""),
        }

        print(f"==================== [{cohort_tag.upper()}] (11) END ====================\n")
        return out


    if __name__ == "__main__":
        print("==============================================================================")
        print("(11) Strategy effectiveness analysis (MOBILE + WEB) + FDR  [ENGLISH OUTPUT]")
        print("==============================================================================\n")

        ensure_dir(SECTION_DIR)

        cohort_summaries = []
        for cohort_tag, file_path in COHORT_FILES.items():
            out = run_section11_for_cohort(file_path, cohort_tag, SECTION_DIR)
            if out is not None:
                cohort_summaries.append(out)

        if len(cohort_summaries) > 0:
            summary_df = pd.DataFrame(cohort_summaries)
            summary_path = os.path.join(SECTION_DIR, "cohort_summary.csv")
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"✅ Cohort summary saved: {summary_path}")
            print(summary_df.to_string(index=False))


def _run_cell_036():
    # ==============================================================================
    # (11-4) Multivariate strategy effects (OLS + robust HC3) + FDR(BH) [ENGLISH]
    # - Adds: PRINT_REPORT option to display full report in console
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    PRINT_REPORT = True          # ✅ 콘솔에 report 전체 출력
    PRINT_MAX_CHARS = None       # None이면 제한 없이 출력, 숫자면 그 길이로 자름(예: 12000)

    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def save_text_report(out_dir, section_number, base_filename, title, content):
        fn = os.path.join(out_dir, f"{section_number}_{base_filename}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write(f"{title}\n")
            f.write("============================================================\n\n")
            f.write(str(content))
        print(f"✅ Saved report: {fn}")

    def pick_accuracy_column(df: pd.DataFrame):
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns:
                return c
        return None

    def coerce_accuracy_to_percent(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        vmax = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else np.nan
        if np.isfinite(vmax) and vmax <= 1.5:
            return s * 100.0, "treated as proportion (0-1) -> converted to %"
        return s, "treated as percent scale (0-100)"

    def normalize_sex_from_gender(df: pd.DataFrame):
        if "sex" in df.columns:
            sex_col, source = "sex", "sex"
        elif "gender" in df.columns:
            sex_col, source = "gender", "gender"
        else:
            return df, None

        s = df[sex_col].astype(str).str.lower().str.strip()
        s = s.replace({
            "nan": np.nan, "none": np.nan, "": np.nan,
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "male": "male", "female": "female",
            "prefer not to say": np.nan,
            "prefer_not_to_say": np.nan,
            "prefer not to": np.nan,
        })
        df["sex"] = s
        return df, source

    def build_strategy_dummies(strategy_series: pd.Series, all_strategies):
        s = strategy_series.astype(str).replace({"on": ",".join(all_strategies)})
        tokens = s.str.split(",")
        tokens = tokens.apply(lambda lst: [x.strip() for x in lst] if isinstance(lst, list) else [])
        token_sets = tokens.apply(set)

        out = {}
        for strat in all_strategies:
            out[f"strat_{strat}"] = token_sets.apply(lambda st: int(strat in st))
        return pd.DataFrame(out)

    def compute_vif(X: pd.DataFrame):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        cols = X.columns.tolist()
        rows = []
        for i, c in enumerate(cols):
            if c == "const":
                continue
            try:
                vif = variance_inflation_factor(X.values, i)
            except Exception:
                vif = np.nan
            rows.append({"term": c, "VIF": float(vif) if np.isfinite(vif) else np.nan})
        return pd.DataFrame(rows).sort_values("VIF", ascending=False)

    def run_section11_4(file_path: str, cohort_tag: str, out_dir: str):
        print(f"\n==================== [{cohort_tag.upper()}] (11-4) START ====================")
        sns.set_theme(style="whitegrid")
        ensure_dir(out_dir)

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"✅ Loaded: {file_path} [{cohort_tag}] (rows={len(df)})")
        except FileNotFoundError:
            print(f"❌ Missing file: {file_path} [{cohort_tag}] -> skip")
            return

        acc_col = pick_accuracy_column(df)
        if acc_col is None:
            print(f"❌ Missing accuracy columns (overallAccuracy*) [{cohort_tag}] -> skip")
            return
        for c in ["strategy", "age"]:
            if c not in df.columns:
                print(f"❌ Missing required column: {c} [{cohort_tag}] -> skip")
                return

        df, sex_source = normalize_sex_from_gender(df.copy())
        if sex_source is None:
            print(f"❌ Missing sex/gender column [{cohort_tag}] -> skip")
            return
        print(f"✅ Sex source: {sex_source}")

        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df[acc_col], acc_note = coerce_accuracy_to_percent(df[acc_col])
        print(f"✅ Accuracy column used: {acc_col} | {acc_note}")

        df = df.dropna(subset=["strategy", "age", "sex", acc_col]).copy()
        df = df[df["sex"].isin(["male", "female"])].copy()
        print(f"✅ N after dropping missing + keeping male/female: {len(df)} [{cohort_tag}]")

        all_strategies = [
            "hands", "eyes", "background", "texture", "painting-like", "lighting",
            "beauty", "symmetry", "text", "feeling", "dont-know", "random", "other"
        ]
        strategy_names_en = {
            "hands": "Hands",
            "eyes": "Eyes",
            "background": "Background",
            "texture": "Texture",
            "painting-like": "Painting-like",
            "lighting": "Lighting",
            "beauty": "Aesthetics/Beauty",
            "symmetry": "Symmetry",
            "text": "Text",
            "feeling": "Feeling/Intuition",
            "dont-know": "Don't know",
            "random": "Random guess",
            "other": "Other"
        }

        strat_df = build_strategy_dummies(df["strategy"], all_strategies)

        sex_dummy = (df["sex"] == "female").astype(int).rename("sex_female")

        X = pd.concat(
            [
                pd.DataFrame({"age": df["age"].values}),
                pd.DataFrame({"sex_female": sex_dummy.values}),
                strat_df.reset_index(drop=True),
            ],
            axis=1
        )
        y = df[acc_col].reset_index(drop=True)

        X = sm.add_constant(X, has_constant="add")

        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")

        keep = X.notna().all(axis=1) & y.notna()
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"⚠️ [{cohort_tag}] Dropping {dropped} rows due to NaN after coercion.")
        X = X.loc[keep].astype(float)
        y = y.loc[keep].astype(float)

        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")
        except Exception as e:
            print(f"❌ OLS fit failed [{cohort_tag}]: {e}")
            dbg = pd.DataFrame({"y": y}).join(X)
            dbg_fn = os.path.join(out_dir, f"11-4_debug_design_matrix_{cohort_tag}.csv")
            dbg.to_csv(dbg_fn, index=False, encoding="utf-8-sig")
            print(f"🧪 Saved debug matrix: {dbg_fn}")
            return

        rows = []
        for s in all_strategies:
            col = f"strat_{s}"
            if col not in model.params.index:
                continue
            rows.append({
                "strategy": s,
                "strategy_label": strategy_names_en.get(s, s),
                "beta_pp": float(model.params[col]),
                "se_hc3": float(model.bse[col]),
                "t": float(model.tvalues[col]),
                "p_raw": float(model.pvalues[col]),
            })
        coef_df = pd.DataFrame(rows).sort_values("p_raw")

        reject, qvals, _, _ = multipletests(coef_df["p_raw"].values, alpha=0.05, method="fdr_bh")
        coef_df["q_fdr_bh"] = qvals
        coef_df["reject_fdr_bh(q<0.05)"] = reject

        out_csv = os.path.join(out_dir, f"11-4_strategy_multivariate_regression_table_{cohort_tag}.csv")
        coef_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Saved table: {out_csv}")

        try:
            vif_df = compute_vif(X)
            vif_fn = os.path.join(out_dir, f"11-4_vif_{cohort_tag}.csv")
            vif_df.to_csv(vif_fn, index=False, encoding="utf-8-sig")
            print(f"✅ Saved VIF: {vif_fn}")
        except Exception as e:
            print(f"⚠️ VIF computation skipped [{cohort_tag}]: {e}")

        report_lines = []
        report_lines.append(f"[Data]\ncohort={cohort_tag}\nN={len(y)}\n")
        report_lines.append(f"Accuracy column: {acc_col}\nScaling: {acc_note}\nSex source: {sex_source}\n\n")
        report_lines.append("[Model]\n")
        report_lines.append("overallAccuracy(%) ~ age + sex(female) + all strategy indicators\n")
        report_lines.append("Robust SE: HC3\n\n")
        report_lines.append("[Model fit]\n")
        report_lines.append(f"R-squared: {model.rsquared:.4f}\n")
        report_lines.append(f"Adj. R-squared: {model.rsquared_adj:.4f}\n")
        report_lines.append(f"F-statistic: {model.fvalue:.4f}\n")
        report_lines.append(f"Prob(F): {model.f_pvalue:.6g}\n\n")
        report_lines.append("[Strategy coefficients (FDR)]\n")
        report_lines.append(coef_df.sort_values("q_fdr_bh").round(4).to_string(index=False))
        report_lines.append("\n\n[statsmodels summary]\n")
        report_lines.append(str(model.summary()))
        report_text = "".join(report_lines)

        # ✅ 콘솔 DISPLAY 추가
        if PRINT_REPORT:
            print("\n" + "="*78)
            print(f"(11-4) FULL REPORT DISPLAY [{cohort_tag}]")
            print("="*78)
            if PRINT_MAX_CHARS is None:
                print(report_text)
            else:
                print(report_text[:PRINT_MAX_CHARS] + "\n... [TRUNCATED] ...\n")

        save_text_report(
            out_dir,
            "11-4",
            f"strategy_multivariate_regression_report_{cohort_tag}",
            f"(11-4) Multivariate regression for strategy effects [{cohort_tag}]",
            report_text
        )

        plot_df = coef_df.sort_values("beta_pp").copy()
        plot_df["ci_low"] = plot_df["beta_pp"] - 1.96 * plot_df["se_hc3"]
        plot_df["ci_high"] = plot_df["beta_pp"] + 1.96 * plot_df["se_hc3"]

        colors = []
        for _, r in plot_df.iterrows():
            if r["reject_fdr_bh(q<0.05)"]:
                colors.append("#DB4437" if r["beta_pp"] > 0 else "#4285F4")
            else:
                colors.append("#BDBDBD")

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.hlines(y=plot_df["strategy_label"], xmin=plot_df["ci_low"], xmax=plot_df["ci_high"],
                  color="gray", linewidth=2, alpha=0.8)
        ax.scatter(plot_df["beta_pp"], plot_df["strategy_label"], color=colors, s=80, alpha=0.9)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Regression coefficient beta (percentage points) [95% CI, HC3]")
        ax.set_title(f"(11-4) Multivariate strategy effects (controls: age/sex/other strategies) [{cohort_tag}]")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#DB4437", label="Significant improvement (FDR q<0.05)"),
            Patch(facecolor="#4285F4", label="Significant decrease (FDR q<0.05)"),
            Patch(facecolor="#BDBDBD", label="Not significant (FDR)")
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        base_fn = os.path.join(out_dir, f"11-4_strategy_multivariate_coeffplot_{cohort_tag}")
        plt.savefig(f"{base_fn}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{base_fn}.svg", dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot: {base_fn}.png/.svg")
        plt.show()

        print(f"==================== [{cohort_tag.upper()}] (11-4) END ====================\n")


    if __name__ == "__main__":
        print("==============================================================================")
        print("(11-4) Multivariate regression for strategy effects (MOBILE + WEB) [ENGLISH]")
        print("==============================================================================\n")

        out_dir = r"outputs\run_20260119_192624\11_strategy_effectiveness"
        cohort_files = {
            "mobile": config.MOBILE_AGE_FILTERED,
            "web": config.WEB_AGE_FILTERED,
        }

        for cohort_tag, fp in cohort_files.items():
            run_section11_4(fp, cohort_tag, out_dir)


def _run_cell_040():
    # ==============================================================================
    # (12) Age-group-by-Sex usage rate differences for "effective strategies"
    # ------------------------------------------------------------------------------
    # FIXED: RUN_DIR fixed to outputs/run_20260119_192624 (no latest-run ambiguity)
    # - Effective strategies are pulled from Section 11-4 multivariate table (q<0.05 & beta>0)
    # - Chi-square per (age_group × strategy): Male vs Female usage rates
    # - Multiple testing correction: FDR(BH)
    # - Saves tables + plots + Prism tables under the SAME run folder
    # ==============================================================================
    import os
    import re
    from pathlib import Path
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    PRINT_REPORT = True
    sns.set_theme(style="whitegrid")

    # -----------------------------
    # ✅ FIXED RUN CONFIG
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    RUN_DIR = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}"
    SECTION_DIR = RUN_DIR / "12_strategy_usage"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    # input files (fixed)
    INPUT_CSV = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web":    config.WEB_AGE_FILTERED,
    }

    # Age bins
    AGE_BINS = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    PALETTE_SEX = {"Male": "#4285F4", "Female": "#DB4437"}

    STRATEGY_LABEL_EN = {
        "hands": "Hands",
        "eyes": "Eyes",
        "background": "Background",
        "texture": "Texture",
        "painting-like": "Painting-like",
        "lighting": "Lighting",
        "beauty": "Aesthetics/Beauty",
        "symmetry": "Symmetry",
        "text": "Text",
        "feeling": "Feeling/Intuition",
        "dont-know": "Don't know",
        "random": "Random guess",
        "other": "Other",
    }
    ALL_STRATEGIES = list(STRATEGY_LABEL_EN.keys())

    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_input_csv(path: Path) -> Path:
        """Use fixed path; if missing, fallback to CWD."""
        if path.exists():
            return path
        alt = Path.cwd() / path.name
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Cannot find input CSV: {path} (also checked {alt})")

    def save_text_report(out_path: Path, title: str, content: str):
        out_path.write_text(
            "============================================================\n"
            f"{title}\n"
            "============================================================\n\n"
            + content,
            encoding="utf-8"
        )

    def normalize_sex_from_gender(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower().str.strip()
        s = s.replace({
            "nan": np.nan, "none": np.nan, "": np.nan,
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "prefer not to say": np.nan,
            "prefer_not_to_say": np.nan,
            "prefer not to": np.nan,
        })
        return s

    def choose_accuracy_column(df: pd.DataFrame) -> str:
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns:
                return c
        raise KeyError("No accuracy column found among: overallAccuracy_y / overallAccuracy / overallAccuracy_x")

    def ensure_accuracy_percent(df: pd.DataFrame, acc_col: str) -> tuple[pd.DataFrame, str]:
        d = df.copy()
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
        mx = d[acc_col].max(skipna=True)
        if pd.notna(mx) and mx <= 1.5:
            d[acc_col] = d[acc_col] * 100.0
            return d, f"{acc_col} treated as proportion (0-1) -> converted to %"
        return d, f"{acc_col} treated as percent scale"

    def parse_strategy_list(cell) -> list[str]:
        if pd.isna(cell):
            return []
        s = str(cell).strip().lower()
        if s == "on":
            return ALL_STRATEGIES.copy()
        tokens = re.split(r"[,;|/\\]+", s)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [t for t in tokens if t in STRATEGY_LABEL_EN]

    def add_strategy_indicators(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        parsed = d["strategy"].apply(parse_strategy_list)
        for k in ALL_STRATEGIES:
            d[f"strat_{k}"] = parsed.apply(lambda lst: int(k in lst))
        return d

    def load_effective_strategies_fixed(run_dir: Path, cohort_tag: str) -> list[str]:
        """
        Pull effective strategies from 11-4 multivariate table within the SAME RUN_DIR.
        Criteria: q_fdr_bh < 0.05 and beta_pp > 0.
        """
        fn = run_dir / "11_strategy_effectiveness" / f"11-4_strategy_multivariate_regression_table_{cohort_tag}.csv"
        if not fn.exists():
            raise FileNotFoundError(f"Missing 11-4 table (required now): {fn}")

        reg = pd.read_csv(fn, encoding="utf-8-sig")
        needed = {"strategy", "beta_pp", "q_fdr_bh"}
        if not needed.issubset(set(reg.columns)):
            raise ValueError(f"11-4 table missing columns {needed}: {fn}")

        eff = (
            reg[(reg["q_fdr_bh"] < 0.05) & (reg["beta_pp"] > 0)]["strategy"]
            .dropna().astype(str).str.strip().unique().tolist()
        )
        eff = [s for s in eff if s in STRATEGY_LABEL_EN]
        if len(eff) == 0:
            print(f"⚠️ [{cohort_tag}] No effective strategies under q<0.05 & beta>0. Using empty list.")
        return eff

    def cramers_v_2x2(chi2: float, n: int) -> float:
        if n <= 0:
            return np.nan
        return float(np.sqrt(chi2 / n))

    # -----------------------------
    # Main runner for each cohort
    # -----------------------------
    def run_section12_for_cohort(cohort_tag: str):
        print(f"\n==================== [{cohort_tag.upper()}] (12) START ====================")
        print(f"📁 section dir: {SECTION_DIR}")

        input_csv = resolve_input_csv(INPUT_CSV[cohort_tag])
        print(f"📄 input: {input_csv}")

        df = pd.read_csv(input_csv, encoding="utf-8-sig")
        print(f"✅ Loaded rows={len(df):,}")

        # required columns
        for c in ["age", "gender", "strategy"]:
            if c not in df.columns:
                raise KeyError(f"[{cohort_tag}] missing required column: {c}")

        acc_col = choose_accuracy_column(df)
        df, scale_note = ensure_accuracy_percent(df, acc_col)
        print(f"✅ Accuracy: {acc_col} | {scale_note}")
        print("✅ Sex source: gender")

        df = df.copy()
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["sex"] = normalize_sex_from_gender(df["gender"])
        df = df[df["sex"].isin(["male", "female"])].copy()
        df = df.dropna(subset=["age", "strategy", "sex"]).copy()

        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        df = df.dropna(subset=["age_group"]).copy()

        df = add_strategy_indicators(df)
        print(f"✅ Analysis N (male/female with age_group): {len(df):,}")

        raw_out = SECTION_DIR / f"12-0_raw_prepped_{cohort_tag}.csv"
        df.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"✅ Saved raw: {raw_out}")

        # ✅ effective strategies from 11-4 (fixed run)
        effective = load_effective_strategies_fixed(RUN_DIR, cohort_tag)
        print(f"✅ [{cohort_tag}] Effective strategies (11-4, q<.05 & beta>0): {effective}")

        if len(effective) == 0:
            print(f"⚠️ [{cohort_tag}] No effective strategies found -> END")
            return

        rows = []
        for ag in AGE_LABELS:
            d_ag = df[df["age_group"] == ag].copy()
            if len(d_ag) < 20:
                continue

            for skey in effective:
                used_col = f"strat_{skey}"
                tmp = d_ag[["sex", used_col]].copy()
                tmp["used"] = tmp[used_col].astype(int)

                ct = pd.crosstab(tmp["sex"], tmp["used"])

                # enforce 2x2 order
                for rr in ["male", "female"]:
                    if rr not in ct.index:
                        ct.loc[rr] = 0
                for cc in [0, 1]:
                    if cc not in ct.columns:
                        ct[cc] = 0
                ct = ct.loc[["male", "female"], [0, 1]]

                male_total = int(ct.loc["male"].sum())
                female_total = int(ct.loc["female"].sum())
                if male_total == 0 or female_total == 0:
                    continue

                male_used = int(ct.loc["male", 1])
                female_used = int(ct.loc["female", 1])
                male_usage = male_used / male_total * 100.0
                female_usage = female_used / female_total * 100.0

                chi2, p_raw, dof, expected = chi2_contingency(ct.values, correction=False)
                n_total = int(ct.values.sum())
                v = cramers_v_2x2(chi2, n_total)

                rows.append({
                    "cohort": cohort_tag,
                    "age_group": ag,
                    "strategy_key": skey,
                    "strategy": STRATEGY_LABEL_EN[skey],
                    "male_usage_pct": male_usage,
                    "female_usage_pct": female_usage,
                    "diff_female_minus_male_pct": female_usage - male_usage,
                    "male_used": male_used,
                    "male_total": male_total,
                    "female_used": female_used,
                    "female_total": female_total,
                    "chi2": float(chi2),
                    "dof": int(dof),
                    "p_raw": float(p_raw),
                    "cramers_v": float(v),
                })

        results_df = pd.DataFrame(rows)
        if results_df.empty:
            print(f"⚠️ [{cohort_tag}] No valid tests produced -> END")
            return

        reject, qvals, _, _ = multipletests(results_df["p_raw"].values, alpha=0.05, method="fdr_bh")
        results_df["q_fdr_bh"] = qvals
        results_df["fdr_reject_q_lt_0_05"] = reject

        out_table = results_df.sort_values(["q_fdr_bh", "p_raw"]).copy()
        out_csv = SECTION_DIR / f"12-1_strategy_usage_summary_table_{cohort_tag}.csv"
        out_table.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"\n--- (12-1) Summary table (Chi-square + FDR) [{cohort_tag}] ---")
        show_cols = [
            "age_group", "strategy",
            "male_usage_pct", "female_usage_pct", "diff_female_minus_male_pct",
            "p_raw", "q_fdr_bh", "fdr_reject_q_lt_0_05",
            "chi2", "cramers_v",
            "male_total", "female_total"
        ]
        print(out_table[show_cols].round(4).to_string(index=False))
        print(f"✅ Saved: {out_csv}")

        # Plot (bar by age group, facet by strategy)
        plot_melt = out_table.melt(
            id_vars=["age_group", "strategy"],
            value_vars=["male_usage_pct", "female_usage_pct"],
            var_name="sex",
            value_name="usage_pct"
        )
        plot_melt["sex"] = plot_melt["sex"].replace({
            "male_usage_pct": "Male",
            "female_usage_pct": "Female"
        })

        g = sns.catplot(
            data=plot_melt,
            x="age_group", y="usage_pct",
            hue="sex", col="strategy",
            kind="bar",
            order=AGE_LABELS,
            height=5, aspect=1.10,
            palette=PALETTE_SEX,
            legend_out=True
        )
        g.fig.suptitle(
            f"(12-2) Usage rate of effective strategies by age group and Sex [{cohort_tag}]",
            y=1.03, fontsize=16, fontweight="bold"
        )
        g.set_axis_labels("Age group", "Usage rate (%)")
        g.set_titles("Strategy: {col_name}")
        g.add_legend(title="Sex")

        base_fn = SECTION_DIR / f"12-2_strategy_usage_comparison_plot_{cohort_tag}"
        plt.savefig(str(base_fn) + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(str(base_fn) + ".svg", dpi=300, bbox_inches="tight")
        plt.show()

        # Print FDR-significant cells
        sig = out_table[out_table["fdr_reject_q_lt_0_05"]].copy()
        print(f"\n--- (12-3) FDR-significant cells only [{cohort_tag}] ---")
        if sig.empty:
            print("No FDR-significant age_group×sex usage differences for effective strategies.\n")
        else:
            print(sig[show_cols].round(4).to_string(index=False))
            print()

        print(f"==================== [{cohort_tag.upper()}] (12) END ====================\n")


    # -----------------------------
    # Run both cohorts
    # -----------------------------
    print("==============================================================================")
    print("(12) Effective strategy usage by age group and Sex (MOBILE + WEB) [FIXED RUN]")
    print("==============================================================================")
    print("RUN_DIR =", RUN_DIR)
    print("SECTION_DIR =", SECTION_DIR)
    print("")

    run_section12_for_cohort("mobile")
    run_section12_for_cohort("web")


def main():
    _run_cell_025()
    _run_cell_028()
    _run_cell_030()
    _run_cell_033()
    _run_cell_036()
    _run_cell_040()


if __name__ == "__main__":
    main()
