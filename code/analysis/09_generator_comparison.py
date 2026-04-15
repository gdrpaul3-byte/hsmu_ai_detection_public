"""Analysis section for ChatGPT-vs-Gemini generator comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_137():
    # ==============================================================================
    # (31 v1.1) MOBILE ONLY — Participant-level: ChatGPT vs Gemini difference on AI discrimination
    # - Auto-detect generator source column by scanning all string columns in responses_export.csv
    # - Primary: AI trials only, participant-level accuracy: acc_ai_chatgpt vs acc_ai_gemini (paired)
    # - Save: Prism tables + stats + plots (y fixed 0~100)
    # ==============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import ttest_rel, wilcoxon
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # -----------------------------
    # Config
    # -----------------------------
    COHORT_FILE = config.MOBILE_AGE_FILTERED   # participant-level (mobile)
    RESP_FILE   = config.RAW_RESPONSES                          # trial-level
    OUTDIR = config.OUTPUTS_DIR / "outputs_31_gpt_vs_gemini_mobile_only_v1_1"

    AGE_MIN, AGE_MAX = 20, 69
    YLIM = (0, 100)
    SCATTER_ALPHA = 0.25
    AUTO_UNIT_FIX = True

    # -----------------------------
    # Helpers
    # -----------------------------
    def ensure_dir(p): os.makedirs(p, exist_ok=True)

    def save_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def sem(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def infer_and_fix_rt_unit(rt_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(rt_series, errors="coerce")
        med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else np.nan
        if np.isnan(med):
            return s
        if AUTO_UNIT_FIX and med < 20:  # likely seconds
            return s * 1000.0
        return s

    def pick_first_existing(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def normalize_sex(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        s = s.replace({"m":"male","f":"female","woman":"female","man":"male"})
        s = s.where(s.isin(["male","female"]), np.nan)
        return s

    def normalize_image_kind(x: str):
        s = str(x).strip().lower()
        if s == "real":
            return "Real"
        if "ai" in s:
            return "AI"
        return "Other"

    def detect_gen_model_from_text(x: str):
        """
        Return {'chatgpt','gemini',None} from free text.
        """
        s = str(x).strip().lower()
        if s in ["nan", "", "none"]:
            return np.nan
        # chatgpt buckets
        if ("chatgpt" in s) or (" openai" in s) or ("openai" in s) or ("gpt-" in s) or (s == "gpt") or ("gpt" in s):
            return "chatgpt"
        # gemini buckets
        if ("gemini" in s) or (" google" in s) or ("google" in s) or ("bard" in s):
            return "gemini"
        return np.nan

    def paired_effect_dz(diff):
        diff = pd.to_numeric(diff, errors="coerce").dropna()
        if len(diff) <= 1:
            return np.nan
        return diff.mean() / diff.std(ddof=1)

    def auto_find_generator_column(df_resp: pd.DataFrame, ai_mask: pd.Series):
        """
        Scan all object/string columns and score how well they classify AI trials into chatgpt vs gemini.
        Choose the best column by coverage (classified non-NA) then by balance.
        """
        obj_cols = [c for c in df_resp.columns if df_resp[c].dtype == "object"]
        if not obj_cols:
            return None, None

        best = None
        best_score = -1
        best_stats = None

        ai_df = df_resp.loc[ai_mask, obj_cols].copy()

        for c in obj_cols:
            s = ai_df[c].astype(str)
            pred = s.map(detect_gen_model_from_text)
            coverage = pred.notna().mean()  # fraction classified
            if coverage < 0.01:
                continue

            # balance: prefer columns that capture both chatgpt and gemini (not only one)
            counts = pred.value_counts(dropna=True).to_dict()
            n_chat = counts.get("chatgpt", 0)
            n_gem  = counts.get("gemini", 0)
            both = (n_chat > 0) and (n_gem > 0)

            # score: coverage heavily + bonus if both present
            score = coverage + (0.10 if both else 0.0)

            if score > best_score:
                best_score = score
                best = c
                best_stats = {
                    "coverage": float(coverage),
                    "n_chatgpt": int(n_chat),
                    "n_gemini": int(n_gem),
                    "both_present": bool(both),
                }

        return best, best_stats

    # -----------------------------
    # Load & prep
    # -----------------------------
    def prep_mobile():
        ensure_dir(OUTDIR)

        df_part = pd.read_csv(COHORT_FILE, encoding="utf-8-sig")

        # age filter
        if "age" in df_part.columns:
            df_part["age"] = pd.to_numeric(df_part["age"], errors="coerce")
            df_part = df_part[(df_part["age"] >= AGE_MIN) & (df_part["age"] <= AGE_MAX)].copy()

        # id col
        id_col = pick_first_existing(df_part, ["participantId","participant_id","pid","id"])
        if id_col is None:
            raise ValueError("participant-level 파일에서 participantId 컬럼을 못 찾음.")
        df_part = df_part.rename(columns={id_col:"participantId"})

        # Sex col (Gender -> Sex)
        sex_src = pick_first_existing(df_part, ["sex","Sex","gender","Gender"])
        if sex_src is not None:
            df_part["sex"] = normalize_sex(df_part[sex_src])
        else:
            df_part["sex"] = np.nan

        # accuracy col
        acc_col = pick_first_existing(df_part, ["overallAccuracy","overallAccuracy_y","overallAccuracy_x","accuracy"])
        if acc_col is not None:
            df_part["overallAccuracy"] = pd.to_numeric(df_part[acc_col], errors="coerce")
        else:
            df_part["overallAccuracy"] = np.nan

        # avgRT col
        rt_part_col = pick_first_existing(df_part, ["avgRT","avgRT_overall","meanRT","rt_mean"])
        if rt_part_col is not None:
            df_part["avgRT_overall"] = pd.to_numeric(df_part[rt_part_col], errors="coerce")
        else:
            df_part["avgRT_overall"] = np.nan

        # trial-level
        df_resp = pd.read_csv(RESP_FILE, encoding="utf-8-sig")

        need = ["participantId","rt","isCorrect","imageType"]
        miss = [c for c in need if c not in df_resp.columns]
        if miss:
            raise ValueError(f"responses_export.csv에서 필수 컬럼이 없음: {miss}")

        # restrict to mobile cohort participants
        pid_set = set(df_part["participantId"].dropna().unique())
        d = df_resp[df_resp["participantId"].isin(pid_set)].copy()

        # remove practice if trial column exists
        if "trial" in d.columns:
            d = d[~d["trial"].astype(str).str.lower().str.startswith("practice")].copy()

        # normalize RT
        d["rt"] = infer_and_fix_rt_unit(d["rt"])
        d["rt"] = pd.to_numeric(d["rt"], errors="coerce")
        d = d.dropna(subset=["rt"])

        # image kind
        d["image_kind"] = d["imageType"].apply(normalize_image_kind)
        d = d[d["image_kind"].isin(["Real","AI"])].copy()

        # Auto-find generator column from AI trials
        ai_mask = (d["image_kind"] == "AI")
        gen_col, stats = auto_find_generator_column(d, ai_mask)

        if gen_col is None:
            # show hint: list object columns
            obj_cols = [c for c in d.columns if d[c].dtype == "object"]
            raise ValueError(
                "responses_export.csv에서 'gpt/gemini' 흔적이 있는 컬럼을 못 찾음.\n"
                "즉, trial-level 데이터에 생성모델 정보가 없거나, gpt/gemini 문자열이 전혀 없음.\n"
                f"참고: object 컬럼 후보들 = {obj_cols[:30]}{'...' if len(obj_cols)>30 else ''}"
            )

        # Build gen_model from that column (AI trials only meaningful)
        d["gen_model"] = d[gen_col].astype(str).map(detect_gen_model_from_text)

        return df_part, d, gen_col, stats

    # -----------------------------
    # Main analysis
    # -----------------------------
    def run_section31_mobile():
        df_part, d, gen_col, stats = prep_mobile()
        out = OUTDIR
        ensure_dir(out)

        print("==============================================================================")
        print("(31 v1.1 | MOBILE ONLY) GPT vs Gemini difference (AI trials) — AUTO DETECT")
        print("==============================================================================")
        print(f"✅ Cohort participants: {df_part['participantId'].nunique()}")
        print(f"✅ Trials after filters: {len(d)}")
        print(f"✅ Auto-selected generator column: {gen_col}")
        print(f"    - coverage (AI trials classified) = {stats['coverage']*100:.2f}%")
        print(f"    - n_chatgpt={stats['n_chatgpt']} | n_gemini={stats['n_gemini']} | both_present={stats['both_present']}")

        # AI trials with known model
        ai = d[d["image_kind"]=="AI"].copy()
        ai = ai.dropna(subset=["gen_model"])
        ai = ai[ai["gen_model"].isin(["chatgpt","gemini"])].copy()

        if len(ai) == 0:
            raise ValueError("AI trials에서 chatgpt/gemini로 분류된 데이터가 0개임. (컬럼 선택은 됐지만 매칭 실패)")

        # participant-level AI accuracy by model
        acc_ai = (
            ai.groupby(["participantId","gen_model"])["isCorrect"]
            .mean()
            .unstack()
            .reset_index()
        )
        for c in ["chatgpt","gemini"]:
            if c in acc_ai.columns:
                acc_ai[c] = acc_ai[c] * 100.0

        meta = df_part[["participantId","age","sex","overallAccuracy","avgRT_overall"]].copy()
        wide = acc_ai.merge(meta, on="participantId", how="left")

        cc = wide.dropna(subset=["chatgpt","gemini"]).copy()
        cc["diff_gpt_minus_gem"] = cc["chatgpt"] - cc["gemini"]

        # Save Prism tables
        prism_wide_path = os.path.join(out, "31v11-0_participant_wide_aiacc_chatgpt_vs_gemini_mobile.csv")
        wide.to_csv(prism_wide_path, index=False, encoding="utf-8-sig")

        prism_long = cc.melt(
            id_vars=["participantId","age","sex","overallAccuracy","avgRT_overall"],
            value_vars=["chatgpt","gemini"],
            var_name="gen_model",
            value_name="ai_accuracy"
        )
        prism_long_path = os.path.join(out, "31v11-0_participant_long_aiacc_chatgpt_vs_gemini_mobile.csv")
        prism_long.to_csv(prism_long_path, index=False, encoding="utf-8-sig")

        print(f"✅ Saved Prism wide: {prism_wide_path}")
        print(f"✅ Saved Prism long (paired only): {prism_long_path}")
        print(f"✅ N paired (complete-case): {len(cc)}")

        # Paired stats
        t_stat, p_t = ttest_rel(cc["chatgpt"], cc["gemini"])
        try:
            w_stat, p_w = wilcoxon(cc["chatgpt"], cc["gemini"])
        except Exception:
            w_stat, p_w = np.nan, np.nan

        dz = paired_effect_dz(cc["diff_gpt_minus_gem"])

        summary_lines = []
        summary_lines.append("===============================================================================")
        summary_lines.append("(31v1.1) Paired comparison (AI accuracy): ChatGPT vs Gemini [MOBILE]")
        summary_lines.append("===============================================================================")
        summary_lines.append(f"Generator column used: {gen_col}")
        summary_lines.append(f"AI classification coverage: {stats['coverage']*100:.2f}% (AI trials)")
        summary_lines.append(f"N paired (complete-case) = {len(cc)}")
        summary_lines.append(f"Mean AI accuracy ChatGPT = {cc['chatgpt'].mean():.3f}%")
        summary_lines.append(f"Mean AI accuracy Gemini  = {cc['gemini'].mean():.3f}%")
        summary_lines.append(f"Mean diff (GPT - Gem)    = {cc['diff_gpt_minus_gem'].mean():.3f} pp")
        summary_lines.append("")
        summary_lines.append(f"Paired t-test: t={t_stat:.3f}, p={p_t:.6g}")
        summary_lines.append(f"Wilcoxon:      W={w_stat}, p={p_w:.6g}")
        summary_lines.append(f"Cohen's dz     = {dz:.3f}")

        rep_path = os.path.join(out, "31v11-1_paired_tests_aiacc_mobile.txt")
        save_text(rep_path, "\n".join(summary_lines))
        print("\n".join(summary_lines))
        print(f"✅ Saved report: {rep_path}")

        # Plots
        config.apply_korean_plot_style()

        # scatter gpt vs gem
        fig, ax = plt.subplots(figsize=(7,7))
        ax.scatter(cc["gemini"], cc["chatgpt"], alpha=SCATTER_ALPHA)
        ax.plot([0,100],[0,100], linestyle="--")
        ax.set_xlim(*YLIM); ax.set_ylim(*YLIM)
        ax.set_xlabel("Gemini AI accuracy (%)")
        ax.set_ylabel("ChatGPT AI accuracy (%)")
        ax.set_title("(31v1.1) AI Accuracy: ChatGPT vs Gemini (paired)")
        out_png = os.path.join(out, "31v11-2_scatter_aiacc_gpt_vs_gem_mobile.png")
        out_svg = os.path.join(out, "31v11-2_scatter_aiacc_gpt_vs_gem_mobile.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # mean±SEM bar
        summ = pd.DataFrame({
            "gen_model":["chatgpt","gemini"],
            "mean":[cc["chatgpt"].mean(), cc["gemini"].mean()],
            "sem":[sem(cc["chatgpt"]), sem(cc["gemini"])],
            "n":[len(cc), len(cc)]
        })
        summ_path = os.path.join(out, "31v11-2_summary_mean_sem_aiacc_mobile.csv")
        summ.to_csv(summ_path, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(6,6))
        ax.bar(summ["gen_model"], summ["mean"], yerr=summ["sem"])
        ax.set_ylim(*YLIM)
        ax.set_ylabel("AI accuracy (%)")
        ax.set_title(f"(31v1.1) Mean AI accuracy (paired N={len(cc)})\npaired t p={p_t:.3g}")
        out_png2 = os.path.join(out, "31v11-2_bar_mean_sem_aiacc_mobile.png")
        out_svg2 = os.path.join(out, "31v11-2_bar_mean_sem_aiacc_mobile.svg")
        fig.savefig(out_png2, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg2, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print(f"✅ Saved plots: {out_png} / {out_png2}")
        print(f"✅ Saved summary csv: {summ_path}")

        # Optional sensitivity: AI-trial GLM with clustered SE
        ai2 = ai.merge(meta, on="participantId", how="left")
        ai2 = ai2.dropna(subset=["isCorrect","gen_model","age","sex"]).copy()
        ai2["isCorrect_num"] = ai2["isCorrect"].astype(int)
        ai2["sex"] = ai2["sex"].astype(str)

        if len(ai2) >= 200:
            glm = smf.glm(
                "isCorrect_num ~ C(gen_model) + age + C(sex)",
                data=ai2,
                family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": ai2["participantId"]})

            glm_txt = os.path.join(out, "31v11-3_glm_clustered_ai_trials_mobile.txt")
            save_text(glm_txt, glm.summary().as_text())
            print(f"✅ Saved AI-trial GLM (clustered SE) report: {glm_txt}")

        print("\n==================== (31 v1.1 | MOBILE ONLY) DONE ====================\n")

    if __name__ == "__main__":
        run_section31_mobile()


def main():
    _run_cell_137()


if __name__ == "__main__":
    main()
