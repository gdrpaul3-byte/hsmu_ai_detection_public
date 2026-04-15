"""Figure 7 assembly: age-by-sex reaction time figure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_037():
    # ==============================================================================
    # Figure 7 (RT): Age-bin × Sex (overall RT) + MixedLM EMM interaction (Correctness × Kind)
    # - Main: Mobile
    # - Supp: PC (interaction only)
    # Saves to:
    #   plots/run_20260119_192624/07_rt_summary/
    # ==============================================================================
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import sem
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # -----------------------------
    # Paths / config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "07_rt_summary"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    MOBILE_MAIN = config.MOBILE_AGE_FILTERED
    PC_MAIN     = config.WEB_AGE_FILTERED

    # From your 26 v1.2 outputs
    RT26_DIR = BASE_DIR / "outputs_26_verification_cost_v1_2"
    MOBILE_TRIAL = RT26_DIR / "mobile" / "26v12-0_trial_level_table.csv"
    MOBILE_COEF  = RT26_DIR / "mobile" / "26v12-1_mixedlm_coeffs.csv"
    PC_TRIAL     = RT26_DIR / "web" / "26v12-0_trial_level_table.csv"
    PC_COEF      = RT26_DIR / "web" / "26v12-1_mixedlm_coeffs.csv"

    for p in [MOBILE_MAIN, PC_MAIN, MOBILE_TRIAL, MOBILE_COEF, PC_TRIAL, PC_COEF]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Big fonts (paper)
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.75)

    SEX_COL = {"male": "#4285F4", "female": "#DB4437"}

    AGE_BINS   = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    def normalize_sex(series):
        s = series.astype(str).str.lower().str.strip()
        s = s.replace({"m":"male","f":"female"})
        s = s.where(s.isin(["male","female"]))
        return s

    def add_age_group(df):
        df = df.copy()
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        return df

    def fmt_p(p):
        if p is None or (isinstance(p,float) and np.isnan(p)): 
            return "NA"
        p = float(p)
        return "p < .001" if p < 0.001 else f"p = {p:.3f}"

    # ==============================================================================
    # Fig7A (Mobile): Overall RT by Age-bin × Sex (mean±SEM) + ANOVA p box
    # ==============================================================================
    def fig7A_overall_rt_mobile():
        df = pd.read_csv(MOBILE_MAIN, encoding="utf-8-sig")
        df = add_age_group(df)
        df["sex"] = normalize_sex(df["gender"] if "gender" in df.columns else df["sex"])
        df = df[df["sex"].isin(["male","female"])].copy()

        # your cohort file uses avgRT as overall RT
        if "avgRT" not in df.columns:
            raise KeyError("Mobile main file missing avgRT (overall RT).")

        df["avgRT"] = pd.to_numeric(df["avgRT"], errors="coerce")
        d = df.dropna(subset=["age_group","sex","avgRT"]).copy()

        # Two-way ANOVA (Type II) like your section 23
        # (use statsmodels OLS on participant-level)
        model = smf.ols("avgRT ~ C(age_group) + C(sex) + C(age_group):C(sex)", data=d).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        p_age = float(anova.loc["C(age_group)", "PR(>F)"])
        p_sex = float(anova.loc["C(sex)", "PR(>F)"])
        p_int = float(anova.loc["C(age_group):C(sex)", "PR(>F)"])

        # summary for plotting
        summ = (d.groupby(["age_group","sex"], observed=True)["avgRT"]
                .agg(mean="mean", sem=lambda x: sem(x, nan_policy="omit"), n="count")
                .reset_index())
        summ["age_group"] = pd.Categorical(summ["age_group"], categories=AGE_LABELS, ordered=True)
        summ = summ.sort_values(["age_group","sex"])

        # plot
        fig, ax = plt.subplots(figsize=(16, 10))
        x = np.arange(len(AGE_LABELS))

        for sex in ["male","female"]:
            ss = summ[summ["sex"]==sex].set_index("age_group").reindex(AGE_LABELS).reset_index()
            ax.errorbar(
                x, ss["mean"].values, yerr=ss["sem"].values,
                marker="o", linewidth=3, capsize=8,
                color=SEX_COL[sex], label=sex.title()
            )

        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS, fontweight="bold")
        ax.set_ylabel("Overall mean RT (ms)", fontweight="bold")
        ax.set_title("Figure 7A. Overall RT by age group and sex (Mobile; mean ± SEM)", fontweight="bold", pad=14)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

        # p-box
        box = (f"Two-way ANOVA (Type II)\n"
               f"Age: {fmt_p(p_age)}\n"
               f"Sex: {fmt_p(p_sex)}\n"
               f"Age×Sex: {fmt_p(p_int)}")
        ax.text(
            0.98, 0.02, box,
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            fontweight="bold"
        )

        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        out_png = OUT_DIR / "fig7A_overallRT_agebin_sex_mobile.png"
        out_svg = OUT_DIR / "fig7A_overallRT_agebin_sex_mobile.svg"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

        # save anova table
        anova_out = OUT_DIR / "fig7A_overallRT_anova_mobile.csv"
        anova.round(8).to_csv(anova_out, encoding="utf-8-sig")
        print("✅ saved:", out_png)
        print("✅ anova:", anova_out)

    # ==============================================================================
    # Fig7B (Mobile main): MixedLM EMM interaction plot (Correctness × Kind)
    # - uses your trial-level table (logRT model)
    # - plots model-predicted means (fixed effects only), holding age at mean and sex proportion-weighted
    # ==============================================================================
    def fig7B_mixedlm_emm(trial_csv: Path, coef_csv: Path, cohort_label: str, out_prefix: str):
        d = pd.read_csv(trial_csv, encoding="utf-8-sig")

        # Expect columns from your 26v12-0_trial_level_table.csv
        needed = {"participantId","rt_ms","logRT","Correctness","Kind","age","sex"}
        if not needed.issubset(set(d.columns)):
            raise KeyError(f"Trial table missing columns. Need {needed}, got {set(d.columns)}")

        d["age"] = pd.to_numeric(d["age"], errors="coerce")
        d["sex"] = normalize_sex(d["sex"])
        d = d.dropna(subset=["logRT","Correctness","Kind","age"]).copy()
        d = d[d["Correctness"].isin(["Correct","Incorrect"]) & d["Kind"].isin(["Real","AI"])].copy()

        # Fit mixed model again (to guarantee coefficient names match)
        # random intercept by participant
        formula = "logRT ~ C(Correctness) * C(Kind) + age + C(sex)"
        md = smf.mixedlm(formula, d, groups=d["participantId"])
        m = md.fit(method="lbfgs", reml=False)

        # Get interaction p from coeff table (your saved one is fine too, but we'll compute fresh)
        coef = pd.DataFrame({
            "term": m.params.index,
            "coef": m.params.values,
            "p": m.pvalues.values
        })
        inter_term = "C(Correctness)[T.Incorrect]:C(Kind)[T.Real]"
        p_inter = float(coef.loc[coef["term"]==inter_term, "p"].iloc[0]) if (coef["term"]==inter_term).any() else np.nan

        # Build EMM grid
        age_mean = float(d["age"].mean())
        # use sex distribution weights (if sex missing, treat as all male)
        sex_counts = d["sex"].value_counts(dropna=True)
        w_m = float(sex_counts.get("male", 0)) / float(sex_counts.sum()) if sex_counts.sum() > 0 else 1.0
        w_f = float(sex_counts.get("female", 0)) / float(sex_counts.sum()) if sex_counts.sum() > 0 else 0.0

        grid = []
        for corr in ["Correct","Incorrect"]:
            for kind in ["Real","AI"]:
                # predict for male and female then weight-average (sex-controlled marginal mean)
                for sex, w in [("male", w_m), ("female", w_f)]:
                    if w == 0:
                        continue
                    grid.append({"Correctness": corr, "Kind": kind, "age": age_mean, "sex": sex, "w": w})
        grid = pd.DataFrame(grid)

        # Predict fixed-effects only: use m.predict(exog=...) works with MixedLMResults and formula
        # We'll use the same dataframe columns used in formula.
        grid["pred_logRT"] = m.predict(grid)
        grid["pred_RT_ms"] = np.exp(grid["pred_logRT"])

        # weighted marginal mean over sex
        emm = (grid.groupby(["Correctness","Kind"])
               .apply(lambda g: np.sum(g["pred_RT_ms"] * g["w"]))
               .reset_index(name="emm_rt_ms"))

        # Plot interaction: x=Correctness, line=Kind
        fig, ax = plt.subplots(figsize=(14, 9))
        x_order = ["Correct","Incorrect"]
        x = np.arange(len(x_order))
        for kind, marker in [("Real","o"), ("AI","s")]:
            sub = emm[emm["Kind"]==kind].set_index("Correctness").reindex(x_order).reset_index()
            ax.plot(x, sub["emm_rt_ms"].values, marker=marker, linewidth=4, markersize=10, label=kind)

        ax.set_xticks(x)
        ax.set_xticklabels(x_order, fontweight="bold")
        ax.set_ylabel("Model-predicted RT (ms)", fontweight="bold")
        ax.set_title(f"Figure 7B. Condition-specific RT (MixedLM EMM) [{cohort_label}]", fontweight="bold", pad=14)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.legend(title="Image kind", frameon=True, loc="upper left")

        ax.text(
            0.98, 0.02,
            f"Correctness×Kind interaction:\n{fmt_p(p_inter)}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            fontweight="bold"
        )

        plt.tight_layout()
        out_png = OUT_DIR / f"{out_prefix}.png"
        out_svg = OUT_DIR / f"{out_prefix}.svg"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

        # save EMM table + coef table
        emm_out = OUT_DIR / f"{out_prefix}_emm_table.csv"
        emm.to_csv(emm_out, index=False, encoding="utf-8-sig")

        coef_out = OUT_DIR / f"{out_prefix}_mixedlm_coeffs.csv"
        coef.to_csv(coef_out, index=False, encoding="utf-8-sig")

        print("✅ saved:", out_png)
        print("✅ emm:", emm_out)

    # -----------------------------
    # Run
    # -----------------------------
    fig7A_overall_rt_mobile()

    # Fig7B mobile main
    fig7B_mixedlm_emm(MOBILE_TRIAL, MOBILE_COEF, "MOBILE", "fig7B_mixedlm_emm_interaction_mobile")

    # Supp S7B PC
    fig7B_mixedlm_emm(PC_TRIAL, PC_COEF, "PC (Supplement)", "figS7B_mixedlm_emm_interaction_pc")

    print("\n✅ Figure 7 outputs saved to:", OUT_DIR)


def main():
    _run_cell_037()


if __name__ == "__main__":
    main()
