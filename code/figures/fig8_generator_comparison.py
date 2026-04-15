"""Figure 8 assembly: generator comparison panels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_039():
    # ==============================================================================
    # Figure 8 (UPDATED): Generator comparison
    # - No in-panel stat boxes (A/B/C clean)
    # - Add slope comparison panel next to scatter (D)
    # - Save stats to CSV (no overlay boxes)
    # Mobile main + PC supplementary
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import ttest_rel, wilcoxon
    import statsmodels.formula.api as smf
    from scipy.stats import linregress


    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "08_generator_comparison"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "pc":     config.WEB_AGE_FILTERED,
    }

    # ===== generator labels used in this study (stimuli created July 2025) =====
    GEN_LABEL_CGPT = "ChatGPT-4o (native images; stimuli July 2025)"
    GEN_LABEL_GEM  = "Gemini (Imagen 3; stimuli July 2025)"

    COL_CGPT = "#DB4437"
    COL_GEM  = "#4285F4"

    # Big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.65)

    def to_percent_series(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        finite = x.dropna()
        if finite.empty: return x
        mx = float(finite.max())
        return x * 100.0 if mx <= 1.5 else x

    def paired_test(x, y):
        x = pd.to_numeric(pd.Series(x), errors="coerce")
        y = pd.to_numeric(pd.Series(y), errors="coerce")
        m = x.notna() & y.notna()
        x = x[m].values; y = y[m].values
        n = len(x)
        if n < 10:
            return {"N": n, "t": np.nan, "p_t": np.nan, "w": np.nan, "p_w": np.nan}
        pt = ttest_rel(x, y)
        try:
            pw = wilcoxon(x, y, zero_method="wilcox")
            w_stat, w_p = float(pw.statistic), float(pw.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        return {"N": n, "t": float(pt.statistic), "p_t": float(pt.pvalue), "w": w_stat, "p_w": w_p}

    def fmt_p(p):
        if p is None or (isinstance(p,float) and np.isnan(p)): return "NA"
        p = float(p)
        return "p < .001" if p < 0.001 else f"p = {p:.3f}"

    def fmt_p_console(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "NA"
        p = float(p)
        if p < 1e-3:
            return f"{p:.2e}"
        return f"{p:.6f}"

    def print_fig8_stats(cohort_label, statA, statB, p_inter,
                         mean_acc_cgpt, mean_acc_gem, se_acc_cgpt, se_acc_gem,
                         mean_rt_cgpt, mean_rt_gem, se_rt_cgpt, se_rt_gem,
                         slope_cgpt, slope_gem, N_scatter):
        print("\n" + "="*78)
        print(f"Figure 8 stats [{cohort_label}]")
        print("="*78)

        print("\n(A) Accuracy (ChatGPT-4o vs Gemini)")
        print(f"  N paired: {statA.get('N', 'NA')}")
        print(f"  Mean±SEM (%): ChatGPT-4o={mean_acc_cgpt:.2f}±{se_acc_cgpt:.2f} | Gemini={mean_acc_gem:.2f}±{se_acc_gem:.2f}")
        print(f"  Paired t-test: t={statA.get('t', np.nan):.3f}, p={fmt_p_console(statA.get('p_t', np.nan))}")
        print(f"  Wilcoxon: W={statA.get('w', np.nan)}, p={fmt_p_console(statA.get('p_w', np.nan))}")

        print("\n(B) Mean RT (seconds) (ChatGPT-4o vs Gemini)")
        print(f"  N paired: {statB.get('N', 'NA')}")
        print(f"  Mean±SEM (s): ChatGPT-4o={mean_rt_cgpt:.3f}±{se_rt_cgpt:.3f} | Gemini={mean_rt_gem:.3f}±{se_rt_gem:.3f}")
        print(f"  Paired t-test: t={statB.get('t', np.nan):.3f}, p={fmt_p_console(statB.get('p_t', np.nan))}")
        print(f"  Wilcoxon: W={statB.get('w', np.nan)}, p={fmt_p_console(statB.get('p_w', np.nan))}")

        print("\n(C–D) Age–accuracy trends by generator")
        print(f"  N scatter complete: {N_scatter}")
        print(f"  Simple slopes (% points/year): ChatGPT-4o={slope_cgpt:.3f} | Gemini={slope_gem:.3f}")
        print(f"  Slope-difference test (age × generator interaction, HC3): p={fmt_p_console(p_inter)}")
        print("="*78 + "\n")



    def run_one(cohort_key="mobile", is_supp=False):
        df = pd.read_csv(FILES[cohort_key], encoding="utf-8-sig")

        req = ["accuracy_chatgpt","accuracy_gemini","avg_rt_chatgpt","avg_rt_gemini","age"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise KeyError(f"[{cohort_key}] Missing columns: {miss}")

        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        # accuracy to %
        df["acc_cgpt"] = to_percent_series(df["accuracy_chatgpt"])
        df["acc_gem"]  = to_percent_series(df["accuracy_gemini"])

        # RT seconds
        df["rt_cgpt_s"] = pd.to_numeric(df["avg_rt_chatgpt"], errors="coerce") / 1000.0
        df["rt_gem_s"]  = pd.to_numeric(df["avg_rt_gemini"], errors="coerce") / 1000.0

        # optional trial-count filter if exists
        if "n_chatgpt" in df.columns:
            df = df[df["n_chatgpt"].fillna(0) >= 3].copy()
        if "n_gemini" in df.columns:
            df = df[df["n_gemini"].fillna(0) >= 3].copy()

        tag = "S8" if is_supp else "8"
        cohort_label = "PC (Supplement)" if cohort_key=="pc" else "Mobile"

        # -------------------- Layout: A, B, C, D --------------------
        fig = plt.figure(figsize=(30, 12))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.35, 0.75])
        axA = fig.add_subplot(gs[0,0])
        axB = fig.add_subplot(gs[0,1])
        axC = fig.add_subplot(gs[0,2])
        axD = fig.add_subplot(gs[0,3])

        # -------------------- A: Accuracy (bar+SEM only) --------------------
        dA = df[["acc_cgpt","acc_gem"]].dropna()
        x1 = dA["acc_cgpt"].values
        x2 = dA["acc_gem"].values
        m1, m2 = np.mean(x1), np.mean(x2)
        se1 = np.std(x1, ddof=1)/np.sqrt(len(x1))
        se2 = np.std(x2, ddof=1)/np.sqrt(len(x2))

        axA.bar([0,1], [m1,m2], yerr=[se1,se2], capsize=8, width=0.55,
                color=[COL_CGPT,COL_GEM], edgecolor="black", alpha=0.85)
        axA.set_xticks([0,1])
        axA.set_xticklabels(["ChatGPT-4o","Gemini"], fontweight="bold")
        axA.set_ylim(0,100)
        axA.set_ylabel("Accuracy (%)", fontweight="bold")
        axA.set_title(f"Figure {tag}A. Accuracy by generator\n({cohort_label})", fontweight="bold", pad=12)
        axA.grid(True, axis="y", linestyle=":", alpha=0.35)

        # (no overlay box) — save stats
        statA = paired_test(x1, x2)

        # generator label note (small)
        axA.text(
            0.5, -0.22,
            f"{GEN_LABEL_CGPT}\n{GEN_LABEL_GEM}",
            transform=axA.transAxes, ha="center", va="top",
            fontsize=BIG*0.55
        )

        # -------------------- B: RT (bar+SEM only) --------------------
        dB = df[["rt_cgpt_s","rt_gem_s"]].dropna()
        r1 = dB["rt_cgpt_s"].values
        r2 = dB["rt_gem_s"].values
        rm1, rm2 = np.mean(r1), np.mean(r2)
        rse1 = np.std(r1, ddof=1)/np.sqrt(len(r1))
        rse2 = np.std(r2, ddof=1)/np.sqrt(len(r2))

        axB.bar([0,1], [rm1,rm2], yerr=[rse1,rse2], capsize=8, width=0.55,
                color=[COL_CGPT,COL_GEM], edgecolor="black", alpha=0.85)
        axB.set_xticks([0,1])
        axB.set_xticklabels(["ChatGPT-4o","Gemini"], fontweight="bold")
        axB.set_ylabel("Mean RT (s)", fontweight="bold")
        axB.set_title(f"Figure {tag}B. RT by generator\n({cohort_label})", fontweight="bold", pad=12)
        axB.grid(True, axis="y", linestyle=":", alpha=0.35)

        statB = paired_test(r1, r2)

        # -------------------- C: Age–accuracy scatter + regression lines --------------------
        dC = df[["age","acc_cgpt","acc_gem"]].dropna().copy()

        # scatter (two sets)
        axC.scatter(dC["age"], dC["acc_cgpt"], s=18, alpha=0.18, color=COL_CGPT, edgecolors="none", label="ChatGPT-4o")
        axC.scatter(dC["age"], dC["acc_gem"],  s=18, alpha=0.18, color=COL_GEM,  edgecolors="none", label="Gemini")

        # regression lines
        for gen, ycol, color in [("ChatGPT-4o","acc_cgpt",COL_CGPT), ("Gemini","acc_gem",COL_GEM)]:
            sub = dC[["age", ycol]].dropna()
            if len(sub) >= 30:
                b1, b0 = np.polyfit(sub["age"].values, sub[ycol].values, 1)
                xx = np.linspace(sub["age"].min(), sub["age"].max(), 200)
                yy = b1*xx + b0
                axC.plot(xx, yy, linewidth=4, color=color)

        axC.set_ylim(0,100)
        axC.set_xlabel("Age", fontweight="bold")
        axC.set_ylabel("Accuracy (%)", fontweight="bold")
        axC.set_title(f"Figure {tag}C. Age–accuracy by generator\n({cohort_label})", fontweight="bold", pad=12)
        axC.grid(True, linestyle=":", alpha=0.35)
        axC.legend(frameon=True, loc="upper right")

        # -------------------- D: Slope comparison + slope-difference test --------------------
        long = pd.concat([
            dC[["age","acc_cgpt"]].rename(columns={"acc_cgpt":"accuracy"}).assign(generator="ChatGPT-4o"),
            dC[["age","acc_gem"]].rename(columns={"acc_gem":"accuracy"}).assign(generator="Gemini"),
        ], ignore_index=True)

        # interaction regression: accuracy ~ age * generator
        model = smf.ols("accuracy ~ age * C(generator)", data=long).fit(cov_type="HC3")
        inter_terms = [t for t in model.params.index if "age:C(generator)" in t]
        p_inter = float(model.pvalues[inter_terms[0]]) if inter_terms else np.nan

        # extract slopes for each generator from model
        # baseline slope = coef(age)
        base_slope = float(model.params.get("age", np.nan))
        # if baseline is ChatGPT-4o, Gemini slope = age + age:C(generator)[T.Gemini]
        # statsmodels chooses baseline alphabetically; ensure we compute by prediction:
        # We'll compute slope by fitting separate simple lines (for display), and keep p_inter from interaction.
        def simple_slope(ycol):
            sub = dC[["age", ycol]].dropna()
            b1, _ = np.polyfit(sub["age"].values, sub[ycol].values, 1)
            return float(b1)

        slope_cgpt = simple_slope("acc_cgpt")
        slope_gem  = simple_slope("acc_gem")

        axD.bar([0,1], [slope_cgpt, slope_gem], width=0.55,
                color=[COL_CGPT, COL_GEM], edgecolor="black", alpha=0.85)
        axD.axhline(0, linestyle="--", linewidth=2, color="black", alpha=0.7)
        axD.set_xticks([0,1])
        axD.set_xticklabels(["ChatGPT-4o","Gemini"], rotation=90, ha="center", fontweight="bold")
        axD.set_ylabel("Slope\n(% points/year)", fontweight="bold")
        axD.set_title(f"Figure {tag}D.\nSlope comparison", fontweight="bold", pad=12)
        axD.grid(True, axis="y", linestyle=":", alpha=0.35)

        # show slope-difference test p here (only here)
        axD.text(
            0.5, 0.02,
            f"age×generator:\n{fmt_p(p_inter)}",
            transform=axD.transAxes,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            fontweight="bold"
        )

        fig.tight_layout()

        out_png = OUT_DIR / (f"fig8_{cohort_key}.png" if not is_supp else f"figS8_{cohort_key}.png")
        out_svg = OUT_DIR / (f"fig8_{cohort_key}.svg" if not is_supp else f"figS8_{cohort_key}.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

        # save stats tables (no overlay)
        tests_df = pd.DataFrame([
            {"cohort": cohort_key, "metric":"accuracy", **statA},
            {"cohort": cohort_key, "metric":"rt_seconds", **statB},
            {"cohort": cohort_key, "metric":"age×generator_slope_test", "N": int(len(dC)), "p_interaction": p_inter},
        ])
        stats_out = OUT_DIR / (f"fig8_stats_{cohort_key}.csv" if not is_supp else f"figS8_stats_{cohort_key}.csv")
        tests_df.to_csv(stats_out, index=False, encoding="utf-8-sig")

        coef_out = OUT_DIR / (f"fig8D_interaction_coeffs_{cohort_key}.csv" if not is_supp else f"figS8D_interaction_coeffs_{cohort_key}.csv")
        pd.DataFrame({
            "term": model.params.index,
            "coef": model.params.values,
            "se_hc3": model.bse.values,
            "p": model.pvalues.values
        }).to_csv(coef_out, index=False, encoding="utf-8-sig")


            # -------------------- Console summary (no figure overlay) --------------------
        mean_acc_cgpt = float(np.mean(x1)); mean_acc_gem = float(np.mean(x2))
        se_acc_cgpt   = float(np.std(x1, ddof=1)/np.sqrt(len(x1)))
        se_acc_gem    = float(np.std(x2, ddof=1)/np.sqrt(len(x2)))

        mean_rt_cgpt  = float(np.mean(r1)); mean_rt_gem = float(np.mean(r2))
        se_rt_cgpt    = float(np.std(r1, ddof=1)/np.sqrt(len(r1)))
        se_rt_gem     = float(np.std(r2, ddof=1)/np.sqrt(len(r2)))

        print_fig8_stats(
            cohort_label=cohort_label,
            statA=statA,
            statB=statB,
            p_inter=p_inter,
            mean_acc_cgpt=mean_acc_cgpt,
            mean_acc_gem=mean_acc_gem,
            se_acc_cgpt=se_acc_cgpt,
            se_acc_gem=se_acc_gem,
            mean_rt_cgpt=mean_rt_cgpt,
            mean_rt_gem=mean_rt_gem,
            se_rt_cgpt=se_rt_cgpt,
            se_rt_gem=se_rt_gem,
            slope_cgpt=slope_cgpt,
            slope_gem=slope_gem,
            N_scatter=len(dC)
        )


        print("✅ saved:", out_png)
        print("✅ stats:", stats_out)
        print("✅ interaction coeffs:", coef_out)

    # Run
    run_one("mobile", is_supp=False)
    run_one("pc", is_supp=True)

    print("✅ Figure 8 outputs saved to:", OUT_DIR)


def _run_cell_041():
    # ==============================================================================
    # Figure 8 (UPDATED): Generator comparison
    # - No in-panel stat boxes (A/B/C clean)
    # - Add slope comparison panel next to scatter (D) with slope ± SE (OLS)
    # - Slope-diff test uses age×generator interaction (HC3)
    # - Save stats to CSV (no overlay boxes)
    # Mobile main + PC supplementary
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import ttest_rel, wilcoxon, linregress
    import statsmodels.formula.api as smf

    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "08_generator_comparison"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "pc":     config.WEB_AGE_FILTERED,
    }

    # ===== generator labels used in this study (stimuli created July 2025) =====
    GEN_LABEL_CGPT = "ChatGPT-4o (native images; stimuli July 2025)"
    GEN_LABEL_GEM  = "Gemini (Imagen 3; stimuli July 2025)"

    COL_CGPT = "#DB4437"
    COL_GEM  = "#4285F4"

    # Big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.65)

    def to_percent_series(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        finite = x.dropna()
        if finite.empty:
            return x
        mx = float(finite.max())
        return x * 100.0 if mx <= 1.5 else x

    def paired_test(x, y):
        x = pd.to_numeric(pd.Series(x), errors="coerce")
        y = pd.to_numeric(pd.Series(y), errors="coerce")
        m = x.notna() & y.notna()
        x = x[m].values
        y = y[m].values
        n = len(x)
        if n < 10:
            return {"N": n, "t": np.nan, "p_t": np.nan, "w": np.nan, "p_w": np.nan}
        pt = ttest_rel(x, y)
        try:
            pw = wilcoxon(x, y, zero_method="wilcox")
            w_stat, w_p = float(pw.statistic), float(pw.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        return {"N": n, "t": float(pt.statistic), "p_t": float(pt.pvalue), "w": w_stat, "p_w": w_p}

    def fmt_p(p):
        if p is None or (isinstance(p,float) and np.isnan(p)):
            return "NA"
        p = float(p)
        return "p < .001" if p < 0.001 else f"p = {p:.3f}"

    def fmt_p_console(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "NA"
        p = float(p)
        if p < 1e-3:
            return f"{p:.2e}"
        return f"{p:.6f}"

    def slope_and_se(age, y):
        sub = pd.DataFrame({"age": age, "y": y}).dropna()
        if len(sub) < 3:
            return np.nan, np.nan, np.nan
        lr = linregress(sub["age"].values, sub["y"].values)
        return float(lr.slope), float(lr.stderr), float(lr.intercept)

    def print_fig8_stats(cohort_label, statA, statB, p_inter,
                         mean_acc_cgpt, mean_acc_gem, se_acc_cgpt, se_acc_gem,
                         mean_rt_cgpt, mean_rt_gem, se_rt_cgpt, se_rt_gem,
                         slope_cgpt, se_slope_cgpt, slope_gem, se_slope_gem,
                         N_scatter):
        print("\n" + "="*78)
        print(f"Figure 8 stats [{cohort_label}]")
        print("="*78)

        print("\n(A) Accuracy (ChatGPT-4o vs Gemini)")
        print(f"  N paired: {statA.get('N', 'NA')}")
        print(f"  Mean±SEM (%): ChatGPT-4o={mean_acc_cgpt:.2f}±{se_acc_cgpt:.2f} | Gemini={mean_acc_gem:.2f}±{se_acc_gem:.2f}")
        print(f"  Paired t-test: t={statA.get('t', np.nan):.3f}, p={fmt_p_console(statA.get('p_t', np.nan))}")
        print(f"  Wilcoxon: W={statA.get('w', np.nan)}, p={fmt_p_console(statA.get('p_w', np.nan))}")

        print("\n(B) Mean RT (seconds) (ChatGPT-4o vs Gemini)")
        print(f"  N paired: {statB.get('N', 'NA')}")
        print(f"  Mean±SEM (s): ChatGPT-4o={mean_rt_cgpt:.3f}±{se_rt_cgpt:.3f} | Gemini={mean_rt_gem:.3f}±{se_rt_gem:.3f}")
        print(f"  Paired t-test: t={statB.get('t', np.nan):.3f}, p={fmt_p_console(statB.get('p_t', np.nan))}")
        print(f"  Wilcoxon: W={statB.get('w', np.nan)}, p={fmt_p_console(statB.get('p_w', np.nan))}")

        print("\n(C–D) Age–accuracy trends by generator")
        print(f"  N scatter complete: {N_scatter}")
        print(f"  Slopes (OLS ± SE): ChatGPT-4o={slope_cgpt:.3f}±{se_slope_cgpt:.3f} | Gemini={slope_gem:.3f}±{se_slope_gem:.3f}")
        print(f"  Slope-difference test (age × generator interaction, HC3): p={fmt_p_console(p_inter)}")
        print("="*78 + "\n")

    def run_one(cohort_key="mobile", is_supp=False):
        df = pd.read_csv(FILES[cohort_key], encoding="utf-8-sig")

        req = ["accuracy_chatgpt","accuracy_gemini","avg_rt_chatgpt","avg_rt_gemini","age"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise KeyError(f"[{cohort_key}] Missing columns: {miss}")

        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        # accuracy to %
        df["acc_cgpt"] = to_percent_series(df["accuracy_chatgpt"])
        df["acc_gem"]  = to_percent_series(df["accuracy_gemini"])

        # RT seconds
        df["rt_cgpt_s"] = pd.to_numeric(df["avg_rt_chatgpt"], errors="coerce") / 1000.0
        df["rt_gem_s"]  = pd.to_numeric(df["avg_rt_gemini"], errors="coerce") / 1000.0

        # optional trial-count filter if exists
        if "n_chatgpt" in df.columns:
            df = df[df["n_chatgpt"].fillna(0) >= 3].copy()
        if "n_gemini" in df.columns:
            df = df[df["n_gemini"].fillna(0) >= 3].copy()

        tag = "S8" if is_supp else "8"
        cohort_label = "PC (Supplement)" if cohort_key=="pc" else "Mobile"

        # -------------------- Layout: A, B, C, D --------------------
        fig = plt.figure(figsize=(30, 12))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.35, 0.75])
        axA = fig.add_subplot(gs[0,0])
        axB = fig.add_subplot(gs[0,1])
        axC = fig.add_subplot(gs[0,2])
        axD = fig.add_subplot(gs[0,3])

        # -------------------- A: Accuracy (bar+SEM only) --------------------
        dA = df[["acc_cgpt","acc_gem"]].dropna()
        x1 = dA["acc_cgpt"].values
        x2 = dA["acc_gem"].values
        m1, m2 = np.mean(x1), np.mean(x2)
        se1 = np.std(x1, ddof=1)/np.sqrt(len(x1))
        se2 = np.std(x2, ddof=1)/np.sqrt(len(x2))

        axA.bar([0,1], [m1,m2], yerr=[se1,se2], capsize=8, width=0.55,
                color=[COL_CGPT,COL_GEM], edgecolor="black", alpha=0.85)
        axA.set_xticks([0,1])
        axA.set_xticklabels(["ChatGPT-4o","Gemini"], fontweight="bold")
        axA.set_ylim(0,100)
        axA.set_ylabel("Accuracy (%)", fontweight="bold")
        axA.set_title(f"Figure {tag}A. Accuracy by generator\n({cohort_label})", fontweight="bold", pad=12)
        axA.grid(True, axis="y", linestyle=":", alpha=0.35)

        statA = paired_test(x1, x2)

        axA.text(
            0.5, -0.22,
            f"{GEN_LABEL_CGPT}\n{GEN_LABEL_GEM}",
            transform=axA.transAxes, ha="center", va="top",
            fontsize=BIG*0.55
        )

        # -------------------- B: RT (bar+SEM only) --------------------
        dB = df[["rt_cgpt_s","rt_gem_s"]].dropna()
        r1 = dB["rt_cgpt_s"].values
        r2 = dB["rt_gem_s"].values
        rm1, rm2 = np.mean(r1), np.mean(r2)
        rse1 = np.std(r1, ddof=1)/np.sqrt(len(r1))
        rse2 = np.std(r2, ddof=1)/np.sqrt(len(r2))

        axB.bar([0,1], [rm1,rm2], yerr=[rse1,rse2], capsize=8, width=0.55,
                color=[COL_CGPT,COL_GEM], edgecolor="black", alpha=0.85)
        axB.set_xticks([0,1])
        axB.set_xticklabels(["ChatGPT-4o","Gemini"], fontweight="bold")
        axB.set_ylabel("Mean RT (s)", fontweight="bold")
        axB.set_title(f"Figure {tag}B. RT by generator\n({cohort_label})", fontweight="bold", pad=12)
        axB.grid(True, axis="y", linestyle=":", alpha=0.35)

        statB = paired_test(r1, r2)

        # -------------------- C: Age–accuracy scatter + OLS regression lines --------------------
        dC = df[["age","acc_cgpt","acc_gem"]].dropna().copy()

        axC.scatter(dC["age"], dC["acc_cgpt"], s=18, alpha=0.18, color=COL_CGPT, edgecolors="none", label="ChatGPT-4o")
        axC.scatter(dC["age"], dC["acc_gem"],  s=18, alpha=0.18, color=COL_GEM,  edgecolors="none", label="Gemini")

        # OLS slopes + SE (Option A) for D and caption
        slope_cgpt, se_slope_cgpt, intercept_cgpt = slope_and_se(dC["age"], dC["acc_cgpt"])
        slope_gem,  se_slope_gem,  intercept_gem  = slope_and_se(dC["age"], dC["acc_gem"])

        if len(dC) >= 30 and np.isfinite(slope_cgpt) and np.isfinite(slope_gem):
            xx = np.linspace(dC["age"].min(), dC["age"].max(), 200)
            axC.plot(xx, slope_cgpt*xx + intercept_cgpt, linewidth=4, color=COL_CGPT)
            axC.plot(xx, slope_gem*xx  + intercept_gem,  linewidth=4, color=COL_GEM)

        axC.set_ylim(0,100)
        axC.set_xlabel("Age", fontweight="bold")
        axC.set_ylabel("Accuracy (%)", fontweight="bold")
        axC.set_title(f"Figure {tag}C. Age–accuracy by generator\n({cohort_label})", fontweight="bold", pad=12)
        axC.grid(True, linestyle=":", alpha=0.35)
        axC.legend(frameon=True, loc="upper right")

        # -------------------- D: Slope comparison (OLS ± SE) + slope-diff test (HC3) --------------------
        long = pd.concat([
            dC[["age","acc_cgpt"]].rename(columns={"acc_cgpt":"accuracy"}).assign(generator="ChatGPT-4o"),
            dC[["age","acc_gem"]].rename(columns={"acc_gem":"accuracy"}).assign(generator="Gemini"),
        ], ignore_index=True)

        model = smf.ols("accuracy ~ age * C(generator)", data=long).fit(cov_type="HC3")
        inter_terms = [t for t in model.params.index if "age:C(generator)" in t]
        p_inter = float(model.pvalues[inter_terms[0]]) if inter_terms else np.nan

        axD.bar([0,1], [slope_cgpt, slope_gem],
                yerr=[se_slope_cgpt, se_slope_gem], capsize=8,
                width=0.55, color=[COL_CGPT, COL_GEM],
                edgecolor="black", alpha=0.85)

        axD.axhline(0, linestyle="--", linewidth=2, color="black", alpha=0.7)
        axD.set_xticks([0,1])
        axD.set_xticklabels(["ChatGPT-4o","Gemini"], rotation=90, ha="center", fontweight="bold")
        axD.set_ylabel("Slope (% points/year)\n± SE (OLS)", fontweight="bold")
        axD.set_title(f"Figure {tag}D.\nSlope comparison", fontweight="bold", pad=12)
        axD.grid(True, axis="y", linestyle=":", alpha=0.35)

        # p-value outside (no in-plot box)
        fig.text(0.985, 0.02, f"age×generator interaction (HC3): {fmt_p(p_inter)}",
                 ha="right", va="bottom", fontsize=BIG*0.65, fontweight="bold")

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        out_png = OUT_DIR / (f"fig8_{cohort_key}.png" if not is_supp else f"figS8_{cohort_key}.png")
        out_svg = OUT_DIR / (f"fig8_{cohort_key}.svg" if not is_supp else f"figS8_{cohort_key}.svg")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

        # save stats tables
        tests_df = pd.DataFrame([
            {"cohort": cohort_key, "metric":"accuracy", **statA},
            {"cohort": cohort_key, "metric":"rt_seconds", **statB},
            {"cohort": cohort_key, "metric":"slopes_ols",
             "N": int(len(dC)),
             "slope_cgpt": slope_cgpt, "se_cgpt": se_slope_cgpt,
             "slope_gem": slope_gem, "se_gem": se_slope_gem,
             "p_interaction_hc3": p_inter},
        ])
        stats_out = OUT_DIR / (f"fig8_stats_{cohort_key}.csv" if not is_supp else f"figS8_stats_{cohort_key}.csv")
        tests_df.to_csv(stats_out, index=False, encoding="utf-8-sig")

        coef_out = OUT_DIR / (f"fig8D_interaction_coeffs_{cohort_key}.csv" if not is_supp else f"figS8D_interaction_coeffs_{cohort_key}.csv")
        pd.DataFrame({
            "term": model.params.index,
            "coef": model.params.values,
            "se_hc3": model.bse.values,
            "p": model.pvalues.values
        }).to_csv(coef_out, index=False, encoding="utf-8-sig")

        # -------------------- Console summary --------------------
        mean_acc_cgpt = float(np.mean(x1)); mean_acc_gem = float(np.mean(x2))
        se_acc_cgpt   = float(np.std(x1, ddof=1)/np.sqrt(len(x1)))
        se_acc_gem    = float(np.std(x2, ddof=1)/np.sqrt(len(x2)))

        mean_rt_cgpt  = float(np.mean(r1)); mean_rt_gem = float(np.mean(r2))
        se_rt_cgpt    = float(np.std(r1, ddof=1)/np.sqrt(len(r1)))
        se_rt_gem     = float(np.std(r2, ddof=1)/np.sqrt(len(r2)))

        print_fig8_stats(
            cohort_label=cohort_label,
            statA=statA,
            statB=statB,
            p_inter=p_inter,
            mean_acc_cgpt=mean_acc_cgpt,
            mean_acc_gem=mean_acc_gem,
            se_acc_cgpt=se_acc_cgpt,
            se_acc_gem=se_acc_gem,
            mean_rt_cgpt=mean_rt_cgpt,
            mean_rt_gem=mean_rt_gem,
            se_rt_cgpt=se_rt_cgpt,
            se_rt_gem=se_rt_gem,
            slope_cgpt=slope_cgpt,
            se_slope_cgpt=se_slope_cgpt,
            slope_gem=slope_gem,
            se_slope_gem=se_slope_gem,
            N_scatter=len(dC)
        )

        print("✅ saved:", out_png)
        print("✅ stats:", stats_out)
        print("✅ interaction coeffs:", coef_out)


    # Run
    run_one("mobile", is_supp=False)
    run_one("pc", is_supp=True)

    print("✅ Figure 8 outputs saved to:", OUT_DIR)


def main():
    _run_cell_039()
    _run_cell_041()


if __name__ == "__main__":
    main()
