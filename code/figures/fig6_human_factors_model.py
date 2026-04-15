"""Figure 6 assembly: strategy story and integrated human-factors model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_022():
    # ==============================================================================
    # Figure 5 (Mobile) Strategy Story
    # ------------------------------------------------------------------------------
    # 5A: Multivariate strategy effects (11-4; controls age/sex/other strategies; HC3; FDR)
    # 5B: Strategy usage frequency (mobile; from analysis_data_mobile_age_filtered_20_69.csv)
    # 5C: Age-bin × Sex usage differences for effective strategies (12; mostly n.s. after FDR)
    #
    # Save to:
    #   plots/run_20260119_192624/05_strategy_story/
    # ==============================================================================
    import json
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    RUN_TAG = config.RUN_TAG

    # --- paths (fixed) ---
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "05_strategy_story"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    RUN_DIR = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}"

    PATH_11_4 = RUN_DIR / "11_strategy_effectiveness" / "11-4_strategy_multivariate_regression_table_mobile.csv"
    PATH_12_1 = RUN_DIR / "12_strategy_usage" / "12-1_strategy_usage_summary_table_mobile.csv"
    PATH_MOBILE = config.MOBILE_AGE_FILTERED

    # --- big fonts (paper; consistent with your 8_plus figures) ---
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.75)

    # --- palettes ---
    COL_POS = "#DB4437"   # significant positive
    COL_NEG = "#4285F4"   # significant negative
    COL_NS  = "#BDBDBD"   # not significant
    SEX_COL = {"male": "#4285F4", "female": "#DB4437"}

    # --- strategy key list / labels (match your section 11) ---
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

    def ensure_exists(p: Path, label: str):
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    ensure_exists(PATH_11_4, "11-4 table")
    ensure_exists(PATH_12_1, "12-1 table")
    ensure_exists(PATH_MOBILE, "mobile analysis csv")

    # ==============================================================================
    # Panel 5A: 11-4 multivariate strategy coefficients (β_pp with 95% CI; HC3)
    # ==============================================================================
    reg = pd.read_csv(PATH_11_4, encoding="utf-8-sig")

    # expected columns: strategy, strategy_label, beta_pp, se_hc3, q_fdr_bh, reject_fdr_bh(q<0.05)
    need_cols = {"strategy","strategy_label","beta_pp","se_hc3","q_fdr_bh","reject_fdr_bh(q<0.05)"}
    miss = need_cols - set(reg.columns)
    if miss:
        raise ValueError(f"11-4 table missing columns: {miss}")

    reg = reg.copy()
    reg["ci_low"]  = reg["beta_pp"] - 1.96 * reg["se_hc3"]
    reg["ci_high"] = reg["beta_pp"] + 1.96 * reg["se_hc3"]

    # sort by beta
    reg_plot = reg.sort_values("beta_pp").copy()

    def coef_color(r):
        if bool(r["reject_fdr_bh(q<0.05)"]):
            return COL_POS if r["beta_pp"] > 0 else COL_NEG
        return COL_NS

    reg_plot["color"] = reg_plot.apply(coef_color, axis=1)

    # save a small summary for paper
    sig_reg = reg_plot[reg_plot["reject_fdr_bh(q<0.05)"]].sort_values("q_fdr_bh")
    sig_out = OUT_DIR / "fig5A_sig_strategies_mobile.csv"
    sig_reg.to_csv(sig_out, index=False, encoding="utf-8-sig")

    # ==============================================================================
    # Panel 5B: usage frequency from mobile CSV (strategy tokens)
    # ==============================================================================
    mob = pd.read_csv(PATH_MOBILE, encoding="utf-8-sig")
    if "strategy" not in mob.columns:
        raise KeyError("mobile CSV missing 'strategy' column")

    def parse_strategy_list(cell):
        if pd.isna(cell):
            return []
        s = str(cell).strip().lower()
        if s == "on":
            return ALL_STRATEGIES.copy()
        # split by common delimiters
        tokens = re.split(r"[,;|/\\]+", s)
        tokens = [t.strip() for t in tokens if t.strip()]
        # keep only known
        return [t for t in tokens if t in STRATEGY_LABEL_EN]

    tokens = mob["strategy"].apply(parse_strategy_list)
    token_sets = tokens.apply(set)

    usage_rows = []
    N_total = len(mob)
    for k in ALL_STRATEGIES:
        used = token_sets.apply(lambda st: k in st).sum()
        usage_rows.append({"strategy": k, "strategy_label": STRATEGY_LABEL_EN[k], "n_used": int(used), "pct_used": float(used/N_total*100.0)})

    usage = pd.DataFrame(usage_rows).sort_values("pct_used", ascending=False)
    usage_out = OUT_DIR / "fig5B_strategy_usage_frequency_mobile.csv"
    usage.to_csv(usage_out, index=False, encoding="utf-8-sig")

    # For B panel, show only most relevant set to keep figure clean:
    #   - effective positives from 11-4 + "Don't know" (negative cue) + optionally "Random"
    top_effective = sig_reg["strategy"].tolist()
    show_keys = []
    for k in ["texture","painting-like","feeling","dont-know","random"]:
        if k in usage["strategy"].values:
            show_keys.append(k)
    # ensure include effective ones even if not in list
    for k in top_effective:
        if k not in show_keys:
            show_keys.append(k)

    usage_show = usage[usage["strategy"].isin(show_keys)].copy()
    # order bars by pct_used
    usage_show = usage_show.sort_values("pct_used", ascending=True)

    # ==============================================================================
    # Panel 5C: 12 usage (age_bin × sex) for effective strategies (mobile)
    # ==============================================================================
    tab12 = pd.read_csv(PATH_12_1, encoding="utf-8-sig")
    # expected columns: age_group, strategy_key, male_usage_pct, female_usage_pct, q_fdr_bh, fdr_reject_q_lt_0_05
    need12 = {"age_group","strategy_key","strategy","male_usage_pct","female_usage_pct","q_fdr_bh","fdr_reject_q_lt_0_05"}
    miss12 = need12 - set(tab12.columns)
    if miss12:
        raise ValueError(f"12-1 table missing columns: {miss12}")

    # choose strategies = effective from 11-4 (mobile)
    eff_keys = top_effective[:]  # ['texture','painting-like','feeling']
    tab12_eff = tab12[tab12["strategy_key"].isin(eff_keys)].copy()

    # long format
    melt = tab12_eff.melt(
        id_vars=["age_group","strategy_key","strategy","q_fdr_bh","fdr_reject_q_lt_0_05"],
        value_vars=["male_usage_pct","female_usage_pct"],
        var_name="sex",
        value_name="usage_pct"
    )
    melt["sex"] = melt["sex"].replace({"male_usage_pct":"male", "female_usage_pct":"female"})
    melt["age_group"] = pd.Categorical(melt["age_group"], categories=["20s","30s","40s","50s","60s"], ordered=True)
    melt["strategy"] = pd.Categorical(melt["strategy"], categories=[STRATEGY_LABEL_EN[k] for k in eff_keys], ordered=True)
    melt = melt.sort_values(["strategy","age_group","sex"])

    melt_out = OUT_DIR / "fig5C_usage_agebin_sex_effective_mobile.csv"
    melt.to_csv(melt_out, index=False, encoding="utf-8-sig")

    # ==============================================================================
    # Build the Figure 5 layout (3 panels)
    #   A spans top row
    #   B bottom-left
    #   C bottom-right
    # ==============================================================================
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.15])

    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ----------------
    # 5A plot
    # ----------------
    axA.set_title("A  Strategy effects on accuracy (Mobile; multivariate OLS with HC3; FDR)", fontweight="bold", pad=14)

    y = np.arange(len(reg_plot))
    axA.hlines(y=y, xmin=reg_plot["ci_low"], xmax=reg_plot["ci_high"], color="#9E9E9E", linewidth=2.5, alpha=0.9)
    axA.scatter(reg_plot["beta_pp"], y, s=110, c=reg_plot["color"], edgecolors="none")

    axA.axvline(0, linestyle="--", linewidth=2, color="black", alpha=0.8)

    axA.set_yticks(y)
    axA.set_yticklabels(reg_plot["strategy_label"])
    axA.set_xlabel("Regression coefficient (percentage points) with 95% CI (HC3)", fontweight="bold")

    # small legend (inside, upper right)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COL_POS, label="FDR sig (+)"),
        Patch(facecolor=COL_NEG, label="FDR sig (−)"),
        Patch(facecolor=COL_NS,  label="Not sig"),
    ]
    axA.legend(handles=legend_elems, loc="lower right", frameon=True)

    # annotate model note succinctly
    axA.text(
        0.01, 0.02,
        "Model controls: Age + Sex (female) + all strategy indicators; Robust SE: HC3",
        transform=axA.transAxes,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    axA.grid(True, axis="x", linestyle=":", alpha=0.35)

    # ----------------
    # 5B plot (usage frequency)
    # ----------------
    axB.set_title("B  Strategy usage frequency (Mobile)", fontweight="bold", pad=14)

    axB.barh(
        usage_show["strategy_label"],
        usage_show["pct_used"],
        edgecolor="black",
        alpha=0.85
    )

    axB.set_xlabel("Used (%)", fontweight="bold")
    axB.set_xlim(0, max(usage_show["pct_used"].max()*1.15, 10))

    # add n labels
    for i, r in usage_show.reset_index(drop=True).iterrows():
        axB.text(r["pct_used"] + 1.0, i, f"n={int(r['n_used'])}", va="center")

    axB.grid(True, axis="x", linestyle=":", alpha=0.35)

    # ----------------
    # 5C plot (agebin×sex usage for effective strategies)
    # ----------------
    axC.set_title("C  Effective strategy usage by age group & sex (Mobile; FDR n.s.)", fontweight="bold", pad=14)

    # We'll plot three lines per strategy? Too busy.
    # Instead: grouped bars by sex, faceted by strategy using y-offset blocks within one axis.
    # Simple approach: plot each strategy as a separate mini-block on y-axis with its own x bins.

    # Build x positions for age bins
    age_bins = ["20s","30s","40s","50s","60s"]
    xpos = np.arange(len(age_bins))

    # spacing between strategies on y axis (we'll use separate "rows" by shifting y)
    # Here we create stacked blocks: each strategy has its own baseline y=0 line but same axis;
    # easier: use multiple small sub-axes inside axC? not allowed with gridspec easily.
    # We'll use offset bars: for each strategy, shift x by small amount and separate by panels using annotation.
    # Better: draw as three small lines (male/female) with different line styles per strategy.

    line_styles = {"Texture":"solid", "Painting-like":"dashed", "Feeling/Intuition":"dotted"}
    for strat_name in [STRATEGY_LABEL_EN[k] for k in eff_keys]:
        sub = melt[melt["strategy"] == strat_name].copy()
        # male line
        m_sub = sub[sub["sex"]=="male"].set_index("age_group").reindex(age_bins).reset_index()
        f_sub = sub[sub["sex"]=="female"].set_index("age_group").reindex(age_bins).reset_index()

        axC.plot(xpos, m_sub["usage_pct"].values, color=SEX_COL["male"], linewidth=3.0,
                 linestyle=line_styles.get(strat_name,"solid"), alpha=0.9)
        axC.plot(xpos, f_sub["usage_pct"].values, color=SEX_COL["female"], linewidth=3.0,
                 linestyle=line_styles.get(strat_name,"solid"), alpha=0.9)

    # axis formatting
    axC.set_xticks(xpos)
    axC.set_xticklabels(age_bins, fontweight="bold")
    axC.set_xlabel("Age group", fontweight="bold")
    axC.set_ylabel("Usage rate (%)", fontweight="bold")
    axC.set_ylim(0, 100)
    axC.grid(True, axis="y", linestyle=":", alpha=0.35)

    # legend (outside)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0],[0], color=SEX_COL["male"], linewidth=4, label="Male"),
        Line2D([0],[0], color=SEX_COL["female"], linewidth=4, label="Female"),
        Line2D([0],[0], color="black", linewidth=3, linestyle="solid", label="Texture"),
        Line2D([0],[0], color="black", linewidth=3, linestyle="dashed", label="Painting-like"),
        Line2D([0],[0], color="black", linewidth=3, linestyle="dotted", label="Feeling/Intuition"),
    ]
    axC.legend(handles=legend_lines, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    # FDR note (since all are n.s.)
    axC.text(
        0.02, 0.02,
        "Note: No age-bin × sex usage differences survive FDR correction (q<0.05).",
        transform=axC.transAxes,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    # overall title
    fig.suptitle("Figure 5. Decision cues (strategies) associated with discrimination accuracy (Mobile)", fontweight="bold", y=0.995)

    # layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_out = OUT_DIR / "fig5_strategy_story_mobile.png"
    svg_out = OUT_DIR / "fig5_strategy_story_mobile.svg"
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # meta
    meta = {
        "created_at": datetime.now().isoformat(),
        "run_tag": RUN_TAG,
        "outputs": {
            "fig_png": str(png_out),
            "fig_svg": str(svg_out),
            "fig5A_sig_csv": str(sig_out),
            "fig5B_usage_csv": str(usage_out),
            "fig5C_long_csv": str(melt_out),
        },
        "inputs": {
            "11_4_table": str(PATH_11_4),
            "12_1_table": str(PATH_12_1),
            "mobile_data": str(PATH_MOBILE),
        },
        "notes": [
            "Panel A: multivariate OLS with robust HC3 SE; CI=±1.96*SE; FDR(BH) on strategy terms.",
            "Panel B: strategy usage frequency computed from 'strategy' tokens in mobile dataset.",
            "Panel C: age-bin × sex strategy usage for effective strategies; none survive FDR (mobile).",
        ]
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Saved Figure 5 to:", OUT_DIR)
    print(" -", png_out)


def _run_cell_035():
    # ==============================================================================
    # Figure 6: Integrated Human-factors model (Mobile main; PC supplementary)
    # ------------------------------------------------------------------------------
    # Outputs:
    #   plots/run_20260119_192624/06_human_factors_model/
    #     - fig6A_mobile_std_beta.png/.svg
    #     - fig6B_mobile_nested_R2.png/.svg
    #     - figS6A_pc_std_beta.png/.svg
    #     - figS6B_pc_nested_R2.png/.svg
    #     - tables: coef tables + model fit tables (csv)
    # ==============================================================================

    import re
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    import statsmodels.api as sm
    from matplotlib.patches import Patch   # ✅ needed

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT   # data location
    RUN_OUT = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "06_human_factors_model"
    RUN_OUT.mkdir(parents=True, exist_ok=True)

    FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "pc":     config.WEB_AGE_FILTERED,
    }

    # big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.75)

    COL_POS = "#DB4437"
    COL_NEG = "#4285F4"
    COL_NS  = "#BDBDBD"

    STRATEGY_KEYS = ["texture", "painting-like", "feeling"]  # effective strategies (11-4 mobile)

    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_acc_col(df):
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns: return c
        raise KeyError("No overallAccuracy column found.")

    def to_percent(s):
        s = pd.to_numeric(s, errors="coerce")
        finite = s.dropna()
        if finite.empty: return s
        return s * 100.0 if float(finite.max()) <= 1.5 else s

    def normalize_sex(df):
        out = df.copy()
        src = None
        for c in ["sex","gender"]:
            if c in out.columns:
                src = c; break
        if src is None:
            out["sex"] = np.nan
            return out
        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "": np.nan})
        s = s.replace({
            "m":"male","man":"male",
            "f":"female","woman":"female",
            "prefer not to say": np.nan,
            "prefer_not_to_say": np.nan,
            "prefer not to": np.nan,
        })
        out["sex"] = s
        return out

    def confidence_to_score(s):
        mapping = {
            "very-not-confident": 1,
            "not-confident": 2,
            "neutral": 3,
            "confident": 4,
            "very-confident": 5
        }
        return s.astype(str).str.lower().str.strip().map(mapping)

    def exposure_to_score(s):
        mapping = {"never":1, "rarely":2, "sometimes":3, "weekly":4, "daily":5}
        return s.astype(str).str.lower().str.strip().map(mapping)

    def attitude_to_score(s):
        mapping = {"very-negative":-2, "negative":-1, "neutral":0, "positive":1, "very-positive":2}
        return s.astype(str).str.lower().str.strip().map(mapping)

    def parse_strategy_list(cell):
        if pd.isna(cell):
            return []
        txt = str(cell).strip().lower()
        if txt == "on":
            return STRATEGY_KEYS.copy()
        tokens = re.split(r"[,;|/\\]+", txt)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [t for t in tokens if t in STRATEGY_KEYS]

    def zscore(series):
        x = pd.to_numeric(series, errors="coerce")
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=series.index)
        return (x - mu) / sd

    def fit_ols_hc3(y, X):
        Xc = sm.add_constant(X, has_constant="add")
        return sm.OLS(y, Xc, missing="drop").fit(cov_type="HC3")

    def fit_ols_plain(y, X):
        Xc = sm.add_constant(X, has_constant="add")
        return sm.OLS(y, Xc, missing="drop").fit()

    # -----------------------------
    # Prepare cohort data
    # -----------------------------
    def prep_cohort(path, cohort_label):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = normalize_sex(df)

        acc_col = resolve_acc_col(df)
        df["accuracy_pct"] = to_percent(df[acc_col])

        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["sex_female"] = (df["sex"] == "female").astype(float)

        # AI self-report measures
        df["confidence_score"] = confidence_to_score(df["aiConfidence"]) if "aiConfidence" in df.columns else np.nan
        df["exposure_score"]   = exposure_to_score(df["aiExposureFrequency"]) if "aiExposureFrequency" in df.columns else np.nan
        df["attitude_score"]   = attitude_to_score(df["aiAttitude"]) if "aiAttitude" in df.columns else np.nan

        # strategy dummies (only 3 keys)
        if "strategy" in df.columns:
            toks = df["strategy"].apply(parse_strategy_list).apply(set)
            for k in STRATEGY_KEYS:
                df[f"strat_{k}"] = toks.apply(lambda st: float(k in st))
        else:
            for k in STRATEGY_KEYS:
                df[f"strat_{k}"] = np.nan

        # keep male/female only (exclude prefer-not)
        d = df[df["sex"].isin(["male","female"])].copy()

        # complete-case
        needed = ["accuracy_pct","age","sex_female","confidence_score","exposure_score","attitude_score"] + [f"strat_{k}" for k in STRATEGY_KEYS]
        d = d.dropna(subset=needed).copy()

        print(f"[{cohort_label}] N complete-case for unified model: {len(d):,}")
        return d

    # -----------------------------
    # Figure 6A: standardized beta plot
    # -----------------------------
    def make_std_beta_plot(d, cohort_label, out_png, out_svg, out_table_csv):
        dz = d.copy()
        dz["y_z"] = zscore(dz["accuracy_pct"])
        dz["age_z"] = zscore(dz["age"])
        dz["conf_z"] = zscore(dz["confidence_score"])
        dz["expo_z"] = zscore(dz["exposure_score"])
        dz["att_z"] = zscore(dz["attitude_score"])

        X = pd.DataFrame({
            "Age (z)": dz["age_z"],
            "Sex (female=1)": dz["sex_female"],
            "Confidence (z)": dz["conf_z"],
            "Exposure (z)": dz["expo_z"],
            "Attitude (z)": dz["att_z"],
            "Strategy: Texture": dz["strat_texture"],
            "Strategy: Painting-like": dz["strat_painting-like"],
            "Strategy: Feeling": dz["strat_feeling"],
        })

        model = fit_ols_hc3(dz["y_z"], X)

        coef = model.params.drop("const", errors="ignore")
        se = model.bse.drop("const", errors="ignore")
        p = model.pvalues.drop("const", errors="ignore")

        table = pd.DataFrame({
            "term": coef.index,
            "beta_std": coef.values,
            "se_hc3": se.values,
            "p": p.values,
            "ci_low": coef.values - 1.96*se.values,
            "ci_high": coef.values + 1.96*se.values,
        }).sort_values("beta_std")

        table.to_csv(out_table_csv, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(16, 10))
        y = np.arange(len(table))

        ax.hlines(y, table["ci_low"], table["ci_high"], color="#9E9E9E", linewidth=2.5, alpha=0.9)

        colors = []
        for _, r in table.iterrows():
            if r["p"] < 0.05:
                colors.append(COL_POS if r["beta_std"] > 0 else COL_NEG)
            else:
                colors.append(COL_NS)

        ax.scatter(table["beta_std"], y, s=140, c=colors, edgecolors="none")
        ax.axvline(0, linestyle="--", linewidth=2, color="black", alpha=0.85)

        ax.set_yticks(y)
        ax.set_yticklabels(table["term"])
        ax.set_xlabel("Standardized coefficient (β) with 95% CI (HC3)", fontweight="bold")
        ax.set_title(f"Figure 6A. Unified human-factors model (standardized β) [{cohort_label}]",
                     fontweight="bold", pad=14)
        ax.grid(True, axis="x", linestyle=":", alpha=0.35)

        legend_elems = [
            Patch(facecolor=COL_POS, label="p<.05 (β>0)"),
            Patch(facecolor=COL_NEG, label="p<.05 (β<0)"),
            Patch(facecolor=COL_NS, label="n.s."),
        ]
        ax.legend(handles=legend_elems, loc="lower right", frameon=True)

        plt.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print(f"\n[{cohort_label}] Unified model (standardized β) summary (HC3):")
        print(table.round(3).to_string(index=False))
        print(f"Saved coef table: {out_table_csv}\n")

    # -----------------------------
    # Figure 6B: nested model ΔR² (labels fixed; "AI self-reports" wording)
    # -----------------------------
    def make_nested_r2_clean_labels(d, cohort_label, out_png, out_svg, out_table_csv):
        y = d["accuracy_pct"]

        X0 = pd.DataFrame({"Age": d["age"], "Sex_female": d["sex_female"]})
        X1 = pd.DataFrame({
            "Age": d["age"],
            "Sex_female": d["sex_female"],
            "Confidence": d["confidence_score"],
            "Exposure": d["exposure_score"],
            "Attitude": d["attitude_score"],
        })
        X2 = pd.DataFrame({
            "Age": d["age"],
            "Sex_female": d["sex_female"],
            "Confidence": d["confidence_score"],
            "Exposure": d["exposure_score"],
            "Attitude": d["attitude_score"],
            "Texture": d["strat_texture"],
            "Painting_like": d["strat_painting-like"],
            "Feeling": d["strat_feeling"],
        })

        m0 = fit_ols_plain(y, X0)
        m1 = fit_ols_plain(y, X1)
        m2 = fit_ols_plain(y, X2)

        tab = pd.DataFrame([
            {"model": "Demographics\n(Age + Sex)", "R2": m0.rsquared, "Adj_R2": m0.rsquared_adj},
            {"model": "+ AI self-reports\n(+ Confidence, Exposure, Attitude)", "R2": m1.rsquared, "Adj_R2": m1.rsquared_adj},
            {"model": "+ Strategies\n(+ Texture, Painting-like, Feeling)", "R2": m2.rsquared, "Adj_R2": m2.rsquared_adj},
        ])
        tab["ΔR2_vs_prev"] = tab["R2"].diff()
        tab.to_csv(out_table_csv, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(tab))
        ax.plot(x, tab["R2"], marker="o", linewidth=3)

        ax.set_xticks(x)
        ax.set_xticklabels(tab["model"], rotation=0, ha="center")
        ax.set_ylabel("R²", fontweight="bold")
        ax.set_title(f"Figure 6B. Incremental variance explained (nested models) [{cohort_label}]",
                     fontweight="bold", pad=14)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

        for i in range(1, len(tab)):
            ax.text(i, tab.loc[i, "R2"] + 0.01,
                    f"ΔR² = {tab.loc[i, 'ΔR2_vs_prev']:.3f}",
                    ha="center", fontweight="bold")

        ax.text(
            0.02, 0.02,
            "ΔR² indicates the added explanatory power\nwhen each block is added sequentially.",
            transform=ax.transAxes,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            fontweight="bold"
        )

        plt.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print(f"\n[{cohort_label}] Nested model R² summary:")
        print(tab.round(4).to_string(index=False))
        print(f"Saved R² table: {out_table_csv}\n")

    # -----------------------------
    # Run: Mobile main + PC supp (✅ fixed labels + filenames)
    # -----------------------------
    for cohort_key in ["mobile", "pc"]:
        d = prep_cohort(FILES[cohort_key], cohort_key.upper())

        if cohort_key == "mobile":
            label = "MOBILE"
            outA_png = RUN_OUT / "fig6A_mobile_std_beta.png"
            outA_svg = RUN_OUT / "fig6A_mobile_std_beta.svg"
            outA_csv = RUN_OUT / "fig6A_mobile_std_beta_table.csv"

            outB_png = RUN_OUT / "fig6B_mobile_nested_R2.png"
            outB_svg = RUN_OUT / "fig6B_mobile_nested_R2.svg"
            outB_csv = RUN_OUT / "fig6B_mobile_nested_R2_table.csv"
        else:
            label = "PC"
            outA_png = RUN_OUT / "figS6A_pc_std_beta.png"
            outA_svg = RUN_OUT / "figS6A_pc_std_beta.svg"
            outA_csv = RUN_OUT / "figS6A_pc_std_beta_table.csv"

            outB_png = RUN_OUT / "figS6B_pc_nested_R2.png"
            outB_svg = RUN_OUT / "figS6B_pc_nested_R2.svg"
            outB_csv = RUN_OUT / "figS6B_pc_nested_R2_table.csv"

        make_std_beta_plot(d, label, outA_png, outA_svg, outA_csv)
        make_nested_r2_clean_labels(d, label, outB_png, outB_svg, outB_csv)

    # meta
    meta = {
        "created_at": datetime.now().isoformat(),
        "run_tag": RUN_TAG,
        "outputs_dir": str(RUN_OUT),
        "notes": [
            "Fig6A: standardized regression (z-scored y and continuous predictors) with HC3 robust SE.",
            "Fig6B: nested-model R² (plain OLS) with sequential blocks: Demographics, AI self-reports, Strategies.",
            "Mobile main + PC supplementary outputs saved with correct labels/filenames."
        ],
    }
    (RUN_OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Saved Figure 6 outputs to:", RUN_OUT)


def main():
    _run_cell_022()
    _run_cell_035()


if __name__ == "__main__":
    main()
