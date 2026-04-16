"""Figure 4 assembly: sex differences panels and bootstrap forest plots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_010():
    # ==============================================================================
    # (Figure 4 Prep | 4A) Sex distribution pie + Age-bin × Sex accuracy plot (Mobile)
    # ------------------------------------------------------------------------------
    # Input:
    #   analysis_data_mobile_age_filtered_20_69.csv  (must include age, overallAccuracy_y, gender/sex)
    #
    # Output:
    #   plots/run_20260119_192624/04_fig4_sex_age/
    #     - 4A-0_sex_distribution_pie_mobile.png/.svg
    #     - 4A-1_agebin_by_sex_accuracy_mobile.png/.svg
    #     - 4A-2_agebin_by_sex_summary_mobile.csv
    #     - meta.json
    # ==============================================================================
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    PLOTS_ROOT = config.PLOTS_DIR / f"run_{config.RUN_TAG}"
    SECTION_DIR = PLOTS_ROOT / "04_fig4_sex_age"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_PATH = config.MOBILE_AGE_FILTERED

    # paper font (>=3x 느낌)
    FONT_SCALE = 3.2
    BASE = 10
    BIG = BASE * FONT_SCALE

    plt.rcParams.update({
        "font.size": BIG,
        "axes.titlesize": BIG * 1.10,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG * 0.90,
        "ytick.labelsize": BIG * 0.90,
        "legend.fontsize": BIG * 0.75,
    })

    # Fixed sex palette (same mapping everywhere)
    SEX_ORDER = ["male", "female", "prefer-not-to-say"]
    SEX_LABEL = {"male": "Male", "female": "Female", "prefer-not-to-say": "Prefer not to say"}
    SEX_COLOR = {"male": "#4285F4", "female": "#DB4437", "prefer-not-to-say": "#F4B400"}

    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_overall_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("overallAccuracy column not found.")

    def to_percent_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        finite = s_num.dropna()
        if finite.empty:
            return s_num
        mx = float(finite.max())
        return s_num * 100.0 if mx <= 1.5 else s_num

    def normalize_sex(df: pd.DataFrame, out_col="sex"):
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
            "prefer-not-to-say": "prefer-not-to-say",
        })

        out[out_col] = s
        return out

    def age_to_bin(age):
        """20-69 -> 20s/30s/40s/50s/60s bins."""
        if not np.isfinite(age):
            return np.nan
        a = int(age)
        if a < 20 or a > 69:
            return np.nan
        if a < 30: return "20s"
        if a < 40: return "30s"
        if a < 50: return "40s"
        if a < 60: return "50s"
        return "60s"

    AGE_BIN_ORDER = ["20s","30s","40s","50s","60s"]

    # -----------------------------
    # Load + prep
    # -----------------------------
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df = normalize_sex(df, out_col="sex")

    acc_col = resolve_overall_accuracy_column(df)
    df["accuracy_pct"] = to_percent_series(df[acc_col])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Keep 20-69 only (this file is already filtered, but keep safe)
    df = df[(df["age"] >= 20) & (df["age"] <= 69)].copy()

    # -----------------------------
    # (4A-0) Sex distribution pie (male/female/prefer-not-to-say)
    # -----------------------------
    counts = df["sex"].value_counts(dropna=False)
    counts = counts.reindex(SEX_ORDER).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(12, 12))
    sizes = [counts[k] for k in SEX_ORDER]
    colors = [SEX_COLOR[k] for k in SEX_ORDER]

    ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(linewidth=2, edgecolor="white")
    )

    legend_labels = [f"{SEX_LABEL[k]} (n={counts[k]})" for k in SEX_ORDER]
    ax.legend(
        legend_labels,
        title="Sex",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True
    )

    ax.set_title("Figure 4A-0. Sex distribution (Mobile)", fontweight="bold", pad=18)

    pie_png = SECTION_DIR / "4A-0_sex_distribution_pie_mobile.png"
    pie_svg = SECTION_DIR / "4A-0_sex_distribution_pie_mobile.svg"
    plt.savefig(pie_png, dpi=300, bbox_inches="tight")
    plt.savefig(pie_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------
    # (4A-1) Age bin × Sex: mean ± SEM line plot (male/female only)
    # -----------------------------
    d = df[df["sex"].isin(["male","female"])].copy()
    d["age_bin"] = d["age"].apply(age_to_bin)
    d = d.dropna(subset=["age_bin","accuracy_pct"]).copy()

    grp = d.groupby(["age_bin","sex"])["accuracy_pct"]
    summary = grp.agg(n="count", mean="mean", std="std").reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["age_bin"] = pd.Categorical(summary["age_bin"], categories=AGE_BIN_ORDER, ordered=True)
    summary["sex"] = pd.Categorical(summary["sex"], categories=["male","female"], ordered=True)
    summary = summary.sort_values(["age_bin","sex"])

    summary_out = SECTION_DIR / "4A-2_agebin_by_sex_summary_mobile.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 10))

    xpos = np.arange(len(AGE_BIN_ORDER))

    for sex in ["male","female"]:
        ss = summary[summary["sex"] == sex].copy()
        ss = ss.set_index("age_bin").reindex(AGE_BIN_ORDER).reset_index()

        y = ss["mean"].values
        yerr = ss["sem"].values

        ax.errorbar(
            xpos, y, yerr=yerr,
            marker="o",
            linewidth=3.0,
            markersize=10,
            capsize=8,
            color=SEX_COLOR[sex],                 # ✅ pie와 동일 색
            markerfacecolor=SEX_COLOR[sex],       # ✅ 점도 동일 색
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=SEX_LABEL[sex]
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels(AGE_BIN_ORDER, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Age group", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Figure 4A-1. Accuracy by age group and sex (Mobile; mean ± SEM)", fontweight="bold", pad=18)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # interaction p (placeholder)
    interaction_text = "Age × Sex interaction: p = (fill)"
    ax.text(
        0.98, 0.98, interaction_text,
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    # ✅ legend를 플롯 바깥으로 이동 (오버랩 방지)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Sex"
    )

    # ✅ 오른쪽에 legend 공간 확보
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    png_out = SECTION_DIR / "4A-1_agebin_by_sex_accuracy_mobile.png"
    svg_out = SECTION_DIR / "4A-1_agebin_by_sex_accuracy_mobile.svg"
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------
    # meta
    # -----------------------------
    meta = {
        "notebook": "ipynb8_plus",
        "section": "04_fig4_sex_age",
        "created_at": datetime.now().isoformat(),
        "input": INPUT_PATH,
        "accuracy_column_used": acc_col,
        "outputs": {
            "pie_png": str(pie_png),
            "pie_svg": str(pie_svg),
            "agebin_plot_png": str(png_out),
            "agebin_plot_svg": str(svg_out),
            "summary_csv": str(summary_out),
        },
        "notes": [
            "4A-0 includes male/female/prefer-not-to-say for descriptive purposes.",
            "4A-1 uses male/female only; y-axis fixed 0-100; mean±SEM per age bin.",
            "Interaction p-value is left as placeholder; paste your ANOVA interaction p there."
        ]
    }
    (SECTION_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("✅ saved meta:", SECTION_DIR / "meta.json")
    print("✅ outputs dir:", SECTION_DIR)


def _run_cell_012():
    # ==============================================================================
    # (Figure 4 | β sex-diff bootstrap) Mobile main figures: 4A/4B/4C/4D (INTEGRATED)
    # - Fix 4B IndexError
    # - Color-code groups in 4B (Male/Female/Female−Male)
    # - Print key effects + interpretation to console
    # ==============================================================================
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from graphviz import Digraph

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "04_fig4_sex_diff_beta"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_MOBILE = config.MOBILE_AGE_FILTERED

    EFF_DIR = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}" / "35_1b_beta_text" / "mobile"
    EFF_MALE   = EFF_DIR / "35.1b-1_effects_male_mobile.csv"
    EFF_FEMALE = EFF_DIR / "35.1b-1_effects_female_mobile.csv"
    EFF_DIFF   = EFF_DIR / "35.1b-3_effects_diff_female_minus_male_mobile.csv"

    # Big text (>=3x)
    FONT_SCALE = 3.2
    BASE = 10
    BIG = BASE * FONT_SCALE
    plt.rcParams.update({
        "font.size": BIG,
        "axes.titlesize": BIG * 1.10,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG * 0.90,
        "ytick.labelsize": BIG * 0.90,
        "legend.fontsize": BIG * 0.75,
    })

    # Sex palette (consistent)
    SEX_COLOR = {"male": "#4285F4", "female": "#DB4437", "prefer-not-to-say": "#F4B400"}
    SEX_LABEL = {"male": "Male", "female": "Female", "prefer-not-to-say": "Prefer not to say"}

    # Forest plot group colors
    GROUP_COLORS = {
        "Male": "#4285F4",
        "Female": "#DB4437",
        "Female−Male": "#6A1B9A",  # diff in purple
    }

    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_overall_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns: return "overallAccuracy_y"
        if "overallAccuracy" in df.columns: return "overallAccuracy"
        if "overallAccuracy_x" in df.columns: return "overallAccuracy_x"
        raise KeyError("overallAccuracy column not found.")

    def to_percent_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        finite = s_num.dropna()
        if finite.empty: return s_num
        mx = float(finite.max())
        return s_num * 100.0 if mx <= 1.5 else s_num

    def normalize_sex(df: pd.DataFrame, out_col="sex"):
        out = df.copy()
        src = None
        for c in ["sex","gender"]:
            if c in out.columns:
                src = c; break
        if src is None:
            out[out_col] = np.nan; return out
        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
        s = s.replace({
            "m": "male", "man": "male",
            "f": "female", "woman": "female",
            "prefer not to say": "prefer-not-to-say",
            "prefer_not_to_say": "prefer-not-to-say",
            "prefer not to": "prefer-not-to-say",
            "prefer-not-to-say": "prefer-not-to-say",
        })
        out[out_col] = s
        return out

    def age_to_bin(age):
        if not np.isfinite(age): return np.nan
        a = int(age)
        if a < 20 or a > 69: return np.nan
        if a < 30: return "20s"
        if a < 40: return "30s"
        if a < 50: return "40s"
        if a < 60: return "50s"
        return "60s"

    AGE_BIN_ORDER = ["20s","30s","40s","50s","60s"]

    def load_effects(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")
        need = {"effect","coef","CI[2.5%]","CI[97.5%]","sig_CI_nonzero"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns in {path}: {miss}")
        return df

    def row_of(df: pd.DataFrame, effect: str):
        hit = df[df["effect"] == effect]
        return hit.iloc[0] if len(hit) else None

    def fmt_ci(row, nd=3):
        if row is None:
            return "NA"
        coef = float(row["coef"])
        lo = float(row["CI[2.5%]"])
        hi = float(row["CI[97.5%]"])
        sig = bool(row["sig_CI_nonzero"])
        return f"{coef:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" + (" *" if sig else "")

    def fmt_coef(row, nd=3):
        if row is None:
            return "NA"
        coef = float(row["coef"])
        sig = bool(row["sig_CI_nonzero"])
        return f"{coef:.{nd}f}" + ("*" if sig else "")

    # -----------------------------
    # Load β sex-diff outputs (mobile)
    # -----------------------------
    for p in [EFF_MALE, EFF_FEMALE, EFF_DIFF]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    male_df = load_effects(EFF_MALE)
    female_df = load_effects(EFF_FEMALE)
    diff_df = load_effects(EFF_DIFF)

    print("==============================================================================")
    print("Figure 4 (Mobile) | β sex-diff bootstrap")
    print("Using files:")
    print(" -", EFF_MALE)
    print(" -", EFF_FEMALE)
    print(" -", EFF_DIFF)
    print("==============================================================================\n")

    # -----------------------------
    # Console summary for key effects (meaning + values)
    # -----------------------------
    KEY_EFF = ["ind2","a2","b2","cprime","ind_total","ind1","ind3"]

    meaning = {
        "ind2": "Indirect via Confidence: (Age→Confidence)×(Confidence→Accuracy)",
        "a2":   "Path a2: Age→Confidence",
        "b2":   "Path b2: Confidence→Accuracy (controlling other mediators + Age)",
        "cprime":"Direct effect c′: Age→Accuracy (controlling mediators)",
        "ind_total":"Total indirect: ind1+ind2+ind3",
        "ind1":"Indirect via Exposure: a1×b1",
        "ind3":"Indirect via Attitude: a3×b3",
    }

    def print_key_table():
        rows=[]
        for eff in KEY_EFF:
            rows.append({
                "effect": eff,
                "meaning": meaning.get(eff,""),
                "male": fmt_ci(row_of(male_df, eff)),
                "female": fmt_ci(row_of(female_df, eff)),
                "female−male": fmt_ci(row_of(diff_df, eff)),
            })
        table=pd.DataFrame(rows)
        print("Key effects (β; bootstrap 95% CI; * = CI excludes 0):")
        print(table.to_string(index=False))
        print("\nInterpretation notes:")
        print("- 'Female−Male' row shows the sex difference in the effect; * indicates a significant sex difference.")
        print("- If Δind2 is significant while Δb2 is not, the ind2 difference is primarily driven by Δa2.\n")

    print_key_table()

    # =============================================================================
    # 4A) Age-bin × Sex (mean±SEM)
    # =============================================================================
    df = pd.read_csv(DATA_MOBILE, encoding="utf-8-sig")
    df = normalize_sex(df, out_col="sex")
    acc_col = resolve_overall_accuracy_column(df)
    df["accuracy_pct"] = to_percent_series(df[acc_col])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"]>=20) & (df["age"]<=69)].copy()

    dA = df[df["sex"].isin(["male","female"])].copy()
    dA["age_bin"] = dA["age"].apply(age_to_bin)
    dA = dA.dropna(subset=["age_bin","accuracy_pct"]).copy()

    summary = (dA.groupby(["age_bin","sex"])["accuracy_pct"]
               .agg(n="count", mean="mean", std="std")
               .reset_index())
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["age_bin"] = pd.Categorical(summary["age_bin"], categories=AGE_BIN_ORDER, ordered=True)
    summary["sex"] = pd.Categorical(summary["sex"], categories=["male","female"], ordered=True)
    summary = summary.sort_values(["age_bin","sex"])

    summary_out = OUT_DIR / "4A_agebin_sex_summary_mobile.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 10))
    xpos = np.arange(len(AGE_BIN_ORDER))

    for sex in ["male","female"]:
        ss = summary[summary["sex"]==sex].set_index("age_bin").reindex(AGE_BIN_ORDER).reset_index()
        ax.errorbar(
            xpos,
            ss["mean"].values,
            yerr=ss["sem"].values,
            marker="o",
            linewidth=3.0,
            markersize=10,
            capsize=8,
            color=SEX_COLOR[sex],
            markerfacecolor=SEX_COLOR[sex],
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=SEX_LABEL[sex],
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels(AGE_BIN_ORDER, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Age group", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Figure 4A. Accuracy by age group and sex (Mobile; mean ± SEM)", fontweight="bold", pad=18)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # interaction p (you already have p=2.6543e-09)
    ax.text(
        0.98, 0.98, "Age × Sex interaction: p < .001",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Sex")
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    outA_png = OUT_DIR / "4A_agebin_by_sex_mobile.png"
    outA_svg = OUT_DIR / "4A_agebin_by_sex_mobile.svg"
    plt.savefig(outA_png, dpi=300, bbox_inches="tight")
    plt.savefig(outA_svg, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # =============================================================================
    # 4B) Forest plot with group colors + legend + console summary
    # =============================================================================
    effects_to_show = ["ind2","a2","b2","cprime"]
    labels = {
        "ind2": "Indirect via Confidence (ind2)",
        "a2":   "Age → Confidence (a2)",
        "b2":   "Confidence → Accuracy (b2)",
        "cprime": "Direct Age effect (c′)",
    }

    rows = []
    for eff in effects_to_show:
        for grp_name, dff in [("Male", male_df), ("Female", female_df), ("Female−Male", diff_df)]:
            r = row_of(dff, eff)
            rows.append({
                "effect": eff,
                "effect_label": labels[eff],
                "group": grp_name,
                "coef": float(r["coef"]) if r is not None else np.nan,
                "lo": float(r["CI[2.5%]"]) if r is not None else np.nan,
                "hi": float(r["CI[97.5%]"]) if r is not None else np.nan,
                "sig": bool(r["sig_CI_nonzero"]) if r is not None else False,
            })
    forest = pd.DataFrame(rows)

    forest_out = OUT_DIR / "4B_forest_table_mobile.csv"
    forest.to_csv(forest_out, index=False, encoding="utf-8-sig")

    # console print: what 4B shows
    print("4B Forest plot contents (β effects with bootstrap 95% CI):")
    print(forest.to_string(index=False))
    print()

    # plot
    fig, ax = plt.subplots(figsize=(16, 10))

    group_order = ["Male","Female","Female−Male"]
    y_pos = []
    x_vals = []
    xerr = []
    colors = []
    yticks = []
    pos = 0

    for eff in effects_to_show:
        sub = forest[forest["effect"]==eff].set_index("group").reindex(group_order).reset_index()
        for _, r in sub.iterrows():
            y_pos.append(pos)
            x_vals.append(r["coef"])
            xerr.append([r["coef"] - r["lo"], r["hi"] - r["coef"]])
            colors.append(GROUP_COLORS.get(r["group"], "black"))
            yticks.append(f"{labels[eff]} | {r['group']}" + (" *" if r["sig"] else ""))
            pos += 1
        pos += 0.8  # gap between effects

    y_pos = np.array(y_pos)
    x_vals = np.array(x_vals)
    xerr = np.array(xerr).T  # shape (2, N)

    # draw each point individually to control color safely (no IndexError)
    for i in range(len(x_vals)):
        ax.errorbar(
            x_vals[i], y_pos[i],
            xerr=np.array([[xerr[0,i]], [xerr[1,i]]]),
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            capsize=4,
            markersize=8
        )

    ax.axvline(0, linestyle="--", linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticks)
    ax.set_title("Figure 4B. Sex-stratified confidence pathway effects (β; bootstrap 95% CI)", fontweight="bold", pad=18)
    ax.set_xlabel("Standardized effect (β)", fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # legend outside
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker="o", color=GROUP_COLORS["Male"], label="Male", linestyle="None"),
        Line2D([0],[0], marker="o", color=GROUP_COLORS["Female"], label="Female", linestyle="None"),
        Line2D([0],[0], marker="o", color=GROUP_COLORS["Female−Male"], label="Female−Male (Δ)", linestyle="None"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Group")

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    outB_png = OUT_DIR / "4B_forest_sex_diff_mobile.png"
    outB_svg = OUT_DIR / "4B_forest_sex_diff_mobile.svg"
    plt.savefig(outB_png, dpi=300, bbox_inches="tight")
    plt.savefig(outB_svg, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # =============================================================================
    # 4C) Sex-stratified β path diagram (male/female + Δ*)
    # =============================================================================
    def getcoef(df, eff):
        r = row_of(df, eff)
        return (float(r["coef"]) if r is not None else np.nan), (bool(r["sig_CI_nonzero"]) if r is not None else False)

    def getdiffsig(eff):
        r = row_of(diff_df, eff)
        return bool(r["sig_CI_nonzero"]) if r is not None else False

    def label_mf(eff, nd=2):
        cm, sm = getcoef(male_df, eff)
        cf, sf = getcoef(female_df, eff)
        dSig = getdiffsig(eff)
        return f"{cm:.{nd}f}{'*' if sm else ''} / {cf:.{nd}f}{'*' if sf else ''}" + ("  Δ*" if dSig else "")

    g = Digraph("fig4C_sex_path", format="png")
    g.attr(rankdir="LR", bgcolor="white")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#E9F3FF", fontname="Arial", fontsize="14")
    g.attr("edge", fontname="Arial", fontsize="12", color="#1f1f1f")

    g.node("Age", "Age (X)")
    g.node("Exp", "Exposure (M1)")
    g.node("Conf", "Confidence (M2)")
    g.node("Att", "Attitude (M3)")
    g.node("Acc", "Accuracy (Y)")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("Exp"); s.node("Conf"); s.node("Att")

    def edge_style(sig_any, emphasize=False):
        style = "solid" if sig_any else "dashed"
        color = "#1f1f1f" if sig_any else "#9E9E9E"
        pen = "2.2" if sig_any else "1.8"
        if emphasize:
            color = "#C62828"
            pen = "4.0"
        return style, color, pen

    def add_edge(src, dst, eff_key, emphasize=False):
        lm = row_of(male_df, eff_key)
        lf = row_of(female_df, eff_key)
        sig_any = (bool(lm["sig_CI_nonzero"]) if lm is not None else False) or (bool(lf["sig_CI_nonzero"]) if lf is not None else False)
        style, color, pen = edge_style(sig_any, emphasize=emphasize)
        g.edge(src, dst, label=f"β: {label_mf(eff_key)}", style=style, color=color, penwidth=pen)

    # emphasize confidence paths
    add_edge("Age","Exp","a1", emphasize=False)
    add_edge("Age","Conf","a2", emphasize=True)
    add_edge("Age","Att","a3", emphasize=False)
    add_edge("Exp","Acc","b1", emphasize=False)
    add_edge("Conf","Acc","b2", emphasize=True)
    add_edge("Att","Acc","b3", emphasize=False)
    add_edge("Age","Acc","cprime", emphasize=False)

    g.attr("node", shape="note", style="filled", fillcolor="#FFF9E5", fontsize="12", fontname="Arial")
    g.node("note", "Male/Female β shown per path. Δ* indicates significant sex difference (bootstrap 95% CI excludes 0).")

    outC_base = OUT_DIR / "4C_sex_stratified_path_mobile"
    g.format = "png"
    outC_png = g.render(str(outC_base), cleanup=True)
    g.format = "svg"
    outC_svg = g.render(str(outC_base), cleanup=True)

    print("4C path diagram saved:")
    print(" -", outC_png)
    print(" -", outC_svg)
    print()

    # =============================================================================
    # 4D) Δa2 vs Δb2 (female−male) bar with 95% CI + console summary
    # =============================================================================
    def diff_row(eff):
        r = row_of(diff_df, eff)
        return float(r["coef"]), float(r["CI[2.5%]"]), float(r["CI[97.5%]"]), bool(r["sig_CI_nonzero"])

    a2c, a2lo, a2hi, a2sig = diff_row("a2")
    b2c, b2lo, b2hi, b2sig = diff_row("b2")

    print("4D decomposition values (Female−Male):")
    print(f" - Δa2 (Age→Confidence): {a2c:.3f} [{a2lo:.3f}, {a2hi:.3f}] {'*' if a2sig else 'n.s.'}")
    print(f" - Δb2 (Confidence→Accuracy): {b2c:.3f} [{b2lo:.3f}, {b2hi:.3f}] {'*' if b2sig else 'n.s.'}")
    print()

    labelsD = ["Δa2\n(Age→Confidence)", "Δb2\n(Confidence→Accuracy)"]
    vals = [a2c, b2c]
    errs = [[a2c-a2lo, b2c-b2lo], [a2hi-a2c, b2hi-b2c]]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(2)
    ax.bar(x, vals, width=0.45, edgecolor="black", alpha=0.85, color=["#C62828", "#1f1f1f"])
    ax.errorbar(x, vals, yerr=np.array(errs), fmt="none", capsize=8, linewidth=2.5, color="black")
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labelsD, fontweight="bold")
    ax.set_ylabel("Female − Male (β)", fontweight="bold")
    ax.set_title("Figure 4D. Decomposing the ind2 sex difference (Δa2 vs Δb2)", fontweight="bold", pad=18)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax.text(x[0], vals[0] + (0.02 if vals[0]>=0 else -0.05), "*" if a2sig else "n.s.", ha="center", fontweight="bold")
    ax.text(x[1], vals[1] + (0.02 if vals[1]>=0 else -0.05), "*" if b2sig else "n.s.", ha="center", fontweight="bold")

    plt.tight_layout()
    outD_png = OUT_DIR / "4D_delta_a2_vs_b2_mobile.png"
    outD_svg = OUT_DIR / "4D_delta_a2_vs_b2_mobile.svg"
    plt.savefig(outD_png, dpi=300, bbox_inches="tight")
    plt.savefig(outD_svg, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # -----------------------------
    # Meta
    # -----------------------------
    meta = {
        "figure": "Figure 4 (Mobile) - β sex-diff bootstrap",
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "mobile_data": DATA_MOBILE,
            "effects_male": str(EFF_MALE),
            "effects_female": str(EFF_FEMALE),
            "effects_diff": str(EFF_DIFF),
        },
        "outputs": {
            "4A_png": str(outA_png), "4A_svg": str(outA_svg),
            "4B_png": str(outB_png), "4B_svg": str(outB_svg),
            "4C_png": str(outC_png), "4C_svg": str(outC_svg),
            "4D_png": str(outD_png), "4D_svg": str(outD_svg),
            "4A_summary_csv": str(summary_out),
            "4B_table_csv": str(forest_out),
            "4B_table_colored_groups": GROUP_COLORS,
        },
        "notes": [
            "4A shows mean±SEM by age bin and sex (male/female only).",
            "4B shows β estimates with bootstrap 95% CI for male, female, and female−male difference.",
            "4C displays male/female β per path; Δ* from diff table; confidence paths emphasized.",
            "4D compares Δa2 vs Δb2 to explain the ind2 sex difference."
        ]
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("✅ Figure 4 outputs saved to:", OUT_DIR)
    print("✅ meta:", OUT_DIR / "meta.json")


def _run_cell_013():
    # ==============================================================================
    # (4B) Forest plot (READABLE): Male vs Female + Δ(F−M) with 95% CI (β)
    # ------------------------------------------------------------------------------
    # - One row per effect (ind2, a2, b2, cprime)
    # - Three points per row: Male, Female, Δ(F−M)
    # - Δ is visually emphasized; if CI excludes 0, Δ point is highlighted
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RUN_TAG = config.RUN_TAG
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "04_fig4_sex_diff_beta"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    EFF_DIR = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}" / "35_1b_beta_text" / "mobile"
    male_path   = EFF_DIR / "35.1b-1_effects_male_mobile.csv"
    female_path = EFF_DIR / "35.1b-1_effects_female_mobile.csv"
    diff_path   = EFF_DIR / "35.1b-3_effects_diff_female_minus_male_mobile.csv"

    male_df   = pd.read_csv(male_path, encoding="utf-8-sig")
    female_df = pd.read_csv(female_path, encoding="utf-8-sig")
    diff_df   = pd.read_csv(diff_path, encoding="utf-8-sig")

    def row_of(df, eff):
        hit = df[df["effect"] == eff]
        return hit.iloc[0] if len(hit) else None

    def get(df, eff):
        r = row_of(df, eff)
        return {
            "coef": float(r["coef"]),
            "lo": float(r["CI[2.5%]"]),
            "hi": float(r["CI[97.5%]"]),
            "sig": bool(r["sig_CI_nonzero"])
        }

    # ---- choose effects (핵심만) ----
    effects = [
        ("ind2",   "Indirect via Confidence (ind2)"),
        ("a2",     "Age → Confidence (a2)"),
        ("b2",     "Confidence → Accuracy (b2)"),
        ("cprime", "Direct Age effect (c′)"),
    ]

    # colors (consistent)
    COL_M = "#4285F4"  # male
    COL_F = "#DB4437"  # female
    COL_D = "#6A1B9A"  # diff

    # plot settings
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 30,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
    })

    fig, ax = plt.subplots(figsize=(16, 9))

    # y positions (top to bottom)
    y = np.arange(len(effects))[::-1]  # reverse so first item at top

    # x offsets so 3 points don't overlap
    off = 0.22
    xoff = {"Male": -off, "Female": 0.0, "Δ(F−M)": +off}

    # vertical reference line at 0
    ax.axvline(0, linestyle="--", linewidth=2.5, color="gray", alpha=0.8)

    # draw each row
    for i, (eff, label) in enumerate(effects):
        yy = y[i]

        m = get(male_df, eff)
        f = get(female_df, eff)
        d = get(diff_df, eff)  # Δ = Female − Male

        # helper to draw point+CI
        def draw_point(dat, xshift, color, marker="o", ms=10, lw=2.5, z=3, emphasize=False):
            coef, lo, hi, sig = dat["coef"], dat["lo"], dat["hi"], dat["sig"]
            # CI line
            ax.hlines(yy, lo, hi, color=color, linewidth=lw, alpha=0.95, zorder=z)
            # point
            edge = "black" if emphasize else color
            ax.plot(
                coef, yy,
                marker=marker,
                markersize=ms,
                color=color,
                markeredgecolor=edge,
                markeredgewidth=2.0 if emphasize else 1.0,
                zorder=z+1
            )

        # Male/Female: small circles
        # (xshift는 그림 배치용이라 여기서는 y는 그대로, 점을 x축 방향으로는 그대로 두고, 레전드로 구분)
        # 대신 시각적으로 “세 점”처럼 보이게 y를 아주 살짝 분리(미세한 y-jitter)하는 방식이 더 선명함
        yj = 0.12
        draw_point(m, xoff["Male"],   COL_M, marker="o", ms=9,  lw=2.2, z=3, emphasize=False)
        draw_point(f, xoff["Female"], COL_F, marker="o", ms=9,  lw=2.2, z=3, emphasize=False)

        # Δ(F−M): diamond + thicker + (sig이면 강조)
        draw_point(d, xoff["Δ(F−M)"], COL_D, marker="D", ms=11, lw=3.2, z=4, emphasize=d["sig"])

        # place a small label near Δ point: "Δ*" or "Δ n.s."
        delta_tag = "Δ*" if d["sig"] else "Δ n.s."
        ax.text(
            d["hi"] + 0.02, yy,
            delta_tag,
            va="center", ha="left",
            fontsize=18, fontweight="bold",
            color="#C62828" if d["sig"] else "gray"
        )

    # y-axis labels: effect names only (short)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in effects])

    ax.set_xlabel("Standardized effect (β) with 95% CI", fontweight="bold")
    ax.set_title("Figure 4B. Sex differences in confidence pathway effects (Mobile)", fontweight="bold", pad=14)

    # legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker="o", color=COL_M, linestyle="None", markersize=10, label="Male"),
        Line2D([0],[0], marker="o", color=COL_F, linestyle="None", markersize=10, label="Female"),
        Line2D([0],[0], marker="D", color=COL_D, linestyle="None", markersize=11, label="Δ(Female−Male)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=True)

    # explain box inside the figure (self-contained)
    explain = (
        "How to read:\n"
        "• Points = β estimates; lines = bootstrap 95% CI\n"
        "• Δ(F−M) is the sex difference (Female−Male)\n"
        "• If Δ CI does NOT cross 0 ⇒ significant sex difference (Δ*)"
    )
    ax.text(
        0.02, 0.02, explain,
        transform=ax.transAxes,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="gray"),
        fontsize=16
    )

    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    plt.tight_layout()

    png_out = OUT_DIR / "4B_forest_readable.png"
    svg_out = OUT_DIR / "4B_forest_readable.svg"
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", png_out)
    print("✅ saved:", svg_out)


def _run_cell_014():
    # ==============================================================================
    # (Figure 4B-right) Difference-only forest plot: Δ(Female−Male) (β; bootstrap 95% CI)
    # ------------------------------------------------------------------------------
    # Reads: outputs/run_20260119_192624/35_1b_beta_text/mobile/
    #        35.1b-3_effects_diff_female_minus_male_mobile.csv
    # Saves: plots/run_20260119_192624/04_fig4_sex_diff_beta/4B_right_delta_only.png/.svg
    # ==============================================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RUN_TAG = config.RUN_TAG
    DIFF_PATH = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}" / "35_1b_beta_text" / "mobile" / "35.1b-3_effects_diff_female_minus_male_mobile.csv"
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "04_fig4_sex_diff_beta"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DIFF_PATH, encoding="utf-8-sig")

    def pick(effect):
        r = df[df["effect"] == effect].iloc[0]
        return float(r["coef"]), float(r["CI[2.5%]"]), float(r["CI[97.5%]"]), bool(r["sig_CI_nonzero"])

    # 핵심만
    effects = [
        ("ind2",   "Δind2  (Confidence indirect)"),
        ("a2",     "Δa2   (Age→Confidence)"),
        ("b2",     "Δb2   (Confidence→Accuracy)"),
        ("cprime", "Δc′   (Direct Age effect)"),
    ]

    rows = []
    for eff, lab in effects:
        coef, lo, hi, sig = pick(eff)
        rows.append({"label": lab, "coef": coef, "lo": lo, "hi": hi, "sig": sig})

    plot_df = pd.DataFrame(rows)
    # 위에서 아래로 보기 좋게 reverse
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 30,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(plot_df))
    ax.axvline(0, linestyle="--", linewidth=2.0, color="gray")

    for i, r in plot_df.iterrows():
        color = "#C62828" if r["sig"] else "#777777"
        lw = 3.2 if r["sig"] else 2.2
        ax.hlines(i, r["lo"], r["hi"], color=color, linewidth=lw)
        ax.plot(r["coef"], i, marker="D", markersize=10, color=color)

        ax.text(
            r["hi"] + 0.01, i,
            "*" if r["sig"] else "n.s.",
            va="center", ha="left",
            fontweight="bold",
            color=color
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Δ(Female − Male) in standardized effect (β)")
    ax.set_title("Sex differences (Δ) in key mediation paths (Mobile)", fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)

    plt.tight_layout()

    png_out = OUT_DIR / "4B_right_delta_only.png"
    svg_out = OUT_DIR / "4B_right_delta_only.svg"
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", png_out)
    print("✅ saved:", svg_out)


def main():
    _run_cell_010()
    _run_cell_012()
    _run_cell_013()
    _run_cell_014()


if __name__ == "__main__":
    main()
