"""Figure 3 assembly: mobile correlation panels and pooled mediation path diagram."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_007():
    # ==============================================================================
    # (8_plus | Figure 3 | Option B) Mobile: Heatmap + (Confidence/Exposure) Scatter + Mediation Placeholder
    # ------------------------------------------------------------------------------
    # Layout (2x2):
    #   A) Spearman correlation heatmap (Accuracy on top/left, RdBu_r)
    #   B) Confidence vs Accuracy scatter (y fixed 0..100, red regression line)
    #   C) Exposure vs Accuracy scatter (y fixed 0..100, red regression line)
    #   D) Placeholder panel for mediation model (no plotting; reserved space)
    #
    # Save:
    #   plots/run_20260119_192624/03_accuracy_correlations_mobile/
    #     - fig3_mobile_optionB.png / .svg
    #     - fig3_mobile_corr_table.csv
    #     - fig3_mobile_corr_p_table.csv
    #     - fig3_mobile_scatter_raw.csv
    #     - meta.json
    # ==============================================================================

    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from scipy.stats import spearmanr

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    PLOTS_ROOT = config.PLOTS_DIR / f"run_{config.RUN_TAG}"
    SECTION_DIR = PLOTS_ROOT / "03_accuracy_correlations_mobile"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_PATH = config.MOBILE_AGE_FILTERED

    # big text (>=3x)
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

    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_accuracy_col(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns: return "overallAccuracy_y"
        if "overallAccuracy" in df.columns: return "overallAccuracy"
        if "overallAccuracy_x" in df.columns: return "overallAccuracy_x"
        raise KeyError("No overallAccuracy column found.")

    def to_percent_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        finite = s_num.dropna()
        if finite.empty: return s_num
        mx = float(finite.max())
        return s_num * 100.0 if mx <= 1.5 else s_num

    def map_scores(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        def norm(s): return s.astype(str).str.lower().str.strip()

        exposure_map = {"never": 1, "rarely": 2, "sometimes": 3, "weekly": 4, "daily": 5}
        confidence_map = {"very-not-confident": 1, "not-confident": 2, "neutral": 3, "confident": 4, "very-confident": 5}
        attitude_map = {"very-negative": -2, "negative": -1, "neutral": 0, "positive": 1, "very-positive": 2}

        if "exposure_score" not in out.columns:
            out["exposure_score"] = norm(out["aiExposureFrequency"]).map(exposure_map) if "aiExposureFrequency" in out.columns else np.nan
        if "confidence_score" not in out.columns:
            out["confidence_score"] = norm(out["aiConfidence"]).map(confidence_map) if "aiConfidence" in out.columns else np.nan
        if "attitude_score" not in out.columns:
            out["attitude_score"] = norm(out["aiAttitude"]).map(attitude_map) if "aiAttitude" in out.columns else np.nan

        return out

    def spearman_corr_p(df_num: pd.DataFrame):
        cols = df_num.columns.tolist()
        rmat = pd.DataFrame(np.nan, index=cols, columns=cols)
        pmat = pd.DataFrame(np.nan, index=cols, columns=cols)

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    rmat.loc[c1, c2] = 1.0
                    pmat.loc[c1, c2] = np.nan  # ✅ diagonal: no p-value / no stars
                elif i < j:
                    x = df_num[c1].values
                    y = df_num[c2].values
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() < 10:
                        r, p = np.nan, np.nan
                    else:
                        r, p = spearmanr(x[m], y[m])
                    rmat.loc[c1, c2] = r; rmat.loc[c2, c1] = r
                    pmat.loc[c1, c2] = p; pmat.loc[c2, c1] = p
        return rmat, pmat

    def add_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    def jitter_x(x, scale=0.08, seed=42):
        rng = np.random.default_rng(seed)
        return x + rng.normal(0, scale, size=len(x))

    # -----------------------------
    # Load + prep
    # -----------------------------
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    acc_col = resolve_accuracy_col(df)
    df["accuracy_pct"] = to_percent_series(df[acc_col])
    df = map_scores(df)

    for c in ["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Save raw for scatter reproducibility
    scatter_keep = [c for c in ["participantId", "age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"] if c in df.columns]
    scatter_raw = df[scatter_keep].copy()
    scatter_raw_out = SECTION_DIR / "fig3_mobile_scatter_raw.csv"
    scatter_raw.to_csv(scatter_raw_out, index=False, encoding="utf-8-sig")

    # Correlation complete-case for the heatmap
    corr_df = df[["age", "accuracy_pct", "exposure_score", "confidence_score", "attitude_score"]].dropna().copy()
    rmat, pmat = spearman_corr_p(corr_df.rename(columns={
        "age": "Age",
        "accuracy_pct": "Accuracy",
        "exposure_score": "Exposure",
        "confidence_score": "Confidence",
        "attitude_score": "Attitude",
    }))

    # Reorder to put Accuracy first (top/left)
    ordered = ["Accuracy", "Age", "Confidence", "Exposure", "Attitude"]
    rmat = rmat.loc[ordered, ordered]
    pmat = pmat.loc[ordered, ordered]

    # Save correlation tables
    r_out = SECTION_DIR / "fig3_mobile_corr_table.csv"
    p_out = SECTION_DIR / "fig3_mobile_corr_p_table.csv"
    rmat.to_csv(r_out, encoding="utf-8-sig")
    pmat.to_csv(p_out, encoding="utf-8-sig")

    # -----------------------------
    # Figure 3: 2x2 panels (Option B)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    # ---- A) heatmap (RdBu_r, smaller in-cell text, no stars on diagonal) ----
    vals = rmat.values.astype(float)
    im = axA.imshow(vals, vmin=-0.6, vmax=0.6, cmap="RdBu_r")

    axA.set_xticks(range(len(rmat.columns)))
    axA.set_yticks(range(len(rmat.index)))
    axA.set_xticklabels(rmat.columns, rotation=30, ha="right")
    axA.set_yticklabels(rmat.index)
    axA.set_title("A  Spearman correlations (Mobile)", fontweight="bold", pad=12)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            r = vals[i, j]
            p = pmat.values[i, j]

            # diagonal: no stars
            stars = "" if i == j else add_stars(p)
            txt = f"{r:.2f}{stars}"

            axA.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if abs(r) > 0.30 else "black",
                fontsize=BIG * 0.55,  # ✅ slightly smaller to avoid overlap
                fontweight="bold"
            )

    cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)
    cbar.set_label("Spearman ρ", rotation=90)

    # ---- scatter helper ----
    def scatter_panel(ax, xcol, xlab, panel_letter, seed):
        dd = df[[xcol, "accuracy_pct"]].dropna().copy()
        x = dd[xcol].values.astype(float)
        y = dd["accuracy_pct"].values.astype(float)

        # jitter x for ordinal (1..5) to see density
        xj = jitter_x(x, scale=0.08, seed=seed)

        ax.scatter(xj, y, s=25, alpha=0.18)

        # red linear fit line
        if len(dd) >= 10:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(np.min(x), np.max(x), 100)
            yy = coef[0]*xx + coef[1]
            ax.plot(xx, yy, color="#e53935", linewidth=4)

            # Spearman rho
            rho, p = spearmanr(x, y)
            ptxt = "p < .001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(
                0.02, 0.02,
                f"ρ = {rho:.2f}, {ptxt}",
                transform=ax.transAxes,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
                fontsize=BIG * 0.65,
                fontweight="bold"
            )

        ax.set_ylim(0, 100)
        ax.set_xlabel(xlab, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontweight="bold")
        ax.set_title(f"{panel_letter}  {xlab} vs Accuracy (Mobile)", fontweight="bold", pad=12)

    # ---- B/C scatters only (Attitude scatter dropped) ----
    scatter_panel(axB, "confidence_score", "Confidence (1–5)", "B", seed=1)
    scatter_panel(axC, "exposure_score",   "Exposure (1–5)",   "C", seed=2)

    # ---- D) placeholder for mediation model (no plotting here) ----
    axD.axis("off")
    axD.set_title("D  Parallel mediation model (Mobile)", fontweight="bold", pad=12)

    placeholder_text = (
        "PLACEHOLDER\n\n"
        "Insert path diagram here:\n"
        "Age → (Exposure, Confidence, Attitude) → Accuracy\n"
        "Direct path: Age → Accuracy\n\n"
        "Sex differences will be presented in Figure 4."
    )

    axD.text(
        0.5, 0.5, placeholder_text,
        ha="center", va="center",
        fontsize=BIG * 0.75,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF9E5", edgecolor="gray", alpha=0.95)
    )

    # border (optional)
    rect = plt.Rectangle((0.05, 0.08), 0.90, 0.80, transform=axD.transAxes,
                         fill=False, linewidth=2.0, edgecolor="gray")
    axD.add_patch(rect)

    # overall title
    fig.suptitle("Figure 3. Accuracy and AI-related factors (Mobile cohort)", fontweight="bold", y=0.995)

    # layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_out = SECTION_DIR / "fig3_mobile_optionB.png"
    svg_out = SECTION_DIR / "fig3_mobile_optionB.svg"
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # meta
    meta = {
        "figure": "Figure 3 (Mobile) - Option B",
        "created_at": datetime.now().isoformat(),
        "input": INPUT_PATH,
        "accuracy_col_used": acc_col,
        "n_complete_corr": int(len(corr_df)),
        "outputs": {
            "png": str(png_out),
            "svg": str(svg_out),
            "corr_r_csv": str(r_out),
            "corr_p_csv": str(p_out),
            "scatter_raw_csv": str(scatter_raw_out),
        },
        "notes": [
            "Spearman correlations used (ordinal scores).",
            "Heatmap uses RdBu_r diverging colormap; Accuracy is first row/column.",
            "Diagonal cells have no stars (not a tested correlation).",
            "Scatter panels show jittered x for ordinal scores; y fixed 0..100; red linear fit line.",
            "Panel D is reserved for mediation model; sex differences will be shown in Figure 4."
        ]
    }
    (SECTION_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("✅ saved figure:", png_out)
    print("✅ saved meta:", SECTION_DIR / "meta.json")


def _run_cell_008():
    # ==============================================================================
    # (36P) Mobile pooled path diagram (no sex split) for Figure 3 Panel D
    # ------------------------------------------------------------------------------
    # - Reads pooled (all participants) mediation effects:
    #   Prefer standardized beta output:
    #     outputs/run_20260119_192624/35_1b_beta_text/mobile/35.1b-1_effects_all_mobile.csv
    #   Fallback to non-beta (if exists):
    #     outputs/run_20260119_192624/35_1_sex_diff_text/mobile/35.1-1_effects_all_mobile.csv  (or similar)
    #
    # - Draws: Age -> (Exposure, Confidence, Attitude) -> Accuracy + direct c'
    # - Styling:
    #     * Confidence paths (a2,b2) emphasized
    #     * Non-significant paths dashed + gray
    # - Saves PNG+SVG to:
    #     plots/run_20260119_192624/03_accuracy_correlations_mobile/fig3_panelD_path_pooled.png/.svg
    # ==============================================================================

    import pandas as pd
    from pathlib import Path
    from graphviz import Digraph

    RUN_TAG = config.RUN_TAG

    RUN_DIR_OUT = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}"
    PLOTS_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "03_accuracy_correlations_mobile"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- auto-find pooled effect csv ----------
    def find_first_existing(paths):
        for p in paths:
            if p.exists():
                return p
        return None

    beta_candidates = [
        RUN_DIR_OUT / "35_1b_beta_text" / "mobile" / "35.1b-1_effects_all_mobile.csv",
    ]
    # fallback (if you ever saved pooled non-beta)
    nonbeta_candidates = [
        RUN_DIR_OUT / "35_1_sex_diff_text" / "mobile" / "35.1-1_effects_all_mobile.csv",
        RUN_DIR_OUT / "35_1_sex_diff_text_console" / "mobile" / "35.1-1_effects_all_mobile.csv",
    ]

    effects_path = find_first_existing(beta_candidates) or find_first_existing(nonbeta_candidates)
    if effects_path is None:
        raise FileNotFoundError("Could not find pooled effects CSV for mobile. "
                                "Expected beta at outputs/run_20260119_192624/35_1b_beta_text/mobile/35.1b-1_effects_all_mobile.csv")

    df = pd.read_csv(effects_path, encoding="utf-8-sig")
    req = {"effect","coef","sig_CI_nonzero"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {effects_path}: {missing}")

    def get(effect):
        hit = df[df["effect"] == effect]
        return hit.iloc[0] if len(hit) else None

    def fmt(effect, nd=2):
        r = get(effect)
        if r is None:
            return "NA", False
        coef = float(r["coef"])
        sig = bool(r.get("sig_CI_nonzero", False))
        return f"{coef:.{nd}f}{'*' if sig else ''}", sig

    # pull paths
    a1, a1_sig = fmt("a1")
    a2, a2_sig = fmt("a2")
    a3, a3_sig = fmt("a3")
    b1, b1_sig = fmt("b1")
    b2, b2_sig = fmt("b2")
    b3, b3_sig = fmt("b3")
    cp, cp_sig = fmt("cprime")

    ind1, _ = fmt("ind1")
    ind2, _ = fmt("ind2")
    ind3, _ = fmt("ind3")
    indt, _ = fmt("ind_total")

    # label prefix
    is_beta = "35_1b_beta_text" in str(effects_path)
    coef_tag = "β" if is_beta else "B"

    # ---------- Graphviz diagram ----------
    g = Digraph("fig3_panelD_pooled", format="png")
    g.attr(rankdir="LR", bgcolor="white")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#E9F3FF",
           fontname="Arial", fontsize="14", color="#1f1f1f", penwidth="1.6")
    g.attr("edge", fontname="Arial", fontsize="12", color="#1f1f1f")

    # nodes
    g.node("Age", "Age (X)")
    g.node("Exp", "AI Exposure (M1)")
    g.node("Conf", "AI Confidence (M2)")
    g.node("Att", "AI Attitude (M3)")
    g.node("Acc", "Accuracy (Y)")

    # force mediators column
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("Exp"); s.node("Conf"); s.node("Att")

    def edge_style(sig, emphasize=False):
        # significant -> solid, else dashed + gray
        style = "solid" if sig else "dashed"
        color = "#1f1f1f" if sig else "#9E9E9E"
        pen = "2.2" if sig else "1.8"
        if emphasize:
            # emphasize confidence paths
            color = "#C62828" if sig else "#6A1B9A"
            pen = "4.0" if sig else "3.0"
        return style, color, pen

    def add_edge(src, dst, label, sig, emphasize=False):
        style, color, pen = edge_style(sig, emphasize=emphasize)
        g.edge(src, dst, label=label, style=style, color=color, penwidth=pen)

    # edges (confidence emphasized)
    add_edge("Age", "Exp",  f"{coef_tag}: {a1}", a1_sig, emphasize=False)
    add_edge("Age", "Conf", f"{coef_tag}: {a2}", a2_sig, emphasize=True)
    add_edge("Age", "Att",  f"{coef_tag}: {a3}", a3_sig, emphasize=False)

    add_edge("Exp",  "Acc", f"{coef_tag}: {b1}", b1_sig, emphasize=False)
    add_edge("Conf", "Acc", f"{coef_tag}: {b2}", b2_sig, emphasize=True)
    add_edge("Att",  "Acc", f"{coef_tag}: {b3}", b3_sig, emphasize=False)

    add_edge("Age", "Acc", f"{coef_tag} (c'): {cp}", cp_sig, emphasize=False)

    # summary note (inside diagram)
    g.attr("node", shape="note", style="filled", fillcolor="#FFF9E5", fontsize="12", fontname="Arial")
    summary = (
        f"Pooled (Mobile) parallel mediation\n"
        f"{coef_tag} coefficients shown (* = 95% CI excludes 0)\n\n"
        f"Indirect via Exposure:   {ind1}\n"
        f"Indirect via Confidence: {ind2}\n"
        f"Indirect via Attitude:   {ind3}\n"
        f"Indirect TOTAL:          {indt}"
    )
    g.node("summary", summary)

    out_base = PLOTS_DIR / "fig3_panelD_path_pooled"
    g.format = "png"
    png_path = g.render(str(out_base), cleanup=True)
    g.format = "svg"
    svg_path = g.render(str(out_base), cleanup=True)

    print("✅ pooled path diagram saved:")
    print(" -", png_path)
    print(" -", svg_path)
    print("ℹ️ source effects file:", effects_path)


def main():
    _run_cell_007()
    _run_cell_008()


if __name__ == "__main__":
    main()
