"""Supplementary figure assembly for S0-S6."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_047():
    # ==============================================================================
    # Fig. S1 (PC) - AI self-reports vs Accuracy (PC cohort replication of Fig3 A–C)
    # Outputs:
    #   plots/run_20260119_192624/supp/figS1_pc_ai_factors/
    #     - figS1.png/.svg  (3-panel)
    #     - figS1_corr_table.csv  (Spearman r + p)
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import spearmanr

    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT
    IN_FP = config.WEB_AGE_FILTERED

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS1_pc_ai_factors"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- big fonts (match your paper style) ----
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    # ---- helpers ----
    def resolve_acc_col(df):
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns:
                return c
        raise KeyError("No overallAccuracy column found.")

    def to_percent(s):
        x = pd.to_numeric(s, errors="coerce")
        finite = x.dropna()
        if finite.empty:
            return x
        return x*100.0 if float(finite.max()) <= 1.5 else x

    def map_scores(df):
        # ordinal mappings
        exposure_map = {"never":1, "rarely":2, "sometimes":3, "weekly":4, "daily":5}
        conf_map = {"very-not-confident":1, "not-confident":2, "neutral":3, "confident":4, "very-confident":5}
        att_map = {"very-negative":-2, "negative":-1, "neutral":0, "positive":1, "very-positive":2}

        out = df.copy()
        out["exposure_score"] = out["aiExposureFrequency"].astype(str).str.lower().str.strip().map(exposure_map) if "aiExposureFrequency" in out.columns else np.nan
        out["confidence_score"] = out["aiConfidence"].astype(str).str.lower().str.strip().map(conf_map) if "aiConfidence" in out.columns else np.nan
        out["attitude_score"] = out["aiAttitude"].astype(str).str.lower().str.strip().map(att_map) if "aiAttitude" in out.columns else np.nan
        return out

    def stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    # ---- load + prep ----
    df = pd.read_csv(IN_FP, encoding="utf-8-sig")
    acc_col = resolve_acc_col(df)
    df["accuracy_pct"] = to_percent(df[acc_col])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = map_scores(df)

    # variables for S1
    vars_order = ["Accuracy", "Age", "Confidence", "Exposure", "Attitude"]
    var_cols = {
        "Accuracy": "accuracy_pct",
        "Age": "age",
        "Confidence": "confidence_score",
        "Exposure": "exposure_score",
        "Attitude": "attitude_score",
    }

    # complete-case for correlation matrix
    corr_df = df[[var_cols[v] for v in vars_order]].rename(columns={var_cols[v]: v for v in vars_order})
    corr_df = corr_df.dropna().copy()

    # compute Spearman r and p matrices
    R = pd.DataFrame(np.eye(len(vars_order)), index=vars_order, columns=vars_order)
    P = pd.DataFrame(np.zeros((len(vars_order), len(vars_order))), index=vars_order, columns=vars_order)

    for i, a in enumerate(vars_order):
        for j, b in enumerate(vars_order):
            if i == j:
                R.loc[a,b] = 1.0
                P.loc[a,b] = np.nan  # diagonal p not meaningful
            elif i < j:
                r, p = spearmanr(corr_df[a].values, corr_df[b].values)
                R.loc[a,b] = r; R.loc[b,a] = r
                P.loc[a,b] = p; P.loc[b,a] = p

    # save correlation table (long)
    rows = []
    for a in vars_order:
        for b in vars_order:
            if a == b: 
                continue
            if (vars_order.index(a) < vars_order.index(b)):
                rows.append({"var1": a, "var2": b, "spearman_r": float(R.loc[a,b]), "p": float(P.loc[a,b])})
    corr_long = pd.DataFrame(rows)
    corr_long.to_csv(OUT_DIR / "figS1_corr_table.csv", index=False, encoding="utf-8-sig")

    # ---- plot: 3 panels (A heatmap, B confidence scatter, C exposure scatter) ----
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])

    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # A) heatmap (RdBu-like, centered)
    vals = R.values.astype(float)
    im = axA.imshow(vals, vmin=-0.6, vmax=0.6, cmap="RdBu_r")
    axA.set_xticks(range(len(vars_order)))
    axA.set_yticks(range(len(vars_order)))
    axA.set_xticklabels(vars_order, rotation=30, ha="right")
    axA.set_yticklabels(vars_order)
    axA.set_title("A  Spearman correlations (PC cohort)", fontweight="bold", pad=12)

    # annotate numbers (slightly smaller to avoid overlap)
    for i in range(len(vars_order)):
        for j in range(len(vars_order)):
            r = R.iloc[i,j]
            p = P.iloc[i,j]
            # no stars on diagonal
            st = "" if i == j else stars(p)
            txt = f"{r:.2f}{st}"
            axA.text(j, i, txt, ha="center", va="center",
                     color="white" if abs(r) > 0.35 else "black",
                     fontsize=BIG*0.55, fontweight="bold")

    cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontweight="bold")

    # B) Confidence vs Accuracy (scatter + fit)
    dB = df[["confidence_score","accuracy_pct"]].dropna().copy()
    # jitter confidence (ordinal)
    rng = np.random.default_rng(42)
    xj = dB["confidence_score"].values + rng.uniform(-0.12, 0.12, size=len(dB))
    axB.scatter(xj, dB["accuracy_pct"].values, s=18, alpha=0.20, edgecolors="none")
    # fit line on non-jittered x
    b1, b0 = np.polyfit(dB["confidence_score"].values, dB["accuracy_pct"].values, 1)
    xx = np.linspace(1, 5, 200)
    axB.plot(xx, b1*xx + b0, linewidth=3, color="#e53935")
    r_cb, p_cb = spearmanr(dB["confidence_score"].values, dB["accuracy_pct"].values)
    axB.set_title("B  Confidence vs Accuracy (PC)", fontweight="bold", pad=10)
    axB.set_xlabel("Confidence (1–5)", fontweight="bold")
    axB.set_ylabel("Accuracy (%)", fontweight="bold")
    axB.set_ylim(0, 100)
    axB.grid(True, linestyle=":", alpha=0.35)
    axB.text(0.02, 0.02, f"ρ = {r_cb:.2f}, {('p < .001' if p_cb<0.001 else f'p = {p_cb:.3f}')}",
             transform=axB.transAxes, ha="left", va="bottom",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontweight="bold")

    # C) Exposure vs Accuracy (scatter + fit)
    dC = df[["exposure_score","accuracy_pct"]].dropna().copy()
    xj2 = dC["exposure_score"].values + rng.uniform(-0.12, 0.12, size=len(dC))
    axC.scatter(xj2, dC["accuracy_pct"].values, s=18, alpha=0.20, edgecolors="none")
    b1, b0 = np.polyfit(dC["exposure_score"].values, dC["accuracy_pct"].values, 1)
    xx = np.linspace(1, 5, 200)
    axC.plot(xx, b1*xx + b0, linewidth=3, color="#e53935")
    r_eb, p_eb = spearmanr(dC["exposure_score"].values, dC["accuracy_pct"].values)
    axC.set_title("C  Exposure vs Accuracy (PC)", fontweight="bold", pad=10)
    axC.set_xlabel("Exposure (1–5)", fontweight="bold")
    axC.set_ylabel("Accuracy (%)", fontweight="bold")
    axC.set_ylim(0, 100)
    axC.grid(True, linestyle=":", alpha=0.35)
    axC.text(0.02, 0.02, f"ρ = {r_eb:.2f}, {('p < .001' if p_eb<0.001 else f'p = {p_eb:.3f}')}",
             transform=axC.transAxes, ha="left", va="bottom",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontweight="bold")

    fig.tight_layout()
    png = OUT_DIR / "figS1_pc_ai_selfreports.png"
    svg = OUT_DIR / "figS1_pc_ai_selfreports.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", png)
    print("✅ corr N complete-case:", len(corr_df))


def _run_cell_049():
    # ==============================================================================
    # Figure S1 (PC): Correlations + Self-reports scatters + Parallel Mediation (Graphviz)
    # ------------------------------------------------------------------------------
    # Panels (2x2):
    #   A) Spearman heatmap (Accuracy-centered)
    #   B) Confidence vs Accuracy (PC)
    #   C) Exposure vs Accuracy (PC)
    #   D) Parallel mediation model (Age -> Exposure/Confidence/Attitude -> Accuracy) [Graphviz]
    #
    # Inputs:
    #   - analysis_data_web_age_filtered_20_69.csv
    #   - outputs/run_20260119_192624/35_1b_beta_text/web/35.1b-1_effects_all_web.csv
    #
    # Outputs:
    #   - plots/run_20260119_192624/supp/figS1_pc_ai_selfreports/figS1_pc.png/.svg
    #   - (D standalone) .../figS1D_pc_mediation_graphviz.png/.svg
    #   - figS1_corr_table.csv (long)
    #   - meta.json
    # ==============================================================================
    import os
    import json
    import re
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
    BASE_DIR = config.PROJECT_ROOT
    IN_FP = config.WEB_AGE_FILTERED

    RUN_DIR = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}"
    EFF_CSV = RUN_DIR / "35_1b_beta_text" / "web" / "35.1b-1_effects_all_web.csv"

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS1_pc_ai_selfreports"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Big fonts (paper style)
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    # -----------------------------
    # Helpers
    # -----------------------------
    def safe_graphviz_import():
        try:
            import graphviz  # noqa
            return True
        except Exception:
            return False

    def resolve_acc_col(df):
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns:
                return c
        raise KeyError("No overallAccuracy column found.")

    def to_percent(s):
        x = pd.to_numeric(s, errors="coerce")
        finite = x.dropna()
        if finite.empty:
            return x
        return x * 100.0 if float(finite.max()) <= 1.5 else x

    def map_scores(df):
        exposure_map = {"never":1, "rarely":2, "sometimes":3, "weekly":4, "daily":5}
        conf_map = {"very-not-confident":1, "not-confident":2, "neutral":3, "confident":4, "very-confident":5}
        att_map = {"very-negative":-2, "negative":-1, "neutral":0, "positive":1, "very-positive":2}

        out = df.copy()
        out["exposure_score"] = out["aiExposureFrequency"].astype(str).str.lower().str.strip().map(exposure_map) if "aiExposureFrequency" in out.columns else np.nan
        out["confidence_score"] = out["aiConfidence"].astype(str).str.lower().str.strip().map(conf_map) if "aiConfidence" in out.columns else np.nan
        out["attitude_score"] = out["aiAttitude"].astype(str).str.lower().str.strip().map(att_map) if "aiAttitude" in out.columns else np.nan
        return out

    def stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    def fmt_coef(coef, sig):
        return f"{coef:+.2f}{'*' if sig else ''}"

    def get_eff_row(eff_df, effect):
        hit = eff_df[eff_df["effect"] == effect]
        return None if hit.empty else hit.iloc[0]

    def coef_sig_from_row(row):
        if row is None:
            return np.nan, False
        coef = float(row["coef"])
        sig = bool(row.get("sig_CI_nonzero", False))
        return coef, sig

    def render_graphviz_mediation(eff_df, out_base: Path):
        """
        Render Graphviz mediation diagram for PC (pooled) using standardized β effects.
        Saves PNG + SVG and returns paths.
        """
        if not safe_graphviz_import():
            raise RuntimeError("graphviz python package not available (python-graphviz).")

        import graphviz

        # effects we need
        a1 = coef_sig_from_row(get_eff_row(eff_df, "a1"))
        a2 = coef_sig_from_row(get_eff_row(eff_df, "a2"))
        a3 = coef_sig_from_row(get_eff_row(eff_df, "a3"))
        b1 = coef_sig_from_row(get_eff_row(eff_df, "b1"))
        b2 = coef_sig_from_row(get_eff_row(eff_df, "b2"))
        b3 = coef_sig_from_row(get_eff_row(eff_df, "b3"))
        cprime = coef_sig_from_row(get_eff_row(eff_df, "cprime"))

        ind1 = coef_sig_from_row(get_eff_row(eff_df, "ind1"))
        ind2 = coef_sig_from_row(get_eff_row(eff_df, "ind2"))
        ind3 = coef_sig_from_row(get_eff_row(eff_df, "ind3"))
        indt = coef_sig_from_row(get_eff_row(eff_df, "ind_total"))

        g = graphviz.Digraph(name="figS1_pc_mediation")
        g.attr(rankdir="LR", bgcolor="white")

        # Stable font / avoid Unicode arrows
        g.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue",
               fontname="Arial", fontsize="14")
        g.attr("edge", fontname="Arial", fontsize="12", color="black")

        # nodes
        g.node("Age", "Age (X)")
        g.node("Exposure", "AI Exposure (M1)")
        g.node("Confidence", "AI Confidence (M2)")
        g.node("Attitude", "AI Attitude (M3)")
        g.node("Accuracy", "Accuracy (Y)")

        # put mediators at same rank
        with g.subgraph() as s:
            s.attr(rank="same")
            s.node("Exposure")
            s.node("Confidence")
            s.node("Attitude")

        def edge(src, dst, key, human_label, coef_sig_tuple, highlight=False):
            coef, sig = coef_sig_tuple
            label = f"{human_label}\\nβ={coef:+.2f}{'*' if sig else ''}"
            style = "solid" if sig else "dashed"
            color = "#6A1B9A" if highlight else "black"
            penwidth = "3.2" if highlight else "2.0"
            g.edge(src, dst, label=label, style=style, color=color, penwidth=penwidth)

        # a paths
        edge("Age", "Exposure", "a1", "Age -> Exposure (a1)", a1, highlight=False)
        edge("Age", "Confidence", "a2", "Age -> Confidence (a2)", a2, highlight=True)
        edge("Age", "Attitude", "a3", "Age -> Attitude (a3)", a3, highlight=False)

        # b paths
        edge("Exposure", "Accuracy", "b1", "Exposure -> Accuracy (b1)", b1, highlight=False)
        edge("Confidence", "Accuracy", "b2", "Confidence -> Accuracy (b2)", b2, highlight=True)
        edge("Attitude", "Accuracy", "b3", "Attitude -> Accuracy (b3)", b3, highlight=False)

        # direct
        edge("Age", "Accuracy", "cprime", "Age -> Accuracy (c')", cprime, highlight=False)

        # summary note (TOP-LEFT to avoid overlap with Attitude)
        g.attr("node", shape="note", style="filled", fillcolor="#FFF9E5", fontsize="12", fontname="Arial")
        box_txt = (
            "Indirect effects (β)\\n"
            f"via Exposure (ind1): {fmt_coef(ind1[0], ind1[1])}\\n"
            f"via Confidence (ind2): {fmt_coef(ind2[0], ind2[1])}\\n"
            f"via Attitude (ind3): {fmt_coef(ind3[0], ind3[1])}\\n"
            f"TOTAL (ind_total): {fmt_coef(indt[0], indt[1])}\\n"
            "\\n* = bootstrap 95% CI excludes 0"
        )
        g.node("summary", box_txt)

        # render
        g.format = "png"
        png_path = g.render(str(out_base), cleanup=True)
        g.format = "svg"
        svg_path = g.render(str(out_base), cleanup=True)

        return Path(png_path), Path(svg_path)

    # -----------------------------
    # Load inputs
    # -----------------------------
    if not IN_FP.exists():
        raise FileNotFoundError(f"Missing input: {IN_FP}")
    if not EFF_CSV.exists():
        raise FileNotFoundError(f"Missing mediation effects CSV: {EFF_CSV}")

    df = pd.read_csv(IN_FP, encoding="utf-8-sig")
    acc_col = resolve_acc_col(df)
    df["accuracy_pct"] = to_percent(df[acc_col])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = map_scores(df)

    # correlation vars
    vars_order = ["Accuracy", "Age", "Confidence", "Exposure", "Attitude"]
    corr_df = df[["accuracy_pct","age","confidence_score","exposure_score","attitude_score"]].copy()
    corr_df.columns = vars_order
    corr_df = corr_df.dropna().copy()

    # Spearman matrices
    R = pd.DataFrame(np.eye(len(vars_order)), index=vars_order, columns=vars_order)
    P = pd.DataFrame(np.zeros((len(vars_order), len(vars_order))), index=vars_order, columns=vars_order)

    for i, a in enumerate(vars_order):
        for j, b in enumerate(vars_order):
            if i == j:
                R.loc[a,b] = 1.0
                P.loc[a,b] = np.nan
            elif i < j:
                r, p = spearmanr(corr_df[a].values, corr_df[b].values)
                R.loc[a,b] = r; R.loc[b,a] = r
                P.loc[a,b] = p; P.loc[b,a] = p

    # Save corr table (long)
    rows = []
    for i in range(len(vars_order)):
        for j in range(i+1, len(vars_order)):
            a, b = vars_order[i], vars_order[j]
            rows.append({"var1": a, "var2": b, "spearman_r": float(R.loc[a,b]), "p": float(P.loc[a,b])})
    pd.DataFrame(rows).to_csv(OUT_DIR / "figS1_corr_table.csv", index=False, encoding="utf-8-sig")

    # Load mediation effects (β + bootstrap CI)
    eff = pd.read_csv(EFF_CSV, encoding="utf-8-sig")

    # Render D (standalone) via graphviz
    out_base = OUT_DIR / "figS1D_pc_mediation_graphviz"
    pngD, svgD = render_graphviz_mediation(eff, out_base)

    # -----------------------------
    # Plot S1 (2×2)
    # -----------------------------
    fig = plt.figure(figsize=(26, 16))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])

    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])
    axC = fig.add_subplot(gs[1,1])
    axD = fig.add_subplot(gs[1,0])

    # A) Heatmap (increase cell text by 1.5x)
    im = axA.imshow(R.values.astype(float), vmin=-0.6, vmax=0.6, cmap="RdBu_r")
    axA.set_xticks(range(len(vars_order)))
    axA.set_yticks(range(len(vars_order)))
    axA.set_xticklabels(vars_order, rotation=30, ha="right")
    axA.set_yticklabels(vars_order)
    axA.set_title("A  Spearman correlations (PC cohort)", fontweight="bold", pad=12)

    for i in range(len(vars_order)):
        for j in range(len(vars_order)):
            r = R.iloc[i,j]
            p = P.iloc[i,j]
            st = "" if i == j else stars(p)  # no stars on diagonal
            txt = f"{r:.2f}{st}"
            axA.text(j, i, txt, ha="center", va="center",
                     color="white" if abs(r) > 0.35 else "black",
                     fontsize=BIG*0.7,  # ✅ 1.5x bigger than ~0.52
                     fontweight="bold")

    cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontweight="bold")

    # B) Confidence scatter
    dB = df[["confidence_score","accuracy_pct"]].dropna().copy()
    rng = np.random.default_rng(42)
    xj = dB["confidence_score"].values + rng.uniform(-0.12, 0.12, size=len(dB))
    axB.scatter(xj, dB["accuracy_pct"].values, s=18, alpha=0.20, edgecolors="none")
    b1, b0 = np.polyfit(dB["confidence_score"].values, dB["accuracy_pct"].values, 1)
    xx = np.linspace(1, 5, 200)
    axB.plot(xx, b1*xx + b0, linewidth=3, color="#e53935")
    r_cb, p_cb = spearmanr(dB["confidence_score"].values, dB["accuracy_pct"].values)
    axB.set_title("B  Confidence vs Accuracy (PC)", fontweight="bold", pad=10)
    axB.set_xlabel("Confidence (1–5)", fontweight="bold")
    axB.set_ylabel("Accuracy (%)", fontweight="bold")
    axB.set_ylim(0, 100)
    axB.grid(True, linestyle=":", alpha=0.35)
    axB.text(0.02, 0.02, f"ρ = {r_cb:.2f}, {('p < .001' if p_cb<0.001 else f'p = {p_cb:.3f}')}",
             transform=axB.transAxes, ha="left", va="bottom",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontweight="bold")

    # C) Exposure scatter
    dC = df[["exposure_score","accuracy_pct"]].dropna().copy()
    xj2 = dC["exposure_score"].values + rng.uniform(-0.12, 0.12, size=len(dC))
    axC.scatter(xj2, dC["accuracy_pct"].values, s=18, alpha=0.20, edgecolors="none")
    b1, b0 = np.polyfit(dC["exposure_score"].values, dC["accuracy_pct"].values, 1)
    xx = np.linspace(1, 5, 200)
    axC.plot(xx, b1*xx + b0, linewidth=3, color="#e53935")
    r_eb, p_eb = spearmanr(dC["exposure_score"].values, dC["accuracy_pct"].values)
    axC.set_title("C  Exposure vs Accuracy (PC)", fontweight="bold", pad=10)
    axC.set_xlabel("Exposure (1–5)", fontweight="bold")
    axC.set_ylabel("Accuracy (%)", fontweight="bold")
    axC.set_ylim(0, 100)
    axC.grid(True, linestyle=":", alpha=0.35)
    axC.text(0.02, 0.02, f"ρ = {r_eb:.2f}, {('p < .001' if p_eb<0.001 else f'p = {p_eb:.3f}')}",
             transform=axC.transAxes, ha="left", va="bottom",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontweight="bold")

    # D) Graphviz diagram image (PNG) embedded
    axD.axis("off")
    img = plt.imread(str(pngD))
    axD.imshow(img)
    axD.set_title("D  Parallel mediation model (PC; Graphviz)", fontweight="bold", pad=10)

    fig.tight_layout()

    png = OUT_DIR / "figS1_pc_ai_selfreports_mediation_graphviz.png"
    svg = OUT_DIR / "figS1_pc_ai_selfreports_mediation_graphviz.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    meta = {
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "pc_data": str(IN_FP),
            "mediation_effects_csv": str(EFF_CSV),
        },
        "outputs": {
            "figS1_png": str(png),
            "figS1_svg": str(svg),
            "figS1D_graphviz_png": str(pngD),
            "figS1D_graphviz_svg": str(svgD),
            "corr_table": str(OUT_DIR / "figS1_corr_table.csv")
        },
        "notes": [
            "Panel D rendered by Graphviz (PNG+SVG). Panel D embedded in FigS1 via PNG raster."
        ]
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("✅ saved:", png)
    print("✅ D standalone:", pngD, svgD)
    print("✅ corr complete-case N:", len(corr_df))


def _run_cell_051():
    # ==============================================================================
    # Figure S2: PC replication of the unified human-factors model (Fig6 PC version)
    # Panels:
    #   A) PC standardized beta forest (from figS6A_pc_std_beta.png)
    #   B) PC nested R² (from figS6B_pc_nested_R2.png)
    # Output:
    #   plots/run_20260119_192624/supp/figS2_pc_human_factors/figS2_pc.png/.svg
    # ==============================================================================
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path

    RUN_TAG = config.RUN_TAG
    BASE = config.PLOTS_DIR / f"run_{config.RUN_TAG}"
    SRC_DIR = BASE / "06_human_factors_model"

    IN_A = SRC_DIR / "figS6A_pc_std_beta.png"
    IN_B = SRC_DIR / "figS6B_pc_nested_R2.png"

    OUT_DIR = BASE / "supp" / "figS2_pc_human_factors"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_PNG = OUT_DIR / "figS2_pc.png"
    OUT_SVG = OUT_DIR / "figS2_pc.svg"

    for p in [IN_A, IN_B]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    # load images
    imgA = mpimg.imread(str(IN_A))
    imgB = mpimg.imread(str(IN_B))

    # canvas
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    axA.imshow(imgA)
    axA.axis("off")
    axA.text(0.01, 0.98, "A", transform=axA.transAxes,
             ha="left", va="top", fontsize=40, fontweight="bold")

    axB.imshow(imgB)
    axB.axis("off")
    axB.text(0.01, 0.98, "B", transform=axB.transAxes,
             ha="left", va="top", fontsize=40, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", OUT_PNG)
    print("✅ saved:", OUT_SVG)
    print("Inputs:", IN_A.name, IN_B.name)


def _run_cell_053():
    # ==============================================================================
    # Figure S2 (PC): Sex differences replication of Fig4 (PC cohort)
    # ------------------------------------------------------------------------------
    # Panels:
    #   A) PC Age-bin × Sex accuracy (mean±SEM) + 2-way ANOVA p-values
    #   B) PC sex-stratified parallel mediation path diagram (Graphviz)
    # Output (combined):
    #   plots/run_20260119_192624/supp/figS2_pc_sex_diff/figS2_pc.png/.svg
    # Also saves:
    #   - S2A standalone png/svg
    #   - S2B graphviz png/svg
    # ==============================================================================
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path
    from datetime import datetime
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT

    PC_FP = config.WEB_AGE_FILTERED

    # 35.1β outputs (preferred) — auto-discovery fallback included
    RUN_DIR = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}"
    BETA_DIR = RUN_DIR / "35_1b_beta_text" / "web"

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS2_pc_sex_diff"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    AGE_BINS   = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    # paper fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    SEX_COLORS = {"male": "#4285F4", "female": "#DB4437"}

    # -----------------------------
    # Helpers
    # -----------------------------
    def resolve_acc_col(df: pd.DataFrame) -> str:
        for c in ["overallAccuracy_y", "overallAccuracy", "overallAccuracy_x"]:
            if c in df.columns:
                return c
        raise KeyError("No overallAccuracy column found.")

    def to_percent(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        finite = x.dropna()
        if finite.empty:
            return x
        return x * 100.0 if float(finite.max()) <= 1.5 else x

    def normalize_sex(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        src = "sex" if "sex" in out.columns else ("gender" if "gender" in out.columns else None)
        if src is None:
            out["sex"] = np.nan
            return out
        s = out[src].astype(str).str.lower().str.strip()
        s = s.replace({"m":"male","man":"male","f":"female","woman":"female"})
        s = s.replace({"prefer not to say": np.nan, "prefer_not_to_say": np.nan, "prefer not to": np.nan})
        out["sex"] = s.where(s.isin(["male","female"]))
        return out

    def fmt_p(p):
        if p is None or (isinstance(p,float) and np.isnan(p)):
            return "NA"
        p = float(p)
        return "p < .001" if p < 0.001 else f"p = {p:.3f}"

    def safe_graphviz_import():
        try:
            import graphviz  # noqa
            return True
        except Exception:
            return False

    def find_beta_files():
        """
        Prefer exact path under outputs/run_20260119_192624/35_1b_beta_text/web.
        Fallback: rglob in outputs/run_20260119_192624.
        """
        patterns = {
            "male":   f"35.1b-1_effects_male_web.csv",
            "female": f"35.1b-1_effects_female_web.csv",
            "diff":   f"35.1b-3_effects_diff_female_minus_male_web.csv",
        }
        found = {}
        for k, fn in patterns.items():
            p = BETA_DIR / fn
            if p.exists():
                found[k] = p
            else:
                hits = list(RUN_DIR.rglob(fn))
                found[k] = hits[0] if hits else None
        return found

    def get_row(df, effect):
        hit = df[df["effect"] == effect]
        return None if hit.empty else hit.iloc[0]

    def fmt_coef(row):
        if row is None:
            return "NA", False
        coef = float(row["coef"])
        sig = bool(row.get("sig_CI_nonzero", False))
        return f"{coef:+.2f}{'*' if sig else ''}", sig

    def diff_sig(row):
        if row is None:
            return False
        return bool(row.get("sig_CI_nonzero", False))

    # -----------------------------
    # Panel A: Age-bin × Sex accuracy (PC)
    # -----------------------------
    def make_S2A():
        df = pd.read_csv(PC_FP, encoding="utf-8-sig")
        df = normalize_sex(df)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        acc_col = resolve_acc_col(df)
        df["accuracy_pct"] = to_percent(df[acc_col])

        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        d = df.dropna(subset=["age_group","sex","accuracy_pct"]).copy()

        # 2-way ANOVA (Type II)
        model = smf.ols("accuracy_pct ~ C(age_group) + C(sex) + C(age_group):C(sex)", data=d).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        p_age = float(anova.loc["C(age_group)", "PR(>F)"])
        p_sex = float(anova.loc["C(sex)", "PR(>F)"])
        p_int = float(anova.loc["C(age_group):C(sex)", "PR(>F)"])

        # summary
        summ = (d.groupby(["age_group","sex"], observed=True)["accuracy_pct"]
                  .agg(mean="mean", sd="std", n="count")
                  .reset_index())
        summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
        summ["age_group"] = pd.Categorical(summ["age_group"], categories=AGE_LABELS, ordered=True)
        summ = summ.sort_values(["age_group","sex"])

        # plot
        fig, ax = plt.subplots(figsize=(10, 8))
        x = np.arange(len(AGE_LABELS))
        for sex in ["male","female"]:
            s = summ[summ["sex"]==sex].set_index("age_group").reindex(AGE_LABELS)
            ax.errorbar(
                x, s["mean"].values, yerr=s["sem"].values,
                marker="o", linewidth=3, capsize=8,
                color=SEX_COLORS[sex], label=sex.title()
            )

        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Age group", fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontweight="bold")
        ax.set_title("A  Accuracy by age group × sex (PC; mean ± SEM)", fontweight="bold", pad=14)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

        # legend outside
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

        # p-box
        ax.text(
            0.02, 0.02,
            "Two-way ANOVA (Type II)\n"
            f"Age: {fmt_p(p_age)}\n"
            f"Sex: {fmt_p(p_sex)}\n"
            f"Age×Sex: {fmt_p(p_int)}",
            transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            fontweight="bold"
        )

        out_png = OUT_DIR / "figS2A_pc_agebin_sex_accuracy.png"
        out_svg = OUT_DIR / "figS2A_pc_agebin_sex_accuracy.svg"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

        # save anova
        anova_out = OUT_DIR / "figS2A_pc_agebin_sex_anova.csv"
        anova.round(8).to_csv(anova_out, encoding="utf-8-sig")

        return out_png, out_svg, anova_out

    # -----------------------------
    # Panel B: Graphviz sex-stratified path diagram (PC)
    # -----------------------------
    ARCS = [
        ("Age", "Exposure",   "a1",     "Age -> Exposure (a1)"),
        ("Age", "Confidence", "a2",     "Age -> Confidence (a2)"),
        ("Age", "Attitude",   "a3",     "Age -> Attitude (a3)"),
        ("Exposure",   "Accuracy", "b1","Exposure -> Accuracy (b1)"),
        ("Confidence", "Accuracy", "b2","Confidence -> Accuracy (b2)"),
        ("Attitude",   "Accuracy", "b3","Attitude -> Accuracy (b3)"),
        ("Age", "Accuracy",   "cprime", "Age -> Accuracy (c')"),
    ]

    def make_S2B_graphviz():
        if not safe_graphviz_import():
            raise RuntimeError("graphviz python package not available. Install python-graphviz + graphviz.")

        import graphviz

        paths = find_beta_files()
        if any(paths[k] is None for k in ["male","female","diff"]):
            raise FileNotFoundError(f"Missing 35.1b beta CSV(s) for PC/web: {paths}")

        male_df = pd.read_csv(paths["male"], encoding="utf-8-sig")
        female_df = pd.read_csv(paths["female"], encoding="utf-8-sig")
        diff_df = pd.read_csv(paths["diff"], encoding="utf-8-sig")

        g = graphviz.Digraph(name="figS2B_pc_sex_stratified_path")
        g.attr(rankdir="LR", bgcolor="white")

        # stable font
        g.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue",
               fontname="Arial", fontsize="14")
        g.attr("edge", fontname="Arial", fontsize="12", color="black")

        # nodes
        g.node("Age", "Age (X)")
        g.node("Exposure", "AI Exposure (M1)")
        g.node("Confidence", "AI Confidence (M2)")
        g.node("Attitude", "AI Attitude (M3)")
        g.node("Accuracy", "Accuracy (Y)")

        # keep mediators same rank
        with g.subgraph() as s:
            s.attr(rank="same")
            s.node("Exposure")
            s.node("Confidence")
            s.node("Attitude")

        # edges with male/female labels + Δ*
        for src, dst, key, human_label in ARCS:
            rm = get_row(male_df, key)
            rf = get_row(female_df, key)
            rd = get_row(diff_df, key)

            male_txt, male_sig = fmt_coef(rm)
            fem_txt, fem_sig = fmt_coef(rf)
            d_sig = diff_sig(rd)

            linestyle = "solid" if (male_sig or fem_sig) else "dashed"

            # highlight confidence pathway
            penwidth = "2.0"
            edgecolor = "black"
            if key in ["a2","b2"]:
                penwidth = "3.2"
                edgecolor = "#6A1B9A"
                if d_sig:
                    penwidth = "4.6"
                    edgecolor = "#C62828"

            delta = "  Delta*" if d_sig else ""
            edge_label = f"{human_label}\\nβ: {male_txt} / {fem_txt}{delta}"

            g.edge(src, dst, label=edge_label, style=linestyle, color=edgecolor, penwidth=penwidth)

        # summary note TOP-LEFT (avoid overlap)
        ind2_m = fmt_coef(get_row(male_df,"ind2"))[0]
        ind2_f = fmt_coef(get_row(female_df,"ind2"))[0]
        ind2_d = get_row(diff_df,"ind2")
        ind2_dsig = diff_sig(ind2_d)

        if ind2_d is not None and ("CI[2.5%]" in ind2_d.index) and ("CI[97.5%]" in ind2_d.index):
            dcoef = float(ind2_d["coef"])
            dlo = float(ind2_d["CI[2.5%]"]); dhi = float(ind2_d["CI[97.5%]"])
            ind2_diff_txt = f"{dcoef:+.2f} [{dlo:+.2f}, {dhi:+.2f}]{' *' if ind2_dsig else ''}"
        else:
            ind2_diff_txt = "NA"

        g.attr("node", shape="note", style="filled", fillcolor="#FFF9E5", fontsize="12", fontname="Arial")
        box_txt = (
            "Sex-stratified mediation (PC)\\n"
            "Male/Female beta shown per path\\n"
            "Delta* = significant sex difference\\n"
            "\\nKey confidence indirect (ind2):\\n"
            f"male:   {ind2_m}\\n"
            f"female: {ind2_f}\\n"
            f"diff(F-M): {ind2_diff_txt}\\n"
            "\\n* = bootstrap 95% CI excludes 0"
        )
        g.node("summary", box_txt)

        out_base = OUT_DIR / "figS2B_pc_sex_stratified_path"

        g.format = "png"
        png_path = g.render(str(out_base), cleanup=True)

        g.format = "svg"
        svg_path = g.render(str(out_base), cleanup=True)

        return Path(png_path), Path(svg_path), paths

    # -----------------------------
    # Build S2 (A+B combined)
    # -----------------------------
    print("==============================================================================")
    print("Figure S2 (PC): Sex differences replication (A: age×sex accuracy, B: graphviz path)")
    print("==============================================================================")

    A_png, A_svg, A_anova = make_S2A()
    B_png, B_svg, beta_paths = make_S2B_graphviz()

    # combine into one figure S2
    imgA = mpimg.imread(str(A_png))
    imgB = mpimg.imread(str(B_png))

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    ax1 = fig.add_subplot(gs[0,0]); ax1.imshow(imgA); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0,1]); ax2.imshow(imgB); ax2.axis("off")

    # panel letters (already in each panel title, but keep big letters too if desired)
    ax1.text(0.01, 0.98, "A", transform=ax1.transAxes, ha="left", va="top", fontsize=60, fontweight="bold")
    ax2.text(0.01, 0.98, "B", transform=ax2.transAxes, ha="left", va="top", fontsize=60, fontweight="bold")

    fig.tight_layout()

    OUT_PNG = OUT_DIR / "figS2_pc.png"
    OUT_SVG = OUT_DIR / "figS2_pc.svg"
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

    meta = {
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "pc_data": str(PC_FP),
            "beta_files": {k: str(v) if v else None for k,v in beta_paths.items()},
        },
        "outputs": {
            "S2A_png": str(A_png), "S2A_svg": str(A_svg), "S2A_anova": str(A_anova),
            "S2B_png": str(B_png), "S2B_svg": str(B_svg),
            "S2_png": str(OUT_PNG), "S2_svg": str(OUT_SVG),
        }
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("✅ saved:", OUT_PNG)
    print("✅ meta:", OUT_DIR / "meta.json")


def _run_cell_057():
    # ==============================================================================
    # Figure S3 (PC): Strategy analysis (vector SVG)
    # ------------------------------------------------------------------------------
    # Panel A: Multivariate strategy effects (OLS + HC3) from section 11-4 table (PC)
    # Panel B: Strategy usage frequency (PC cohort)
    # Outputs:
    #   plots/run_20260119_192624/supp/figS3_pc_strategy/
    #     - figS3_pc.svg   (editable in Illustrator)
    #     - figS3_pc.png
    #     - tables used / derived
    # ==============================================================================
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT

    # inputs
    REG_TABLE = config.OUTPUTS_DIR / f"run_{config.RUN_TAG}" / "11_strategy_effectiveness" / "11-4_strategy_multivariate_regression_table_web.csv"
    PC_DATA   = config.WEB_AGE_FILTERED

    # output
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS3_pc_strategy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    # colors
    COL_POS = "#DB4437"   # significant + (positive)
    COL_NEG = "#4285F4"   # significant - (negative)
    COL_NS  = "#BDBDBD"

    # strategy label mapping (must match your section 11)
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

    ALL_KEYS = list(STRATEGY_LABEL_EN.keys())

    def parse_strategy_list(cell):
        if pd.isna(cell):
            return []
        s = str(cell).strip().lower()
        if s == "on":
            return ALL_KEYS.copy()
        tokens = re.split(r"[,;|/\\]+", s)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [t for t in tokens if t in STRATEGY_LABEL_EN]

    def load_reg_table():
        if not REG_TABLE.exists():
            raise FileNotFoundError(f"Missing: {REG_TABLE}")

        reg = pd.read_csv(REG_TABLE, encoding="utf-8-sig")

        need = {"strategy","strategy_label","beta_pp","se_hc3","p_raw","q_fdr_bh","reject_fdr_bh(q<0.05)"}
        if not need.issubset(set(reg.columns)):
            raise KeyError(f"11-4 table missing columns. Need {need}, got {set(reg.columns)}")

        reg = reg.copy()
        reg["ci_low"]  = reg["beta_pp"] - 1.96 * reg["se_hc3"]
        reg["ci_high"] = reg["beta_pp"] + 1.96 * reg["se_hc3"]
        return reg

    def compute_usage_topk(df, topk=8):
        if "strategy" not in df.columns:
            raise KeyError("PC data missing 'strategy' column")

        tokens = df["strategy"].apply(parse_strategy_list)
        # count per strategy (participant-level)
        counts = {k: 0 for k in ALL_KEYS}
        n_total = len(df)

        for lst in tokens:
            st = set(lst)
            for k in st:
                counts[k] += 1

        usage = pd.DataFrame({
            "strategy": list(counts.keys()),
            "used_n": list(counts.values()),
            "used_pct": [counts[k] / n_total * 100.0 for k in counts.keys()],
        })
        usage["strategy_label"] = usage["strategy"].map(STRATEGY_LABEL_EN)
        usage = usage.sort_values("used_pct", ascending=False).head(topk)
        return usage, n_total

    # -----------------------------
    # Load inputs
    # -----------------------------
    reg = load_reg_table()
    df_pc = pd.read_csv(PC_DATA, encoding="utf-8-sig")
    usage, n_pc = compute_usage_topk(df_pc, topk=8)

    # save usage table
    usage_out = OUT_DIR / "figS3B_pc_strategy_usage_top.csv"
    usage.to_csv(usage_out, index=False, encoding="utf-8-sig")

    # -----------------------------
    # Plot: 2 panels (A,B)
    # -----------------------------
    fig = plt.figure(figsize=(26, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])

    # ---- A) coefficient plot (OLS+HC3) ----
    plotA = reg.sort_values("beta_pp").copy()

    colors = []
    for _, r in plotA.iterrows():
        if bool(r["reject_fdr_bh(q<0.05)"]):
            colors.append(COL_POS if r["beta_pp"] > 0 else COL_NEG)
        else:
            colors.append(COL_NS)

    y = np.arange(len(plotA))
    axA.hlines(y, plotA["ci_low"], plotA["ci_high"], color="#9E9E9E", lw=3, alpha=0.9)
    axA.scatter(plotA["beta_pp"], y, s=140, c=colors, edgecolors="none")
    axA.axvline(0, ls="--", lw=2, color="black", alpha=0.75)

    axA.set_yticks(y)
    axA.set_yticklabels(plotA["strategy_label"])
    axA.set_xlabel("Regression coefficient (pp) with 95% CI (HC3)", fontweight="bold")
    axA.set_title("A  Strategy effects on accuracy (PC)\nMultivariate OLS (age + sex + all strategies)", fontweight="bold", pad=12)
    axA.grid(True, axis="x", ls=":", alpha=0.35)

    # legend
    from matplotlib.patches import Patch
    axA.legend(
        handles=[
            Patch(facecolor=COL_POS, label="FDR sig (+)"),
            Patch(facecolor=COL_NEG, label="FDR sig (−)"),
            Patch(facecolor=COL_NS,  label="Not sig"),
        ],
        loc="lower left", frameon=True
    )

    # ---- B) usage frequency ----
    axB.barh(usage["strategy_label"][::-1], usage["used_pct"][::-1], color="#4A90E2", alpha=0.90, edgecolor="black")
    axB.set_xlabel("Used (%)", fontweight="bold")
    axB.set_title("B  Strategy usage frequency (PC)", fontweight="bold", pad=12)
    axB.grid(True, axis="x", ls=":", alpha=0.35)

    # annotate n
    for i, (_, r) in enumerate(usage.iloc[::-1].reset_index(drop=True).iterrows()):
        axB.text(r["used_pct"] + 1.0, i, f"n={int(r['used_n'])}", va="center", fontweight="bold", fontsize=BIG*0.55)

    axB.text(
        0.98, 0.02,
        f"PC cohort N={n_pc}",
        transform=axB.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    fig.tight_layout()

    out_png = OUT_DIR / "figS3_pc.png"
    out_svg = OUT_DIR / "figS3_pc.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", out_png)
    print("✅ saved:", out_svg)
    print("✅ usage table:", usage_out)
    print("✅ reg table used:", REG_TABLE)


def _run_cell_059():
    # ==============================================================================
    # Fig S3B (PC): Strategy endorsement rate (ALL strategies)
    # ------------------------------------------------------------------------------
    # - Computes % of participants endorsing each strategy (multiple selection allowed)
    # - Uses the raw 'strategy' column in analysis_data_web_age_filtered_20_69.csv
    # - Saves: figS3B_pc_strategy_endorsement_all.csv / .png / .svg
    # ==============================================================================

    import re
    import ast
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT

    PC_DATA = config.WEB_AGE_FILTERED

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS3_pc_strategy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Big fonts (paper style)
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    # Strategy label mapping (must match your Section 11 keys)
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
    ALL_KEYS = list(STRATEGY_LABEL_EN.keys())

    def parse_strategy_list(cell):
        """Parse the raw strategy cell into a list of normalized keys."""
        if pd.isna(cell):
            return []
        s = str(cell).strip().lower()
        if not s:
            return []

        # if stored as python list-like string
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                items = list(ast.literal_eval(s))
                tokens = [str(x).strip().lower() for x in items]
            except Exception:
                tokens = [s]
        else:
            tokens = re.split(r"[,;|/\\]+", s)
            tokens = [t.strip() for t in tokens if t.strip()]

        # keep only known keys
        return [t for t in tokens if t in STRATEGY_LABEL_EN]

    def compute_usage_all(df):
        """Compute endorsement counts/rates for ALL strategies (participant-level, multi-select)."""
        tokens = df["strategy"].apply(parse_strategy_list)
        counts = {k: 0 for k in ALL_KEYS}
        n_total = len(df)

        for lst in tokens:
            st = set(lst)  # multi-select; count participant once per strategy
            for k in st:
                counts[k] += 1

        usage = pd.DataFrame({
            "strategy": list(counts.keys()),
            "strategy_label": [STRATEGY_LABEL_EN[k] for k in counts.keys()],
            "used_n": list(counts.values()),
            "used_pct": [counts[k] / n_total * 100.0 for k in counts.keys()],
        }).sort_values("used_pct", ascending=False)

        return usage, n_total

    # -----------------------------
    # Load + compute
    # -----------------------------
    df_pc = pd.read_csv(PC_DATA, encoding="utf-8-sig")

    if "strategy" not in df_pc.columns:
        raise KeyError("PC data missing 'strategy' column")

    # Define respondents for S3B: non-empty strategy responses
    resp_mask = df_pc["strategy"].notna() & (df_pc["strategy"].astype(str).str.strip() != "")
    df_pc_resp = df_pc.loc[resp_mask].copy()

    usage, n_pc = compute_usage_all(df_pc_resp)

    # Save table
    usage_csv = OUT_DIR / "figS3B_pc_strategy_endorsement_all.csv"
    usage.to_csv(usage_csv, index=False, encoding="utf-8-sig")
    print("✅ saved usage table:", usage_csv)
    print("✅ PC strategy respondents N:", n_pc)

    # -----------------------------
    # Plot (standalone S3B)
    # -----------------------------
    plt.figure(figsize=(12, 10))
    plt.barh(usage["strategy_label"][::-1], usage["used_pct"][::-1],
             color="#4A90E2", alpha=0.90, edgecolor="black")
    plt.xlabel("Participants endorsing strategy (%)", fontweight="bold")
    plt.title("Strategy endorsement rate (PC)", fontweight="bold", pad=12)
    plt.grid(True, axis="x", ls=":", alpha=0.35)

    plt.tight_layout()
    out_png = OUT_DIR / "figS3B_pc_strategy_endorsement_all.png"
    out_svg = OUT_DIR / "figS3B_pc_strategy_endorsement_all.svg"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("✅ saved:", out_png)
    print("✅ saved:", out_svg)


def _run_cell_061():
    # ==============================================================================
    # Figure S4 (PC): RT replication of Figure 7 (vector SVG)
    # Panels:
    #   A) Overall mean RT by age group × sex (PC) + 2-way ANOVA p-values
    #   B) Condition-specific RT (PC): LMM EMM interaction + posthoc (Real vs AI at Correct/Incorrect)
    # Outputs:
    #   plots/run_20260119_192624/supp/figS4_pc_rt/figS4_pc.svg/.png
    #   + stats tables (csv)
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import norm

    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT

    PC_MAIN = config.WEB_AGE_FILTERED
    PC_TRIAL = config.OUTPUTS_DIR / "outputs_26_verification_cost_v1_2" / "web" / "26v12-0_trial_level_table.csv"

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS4_pc_rt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    AGE_BINS   = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    # big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    SEX_COLORS = {"male": "#4285F4", "female": "#DB4437"}

    def fmt_p(p):
        if p is None or (isinstance(p,float) and np.isnan(p)):
            return "NA"
        p = float(p)
        return "p < .001" if p < 0.001 else f"p = {p:.3f}"

    def normalize_sex(df):
        src = "sex" if "sex" in df.columns else ("gender" if "gender" in df.columns else None)
        if src is None:
            df["sex"] = np.nan
            return df
        s = df[src].astype(str).str.lower().str.strip()
        s = s.replace({"m":"male","man":"male","f":"female","woman":"female"})
        s = s.replace({"prefer not to say": np.nan, "prefer_not_to_say": np.nan, "prefer not to": np.nan})
        df["sex"] = s.where(s.isin(["male","female"]))
        return df

    def pick_rt_col(df):
        for c in ["avgRT", "mean_rt", "avg_rt", "MeanRT", "meanRT"]:
            if c in df.columns:
                return c
        return None

    def sem(x):
        x = pd.Series(x).dropna()
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1)/np.sqrt(len(x))

    def wald_contrast(m, L):
        params = m.params.values.reshape(-1,1)
        cov = m.cov_params().values
        L = np.asarray(L, dtype=float).reshape(-1,1)
        est = float(L.T @ params)
        var = float(L.T @ cov @ L)
        se = np.sqrt(var) if var >= 0 else np.nan
        z = est/se if (se and np.isfinite(se) and se > 0) else np.nan
        p = 2*(1-norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        return est, se, z, p

    def add_bracket(ax, x1, x2, y, text, barh=0.02):
        ylim = ax.get_ylim()
        yr = ylim[1] - ylim[0]
        h = barh * yr
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c="black")
        ax.text((x1+x2)/2, y+h*1.10, text, ha="center", va="bottom", fontweight="bold", fontsize=BIG*0.65)

    # =============================================================================
    # Panel A: Overall RT by age group × sex (PC)
    # =============================================================================
    df = pd.read_csv(PC_MAIN, encoding="utf-8-sig")
    df = normalize_sex(df)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    rt_col = pick_rt_col(df)
    if rt_col is None:
        raise KeyError("No RT column found in PC main file (avgRT etc).")

    df["rt_s"] = pd.to_numeric(df[rt_col], errors="coerce") / 1000.0  # seconds
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    dA = df.dropna(subset=["age_group","sex","rt_s"]).copy()

    # 2-way ANOVA (Type II)
    mA = smf.ols("rt_s ~ C(age_group) + C(sex) + C(age_group):C(sex)", data=dA).fit()
    anovaA = sm.stats.anova_lm(mA, typ=2)
    p_age = float(anovaA.loc["C(age_group)", "PR(>F)"])
    p_sex = float(anovaA.loc["C(sex)", "PR(>F)"])
    p_int = float(anovaA.loc["C(age_group):C(sex)", "PR(>F)"])

    summA = (dA.groupby(["age_group","sex"], observed=True)["rt_s"]
               .agg(mean="mean", sd="std", n="count").reset_index())
    summA["sem"] = summA["sd"]/np.sqrt(summA["n"])
    summA["age_group"] = pd.Categorical(summA["age_group"], categories=AGE_LABELS, ordered=True)
    summA = summA.sort_values(["age_group","sex"])

    # =============================================================================
    # Panel B: Trial-level LMM EMM + posthoc (PC)
    # =============================================================================
    dt = pd.read_csv(PC_TRIAL, encoding="utf-8-sig")
    need = {"participantId","logRT","Correctness","Kind","age","sex"}
    if not need.issubset(set(dt.columns)):
        raise KeyError(f"Trial table missing columns. Need {need}")

    dt["age"] = pd.to_numeric(dt["age"], errors="coerce")
    dt["sex"] = dt["sex"].astype(str).str.lower().str.strip()
    dt.loc[~dt["sex"].isin(["male","female"]), "sex"] = np.nan
    dt = dt.dropna(subset=["logRT","Correctness","Kind","age"]).copy()
    dt = dt[dt["Correctness"].isin(["Correct","Incorrect"]) & dt["Kind"].isin(["Real","AI"])].copy()

    formula = "logRT ~ C(Correctness) * C(Kind) + age + C(sex)"
    md = smf.mixedlm(formula, dt, groups=dt["participantId"])
    m = md.fit(method="lbfgs", reml=False)

    inter_term = "C(Correctness)[T.Incorrect]:C(Kind)[T.Real]"
    p_inter = float(m.pvalues[inter_term]) if inter_term in m.pvalues.index else np.nan

    # EMM at mean age, sex-weighted
    age_mean = float(dt["age"].mean())
    sex_counts = dt["sex"].value_counts(dropna=True)
    w_m = float(sex_counts.get("male",0))/float(sex_counts.sum()) if sex_counts.sum()>0 else 1.0
    w_f = float(sex_counts.get("female",0))/float(sex_counts.sum()) if sex_counts.sum()>0 else 0.0

    grid=[]
    for corr in ["Correct","Incorrect"]:
        for kind in ["Real","AI"]:
            for sex,w in [("male",w_m),("female",w_f)]:
                if w==0: continue
                grid.append({"Correctness":corr,"Kind":kind,"age":age_mean,"sex":sex,"w":w})
    grid=pd.DataFrame(grid)
    grid["pred_logRT"]=m.predict(grid)
    grid["pred_rt_s"]=np.exp(grid["pred_logRT"])

    emm=(grid.groupby(["Correctness","Kind"])
           .apply(lambda g: float(np.sum(g["pred_rt_s"]*g["w"])))
           .reset_index(name="emm_rt_s"))

    # Posthoc contrasts (logRT scale)
    terms = list(m.params.index)

    # Correct: Real-AI == C(Kind)[T.Real]
    Lc = np.zeros(len(terms))
    if "C(Kind)[T.Real]" in terms:
        Lc[terms.index("C(Kind)[T.Real]")] = 1.0
    _,_,z_c,p_c = wald_contrast(m, Lc)

    # Incorrect: Real-AI == C(Kind)[T.Real] + interaction
    Li = np.zeros(len(terms))
    if "C(Kind)[T.Real]" in terms:
        Li[terms.index("C(Kind)[T.Real]")] = 1.0
    if inter_term in terms:
        Li[terms.index(inter_term)] = 1.0
    _,_,z_i,p_i = wald_contrast(m, Li)

    # =============================================================================
    # Plot S4 (vector)
    # =============================================================================
    fig = plt.figure(figsize=(36, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    # ---- S4A ----
    x = np.arange(len(AGE_LABELS))
    for sex in ["male","female"]:
        s = summA[summA["sex"]==sex].set_index("age_group").reindex(AGE_LABELS)
        ax1.errorbar(x, s["mean"].values, yerr=s["sem"].values,
                     marker="o", linewidth=3, capsize=8,
                     color=SEX_COLORS[sex], label=sex.title())

    ax1.set_xticks(x)
    ax1.set_xticklabels(AGE_LABELS, fontweight="bold")
    ax1.set_xlabel("Age group", fontweight="bold")
    ax1.set_ylabel("Overall mean RT (s)", fontweight="bold")
    ax1.set_title("A  Overall RT by age group × sex (PC; mean ± SEM)", fontweight="bold", pad=14)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

    ax1.text(
        1.02, 0.02,
        "Two-way ANOVA (Type II)\n..."
        , transform=ax1.transAxes, ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )


    # ---- S4B ----
    x_order = ["Correct","Incorrect"]
    xx = np.arange(len(x_order))

    for kind, marker, color in [("Real","o","#1f77b4"), ("AI","s","#ff7f0e")]:
        sub = emm[emm["Kind"]==kind].set_index("Correctness").reindex(x_order).reset_index()
        ax2.plot(xx, sub["emm_rt_s"].values, marker=marker, linewidth=4, markersize=10, label=kind, color=color)

    ax2.set_xticks(xx)
    ax2.set_xticklabels(x_order, fontweight="bold")
    ax2.set_ylabel("Model-predicted RT (s)", fontweight="bold")
    ax2.set_title("B  Condition-specific RT (PC)\nLMM EMM + posthoc", fontweight="bold", pad=14)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax2.legend(title="Image kind", frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1))

    ax2.text(
        1.02, 0.02,
        f"Interaction...\n{fmt_p(p_inter)}",
        transform=ax2.transAxes, ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )


    # posthoc brackets (Real vs AI at each correctness)
    y_correct = float(emm[emm["Correctness"]=="Correct"]["emm_rt_s"].max())
    y_incor = float(emm[emm["Correctness"]=="Incorrect"]["emm_rt_s"].max())
    add_bracket(ax2, 0-0.12, 0+0.12, y_correct*1.03, fmt_p(p_c))
    add_bracket(ax2, 1-0.12, 1+0.12, y_incor*1.03,   fmt_p(p_i))

    fig.tight_layout(rect=[0, 0, 0.80, 1])


    out_png = OUT_DIR / "figS4_pc.png"
    out_svg = OUT_DIR / "figS4_pc.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # save tables
    anovaA.round(8).to_csv(OUT_DIR/"figS4A_pc_overall_rt_anova.csv", encoding="utf-8-sig")
    emm.to_csv(OUT_DIR/"figS4B_pc_emm_table.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "cohort": "pc",
        "interaction_p": p_inter,
        "posthoc_real_vs_ai_correct_p": p_c,
        "posthoc_real_vs_ai_incorrect_p": p_i
    }]).to_csv(OUT_DIR/"figS4B_pc_posthoc_pvalues.csv", index=False, encoding="utf-8-sig")

    print("✅ saved:", out_png)
    print("✅ saved:", out_svg)
    print("✅ tables saved to:", OUT_DIR)


def _run_cell_062():
    # ==============================================================================
    # Figure S4 (PC): Unified human-factors model (vector SVG)
    # ------------------------------------------------------------------------------
    # Panels:
    #   A) Standardized β forest plot (PC) with 95% CI (HC3)
    #   B) Nested model R² (PC): Demographics -> AI self-reports -> Strategies
    #
    # Inputs (from your Fig6 pipeline):
    #   - plots/run_20260119_192624/06_human_factors_model/figS6A_pc_std_beta_table.csv
    #   - plots/run_20260119_192624/06_human_factors_model/figS6B_pc_nested_R2_table.csv
    #
    # Output:
    #   plots/run_20260119_192624/supp/figS4_pc_unified_model/figS4_pc.svg/.png
    # ==============================================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RUN_TAG = config.RUN_TAG

    SRC_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "06_human_factors_model"
    IN_A = SRC_DIR / "figS6A_pc_std_beta_table.csv"
    IN_B = SRC_DIR / "figS6B_pc_nested_R2_table.csv"

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS4_pc_unified_model"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_PNG = OUT_DIR / "figS4_pc.png"
    OUT_SVG = OUT_DIR / "figS4_pc.svg"

    for p in [IN_A, IN_B]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    # ---- big fonts (paper style) ----
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    COL_POS = "#DB4437"
    COL_NEG = "#4285F4"
    COL_NS  = "#BDBDBD"

    def sig_color(beta, p):
        if np.isfinite(p) and p < 0.05:
            return COL_POS if beta > 0 else COL_NEG
        return COL_NS

    # -----------------------------
    # Load tables
    # -----------------------------
    A = pd.read_csv(IN_A, encoding="utf-8-sig")
    needA = {"term","beta_std","se_hc3","p","ci_low","ci_high"}
    missA = needA - set(A.columns)
    if missA:
        raise KeyError(f"Missing columns in {IN_A.name}: {missA}")

    B = pd.read_csv(IN_B, encoding="utf-8-sig")
    needB = {"model","R2","Adj_R2","ΔR2_vs_prev"}
    missB = needB - set(B.columns)
    if missB:
        raise KeyError(f"Missing columns in {IN_B.name}: {missB}")

    # sort Panel A by beta
    A = A.copy().sort_values("beta_std", ascending=True).reset_index(drop=True)
    A["color"] = [sig_color(b, p) for b, p in zip(A["beta_std"].values, A["p"].values)]

    # Ensure Panel B order (as saved)
    B = B.copy().reset_index(drop=True)

    # -----------------------------
    # Plot (2 panels)
    # -----------------------------
    fig = plt.figure(figsize=(26, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    # ---- Panel A: β forest ----
    y = np.arange(len(A))
    ax1.hlines(y, A["ci_low"], A["ci_high"], color="#9E9E9E", lw=3, alpha=0.9)
    ax1.scatter(A["beta_std"], y, s=140, c=A["color"], edgecolors="none")
    ax1.axvline(0, ls="--", lw=2, color="black", alpha=0.75)

    ax1.set_yticks(y)
    ax1.set_yticklabels(A["term"])
    ax1.set_xlabel("Standardized coefficient (β) with 95% CI (HC3)", fontweight="bold")
    ax1.set_title("A  Unified human-factors model (PC)\nStandardized β (HC3 robust SE)", fontweight="bold", pad=12)
    ax1.grid(True, axis="x", ls=":", alpha=0.35)

    from matplotlib.patches import Patch
    ax1.legend(
        handles=[
            Patch(facecolor=COL_POS, label="p<.05 (β>0)"),
            Patch(facecolor=COL_NEG, label="p<.05 (β<0)"),
            Patch(facecolor=COL_NS,  label="n.s."),
        ],
        loc="lower left",
        frameon=True
    )

    # ---- Panel B: Nested R² ----
    x = np.arange(len(B))
    ax2.plot(x, B["R2"].values, marker="o", linewidth=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(B["model"].values, rotation=0, ha="center")
    ax2.set_ylabel("R²", fontweight="bold")
    ax2.set_title("B  Incremental variance explained (PC)\nNested models (ΔR² shown)", fontweight="bold", pad=12)
    ax2.grid(True, axis="y", ls=":", alpha=0.35)

    # annotate ΔR² (for rows after first)
    for i in range(1, len(B)):
        dr2 = B.loc[i, "ΔR2_vs_prev"]
        if np.isfinite(dr2):
            ax2.text(i, B.loc[i, "R2"] + 0.01, f"ΔR²={dr2:.3f}", ha="center", fontweight="bold")

    ax2.text(
        0.02, 0.02,
        "Blocks added sequentially:\nDemographics → AI self-reports → Strategies",
        transform=ax2.transAxes, ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", OUT_PNG)
    print("✅ saved:", OUT_SVG)
    print("Inputs:", IN_A.name, IN_B.name)


def _run_cell_063():
    # ==============================================================================
    # Figure S4 (PC) CLEAN: no boxes inside plots (vector SVG)
    # - Removes: legend box, note box, ΔR² text annotations
    # Outputs:
    #   plots/run_20260119_192624/supp/figS4_pc_unified_model/figS4_pc_clean.svg/.png
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RUN_TAG = config.RUN_TAG

    SRC_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "06_human_factors_model"
    IN_A = SRC_DIR / "figS6A_pc_std_beta_table.csv"
    IN_B = SRC_DIR / "figS6B_pc_nested_R2_table.csv"

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS4_pc_unified_model"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_PNG = OUT_DIR / "figS4_pc_clean.png"
    OUT_SVG = OUT_DIR / "figS4_pc_clean.svg"

    # big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    COL_POS = "#DB4437"
    COL_NEG = "#4285F4"
    COL_NS  = "#BDBDBD"

    def sig_color(beta, p):
        if np.isfinite(p) and p < 0.05:
            return COL_POS if beta > 0 else COL_NEG
        return COL_NS

    # load
    A = pd.read_csv(IN_A, encoding="utf-8-sig").sort_values("beta_std").reset_index(drop=True)
    B = pd.read_csv(IN_B, encoding="utf-8-sig").reset_index(drop=True)

    A["color"] = [sig_color(b, p) for b, p in zip(A["beta_std"].values, A["p"].values)]

    # plot
    fig = plt.figure(figsize=(26, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    # A
    y = np.arange(len(A))
    ax1.hlines(y, A["ci_low"], A["ci_high"], color="#9E9E9E", lw=3, alpha=0.9)
    ax1.scatter(A["beta_std"], y, s=140, c=A["color"], edgecolors="none")
    ax1.axvline(0, ls="--", lw=2, color="black", alpha=0.75)
    ax1.set_yticks(y)
    ax1.set_yticklabels(A["term"])
    ax1.set_xlabel("Standardized coefficient (β) with 95% CI (HC3)", fontweight="bold")
    ax1.set_title("A  Unified human-factors model (PC)\nStandardized β (HC3 robust SE)", fontweight="bold", pad=12)
    ax1.grid(True, axis="x", ls=":", alpha=0.35)

    # B
    x = np.arange(len(B))
    ax2.plot(x, B["R2"].values, marker="o", linewidth=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(B["model"].values, rotation=0, ha="center")
    ax2.set_ylabel("R²", fontweight="bold")
    ax2.set_title("B  Incremental variance explained (PC)\nNested models", fontweight="bold", pad=12)
    ax2.grid(True, axis="y", ls=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", OUT_PNG)
    print("✅ saved:", OUT_SVG)


def _run_cell_065():
    # ==============================================================================
    # Figure S6 (PC): Generator comparison (ChatGPT-4o vs Gemini/Imagen3) - VECTOR SVG
    # ------------------------------------------------------------------------------
    # Panels:
    #   A) Accuracy (mean ± SEM)
    #   B) Mean RT seconds (mean ± SEM)
    #   C) Age–accuracy scatter + regression lines
    #   D) Slope comparison (slope ± SE from OLS/linregress) + age×generator interaction (HC3)
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy import stats
    import statsmodels.formula.api as smf
    from scipy.stats import linregress

    # -----------------------------
    # Config
    # -----------------------------
    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT
    IN_FP = config.WEB_AGE_FILTERED

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS6_pc_generator"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    GEN_LABEL_CGPT = "ChatGPT-4o (native images; July 2025 stimuli)"
    GEN_LABEL_GEM  = "Gemini (Imagen 3; July 2025 stimuli)"

    # big fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    # colors
    COL_CGPT = "#4285F4"  # blue
    COL_GEM  = "#DB4437"  # red

    # -----------------------------
    # helpers
    # -----------------------------
    def to_percent_series(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        finite = x.dropna()
        if finite.empty:
            return x
        return x * 100.0 if float(finite.max()) <= 1.5 else x

    def sem(x):
        x = pd.Series(x).dropna().astype(float)
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))

    def fmt_p(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "NA"
        p = float(p)
        return "p < .001" if p < 0.001 else f"p = {p:.3f}"

    def paired_tests(x1, x2):
        x1 = pd.to_numeric(pd.Series(x1), errors="coerce")
        x2 = pd.to_numeric(pd.Series(x2), errors="coerce")
        d = pd.concat([x1, x2], axis=1).dropna()
        a = d.iloc[:, 0].values
        b = d.iloc[:, 1].values
        n = len(a)

        t = stats.ttest_rel(a, b, nan_policy="omit")
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        return {
            "N": int(n),
            "mean_cgpt": float(np.mean(a)),
            "sem_cgpt": float(sem(a)),
            "mean_gem": float(np.mean(b)),
            "sem_gem": float(sem(b)),
            "t": float(t.statistic),
            "p_t": float(t.pvalue),
            "wilcoxon_W": w_stat,
            "p_wilcoxon": w_p
        }

    def slope_and_se(age, y):
        sub = pd.DataFrame({"age": age, "y": y}).dropna()
        if len(sub) < 3:
            return np.nan, np.nan, np.nan, np.nan
        lr = linregress(sub["age"].values, sub["y"].values)
        # slope, slope_se, intercept, p
        return float(lr.slope), float(lr.stderr), float(lr.intercept), float(lr.pvalue)

    # -----------------------------
    # Load + prep
    # -----------------------------
    df = pd.read_csv(IN_FP, encoding="utf-8-sig")

    req = ["accuracy_chatgpt", "accuracy_gemini", "avg_rt_chatgpt", "avg_rt_gemini", "age"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in PC file: {miss}")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["acc_cgpt"] = to_percent_series(df["accuracy_chatgpt"])
    df["acc_gem"]  = to_percent_series(df["accuracy_gemini"])

    df["rt_cgpt_s"] = pd.to_numeric(df["avg_rt_chatgpt"], errors="coerce") / 1000.0
    df["rt_gem_s"]  = pd.to_numeric(df["avg_rt_gemini"], errors="coerce") / 1000.0

    # optional: ensure minimum trials if available
    if "n_chatgpt" in df.columns:
        df = df[df["n_chatgpt"].fillna(0) >= 3].copy()
    if "n_gemini" in df.columns:
        df = df[df["n_gemini"].fillna(0) >= 3].copy()

    # -----------------------------
    # Stats for A/B
    # -----------------------------
    statA = paired_tests(df["acc_cgpt"], df["acc_gem"])
    statB = paired_tests(df["rt_cgpt_s"], df["rt_gem_s"])

    # -----------------------------
    # Scatter + interaction model
    # -----------------------------
    dC = df[["age", "acc_cgpt", "acc_gem"]].dropna().copy()

    long = pd.concat([
        dC[["age", "acc_cgpt"]].rename(columns={"acc_cgpt": "accuracy"}).assign(generator="ChatGPT-4o"),
        dC[["age", "acc_gem"]].rename(columns={"acc_gem": "accuracy"}).assign(generator="Gemini"),
    ], ignore_index=True)

    # interaction regression (HC3)
    model = smf.ols("accuracy ~ age * C(generator)", data=long).fit(cov_type="HC3")
    inter_terms = [t for t in model.params.index if "age:C(generator)" in t]
    p_inter = float(model.pvalues[inter_terms[0]]) if inter_terms else np.nan

    # slope ± SE (OLS) for caption + error bars (Option A)
    slope_cgpt, se_cgpt, intercept_cgpt, p_slope_cgpt = slope_and_se(dC["age"], dC["acc_cgpt"])
    slope_gem,  se_gem,  intercept_gem,  p_slope_gem  = slope_and_se(dC["age"], dC["acc_gem"])

    print(f"[PC] slope (ChatGPT-4o) = {slope_cgpt:.3f} ± {se_cgpt:.3f}  (OLS)")
    print(f"[PC] slope (Gemini)     = {slope_gem:.3f} ± {se_gem:.3f}  (OLS)")
    print(f"[PC] slope-diff test (age×generator, HC3) p = {p_inter:.3g}")

    # -----------------------------
    # Plot (vector)
    # -----------------------------
    fig = plt.figure(figsize=(30, 12))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.35, 0.85])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[0, 3])

    # A) Accuracy bar
    axA.bar([0, 1], [statA["mean_cgpt"], statA["mean_gem"]],
            yerr=[statA["sem_cgpt"], statA["sem_gem"]],
            capsize=8, width=0.55, color=[COL_CGPT, COL_GEM],
            edgecolor="black", alpha=0.85)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["ChatGPT-4o", "Gemini"], fontweight="bold")
    axA.set_ylim(0, 100)
    axA.set_ylabel("Accuracy (%)", fontweight="bold")
    axA.set_title("A  Accuracy (PC)", fontweight="bold", pad=12)
    axA.grid(True, axis="y", linestyle=":", alpha=0.35)

    axA.text(0.5, -0.20, f"{GEN_LABEL_CGPT}\n{GEN_LABEL_GEM}",
             transform=axA.transAxes, ha="center", va="top",
             fontsize=BIG * 0.55)

    # B) RT bar
    axB.bar([0, 1], [statB["mean_cgpt"], statB["mean_gem"]],
            yerr=[statB["sem_cgpt"], statB["sem_gem"]],
            capsize=8, width=0.55, color=[COL_CGPT, COL_GEM],
            edgecolor="black", alpha=0.85)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["ChatGPT-4o", "Gemini"], fontweight="bold")
    axB.set_ylabel("Mean RT (s)", fontweight="bold")
    axB.set_title("B  Mean RT (PC)", fontweight="bold", pad=12)
    axB.grid(True, axis="y", linestyle=":", alpha=0.35)

    # C) Age–accuracy scatter + lines (OLS lines)
    axC.scatter(dC["age"], dC["acc_cgpt"], s=18, alpha=0.18, color=COL_CGPT, edgecolors="none", label="ChatGPT-4o")
    axC.scatter(dC["age"], dC["acc_gem"],  s=18, alpha=0.18, color=COL_GEM,  edgecolors="none", label="Gemini")

    if len(dC) >= 30:
        xx = np.linspace(dC["age"].min(), dC["age"].max(), 200)
        axC.plot(xx, slope_cgpt * xx + intercept_cgpt, linewidth=4, color=COL_CGPT)
        axC.plot(xx, slope_gem  * xx + intercept_gem,  linewidth=4, color=COL_GEM)

    axC.set_ylim(0, 100)
    axC.set_xlabel("Age", fontweight="bold")
    axC.set_ylabel("Accuracy (%)", fontweight="bold")
    axC.set_title("C  Age–accuracy by generator (PC)", fontweight="bold", pad=12)
    axC.grid(True, linestyle=":", alpha=0.35)
    axC.legend(frameon=True, loc="upper right")

    # D) slope bars with SE (OLS)
    axD.bar([0, 1], [slope_cgpt, slope_gem],
            yerr=[se_cgpt, se_gem],
            capsize=8, width=0.55,
            color=[COL_CGPT, COL_GEM],
            edgecolor="black", alpha=0.85)

    axD.axhline(0, linestyle="--", linewidth=2, color="black", alpha=0.7)
    axD.set_xticks([0, 1])
    axD.set_xticklabels(["ChatGPT-4o", "Gemini"], rotation=90, fontweight="bold")
    axD.set_ylabel("Slope (% points/year)\n± SE (OLS)", fontweight="bold")
    axD.set_title("D  Slope comparison (PC)", fontweight="bold", pad=12)
    axD.grid(True, axis="y", linestyle=":", alpha=0.35)

    # put p_inter outside
    fig.text(0.985, 0.02, f"age×generator interaction (HC3): {fmt_p(p_inter)}",
             ha="right", va="bottom", fontsize=BIG * 0.65, fontweight="bold")

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_png = OUT_DIR / "figS6_pc.png"
    out_svg = OUT_DIR / "figS6_pc.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------
    # Save stats tables + console output
    # -----------------------------
    tests_df = pd.DataFrame([
        {"cohort": "pc", "metric": "accuracy", **statA},
        {"cohort": "pc", "metric": "rt_seconds", **statB},
        {"cohort": "pc", "metric": "slopes_ols",
         "N": int(len(dC)),
         "slope_cgpt": slope_cgpt, "se_cgpt": se_cgpt,
         "slope_gem": slope_gem, "se_gem": se_gem,
         "p_interaction_hc3": p_inter}
    ])

    stats_out = OUT_DIR / "figS6_stats_pc.csv"
    tests_df.to_csv(stats_out, index=False, encoding="utf-8-sig")

    coef_out = OUT_DIR / "figS6D_interaction_coeffs_pc.csv"
    pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "se_hc3": model.bse.values,
        "p": model.pvalues.values
    }).to_csv(coef_out, index=False, encoding="utf-8-sig")

    print("\n==============================================================================")
    print("Figure S6 stats [PC]")
    print("==============================================================================")
    print("(A) Accuracy (ChatGPT-4o vs Gemini)")
    print(f"  N paired: {statA['N']}")
    print(f"  Mean±SEM (%): ChatGPT-4o={statA['mean_cgpt']:.2f}±{statA['sem_cgpt']:.2f} | Gemini={statA['mean_gem']:.2f}±{statA['sem_gem']:.2f}")
    print(f"  Paired t-test: t={statA['t']:.3f}, p={statA['p_t']:.3g}")

    print("\n(B) Mean RT (seconds) (ChatGPT-4o vs Gemini)")
    print(f"  N paired: {statB['N']}")
    print(f"  Mean±SEM (s): ChatGPT-4o={statB['mean_cgpt']:.3f}±{statB['sem_cgpt']:.3f} | Gemini={statB['mean_gem']:.3f}±{statB['sem_gem']:.3f}")
    print(f"  Paired t-test: t={statB['t']:.3f}, p={statB['p_t']:.3g}")

    print("\n(C–D) Age–accuracy trends by generator")
    print(f"  N scatter complete: {len(dC)}")
    print(f"  Slopes (OLS ± SE): ChatGPT-4o={slope_cgpt:.3f}±{se_cgpt:.3f} | Gemini={slope_gem:.3f}±{se_gem:.3f}")
    print(f"  Slope-difference test (age×generator interaction, HC3): p={p_inter:.3g}")
    print("==============================================================================")

    print("✅ saved:", out_png)
    print("✅ stats:", stats_out)
    print("✅ interaction coeffs:", coef_out)


def _run_cell_067():
    # ==============================================================================
    # Figure S0 (for Fig2): Age-bin participant counts by device, with Sex breakdown
    #   - includes prefer-not-to-say
    # Panels (2x2):
    #   A) Mobile total counts by age bin
    #   B) PC total counts by age bin
    #   C) Mobile counts by age bin, stacked by sex (male/female/prefer-not-to-say)
    #   D) PC counts by age bin, stacked by sex (male/female/prefer-not-to-say)
    # - All y-axes matched within row groups
    # - Saves vector SVG + PNG + tables
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RUN_TAG = config.RUN_TAG
    BASE_DIR = config.PROJECT_ROOT

    MOBILE_FP = config.MOBILE_AGE_FILTERED
    PC_FP     = config.WEB_AGE_FILTERED

    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "supp" / "figS0_age_counts_device"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    AGE_BINS   = [19, 29, 39, 49, 59, 69]
    AGE_LABELS = ["20s", "30s", "40s", "50s", "60s"]

    # paper fonts
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.70)

    SEX_ORDER = ["male", "female", "prefer-not-to-say"]
    SEX_LABEL = {"male":"Male", "female":"Female", "prefer-not-to-say":"Prefer not to say"}
    SEX_COLORS = {"male": "#4285F4", "female": "#DB4437", "prefer-not-to-say": "#9E9E9E"}

    def normalize_sex(df):
        src = "sex" if "sex" in df.columns else ("gender" if "gender" in df.columns else None)
        if src is None:
            df["sex"] = np.nan
            return df

        s = df[src].astype(str).str.lower().str.strip()
        s = s.replace({"m":"male","man":"male","f":"female","woman":"female"})
        s = s.replace({
            "prefer not to say":"prefer-not-to-say",
            "prefer_not_to_say":"prefer-not-to-say",
            "prefer not to":"prefer-not-to-say",
            "prefer-not-to-say":"prefer-not-to-say"
        })
        # anything else -> NaN (optional)
        s = s.where(s.isin(SEX_ORDER))
        df["sex"] = s
        return df

    def load_counts(fp: Path, cohort_label: str):
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        df = pd.read_csv(fp, encoding="utf-8-sig")
        if "age" not in df.columns:
            raise KeyError(f"'age' column missing in {fp.name}")

        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
        df = normalize_sex(df)

        # total counts (all rows with age_group)
        total = df["age_group"].value_counts().reindex(AGE_LABELS).fillna(0).astype(int)
        total_df = pd.DataFrame({"cohort": cohort_label, "age_group": AGE_LABELS, "n": total.values})

        # sex counts (male/female/prefer-not-to-say only)
        sex_df = df[df["sex"].isin(SEX_ORDER)].copy()
        sex_tab = (sex_df.groupby(["age_group","sex"], observed=True)
                          .size()
                          .unstack(fill_value=0)
                          .reindex(index=AGE_LABELS, columns=SEX_ORDER, fill_value=0))
        sex_tab = sex_tab.reset_index()
        sex_tab["cohort"] = cohort_label
        sex_tab = sex_tab[["cohort","age_group"] + SEX_ORDER]

        return total_df, sex_tab

    def plot_total(ax, cdf, title, ymax_use):
        x = np.arange(len(AGE_LABELS))
        ax.bar(x, cdf["n"].values, width=0.80, edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS, fontweight="bold")
        ax.set_ylabel("Number of participants", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=12)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.set_ylim(0, ymax_use)

        off = max(ymax_use*0.02, 1)
        for i, v in enumerate(cdf["n"].values):
            ax.text(i, v + off, f"{int(v)}", ha="center", va="bottom",
                    fontweight="bold", fontsize=BIG*0.65)

    def plot_sex_stacked(ax, sdf, title, ymax_use):
        x = np.arange(len(AGE_LABELS))
        bottoms = np.zeros(len(AGE_LABELS), dtype=float)

        for sex in SEX_ORDER:
            vals = sdf[sex].values.astype(float)
            ax.bar(x, vals, bottom=bottoms, width=0.80,
                   color=SEX_COLORS[sex], edgecolor="black", alpha=0.90,
                   label=SEX_LABEL[sex])
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS, fontweight="bold")
        ax.set_ylabel("Number of participants", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=12)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.set_ylim(0, ymax_use)

        off = max(ymax_use*0.02, 1)
        # annotate total
        for i, v in enumerate(bottoms):
            ax.text(i, v + off, f"{int(v)}", ha="center", va="bottom",
                    fontweight="bold", fontsize=BIG*0.65)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

    # --- load ---
    t_mobile, s_mobile = load_counts(MOBILE_FP, "Mobile")
    t_pc,     s_pc     = load_counts(PC_FP, "PC")

    # save tables
    pd.concat([t_mobile, t_pc], ignore_index=True).to_csv(
        OUT_DIR / "figS0_total_counts_mobile_vs_pc.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([s_mobile, s_pc], ignore_index=True).to_csv(
        OUT_DIR / "figS0_sex_counts_mobile_vs_pc.csv", index=False, encoding="utf-8-sig"
    )

    # y-limits (common within row)
    ymax_total = int(max(t_mobile["n"].max(), t_pc["n"].max()))
    ymax_total = int(np.ceil(ymax_total * 1.12))

    ymax_sex = int(max((s_mobile[SEX_ORDER].sum(axis=1)).max(),
                       (s_pc[SEX_ORDER].sum(axis=1)).max()))
    ymax_sex = int(np.ceil(ymax_sex * 1.12))

    # --- plot 2x2 ---
    fig = plt.figure(figsize=(26, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])

    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])
    axC = fig.add_subplot(gs[1,0])
    axD = fig.add_subplot(gs[1,1])

    plot_total(axA, t_mobile, "A  Mobile: age-bin counts (final analytic sample)", ymax_total)
    plot_total(axB, t_pc,     "B  PC: age-bin counts (final analytic sample)",     ymax_total)

    plot_sex_stacked(axC, s_mobile, "C  Mobile: age-bin counts by sex", ymax_sex)
    plot_sex_stacked(axD, s_pc,     "D  PC: age-bin counts by sex",     ymax_sex)

    fig.suptitle("Supplementary Figure S0. Participant age distributions by device and sex (final filtered samples)",
                 fontweight="bold", y=1.02)

    fig.tight_layout()

    out_png = OUT_DIR / "figS0_age_counts_device_sex.png"
    out_svg = OUT_DIR / "figS0_age_counts_device_sex.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("✅ saved:", out_png)
    print("✅ saved:", out_svg)
    print("✅ tables saved in:", OUT_DIR)


def main():
    _run_cell_047()
    _run_cell_049()
    _run_cell_051()
    _run_cell_053()
    _run_cell_057()
    _run_cell_059()
    _run_cell_061()
    _run_cell_062()
    _run_cell_063()
    _run_cell_065()
    _run_cell_067()


if __name__ == "__main__":
    main()
