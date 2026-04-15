"""Figure 5 assembly: reaction-time and verification-cost visualizations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_016():
    # ==============================================================================
    # (Figure 5) Verification cost: Reaction time (RT) analyses
    # ------------------------------------------------------------------------------
    # Outputs:
    #   plots/run_20260119_192624/05_verification_cost_rt/
    #     - fig5_verification_cost_rt.png/.svg
    #     - 5-0_prepped_mobile.csv / 5-0_prepped_pc.csv
    #     - 5-1_rt_correctness_summary.csv
    #     - 5-2_rt_accuracy_scatter_raw.csv
    #     - 5-3_rt_by_confidence_summary_mobile.csv
    #     - meta.json
    # ------------------------------------------------------------------------------
    # Panels:
    #   A) Mean RT (ms): Correct vs Incorrect (Mobile / PC)  [bar+SEM + dots]
    #   B) Accuracy vs Mean RT (participant-level; Mobile vs PC)
    #   C) Mobile: Mean RT by AI Confidence level (1–5)      [dots + mean±SEM]
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
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "05_verification_cost_rt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    COHORT_FILES = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "pc":     config.WEB_AGE_FILTERED,  # web -> pc label
    }

    # big fonts (>=3x)
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

    def confidence_to_score(s: pd.Series) -> pd.Series:
        # 1..5 (ordinal)
        mapping = {
            "very-not-confident": 1,
            "not-confident": 2,
            "neutral": 3,
            "confident": 4,
            "very-confident": 5,
        }
        return s.astype(str).str.lower().str.strip().map(mapping)

    def choose_rt_columns(df: pd.DataFrame):
        """
        Prefer participant-level RT columns already computed:
          - overallAvgRT
          - avg_rt_correct
          - avg_rt_incorrect
        If not available, returns Nones and you can add a fallback merge later.
        """
        overall = "overallAvgRT" if "overallAvgRT" in df.columns else None
        rt_c = "avg_rt_correct" if "avg_rt_correct" in df.columns else None
        rt_i = "avg_rt_incorrect" if "avg_rt_incorrect" in df.columns else None
        return overall, rt_c, rt_i

    def cap_outliers(y, q=0.99):
        """For plotting only: cap at q-quantile to avoid a few huge RT outliers dominating."""
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return None
        return float(np.quantile(y, q))

    # -----------------------------
    # Load + prep
    # -----------------------------
    prepped = {}
    for tag, path in COHORT_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        acc_col = resolve_accuracy_col(df)
        df["accuracy_pct"] = to_percent_series(df[acc_col])

        # confidence score
        if "aiConfidence" in df.columns:
            df["confidence_score"] = confidence_to_score(df["aiConfidence"])
        else:
            df["confidence_score"] = np.nan

        # RT columns
        overall_rt_col, rt_c_col, rt_i_col = choose_rt_columns(df)

        # keep numeric
        df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"], errors="coerce")
        if overall_rt_col:
            df[overall_rt_col] = pd.to_numeric(df[overall_rt_col], errors="coerce")
        if rt_c_col:
            df[rt_c_col] = pd.to_numeric(df[rt_c_col], errors="coerce")
        if rt_i_col:
            df[rt_i_col] = pd.to_numeric(df[rt_i_col], errors="coerce")
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

        # minimal columns to save
        keep_cols = [c for c in [
            "participantId", "age", "accuracy_pct", "aiConfidence", "confidence_score",
            overall_rt_col, rt_c_col, rt_i_col
        ] if c is not None and c in df.columns]

        out_csv = OUT_DIR / f"5-0_prepped_{tag}.csv"
        df[keep_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
        prepped[tag] = (df, out_csv, {"acc_col": acc_col, "rt_cols": (overall_rt_col, rt_c_col, rt_i_col)})

    print("✅ Prepped saved:")
    for tag, (_, out_csv, meta) in prepped.items():
        print(" -", tag, out_csv, meta)

    # -----------------------------
    # Panel A: Correct vs Incorrect RT (Mobile/PC) bar+SEM + dots
    # -----------------------------
    def make_rt_correctness_summary(df, tag):
        _, rt_c, rt_i = choose_rt_columns(df)
        if rt_c is None or rt_i is None:
            raise KeyError(f"[{tag}] Missing avg_rt_correct/avg_rt_incorrect columns. (Add fallback if needed.)")

        d = df[[rt_c, rt_i]].copy()
        d = d.dropna()
        if len(d) < 10:
            raise ValueError(f"[{tag}] Too few complete RT rows: N={len(d)}")

        return pd.DataFrame({
            "device": [tag, tag],
            "condition": ["Correct", "Incorrect"],
            "mean_rt_ms": [d[rt_c].mean(), d[rt_i].mean()],
            "sem_rt_ms":  [d[rt_c].std(ddof=1)/np.sqrt(len(d)), d[rt_i].std(ddof=1)/np.sqrt(len(d))],
            "n": [len(d), len(d)]
        })

    rt_sum = pd.concat([
        make_rt_correctness_summary(prepped["mobile"][0], "Mobile"),
        make_rt_correctness_summary(prepped["pc"][0], "PC"),
    ], ignore_index=True)

    rt_sum_out = OUT_DIR / "5-1_rt_correctness_summary.csv"
    rt_sum.to_csv(rt_sum_out, index=False, encoding="utf-8-sig")
    print("✅ Saved:", rt_sum_out)

    # -----------------------------
    # Panel B: Accuracy vs Mean RT scatter (Mobile vs PC)
    # -----------------------------
    scatter_rows = []
    for tag, (df, _, meta) in prepped.items():
        overall_rt, _, _ = meta["rt_cols"]
        if overall_rt is None:
            continue
        d = df[["accuracy_pct", overall_rt]].dropna().copy()
        d["device"] = "Mobile" if tag == "mobile" else "PC"
        d = d.rename(columns={overall_rt: "mean_rt_ms"})
        scatter_rows.append(d)

    scatter_df = pd.concat(scatter_rows, ignore_index=True)
    scatter_out = OUT_DIR / "5-2_rt_accuracy_scatter_raw.csv"
    scatter_df.to_csv(scatter_out, index=False, encoding="utf-8-sig")
    print("✅ Saved:", scatter_out)

    # -----------------------------
    # Panel C: Mobile RT by confidence level (1–5)
    # -----------------------------
    mob = prepped["mobile"][0].copy()
    overall_rt_m, _, _ = prepped["mobile"][2]["rt_cols"]
    if overall_rt_m is None:
        raise KeyError("Mobile is missing overallAvgRT (mean RT).")

    c_df = mob[["confidence_score", overall_rt_m]].dropna().copy()
    c_df = c_df[(c_df["confidence_score"]>=1) & (c_df["confidence_score"]<=5)].copy()

    conf_summary = (c_df.groupby("confidence_score")[overall_rt_m]
                    .agg(n="count", mean="mean", std="std")
                    .reset_index())
    conf_summary["sem"] = conf_summary["std"] / np.sqrt(conf_summary["n"])
    conf_out = OUT_DIR / "5-3_rt_by_confidence_summary_mobile.csv"
    conf_summary.to_csv(conf_out, index=False, encoding="utf-8-sig")
    print("✅ Saved:", conf_out)

    # -----------------------------
    # Make Figure 5 (A–C)
    # -----------------------------
    fig = plt.figure(figsize=(22, 16))

    # --- A: bar+SEM + dots (Mobile/PC × Correct/Incorrect)
    axA = fig.add_subplot(2, 2, 1)
    axA.set_title("A  Verification cost: RT by correctness", fontweight="bold", pad=12)

    # positions: 0,1 for Mobile; 3,4 for PC (gap between)
    pos_map = {
        ("Mobile", "Correct"): 0,
        ("Mobile", "Incorrect"): 1,
        ("PC", "Correct"): 3,
        ("PC", "Incorrect"): 4,
    }
    xpos = [pos_map[(r["device"], r["condition"])] for _, r in rt_sum.iterrows()]
    means = rt_sum["mean_rt_ms"].values
    sems  = rt_sum["sem_rt_ms"].values

    axA.bar(xpos, means, yerr=sems, capsize=8, width=0.75, edgecolor="black", alpha=0.85)

    # overlay dots (participant-level), capped for display
    rng = np.random.default_rng(42)
    for device in ["Mobile","PC"]:
        df = prepped["mobile"][0] if device=="Mobile" else prepped["pc"][0]
        _, rt_c, rt_i = choose_rt_columns(df)
        dd = df[[rt_c, rt_i]].dropna().copy()

        # cap y for visualization
        y_cap = cap_outliers(np.concatenate([dd[rt_c].values, dd[rt_i].values]), q=0.99)

        for cond, col in [("Correct", rt_c), ("Incorrect", rt_i)]:
            x0 = pos_map[(device, cond)]
            y = dd[col].values.astype(float)
            if y_cap is not None:
                y = np.clip(y, 0, y_cap)
            x = x0 + rng.uniform(-0.18, 0.18, size=len(y))
            axA.scatter(x, y, s=10, alpha=0.12, edgecolors="none")

    axA.set_xticks([0,1,3,4])
    axA.set_xticklabels(["Mobile\nCorrect","Mobile\nIncorrect","PC\nCorrect","PC\nIncorrect"], fontweight="bold")
    axA.set_ylabel("Mean RT (ms)", fontweight="bold")
    axA.grid(True, axis="y", linestyle=":", alpha=0.4)

    # --- B: Accuracy vs Mean RT scatter
    axB = fig.add_subplot(2, 2, 2)
    axB.set_title("B  Accuracy vs mean RT (participant-level)", fontweight="bold", pad=12)

    colors = {"Mobile":"#4285F4", "PC":"#34A853"}
    for dev in ["Mobile","PC"]:
        dd = scatter_df[scatter_df["device"]==dev].dropna()
        axB.scatter(dd["mean_rt_ms"], dd["accuracy_pct"], s=12, alpha=0.18, color=colors[dev], edgecolors="none", label=dev)

    # optional trend lines (linear fit per device)
    for dev in ["Mobile","PC"]:
        dd = scatter_df[scatter_df["device"]==dev].dropna()
        if len(dd) >= 20:
            x = dd["mean_rt_ms"].values
            y = dd["accuracy_pct"].values
            b1, b0 = np.polyfit(x, y, 1)
            xx = np.linspace(np.min(x), np.max(x), 200)
            yy = b1*xx + b0
            axB.plot(xx, yy, linewidth=3, color=colors[dev])

    # annotate Spearman (overall)
    rho, p = spearmanr(scatter_df["mean_rt_ms"], scatter_df["accuracy_pct"], nan_policy="omit")
    p_txt = "p < .001" if p < 0.001 else f"p = {p:.3f}"
    axB.text(
        0.02, 0.02,
        f"Spearman ρ = {rho:.2f}, {p_txt}",
        transform=axB.transAxes,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    axB.set_xlabel("Mean RT (ms)", fontweight="bold")
    axB.set_ylabel("Accuracy (%)", fontweight="bold")
    axB.set_ylim(0, 100)
    axB.grid(True, linestyle=":", alpha=0.35)
    axB.legend(frameon=True, loc="upper right")

    # --- C: Mobile RT by confidence (1–5)
    axC = fig.add_subplot(2, 2, 3)
    axC.set_title("C  Mobile: Mean RT by AI confidence level", fontweight="bold", pad=12)

    # dot cloud
    y_cap = cap_outliers(c_df[overall_rt_m].values, q=0.99)
    for lvl in range(1,6):
        vals = c_df.loc[c_df["confidence_score"]==lvl, overall_rt_m].values.astype(float)
        if y_cap is not None:
            vals = np.clip(vals, 0, y_cap)
        x = lvl + rng.uniform(-0.18, 0.18, size=len(vals))
        axC.scatter(x, vals, s=10, alpha=0.12, edgecolors="none")

    # mean±SEM line
    conf_summary = conf_summary.sort_values("confidence_score")
    axC.errorbar(conf_summary["confidence_score"], conf_summary["mean"], yerr=conf_summary["sem"],
                 fmt="o-", linewidth=3, markersize=8, capsize=6)

    # correlation annotation
    rho_c, p_c = spearmanr(c_df["confidence_score"], c_df[overall_rt_m], nan_policy="omit")
    p_c_txt = "p < .001" if p_c < 0.001 else f"p = {p_c:.3f}"
    axC.text(
        0.02, 0.02,
        f"Spearman ρ = {rho_c:.2f}, {p_c_txt}",
        transform=axC.transAxes,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        fontweight="bold"
    )

    axC.set_xlim(0.5, 5.5)
    axC.set_xticks([1,2,3,4,5])
    axC.set_xlabel("AI confidence (1–5)", fontweight="bold")
    axC.set_ylabel("Mean RT (ms)", fontweight="bold")
    axC.grid(True, axis="y", linestyle=":", alpha=0.35)

    # (optional) empty panel for later
    axD = fig.add_subplot(2, 2, 4)
    axD.axis("off")
    axD.text(0.5, 0.5, "Panel reserved (optional)\n(e.g., RT by Exposure or by Sex)", ha="center", va="center",
             bbox=dict(boxstyle="round", fc="#FFF9E5", alpha=0.9))

    fig.suptitle("Figure 5. Verification cost in real-vs-AI discrimination (Reaction time)", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])

    fig_png = OUT_DIR / "fig5_verification_cost_rt.png"
    fig_svg = OUT_DIR / "fig5_verification_cost_rt.svg"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # -----------------------------
    # Meta
    # -----------------------------
    meta = {
        "figure": "Figure 5 (RT / verification cost)",
        "created_at": datetime.now().isoformat(),
        "inputs": COHORT_FILES,
        "outputs": {
            "fig_png": str(fig_png),
            "fig_svg": str(fig_svg),
            "prepped_mobile_csv": str(prepped["mobile"][1]),
            "prepped_pc_csv": str(prepped["pc"][1]),
            "rt_correctness_summary_csv": str(rt_sum_out),
            "rt_accuracy_scatter_raw_csv": str(scatter_out),
            "rt_by_confidence_summary_csv": str(conf_out),
        },
        "notes": [
            "Panel A uses avg_rt_correct/avg_rt_incorrect if present; dots are capped at 99th percentile for display only.",
            "Panel B uses overallAvgRT as mean RT; y-axis fixed to 0–100 for accuracy.",
            "Panel C focuses on mobile cohort and uses confidence_score mapped from aiConfidence.",
        ],
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("✅ Saved Figure 5 to:", OUT_DIR)
    print(" -", fig_png)
    print(" -", fig_svg)


def _run_cell_017():
    # ==============================================================================
    # Figure 5 (Story version): Verification cost by error type using trial-level RT
    # - Main: Mobile
    # - Supp: PC (web)
    # ------------------------------------------------------------------------------
    # Outputs:
    #   plots/run_20260119_192624/05_verification_cost_story/
    #     - fig5_mobile.png/.svg
    #     - figS5_pc.png/.svg
    #     - mobile_participant_costs.csv
    #     - pc_participant_costs.csv
    #     - meta.json
    # ==============================================================================
    import os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    RUN_TAG = config.RUN_TAG
    OUT_DIR = config.PLOTS_DIR / f"run_{config.RUN_TAG}" / "05_verification_cost_story"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    RESP_PATH = config.RAW_RESPONSES
    MOBILE_PATH = config.MOBILE_AGE_FILTERED
    PC_PATH = config.WEB_AGE_FILTERED   # web -> pc

    # --- big fonts (paper) ---
    FONT_SCALE = 3.0
    BASE = 10
    BIG = config.apply_plot_style(font_scale=FONT_SCALE, base=BASE, legend_scale=0.75)

    # -----------------------------
    # Helpers: map confidence
    # -----------------------------
    def confidence_to_score(s: pd.Series) -> pd.Series:
        mapping = {
            "very-not-confident": 1,
            "not-confident": 2,
            "neutral": 3,
            "confident": 4,
            "very-confident": 5,
        }
        return s.astype(str).str.lower().str.strip().map(mapping)

    def infer_device_from_trial_df(r: pd.DataFrame):
        # If deviceType exists use it, else infer from participantId suffix
        if "deviceType" in r.columns:
            d = r["deviceType"].astype(str).str.lower().str.strip()
            d = d.replace({"web":"pc", "desktop":"pc", "browser":"pc", "mobile":"mobile"})
            return d
        if "participantId" in r.columns:
            pid = r["participantId"].astype(str).str.lower()
            suf = pid.str.extract(r"_(mobile|web)$", expand=False)
            return suf.replace({"web":"pc"})
        return pd.Series([np.nan]*len(r), index=r.index)

    # -----------------------------
    # Helpers: infer truth/response columns robustly
    # -----------------------------
    def find_first(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    def normalize_truth_response(resp_df: pd.DataFrame):
        """
        Goal: create two boolean columns:
          - truth_is_ai: True if ground-truth is AI
          - resp_is_ai:  True if participant responded "AI"
        Also keep rt_ms numeric.
        This function tries common column names; if not found, prints columns and raises.
        """
        df = resp_df.copy()

        # RT column
        rt_col = find_first(df.columns, ["rt", "RT", "response_time", "reaction_time", "time_ms"])
        if rt_col is None:
            raise KeyError(f"RT column not found. Available cols: {list(df.columns)[:50]}")
        df["rt_ms"] = pd.to_numeric(df[rt_col], errors="coerce")

        # response (AI vs REAL)
        resp_col = find_first(df.columns, ["response", "answer", "choice", "guess", "userAnswer", "user_answer"])
        # truth (AI vs REAL)
        truth_col = find_first(df.columns, ["truth", "groundTruth", "ground_truth", "label", "isAI", "is_ai", "is_fake", "is_real"])

        if resp_col is None or truth_col is None:
            print("Available columns:", list(df.columns))
            raise KeyError("Could not infer response/truth columns. "
                           "Need columns for participant response and ground truth (AI vs REAL).")

        # normalize response
        r = df[resp_col].astype(str).str.lower().str.strip()
        # common encodings
        resp_is_ai = r.isin(["ai", "fake", "generated", "1", "true"])
        resp_is_real = r.isin(["real", "human", "0", "false"])
        # if response is boolean-like but inverted, fallback using keywords
        df["resp_is_ai"] = np.where(resp_is_ai, True, np.where(resp_is_real, False, np.nan))

        # normalize truth
        t = df[truth_col]
        # if truth is boolean/numeric already
        if pd.api.types.is_bool_dtype(t) or pd.api.types.is_numeric_dtype(t):
            # assume 1/True means AI unless column is "is_real"
            if truth_col.lower() in ["is_real"]:
                df["truth_is_ai"] = ~pd.to_numeric(t, errors="coerce").astype(bool)
            else:
                df["truth_is_ai"] = pd.to_numeric(t, errors="coerce").astype(bool)
        else:
            tt = t.astype(str).str.lower().str.strip()
            truth_ai = tt.isin(["ai","fake","generated","1","true"])
            truth_real = tt.isin(["real","human","0","false"])
            # if column is is_real in string
            if truth_col.lower() == "is_real":
                df["truth_is_ai"] = np.where(truth_real, False, np.where(truth_ai, True, np.nan))
            else:
                df["truth_is_ai"] = np.where(truth_ai, True, np.where(truth_real, False, np.nan))

        # drop practice trials if identifiable
        if "trial" in df.columns:
            tr = df["trial"].astype(str)
            df = df[~tr.str.startswith("Practice", na=False)].copy()

        # must have participantId
        if "participantId" not in df.columns:
            raise KeyError("responses_export.csv must contain participantId.")

        # drop rows with missing key
        df = df.dropna(subset=["participantId","rt_ms","truth_is_ai","resp_is_ai"]).copy()

        # device label (mobile/pc)
        df["device"] = infer_device_from_trial_df(df)

        return df

    # -----------------------------
    # Compute per-participant RT confusion cells + costs
    # -----------------------------
    def compute_participant_costs(df_trials: pd.DataFrame):
        """
        Returns participant-level table with:
          rt_TN (Real->Real), rt_FP (AI->Real), rt_FN (Real->AI), rt_TP (AI->AI)
          FP_cost = rt_FP - rt_TP
          FN_cost = rt_FN - rt_TN
        """
        d = df_trials.copy()

        # confusion labels
        truth_ai = d["truth_is_ai"].astype(bool)
        resp_ai = d["resp_is_ai"].astype(bool)

        # TN: truth real (False), resp real (False)
        d["cell"] = np.where((~truth_ai) & (~resp_ai), "TN_Real→Real",
                     np.where((truth_ai) & (~resp_ai), "FP_AI→Real",
                     np.where((~truth_ai) & (resp_ai), "FN_Real→AI",
                     np.where((truth_ai) & (resp_ai), "TP_AI→AI", "NA"))))

        # mean RT per participant per cell
        pv = d.pivot_table(index="participantId", columns="cell", values="rt_ms", aggfunc="mean")
        pv = pv.rename(columns={
            "TN_Real→Real":"rt_TN",
            "FP_AI→Real":"rt_FP",
            "FN_Real→AI":"rt_FN",
            "TP_AI→AI":"rt_TP",
        }).reset_index()

        # costs
        pv["FP_cost"] = pv["rt_FP"] - pv["rt_TP"]   # extra time when AI is mistaken as Real vs correctly judged AI
        pv["FN_cost"] = pv["rt_FN"] - pv["rt_TN"]   # extra time when Real is mistaken as AI vs correctly judged Real

        return pv

    # -----------------------------
    # Plotting: build a 3-panel figure (A/B/C)
    # -----------------------------
    def plot_fig5_story(part_df, trial_df, cohort_label, out_png, out_svg, conf_df=None):
        """
        cohort_label: "Mobile" or "PC"
        part_df: participant-level with rt_TN/rt_FP/rt_FN/rt_TP, FP_cost, FN_cost
        trial_df: trial-level with confusion cell for heatmap
        conf_df: participant-level confidence_score (optional; used only for Mobile main)
        """
        fig = plt.figure(figsize=(22, 16))

        # Panel A: RT confusion matrix (mean RT by cell)
        axA = fig.add_subplot(2, 2, 1)
        axA.set_title(f"A  RT confusion matrix ({cohort_label})", fontweight="bold", pad=12)

        # compute mean RT by cell
        tmp = trial_df.copy()
        truth_ai = tmp["truth_is_ai"].astype(bool)
        resp_ai = tmp["resp_is_ai"].astype(bool)
        tmp["cell"] = np.where((~truth_ai) & (~resp_ai), "Real→Real",
                       np.where((truth_ai) & (~resp_ai), "AI→Real",
                       np.where((~truth_ai) & (resp_ai), "Real→AI",
                       np.where((truth_ai) & (resp_ai), "AI→AI", "NA"))))

        mat = tmp.pivot_table(index="truth_is_ai", columns="resp_is_ai", values="rt_ms", aggfunc="mean")
        # arrange in logical order: rows truth Real/AI, cols resp Real/AI
        # truth_is_ai False=Real row first; resp_is_ai False=Real col first
        mat = mat.reindex(index=[False, True], columns=[False, True])
        # labels
        row_labels = ["Truth: Real", "Truth: AI"]
        col_labels = ["Resp: Real", "Resp: AI"]

        im = axA.imshow(mat.values, aspect="auto")
        axA.set_xticks([0,1]); axA.set_xticklabels(col_labels)
        axA.set_yticks([0,1]); axA.set_yticklabels(row_labels)

        # annotate each cell with mean RT
        for i in range(2):
            for j in range(2):
                v = mat.values[i,j]
                txt = "NA" if not np.isfinite(v) else f"{v:.0f} ms"
                axA.text(j, i, txt, ha="center", va="center", fontweight="bold", color="white")

        cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)
        cbar.set_label("Mean RT (ms)", rotation=90)

        # Panel B: Verification costs (FP vs FN)
        axB = fig.add_subplot(2, 2, 2)
        axB.set_title(f"B  Verification cost contrasts ({cohort_label})", fontweight="bold", pad=12)

        # compute means + SEM (participant-level)
        def mean_sem(x):
            x = pd.to_numeric(x, errors="coerce").dropna()
            if len(x) < 5:
                return np.nan, np.nan, len(x)
            return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x))), int(len(x))

        fp_mean, fp_sem, fp_n = mean_sem(part_df["FP_cost"])
        fn_mean, fn_sem, fn_n = mean_sem(part_df["FN_cost"])

        x = np.arange(2)
        axB.bar(x, [fp_mean, fn_mean], yerr=[fp_sem, fn_sem], capsize=8, width=0.6, edgecolor="black", alpha=0.85)
        axB.axhline(0, linestyle="--", linewidth=2)
        axB.set_xticks(x)
        axB.set_xticklabels([f"FP cost\n(AI→Real − AI→AI)\n(n={fp_n})",
                             f"FN cost\n(Real→AI − Real→Real)\n(n={fn_n})"],
                            fontweight="bold")
        axB.set_ylabel("ΔRT (ms)", fontweight="bold")
        axB.grid(True, axis="y", linestyle=":", alpha=0.35)

        # Panel C: Confidence vs FP cost (Mobile main only; otherwise leave blank)
        axC = fig.add_subplot(2, 2, 3)
        axC.set_title(f"C  Confidence vs FP cost ({cohort_label})", fontweight="bold", pad=12)

        if conf_df is not None:
            merged = part_df.merge(conf_df, on="participantId", how="left")
            merged = merged.dropna(subset=["confidence_score","FP_cost"]).copy()
            # jitter x
            rng = np.random.default_rng(42)
            xj = merged["confidence_score"].values + rng.normal(0, 0.08, size=len(merged))
            axC.scatter(xj, merged["FP_cost"].values, s=14, alpha=0.18, edgecolors="none")

            # trend line
            x0 = merged["confidence_score"].values.astype(float)
            y0 = merged["FP_cost"].values.astype(float)
            if len(merged) >= 20:
                b1, b0 = np.polyfit(x0, y0, 1)
                xx = np.linspace(1,5,100)
                yy = b1*xx + b0
                axC.plot(xx, yy, color="#e53935", linewidth=4)

            rho, p = spearmanr(x0, y0, nan_policy="omit")
            ptxt = "p < .001" if p < 0.001 else f"p = {p:.3f}"
            axC.text(0.02, 0.02, f"Spearman ρ = {rho:.2f}, {ptxt}",
                     transform=axC.transAxes, ha="left", va="bottom",
                     bbox=dict(boxstyle="round", fc="white", alpha=0.85),
                     fontweight="bold")

            axC.set_xlim(0.5, 5.5)
            axC.set_xticks([1,2,3,4,5])
            axC.set_xlabel("AI confidence (1–5)", fontweight="bold")
            axC.set_ylabel("FP cost (ms)", fontweight="bold")
            axC.grid(True, axis="y", linestyle=":", alpha=0.35)
        else:
            axC.axis("off")
            axC.text(0.5,0.5,"(Supplement)\nConfidence vs FP cost", ha="center", va="center",
                     bbox=dict(boxstyle="round", fc="#FFF9E5", alpha=0.9))

        # Panel D placeholder (optional): could add Age vs FP cost, or Sex split later
        axD = fig.add_subplot(2, 2, 4)
        axD.axis("off")
        axD.text(0.5,0.5,"(Optional)\nAdd: Age vs FP cost\nor Sex-stratified FP cost",
                 ha="center", va="center",
                 bbox=dict(boxstyle="round", fc="#FFF9E5", alpha=0.9))

        fig.suptitle(f"Figure 5. Verification cost in real-vs-AI discrimination ({cohort_label})", fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0,0,1,0.97])

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_svg, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # -----------------------------
    # Run (Mobile main + PC supp)
    # -----------------------------
    if not Path(RESP_PATH).exists():
        raise FileNotFoundError(f"Missing {RESP_PATH} in current directory.")

    resp_raw = pd.read_csv(RESP_PATH, encoding="utf-8-sig")
    resp = normalize_truth_response(resp_raw)

    # Load participant-level confidence for mobile
    mob = pd.read_csv(MOBILE_PATH, encoding="utf-8-sig")
    mob_conf = mob[["participantId","aiConfidence"]].copy() if "aiConfidence" in mob.columns else mob[["participantId"]].copy()
    if "aiConfidence" in mob_conf.columns:
        mob_conf["confidence_score"] = confidence_to_score(mob_conf["aiConfidence"])
    else:
        mob_conf["confidence_score"] = np.nan

    # ---- Mobile ----
    resp_mobile = resp[resp["device"]=="mobile"].copy()
    part_mobile = compute_participant_costs(resp_mobile)
    part_mobile_out = OUT_DIR / "mobile_participant_costs.csv"
    part_mobile.to_csv(part_mobile_out, index=False, encoding="utf-8-sig")

    fig5_png = OUT_DIR / "fig5_mobile.png"
    fig5_svg = OUT_DIR / "fig5_mobile.svg"
    plot_fig5_story(part_mobile, resp_mobile, "Mobile", fig5_png, fig5_svg, conf_df=mob_conf)

    # ---- PC (supp) ----
    resp_pc = resp[resp["device"]=="pc"].copy()
    part_pc = compute_participant_costs(resp_pc)
    part_pc_out = OUT_DIR / "pc_participant_costs.csv"
    part_pc.to_csv(part_pc_out, index=False, encoding="utf-8-sig")

    figS_png = OUT_DIR / "figS5_pc.png"
    figS_svg = OUT_DIR / "figS5_pc.svg"
    plot_fig5_story(part_pc, resp_pc, "PC", figS_png, figS_svg, conf_df=None)

    meta = {
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "responses_export": RESP_PATH,
            "mobile_analysis_data": MOBILE_PATH,
            "pc_analysis_data": PC_PATH,
        },
        "outputs": {
            "fig5_mobile_png": str(fig5_png),
            "fig5_mobile_svg": str(fig5_svg),
            "figS5_pc_png": str(figS_png),
            "figS5_pc_svg": str(figS_svg),
            "mobile_participant_costs": str(part_mobile_out),
            "pc_participant_costs": str(part_pc_out),
        },
        "notes": [
            "Panel A is a 2×2 RT confusion matrix (Truth×Response).",
            "Panel B shows verification costs: FP cost = RT(AI→Real)-RT(AI→AI), FN cost = RT(Real→AI)-RT(Real→Real).",
            "Panel C relates confidence (1–5) to FP cost in Mobile.",
            "PC version is saved as Supplementary (S5).",
        ]
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Saved Figure 5 story to:", OUT_DIR)
    print(" -", fig5_png)
    print(" -", figS_png)


def main():
    _run_cell_016()
    _run_cell_017()


if __name__ == "__main__":
    main()
