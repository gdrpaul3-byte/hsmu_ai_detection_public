"""Figure 2 assembly: participant overview, accuracy histogram, age-accuracy plots, and slope comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def _run_cell_000():
    # ==============================================================================
    # (8_plus | Section 1) Participant Overview (Pie) + Filtered Accuracy
    #   - SAVE TO: plots/run_20260119_192624/01_filter_overview_plus/
    #   - Figures: Pie + Histogram(no spacing) + Violin(+dots), y=0..100
    #   - Big fonts for manuscript figures
    # ==============================================================================

    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    # -----------------------------
    # Run/Section folder utils (SAVE UNDER "plots")
    # -----------------------------
    def _get_latest_run_dir(root=config.PLOTS_DIR):
        root = Path(root)
        if not root.exists():
            return None
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        return runs[-1] if runs else None

    def _make_section_dir(section_name, root=config.PLOTS_DIR, run_tag=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        if run_tag is not None:
            run_dir = root / f"run_{run_tag}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = _get_latest_run_dir(root)
            if run_dir is None:
                auto_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = root / f"run_{auto_tag}"
                run_dir.mkdir(parents=True, exist_ok=True)

        section_dir = run_dir / section_name
        section_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, section_dir

    # -----------------------------
    # Accuracy utils
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
        if mx <= 1.5:
            return s_num * 100.0
        return s_num

    # -----------------------------
    # deviceType normalize
    # -----------------------------
    def normalize_device_type(df: pd.DataFrame, device_col="deviceType", pid_col="participantId") -> pd.DataFrame:
        out = df.copy()
        if device_col in out.columns:
            s = out[device_col].astype(str).str.lower().str.strip()
            s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
            s = s.replace({"desktop": "web", "pc": "web", "browser": "web", "iphone": "mobile", "android": "mobile"})
            out[device_col] = s
        else:
            out[device_col] = np.nan

        if pid_col in out.columns and out[device_col].isna().all():
            out[device_col] = out[pid_col].astype(str).str.lower().str.extract(r"_(mobile|web)$", expand=False)

        return out

    # -----------------------------
    # Filter flags
    # -----------------------------
    def make_filter_flags(df: pd.DataFrame) -> pd.DataFrame:
        d = normalize_device_type(df.copy(), "deviceType", "participantId")

        ft = d["firstTime"].astype(str).str.lower().str.strip() if "firstTime" in d.columns else pd.Series(["nan"]*len(d), index=d.index)
        email = d["email"].astype(str).str.lower().str.strip() if "email" in d.columns else pd.Series([""]*len(d), index=d.index)
        age = pd.to_numeric(d["age"], errors="coerce") if "age" in d.columns else pd.Series([np.nan]*len(d), index=d.index)
        device = d["deviceType"].astype(str).str.lower().str.strip() if "deviceType" in d.columns else pd.Series(["nan"]*len(d), index=d.index)

        acc_col = resolve_overall_accuracy_column(d)
        acc = pd.to_numeric(d[acc_col], errors="coerce")

        reasons = []
        reasons.append(((email == "test"), "excluded: test email"))
        if "firstTime" in d.columns:
            reasons.append(((ft != "yes"), "excluded: not firstTime"))
        reasons.append((~device.isin(["mobile", "web"]), "excluded: invalid deviceType"))
        reasons.append((~age.between(20, 69), "excluded: age outside 20-69"))
        reasons.append((acc.isna(), "excluded: missing accuracy"))

        reason = pd.Series(["kept"] * len(d), index=d.index, dtype=object)
        keep = pd.Series([True] * len(d), index=d.index)

        for mask, label in reasons:
            mask = mask.fillna(False)
            to_set = keep & mask
            reason.loc[to_set] = label
            keep.loc[to_set] = False

        return pd.DataFrame({"keep": keep.astype(bool), "reason": reason.astype(str)}, index=d.index)

    # -----------------------------
    # Plot: Pie
    # -----------------------------
    def plot_pie_overview(flags: pd.DataFrame, section_dir: Path, FONT=36):
        kept_n = int(flags["keep"].sum())
        excl_n = int((~flags["keep"]).sum())

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.pie(
            [kept_n, excl_n],
            labels=[f"Kept\n(n={kept_n})", f"Excluded\n(n={excl_n})"],
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            counterclock=False,
            wedgeprops=dict(linewidth=2.0, edgecolor="white"),
            textprops=dict(fontsize=FONT, fontweight="bold"),
        )
        ax.set_title("(01-1) Participant Overview: Kept vs Excluded", fontsize=int(FONT*1.1), fontweight="bold", pad=20)

        pie_png = section_dir / "01-1_pie_kept_vs_excluded.png"
        pie_svg = section_dir / "01-1_pie_kept_vs_excluded.svg"
        plt.savefig(pie_png, dpi=300, bbox_inches="tight")
        plt.savefig(pie_svg, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        reason_counts = (
            flags.loc[~flags["keep"], "reason"]
            .value_counts()
            .rename_axis("reason")
            .reset_index(name="count")
        )
        reason_csv = section_dir / "01-2_exclusion_reasons.csv"
        reason_counts.to_csv(reason_csv, index=False, encoding="utf-8-sig")

        return pie_png, pie_svg, reason_csv, reason_counts

    # -----------------------------
    # Plot: Histogram (bars attached) + mean/median different colors, y not fixed
    # -----------------------------
    def plot_accuracy_hist_no_spacing(df_kept: pd.DataFrame, acc_col: str, section_dir: Path, bins=40, FONT=36):
        acc_pct = to_percent_series(df_kept[acc_col]).dropna().astype(float).values
        if acc_pct.size == 0:
            raise ValueError("No valid accuracy values after filtering.")

        lo = 0.0
        hi = 100.0
        bin_edges = np.linspace(lo, hi, bins + 1)

        mean_val = float(np.mean(acc_pct))
        median_val = float(np.median(acc_pct))
        std_val = float(np.std(acc_pct, ddof=1)) if acc_pct.size > 1 else np.nan
        n = int(acc_pct.size)

        fig, ax = plt.subplots(figsize=(18, 9))

        # ✅ bars 붙게
        ax.hist(acc_pct, bins=bin_edges, rwidth=1.0, linewidth=0)

        # ✅ mean/median 색 다르게
        ax.axvline(mean_val, linestyle="--", linewidth=4, color="#1f77b4", label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, linestyle=":",  linewidth=4, color="#d62728", label=f"Median: {median_val:.2f}")

        ax.set_title("(01-7) Accuracy Distribution (Filtered Participants)", fontsize=int(FONT*1.15), fontweight="bold", pad=18)
        ax.set_xlabel("Overall Accuracy (%)", fontsize=FONT, fontweight="bold")
        ax.set_ylabel("Number of Participants", fontsize=FONT, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.tick_params(axis="both", labelsize=int(FONT*0.9))

        stats_text = f"N: {n}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nSD: {std_val:.2f}"
        ax.text(
            0.03, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=int(FONT*0.75),
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6, edgecolor="none")
        )

        ax.legend(fontsize=int(FONT*0.7), loc="upper left", frameon=True)

        hist_png = section_dir / "01-7_accuracy_hist_filtered.png"
        hist_svg = section_dir / "01-7_accuracy_hist_filtered.svg"
        plt.savefig(hist_png, dpi=300, bbox_inches="tight")
        plt.savefig(hist_svg, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # bins csv
        counts, edges = np.histogram(acc_pct, bins=bin_edges)
        bins_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
        bins_csv = section_dir / "01-7_accuracy_hist_filtered__bins.csv"
        bins_df.to_csv(bins_csv, index=False, encoding="utf-8-sig")

        return {
            "n": n, "mean": mean_val, "median": median_val, "std": std_val,
            "bins": bins,
            "hist_png": str(hist_png), "hist_svg": str(hist_svg), "bins_csv": str(bins_csv),
        }

    # -----------------------------
    # Plot: Violin + dots (y=0..100 fixed)
    #  - Use matplotlib only (no seaborn dependency)
    # -----------------------------
    def plot_violin_with_dots(df_kept: pd.DataFrame, acc_col: str, section_dir: Path, FONT=36, dot_alpha=0.25):
        acc = to_percent_series(df_kept[acc_col]).dropna().astype(float).values
        if acc.size == 0:
            raise ValueError("No valid accuracy values for violin.")

        n = int(acc.size)
        mean_val = float(np.mean(acc))
        median_val = float(np.median(acc))

        fig, ax = plt.subplots(figsize=(10, 10))

        # violin (single group)
        parts = ax.violinplot([acc], positions=[1], showmeans=False, showmedians=False, showextrema=False)

        # scatter dots with jitter
        rng = np.random.default_rng(42)
        jitter = rng.normal(loc=0.0, scale=0.04, size=n)
        x = 1 + jitter
        ax.scatter(x, acc, s=30, alpha=dot_alpha)

        # mean/median lines (horizontal)
        ax.hlines(mean_val, 0.7, 1.3, linestyles="--", linewidth=4, color="#1f77b4", label=f"Mean: {mean_val:.2f}")
        ax.hlines(median_val, 0.7, 1.3, linestyles=":",  linewidth=4, color="#d62728", label=f"Median: {median_val:.2f}")

        ax.set_title("(01-8) Accuracy (Filtered) – Violin + Individual Dots", fontsize=int(FONT*1.15), fontweight="bold", pad=18)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=FONT, fontweight="bold")
        ax.set_xticks([1])
        ax.set_xticklabels(["Filtered"], fontsize=FONT, fontweight="bold")
        ax.set_ylim(0, 100)   # ✅ y-axis fixed 0..100
        ax.tick_params(axis="y", labelsize=int(FONT*0.9))

        ax.legend(fontsize=int(FONT*0.7), loc="upper left", frameon=True)

        out_png = section_dir / "01-8_accuracy_violin_filtered.png"
        out_svg = section_dir / "01-8_accuracy_violin_filtered.svg"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_svg, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # raw values for Prism
        raw_csv = section_dir / "01-8_accuracy_violin_filtered__raw.csv"
        pd.DataFrame({"accuracy_pct": acc}).to_csv(raw_csv, index=False, encoding="utf-8-sig")

        return {"violin_png": str(out_png), "violin_svg": str(out_svg), "raw_csv": str(raw_csv)}

    # -----------------------------
    # JSON safe
    # -----------------------------
    def _json_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    # -----------------------------
    # MAIN
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(8_plus | Section 1) Participant Overview (Pie) + Filtered Accuracy (SAVE: plots/)")
        print("==============================================================================\n")

        ROOT = "plots"  # ✅ outputs 대신 plots
        RUN_TAG = config.RUN_TAG
        SECTION_NAME = "01_filter_overview_plus"
        FILE_PATH = config.ENRICHED_SURVEYS

        BASE_FONT = 36  # ✅ 논문용 크게

        plt.rcParams.update({
            "font.size": BASE_FONT,
            "axes.titlesize": int(BASE_FONT * 1.15),
            "axes.labelsize": BASE_FONT,
            "legend.fontsize": int(BASE_FONT * 0.7),
            "xtick.labelsize": int(BASE_FONT * 0.9),
            "ytick.labelsize": int(BASE_FONT * 0.9),
        })

        run_dir, section_dir = _make_section_dir(SECTION_NAME, root=ROOT, run_tag=RUN_TAG)

        df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
        print(f"✅ Loaded: {FILE_PATH} rows={len(df):,}")
        print(f"📁 output section: {section_dir}")

        flags = make_filter_flags(df)
        kept = df.loc[flags["keep"]].copy()

        # save filtered raw
        kept_out = section_dir / "01-3_filtered_participants_raw.csv"
        kept.to_csv(kept_out, index=False, encoding="utf-8-sig")
        print(f"✅ saved: {kept_out}")

        # pie
        pie_png, pie_svg, reason_csv, reason_counts = plot_pie_overview(flags, section_dir, FONT=BASE_FONT)

        # hist
        acc_col = resolve_overall_accuracy_column(df)
        hist_info = plot_accuracy_hist_no_spacing(kept, acc_col=acc_col, section_dir=section_dir, bins=40, FONT=BASE_FONT)

        # violin + dots
        violin_info = plot_violin_with_dots(kept, acc_col=acc_col, section_dir=section_dir, FONT=BASE_FONT, dot_alpha=0.25)

        # meta
        meta = {
            "notebook": "ipynb8_plus",
            "section": "01_filter_overview_plus",
            "created_at": datetime.now().isoformat(),
            "root_dir": ROOT,
            "run_tag": RUN_TAG,
            "input": {
                "file": FILE_PATH,
                "rows_total": int(len(df)),
                "accuracy_column_used": acc_col,
            },
            "filter": {
                "rows_kept": int(flags["keep"].sum()),
                "rows_excluded": int((~flags["keep"]).sum()),
                "exclusion_reason_counts": reason_counts.to_dict(orient="records"),
            },
            "outputs": {
                "filtered_csv": str(kept_out),
                "pie_png": str(pie_png),
                "pie_svg": str(pie_svg),
                "exclusion_reasons_csv": str(reason_csv),
                "hist_png": hist_info["hist_png"],
                "hist_svg": hist_info["hist_svg"],
                "hist_bins_csv": hist_info["bins_csv"],
                "violin_png": violin_info["violin_png"],
                "violin_svg": violin_info["violin_svg"],
                "violin_raw_csv": violin_info["raw_csv"],
            },
            "figure_style": {
                "font_base": BASE_FONT,
                "hist_bar_spacing": "none (rwidth=1.0, linewidth=0)",
                "mean_line": {"style": "--", "color": "#1f77b4"},
                "median_line": {"style": ":", "color": "#d62728"},
                "violin_y_lim": [0, 100],
                "dot_alpha": 0.25
            }
        }

        meta_out = section_dir / "01_meta.json"
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=_json_safe), encoding="utf-8")
        print(f"\n✅ meta saved: {meta_out}")
        print("\n==================== DONE ====================")


def _run_cell_003():
    # ==============================================================================
    # (8_plus | Unified Section) Pie (Included device) + Histogram (no gaps) + Bar+SEM+Dots
    # ------------------------------------------------------------------------------
    # - Input: enriched_surveys_data.csv
    # - Output root: plots/run_<RUN_TAG>/<SECTION_NAME>/
    # - Device label: web -> PC
    # - Accuracy scale: always percent (0~100)
    # - Big text for paper: font_scale >= 3.0
    # ==============================================================================

    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from scipy.stats import ttest_ind

    # -----------------------------
    # Run/Section dir (plots)
    # -----------------------------
    def _get_latest_run_dir(root=config.PLOTS_DIR):
        root = Path(root)
        if not root.exists():
            return None
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        return runs[-1] if runs else None

    def _make_section_dir(section_name, outputs_root=config.PLOTS_DIR, run_tag=None):
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
    # Column utilities
    # -----------------------------
    def resolve_overall_accuracy_column(df: pd.DataFrame) -> str:
        if "overallAccuracy_y" in df.columns:
            return "overallAccuracy_y"
        if "overallAccuracy" in df.columns:
            return "overallAccuracy"
        if "overallAccuracy_x" in df.columns:
            return "overallAccuracy_x"
        raise KeyError("overallAccuracy column not found (overallAccuracy_y/overallAccuracy/overallAccuracy_x).")

    def to_percent_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        finite = s_num.dropna()
        if finite.empty:
            return s_num
        mx = float(finite.max())
        if mx <= 1.5:  # ratio
            return s_num * 100.0
        return s_num

    def infer_device_type(df: pd.DataFrame, device_col="deviceType", pid_col="participantId") -> pd.DataFrame:
        out = df.copy()
        if device_col in out.columns:
            s = out[device_col].astype(str).str.lower().str.strip()
            s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
            s = s.replace({
                "desktop": "web", "pc": "web", "browser": "web",
                "iphone": "mobile", "android": "mobile",
            })
            out[device_col] = s
        else:
            out[device_col] = np.nan

        if pid_col in out.columns and out[device_col].isna().all():
            extracted = out[pid_col].astype(str).str.lower().str.extract(r"_(mobile|web)$", expand=False)
            out[device_col] = extracted

        return out

    # -----------------------------
    # Fonts (paper-scale)
    # -----------------------------
    def set_paper_fonts(font_scale=3.2):
        base = 10
        plt.rcParams.update({
            "font.size": base * font_scale,
            "axes.titlesize": base * font_scale * 1.15,
            "axes.labelsize": base * font_scale * 1.05,
            "xtick.labelsize": base * font_scale,
            "ytick.labelsize": base * font_scale,
            "legend.fontsize": base * font_scale * 0.85,
            "figure.titlesize": base * font_scale * 1.2,
        })

    # -----------------------------
    # Plot 1) Pie: Included device
    # -----------------------------
    def plot_pie_included_device(included: pd.DataFrame, section_dir: Path, font_scale=3.2):
        set_paper_fonts(font_scale)

        counts = included["device_label"].value_counts().reindex(["MOBILE", "PC"]).fillna(0).astype(int)
        labels = [f"{k}\n(n={counts[k]})" for k in ["MOBILE", "PC"]]
        sizes = [counts["MOBILE"], counts["PC"]]

        fig, ax = plt.subplots(figsize=(12, 12))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            counterclock=False,
            wedgeprops=dict(linewidth=2.0, edgecolor="white"),
            pctdistance=0.72,
            labeldistance=1.10,
        )
        for t in texts: t.set_fontweight("bold")
        for at in autotexts: at.set_fontweight("bold")

        ax.set_title("(01-1) Included Participants: Mobile vs PC", fontweight="bold", pad=22)

        png_out = section_dir / "01-1_pie_included_mobile_vs_pc.png"
        svg_out = section_dir / "01-1_pie_included_mobile_vs_pc.svg"
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.savefig(svg_out, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        return {"png": str(png_out), "svg": str(svg_out), "counts": counts.to_dict()}

    # -----------------------------
    # Plot 2) Histogram: no gaps + mean/median different styles
    # -----------------------------
    def plot_hist_no_gaps(included: pd.DataFrame, section_dir: Path, bins=40, font_scale=3.2):
        set_paper_fonts(font_scale)

        x = included["accuracy_pct"].dropna().astype(float).values
        if len(x) == 0:
            raise ValueError("No accuracy values for histogram.")

        edges = np.linspace(0, 100, bins + 1)
        counts, _ = np.histogram(x, bins=edges)

        fig, ax = plt.subplots(figsize=(18, 10))

        # ✅ 막대 붙게: align='edge' + width = bin_width
        bin_width = edges[1] - edges[0]
        ax.bar(
            edges[:-1], counts,
            width=bin_width,
            align="edge",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85
        )

        mean_val = float(np.mean(x))
        median_val = float(np.median(x))
        sd_val = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan

        # ✅ mean / median line 다른 색 + 스타일
        ax.axvline(mean_val, linestyle="--", linewidth=4.0, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, linestyle=":", linewidth=4.0, label=f"Median: {median_val:.2f}")

        ax.set_title("(01-2) Accuracy Distribution (Included Participants)", fontweight="bold", pad=18)
        ax.set_xlabel("Overall Accuracy (%)", fontweight="bold")
        ax.set_ylabel("Number of Participants", fontweight="bold")
        ax.set_xlim(0, 100)

        # ✅ 통계 박스
        stats_txt = f"N: {len(x):,}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nSD: {sd_val:.2f}"
        ax.text(
            0.02, 0.98, stats_txt,
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85),
            fontweight="bold"
        )

        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        png_out = section_dir / "01-2_hist_accuracy_included_nogaps.png"
        svg_out = section_dir / "01-2_hist_accuracy_included_nogaps.svg"
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.savefig(svg_out, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        # reproducibility
        bins_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
        bins_csv = section_dir / "01-2_hist_accuracy_included_nogaps__bins.csv"
        bins_df.to_csv(bins_csv, index=False, encoding="utf-8-sig")

        return {
            "png": str(png_out), "svg": str(svg_out),
            "bins_csv": str(bins_csv),
            "N": int(len(x)), "mean": mean_val, "median": median_val, "sd": sd_val,
            "bins": int(bins)
        }

    # -----------------------------
    # Plot 3) Bar (Mean±SEM) + dots : thinner bar + de-crowd
    # -----------------------------
    def hedges_g(x, y):
        x = np.array(x); y = np.array(y)
        nx, ny = len(x), len(y)
        sx2 = np.var(x, ddof=1); sy2 = np.var(y, ddof=1)
        sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2))
        d = (np.mean(x) - np.mean(y)) / sp
        J = 1 - (3 / (4*(nx+ny) - 9))
        return float(d * J)

    def plot_bar_sem_with_dots(included: pd.DataFrame, section_dir: Path, font_scale=3.2):
        set_paper_fonts(font_scale)

        order = ["MOBILE", "PC"]
        d = included[included["device_label"].isin(order)].copy()
        d["accuracy_pct"] = pd.to_numeric(d["accuracy_pct"], errors="coerce")
        d = d.dropna(subset=["accuracy_pct"]).copy()

        g1 = d.loc[d["device_label"]=="MOBILE", "accuracy_pct"].values.astype(float)
        g2 = d.loc[d["device_label"]=="PC", "accuracy_pct"].values.astype(float)

        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        sem1 = float(np.std(g1, ddof=1) / np.sqrt(len(g1)))
        sem2 = float(np.std(g2, ddof=1) / np.sqrt(len(g2)))

        # Welch t-test
        t, p = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        g = hedges_g(g1, g2)  # MOBILE - PC

        fig, ax = plt.subplots(figsize=(14, 10))
        x = np.arange(2)

        # ✅ bar thinner
        bar_width = 0.45
        ax.bar(
            x, [m1, m2],
            yerr=[sem1, sem2],
            width=bar_width,
            capsize=12,
            edgecolor="black",
            linewidth=2.0,
            alpha=0.85
        )

        # ✅ dots: smaller + more jitter + more transparent
        rng = np.random.default_rng(42)
        jitter = 0.20
        ax.scatter(rng.normal(x[0], jitter, size=len(g1)), g1, s=22, alpha=0.18, edgecolors="none", zorder=3)
        ax.scatter(rng.normal(x[1], jitter, size=len(g2)), g2, s=22, alpha=0.18, edgecolors="none", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(order, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontweight="bold")

        # ✅ 천장 몰림 완화: 상단 여유를 둠(100에서 살짝 위로)
        ax.set_ylim(0, 105)

        ax.set_title("(01-3) Mobile vs PC Accuracy (Mean ± SEM) + Individual Dots", fontweight="bold", pad=30)

        # ✅ 텍스트 겹침 방지: 축 밖 위쪽에 한 줄로
        stat_line = f"Welch t-test p={p:.2e} | Hedges' g(MOBILE−PC)={g:.3f}"
        ax.text(
            0.5, 1.02, stat_line,
            transform=ax.transAxes,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="none", alpha=0.9),
            fontweight="bold"
        )

        # n 표시는 막대 위에 작게(박스)
        ax.text(x[0], m1 + sem1 + 2.0, f"n={len(g1)}", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
                fontweight="bold")
        ax.text(x[1], m2 + sem2 + 2.0, f"n={len(g2)}", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
                fontweight="bold")

        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        png_out = section_dir / "01-3_bar_sem_dots_mobile_vs_pc.png"
        svg_out = section_dir / "01-3_bar_sem_dots_mobile_vs_pc.svg"
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.savefig(svg_out, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        return {"png": str(png_out), "svg": str(svg_out), "welch_p": float(p), "hedges_g": float(g),
                "means": {"MOBILE": m1, "PC": m2}, "sems": {"MOBILE": sem1, "PC": sem2},
                "ns": {"MOBILE": int(len(g1)), "PC": int(len(g2))}}

    # -----------------------------
    # Main runner
    # -----------------------------
    print("==============================================================================")
    print("(8_plus | Unified) Pie + Histogram(no gaps) + Bar+SEM+Dots  [PLOTS]")
    print("==============================================================================")

    FILE_PATH = config.ENRICHED_SURVEYS
    RUN_TAG = config.RUN_TAG         # ✅ 너 run 고정
    SECTION_NAME = "01_device_accuracy_mobile_vs_pc"  # ✅ 섹션 이름(원하면 바꿔)

    OUTPUTS_ROOT = "plots"
    FONT_SCALE = 3.2
    HIST_BINS = 40

    run_dir, section_dir = _make_section_dir(SECTION_NAME, outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
    print(f"📁 output section: {section_dir}")

    df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
    print(f"✅ Loaded: {FILE_PATH} rows={len(df):,}")

    # deviceType 확보 + label 변환(web->PC)
    df = infer_device_type(df, device_col="deviceType", pid_col="participantId")
    df["device_label"] = df["deviceType"].replace({"web": "PC", "mobile": "MOBILE"})
    df["device_label"] = df["device_label"].astype(str).str.upper().replace({"WEB": "PC"})

    # accuracy(%)
    acc_col = resolve_overall_accuracy_column(df)
    df["accuracy_pct"] = to_percent_series(df[acc_col])

    # ✅ included 정의(너 기존 필터 기준이 따로 있으면 여기만 바꿔)
    # 지금은: email != test, firstTime==yes 있으면 yes만, device mobile/web만, age 20-69 있으면 적용
    included = df.copy()

    if "email" in included.columns:
        included = included[included["email"] != "test"].copy()

    if "firstTime" in included.columns:
        included = included[included["firstTime"].astype(str).str.lower().str.strip() == "yes"].copy()

    included = included[included["deviceType"].isin(["mobile", "web"])].copy()
    included = included[included["device_label"].isin(["MOBILE", "PC"])].copy()

    if "age" in included.columns:
        age = pd.to_numeric(included["age"], errors="coerce")
        included = included[(age >= 20) & (age <= 69)].copy()

    included = included.dropna(subset=["accuracy_pct"]).copy()

    # 저장용 raw/describe
    raw_out = section_dir / "01-0_included_raw.csv"
    included.to_csv(raw_out, index=False, encoding="utf-8-sig")

    desc = included.groupby("device_label")["accuracy_pct"].describe()
    desc_out = section_dir / "01-0_included_describe_by_device.csv"
    desc.to_csv(desc_out, encoding="utf-8-sig")

    print(f"✅ saved: {raw_out}")
    print(f"✅ saved: {desc_out}")

    # plots
    pie_info  = plot_pie_included_device(included, section_dir, font_scale=FONT_SCALE)
    hist_info = plot_hist_no_gaps(included, section_dir, bins=HIST_BINS, font_scale=FONT_SCALE)
    bar_info  = plot_bar_sem_with_dots(included, section_dir, font_scale=FONT_SCALE)

    # meta (Path -> str 변환)
    meta = {
        "notebook": "ipynb8_plus",
        "section": SECTION_NAME,
        "run_tag": RUN_TAG,
        "input": FILE_PATH,
        "accuracy_column_used": acc_col,
        "filters_applied": {
            "exclude_email_test_if_exists": True,
            "firstTime_yes_if_exists": True,
            "deviceType_in_mobile_web": True,
            "age_20_69_if_exists": True
        },
        "counts": {
            "included_rows": int(len(included)),
            "included_by_device": included["device_label"].value_counts().to_dict()
        },
        "outputs": {
            "raw_csv": str(raw_out),
            "describe_csv": str(desc_out),
            "pie": pie_info,
            "hist": hist_info,
            "bar": bar_info
        }
    }
    meta_out = section_dir / "01_meta.json"
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"✅ meta saved: {meta_out}")

    print("\n==================== DONE ====================")


def _run_cell_004():
    # ==============================================================================
    # (4-2 ONLY) Relationship Between Age and Accuracy (MOBILE + WEB)
    # - Big text (>=3x), y-axis fixed to 0-100
    # - Saves: raw/stats/meta + png/svg
    # ==============================================================================

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, linregress
    from pathlib import Path
    from datetime import datetime
    import json
    import shutil

    # -----------------------------
    # Run/Section 폴더 유틸
    # -----------------------------
    def _get_latest_run_dir(outputs_root=config.PLOTS_DIR):
        root = Path(outputs_root)
        if not root.exists():
            return None
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        return runs[-1] if runs else None

    def _make_section_dir(section_name, outputs_root=config.PLOTS_DIR, run_tag=None):
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
    # Accuracy 컬럼 선택/스케일 유틸
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
    # (4-2) regplot 저장 (Big text + y=0~100)
    # -----------------------------
    def save_age_accuracy_regplot_bigtext(
        df: pd.DataFrame,
        cohort_tag: str,
        section_dir: Path,
        acc_col: str,
        section_number="4-2",
        y_min=0, y_max=100,
        text_scale=3.0
    ):
        d = df.copy()
        d["age"] = pd.to_numeric(d["age"], errors="coerce")
        d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
        d = d.dropna(subset=["age", acc_col]).copy()
        d["accuracy_pct"] = to_percent_series(d[acc_col])

        base = f"{section_number}_age_accuracy_regplot_{cohort_tag}_bigtext"
        raw_out = section_dir / f"{base}__raw.csv"
        stats_out = section_dir / f"{base}__stats.csv"
        meta_out = section_dir / f"{base}__meta.json"

        keep_cols = [c for c in ["participantId", "deviceType", "firstTime", "age", acc_col, "accuracy_pct"] if c in d.columns]
        d[keep_cols].to_csv(raw_out, index=False, encoding="utf-8-sig")

        # stats
        r, p = pearsonr(d["age"], d["accuracy_pct"])
        lr = linregress(d["age"], d["accuracy_pct"])

        stats_df = pd.DataFrame([{
            "cohort": cohort_tag,
            "N": int(len(d)),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "slope_pct_per_year": float(lr.slope),
            "intercept_pct": float(lr.intercept),
            "rvalue_linreg": float(lr.rvalue),
            "pvalue_linreg": float(lr.pvalue),
            "stderr_slope": float(lr.stderr) if lr.stderr is not None else np.nan,
        }])
        stats_df.to_csv(stats_out, index=False, encoding="utf-8-sig")

        # -----------------------------
        # Big text styling (>=3x)
        # -----------------------------
        # 기준 폰트들을 전역으로 크게 올려서 "겹침" 줄이려면, figure 크기도 키워야 함.
        BASE = 12
        F_LABEL = int(BASE * text_scale)      # 축 라벨
        F_TICK  = int(10 * text_scale)        # tick
        F_SUP   = int(16 * text_scale)        # suptitle
        F_TITLE = int(12 * text_scale)        # subtitle

        sns.set_theme(style="ticks")
        fig = plt.figure(figsize=(18, 12))  # 텍스트 커진 만큼 캔버스도 키움

        ax = sns.regplot(
        data=d,
        x="age",
        y="accuracy_pct",
        scatter_kws={"alpha": 0.20, "s": 70, "edgecolor": "none"},
        line_kws={"linewidth": 4.0, "color": "red"}  # 회귀선 색상을 빨간색으로 지정
        )

        # y축 고정
        ax.set_ylim(y_min, y_max)

        # grid / spines
        ax.grid(True, linestyle=":", alpha=0.6)
        sns.despine()

        # titles
        # p 표기는 너무 길어지면 겹치기 쉬워서 subtitle에 r만, p는 아래 note로 따로 넣는 것도 가능
        ax.set_title(
            f"Filtered Data (N={len(d):,}), Pearson r={r:.3f}",
            fontsize=F_TITLE, pad=20, fontweight="bold"
        )
        fig.suptitle(
            f"({section_number}) Relationship Between Age and Accuracy ({cohort_tag.upper()})",
            fontsize=F_SUP, fontweight="bold", y=0.98
        )

        ax.set_xlabel("Age", fontsize=F_LABEL, fontweight="bold", labelpad=18)
        ax.set_ylabel("Accuracy (%)", fontsize=F_LABEL, fontweight="bold", labelpad=18)

        ax.tick_params(axis="both", labelsize=F_TICK)

        # annotation box (밖으로 빼고 싶으면 transform을 figure로)
        p_str = "p < .001" if p < 0.001 else f"p = {p:.3f}"
        note = (
            f"Pearson r = {r:.3f}, {p_str}\n"
            f"Slope = {lr.slope:.3f} (%/year)"
        )
        ax.text(
            0.02, 0.02, note,
            transform=ax.transAxes,
            fontsize=int(10 * text_scale),
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
        )

        png_out = section_dir / f"{base}.png"
        svg_out = section_dir / f"{base}.svg"
        plt.savefig(png_out, dpi=300, bbox_inches="tight")
        plt.savefig(svg_out, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        meta = {
            "section": "04_age_accuracy_bigtext",
            "section_number": section_number,
            "cohort": cohort_tag,
            "accuracy_column_used": acc_col,
            "y_scale": "percent",
            "y_range_fixed": [y_min, y_max],
            "text_scale": float(text_scale),
            "outputs": {
                "raw_csv": str(raw_out),
                "stats_csv": str(stats_out),
                "png": str(png_out),
                "svg": str(svg_out),
            }
        }
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        print(f"✅ saved: {png_out}")
        print(f"✅ saved: {svg_out}")
        print(f"✅ stats: {stats_out}")
        return r, p, lr, int(len(d))

    # -----------------------------
    # main (4-2 only)
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(4-2 ONLY) Relationship Between Age and Accuracy (MOBILE + WEB) - BIG TEXT")
        print("==============================================================================\n")

        # 원하는 저장 위치로 바꿔도 됨: "outputs" 또는 "plots"
        OUTPUTS_ROOT = "plots"
        RUN_TAG = config.RUN_TAG   # 고정하고 싶으면 사용, 아니면 None
        AGE_MIN, AGE_MAX = 20, 69

        # 섹션 폴더
        run_dir, section_dir = _make_section_dir("04_age_accuracy_bigtext", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)
        print(f"📁 section dir: {section_dir}")

        cohort_files = {
            "mobile": config.FIRST_TIMERS_MOBILE,
            "web":    config.FIRST_TIMERS_WEB,
        }

        for cohort_tag, file_path in cohort_files.items():
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ '{file_path}' not found. ({cohort_tag}) skip")
                continue

            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"\n✅ Loaded: {file_path.name} ({cohort_tag}) rows={len(df):,}")

            if "age" not in df.columns:
                print(f"❌ [{cohort_tag}] no age column. skip")
                continue

            acc_col = resolve_overall_accuracy_column(df)
            print(f"✅ [{cohort_tag}] accuracy column used: {acc_col}")

            # 필터 + 결측 제거
            d = df.copy()
            d["age"] = pd.to_numeric(d["age"], errors="coerce")
            d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
            d = d.dropna(subset=["age", acc_col]).copy()

            d_f = d[(d["age"] >= AGE_MIN) & (d["age"] <= AGE_MAX)].copy()
            print(f"✅ [{cohort_tag}] age {AGE_MIN}~{AGE_MAX}: {len(d):,} -> {len(d_f):,}")

            # (선택) 필터 CSV 저장(재사용용)
            out_main = Path(f"analysis_data_{cohort_tag}_age_filtered_{AGE_MIN}_{AGE_MAX}.csv")
            d_f.to_csv(out_main, index=False, encoding="utf-8-sig")
            shutil.copy2(out_main, section_dir / out_main.name)

            if len(d_f) < 3:
                print(f"⚠️ [{cohort_tag}] too small after filter. skip plot")
                continue

            save_age_accuracy_regplot_bigtext(
                d_f,
                cohort_tag=cohort_tag,
                section_dir=section_dir,
                acc_col=acc_col,
                section_number="4-2",
                y_min=0, y_max=100,
                text_scale=3.0
            )

        print("\n✅ DONE")


def _run_cell_005():
    # ==============================================================================
    # (8_plus | NEW Section) Slope Bar (Mobile vs PC) - ALL TEXT OUTSIDE AXES
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    import json
    from scipy.stats import norm

    # -----------------------------
    # Config
    # -----------------------------
    PLOTS_ROOT = config.PLOTS_DIR
    RUN_TAG = config.RUN_TAG
    SECTION_NAME = "02_slope_bar_mobile_vs_pc"
    SECTION_DIR = PLOTS_ROOT / f"run_{RUN_TAG}" / SECTION_NAME
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    # 논문용 크게
    FIG_W, FIG_H = 16, 10
    TITLE_FS = 34
    LABEL_FS = 28
    TICK_FS  = 26
    TEXT_FS  = 24
    LINE_W   = 3.0

    BAR_WIDTH = 0.22  # 얇게

    def p_report(p):
        if p < 0.001: return "p < .001"
        if p < 0.01:  return "p < .01"
        if p < 0.05:  return "p < .05"
        return f"p = {p:.3f}"

    # -----------------------------
    # Inputs (네가 준 값)
    # -----------------------------
    mobile_slope = -0.607475
    mobile_se    =  0.029333
    mobile_n     =  1330

    pc_slope     = -0.229760
    pc_se        =  0.058418
    pc_n         =  334

    # slope diff test
    Z = (pc_slope - mobile_slope) / np.sqrt(pc_se**2 + mobile_se**2)
    p_diff = 2 * (1 - norm.cdf(abs(Z)))

    # -----------------------------
    # Plot
    # -----------------------------
    devices = ["Mobile", "PC"]
    slopes  = [mobile_slope, pc_slope]
    ses     = [mobile_se, pc_se]
    ns      = [mobile_n, pc_n]

    x = np.arange(len(devices))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # ✅ 오른쪽 바깥 영역 확보 (legend 박스 자리)
    fig.subplots_adjust(right=0.72, top=0.88, bottom=0.12)

    bars = ax.bar(
        x, slopes,
        width=BAR_WIDTH,
        yerr=ses,
        capsize=10,
        linewidth=0,
    )

    # baseline
    ax.axhline(0, linewidth=LINE_W)

    # x tick
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(N={n})" for d, n in zip(devices, ns)], fontsize=TICK_FS)

    # labels
    ax.set_ylabel("Slope (Accuracy % per year)", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

    # y range: 막대가 천장에 붙지 않게 여유
    y_min = min(slopes) - 0.20
    y_max = 0.10
    ax.set_ylim(y_min, y_max)

    # grid / spines
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # title (figure-level)
    fig.suptitle(
        "Age–Accuracy Slope by Device (Mobile vs PC)",
        fontsize=TITLE_FS, fontweight="bold", y=0.97
    )

    # -----------------------------
    # ✅ ALL TEXT OUTSIDE AXES (오른쪽 박스)
    # -----------------------------
    info_lines = [
        "Estimates (± SE):",
        f"  Mobile: {mobile_slope:.3f} ± {mobile_se:.3f}",
        f"  PC:     {pc_slope:.3f} ± {pc_se:.3f}",
        "",
        "Slope difference test (PC − Mobile):",
        f"  Z = {Z:.2f}",
        f"  {p_report(p_diff)}",
    ]

    info_text = "\n".join(info_lines)

    # figure 좌표(0~1)로 오른쪽 영역에 배치
    fig.text(
        0.75, 0.50,
        info_text,
        ha="left", va="center",
        fontsize=TEXT_FS,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="gray", lw=1.5)
    )

    # -----------------------------
    # Save
    # -----------------------------
    base = "02-1_slope_bar_mobile_vs_pc"
    png_out = SECTION_DIR / f"{base}.png"
    svg_out = SECTION_DIR / f"{base}.svg"
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # stats/meta
    stats_df = pd.DataFrame([
        {"device": "mobile", "N": mobile_n, "slope_pct_per_year": mobile_slope, "stderr_slope": mobile_se},
        {"device": "pc",     "N": pc_n,     "slope_pct_per_year": pc_slope,     "stderr_slope": pc_se},
        {"device": "diff_pc_minus_mobile", "Z": Z, "p": p_diff}
    ])
    stats_out = SECTION_DIR / f"{base}__stats.csv"
    stats_df.to_csv(stats_out, index=False, encoding="utf-8-sig")

    meta = {
        "notebook": "ipynb8_plus",
        "section": SECTION_NAME,
        "created_at": datetime.now().isoformat(),
        "figure": {"png": str(png_out), "svg": str(svg_out)},
        "stats_csv": str(stats_out),
        "test": "Wald Z test using two independent slope estimates",
        "formula": "Z = (b_pc - b_mobile)/sqrt(se_pc^2 + se_mobile^2)",
    }
    meta_out = SECTION_DIR / f"{base}__meta.json"
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("✅ saved:")
    print(" -", png_out)
    print(" -", svg_out)
    print(" -", stats_out)
    print(" -", meta_out)
    print("p_diff raw =", p_diff)


def main():
    _run_cell_000()
    _run_cell_003()
    _run_cell_004()
    _run_cell_005()


if __name__ == "__main__":
    main()
