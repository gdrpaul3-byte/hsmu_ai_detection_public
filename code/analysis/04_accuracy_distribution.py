"""Analysis step 4: accuracy distributions and age-accuracy relationships."""
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _get_latest_run_dir(outputs_root=config.OUTPUTS_DIR):
    root = Path(outputs_root)
    if not root.exists():
        return None
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    return runs[-1] if runs else None


def _make_section_dir(section_name, outputs_root=config.OUTPUTS_DIR, run_tag=config.RUN_TAG):
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


def resolve_overall_accuracy_column(df: pd.DataFrame) -> str:
    if "overallAccuracy_y" in df.columns:
        return "overallAccuracy_y"
    if "overallAccuracy" in df.columns:
        return "overallAccuracy"
    if "overallAccuracy_x" in df.columns:
        return "overallAccuracy_x"
    raise KeyError("overallAccuracy ?? ??? ????.")


def to_percent_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    finite = s_num.dropna()
    if finite.empty:
        return s_num
    return s_num * 100.0 if float(finite.max()) <= 1.5 else s_num


def save_accuracy_distribution(file_path, section_dir: Path, section_number="2-1"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    print(f"? '{file_path}' ?? ??")
    acc_col = resolve_overall_accuracy_column(df)
    acc_pct = to_percent_series(df[acc_col]).dropna()

    base = f"{section_number}_accuracy_distribution_all_participants"
    raw_out = section_dir / f"{base}__raw.csv"
    desc_out = section_dir / f"{base}__describe.csv"
    bins_out = section_dir / f"{base}__bins.csv"
    meta_out = section_dir / f"{base}__meta.json"

    pd.DataFrame({"accuracy_pct": acc_pct}).to_csv(raw_out, index=False, encoding="utf-8-sig")
    acc_pct.describe().round(6).to_frame("value").to_csv(desc_out, encoding="utf-8-sig")

    bins = np.arange(0, 102, 2)
    counts, edges = np.histogram(acc_pct, bins=bins)
    pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    }).to_csv(bins_out, index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(acc_pct, bins=bins, edgecolor="white")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Count")
    plt.title("Overall Accuracy Distribution")
    png_out = section_dir / f"{base}.png"
    svg_out = section_dir / f"{base}.svg"
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)

    meta_out.write_text(json.dumps({
        "section": "04_accuracy_distribution",
        "input": str(file_path),
        "accuracy_column": acc_col,
        "outputs": {
            "raw_csv": str(raw_out),
            "describe_csv": str(desc_out),
            "bins_csv": str(bins_out),
            "png": str(png_out),
            "svg": str(svg_out),
        },
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_age_accuracy_regplot(df: pd.DataFrame, cohort_tag: str, section_dir: Path, acc_col: str, section_number="4-2"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, linregress

    d = df.copy()
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d[acc_col] = pd.to_numeric(d[acc_col], errors="coerce")
    d = d.dropna(subset=["age", acc_col]).copy()
    d["accuracy_pct"] = to_percent_series(d[acc_col])

    base = f"{section_number}_age_accuracy_regplot_{cohort_tag}"
    raw_out = section_dir / f"{base}__raw.csv"
    stats_out = section_dir / f"{base}__stats.csv"
    meta_out = section_dir / f"{base}__meta.json"

    keep_cols = [c for c in ["participantId", "deviceType", "firstTime", "age", acc_col, "accuracy_pct"] if c in d.columns]
    d[keep_cols].to_csv(raw_out, index=False, encoding="utf-8-sig")

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

    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(12, 7))
    sns.regplot(
        data=d,
        x="age",
        y="accuracy_pct",
        scatter_kws={"alpha": 0.25, "edgecolor": "w", "s": 50},
        line_kws={"color": "#e53935", "linewidth": 2.8},
    )
    plt.suptitle(f"({section_number}) Relationship Between Age and Accuracy ({cohort_tag})", fontsize=16, fontweight="bold")
    plt.title(f"Filtered Data (N={len(d)}), Pearson r={r:.3f}", fontsize=12, pad=10)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    sns.despine()
    png_out = section_dir / f"{base}.png"
    svg_out = section_dir / f"{base}.svg"
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)

    meta_out.write_text(json.dumps({
        "section": "04_age_accuracy",
        "cohort": cohort_tag,
        "accuracy_column_used": acc_col,
        "outputs": {
            "raw_csv": str(raw_out),
            "stats_csv": str(stats_out),
            "png": str(png_out),
            "svg": str(svg_out),
        },
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return r, p, lr, int(len(d))


def save_pearson_report(cohort_tag: str, section_dir: Path, r: float, p: float, n_used: int, section_number="4-3"):
    p_report = "p < .001" if p < 0.001 else f"p = {p:.3f}"
    txt_out = section_dir / f"{section_number}_age_accuracy_pearson_{cohort_tag}.txt"
    txt_out.write_text(
        f"Pearson correlation report ({cohort_tag})\nN={n_used}\nr={r:.3f}\np={p:.6g} ({p_report})\n",
        encoding="utf-8",
    )
    pd.DataFrame([{
        "cohort": cohort_tag,
        "N": n_used,
        "df": n_used - 2,
        "pearson_r": r,
        "pearson_p": p,
        "p_report": p_report,
    }]).to_csv(section_dir / f"{section_number}_age_accuracy_pearson_{cohort_tag}__stats.csv", index=False, encoding="utf-8-sig")


def main():
    run_dir, section_dir = _make_section_dir("04_age_accuracy")
    save_accuracy_distribution(config.ENRICHED_SURVEYS, section_dir)

    cohort_files = {
        "mobile": config.MOBILE_AGE_FILTERED,
        "web": config.WEB_AGE_FILTERED,
    }
    summary = []
    for cohort_tag, file_path in cohort_files.items():
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        acc_col = resolve_overall_accuracy_column(df)
        r, p, lr, n_used = save_age_accuracy_regplot(df, cohort_tag, section_dir, acc_col)
        save_pearson_report(cohort_tag, section_dir, float(r), float(p), int(n_used))
        summary.append({
            "cohort": cohort_tag,
            "N_filtered": int(n_used),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "slope_pct_per_year": float(lr.slope),
            "intercept_pct": float(lr.intercept),
        })
    pd.DataFrame(summary).to_csv(section_dir / "cohort_summary.csv", index=False, encoding="utf-8-sig")
    (section_dir / "section_meta.json").write_text(json.dumps({
        "section": "04_age_accuracy",
        "run_dir": str(run_dir),
        "cohorts": [str(config.MOBILE_AGE_FILTERED), str(config.WEB_AGE_FILTERED)],
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
