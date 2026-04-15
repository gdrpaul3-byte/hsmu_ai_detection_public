"""Preprocessing step 3: filter first-timer cohorts to ages 20-69 and save final analytic cohorts."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_overall_accuracy_column(df: pd.DataFrame) -> str:
    if "overallAccuracy_y" in df.columns:
        return "overallAccuracy_y"
    if "overallAccuracy" in df.columns:
        return "overallAccuracy"
    if "overallAccuracy_x" in df.columns:
        return "overallAccuracy_x"
    raise KeyError("overallAccuracy ?? ??(overallAccuracy_y/overallAccuracy/overallAccuracy_x)? ????.")


def filter_age_range(input_path: Path, output_path: Path, cohort_tag: str, age_min: int = 20, age_max: int = 69) -> None:
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    acc_col = resolve_overall_accuracy_column(df)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    filtered = df.dropna(subset=["age", acc_col]).copy()
    filtered = filtered[(filtered["age"] >= age_min) & (filtered["age"] <= age_max)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(f"? [{cohort_tag}] saved {len(filtered):,} rows to {output_path}")


def main() -> None:
    print("==============================================================================")
    print("(3) Age filtering for final analytic cohorts")
    print("------------------------------------------------------------------------------")
    print("??: first-timer mobile/web ????? 20-69? ???? ????.")
    print("==============================================================================\n")

    filter_age_range(config.FIRST_TIMERS_MOBILE, config.MOBILE_AGE_FILTERED, "mobile")
    filter_age_range(config.FIRST_TIMERS_WEB, config.WEB_AGE_FILTERED, "web")


if __name__ == "__main__":
    main()
