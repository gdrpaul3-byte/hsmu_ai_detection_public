"""Preprocessing step 2: keep first attempts and split cohorts by device type."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def _run_cell_005():
    # ==============================================================================
    # (3) 첫 참여자 데이터 필터링 + deviceType(mobile/web) 분리 + 정확도 분포 저장 (Version: 재현용 CSV 포함)
    # ------------------------------------------------------------------------------
    # 저장 규칙:
    #   - 글로벌 재사용 CSV(필터 결과): 메인 디렉토리에 저장
    #       * analysis_data_first_timers.csv
    #       * analysis_data_first_timers_mobile.csv
    #       * analysis_data_first_timers_web.csv
    #   - 섹션 결과(그림/통계/재현용 CSV/meta): outputs/run_.../03_first_time_device_split/ 저장
    #       * __raw.csv / __describe.csv / __bins.csv / .png / .svg / __meta.json
    #       * device_counts.csv / filter_summary.csv / section_meta.json
    # ------------------------------------------------------------------------------
    # deviceType 처리:
    #   - deviceType 컬럼이 있으면 정규화해서 사용
    #   - 전부 NaN이면 participantId 끝의 _mobile/_web로 추출
    #   - KEEP_ONLY_VALID_DEVICE=True이면 mobile/web만 남김
    # ------------------------------------------------------------------------------
    # 정확도 처리:
    #   - overallAccuracy는 overallAccuracy_y(0~1)를 "정식"으로 사용
    #   - 그래프는 %로 변환해 표시 (0~100)
    # ==============================================================================

    import pandas as pd
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        plt = None
        sns = None
    from pathlib import Path
    import json
    from datetime import datetime
    import shutil

    # -----------------------------
    # Run/Section 폴더 유틸
    # -----------------------------
    def _get_latest_run_dir(outputs_root=config.OUTPUTS_DIR):
        root = Path(outputs_root)
        if not root.exists():
            return None
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        return runs[-1] if runs else None

    def _make_section_dir(section_name, outputs_root=config.OUTPUTS_DIR, run_tag=None):
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
        # responses 기반 y(0~1) 우선
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
        # 0~1이면 %로
        if mx <= 1.5:
            return s_num * 100.0
        return s_num

    # -----------------------------
    # deviceType 정리/추론 유틸
    # -----------------------------
    def normalize_device_type_from_column(df: pd.DataFrame, col: str = "deviceType") -> pd.DataFrame:
        out = df.copy()
        if col in out.columns:
            s = out[col].astype(str).str.lower().str.strip()
            s = s.replace({"nan": np.nan, "none": np.nan, "null": np.nan, "": np.nan})
            # 흔한 변형 정리
            s = s.replace({
                "desktop": "web",
                "pc": "web",
                "browser": "web",
                "iphone": "mobile",
                "android": "mobile",
            })
            out[col] = s
        else:
            out[col] = np.nan
        return out

    def infer_device_type(df: pd.DataFrame, device_col="deviceType", pid_col="participantId") -> pd.DataFrame:
        out = normalize_device_type_from_column(df, col=device_col)

        if pid_col in out.columns:
            out[pid_col] = out[pid_col].astype(str)
        else:
            return out

        # 전부 NaN이면 participantId suffix에서 추출
        if out[device_col].isna().all():
            extracted = out[pid_col].str.lower().str.extract(r"_(mobile|web)$", expand=False)
            out[device_col] = extracted

        return out

    def make_device_counts_table(df: pd.DataFrame, device_col="deviceType", pid_col="participantId"):
        vc = df[device_col].value_counts(dropna=False).rename("rows").to_frame()
        if pid_col in df.columns:
            nunq = df.groupby(device_col, dropna=False)[pid_col].nunique().rename("unique_participants")
            vc = vc.join(nunq, how="left")
        return vc.reset_index().rename(columns={"index": device_col})

    # -----------------------------
    # 히스토그램 저장(재현용 CSV 포함)
    # -----------------------------
    def save_accuracy_histogram(
        df: pd.DataFrame,
        acc_col: str,
        section_dir: Path,
        section_number: str,
        title_suffix: str,
        bins: int = 20,
        use_kde: bool = True
    ):
        safe_suffix = title_suffix.lower().replace(" ", "_").replace("/", "_")
        base_filename = f"{section_number}_accuracy_histogram_{safe_suffix}"

        # percent 스케일 값 만들기
        acc_pct = to_percent_series(df[acc_col])
        plot_col = f"{acc_col}__pct"
        plot_df = df.copy()
        plot_df[plot_col] = acc_pct

        # raw 저장 (Prism/Excel: participant 단위)
        raw_out = section_dir / f"{base_filename}__raw.csv"
        raw_cols = [c for c in ["participantId", acc_col, plot_col, "deviceType", "firstTime"] if c in plot_df.columns]
        plot_df[raw_cols].to_csv(raw_out, index=False, encoding="utf-8-sig")

        # describe 저장
        desc = plot_df[plot_col].describe()
        describe_out = section_dir / f"{base_filename}__describe.csv"
        desc.round(6).to_frame("value").to_csv(describe_out, encoding="utf-8-sig")

        # bins 저장 (재현성 위해 edges 고정)
        finite_vals = plot_df[plot_col].dropna().astype(float).values
        lo = min(0.0, float(np.min(finite_vals))) if finite_vals.size else 0.0
        hi = max(100.0, float(np.max(finite_vals))) if finite_vals.size else 100.0
        bin_edges = np.linspace(lo, hi, bins + 1)
        counts, edges = np.histogram(finite_vals, bins=bin_edges)

        bins_df = pd.DataFrame({
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts
        })
        bins_out = section_dir / f"{base_filename}__bins.csv"
        bins_df.to_csv(bins_out, index=False, encoding="utf-8-sig")

        mean_val = float(np.nanmean(finite_vals)) if finite_vals.size else np.nan
        median_val = float(np.nanmedian(finite_vals)) if finite_vals.size else np.nan
        std_val = float(np.nanstd(finite_vals, ddof=1)) if finite_vals.size > 1 else np.nan

        png_out = section_dir / f"{base_filename}.png"
        svg_out = section_dir / f"{base_filename}.svg"
        plotting_available = plt is not None and sns is not None
        if plotting_available:
            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(14, 7))
            ax = sns.histplot(
                data=plot_df,
                x=plot_col,
                bins=bin_edges,
                kde=use_kde,
                alpha=0.6,
                edgecolor="black"
            )

            plt.axvline(mean_val, linestyle="--", linewidth=2.5, label=f"Mean: {mean_val:.2f}")
            plt.axvline(median_val, linestyle=":", linewidth=2.5, label=f"Median: {median_val:.2f}")

            plt.suptitle(f"({section_number}) Distribution of Overall Accuracy ({title_suffix})", fontsize=20, fontweight="bold")
            plt.xlabel("Overall Accuracy (%)", fontsize=12)
            plt.ylabel("Number of Participants", fontsize=12)

            text_str = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd. Dev.: {std_val:.2f}\nN: {int(desc['count'])}"
            ax.text(
                0.05, 0.95, text_str,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            sns.despine(left=True)
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
            plt.savefig(png_out, dpi=300, bbox_inches="tight")
            plt.savefig(svg_out, bbox_inches="tight")
            plt.show()
            plt.close(fig)

        # meta
        meta = {
            "section": "03_first_time_device_split",
            "section_number": section_number,
            "accuracy_column_used": acc_col,
            "plot_scale": "percent",
            "bins": int(bins),
            "bin_edges": edges.tolist(),
            "use_kde": bool(use_kde),
            "title_suffix": title_suffix,
            "outputs": {
                "raw_csv": str(raw_out),
                "describe_csv": str(describe_out),
                "bins_csv": str(bins_out),
                "png": str(png_out) if plotting_available else None,
                "svg": str(svg_out) if plotting_available else None,
            },
            "summary": {
                "n": int(desc["count"]),
                "mean_pct": float(mean_val) if np.isfinite(mean_val) else None,
                "median_pct": float(median_val) if np.isfinite(median_val) else None,
                "std_pct": float(std_val) if np.isfinite(std_val) else None,
                "min_pct": float(desc["min"]) if np.isfinite(desc["min"]) else None,
                "max_pct": float(desc["max"]) if np.isfinite(desc["max"]) else None,
            },
            "plotting_available": plotting_available,
        }
        meta_out = section_dir / f"{base_filename}__meta.json"
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # 콘솔용 요약 반환
        return {
            "section_number": section_number,
            "title": title_suffix,
            "n": int(desc["count"]),
            "mean_pct": float(mean_val),
            "median_pct": float(median_val),
            "std_pct": float(std_val),
            "outputs": meta["outputs"]
        }

    # -----------------------------
    # MAIN
    # -----------------------------
    if __name__ == "__main__":
        print("==============================================================================")
        print("(3) 첫 참여자 데이터 필터링 및 정확도 분포 시각화 + deviceType 분리 저장")
        print("------------------------------------------------------------------------------")
        print("목적: firstTime 필터 + deviceType(mobile/web) 정리 + 분리 저장 + 분포 그림/재현용 데이터 저장")
        print("==============================================================================\n")

        # 설정
        KEEP_ONLY_VALID_DEVICE = True
        OUTPUTS_ROOT = config.OUTPUTS_DIR
        RUN_TAG = config.RUN_TAG  # 섹션2와 같은 run 폴더 쓰려면 config.RUN_TAG 같은 값으로 지정

        # 섹션 폴더
        run_dir, section_dir = _make_section_dir("03_first_time_device_split", outputs_root=OUTPUTS_ROOT, run_tag=RUN_TAG)

        # 1) 데이터 로드
        file_path = config.ENRICHED_SURVEYS
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✅ '{file_path}' 로딩 성공")
        print(f"원본 참여자 수(rows): {len(df):,}")

        # 2) 테스트 계정 제외 + 첫 참여자만
        if "email" in df.columns:
            df = df[df["email"] != "test"].copy()

        if "firstTime" not in df.columns:
            raise KeyError("firstTime 컬럼이 없습니다. 섹션3 필터링 불가.")

        df_first = df[df["firstTime"].astype(str).str.lower().str.strip() == "yes"].copy()
        print(f"필터링 후(firstTime==yes) rows: {len(df_first):,}")

        # 3) deviceType 확보/정리
        df_first = infer_device_type(df_first, device_col="deviceType", pid_col="participantId")

        if KEEP_ONLY_VALID_DEVICE:
            before = len(df_first)
            df_first = df_first[df_first["deviceType"].isin(["mobile", "web"])].copy()
            print(f"✅ deviceType mobile/web로 제한: {before:,} -> {len(df_first):,}")

        # device 분포표 저장
        device_counts = make_device_counts_table(df_first, device_col="deviceType", pid_col="participantId")
        device_counts_out = section_dir / "device_counts.csv"
        device_counts.to_csv(device_counts_out, index=False, encoding="utf-8-sig")
        print("\n[Device Type 분포]")
        print(device_counts)

        # 4) mobile/web 분리
        df_mobile = df_first[df_first["deviceType"] == "mobile"].copy()
        df_web = df_first[df_first["deviceType"] == "web"].copy()
        print("\n[분리 결과]")
        print("mobile:", df_mobile.shape, "| web:", df_web.shape)

        # 5) 정확도 컬럼 결정
        acc_col = resolve_overall_accuracy_column(df_first)
        print(f"\n✅ accuracy column used: {acc_col} (plot will be in %)")

        # 6) 히스토그램 (전체 + 모바일 + 웹) 저장
        summaries = []
        if len(df_first) > 0:
            summaries.append(save_accuracy_histogram(
                df_first, acc_col, section_dir,
                section_number="3-1",
                title_suffix="First-Time Participants (All Devices)",
                bins=50, use_kde=True
            ))
        if len(df_mobile) > 0:
            summaries.append(save_accuracy_histogram(
                df_mobile, acc_col, section_dir,
                section_number="3-2",
                title_suffix="First-Time Participants (Mobile)",
                bins=50, use_kde=True
            ))
        else:
            print("\n⚠️ mobile 데이터가 0이라 mobile 분포는 스킵")
        if len(df_web) > 0:
            summaries.append(save_accuracy_histogram(
                df_web, acc_col, section_dir,
                section_number="3-3",
                title_suffix="First-Time Participants (Web)",
                bins=50, use_kde=True
            ))
        else:
            print("\n⚠️ web 데이터가 0이라 web 분포는 스킵")

        # 요약표 저장(논문/리포트에 바로 쓰기 좋음)
        summary_df = pd.DataFrame(summaries)
        summary_out = section_dir / "filter_summary.csv"
        summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

        # 7) 글로벌 필터 CSV 저장 (메인 디렉토리)
        out_all = Path(config.FIRST_TIMERS_ALL)
        out_mobile = Path(config.FIRST_TIMERS_MOBILE)
        out_web = Path(config.FIRST_TIMERS_WEB)

        df_first.to_csv(out_all, index=False, encoding="utf-8-sig")
        df_mobile.to_csv(out_mobile, index=False, encoding="utf-8-sig")
        df_web.to_csv(out_web, index=False, encoding="utf-8-sig")

        print(f"\n✅ (메인) '{out_all}' 저장 완료")
        print(f"✅ (메인) '{out_mobile}' 저장 완료")
        print(f"✅ (메인) '{out_web}' 저장 완료")

        # 섹션 폴더에도 복사본 저장(추적/재현성)
        for p in [out_all, out_mobile, out_web]:
            shutil.copy2(p, section_dir / p.name)

        # 섹션 메타 저장
        section_meta = {
            "section": "03_first_time_device_split",
            "inputs": {"file_path": str(file_path)},
            "filters": {
                "exclude_email_test": "email != 'test' (if exists)",
                "firstTime": "yes",
                "KEEP_ONLY_VALID_DEVICE": KEEP_ONLY_VALID_DEVICE,
            },
            "counts": {
                "rows_firstTime": int(len(df_first)),
                "rows_mobile": int(len(df_mobile)),
                "rows_web": int(len(df_web)),
            },
            "accuracy_column_used": acc_col,
            "saved_main": [str(out_all), str(out_mobile), str(out_web)],
            "saved_section_dir": str(section_dir),
            "device_counts_csv": str(device_counts_out),
            "summary_csv": str(summary_out),
        }
        (section_dir / "section_meta.json").write_text(json.dumps(section_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n📁 섹션3 저장 위치: {section_dir}")
        print(f" - device counts: {device_counts_out}")
        print(f" - summary table: {summary_out}")
        print(f" - section meta:  {section_dir / 'section_meta.json'}")


def main():
    _run_cell_005()


if __name__ == "__main__":
    main()
