"""Preprocessing step 1: merge raw exports and aggregate participant-level metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def _run_cell_000():
    # ==============================================================================
    # (1) 데이터 전처리 및 참여자별 통계 계산 Version7 (섹션별 폴더 저장 + 재현용 CSV 강화)
    # ==============================================================================

    import pandas as pd
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    import json
    import shutil

    def _safe_read_csv(path, encodings=("utf-8", "utf-8-sig", "cp949")):
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc), enc
            except Exception as e:
                last_err = e
        raise last_err

    def _to_bool_series(x: pd.Series) -> pd.Series:
        """
        다양한 타입의 isCorrect(0/1, True/False, 'true'/'false', '1'/'0')를 안전하게 bool로 변환
        """
        if x.dtype == bool:
            return x
        # 숫자면 0/1 처리
        if np.issubdtype(x.dtype, np.number):
            return x.fillna(0).astype(int).astype(bool)

        s = x.astype(str).str.strip().str.lower()
        true_set = {"true", "1", "t", "yes", "y"}
        false_set = {"false", "0", "f", "no", "n", "nan", "none", ""}

        out = pd.Series(np.nan, index=x.index, dtype="float")
        out[s.isin(true_set)] = 1
        out[s.isin(false_set)] = 0
        # 애매한 값은 NaN으로 남김 (원하면 여기서 에러로 바꿀 수도 있음)
        return out.fillna(0).astype(int).astype(bool)

    def _require_columns(df: pd.DataFrame, required, df_name="df"):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[{df_name}] 필수 컬럼 누락: {missing}")

    def process_experiment_data(
        responses_path,
        surveys_path,
        output_main_dir=".",        # 글로벌 산출물 저장 위치(메인)
        outputs_root=config.OUTPUTS_DIR,      # 섹션별 폴더가 생기는 루트
        run_tag=None                 # None이면 자동 타임스탬프
    ):
        print("==============================================================================")
        print("(1) 데이터 전처리 및 참여자별 통계 계산")
        print("------------------------------------------------------------------------------")
        print("목적: 분리된 실험 데이터와 설문 데이터를 병합하고, 참여자별 상세 통계를 계산합니다.")
        print("==============================================================================\n")

        responses_path = Path(responses_path)
        surveys_path   = Path(surveys_path)
        output_main_dir = Path(output_main_dir)
        outputs_root = Path(outputs_root)

        # run 폴더 / 섹션 폴더
        if run_tag is None:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        section_dir = outputs_root / f"run_{run_tag}" / "01_preprocess"
        section_dir.mkdir(parents=True, exist_ok=True)

        # 1) 데이터 로딩 (인코딩 fallback)
        try:
            responses_df, enc_r = _safe_read_csv(responses_path)
            surveys_df, enc_s   = _safe_read_csv(surveys_path)
            print(f"✅ 파일 로딩 성공 (responses: {enc_r}, surveys: {enc_s})")
        except FileNotFoundError as e:
            print(f"❌ 파일을 찾을 수 없습니다: {e}")
            return
        except Exception as e:
            print(f"❌ CSV 로딩 실패: {e}")
            return

        # 2) 필수 컬럼 체크
        try:
            _require_columns(responses_df, ["participantId", "trial", "isCorrect", "rt", "imageType"], "responses_df")
            _require_columns(surveys_df,   ["participantId"], "surveys_df")
        except Exception as e:
            print(f"❌ 컬럼 체크 실패: {e}")
            return

        # 3) 설문 완료자 필터
        completed_ids = surveys_df["participantId"].dropna().unique()
        filtered_responses_df = responses_df[responses_df["participantId"].isin(completed_ids)].copy()

        # 타입 정리
        filtered_responses_df["isCorrect"] = _to_bool_series(filtered_responses_df["isCorrect"])
        filtered_responses_df["rt"] = pd.to_numeric(filtered_responses_df["rt"], errors="coerce")

        print(f"✅ 설문 완료자 {len(completed_ids)}명 기준으로 trial 데이터 필터링 완료")
        print(f"   - responses 원본 row: {len(responses_df):,}")
        print(f"   - responses 필터 row: {len(filtered_responses_df):,}\n")

        # 4) 참여자별 통계 계산
        def calculate_participant_stats(group: pd.DataFrame) -> pd.Series:
            # practice vs main 구분
            practice_mask = group["trial"].astype(str).str.startswith("Practice_", na=False)
            practice_trials = group[practice_mask]
            main_trials = group[~practice_mask]

            stats = {}

            # 연습 시행 통계
            if not practice_trials.empty:
                stats["practice_accuracy"] = practice_trials["isCorrect"].mean()
                stats["practice_avg_rt"] = practice_trials["rt"].mean()

                practice_correct = practice_trials[practice_trials["isCorrect"]]
                practice_incorrect = practice_trials[~practice_trials["isCorrect"]]
                stats["practice_avg_rt_correct"] = practice_correct["rt"].mean() if not practice_correct.empty else np.nan
                stats["practice_avg_rt_incorrect"] = practice_incorrect["rt"].mean() if not practice_incorrect.empty else np.nan
                stats["practice_n"] = len(practice_trials)
            else:
                stats.update({
                    "practice_accuracy": np.nan,
                    "practice_avg_rt": np.nan,
                    "practice_avg_rt_correct": np.nan,
                    "practice_avg_rt_incorrect": np.nan,
                    "practice_n": 0
                })

            # 본 실험 통계
            if not main_trials.empty:
                correct_responses = main_trials[main_trials["isCorrect"]]
                incorrect_responses = main_trials[~main_trials["isCorrect"]]

                stats["avg_rt_correct"] = correct_responses["rt"].mean() if not correct_responses.empty else np.nan
                stats["avg_rt_incorrect"] = incorrect_responses["rt"].mean() if not incorrect_responses.empty else np.nan
                stats["main_n"] = len(main_trials)

                # imageType별
                for key, label in [("ai_chatgpt", "chatgpt"), ("ai_gemini", "gemini")]:
                    sub = main_trials[main_trials["imageType"] == key]
                    stats[f"accuracy_{label}"] = sub["isCorrect"].mean() if not sub.empty else np.nan
                    stats[f"avg_rt_{label}"] = sub["rt"].mean() if not sub.empty else np.nan
                    stats[f"n_{label}"] = len(sub)

                # 전체 정확도도 같이 저장(많이 쓰니까)
                stats["overallAccuracy"] = main_trials["isCorrect"].mean()
                stats["overallAvgRT"] = main_trials["rt"].mean()
            else:
                stats.update({
                    "avg_rt_correct": np.nan,
                    "avg_rt_incorrect": np.nan,
                    "main_n": 0,
                    "accuracy_chatgpt": np.nan,
                    "avg_rt_chatgpt": np.nan,
                    "n_chatgpt": 0,
                    "accuracy_gemini": np.nan,
                    "avg_rt_gemini": np.nan,
                    "n_gemini": 0,
                    "overallAccuracy": np.nan,
                    "overallAvgRT": np.nan,
                })

            return pd.Series(stats)

        # (섹션1) participant_stats 계산 부분만 이렇게 바꿔줘
        print("⏳ 참여자별 상세 통계치 계산 중...")
        g = filtered_responses_df.groupby("participantId", sort=False)[["trial", "isCorrect", "rt", "imageType"]]
        participant_stats = (
            g.apply(calculate_participant_stats)
             .reset_index()
        )
        print("✅ 통계치 계산 완료\n")


        # 5) 최종 데이터셋 생성 (surveys + stats)
        enriched_surveys_df = pd.merge(surveys_df, participant_stats, on="participantId", how="left")
        print("✅ 최종 데이터셋(enriched_surveys_df) 생성 완료\n")

        # 6) 저장: 글로벌(메인) + 섹션 폴더(복사본/로그)
        output_main_dir.mkdir(parents=True, exist_ok=True)

        out_filtered = output_main_dir / config.FILTERED_RESPONSES
        out_stats_wide = output_main_dir / config.PARTICIPANT_STATS_WIDE
        out_stats_long = output_main_dir / config.PARTICIPANT_STATS_LONG
        out_enriched = output_main_dir / config.ENRICHED_SURVEYS

        # 재현용 저장(UTF-8-SIG: 엑셀 한글 호환)
        filtered_responses_df.to_csv(out_filtered, index=False, encoding="utf-8-sig")
        participant_stats.to_csv(out_stats_wide, index=False, encoding="utf-8-sig")

        # Prism/Excel 편한 long format도 추가 저장
        stats_long = participant_stats.melt(id_vars=["participantId"], var_name="metric", value_name="value")
        stats_long.to_csv(out_stats_long, index=False, encoding="utf-8-sig")

        enriched_surveys_df.to_csv(out_enriched, index=False, encoding="utf-8-sig")

        # 섹션 폴더에 복사본 저장
        for p in [out_filtered, out_stats_wide, out_stats_long, out_enriched]:
            shutil.copy2(p, section_dir / p.name)

        # 메타/로그 저장(재현성)
        meta = {
            "run_tag": run_tag,
            "section": "01_preprocess",
            "inputs": {
                "responses_path": str(responses_path),
                "surveys_path": str(surveys_path),
            },
            "outputs_main": {
                "filtered_responses_completed": str(out_filtered),
                "participant_stats_wide": str(out_stats_wide),
                "participant_stats_long": str(out_stats_long),
                "enriched_surveys_data": str(out_enriched),
            },
            "counts": {
                "responses_rows_raw": int(len(responses_df)),
                "responses_rows_filtered": int(len(filtered_responses_df)),
                "surveys_rows": int(len(surveys_df)),
                "n_completed_ids": int(len(completed_ids)),
            },
            "notes": [
                "filtered_responses_completed.csv = 설문 완료자 기준 trial-level 데이터(재현/검증용)",
                "participant_stats_long.csv = Prism에서 metric별로 그래프 재구성하기 쉬운 형태",
            ],
        }
        (section_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        print("📁 저장 완료:")
        print(f" - (메인) {out_filtered.name}")
        print(f" - (메인) {out_stats_wide.name}")
        print(f" - (메인) {out_stats_long.name}")
        print(f" - (메인) {out_enriched.name}")
        print(f" - (섹션 폴더) {section_dir}\n")


    if __name__ == "__main__":
        responses_file = config.RAW_RESPONSES
        surveys_file = config.RAW_SURVEYS

        # 글로벌 산출물은 메인 디렉토리에 저장(네 요구사항)
        process_experiment_data(
            responses_path=responses_file,
            surveys_path=surveys_file,
            output_main_dir=".",   # 메인
            outputs_root=config.OUTPUTS_DIR # 섹션별 폴더 루트
        )


def main():
    _run_cell_000()


if __name__ == "__main__":
    main()
