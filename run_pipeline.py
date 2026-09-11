import os
import sys

# Windows 콘솔 출력 UTF-8 인코딩 강제 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_collector import collect_all_data
from src.feature_engineering import process_features
from src.model import run_training_pipeline
from src.predictor import MatchPredictor

def main():
    print("="*70)
    print("[EPL & La Liga AI Soccer Prediction Pipeline]")
    print("="*70)

    # 1. 데이터 수집
    print("\n[Step 1/4] 5개년 데이터 수집 중 (EPL & La Liga)...")
    league_data = collect_all_data(output_dir="data/raw")
    
    # 2. 피처 엔지니어링 (동기부여, 챔스 로테이션, Elo, 롤링 폼 등)
    print("\n[Step 2/4] 특성 공학 (Feature Engineering) 수행 중...")
    features_df, engineer = process_features(
        input_path="data/raw/football_5years_all.csv",
        output_path="data/processed/features_dataset.csv"
    )

    # 3. 머신러닝 모델 학습 및 검증
    print("\n[Step 3/4] AI 모델 학습 및 2024-25 시즌 백테스트 평가 중...")
    trainer, results = run_training_pipeline(
        data_path="data/processed/features_dataset.csv",
        engineer_state=engineer.latest_state
    )

    # 4. 실시간 예측 시뮬레이션 테스트
    print("\n[Step 4/4] 예측기 시뮬레이션 테스트...")
    predictor = MatchPredictor(pkg_path="models/football_predictor_pkg.pkl")
    
    # 샘플 매치 1: 북런던 더비 (아스날 vs 토트넘)
    sample_match = predictor.predict_match("Arsenal", "Tottenham", rest_home=3, rest_away=7, match_month=4)
    print(f"\n[샘플 예측 결과 1] {sample_match['home_team']} vs {sample_match['away_team']}")
    print(f" -> 예측 결과: {sample_match['predicted_outcome']}")
    print(f" -> 확률: 홈승 {sample_match['probabilities']['home_win']*100:.1f}% | "
          f"무승부 {sample_match['probabilities']['draw']*100:.1f}% | "
          f"원정승 {sample_match['probabilities']['away_win']*100:.1f}%")
    for insight in sample_match['insights']:
        print(f"    {insight}")

    # 샘플 매치 2: 엘 클라시코 (레알 마드리드 vs 바르셀로나)
    sample_match2 = predictor.predict_match("Real Madrid", "Barcelona", rest_home=6, rest_away=6, match_month=10)
    print(f"\n[샘플 예측 결과 2] {sample_match2['home_team']} vs {sample_match2['away_team']}")
    print(f" -> 예측 결과: {sample_match2['predicted_outcome']}")
    print(f" -> 확률: 홈승 {sample_match2['probabilities']['home_win']*100:.1f}% | "
          f"무승부 {sample_match2['probabilities']['draw']*100:.1f}% | "
          f"원정승 {sample_match2['probabilities']['away_win']*100:.1f}%")
    for insight in sample_match2['insights']:
        print(f"    {insight}")

    print("\n" + "="*70)
    print("🎉 [파이프라인 전체 완료! 'streamlit run app.py'로 대시보드를 실행할 수 있습니다.]")
    print("="*70)

if __name__ == "__main__":
    main()
