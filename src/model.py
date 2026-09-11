import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 학습에 사용할 전체 178개 Feature 목록
FEATURE_COLUMNS = [
    # 1. Elo 전력 레이팅
    'Elo_Home', 'Elo_Away', 'Elo_Diff', 'Elo_Exp_Home', 'Elo_Exp_Away',
    
    # 2. 일정 및 피로도
    'Rest_Days_Home', 'Rest_Days_Away', 'Rest_Days_Diff',
    
    # 3. 다음 경기 함정 경기(Trap Game) & 동기부여 & 라이벌 더비
    'Trap_Risk_Home', 'Trap_Risk_Away', 'Trap_Risk_Diff',
    'Motivation_Home', 'Motivation_Away', 'Motivation_Diff',
    'Is_Derby',
    
    # 4. 팀별 5대 전술 장단점 & 상성(카운터) 지표
    'Tactics_Pressing_Home', 'Tactics_Pressing_Away',
    'Tactics_Attack_Home', 'Tactics_Attack_Away',
    'Tactics_Finishing_Home', 'Tactics_Finishing_Away',
    'Tactics_Defense_Home', 'Tactics_Defense_Away',
    'Tactics_SetPiece_Home', 'Tactics_SetPiece_Away',
    'Counter_Advantage_Home', 'Counter_Advantage_Away', 'Counter_Advantage_Diff',
    'Tactics_Clash_Pressing', 'Tactics_Clash_Attack_vs_Def', 'Tactics_Clash_Def_vs_Attack', 'Tactics_Clash_SetPiece',
    
    # 5. 최근 5경기 롤링 및 가중 폼
    'Home_L5_goals_scored_avg', 'Home_L5_goals_conceded_avg', 'Home_L5_points_avg',
    'Home_L5_shots_avg', 'Home_L5_shots_target_avg', 'Home_L5_shots_conceded_avg',
    'Home_L5_shots_target_conceded_avg', 'Home_L5_corners_avg', 'Home_L5_win_rate',
    'Home_L5_draw_rate', 'Home_L5_loss_rate', 'Home_L5_shot_accuracy',
    'Home_L5_weighted_points', 'Home_L5_unbeaten_streak',
    
    'Away_L5_goals_scored_avg', 'Away_L5_goals_conceded_avg', 'Away_L5_points_avg',
    'Away_L5_shots_avg', 'Away_L5_shots_target_avg', 'Away_L5_shots_conceded_avg',
    'Away_L5_shots_target_conceded_avg', 'Away_L5_corners_avg', 'Away_L5_win_rate',
    'Away_L5_draw_rate', 'Away_L5_loss_rate', 'Away_L5_shot_accuracy',
    'Away_L5_weighted_points', 'Away_L5_unbeaten_streak',
    
    # 6. 롤링 L3 & L10
    'Home_L3_points_avg', 'Away_L3_points_avg',
    'Home_L10_points_avg', 'Away_L10_points_avg',
    'Home_L10_goals_scored_avg', 'Away_L10_goals_scored_avg',
    'Home_L10_goals_conceded_avg', 'Away_L10_goals_conceded_avg',
    
    # 7. 홈 전용 / 원정 전용 폼
    'Home_AtHome_L5_points_avg', 'Home_AtHome_L5_goals_scored_avg', 'Home_AtHome_L5_goals_conceded_avg',
    'Away_AtAway_L5_points_avg', 'Away_AtAway_L5_goals_scored_avg', 'Away_AtAway_L5_goals_conceded_avg',
    
    # 8. H2H 상대전적
    'H2H_Home_Avg_Points', 'H2H_Home_Avg_GoalDiff', 'H2H_Count',
    
    # 9. 상대적 차이
    'Diff_L5_Points', 'Diff_L5_Weighted_Points', 'Diff_L5_Goals_Scored', 
    'Diff_L5_Goals_Conceded', 'Diff_L5_Shots_Target'
]

class FootballModelTrainer:
    def __init__(self, data_path: str = "data/processed/features_dataset.csv"):
        self.data_path = data_path
        self.feature_cols = FEATURE_COLUMNS
        self.models = {}
        self.best_model = None
        self.feature_importances = {}

    def train_full_5years(self):
        """5개년 3,800개 전체 경기 데이터를 100% 학습시킵니다."""
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date']).reset_index(drop=True)
        df = df[df['Target'].isin([0, 1, 2])].copy()

        X_full = df[self.feature_cols].fillna(0)
        y_full = df['Target'].astype(int)

        print("\n" + "="*70)
        print(f"🔥 [AI 5개년 전체 경기 100% 풀 러닝 (Full Training)]")
        print(f" -> 총 학습 경기 수 : {len(df)}경기 (EPL 1,900경기 + 라리가 1,900경기 전량)")
        print(f" -> 학습 시즌 범위  : 2020-21 ~ 2024-25 전 시즌")
        print("="*70)

        # 1. LightGBM Classifier (Full Data)
        print(" -> 🧠 [1/4] LightGBM Gradient Boosting 5년치 전체 학습 중...")
        lgbm = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=28,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(X_full, y_full)
        self.models['LightGBM'] = lgbm

        # 2. XGBoost Classifier (Full Data)
        print(" -> 🧠 [2/4] XGBoost Deep Tree 5년치 전체 학습 중...")
        xgb = XGBClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb.fit(X_full, y_full)
        self.models['XGBoost'] = xgb

        # 3. 딥 뉴럴 네트워크 (MLP Neural Network)
        print(" -> 🧠 [3/4] 딥러닝 신경망 (Multi-Layer Perceptron Neural Network) 학습 중...")
        mlp_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=300,
                alpha=0.001,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=15
            ))
        ])
        mlp_pipe.fit(X_full, y_full)
        self.models['NeuralNetwork'] = mlp_pipe

        # 4. Logistic Regression
        print(" -> 🧠 [4/4] 로지스틱 확률 캘리브레이터 학습 중...")
        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=500, C=0.5, random_state=42))
        ])
        lr_pipe.fit(X_full, y_full)
        self.models['LogisticRegression'] = lr_pipe

        # Feature Importance 추출
        importances = lgbm.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        self.feature_importances['LightGBM'] = fi_df

        # 전체 데이터 적합도 평가
        preds = lgbm.predict(X_full)
        acc = accuracy_score(y_full, preds)
        print(f"\n✅ [학습 완료] 5개년 3,800경기 전체 패턴 학습 완료 (기억 적합도: {acc*100:.2f}%)")
        print("\n🔥 [AI가 5년간 배운 가장 중요한 핵심 승패 결정 요인 TOP 10]")
        for i, r in fi_df.head(10).iterrows():
            print(f"  {i+1:2d}. {r['Feature']:<35} : {r['Importance']}")

    def save_model_package(self, engineer_state: dict, output_dir: str = "models"):
        os.makedirs(output_dir, exist_ok=True)
        package = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'feature_importances': self.feature_importances,
            'engineer_state': engineer_state
        }
        save_path = os.path.join(output_dir, "football_predictor_pkg.pkl")
        joblib.dump(package, save_path)
        print(f"\n[최종 AI 모델 패키지 저장 완료] -> {save_path}")

def run_training_pipeline(data_path: str = "data/processed/features_dataset.csv",
                          engineer_state: dict = None) -> Tuple[FootballModelTrainer, dict]:
    trainer = FootballModelTrainer(data_path=data_path)
    trainer.train_full_5years()
    
    if engineer_state is not None:
        trainer.save_model_package(engineer_state)
        
    return trainer, {}

if __name__ == "__main__":
    run_training_pipeline()
