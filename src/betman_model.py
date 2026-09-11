import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.predictor import MatchPredictor

class BetmanTotoModelTrainer:
    """
    작년 시즌(250001회)부터 올해(260051회)까지의 배트맨 축구토토 승무패 111개 회차(1,554경기)
    대중 투표율(%) + 전술 상성 + 괴리율(Edge) 전용 AI 학습기
    """
    def __init__(self, toto_path: str = "data/processed/betman_toto_25_26.csv",
                 predictor_pkg_path: str = "models/football_predictor_pkg.pkl"):
        self.toto_df = pd.read_csv(toto_path)
        self.toto_df['gmTs'] = self.toto_df['gmTs'].astype(str)
        self.predictor = MatchPredictor(predictor_pkg_path)
        self.feature_cols = []
        self.models = {}

    def prepare_toto_features(self) -> pd.DataFrame:
        """베트맨 14경기 데이터에 5개년 전술 피처와 대중 투표율 괴리율을 결합합니다."""
        print("\n[Step 1/3] 배트맨 승무패 250001~260051회차 전술 & 투표율 피처 합성 중...")
        
        rows = []
        target_map = {'H': 0, 'D': 1, 'A': 2}
        
        for idx, row in self.toto_df.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            ftr = row['FTR']
            
            v_h = float(row['Vote_Home'])
            v_d = float(row['Vote_Draw'])
            v_a = float(row['Vote_Away'])

            # 기본 5개년 전술 예측
            res = self.predictor.predict_match(h_team, a_team)
            p_h = res['probabilities']['home_win'] * 100
            p_d = res['probabilities']['draw'] * 100
            p_a = res['probabilities']['away_win'] * 100
            
            t_h = res['tactics']['home']
            t_a = res['tactics']['away']

            # 피처 구성
            feat = {
                'gmTs': row['gmTs'],
                'Match_No': row['Match_No'],
                'Vote_H': v_h, 'Vote_D': v_d, 'Vote_A': v_a,
                'Vote_Max': max(v_h, v_d, v_a),
                'Vote_Diff_HA': v_h - v_a,
                'AI_Prob_H': p_h, 'AI_Prob_D': p_d, 'AI_Prob_A': p_a,
                'Edge_H': p_h - v_h,
                'Edge_D': p_d - v_d,
                'Edge_A': p_a - v_a,
                'Elo_Home': res['elo']['home'],
                'Elo_Away': res['elo']['away'],
                'Elo_Diff': res['elo']['home'] - res['elo']['away'],
                'Pressing_H': t_h['pressing_intensity'],
                'Pressing_A': t_a['pressing_intensity'],
                'Attack_H': t_h['attack_firepower'],
                'Attack_A': t_a['attack_firepower'],
                'Finishing_H': t_h['finishing_efficiency'],
                'Finishing_A': t_a['finishing_efficiency'],
                'Defense_H': t_h['defensive_wall'],
                'Defense_A': t_a['defensive_wall'],
                'Counter_Adv_H': res['tactics']['counter_score_home'],
                'Counter_Adv_A': res['tactics']['counter_score_away'],
                'Trap_Risk_H': res['trap_risk']['home'],
                'Trap_Risk_A': res['trap_risk']['away'],
                'Motivation_H': res['motivation']['home'],
                'Motivation_A': res['motivation']['away'],
                'Target': target_map.get(ftr, -1)
            }
            rows.append(feat)

        feat_df = pd.DataFrame(rows)
        self.feature_cols = [c for c in feat_df.columns if c not in ['gmTs', 'Match_No', 'Target']]
        return feat_df

    def train_betman_ai(self, output_pkg_path: str = "models/betman_toto_model.pkl"):
        """배트맨 승무패 111개 회차 전용 AI 모델을 학습합니다."""
        df = self.prepare_toto_features()
        df = df[df['Target'].isin([0, 1, 2])].copy()

        X = df[self.feature_cols].fillna(0)
        y = df['Target'].astype(int)

        print("\n" + "="*70)
        print(f"🔥 [배트맨 축구토토 승무패 전용 AI 풀 학습 (250001회 ~ 260051회)]")
        print(f" -> 총 학습 회차 수 : {df['gmTs'].nunique()}개 회차 (2025년 1회 ~ 2026년 51회)")
        print(f" -> 총 학습 경기 수 : {len(df)}경기 (14경기 x 111회차 전량)")
        print("="*70)

        # 1. LightGBM Toto Classifier
        print(" -> 🧠 [1/4] LightGBM 대중 심리 & 괴리율 패턴 학습 중...")
        lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=24,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(X, y)
        self.models['LightGBM'] = lgbm

        # 2. XGBoost Toto Classifier
        print(" -> 🧠 [2/4] XGBoost 몰표 함정(Trap) 분류기 학습 중...")
        xgb = XGBClassifier(
            n_estimators=160,
            learning_rate=0.035,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb.fit(X, y)
        self.models['XGBoost'] = xgb

        # 3. 딥러닝 신경망 (MLP)
        print(" -> 🧠 [3/4] 딥 뉴럴 네트워크 (MLP Deep Neural Net) 학습 중...")
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
        mlp_pipe.fit(X, y)
        self.models['NeuralNetwork'] = mlp_pipe

        # 4. Logistic Regression
        print(" -> 🧠 [4/4] 로지스틱 확률 캘리브레이터 학습 중...")
        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=500, C=0.5, random_state=42))
        ])
        lr_pipe.fit(X, y)
        self.models['LogisticRegression'] = lr_pipe

        # 적합도 및 중요도
        preds = lgbm.predict(X)
        acc = accuracy_score(y, preds)
        print(f"\n✅ [학습 완료] 배트맨 승무패 111개 회차 1,554경기 풀 학습 완료 (적합도: {acc*100:.2f}%)")

        # Feature Importance
        fi_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': lgbm.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print("\n🔥 [승무패 14경기에서 이변과 승패를 가르는 핵심 요인 TOP 8]")
        for i, r in fi_df.head(8).iterrows():
            print(f"  {i+1:2d}. {r['Feature']:<25} : {r['Importance']}")

        # 모델 저장
        os.makedirs(os.path.dirname(output_pkg_path), exist_ok=True)
        pkg = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'feature_importances': fi_df
        }
        joblib.dump(pkg, output_pkg_path)
        print(f"\n[배트맨 승무패 전용 AI 패키지 저장 완료] -> {output_pkg_path}")

def run_betman_training():
    trainer = BetmanTotoModelTrainer()
    trainer.train_betman_ai()

if __name__ == "__main__":
    run_betman_training()
