import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.predictor import MatchPredictor

class BetmanTotoAnalyzer:
    def __init__(self, toto_data_path: str = "data/processed/betman_toto_25_26.csv",
                 model_pkg_path: str = "models/football_predictor_pkg.pkl",
                 betman_model_path: str = "models/betman_toto_model.pkl"):
        self.toto_df = pd.read_csv(toto_data_path)
        self.toto_df['gmTs'] = self.toto_df['gmTs'].astype(str)
        self.predictor = MatchPredictor(model_pkg_path)
        
        # 베트맨 승무패 전용 학습 모델 로드
        if os.path.exists(betman_model_path):
            pkg = joblib.load(betman_model_path)
            self.toto_models = pkg['models']
            self.toto_feature_cols = pkg['feature_cols']
        else:
            self.toto_models = None

    def get_available_rounds(self) -> List[str]:
        return sorted(self.toto_df['gmTs'].unique().tolist(), reverse=True)

    def analyze_round(self, gmTs: str) -> Dict[str, Any]:
        gmTs_str = str(gmTs)
        round_df = self.toto_df[self.toto_df['gmTs'] == gmTs_str].sort_values(by='Match_No').reset_index(drop=True)
        
        matches_analysis = []
        correct_count = 0
        trap_detected_count = 0

        for _, row in round_df.iterrows():
            m_no = int(row['Match_No'])
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            ftr = row['FTR']
            
            v_h = float(row['Vote_Home'])
            v_d = float(row['Vote_Draw'])
            v_a = float(row['Vote_Away'])

            # 5개년 전술 예측
            res = self.predictor.predict_match(h_team, a_team)
            p_h = res['probabilities']['home_win'] * 100
            p_d = res['probabilities']['draw'] * 100
            p_a = res['probabilities']['away_win'] * 100
            t_h = res['tactics']['home']
            t_a = res['tactics']['away']

            # 승무패 전용 AI 모델(투표율+괴리율 융합) 예측
            if self.toto_models:
                feat = {
                    'Vote_H': v_h, 'Vote_D': v_d, 'Vote_A': v_a,
                    'Vote_Max': max(v_h, v_d, v_a),
                    'Vote_Diff_HA': v_h - v_a,
                    'AI_Prob_H': p_h, 'AI_Prob_D': p_d, 'AI_Prob_A': p_a,
                    'Edge_H': p_h - v_h, 'Edge_D': p_d - v_d, 'Edge_A': p_a - v_a,
                    'Elo_Home': res['elo']['home'], 'Elo_Away': res['elo']['away'],
                    'Elo_Diff': res['elo']['home'] - res['elo']['away'],
                    'Pressing_H': t_h['pressing_intensity'], 'Pressing_A': t_a['pressing_intensity'],
                    'Attack_H': t_h['attack_firepower'], 'Attack_A': t_a['attack_firepower'],
                    'Finishing_H': t_h['finishing_efficiency'], 'Finishing_A': t_a['finishing_efficiency'],
                    'Defense_H': t_h['defensive_wall'], 'Defense_A': t_a['defensive_wall'],
                    'Counter_Adv_H': res['tactics']['counter_score_home'],
                    'Counter_Adv_A': res['tactics']['counter_score_away'],
                    'Trap_Risk_H': res['trap_risk']['home'], 'Trap_Risk_A': res['trap_risk']['away'],
                    'Motivation_H': res['motivation']['home'], 'Motivation_A': res['motivation']['away']
                }
                feat_df = pd.DataFrame([feat])[self.toto_feature_cols].fillna(0)
                
                l_prob = self.toto_models['LightGBM'].predict_proba(feat_df)[0]
                x_prob = self.toto_models['XGBoost'].predict_proba(feat_df)[0]
                n_prob = self.toto_models['NeuralNetwork'].predict_proba(feat_df)[0]
                
                # 앙상블 확률로 최종 보정
                final_prob = (l_prob * 0.45 + x_prob * 0.35 + n_prob * 0.20)
                p_h, p_d, p_a = final_prob[0] * 100, final_prob[1] * 100, final_prob[2] * 100

            edge_h = p_h - v_h
            edge_d = p_d - v_d
            edge_a = p_a - v_a

            # 대중 몰표 함정(Trap) 감지
            is_trap_warning = False
            trap_reason = ""

            if v_h >= 65 and edge_h <= -15:
                is_trap_warning = True
                trap_reason = f"🚨 [홈팀 몰표 함정] 대중 {v_h:.0f}% 몰림 vs AI 확률 {p_h:.0f}% (무/패 이변 주의!)"
                trap_detected_count += 1
            elif v_a >= 65 and edge_a <= -15:
                is_trap_warning = True
                trap_reason = f"🚨 [원정팀 몰표 함정] 대중 {v_a:.0f}% 몰림 vs AI 확률 {p_a:.0f}% (홈승/무 이변 주의!)"
                trap_detected_count += 1

            ai_probs = {'H': p_h, 'D': p_d, 'A': p_a}
            best_pick = max(ai_probs, key=ai_probs.get)

            sorted_picks = sorted(ai_probs.items(), key=lambda x: x[1], reverse=True)
            if is_trap_warning:
                if v_h >= 65:
                    double_pick = "D,A (무/패 복식)"
                else:
                    double_pick = "H,D (승/무 복식)"
            else:
                top1, top2 = sorted_picks[0][0], sorted_picks[1][0]
                double_pick = f"{top1},{top2}"

            is_correct = (best_pick == ftr)
            if is_correct:
                correct_count += 1

            matches_analysis.append({
                'Match_No': m_no,
                'HomeTeam': h_team,
                'AwayTeam': a_team,
                'Vote_H': v_h, 'Vote_D': v_d, 'Vote_A': v_a,
                'AI_Prob_H': round(p_h, 1), 'AI_Prob_D': round(p_d, 1), 'AI_Prob_A': round(p_a, 1),
                'Edge_H': round(edge_h, 1), 'Edge_D': round(edge_d, 1), 'Edge_A': round(edge_a, 1),
                'AI_Single_Pick': best_pick,
                'AI_Double_Pick': double_pick,
                'Actual_Result': ftr,
                'Is_Correct': is_correct,
                'Is_Trap_Warning': is_trap_warning,
                'Trap_Reason': trap_reason,
                'Insights': res['insights'][:2]
            })

        return {
            'gmTs': gmTs,
            'matches': matches_analysis,
            'total_matches': len(matches_analysis),
            'correct_count': correct_count,
            'accuracy_rate': round((correct_count / len(matches_analysis)) * 100, 1),
            'trap_detected_count': trap_detected_count
        }

if __name__ == "__main__":
    analyzer = BetmanTotoAnalyzer()
    res = analyzer.analyze_round("260051")
    print(f"\n[260051회 승무패 전용 AI 분석 결과] 14경기 중 {res['correct_count']}경기 적중 (적중률: {res['accuracy_rate']}%)")
