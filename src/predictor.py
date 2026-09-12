import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.feature_engineering import FootballFeatureEngineer, RIVALRIES, EUROPEAN_CONTENDERS

class MatchPredictor:
    def __init__(self, pkg_path: str = "models/football_predictor_pkg.pkl"):
        if not os.path.exists(pkg_path):
            raise FileNotFoundError(f"모델 패키지 파일({pkg_path})이 존재하지 않습니다. 먼저 학습을 실행하세요.")
            
        pkg = joblib.load(pkg_path)
        self.models = pkg['models']
        self.feature_cols = pkg['feature_cols']
        self.feature_importances = pkg['feature_importances']
        self.state = pkg['engineer_state']
        self.engineer = FootballFeatureEngineer()

    def get_available_teams(self) -> list:
        elo_ratings = self.state.get('elo_ratings', {})
        return sorted(list(elo_ratings.keys()))

    def get_team_tactical_profile(self, team: str) -> Dict[str, float]:
        team_history = self.state.get('team_history', {})
        hist = team_history.get(team, [])
        return self.engineer._extract_tactical_profile(hist, n_matches=10)

    def get_team_rankings(self) -> pd.DataFrame:
        elo_dict = self.state.get('elo_ratings', {})
        rows = []
        for team, rating in elo_dict.items():
            tac = self.get_team_tactical_profile(team)
            rows.append({
                '팀명': team,
                'Elo 레이팅': round(rating, 1),
                '압박 강도': tac.get('pressing_intensity', 50),
                '공격력': tac.get('attack_firepower', 50),
                '골 결정력': tac.get('finishing_efficiency', 50),
                '수비 조직력': tac.get('defensive_wall', 50),
                '세트피스': tac.get('set_piece_threat', 50)
            })
        df = pd.DataFrame(rows).sort_values(by='Elo 레이팅', ascending=False).reset_index(drop=True)
        return df

    def predict_match(self, home_team: str, away_team: str, 
                      rest_home: int = 7, rest_away: int = 7, 
                      match_month: int = 4,
                      next_match_home: int = 0,
                      next_match_away: int = 0) -> Dict[str, Any]:
        elo_ratings = self.state.get('elo_ratings', {})
        team_history = self.state.get('team_history', {})
        team_home_history = self.state.get('team_home_history', {})
        team_away_history = self.state.get('team_away_history', {})
        h2h_history = self.state.get('h2h_history', {})
        season_team_pts = self.state.get('season_team_pts', {})
        season_team_matches = self.state.get('season_team_matches', {})

        # 1. Elo
        elo_h = elo_ratings.get(home_team, 1500.0)
        elo_a = elo_ratings.get(away_team, 1500.0)
        exp_h, exp_a = self.engineer._calc_elo_expected(elo_h, elo_a)
        elo_diff = (elo_h + self.engineer.home_adv) - elo_a

        # 2. 다음 경기 함정 경기 지수
        fake_date = pd.Timestamp(year=2025, month=match_month, day=15)
        trap_h, trap_h_desc = self.engineer._calc_trap_and_lookahead(
            home_team, elo_h, elo_a, rest_home, fake_date, next_match_importance=next_match_home
        )
        trap_a, trap_a_desc = self.engineer._calc_trap_and_lookahead(
            away_team, elo_a, elo_h, rest_away, fake_date, next_match_importance=next_match_away
        )
        trap_diff = trap_h - trap_a

        # 3. 동기부여
        h_pts = season_team_pts.get(home_team, 40)
        a_pts = season_team_pts.get(away_team, 40)
        h_games = season_team_matches.get(home_team, 28)
        a_games = season_team_matches.get(away_team, 28)

        h_recent_res = [m['result'] for m in team_history.get(home_team, [])[-5:]]
        a_recent_res = [m['result'] for m in team_history.get(away_team, [])[-5:]]

        motive_h, motive_h_desc = self.engineer._calculate_motivation_score(h_pts, h_games, h_recent_res, is_home=True, trap_risk=trap_h)
        motive_a, motive_a_desc = self.engineer._calculate_motivation_score(a_pts, a_games, a_recent_res, is_home=False, trap_risk=trap_a)
        motive_diff = motive_h - motive_a

        # 4. 더비
        match_pair = frozenset([home_team, away_team])
        is_derby = 1.0 if match_pair in RIVALRIES else 0.0
        derby_name = RIVALRIES.get(match_pair, 'None')
        if is_derby:
            motive_h = max(motive_h, 0.90)
            motive_a = max(motive_a, 0.90)
            motive_diff = motive_h - motive_a

        # 5. 전술 장단점 & 상성 카운터
        h_hist = team_history.get(home_team, [])
        a_hist = team_history.get(away_team, [])
        h_tactics = self.engineer._extract_tactical_profile(h_hist, n_matches=10)
        a_tactics = self.engineer._extract_tactical_profile(a_hist, n_matches=10)
        h_counter, a_counter, clash_reasons = self.engineer._calc_counter_advantage(h_tactics, a_tactics)
        counter_diff = h_counter - a_counter

        press_clash = h_tactics['pressing_intensity'] - a_tactics['defensive_wall']
        attack_vs_def = h_tactics['attack_firepower'] - a_tactics['defensive_wall']
        def_vs_attack = h_tactics['defensive_wall'] - a_tactics['attack_firepower']
        set_piece_mismatch = h_tactics['set_piece_threat'] - a_tactics['set_piece_threat']

        # 6. 롤링 헬퍼
        def compute_rolling(hist: list, n_matches: int, prefix: str) -> dict:
            res = {}
            recent = hist[-n_matches:] if len(hist) >= n_matches else hist
            count = len(recent)
            if count == 0:
                res[f'{prefix}_goals_scored_avg'] = 1.2
                res[f'{prefix}_goals_conceded_avg'] = 1.2
                res[f'{prefix}_points_avg'] = 1.2
                res[f'{prefix}_shots_avg'] = 11.0
                res[f'{prefix}_shots_target_avg'] = 4.0
                res[f'{prefix}_shots_conceded_avg'] = 11.0
                res[f'{prefix}_shots_target_conceded_avg'] = 4.0
                res[f'{prefix}_corners_avg'] = 4.5
                res[f'{prefix}_fouls_avg'] = 10.0
                res[f'{prefix}_win_rate'] = 0.35
                res[f'{prefix}_draw_rate'] = 0.30
                res[f'{prefix}_loss_rate'] = 0.35
                res[f'{prefix}_shot_accuracy'] = 0.35
                res[f'{prefix}_weighted_points'] = 1.2
                res[f'{prefix}_unbeaten_streak'] = 0
            else:
                gs = sum(m['goals_scored'] for m in recent) / count
                gc = sum(m['goals_conceded'] for m in recent) / count
                pts = sum(m['points'] for m in recent) / count
                shots = sum(m['shots'] for m in recent) / count
                sot = sum(m['shots_target'] for m in recent) / count
                shots_c = sum(m['shots_conceded'] for m in recent) / count
                sot_c = sum(m['shots_target_conceded'] for m in recent) / count
                corners = sum(m['corners'] for m in recent) / count
                fouls = sum(m['fouls'] for m in recent) / count
                
                wins = sum(1 for m in recent if m['result'] == 'W') / count
                draws = sum(1 for m in recent if m['result'] == 'D') / count
                losses = sum(1 for m in recent if m['result'] == 'L') / count
                
                weights = [i + 1 for i in range(count)]
                w_sum = sum(weights)
                w_pts = sum(m['points'] * w for m, w in zip(recent, weights)) / w_sum
                
                unbeaten = 0
                for m in reversed(recent):
                    if m['result'] in ['W', 'D']:
                        unbeaten += 1
                    else:
                        break
                
                res[f'{prefix}_goals_scored_avg'] = gs
                res[f'{prefix}_goals_conceded_avg'] = gc
                res[f'{prefix}_points_avg'] = pts
                res[f'{prefix}_shots_avg'] = shots
                res[f'{prefix}_shots_target_avg'] = sot
                res[f'{prefix}_shots_conceded_avg'] = shots_c
                res[f'{prefix}_shots_target_conceded_avg'] = sot_c
                res[f'{prefix}_corners_avg'] = corners
                res[f'{prefix}_fouls_avg'] = fouls
                res[f'{prefix}_win_rate'] = wins
                res[f'{prefix}_draw_rate'] = draws
                res[f'{prefix}_loss_rate'] = losses
                res[f'{prefix}_shot_accuracy'] = (sot / (shots + 1e-5))
                res[f'{prefix}_weighted_points'] = w_pts
                res[f'{prefix}_unbeaten_streak'] = unbeaten
            return res

        h_home_hist = team_home_history.get(home_team, [])
        a_away_hist = team_away_history.get(away_team, [])

        f_dict = {
            'Elo_Home': elo_h, 'Elo_Away': elo_a, 'Elo_Diff': elo_diff,
            'Elo_Exp_Home': exp_h, 'Elo_Exp_Away': exp_a,
            'Rest_Days_Home': rest_home, 'Rest_Days_Away': rest_away,
            'Rest_Days_Diff': rest_home - rest_away,
            'Trap_Risk_Home': trap_h, 'Trap_Risk_Away': trap_a,
            'Trap_Risk_Diff': trap_diff,
            'Motivation_Home': motive_h, 'Motivation_Away': motive_a,
            'Motivation_Diff': motive_diff,
            'Is_Derby': is_derby,
            'Tactics_Pressing_Home': h_tactics['pressing_intensity'],
            'Tactics_Pressing_Away': a_tactics['pressing_intensity'],
            'Tactics_Attack_Home': h_tactics['attack_firepower'],
            'Tactics_Attack_Away': a_tactics['attack_firepower'],
            'Tactics_Finishing_Home': h_tactics['finishing_efficiency'],
            'Tactics_Finishing_Away': a_tactics['finishing_efficiency'],
            'Tactics_Defense_Home': h_tactics['defensive_wall'],
            'Tactics_Defense_Away': a_tactics['defensive_wall'],
            'Tactics_SetPiece_Home': h_tactics['set_piece_threat'],
            'Tactics_SetPiece_Away': a_tactics['set_piece_threat'],
            'Counter_Advantage_Home': h_counter,
            'Counter_Advantage_Away': a_counter,
            'Counter_Advantage_Diff': counter_diff,
            'Tactics_Clash_Pressing': press_clash,
            'Tactics_Clash_Attack_vs_Def': attack_vs_def,
            'Tactics_Clash_Def_vs_Attack': def_vs_attack,
            'Tactics_Clash_SetPiece': set_piece_mismatch
        }

        for n in [3, 5, 10]:
            f_dict.update(compute_rolling(h_hist, n, f'Home_L{n}'))
            f_dict.update(compute_rolling(a_hist, n, f'Away_L{n}'))
        
        f_dict.update(compute_rolling(h_home_hist, 5, 'Home_AtHome_L5'))
        f_dict.update(compute_rolling(a_away_hist, 5, 'Away_AtAway_L5'))

        # H2H
        h2h_key = tuple(sorted([home_team, away_team]))
        h2h_matches = h2h_history.get(h2h_key, [])[-5:]
        if h2h_matches:
            h2h_pts = []
            h2h_gd = []
            for m in h2h_matches:
                if m['home'] == home_team:
                    h2h_pts.append(m['home_pts'])
                    h2h_gd.append(m['home_goals'] - m['away_goals'])
                else:
                    h2h_pts.append(m['away_pts'])
                    h2h_gd.append(m['away_goals'] - m['home_goals'])
            f_dict['H2H_Home_Avg_Points'] = sum(h2h_pts) / len(h2h_pts)
            f_dict['H2H_Home_Avg_GoalDiff'] = sum(h2h_gd) / len(h2h_gd)
            f_dict['H2H_Count'] = len(h2h_matches)
        else:
            f_dict['H2H_Home_Avg_Points'] = 1.2
            f_dict['H2H_Home_Avg_GoalDiff'] = 0.0
            f_dict['H2H_Count'] = 0

        f_dict['Diff_L5_Points'] = f_dict['Home_L5_points_avg'] - f_dict['Away_L5_points_avg']
        f_dict['Diff_L5_Weighted_Points'] = f_dict['Home_L5_weighted_points'] - f_dict['Away_L5_weighted_points']
        f_dict['Diff_L5_Goals_Scored'] = f_dict['Home_L5_goals_scored_avg'] - f_dict['Away_L5_goals_scored_avg']
        f_dict['Diff_L5_Goals_Conceded'] = f_dict['Home_L5_goals_conceded_avg'] - f_dict['Away_L5_goals_conceded_avg']
        f_dict['Diff_L5_Shots_Target'] = f_dict['Home_L5_shots_target_avg'] - f_dict['Away_L5_shots_target_avg']

        input_df = pd.DataFrame([f_dict])[self.feature_cols].fillna(0)

        # 모델별 예측 확률
        lgbm_prob = self.models['LightGBM'].predict_proba(input_df)[0]
        xgb_prob = self.models['XGBoost'].predict_proba(input_df)[0]
        lr_prob = self.models['LogisticRegression'].predict_proba(input_df)[0]
        
        if 'NeuralNetwork' in self.models:
            mlp_prob = self.models['NeuralNetwork'].predict_proba(input_df)[0]
            ens_prob = (lgbm_prob * 0.40 + xgb_prob * 0.30 + mlp_prob * 0.20 + lr_prob * 0.10)
        else:
            mlp_prob = lr_prob
            ens_prob = (lgbm_prob * 0.50 + xgb_prob * 0.30 + lr_prob * 0.20)

        pred_idx = int(np.argmax(ens_prob))
        outcomes = ['홈팀 승리 (Home Win)', '무승부 (Draw)', '원정팀 승리 (Away Win)']

        # 종합 분석 인사이트
        insights = []
        if is_derby:
            insights.append(f"⚔️ **[{derby_name}]** 치열한 라이벌 더비 경기로 양 팀 모두 동기부여가 100%입니다.")
        
        if trap_h > 0.15:
            insights.append(f"🚨 **{home_team}**: {trap_h_desc} -> 리그 경기 집중력 분산 및 이변(무/패) 위험 발생!")
        if trap_a > 0.15:
            insights.append(f"🚨 **{away_team}**: {trap_a_desc} -> 다음 일정 대비 주전 체력 안배 가능성.")
            
        for cr in clash_reasons:
            insights.append(cr)
            
        insights.append(f"🎯 **동기부여 상태**: {home_team}({motive_h_desc}, 지수: {motive_h*100:.0f}점) vs {away_team}({motive_a_desc}, 지수: {motive_a*100:.0f}점)")
        
        h_streak = f_dict.get('Home_L5_unbeaten_streak', 0)
        a_streak = f_dict.get('Away_L5_unbeaten_streak', 0)
        if h_streak >= 3:
            insights.append(f"🔥 **{home_team}** 최근 {h_streak}경기 연속 무패 행진 중!")
        if a_streak >= 3:
            insights.append(f"🔥 **{away_team}** 최근 {a_streak}경기 연속 무패 행진 중!")

        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_outcome': outcomes[pred_idx],
            'probabilities': {
                'home_win': float(ens_prob[0]),
                'draw': float(ens_prob[1]),
                'away_win': float(ens_prob[2])
            },
            'model_breakdown': {
                'LightGBM': {'home': float(lgbm_prob[0]), 'draw': float(lgbm_prob[1]), 'away': float(lgbm_prob[2])},
                'XGBoost': {'home': float(xgb_prob[0]), 'draw': float(xgb_prob[1]), 'away': float(xgb_prob[2])},
                'NeuralNetwork (Deep Learning)': {'home': float(mlp_prob[0]), 'draw': float(mlp_prob[1]), 'away': float(mlp_prob[2])},
                'LogisticRegression': {'home': float(lr_prob[0]), 'draw': float(lr_prob[1]), 'away': float(lr_prob[2])}
            },
            'elo': {'home': round(elo_h, 1), 'away': round(elo_a, 1)},
            'tactics': {
                'home': h_tactics,
                'away': a_tactics,
                'counter_score_home': h_counter,
                'counter_score_away': a_counter
            },
            'trap_risk': {'home': round(trap_h, 2), 'away': round(trap_a, 2), 'home_desc': trap_h_desc, 'away_desc': trap_a_desc},
            'motivation': {'home': round(motive_h, 2), 'away': round(motive_a, 2), 'home_desc': motive_h_desc, 'away_desc': motive_a_desc},
            'h2h': {'count': f_dict.get('H2H_Count', 0), 'home_avg_pts': round(f_dict.get('H2H_Home_Avg_Points', 0), 2)},
            'recent_form': {
                'home_l5_pts': round(f_dict.get('Home_L5_points_avg', 0), 2),
                'away_l5_pts': round(f_dict.get('Away_L5_points_avg', 0), 2),
                'home_l5_weighted': round(f_dict.get('Home_L5_weighted_points', 0), 2),
                'away_l5_weighted': round(f_dict.get('Away_L5_weighted_points', 0), 2),
            },
            'insights': insights
        }

if __name__ == "__main__":
    predictor = MatchPredictor()
    result = predictor.predict_match('Arsenal', 'Chelsea', rest_home=3, rest_away=7, match_month=4)
    print("예측 결과:")
    print(result)
