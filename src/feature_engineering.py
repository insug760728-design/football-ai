import os
import io
import sys
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 유럽 대항전(UCL/UEL) 단골 출전 빅클럽 리스트
EUROPEAN_CONTENDERS = {
    'Man City', 'Arsenal', 'Liverpool', 'Aston Villa', 'Tottenham', 'Chelsea', 'Man United', 'Newcastle',
    'Real Madrid', 'Barcelona', 'Ath Madrid', 'Real Sociedad', 'Ath Bilbao', 'Betis', 'Villarreal', 'Sevilla'
}

# 유명 더비 및 라이벌 매치 정의
RIVALRIES = {
    # EPL 더비
    frozenset(['Arsenal', 'Tottenham']): 'North London Derby',
    frozenset(['Liverpool', 'Everton']): 'Merseyside Derby',
    frozenset(['Man United', 'Man City']): 'Manchester Derby',
    frozenset(['Liverpool', 'Man United']): 'North West Derby',
    frozenset(['Chelsea', 'Arsenal']): 'London Derby',
    frozenset(['Chelsea', 'Tottenham']): 'London Derby',
    frozenset(['Aston Villa', 'Wolves']): 'Midlands Derby',
    frozenset(['Newcastle', 'Sunderland']): 'Tyne-Wear Derby',
    
    # La Liga 더비
    frozenset(['Real Madrid', 'Barcelona']): 'El Clasico',
    frozenset(['Real Madrid', 'Ath Madrid']): 'Madrid Derby',
    frozenset(['Sevilla', 'Betis']): 'Seville Derby (El Gran Derbi)',
    frozenset(['Ath Bilbao', 'Sociedad']): 'Basque Derby',
    frozenset(['Barcelona', 'Espanol']): 'Barcelona Derby',
    frozenset(['Valencia', 'Villarreal']): 'Derbi de la Comunitat',
}

class FootballFeatureEngineer:
    def __init__(self, k_factor: float = 30.0, home_adv: float = 65.0, initial_elo: float = 1500.0):
        self.k_factor = k_factor
        self.home_adv = home_adv
        self.initial_elo = initial_elo

    def _calc_elo_expected(self, elo_home: float, elo_away: float) -> Tuple[float, float]:
        diff = (elo_away - (elo_home + self.home_adv)) / 400.0
        exp_home = 1.0 / (1.0 + math.pow(10.0, diff))
        exp_away = 1.0 - exp_home
        return exp_home, exp_away

    def _get_goal_multiplier(self, goal_diff: int) -> float:
        abs_diff = abs(goal_diff)
        if abs_diff <= 1:
            return 1.0
        elif abs_diff == 2:
            return 1.5
        else:
            return (11.0 + abs_diff) / 8.0

    def _calc_trap_and_lookahead(self, team: str, elo_self: float, elo_opp: float, 
                                rest_days: int, date: pd.Timestamp, 
                                next_match_importance: int = 0) -> Tuple[float, str]:
        """
        [다음 경기 중요도 & 함정 경기(Trap Game / Lookahead) 지수 산출]
        강팀(1위권/빅클럽)이 약팀을 상대할 때, 3~4일 뒤에 챔피언스리그 토너먼트나
        우승 결정전급 빅매치가 잡혀 있는 경우 발생하는 주전 로테이션 및 집중력 분산(방심) 패널티.
        """
        is_big_team = team in EUROPEAN_CONTENDERS or elo_self > 1650
        is_weaker_opp = (elo_self - elo_opp) > 150 # 상당한 전력 차이
        
        month = date.month
        is_ucl_time = month in [2, 3, 4, 5, 9, 10, 11]

        trap_risk = 0.0
        reasons = []

        # 명시적 다음 경기 중요도가 주어진 경우
        if next_match_importance >= 2: # 챔스 8강/4강 또는 결승
            trap_risk += 0.35
            reasons.append("🚨 [Lookahead 경보] 3일 뒤 챔피언스리그/결승 빅매치 대비 극심한 주전 로테이션 & 힘빼기")
        elif next_match_importance == 1: # 라이벌전 또는 챔스 조별
            trap_risk += 0.20
            reasons.append("⚠️ [Lookahead 주의] 다음 경기 라이벌전/유럽대항전 집중으로 인한 방심 위험")
        else:
            # 일정 데이터 기반 자동 추정 (봄 챔스 기간에 빡빡한 일정 + 약팀 상대)
            if is_big_team and is_weaker_opp and rest_days <= 4 and is_ucl_time:
                trap_risk += 0.25
                reasons.append("⚠️ [함정 경기(Trap Game)] 주중 유럽대항전 여파 및 주전 체력 안배로 하위권 상대 고전 가능성")
            elif is_big_team and rest_days <= 4:
                trap_risk += 0.15
                reasons.append("🗓️ 주중 경기 병행에 따른 체력 과부하")

        reason_str = ", ".join(reasons) if reasons else "일정 부담 없음"
        return min(trap_risk, 0.50), reason_str

    def _calculate_motivation_score(self, current_points: float, games_played: int, 
                                   recent_results: List[str], is_home: bool, 
                                   trap_risk: float) -> Tuple[float, str]:
        if games_played < 5:
            base_motive = 0.60
            if trap_risk > 0:
                base_motive -= (trap_risk * 0.5)
            return max(0.1, base_motive), "시즌 초반 순위 선점"

        progress = min(games_played / 38.0, 1.0)
        projected_pts = (current_points / max(games_played, 1)) * 38.0

        motivation = 0.50
        reasons = []

        # 1. 꼴찌/강등권 팀의 결사항전 (18~20위권)
        if projected_pts < 38.0:
            relegation_urgency = 0.30 * (progress ** 1.5)
            motivation += (0.25 + relegation_urgency)
            reasons.append("🚨 강등권 탈출 결사항전 (극도 절박함)")
        elif projected_pts >= 65.0:
            top_urgency = 0.20 * progress
            motivation += (0.20 + top_urgency)
            reasons.append("🏆 우승/챔스 티켓 쟁탈전")
        elif 50.0 <= projected_pts < 65.0:
            motivation += (0.10 * progress)
            reasons.append("🇪🇺 유럽대항전 진출 경쟁")
        elif 40.0 <= projected_pts < 50.0 and progress > 0.70:
            motivation -= (0.20 * progress)
            reasons.append("🏖️ 동기부여 저하 (잔류 안정권 중위권)")

        # 2. 연패 탈출 배수진
        if len(recent_results) >= 3:
            last_3 = recent_results[-3:]
            if all(r == 'L' for r in last_3):
                desperation = 0.15 if is_home else 0.08
                motivation += desperation
                reasons.append("🔥 연패 탈출 배수진")
            elif all(r == 'W' for r in last_3):
                motivation += 0.05
                reasons.append("⚡ 3연승 상승세 모멘텀")

        # 3. 다음 경기 대비 힘빼기(Trap Game) 감점
        if trap_risk > 0:
            motivation -= trap_risk
            reasons.append(f"⚠️ 다음 경기 집중 힘빼기 패널티 (-{int(trap_risk*100)}%)")

        motivation = max(0.1, min(1.0, motivation))
        reason_str = ", ".join(reasons) if reasons else "일반 정규 리그 경기"
        return motivation, reason_str

    def _extract_tactical_profile(self, hist: List[dict], n_matches: int = 10) -> Dict[str, float]:
        recent = hist[-n_matches:] if len(hist) >= n_matches else hist
        if not recent:
            return {
                'pressing_intensity': 50.0,
                'attack_firepower': 50.0,
                'finishing_efficiency': 50.0,
                'defensive_wall': 50.0,
                'set_piece_threat': 50.0
            }

        count = len(recent)
        avg_fouls = sum(m['fouls'] for m in recent) / count
        avg_shots = sum(m['shots'] for m in recent) / count
        avg_sot = sum(m['shots_target'] for m in recent) / count
        avg_goals = sum(m['goals_scored'] for m in recent) / count
        avg_conceded = sum(m['goals_conceded'] for m in recent) / count
        avg_shots_c = sum(m['shots_conceded'] for m in recent) / count
        avg_corners = sum(m['corners'] for m in recent) / count

        foul_score = min(max((avg_fouls - 7.0) / 8.0, 0.0), 1.0) * 50.0
        suppression_score = min(max((18.0 - avg_shots_c) / 12.0, 0.0), 1.0) * 50.0
        pressing = foul_score + suppression_score

        shot_score = min(max((avg_shots - 8.0) / 12.0, 0.0), 1.0) * 50.0
        goal_score = min(max((avg_goals - 0.5) / 2.5, 0.0), 1.0) * 50.0
        attack = shot_score + goal_score

        shot_acc = (avg_sot / (avg_shots + 1e-5))
        goal_conv = (avg_goals / (avg_sot + 1e-5))
        acc_score = min(max((shot_acc - 0.25) / 0.20, 0.0), 1.0) * 50.0
        conv_score = min(max((goal_conv - 0.20) / 0.25, 0.0), 1.0) * 50.0
        finishing = acc_score + conv_score

        concede_score = min(max((2.5 - avg_conceded) / 2.0, 0.0), 1.0) * 60.0
        shot_c_score = min(max((18.0 - avg_shots_c) / 12.0, 0.0), 1.0) * 40.0
        defense = concede_score + shot_c_score

        corner_score = min(max((avg_corners - 2.0) / 6.0, 0.0), 1.0) * 100.0
        set_piece = corner_score

        return {
            'pressing_intensity': round(pressing, 1),
            'attack_firepower': round(attack, 1),
            'finishing_efficiency': round(finishing, 1),
            'defensive_wall': round(defense, 1),
            'set_piece_threat': round(set_piece, 1)
        }

    def _calc_counter_advantage(self, t_home: Dict[str, float], t_away: Dict[str, float]) -> Tuple[float, float, List[str]]:
        reasons = []
        h_counter = 0.0
        if t_home['pressing_intensity'] > 65 and t_away['defensive_wall'] < 45:
            h_counter += 25.0
            reasons.append("🎯 [상성 우위] 홈팀의 강력한 전방 압박이 상대의 불안한 빌드업을 완벽히 공략함")
        if t_home['attack_firepower'] > 65 and t_away['defensive_wall'] < 45:
            h_counter += 30.0
            reasons.append("🎯 [상성 우위] 홈팀의 막강한 화력이 상대의 헐거운 수비벽을 압도함")
        if t_home['finishing_efficiency'] > 65 and t_away['defensive_wall'] < 50:
            h_counter += 20.0
            reasons.append("🎯 [상성 우위] 홈팀의 예리한 골 결정력이 찬스를 놓치지 않고 득점 전환")
        if t_home['set_piece_threat'] > 65 and t_away['defensive_wall'] < 50:
            h_counter += 20.0
            reasons.append("🎯 [상성 우위] 홈팀의 세트피스/코너킥 제공권 장악력이 상대 수비에 치명적")

        a_counter = 0.0
        if t_away['pressing_intensity'] > 65 and t_home['defensive_wall'] < 45:
            a_counter += 25.0
            reasons.append("⚠️ [원정 역상성] 원정팀의 거센 압박이 홈팀 수비 실수를 유발할 위험 높음")
        if t_away['attack_firepower'] > 65 and t_home['defensive_wall'] < 45:
            a_counter += 30.0
            reasons.append("⚠️ [원정 역상성] 원정팀의 카운터 어택 화력이 홈팀의 수비 뒷공간을 위협함")
        if t_away['finishing_efficiency'] > 65 and t_home['defensive_wall'] < 50:
            a_counter += 20.0
            reasons.append("⚠️ [원정 역상성] 원정팀의 높은 골 결정력으로 한두 번의 찬스가 실점으로 이어질 위험")
        if t_away['set_piece_threat'] > 65 and t_home['defensive_wall'] < 50:
            a_counter += 20.0
            reasons.append("⚠️ [원정 역상성] 원정팀의 세트피스 헤더 공중볼 장악 주의")

        return h_counter, a_counter, reasons

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
        df = df.sort_values(by=['Date']).reset_index(drop=True)

        elo_ratings: Dict[str, float] = {}
        current_season = None
        season_team_pts: Dict[str, int] = {}
        season_team_matches: Dict[str, int] = {}

        team_history: Dict[str, List[dict]] = {}
        team_home_history: Dict[str, List[dict]] = {}
        team_away_history: Dict[str, List[dict]] = {}
        h2h_history: Dict[Tuple[str, str], List[dict]] = {}
        team_last_date: Dict[str, pd.Timestamp] = {}

        feature_rows = []

        for idx, row in df.iterrows():
            date = row['Date']
            home_team = str(row['HomeTeam'])
            away_team = str(row['AwayTeam'])
            league = row.get('League', 'Unknown')
            season = str(row.get('Season', 'Unknown'))

            if season != current_season:
                current_season = season
                season_team_pts = {}
                season_team_matches = {}

            # 1. Elo
            elo_h = elo_ratings.get(home_team, self.initial_elo)
            elo_a = elo_ratings.get(away_team, self.initial_elo)
            exp_h, exp_a = self._calc_elo_expected(elo_h, elo_a)
            elo_diff = (elo_h + self.home_adv) - elo_a

            # 2. 휴식일수
            rest_h = (date - team_last_date[home_team]).days if home_team in team_last_date else 10
            rest_a = (date - team_last_date[away_team]).days if away_team in team_last_date else 10
            rest_h = min(max(rest_h, 1), 30)
            rest_a = min(max(rest_a, 1), 30)

            # 3. [다음 경기 중요도 & 함정 경기(Trap Game) 리스크]
            trap_h, trap_h_desc = self._calc_trap_and_lookahead(home_team, elo_h, elo_a, rest_h, date)
            trap_a, trap_a_desc = self._calc_trap_and_lookahead(away_team, elo_a, elo_h, rest_a, date)
            trap_diff = trap_h - trap_a

            # 4. 동기부여 지수
            h_season_pts = season_team_pts.get(home_team, 0)
            a_season_pts = season_team_pts.get(away_team, 0)
            h_games = season_team_matches.get(home_team, 0)
            a_games = season_team_matches.get(away_team, 0)
            
            h_recent_res = [m['result'] for m in team_history.get(home_team, [])[-5:]]
            a_recent_res = [m['result'] for m in team_history.get(away_team, [])[-5:]]

            motive_h, motive_h_desc = self._calculate_motivation_score(h_season_pts, h_games, h_recent_res, is_home=True, trap_risk=trap_h)
            motive_a, motive_a_desc = self._calculate_motivation_score(a_season_pts, a_games, a_recent_res, is_home=False, trap_risk=trap_a)
            motive_diff = motive_h - motive_a

            # 5. 라이벌리 / 더비 경기
            match_pair = frozenset([home_team, away_team])
            is_derby = 1.0 if match_pair in RIVALRIES else 0.0
            derby_name = RIVALRIES.get(match_pair, 'None')
            if is_derby:
                motive_h = max(motive_h, 0.90)
                motive_a = max(motive_a, 0.90)
                motive_diff = motive_h - motive_a

            # 6. 전술 장단점 & 상성 카운터
            h_hist = team_history.get(home_team, [])
            a_hist = team_history.get(away_team, [])
            
            h_tactics = self._extract_tactical_profile(h_hist, n_matches=10)
            a_tactics = self._extract_tactical_profile(a_hist, n_matches=10)

            h_counter, a_counter, clash_reasons = self._calc_counter_advantage(h_tactics, a_tactics)
            counter_diff = h_counter - a_counter

            press_clash = h_tactics['pressing_intensity'] - a_tactics['defensive_wall']
            attack_vs_def = h_tactics['attack_firepower'] - a_tactics['defensive_wall']
            def_vs_attack = h_tactics['defensive_wall'] - a_tactics['attack_firepower']
            set_piece_mismatch = h_tactics['set_piece_threat'] - a_tactics['set_piece_threat']

            # 7. 최근 가중 폼 및 롤링 스탯
            def compute_rolling(hist: List[dict], n_matches: int, prefix: str) -> dict:
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
                'Match_ID': idx,
                'Date': date,
                'Season': season,
                'League': league,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Elo_Home': elo_h,
                'Elo_Away': elo_a,
                'Elo_Diff': elo_diff,
                'Elo_Exp_Home': exp_h,
                'Elo_Exp_Away': exp_a,
                'Rest_Days_Home': rest_h,
                'Rest_Days_Away': rest_a,
                'Rest_Days_Diff': rest_h - rest_a,
                
                # 함정 경기(Trap Game) 및 동기부여 피처
                'Trap_Risk_Home': trap_h,
                'Trap_Risk_Away': trap_a,
                'Trap_Risk_Diff': trap_diff,
                'Motivation_Home': motive_h,
                'Motivation_Away': motive_a,
                'Motivation_Diff': motive_diff,
                'Is_Derby': is_derby,
                'Derby_Name': derby_name,
                
                # 전술 장단점 & 상성 카운터
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
                'Tactics_Clash_SetPiece': set_piece_mismatch,
                
                'Games_Played_Home': h_games,
                'Games_Played_Away': a_games,
                'Season_Points_Home': h_season_pts,
                'Season_Points_Away': a_season_pts,
            }

            for n in [3, 5, 10]:
                f_dict.update(compute_rolling(h_hist, n, f'Home_L{n}'))
                f_dict.update(compute_rolling(a_hist, n, f'Away_L{n}'))
            
            f_dict.update(compute_rolling(h_home_hist, 5, 'Home_AtHome_L5'))
            f_dict.update(compute_rolling(a_away_hist, 5, 'Away_AtAway_L5'))

            # 8. H2H
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

            # 9. 상대적 차이
            f_dict['Diff_L5_Points'] = f_dict['Home_L5_points_avg'] - f_dict['Away_L5_points_avg']
            f_dict['Diff_L5_Weighted_Points'] = f_dict['Home_L5_weighted_points'] - f_dict['Away_L5_weighted_points']
            f_dict['Diff_L5_Goals_Scored'] = f_dict['Home_L5_goals_scored_avg'] - f_dict['Away_L5_goals_scored_avg']
            f_dict['Diff_L5_Goals_Conceded'] = f_dict['Home_L5_goals_conceded_avg'] - f_dict['Away_L5_goals_conceded_avg']
            f_dict['Diff_L5_Shots_Target'] = f_dict['Home_L5_shots_target_avg'] - f_dict['Away_L5_shots_target_avg']

            # 10. 배당률 정보
            if 'B365H' in row and pd.notna(row['B365H']):
                f_dict['B365H'] = row['B365H']
                f_dict['B365D'] = row['B365D']
                f_dict['B365A'] = row['B365A']

            # 11. 타겟 레이블
            ftr = row['FTR']
            f_dict['FTHG'] = row['FTHG']
            f_dict['FTAG'] = row['FTAG']
            f_dict['FTR'] = ftr
            target_map = {'H': 0, 'D': 1, 'A': 2}
            f_dict['Target'] = target_map.get(ftr, -1)

            feature_rows.append(f_dict)

            # === 경기 종료 후 상태 업데이트 ===
            fthg = int(row['FTHG'])
            ftag = int(row['FTAG'])
            hs = float(row.get('HS', 11))
            as_ = float(row.get('AS', 11))
            hst = float(row.get('HST', 4))
            ast = float(row.get('AST', 4))
            hc = float(row.get('HC', 5))
            ac = float(row.get('AC', 5))
            hf = float(row.get('HF', 10))
            af = float(row.get('AF', 10))

            if ftr == 'H':
                home_pts, away_pts = 3, 0
                home_res, away_res = 'W', 'L'
                actual_home, actual_away = 1.0, 0.0
            elif ftr == 'D':
                home_pts, away_pts = 1, 1
                home_res, away_res = 'D', 'D'
                actual_home, actual_away = 0.5, 0.5
            else:
                home_pts, away_pts = 0, 3
                home_res, away_res = 'L', 'W'
                actual_home, actual_away = 0.0, 1.0

            season_team_pts[home_team] = season_team_pts.get(home_team, 0) + home_pts
            season_team_pts[away_team] = season_team_pts.get(away_team, 0) + away_pts
            season_team_matches[home_team] = season_team_matches.get(home_team, 0) + 1
            season_team_matches[away_team] = season_team_matches.get(away_team, 0) + 1

            h_match_data = {
                'goals_scored': fthg, 'goals_conceded': ftag, 'points': home_pts,
                'shots': hs, 'shots_target': hst, 'shots_conceded': as_, 'shots_target_conceded': ast,
                'corners': hc, 'fouls': hf, 'result': home_res
            }
            a_match_data = {
                'goals_scored': ftag, 'goals_conceded': fthg, 'points': away_pts,
                'shots': as_, 'shots_target': ast, 'shots_conceded': hs, 'shots_target_conceded': hst,
                'corners': ac, 'fouls': af, 'result': away_res
            }

            team_history.setdefault(home_team, []).append(h_match_data)
            team_history.setdefault(away_team, []).append(a_match_data)
            team_home_history.setdefault(home_team, []).append(h_match_data)
            team_away_history.setdefault(away_team, []).append(a_match_data)

            h2h_history.setdefault(h2h_key, []).append({
                'home': home_team, 'away': away_team,
                'home_goals': fthg, 'away_goals': ftag,
                'home_pts': home_pts, 'away_pts': away_pts
            })

            team_last_date[home_team] = date
            team_last_date[away_team] = date

            margin_mult = self._get_goal_multiplier(fthg - ftag)
            elo_ratings[home_team] = elo_h + self.k_factor * margin_mult * (actual_home - exp_h)
            elo_ratings[away_team] = elo_a + self.k_factor * margin_mult * (actual_away - exp_a)

        feat_df = pd.DataFrame(feature_rows)
        self.latest_state = {
            'elo_ratings': elo_ratings,
            'team_history': team_history,
            'team_home_history': team_home_history,
            'team_away_history': team_away_history,
            'h2h_history': h2h_history,
            'team_last_date': team_last_date,
            'season_team_pts': season_team_pts,
            'season_team_matches': season_team_matches
        }
        return feat_df

def process_features(input_path: str = "data/raw/football_5years_all.csv", 
                     output_path: str = "data/processed/features_dataset.csv") -> Tuple[pd.DataFrame, FootballFeatureEngineer]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    
    engineer = FootballFeatureEngineer()
    features_df = engineer.generate_features(df)
    
    features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[특성 공학 완료] 총 {len(features_df)}개 경기, {features_df.shape[1]}개 피처 생성 -> {output_path}")
    return features_df, engineer

if __name__ == "__main__":
    process_features()
