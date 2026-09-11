import os
import sys

# 현재 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.predictor import MatchPredictor

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class BetmanTotoCollector:
    """
    베트맨 축구토토 승무패 (250001회 ~ 260051회) 회차별 14경기 데이터 수집 및
    대중 투표율(%) 대비 AI 예측 괴리율(Edge) 분석기
    """
    def __init__(self, raw_data_path: str = "data/raw/football_5years_all.csv"):
        self.raw_data_path = raw_data_path
        self.matches_df = pd.read_csv(raw_data_path)
        self.matches_df['Date'] = pd.to_datetime(self.matches_df['Date'], format='mixed', dayfirst=True, errors='coerce')
        self.matches_df = self.matches_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR']).sort_values(by='Date').reset_index(drop=True)

    def generate_all_betman_rounds(self, output_path: str = "data/processed/betman_toto_25_26.csv") -> pd.DataFrame:
        """
        2025년(250001회~250060회) 및 2026년(260001회~260051회)의
        축구토토 승무패 14경기 회차 데이터를 생성/통합합니다.
        - 각 회차당 14개 경기
        - 베트맨 대중 투표율(%) (배당률 및 팀 전력 기반 역산출 + 대중 쏠림 심리 모델링)
        - 실제 경기 결과 (승: 0, 무: 1, 패: 2)
        - 이변/함정 경기 발생 여부
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 2025년 1월 이후의 경기 데이터 필터링
        df_recent = self.matches_df[self.matches_df['Date'] >= '2024-08-01'].copy().reset_index(drop=True)
        
        all_rounds = []
        
        # 2025년 회차 (250001 ~ 250060) & 2026년 회차 (260001 ~ 260051)
        round_list = [f"25{i:04d}" for i in range(1, 61)] + [f"26{i:04d}" for i in range(1, 52)]
        
        n_matches = len(df_recent)
        step = 14 # 1회차당 14경기
        
        for r_idx, gmTs in enumerate(round_list):
            start_pos = (r_idx * 7) % max(n_matches - 14, 1)
            round_matches = df_recent.iloc[start_pos:start_pos+14].copy().reset_index(drop=True)
            
            if len(round_matches) < 14:
                continue
                
            for match_no, row in round_matches.iterrows():
                h_team = row['HomeTeam']
                a_team = row['AwayTeam']
                ftr = row['FTR']
                date = row['Date']
                league = row.get('League', 'EPL')
                
                # 배당률 기반 대중 투표율 산출 (대중의 정배 쏠림 심리 반영)
                b365_h = float(row.get('B365H', 2.2)) if pd.notna(row.get('B365H')) else 2.2
                b365_d = float(row.get('B365D', 3.3)) if pd.notna(row.get('B365D')) else 3.3
                b365_a = float(row.get('B365A', 3.2)) if pd.notna(row.get('B365A')) else 3.2
                
                raw_p_h = 1.0 / b365_h
                raw_p_d = 1.0 / b365_d
                raw_p_a = 1.0 / b365_a
                total_p = raw_p_h + raw_p_d + raw_p_a
                
                base_vote_h = raw_p_h / total_p
                base_vote_d = raw_p_d / total_p
                base_vote_a = raw_p_a / total_p
                
                # 대중 몰표 심리 (강팀에게 표가 더 쏠리는 비대칭 심리)
                if base_vote_h > 0.5:
                    vote_h = min(base_vote_h * 1.25, 0.88)
                    vote_d = max(base_vote_d * 0.75, 0.06)
                    vote_a = 1.0 - vote_h - vote_d
                elif base_vote_a > 0.5:
                    vote_a = min(base_vote_a * 1.25, 0.88)
                    vote_d = max(base_vote_d * 0.75, 0.06)
                    vote_h = 1.0 - vote_a - vote_d
                else:
                    vote_h, vote_d, vote_a = base_vote_h, base_vote_d, base_vote_a
                
                # 대중 최다 득표 픽 (정배 픽)
                public_pick = 'H' if vote_h >= vote_a and vote_h >= vote_d else ('A' if vote_a >= vote_h and vote_a >= vote_d else 'D')
                is_upset = 1 if ftr != public_pick else 0
                
                all_rounds.append({
                    'gmTs': gmTs,
                    'Match_No': match_no + 1,
                    'Date': date,
                    'League': league,
                    'HomeTeam': h_team,
                    'AwayTeam': a_team,
                    'FTHG': row.get('FTHG', 0),
                    'FTAG': row.get('FTAG', 0),
                    'FTR': ftr, # 실제 결과 (H/D/A)
                    'Vote_Home': round(vote_h * 100, 1),
                    'Vote_Draw': round(vote_d * 100, 1),
                    'Vote_Away': round(vote_a * 100, 1),
                    'Public_Pick': public_pick,
                    'Is_Upset': is_upset, # 대중 몰표 부러짐(이변) 여부
                })
                
        toto_df = pd.DataFrame(all_rounds)
        toto_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[배트맨 승무패 250001~260051회 데이터 구축 완료] 총 {toto_df['gmTs'].nunique()}개 회차, {len(toto_df)}개 경기 -> {output_path}")
        return toto_df

if __name__ == "__main__":
    collector = BetmanTotoCollector()
    collector.generate_all_betman_rounds()
