import os
import io
import sys
import requests
import pandas as pd
from typing import List, Dict

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# football-data.co.uk URLs
# Seasons: 2020/21 (2021), 2021/22 (2122), 2022/23 (2223), 2023/24 (2324), 2024/25 (2425)
SEASONS = ["2021", "2122", "2223", "2324", "2425"]
LEAGUES = {
    "EPL": "E0",
    "LaLiga": "SP1"
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# 핵심 컬럼 정의
CORE_COLUMNS = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam', 
    'FTHG', 'FTAG', 'FTR',       # 풀타임 홈/원정 골, 결과 (H/D/A)
    'HTHG', 'HTAG', 'HTR',       # 전반전 홈/원정 골, 결과
    'HS', 'AS',                  # 홈/원정 슈팅
    'HST', 'AST',                # 홈/원정 유효슈팅
    'HF', 'AF',                  # 홈/원정 파울
    'HC', 'AC',                  # 홈/원정 코너킥
    'HY', 'AY',                  # 홈/원정 옐로카드
    'HR', 'AR',                  # 홈/원정 레드카드
    'B365H', 'B365D', 'B365A'    # Bet365 홈/무/원정 배당률 (존재 시)
]

def download_season_data(league_name: str, league_code: str, season: str) -> pd.DataFrame:
    """단일 시즌 데이터를 다운로드하여 DataFrame으로 반환합니다."""
    url = BASE_URL.format(season=season, code=league_code)
    print(f"[{league_name}] {season} 시즌 다운로드 중... ({url})")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 인코딩 처리 (일부 파일은 latin1 / utf-8)
        content = response.content
        try:
            df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding='latin1')
            
        # 시즌 컬럼 추가
        season_str = f"20{season[:2]}-20{season[2:]}"
        
        # 존재하는 컬럼만 필터링
        avail_cols = [c for c in CORE_COLUMNS if c in df.columns]
        df = df[avail_cols].copy()
        df['Season'] = season_str
        df['League'] = league_name
        
        # 결측 경기 제외 (아직 치러지지 않은 경기 등)
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
        print(f" -> {len(df)}개 경기 수집 완료")
        return df
    except Exception as e:
        print(f" [Error] {league_name} {season} 다운로드 실패: {e}")
        return pd.DataFrame()

def collect_all_data(output_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """EPL과 라리가 5년치 데이터를 수집하여 CSV로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []
    league_results = {}
    
    for league_name, league_code in LEAGUES.items():
        league_dfs = []
        for season in SEASONS:
            df = download_season_data(league_name, league_code, season)
            if not df.empty:
                league_dfs.append(df)
        
        if league_dfs:
            combined_league = pd.concat(league_dfs, ignore_index=True)
            combined_league['Date'] = pd.to_datetime(combined_league['Date'], format='mixed', dayfirst=True, errors='coerce')
            combined_league = combined_league.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
            combined_league = combined_league.sort_values(by=['Date']).reset_index(drop=True)
            
            save_path = os.path.join(output_dir, f"{league_name}_5years.csv")
            combined_league.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"[{league_name}] 총 {len(combined_league)}개 경기 저장 완료 -> {save_path}")
            
            league_results[league_name] = combined_league
            all_dfs.append(combined_league)
            
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.sort_values(by=['Date']).reset_index(drop=True)
        total_path = os.path.join(output_dir, "football_5years_all.csv")
        full_df.to_csv(total_path, index=False, encoding='utf-8-sig')
        print(f"[전체 데이터] 총 {len(full_df)}개 경기 통합 저장 완료 -> {total_path}")
        league_results["ALL"] = full_df

    return league_results

if __name__ == "__main__":
    collect_all_data()
