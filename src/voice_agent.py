import re
from typing import Dict, Any, Tuple
from src.predictor import MatchPredictor

# 한국어 팀명 및 별칭 매핑 사전
TEAM_ALIASES = {
    # EPL
    '아스날': 'Arsenal', '아스널': 'Arsenal',
    '첼시': 'Chelsea',
    '맨시티': 'Man City', '맨체스터 시티': 'Man City', '맨체스터시티': 'Man City',
    '맨유': 'Man United', '맨체스터 유나이티드': 'Man United', '맨체스터유나이티드': 'Man United',
    '리버풀': 'Liverpool',
    '토트넘': 'Tottenham', '토튼햄': 'Tottenham',
    '아스톤빌라': 'Aston Villa', '아스톤 빌라': 'Aston Villa', '빌라': 'Aston Villa',
    '뉴캐슬': 'Newcastle', '뉴캐슬 유나이티드': 'Newcastle',
    '브라이튼': 'Brighton', '브라이턴': 'Brighton',
    '웨스트햄': 'West Ham', '웨스트 햄': 'West Ham',
    '에버튼': 'Everton', '에버턴': 'Everton',
    '울버햄튼': 'Wolves', '울브스': 'Wolves',
    '풀럼': 'Fulham', '크리스탈 팰리스': 'Crystal Palace', '크리스탈팰리스': 'Crystal Palace',
    '본머스': 'Bournemouth', '브렌트포드': 'Brentford', '브렌트퍼드': 'Brentford',
    '노팅엄': 'Nott\'m Forest', '노팅엄 포레스트': 'Nott\'m Forest',
    '사우샘프턴': 'Southampton', '사우스햄튼': 'Southampton',
    '레스터': 'Leicester', '레스터 시티': 'Leicester',
    '리즈': 'Leeds', '리즈 유나이티드': 'Leeds',
    '입스위치': 'Ipswich', '루턴': 'Luton', '번리': 'Burnley', '셰필드': 'Sheffield United',
    
    # La Liga
    '레알': 'Real Madrid', '레알 마드리드': 'Real Madrid', '레알마드리드': 'Real Madrid',
    '바르샤': 'Barcelona', '바르셀로나': 'Barcelona',
    '아틀레티코': 'Ath Madrid', '아틀레티코 마드리드': 'Ath Madrid', '꼬마': 'Ath Madrid', 'at마드리드': 'Ath Madrid',
    '소시에다드': 'Real Sociedad', '레알 소시에다드': 'Real Sociedad',
    '빌바오': 'Ath Bilbao', '아틀레틱 빌바오': 'Ath Bilbao',
    '베티스': 'Betis', '레알 베티스': 'Betis',
    '비야레알': 'Villarreal',
    '세비야': 'Sevilla',
    '발렌시아': 'Valencia',
    '지로나': 'Girona',
    '오사수나': 'Osasuna', '헤타페': 'Getafe', '에스파뇰': 'Espanol', '셀타': 'Celta', '셀타비고': 'Celta',
    '마요르카': 'Mallorca', '라요': 'Vallecano', '바예카노': 'Vallecano', '알라베스': 'Alaves',
    '라스팔마스': 'Las Palmas', '레가네스': 'Leganes', '알메리아': 'Almeria', '카디스': 'Cadiz', '그라나다': 'Granada'
}

class FootballVoiceAgent:
    def __init__(self, predictor: MatchPredictor):
        self.predictor = predictor

    def parse_voice_query(self, text: str) -> Dict[str, Any]:
        """
        사용자의 음성/자연어 질의에서 홈팀, 원정팀, 챔스/다음경기 여부 등을 파싱합니다.
        예: "아스날이랑 첼시 경기 분석해줘", "레알 마드리드 다음 경기 챔스인데 바르샤전 어때?"
        """
        text_lower = text.lower()
        found_teams = []

        # 팀명 추출
        # 긴 이름부터 매칭
        sorted_aliases = sorted(TEAM_ALIASES.keys(), key=lambda x: len(x), reverse=True)
        for alias in sorted_aliases:
            if alias.lower() in text_lower:
                eng_team = TEAM_ALIASES[alias]
                if eng_team not in found_teams:
                    found_teams.append(eng_team)
                if len(found_teams) == 2:
                    break

        # 다음 경기(챔스/로테이션) 키워드 감지
        next_ucl_keywords = ['챔스', '챔피언스리그', '유럽대항전', '로테이션', '힘빼', '다음 경기', '다음경기', '토너먼트']
        has_ucl_concern = any(kw in text_lower for kw in next_ucl_keywords)
        next_importance = 2 if has_ucl_concern else 0

        # 강등/꼴찌 키워드
        is_relegation_talk = any(kw in text_lower for kw in ['강등', '꼴찌', '결사항전', '탈출', '절박'])

        return {
            'found_teams': found_teams,
            'next_match_importance': next_importance,
            'is_relegation_talk': is_relegation_talk,
            'raw_text': text
        }

    def generate_voice_response(self, user_voice_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        사용자의 음성 질의를 분석하여 축구 AI 예측을 수행하고,
        음성으로 읽어주기 가장 적합한 자연스러운 음성 해설 대본(Speech Script)을 생성합니다.
        """
        parsed = self.parse_voice_query(user_voice_text)
        teams = parsed['found_teams']

        if len(teams) < 2:
            if len(teams) == 1:
                return (f"네, {teams[0]}을 찾았습니다! 상대 팀도 함께 말씀해 주시면 승패를 분석해 드릴게요. 예를 들어 '{teams[0]} 대 첼시 경기 분석해줘'라고 말씀해 보세요.", {})
            else:
                return ("팀 이름을 정확히 인식하지 못했습니다. '아스날 대 토트넘 경기 예측해줘' 또는 '레알 마드리드 다음 경기 챔스인데 바르샤전 어때?'처럼 말씀해 주세요!", {})

        home_team, away_team = teams[0], teams[1]
        next_imp = parsed['next_match_importance']

        # 예측 실행
        res = self.predictor.predict_match(
            home_team, away_team,
            rest_home=3 if next_imp > 0 else 7,
            rest_away=7,
            match_month=4,
            next_match_home=next_imp,
            next_match_away=0
        )

        probs = res['probabilities']
        h_prob = int(probs['home_win'] * 100)
        d_prob = int(probs['draw'] * 100)
        a_prob = int(probs['away_win'] * 100)

        # 음성으로 들려줄 대본 구성
        speech_lines = []
        speech_lines.append(f"네! 지난 5개년 데이터와 전술 상성을 분석한 결과입니다.")
        
        # 예측 결과
        if res['predicted_outcome'] == '홈팀 승리 (Home Win)':
            speech_lines.append(f"홈팀 {home_team}의 승리 확률이 {h_prob}퍼센트로 가장 우세합니다.")
        elif res['predicted_outcome'] == '원정팀 승리 (Away Win)':
            speech_lines.append(f"원정팀 {away_team}의 승리 확률이 {a_prob}퍼센트로 역배당 또는 우세가 예상됩니다.")
        else:
            speech_lines.append(f"치열한 접전으로 무승부 확률이 {d_prob}퍼센트로 매우 높게 나타났습니다.")

        speech_lines.append(f"확률 분포는 {home_team} 승리 {h_prob}퍼센트, 무승부 {d_prob}퍼센트, {away_team} 승리 {a_prob}퍼센트입니다.")

        # 핵심 인사이트 추가
        for ins in res['insights'][:2]:
            clean_ins = ins.replace('**', '').replace('🚨', '').replace('⚠️', '').replace('🎯', '').replace('🔥', '').replace('⚔️', '')
            speech_lines.append(clean_ins)

        voice_script = " ".join(speech_lines)
        return voice_script, res
