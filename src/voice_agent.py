import re
from typing import Dict, Any, Tuple
from src.predictor import MatchPredictor

# 한국어 팀명, 선수명, 감독명 및 별칭 매핑 사전
TEAM_ALIASES = {
    # 선수 및 감독 -> 팀 매핑
    '손흥민': 'Tottenham', '손': 'Tottenham', '쏘니': 'Tottenham', '포스테코글루': 'Tottenham',
    '황희찬': 'Wolves',
    '홀란드': 'Man City', '엘링 홀란드': 'Man City', '데브라이너': 'Man City', '더브라위너': 'Man City', '덕배': 'Man City', '펩': 'Man City', '과르디올라': 'Man City',
    '살라': 'Liverpool', '모하메드 살라': 'Liverpool', '반다이크': 'Liverpool', '알리송': 'Liverpool', '슬롯': 'Liverpool', '클롭': 'Liverpool',
    '사카': 'Arsenal', '부카요 사카': 'Arsenal', '외데고르': 'Arsenal', '하베르츠': 'Arsenal', '아르테타': 'Arsenal',
    '콜 파머': 'Chelsea', '파머': 'Chelsea', '엔조': 'Chelsea', '마레스카': 'Chelsea',
    '브루노': 'Man United', '브페': 'Man United', '래시포드': 'Man United', '아모림': 'Man United', '텐하흐': 'Man United',
    '음바페': 'Real Madrid', '비니시우스': 'Real Madrid', '벨링엄': 'Real Madrid', '모드리치': 'Real Madrid', '안첼로티': 'Real Madrid',
    '야말': 'Barcelona', '라민 야말': 'Barcelona', '레반도프스키': 'Barcelona', '레반돕': 'Barcelona', '페드리': 'Barcelona', '가비': 'Barcelona', '하피냐': 'Barcelona', '플릭': 'Barcelona',
    '그리즈만': 'Ath Madrid', '시메오네': 'Ath Madrid',
    '쿠보': 'Real Sociedad',

    # EPL 팀
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
    
    # La Liga 팀
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
    def __init__(self, predictor: MatchPredictor, toto_analyzer=None):
        self.predictor = predictor
        self.toto_analyzer = toto_analyzer

    def parse_voice_query(self, text: str) -> Dict[str, Any]:
        """
        자연어 질의에서 팀명, 선수명, 승무패 회차/경기번호, 챔스 여부 등을 유연하게 파싱합니다.
        """
        text_lower = text.lower()
        found_teams = []

        # 팀명 및 선수/감독명 추출 (긴 이름부터 매칭)
        sorted_aliases = sorted(TEAM_ALIASES.keys(), key=lambda x: len(x), reverse=True)
        for alias in sorted_aliases:
            if alias.lower() in text_lower:
                eng_team = TEAM_ALIASES[alias]
                if eng_team not in found_teams:
                    found_teams.append(eng_team)
                if len(found_teams) == 2:
                    break

        # 승무패 회차 및 경기 번호 추출 (52회분석, 52회차, 52회, 260051 등)
        round_match = re.search(r'(\d+)\s*(?:회차|회|번째)?', text)
        round_num = None
        if round_match:
            val = int(round_match.group(1))
            if val > 1000:
                round_num = val
            elif 1 <= val <= 99:
                # 250000 / 260000 포맷 매핑
                round_num = 260000 + val
        
        match_no_match = re.search(r'(\d+)\s*번(?:\s*경기)?', text)
        match_no = int(match_no_match.group(1)) if match_no_match else None

        # 다음 경기(챔스/로테이션) 키워드 감지
        next_ucl_keywords = ['챔스', '챔피언스리그', '유럽대항전', '로테이션', '힘빼', '다음 경기', '다음경기', '토너먼트', '일정', '체력']
        has_ucl_concern = any(kw in text_lower for kw in next_ucl_keywords)
        next_importance = 2 if has_ucl_concern else 0

        # 강등/꼴찌 키워드
        is_relegation_talk = any(kw in text_lower for kw in ['강등', '꼴찌', '결사항전', '탈출', '절박', '생존'])

        return {
            'found_teams': found_teams,
            'round_num': round_num,
            'match_no': match_no,
            'next_match_importance': next_importance,
            'is_relegation_talk': is_relegation_talk,
            'raw_text': text
        }

    def generate_voice_response(self, user_voice_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        사용자의 음성/채팅 질의를 축구 전문가 관점에서 대화형으로 풀어냅니다.
        """
        text = user_voice_text.strip()
        text_lower = text.lower()
        parsed = self.parse_voice_query(text)

        # 1. 인사 및 친근한 대화
        if any(w in text_lower for w in ['안녕', '반가워', '하이', '누구야', '자기소개', '뭐해', '반갑']):
            return ("반가워요! 저는 축구 승패 예측과 배트맨 승무패 14경기를 전문으로 분석하는 AI 여성 파트너예요. 궁금한 경기나 이번 주 이변 픽, 마킹 조합에 대해 편하게 물어보세요!", {})

        # 2. 이변 / 역배 / 함정 경기 질문
        if any(w in text_lower for w in ['이변', '역배', '함정', '터질', '부러질', '위험한', '몰표']):
            if self.toto_analyzer:
                r_num = parsed['round_num'] or 260051
                try:
                    r_data = self.toto_analyzer.analyze_round(r_num)
                    traps = [m for m in r_data['matches'] if m['Is_Trap_Warning']]
                    if traps:
                        top_trap = traps[0]
                        return (
                            f"승무패 {r_num}회차에서 가장 조심해야 할 이변 경기는 {top_trap['Match_No']}번 [{top_trap['HomeTeam']} vs {top_trap['AwayTeam']}] 경기예요. "
                            f"대중 투표율은 한쪽으로 몰렸지만, AI 모델 분석 결과 {top_trap['Trap_Reason']} 이유로 이변 가능성이 매우 높습니다. "
                            f"단통보다는 반드시 [{top_trap['AI_Double_Pick']}] 복식 마킹으로 방어하시는 것을 강력 추천드려요!",
                            {'traps': traps}
                        )
                except Exception:
                    pass
            return ("빅클럽이 다음 주 챔피언스리그 16강이나 8강 원정 경기를 앞두고 있거나, 강등권 팀과 맞붙을 때 대중 몰표 함정이 가장 자주 터져요! 이런 경기는 단통을 피하고 무승부나 역배 복식을 섞는 게 1등의 핵심 비결입니다.", {})

        # 3. 1등 조합 / 배팅 노하우 / 마킹 추천 질문
        if any(w in text_lower for w in ['조합', '1등', '마킹', '배팅', '공략', '비결', '어떻게 사', '구매']):
            return ("승무패 1등을 노릴 때는 모두가 찍는 정배당만 가면 상금이 얼마 안 돼요. AI 분석상 확실한 9경기는 '단통'으로 줄이고, 대중 몰표 함정이 감지된 5경기는 '승무' 또는 '무패' 같은 '복식(투픽)'으로 커버하는 '9단통 + 5복식' 조합이 가장 적중률과 배당 가성비가 뛰어납니다!", {})

        # 4. 강등권 / 생존 싸움 질문
        if parsed['is_relegation_talk']:
            return ("시즌 후반기인 3월부터 5월까지는 17위에서 20위 강등권 팀들의 결사항전 동기부여가 엄청나요. 데이터상 강등권 팀은 안방 경기에서 승점 획득 확률이 평소보다 약 15% 이상 급상승하니 역배 이변을 꼭 염두에 두셔야 해요!", {})

        # 5. 챔피언스리그 / 로테이션 리스크 질문
        if parsed['next_match_importance'] > 0 and len(parsed['found_teams']) == 0:
            return ("유럽 챔피언스리그 토너먼트를 치르는 맨시티, 아스날, 레알 마드리드 같은 강팀들은 주말 리그 경기에서 주전 체력 안배나 로테이션을 돌릴 확률이 매우 높아요. 이런 경기가 바로 승무패에서 가장 많이 부러지는 대표적인 '룩어헤드 트랩' 경기입니다.", {})

        # 6. 현재 가장 강한 팀 / 랭킹 / 폼 질문
        if any(w in text_lower for w in ['제일 잘해', '1위', '우승', '폼 좋은', '순위', '랭킹', '누가 강해']):
            rank_df = self.predictor.get_team_rankings()
            top3 = rank_df.head(3)
            names = ", ".join([f"{r['팀명']}(Elo {int(r['Elo 레이팅'])}점)" for _, r in top3.iterrows()])
            return (f"현재 데이터 기준 파워 랭킹 톱3는 {names}입니다! 이 팀들은 공수 밸런스와 최근 승점 획득 페이스가 리그 최상위권을 유지하고 있어요.", {})

        # 7. 승무패 특정 회차/경기 질문 (52회, 51회, 14경기 등)
        if ('승무패' in text_lower or '토토' in text_lower or parsed['match_no'] is not None or parsed['round_num'] is not None) and self.toto_analyzer:
            avail_rounds = self.toto_analyzer.get_available_rounds()
            r_num = str(parsed['round_num'] or '260051')
            
            # 입력된 회차가 데이터셋에 없을 경우 (예: 260052 등 최신 예정 회차)
            if r_num not in avail_rounds:
                short_r = r_num[-2:] if len(r_num) >= 2 else r_num
                return (
                    f"승무패 {short_r}회차는 현재 최신 발매/진행 예정 회차입니다! "
                    f"가장 최근 완료된 51회차(260051) 분석표는 '배트맨 승무패 14경기' 탭에서 확인하실 수 있으며, "
                    f"{short_r}회차에 편성된 경기(예: '아스날 대 첼시 누가 이겨?' 또는 '손흥민 토트넘 경기 어때?')를 팀명으로 말씀해 주시면 5개년 전력 빅데이터를 바탕으로 즉시 승무패 확률과 전술 상성을 정밀 분석해 드릴게요!",
                    {}
                )
            
            m_no = parsed['match_no']
            try:
                r_data = self.toto_analyzer.analyze_round(r_num)
                if m_no:
                    match_item = next((m for m in r_data['matches'] if m['Match_No'] == m_no), None)
                    if match_item:
                        trap_alert = f" 🚨 주의하세요! {match_item['Trap_Reason']}" if match_item['Is_Trap_Warning'] else " 전력상 비교적 안정적인 경기입니다."
                        script = (
                            f"승무패 {r_num[-2:]}회차 {m_no}번 [{match_item['HomeTeam']} vs {match_item['AwayTeam']}] 분석입니다. "
                            f"대중 투표율은 홈승 {match_item['Vote_H']}%, 무 {match_item['Vote_D']}%, 원정승 {match_item['Vote_A']}%이지만, "
                            f"AI 정밀 분석 확률은 홈승 {match_item['AI_Prob_H']}%, 무승부 {match_item['AI_Prob_D']}%, 원정승 {match_item['AI_Prob_A']}%예요. "
                            f"AI의 단통 추천은 [{match_item['AI_Single_Pick']}], 복식 추천은 [{match_item['AI_Double_Pick']}]입니다.{trap_alert}"
                        )
                        return (script, {'match': match_item})
                else:
                    traps = [m for m in r_data['matches'] if m['Is_Trap_Warning']]
                    script = (
                        f"승무패 {r_num[-2:]}회차 14경기 전체 분석 결과예요. "
                        f"AI 단통 기준 14경기 중 {r_data['correct_count']}경기를 적중시켰고, 대중 몰표 함정은 총 {len(traps)}경기가 감지되었습니다. "
                        f"상세한 14경기 표와 마킹 조합은 '배트맨 승무패 14경기' 탭에서 한눈에 확인하실 수 있어요!"
                    )
                    return (script, r_data)
            except Exception as e:
                pass

        # 8. 팀 매치업 분석
        teams = parsed['found_teams']

        if len(teams) < 2:
            if len(teams) == 1:
                t = teams[0]
                rank_df = self.predictor.get_team_rankings()
                team_row = rank_df[rank_df['팀명'] == t]
                if not team_row.empty:
                    elo = int(team_row.iloc[0]['Elo 레이팅'])
                    atk = round(team_row.iloc[0]['공격력'], 1)
                    dfn = round(team_row.iloc[0]['수비 조직력'], 1)
                    return (f"{t}은 현재 파워 레이팅 {elo}점, 공격력 {atk}점, 수비 조직력 {dfn}점을 기록 중이에요. 상대할 팀도 함께 말씀해 주시면(예: '{t} 대 첼시 누가 이겨?') 맞춤 전술 상성과 승패 확률을 정확히 계산해 드릴게요!", {})
                return (f"네, {t}을 확인했어요! 상대할 팀도 함께 알려주시면 승패를 분석해 드릴게요. 예를 들어 '{t} 대 첼시 경기 분석해줘'라고 말씀해 보세요.", {})
            else:
                return ("축구 경기나 승무패에 대해 궁금한 점을 편하게 물어보세요! 예를 들어 '아스날 대 첼시 누가 이겨?', '이번 주 이변 경기 알려줘', '승무패 1등 조합 노하우 알려줘'처럼 질문하시면 똑똑하게 답변해 드릴게요!", {})

        home_team, away_team = teams[0], teams[1]
        next_imp = parsed['next_match_importance']

        # 정밀 승패 예측
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

        # 음성 대본
        speech_lines = []
        speech_lines.append(f"네! 5개년 전력과 전술 상성을 분석해 드립니다.")
        
        if res['predicted_outcome'] == '홈팀 승리 (Home Win)':
            speech_lines.append(f"홈팀 {home_team}의 승리 확률이 {h_prob}퍼센트로 가장 우세해요.")
        elif res['predicted_outcome'] == '원정팀 승리 (Away Win)':
            speech_lines.append(f"원정팀 {away_team}의 승리 확률이 {a_prob}퍼센트로 역배당 또는 우세가 예상돼요.")
        else:
            speech_lines.append(f"두 팀의 팽팽한 상성으로 무승부 확률이 {d_prob}퍼센트로 매우 높게 나타났어요.")

        speech_lines.append(f"확률 분포는 {home_team} 승 {h_prob}%, 무 {d_prob}%, {away_team} 승 {a_prob}%입니다.")

        for ins in res['insights'][:2]:
            clean_ins = ins.replace('**', '').replace('🚨', '').replace('⚠️', '').replace('🎯', '').replace('🔥', '').replace('⚔️', '')
        voice_script = " ".join(speech_lines)
        return voice_script, res
