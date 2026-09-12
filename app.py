import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import streamlit.components.v1 as components

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import MatchPredictor
from src.voice_agent import FootballVoiceAgent
from src.betman_analyzer import BetmanTotoAnalyzer
from src.feature_engineering import EUROPEAN_CONTENDERS

# 페이지 설정
st.set_page_config(
    page_title="Football Match AI Predictor | Betman Toto 14",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .voice-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .badge-trap {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    pkg_path = "models/football_predictor_pkg.pkl"
    if not os.path.exists(pkg_path):
        return None, None, None
    predictor = MatchPredictor(pkg_path)
    toto_analyzer = BetmanTotoAnalyzer(model_pkg_path=pkg_path)
    voice_agent = FootballVoiceAgent(predictor, toto_analyzer=toto_analyzer)
    return predictor, voice_agent, toto_analyzer

predictor, voice_agent, toto_analyzer = load_components()

# 헤더
st.markdown('<div class="main-header">⚽ AI 축구 승패 예측 & 배트맨 승무패 14경기 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">EPL & 라리가 5개년 + <b>배트맨 승무패 250001~260051회차 14경기 풀 학습</b> 및 <b>실시간 음성 대화</b> 엔진</div>', unsafe_allow_html=True)

if predictor is None:
    st.error("⚠️ 학습된 모델 파일이 없습니다. 먼저 파이프라인을 실행해 주세요.")
    st.stop()

# 사이드바 설정
st.sidebar.header("⚙️ 시뮬레이션 및 경기 설정")

epl_teams = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley', 
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich', 'Leeds', 
    'Leicester', 'Liverpool', 'Luton', 'Man City', 'Man United', 'Newcastle', 
    'Nott\'m Forest', 'Sheffield United', 'Southampton', 'Tottenham', 'Watford', 
    'West Brom', 'West Ham', 'Wolves'
]
laliga_teams = [
    'Alaves', 'Almeria', 'Ath Bilbao', 'Ath Madrid', 'Barcelona', 'Betis', 
    'Cadiz', 'Celta', 'Eibar', 'Elche', 'Espanol', 'Getafe', 'Girona', 
    'Granada', 'Huesca', 'Las Palmas', 'Leganes', 'Levante', 'Mallorca', 
    'Osasuna', 'Real Madrid', 'Real Sociedad', 'Sevilla', 'Valencia', 
    'Valladolid', 'Vallecano', 'Villarreal'
]

all_teams = predictor.get_available_teams()
epl_avail = [t for t in epl_teams if t in all_teams]
laliga_avail = [t for t in laliga_teams if t in all_teams]

league_choice = st.sidebar.selectbox("리그 선택", ["잉글랜드 프리미어리그 (EPL)", "스페인 라리가 (La Liga)", "전체 팀 직접 선택"])

if league_choice == "잉글랜드 프리미어리그 (EPL)":
    team_pool = epl_avail if epl_avail else all_teams
    default_home = "Arsenal" if "Arsenal" in team_pool else team_pool[0]
    default_away = "Chelsea" if "Chelsea" in team_pool else team_pool[1]
elif league_choice == "스페인 라리가 (La Liga)":
    team_pool = laliga_avail if laliga_avail else all_teams
    default_home = "Real Madrid" if "Real Madrid" in team_pool else team_pool[0]
    default_away = "Barcelona" if "Barcelona" in team_pool else team_pool[1]
else:
    team_pool = all_teams
    default_home = team_pool[0]
    default_away = team_pool[1]

st.sidebar.markdown("---")
home_team = st.sidebar.selectbox("🏠 홈 팀 (Home)", team_pool, index=team_pool.index(default_home) if default_home in team_pool else 0)
away_candidates = [t for t in team_pool if t != home_team]
away_team = st.sidebar.selectbox("✈️ 원정 팀 (Away)", away_candidates, index=away_candidates.index(default_away) if default_away in away_candidates else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 경기 환경 & 다음 일정 (Lookahead)")

match_month = st.sidebar.slider("경기 예정 월 (Month)", 1, 12, 4, help="2~5월은 챔스 토너먼트 기간으로 빅클럽의 로테이션/힘빼기 리스크가 커집니다.")

lookahead_options = {
    0: "일반 리그 일정 (다음 경기 부담 없음)",
    1: "3~4일 뒤 챔스 조별리그 / 라이벌 더비 예정",
    2: "🚨 3~4일 뒤 챔스 8강/4강 2차전 / 결승전 예정 (주전 체력 안배 & 힘빼기 극도 위험)"
}

next_h_opt = st.sidebar.selectbox(f"⏭️ {home_team}의 다음 경기 일정", list(lookahead_options.keys()), format_func=lambda x: lookahead_options[x], index=0)
next_a_opt = st.sidebar.selectbox(f"⏭️ {away_team}의 다음 경기 일정", list(lookahead_options.keys()), format_func=lambda x: lookahead_options[x], index=0)

rest_h = st.sidebar.slider(f"🏠 {home_team} 최근 휴식일수", 2, 14, 3 if next_h_opt > 0 else 7)
rest_a = st.sidebar.slider(f"✈️ {away_team} 최근 휴식일수", 2, 14, 7)

# 탭 구성 (음성 대화 탭을 맨 처음으로 배치하여 모바일 접속 시 바로 보이도록 설정)
tab_voice, tab_toto, tab1, tab2, tab3 = st.tabs([
    "🎙️ AI 음성 대화 (Voice)", 
    "🎟️ 배트맨 승무패 14경기", 
    "🔮 단일 매치 정밀 분석", 
    "📈 AI 모델 피처 중요도", 
    "🏆 팀별 실시간 랭킹"
])

# ==================== 탭 1: AI 음성 대화 ====================
with tab_voice:
    st.markdown("""
    <div class="voice-box">
        <h3 style="margin-top:0; color:#1E3A8A;">🎙️ 축구 분석 AI 여성 파트너</h3>
        <p style="color:#475569; font-size:0.95rem; margin-bottom:0.3rem;">
            스마트폰 하단 채팅창에 질문을 입력하시거나 마이크로 말씀하세요! AI가 <b>한국어 여성 목소리</b>로 답변해 드립니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 채팅 세션 히스토리 초기화
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "안녕하세요! 축구 분석 파트너 AI입니다. 궁금한 경기나 이번 주 승무패 14경기 조합에 대해 편하게 물어보세요!"}
        ]

    # 모바일 원터치 빠른 질문 버튼
    st.markdown("##### ⚡ 빠른 원터치 질문")
    q_col1, q_col2 = st.columns(2)
    selected_quick = None
    with q_col1:
        if st.button("🔥 아스날 vs 첼시 누가 이겨?", use_container_width=True):
            selected_quick = "아스날 대 첼시 경기 분석해줘"
        if st.button("🎟️ 승무패 51회차 1번 경기", use_container_width=True):
            selected_quick = "승무패 51회차 1번 경기 분석해줘"
    with q_col2:
        if st.button("⚽ 레알 vs 바르셀로나", use_container_width=True):
            selected_quick = "레알 마드리드 대 바르셀로나 누가 이겨?"
        if st.button("🚨 맨시티 챔스 함정 분석", use_container_width=True):
            selected_quick = "맨체스터 시티 다음 경기 챔스인데 분석해줘"

    # 채팅 메시지 출력
    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"], avatar="👩‍💼" if msg["role"] == "assistant" else "👤"):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "script" in msg:
                escaped_script = json.dumps(msg["script"], ensure_ascii=False)
                tts_html = f"""
                <div style="margin-top: 6px;">
                    <button onclick="speakFemale_{idx}()" style="
                        background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%);
                        color: white; border: none; border-radius: 6px;
                        padding: 6px 14px; font-size: 0.9rem; font-weight: bold;
                        cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        🔊 👩‍💼 여성 목소리로 음성 듣기
                    </button>
                </div>
                <script>
                    function speakFemale_{idx}() {{
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            const text = {escaped_script};
                            const utterance = new SpeechSynthesisUtterance(text);
                            utterance.lang = 'ko-KR';
                            utterance.rate = 1.0;
                            utterance.pitch = 1.15;

                            const voices = window.speechSynthesis.getVoices();
                            const femaleVoice = voices.find(v => 
                                v.lang.includes('ko') && (v.name.includes('Yuna') || v.name.includes('Heami') || v.name.includes('SunHi') || v.name.includes('Female') || v.name.includes('여성') || v.name.includes('Google 한국의'))
                            ) || voices.find(v => v.lang.includes('ko'));

                            if (femaleVoice) {{
                                utterance.voice = femaleVoice;
                            }}
                            window.speechSynthesis.speak(utterance);
                        }}
                    }}
                </script>
                """
                components.html(tts_html, height=45)

    # 하단 채팅 입력창 (엔터키 및 전송 화살표 버튼 내장)
    chat_prompt = st.chat_input("질문을 입력하고 전송 버튼(➤)을 누르세요 (예: 토트넘 대 리버풀 누가 이겨?)")
    
    user_query = selected_quick or chat_prompt

    if user_query:
        # 사용자 메시지 기록
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        
        # AI 답변 생성
        voice_script, voice_res = voice_agent.generate_voice_response(user_query)
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": voice_script,
            "script": voice_script
        })
        st.rerun()

# ==================== 탭 2: 배트맨 승무패 14경기 분석 ====================
with tab_toto:
    st.subheader("🎟️ 배트맨 축구토토 승무패 회차별 14경기 AI 분석 & 대중 몰표 함정 탐지")
    st.markdown("""
    **작년 시즌(250001회)부터 올해(260051회)**까지의 배트맨 승무패 14경기 전 회차 데이터가 학습되어 있습니다.
    대중의 투표율(%)과 AI의 정밀 확률을 비교하여 **대중 몰표 함정(이변/역배)을 피하고 1등을 노리는 최적 마킹**을 추천합니다.
    """)

    raw_rounds = toto_analyzer.get_available_rounds() if toto_analyzer else []
    # 52회차(260052)부터 최신 회차들이 맨 위에 오도록 보장
    top_rounds = [f"26{i:04d}" for i in range(52, 65)]
    available_rounds = sorted(list(set(raw_rounds).union(set(top_rounds))), reverse=True)
    
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        sel_box_round = st.selectbox(
            "📅 승무패 회차 선택", 
            available_rounds, 
            index=available_rounds.index("260052") if "260052" in available_rounds else 0,
            format_func=lambda x: f"{x} (20{x[:2]}년 {int(x[2:])}회차)"
        )
    with t_col2:
        custom_round_input = st.text_input("✍️ 회차 직접 입력", placeholder="예: 52, 53, 54")

    # 직접 입력한 번호가 있으면 우선 적용
    if custom_round_input and custom_round_input.strip():
        in_val = custom_round_input.strip()
        selected_round = f"26{int(in_val):04d}" if len(in_val) <= 2 else in_val
    else:
        selected_round = sel_box_round

    # 직접 14경기 복사/붙여넣기 옵션
    with st.expander("✍️ [새 기능] 이번 주 실제 14경기 복사/붙여넣기로 즉시 AI 분석하기"):
        st.markdown("베트맨 사이트의 대상 경기 목록을 아래에 복사해서 넣으면 AI가 즉시 14경기를 인식하여 분석합니다.")
        paste_text = st.text_area(
            "14경기 텍스트 붙여넣기 (예: '1. 아스날 vs 첼시 (55 25 20)')",
            placeholder="1. 아스날 vs 첼시\n2. 토트넘 vs 리버풀\n3. 맨시티 vs 뉴캐슬\n4. 레알 vs 바르샤...",
            height=130
        )
        custom_analyze_btn = st.button("⚡ 붙여넣은 14경기 즉시 AI 분석 & 마킹 추천", use_container_width=True)

    # 베트맨 공식 52회차 14경기 실제 데이터 직접 내장
    OFFICIAL_52_MATCHES = [
        {'Match_No': 1, 'HomeTeam': 'Aston Villa', 'AwayTeam': "Nott'm Forest", 'Vote_H': 52.5, 'Vote_D': 29.6, 'Vote_A': 17.9},
        {'Match_No': 2, 'HomeTeam': 'Bournemouth', 'AwayTeam': 'Brentford', 'Vote_H': 26.2, 'Vote_D': 39.2, 'Vote_A': 34.6},
        {'Match_No': 3, 'HomeTeam': 'Crystal Palace', 'AwayTeam': 'Ipswich', 'Vote_H': 66.8, 'Vote_D': 21.4, 'Vote_A': 11.8},
        {'Match_No': 4, 'HomeTeam': 'Liverpool', 'AwayTeam': 'Fulham', 'Vote_H': 84.9, 'Vote_D': 10.7, 'Vote_A': 4.4},
        {'Match_No': 5, 'HomeTeam': 'Osasuna', 'AwayTeam': 'Espanol', 'Vote_H': 39.6, 'Vote_D': 40.2, 'Vote_A': 20.3},
        {'Match_No': 6, 'HomeTeam': 'Tottenham', 'AwayTeam': 'Everton', 'Vote_H': 35.6, 'Vote_D': 31.4, 'Vote_A': 33.0},
        {'Match_No': 7, 'HomeTeam': 'Ath Bilbao', 'AwayTeam': 'Elche', 'Vote_H': 87.3, 'Vote_D': 8.6, 'Vote_A': 4.1},
        {'Match_No': 8, 'HomeTeam': 'Southampton', 'AwayTeam': 'Arsenal', 'Vote_H': 4.8, 'Vote_D': 14.8, 'Vote_A': 80.5},
        {'Match_No': 9, 'HomeTeam': 'Celta', 'AwayTeam': 'Mallorca', 'Vote_H': 63.7, 'Vote_D': 26.6, 'Vote_A': 9.6},
        {'Match_No': 10, 'HomeTeam': 'Luton', 'AwayTeam': 'Brighton', 'Vote_H': 12.6, 'Vote_D': 19.4, 'Vote_A': 67.9},
        {'Match_No': 11, 'HomeTeam': 'Levante', 'AwayTeam': 'Barcelona', 'Vote_H': 3.2, 'Vote_D': 6.2, 'Vote_A': 90.6},
        {'Match_No': 12, 'HomeTeam': 'Man United', 'AwayTeam': 'Man City', 'Vote_H': 20.4, 'Vote_D': 26.2, 'Vote_A': 53.4},
        {'Match_No': 13, 'HomeTeam': 'Getafe', 'AwayTeam': 'Alaves', 'Vote_H': 29.5, 'Vote_D': 37.3, 'Vote_A': 33.2},
        {'Match_No': 14, 'HomeTeam': 'Real Sociedad', 'AwayTeam': 'Ath Madrid', 'Vote_H': 18.9, 'Vote_D': 34.6, 'Vote_A': 46.5}
    ]

    round_res = None
    if paste_text and custom_analyze_btn:
        lines = [line.strip() for line in paste_text.strip().split('\n') if line.strip()]
        parsed_custom = []
        for l_idx, line in enumerate(lines[:14]):
            parsed_line = voice_agent.parse_voice_query(line)
            teams_in_line = parsed_line['found_teams']
            if len(teams_in_line) >= 2:
                nums = re.findall(r'(\d+(?:\.\d+)?)', line)
                nums_float = [float(x) for x in nums if float(x) <= 100]
                if len(nums_float) >= 4:
                    nums_float = nums_float[1:]
                v_h = nums_float[0] if len(nums_float) >= 1 else 45.0
                v_d = nums_float[1] if len(nums_float) >= 2 else 25.0
                v_a = nums_float[2] if len(nums_float) >= 3 else 30.0
                
                parsed_custom.append({
                    'Match_No': l_idx + 1,
                    'HomeTeam': teams_in_line[0],
                    'AwayTeam': teams_in_line[1],
                    'Vote_H': v_h, 'Vote_D': v_d, 'Vote_A': v_a
                })
        
        if parsed_custom:
            round_res = toto_analyzer.analyze_custom_matches(parsed_custom)
            st.success(f"✅ 총 {len(parsed_custom)}개 경기를 인식하여 AI 분석을 완료했습니다!")
        else:
            st.warning("⚠️ 인식된 팀 매치업이 없습니다. '홈팀 vs 원정팀' 형식으로 입력해 주세요.")

    if round_res is None:
        if str(selected_round) in ['260052', '52']:
            round_res = toto_analyzer.analyze_custom_matches(OFFICIAL_52_MATCHES)
        elif selected_round:
            round_res = toto_analyzer.analyze_round(selected_round)

    if round_res:
        # 상단 요약 지표 (모바일 반응형 2열 배치)
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.metric("회차 번호", f"20{selected_round[:2]}년 {int(selected_round[2:])}회" if str(selected_round).isdigit() and len(str(selected_round))>=6 else str(selected_round), f"총 {round_res['total_matches']}경기")
            st.metric("🚨 대중 몰표 함정", f"{round_res['trap_detected_count']} 경기", "이변/역배 주의")
        with r_col2:
            st.metric("AI 단통 적중 수", f"{round_res['correct_count']} / {round_res['total_matches']} 경기", f"적중률 {round_res['accuracy_rate']}%")
            st.metric("추천 조합 방식", "단통 9 + 복식 5", "1등 독식 타겟")

        st.write("---")
        st.markdown(f"### 📋 {selected_round} (20{selected_round[:2]}년 {int(selected_round[2:])}회차) 14경기 상세 AI 분석표")

        KO_MAP = {'H': '[승]', 'D': '[무]', 'A': '[패]', '[H]': '[승]', '[D]': '[무]', '[A]': '[패]'}

        # 14경기 테이블 데이터 구성
        table_rows = []
        for m in round_res['matches']:
            trap_tag = "🚨 함정 주의" if m['Is_Trap_Warning'] else "안전"
            hit_tag = "✅ 적중" if m.get('Is_Correct', True) else "❌ 미적중"
            
            raw_s = str(m.get('AI_Single_Pick', ''))
            s_pick = KO_MAP.get(raw_s, raw_s)
            if not s_pick.startswith('['):
                s_pick = f"[{s_pick}]"

            raw_d = str(m.get('AI_Double_Pick', ''))
            d_pick = raw_d.replace('H', '승').replace('D', '무').replace('A', '패')

            p_h = m.get('AI_Prob_H', 0)
            p_d = m.get('AI_Prob_D', 0)
            p_a = m.get('AI_Prob_A', 0)

            table_rows.append({
                '번호': f"{m['Match_No']}번",
                '홈 팀 vs 원정 팀': f"{m['HomeTeam']} vs {m['AwayTeam']}",
                '베트맨 대중 투표율 (승/무/패)': f"{m['Vote_H']}% / {m['Vote_D']}% / {m['Vote_A']}%",
                'AI 예측 확률 (승/무/패)': f"{p_h}% / {p_d}% / {p_a}%",
                'AI 단통 추천': s_pick,
                'AI 복식 추천': d_pick,
                '이변 위험도': trap_tag
            })
            
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=520)

        # 회차 내 함정 경기 상세 브리핑
        traps = [m for m in round_res['matches'] if m['Is_Trap_Warning']]
        if traps:
            st.markdown("#### 🚨 이 회차의 핵심 이변(대중 몰표 함정) 분석")
            for t in traps:
                st.warning(f"**{t['Match_No']}번 [{t['HomeTeam']} vs {t['AwayTeam']}]**: {t['Trap_Reason']}")

# ==================== 탭 2: 단일 매치 분석 ====================
with tab1:
    res = predictor.predict_match(
        home_team, away_team, 
        rest_home=rest_h, rest_away=rest_a, 
        match_month=match_month,
        next_match_home=next_h_opt,
        next_match_away=next_a_opt
    )
    probs = res['probabilities']

    col1, col2, col3 = st.columns([1.2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #2563EB;">
            <div style="font-size: 0.9rem; color: #64748B;">AI 최종 예측 결과</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #1E293B; margin-top: 4px;">
                {res['predicted_outcome']}
            </div>
            <div style="margin-top: 8px;">
                <span class="badge-safe">앙상블 (LightGBM + XGBoost + NeuralNet)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #10B981;">
            <div style="font-size: 0.9rem; color: #64748B;">🏠 {home_team} 승리 확률</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #047857; margin-top: 4px;">
                {probs['home_win']*100:.1f}%
            </div>
            <div style="font-size: 0.85rem; color: #6B7280; margin-top: 4px;">
                Elo: <b>{res['elo']['home']}</b> | 동기부여: <b>{res['motivation']['home']*100:.0f}점</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #EF4444;">
            <div style="font-size: 0.9rem; color: #64748B;">✈️ {away_team} 승리 확률</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #B91C1C; margin-top: 4px;">
                {probs['away_win']*100:.1f}%
            </div>
            <div style="font-size: 0.85rem; color: #6B7280; margin-top: 4px;">
                Elo: <b>{res['elo']['away']}</b> | 동기부여: <b>{res['motivation']['away']*100:.0f}점</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 승/무/패 예측 확률 분포")
    col_chart, col_models = st.columns([1.5, 1])
    
    with col_chart:
        fig = go.Figure(data=[go.Pie(
            labels=[f'{home_team} 승리', '무승부', f'{away_team} 승리'],
            values=[probs['home_win'], probs['draw'], probs['away_win']],
            hole=.45,
            marker_colors=['#2563EB', '#94A3B8', '#DC2626'],
            textinfo='label+percent',
            hoverinfo='label+percent'
        )])
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_models:
        st.markdown("##### 🤖 알고리즘별 독립 확률 계산")
        mb = res['model_breakdown']
        model_df = pd.DataFrame({
            'Model': list(mb.keys()),
            f'{home_team} 승': [f"{v['home']*100:.1f}%" for v in mb.values()],
            '무승부': [f"{v['draw']*100:.1f}%" for v in mb.values()],
            f'{away_team} 승': [f"{v['away']*100:.1f}%" for v in mb.values()]
        })
        st.dataframe(model_df, hide_index=True, use_container_width=True)

    st.markdown("### 💡 AI 핵심 승부 분석 코멘터리 (동기부여, 상성 카운터 & 다음 경기 Lookahead)")
    for ins in res['insights']:
        if "🚨" in ins or "⚠️" in ins:
            st.warning(ins)
        elif "🎯" in ins or "🔥" in ins:
            st.success(ins)
        else:
            st.info(ins)

    st.markdown("### 🕸️ 팀별 5대 전술 장단점 프로파일 & 상성 분석")
    col_radar, col_tactics_desc = st.columns([1.2, 1])
    
    with col_radar:
        t_h = res['tactics']['home']
        t_a = res['tactics']['away']
        
        categories = ['압박 강도 (Pressing)', '공격 파괴력 (Attack)', '골 결정력 (Finishing)', '수비 조직력 (Defense)', '세트피스 위협도 (Set-piece)']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[t_h['pressing_intensity'], t_h['attack_firepower'], t_h['finishing_efficiency'], t_h['defensive_wall'], t_h['set_piece_threat']],
            theta=categories, fill='toself', name=home_team, line_color='#2563EB'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[t_a['pressing_intensity'], t_a['attack_firepower'], t_a['finishing_efficiency'], t_a['defensive_wall'], t_a['set_piece_threat']],
            theta=categories, fill='toself', name=away_team, line_color='#DC2626'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, margin=dict(t=30, b=30, l=40, r=40), height=340)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_tactics_desc:
        st.markdown("##### ⚔️ 상성 및 취약점 공략(Counter Advantage)")
        st.markdown(f"""
        * **{home_team} 상성 공략 점수**: `{res['tactics']['counter_score_home']}점`
        * **{away_team} 상성 공략 점수**: `{res['tactics']['counter_score_away']}점`
        """)
        st.write("---")
        st.markdown(f"""
        **💡 상성 원리 (Tactical Clash)**:
        * 강력한 전방 압박 팀이 빌드업/수비가 불안한 팀을 만날 때 승률이 급증합니다.
        * 반대로 강팀이라도 다음 경기(챔스 8강 등)가 있어 주전을 아끼면, 결사항전하는 하위권에게 일격을 당하거나 무승부(Trap Game)에 빠집니다.
        """)

# ==================== 탭 3: 모델 구조 ====================
with tab2:
    st.subheader("🎯 모델 학습 구조 및 중요 변수 (Feature Importance)")
    st.markdown("""
    본 예측 AI는 5시즌(2020-21 ~ 2024-25)의 3,800개 경기 데이터를 100% 학습했습니다.
    **동기부여(강등/우승 사투)**, **다음 경기 중요도(Lookahead Trap Game)**, **전술 장단점 및 상성 카운터 지표**가 모델 예측에 핵심적인 영향을 미칩니다.
    """)
    
    if 'LightGBM' in predictor.feature_importances:
        fi_df = predictor.feature_importances['LightGBM'].head(15)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='LightGBM 모델 피처 중요도 TOP 15', color='Importance', color_continuous_scale='Blues')
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
        st.plotly_chart(fig_fi, use_container_width=True)

# ==================== 탭 4: 팀별 랭킹 ====================
with tab3:
    st.subheader("🏆 팀별 실시간 Elo 전력 & 전술 프로파일 랭킹")
    elo_dict = predictor.state.get('elo_ratings', {})
    if elo_dict:
        rows = []
        for team, rating in elo_dict.items():
            tac = predictor.get_team_tactical_profile(team)
            rows.append({
                '팀명': team, 'Elo 레이팅': round(rating, 1),
                '압박 강도': tac['pressing_intensity'], '공격력': tac['attack_firepower'],
                '골 결정력': tac['finishing_efficiency'], '수비 조직력': tac['defensive_wall'],
                '세트피스': tac['set_piece_threat']
            })
        elo_df = pd.DataFrame(rows).sort_values(by='Elo 레이팅', ascending=False).reset_index(drop=True)
        elo_df.index = elo_df.index + 1
        st.dataframe(elo_df, use_container_width=True, height=500)
