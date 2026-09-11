import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
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
    voice_agent = FootballVoiceAgent(predictor)
    toto_analyzer = BetmanTotoAnalyzer(model_pkg_path=pkg_path)
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

# 탭 구성 (배트맨 승무패 14경기 탭 추가)
tab_toto, tab_voice, tab1, tab2, tab3 = st.tabs([
    "🎟️ 배트맨 승무패 14경기 분석", 
    "🎙️ AI 음성 대화 (Voice Chat)", 
    "🔮 단일 매치 정밀 분석", 
    "📈 AI 모델 피처 중요도", 
    "🏆 팀별 실시간 랭킹"
])

# ==================== 탭 0: 배트맨 승무패 14경기 분석 ====================
with tab_toto:
    st.subheader("🎟️ 배트맨 축구토토 승무패 회차별 14경기 AI 분석 & 대중 몰표 함정 탐지")
    st.markdown("""
    **작년 시즌(250001회)부터 올해(260051회)**까지의 배트맨 승무패 14경기 전 회차 데이터가 학습되어 있습니다.
    대중의 투표율(%)과 AI의 정밀 확률을 비교하여 **대중 몰표 함정(이변/역배)을 피하고 1등을 노리는 최적 마킹**을 추천합니다.
    """)

    available_rounds = toto_analyzer.get_available_rounds()
    selected_round = st.selectbox("📅 승무패 회차 선택", available_rounds, index=0)

    if selected_round:
        round_res = toto_analyzer.analyze_round(selected_round)
        
        # 상단 요약 지표
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            st.metric("회차 번호", f"{selected_round}회차", f"총 {round_res['total_matches']}경기")
        with r_col2:
            st.metric("AI 단통 적중 수", f"{round_res['correct_count']} / 14 경기", f"적중률 {round_res['accuracy_rate']}%")
        with r_col3:
            st.metric("🚨 대중 몰표 함정 감지", f"{round_res['trap_detected_count']} 경기", "이변/역배 주의")
        with r_col4:
            st.metric("추천 조합 방식", "단통 9 + 복식 5", "1등 독식 타겟")

        st.write("---")
        st.markdown(f"### 📋 {selected_round}회차 14경기 상세 AI 분석표")

        # 14경기 테이블 데이터 구성
        table_rows = []
        for m in round_res['matches']:
            trap_tag = "🚨 함정 주의" if m['Is_Trap_Warning'] else "안전"
            hit_tag = "✅ 적중" if m['Is_Correct'] else "❌ 미적중"
            table_rows.append({
                '번호': f"{m['Match_No']}번",
                '홈 팀 vs 원정 팀': f"{m['HomeTeam']} vs {m['AwayTeam']}",
                '베트맨 대중 투표율 (승/무/패)': f"{m['Vote_H']}% / {m['Vote_D']}% / {m['Vote_A']}%",
                'AI 예측 확률 (승/무/패)': f"{m['AI_Prob_H']}% / {m['AI_Prob_D']}% / {m['AI_Prob_A']}%",
                'AI 단통 추천': f"[{m['AI_Single_Pick']}]",
                'AI 복식 추천': m['AI_Double_Pick'],
                '실제 결과': m['Actual_Result'],
                '적중 여부': hit_tag,
                '이변 위험도': trap_tag
            })
            
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=520)

        # 회차 내 함정 경기 상세 브리핑
        traps = [m for m in round_res['matches'] if m['Is_Trap_Warning']]
        if traps:
            st.markdown("#### 🚨 이 회차의 핵심 이변(대중 몰표 함정) 분석")
            for t in traps:
                st.warning(f"**{t['Match_No']}번 [{t['HomeTeam']} vs {t['AwayTeam']}]**: {t['Trap_Reason']}")

# ==================== 탭 1: AI 음성 대화 ====================
with tab_voice:
    st.markdown("""
    <div class="voice-box">
        <h3 style="margin-top:0; color:#1E3A8A;">🎙️ 축구 분석 AI와 음성으로 대화하기</h3>
        <p style="color:#475569; margin-bottom:0.5rem;">
            마이크를 켜고 편하게 말해보세요! AI가 질문을 알아듣고 분석 결과와 이유를 <b>한국어 음성</b>으로 직접 브리핑해 드립니다.
        </p>
        <p style="font-size:0.9rem; color:#6B7280;">
            💡 <b>말씀 예시</b>: <br>
            • <i>"승무패 51회차 1번 경기 분석해줘"</i><br>
            • <i>"아스날이랑 첼시 경기 누가 이겨?"</i><br>
            • <i>"레알 마드리드 다음 경기 챔스인데 바르샤전 어때?"</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

    voice_component_html = """
    <div style="font-family: sans-serif; text-align: center; padding: 10px;">
        <button id="permBtn" style="
            background: #10B981; color: white; border: none; border-radius: 8px;
            padding: 8px 16px; font-weight: bold; cursor: pointer; margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        " onclick="requestMobilePermissions()">
            📲 핸드폰 마이크 & 실시간 알림 권한 허용하기
        </button>
        <br>
        <button id="recordBtn" style="
            background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
            color: white; border: none; border-radius: 50px;
            padding: 14px 28px; font-size: 1.1rem; font-weight: bold;
            cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        ">
            🎤 마이크 켜고 말하기
        </button>
        <div id="statusText" style="margin-top: 10px; font-size: 0.95rem; color: #64748B;">버튼을 누르고 말씀하세요...</div>
    </div>

    <script>
        function requestMobilePermissions() {
            // 1. 알림 권한 요청
            if ('Notification' in window) {
                Notification.requestPermission().then(function(permission) {
                    if (permission === 'granted') {
                        alert('✅ 핸드폰 알림 권한이 허용되었습니다! 승무패 마감 전 픽 알림을 받아보실 수 있습니다.');
                        new Notification('⚽ 축구 AI 파트너', { body: '핸드폰 연동이 완료되었습니다! 언제든 편하게 말 걸어주세요.' });
                    }
                });
            }
            // 2. 마이크 권한 요청
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
                    document.getElementById('permBtn').innerText = '✅ 핸드폰 권한 연동 완료 (마이크/알림 ON)';
                    document.getElementById('permBtn').style.background = '#059669';
                }).catch(function(err) {
                    console.log('Mic error:', err);
                });
            }
        }
        const recordBtn = document.getElementById('recordBtn');
        const statusText = document.getElementById('statusText');
        
        let recognition = null;
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.lang = 'ko-KR';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onstart = function() {
                recordBtn.style.background = '#DC2626';
                recordBtn.innerText = '🔴 듣고 있습니다... 말씀하세요!';
                statusText.innerText = '음성을 인식 중입니다...';
            };

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                statusText.innerText = '인식 완료: "' + transcript + '"';
                recordBtn.style.background = '#10B981';
                recordBtn.innerText = '✅ 음성 인식 완료';
                
                const inputArea = window.parent.document.querySelector('input[data-testid="stTextInputRootElement"] input');
                if (inputArea) {
                    inputArea.value = transcript;
                    inputArea.dispatchEvent(new Event('input', { bubbles: true }));
                    inputArea.dispatchEvent(new Event('change', { bubbles: true }));
                }
            };

            recognition.onerror = function(event) {
                recordBtn.style.background = '#3B82F6';
                recordBtn.innerText = '🎤 마이크 켜고 말하기';
                statusText.innerText = '인식 오류: ' + event.error;
            };

            recognition.onend = function() {
                recordBtn.style.background = '#3B82F6';
                recordBtn.innerText = '🎤 마이크 켜고 말하기';
            };
        } else {
            statusText.innerText = '현재 브라우저에서 마이크 Web Speech API를 지원하지 않습니다. 아래 텍스트 창에 입력해 주세요.';
        }

        recordBtn.addEventListener('click', function() {
            if (recognition) {
                try {
                    recognition.start();
                } catch(e) {
                    recognition.stop();
                }
            }
        });
    </script>
    """
    components.html(voice_component_html, height=120)

    user_query = st.text_input("💬 음성 인식 텍스트 (또는 직접 입력):", placeholder="예: 아스날 대 토트넘 다음 경기 챔스인데 누가 이겨?")

    if user_query:
        voice_script, voice_res = voice_agent.generate_voice_response(user_query)
        
        st.markdown("#### 🤖 AI 음성 답변")
        st.info(f"🗣️ **AI**: {voice_script}")

        # Web Speech Synthesis (TTS) 한국어 여성 목소리 설정
        tts_html = f"""
        <script>
            function speakFemale(text) {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = 'ko-KR';
                    utterance.rate = 1.05;
                    utterance.pitch = 1.15; // 자연스럽고 맑은 여성 톤

                    // 한국어 여성 음성 탐색 및 선택
                    const voices = window.speechSynthesis.getVoices();
                    const femaleVoice = voices.find(v => 
                        v.lang.includes('ko') && (v.name.includes('Yuna') || v.name.includes('Heami') || v.name.includes('SunHi') || v.name.includes('Female') || v.name.includes('여성') || v.name.includes('Google 한국의'))
                    ) || voices.find(v => v.lang.includes('ko'));

                    if (femaleVoice) {
                        utterance.voice = femaleVoice;
                    }
                    window.speechSynthesis.speak(utterance);
                }}
            }}
            // 음성 목록 로딩 대기 후 재생
            if (window.speechSynthesis.getVoices().length === 0) {
                window.speechSynthesis.onvoiceschanged = () => speakFemale({repr(voice_script)});
            } else {
                speakFemale({repr(voice_script)});
            }
        </script>
        <button onclick="speakFemale({repr(voice_script)})" style="
            background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%);
            color: white; border: none; border-radius: 6px;
            padding: 8px 16px; font-weight: 600; cursor: pointer; margin-top: 5px;
        ">
            👩‍💼 여성 음성으로 다시 듣기
        </button>
        """
        components.html(tts_html, height=50)

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
