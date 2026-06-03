"""
EVE v37.7 - Streamlit Dashboard
================================

EVE의 실시간 상태 시각화 + 채팅 + 자율 거주.

실행 (Colab Pro):
    !pip install streamlit plotly
    !streamlit run dashboard.py --server.port 8501 &
    # ngrok 또는 Colab tunnel 활용

실행 (로컬):
    pip install streamlit plotly
    streamlit run dashboard.py

요소:
- 추상 EVE 시각화 (별자리/원)
- 호르몬 막대 (8 주요)
- 자기 방 객체 + 위치
- 활성 카테고리 + 속마음
- 채팅창 (사용자 ↔ EVE)
- 자율 tick (1tick / 1h / 24h)
- 호르몬 시계열 차트
"""
import sys
import os
import time as _time
from pathlib import Path

# v36 + v35 경로 자동 감지
_HERE = Path(__file__).parent if '__file__' in globals() else Path('.')
_ROOT = _HERE.parent if _HERE.name == 'v36_modules' else _HERE
_V35 = _ROOT / 'eve_v35_release'
sys.path.insert(0, str(_ROOT / 'v36_modules'))
if _V35.exists():
    sys.path.insert(0, str(_V35))
    sys.path.insert(0, str(_V35 / 'eve_modules'))

import streamlit as st
import plotly.graph_objects as go

from eve_v36 import EVE_v36
from dashboard_data import (get_eve_state, get_eve_visualization_data,
                            HormoneTracker, hormone_color)


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="EVE — Living AGI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session state — EVE 인스턴스 영속
# ============================================================
def init_eve():
    """EVE 인스턴스 생성/로드 — session_state 보관"""
    if 'eve' not in st.session_state:
        eve = EVE_v36(seed_common_sense_on_init=True, enhance_say=True)
        eve.hs.sim_hour = 9.0  # 오전 9시 시작
        eve.enter_room()
        st.session_state.eve = eve
        st.session_state.tracker = HormoneTracker(max_history=200)
        st.session_state.dialogue_log = []  # 사용자 표시용
    return st.session_state.eve, st.session_state.tracker


# ============================================================
# 추상 EVE 시각화 (plotly)
# ============================================================
def render_eve_constellation(eve):
    """별자리/원 추상 시각화"""
    viz = get_eve_visualization_data(eve)
    center = viz['center']
    nodes = viz['nodes']
    connections = viz['connections']

    fig = go.Figure()

    # 1. 연결선 (focus → 활성 카테고리)
    focus_node = next((n for n in nodes if n['is_focus']), None)
    if focus_node:
        for conn in connections:
            target = next((n for n in nodes if n['name'] == conn['to']), None)
            if target:
                fig.add_trace(go.Scatter(
                    x=[focus_node['x'], target['x']],
                    y=[focus_node['y'], target['y']],
                    mode='lines',
                    line=dict(
                        color=f'rgba(180,180,180,{0.2 + conn["strength"] * 0.5})',
                        width=1,
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                ))

    # 2. 활성 카테고리 노드
    for nd in nodes:
        c = nd['color']
        rgb = f"rgb({c[0]},{c[1]},{c[2]})"
        line_color = 'gold' if nd['is_focus'] else 'rgba(255,255,255,0.3)'
        line_width = 3 if nd['is_focus'] else 1
        fig.add_trace(go.Scatter(
            x=[nd['x']], y=[nd['y']],
            mode='markers+text',
            marker=dict(
                size=nd['size'],
                color=rgb,
                line=dict(color=line_color, width=line_width),
            ),
            text=[nd['name']],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            hoverinfo='text',
            hovertext=f"{nd['name']} (활성 {nd['level']:.2f})",
            showlegend=False,
        ))

    # 3. 중심 EVE
    c = center['color']
    rgb = f"rgb({c[0]},{c[1]},{c[2]})"
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(
            size=center['size'] * 1.8,
            color=rgb,
            line=dict(color='white', width=2),
            opacity=0.9,
        ),
        text=['EVE'],
        textposition='middle center',
        textfont=dict(size=14, color='black' if sum(c) > 380 else 'white',
                     family='Arial Black'),
        hoverinfo='text',
        hovertext=(f"EVE<br>"
                   f"feeling: {center.get('feeling','?')}<br>"
                   f"period: {center.get('period','?')}"),
        showlegend=False,
    ))

    fig.update_layout(
        title=f"🌌 EVE 의식 — {viz['time_label']} ({center.get('period','')})",
        xaxis=dict(visible=False, range=[-200, 200]),
        yaxis=dict(visible=False, range=[-200, 200], scaleanchor='x'),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ============================================================
# 호르몬 시계열 차트
# ============================================================
def render_hormone_timeline(tracker, names=None):
    if names is None:
        names = ['dopamine', 'oxytocin', 'serotonin', 'cortisol',
                 'norepinephrine', 'melatonin']
    fig = go.Figure()
    for name in names:
        ts, vals = tracker.series(name)
        if not vals:
            continue
        c = hormone_color(name)
        rgb = f"rgb({c[0]},{c[1]},{c[2]})"
        fig.add_trace(go.Scatter(
            x=list(range(len(vals))),
            y=vals,
            mode='lines',
            line=dict(color=rgb, width=2),
            name=name,
            hovertemplate=f'{name}: %{{y:.2f}}<extra></extra>',
        ))
    fig.update_layout(
        title="호르몬 시계열",
        xaxis=dict(title="시간 (스냅샷)"),
        yaxis=dict(title="레벨", range=[0, 1.1]),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.2),
    )
    return fig


# ============================================================
# 메인 — 페이지 레이아웃
# ============================================================
def main():
    eve, tracker = init_eve()
    state = get_eve_state(eve)

    # ===== 헤더 =====
    col_h1, col_h2, col_h3, col_h4 = st.columns([2, 2, 2, 2])
    with col_h1:
        st.metric("⏰ 시간", f"{state['time']['sim_hour_str']}",
                 delta=state['time']['period'])
    with col_h2:
        st.metric("📍 위치", state['space']['env_name'] or '-')
    with col_h3:
        st.metric("💭 feeling", state['feeling'] or '-')
    with col_h4:
        focus = state['mind']['focus']
        st.metric("🎯 focus", focus or '-')

    st.divider()

    # ===== 사이드바: 호르몬 + 컨트롤 =====
    with st.sidebar:
        st.markdown("### 💊 호르몬")
        primary = ['dopamine', 'oxytocin', 'serotonin', 'cortisol',
                  'norepinephrine', 'melatonin', 'endorphin', 'adenosine']
        for name in primary:
            h = state['hormones'].get(name)
            if h:
                c = h['color']
                pct = int(h['level'] * 100)
                arrow = '↑' if h['is_elevated'] else ('↓' if h['is_depleted'] else ' ')
                bar = '█' * (pct // 10) + '░' * (10 - pct // 10)
                st.markdown(
                    f"<span style='color:rgb{c}'>{bar}</span>"
                    f" <small>{name[:8]}{arrow} {h['level']:.2f}</small>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("### 🎮 컨트롤")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("⏩ 1 tick", use_container_width=True):
                eve.tick(dt=0.5)
                tracker.snapshot(eve)
                st.rerun()
            if st.button("📅 1시간", use_container_width=True):
                eve.live_in_room(hours=1.0, sim_hour_per_step=0.25)
                tracker.snapshot(eve)
                st.rerun()
        with col_c2:
            if st.button("🌅 8시간", use_container_width=True):
                eve.live_in_room(hours=8.0, sim_hour_per_step=1.0)
                tracker.snapshot(eve)
                st.rerun()
            if st.button("🌙 24시간", use_container_width=True):
                eve.live_in_room(hours=24.0, sim_hour_per_step=1.0)
                for _ in range(8):
                    tracker.snapshot(eve)
                st.rerun()

        st.divider()
        st.markdown("### ⚙️ 환경")
        env_choice = st.radio(
            "현재 위치",
            ["내방 (거주)", "거실 (사회 시뮬)", "주방 (학습 시뮬)"],
            label_visibility='collapsed',
        )
        if st.button("환경 전환", use_container_width=True):
            try:
                eve.exit_environment()
            except Exception:
                pass
            if env_choice.startswith("내방"):
                eve.enter_room()
            elif env_choice.startswith("거실"):
                from social_env import make_social_living_room
                eve.enter_environment(make_social_living_room())
            else:
                from symbolic_env import make_kitchen_env
                eve.enter_environment(make_kitchen_env())
            st.rerun()

        st.divider()
        st.markdown("### 📊 상태")
        st.caption(f"보호 카테고리: {state['protected']['protected_count']}")
        st.caption(f"신념: {state['protected']['beliefs_count']}")
        st.caption(f"카테고리 그래프: {state['protected']['sa_categories']}")

    # ===== 메인 — 시각화 + 머릿속 =====
    col_main, col_mind = st.columns([3, 2])

    with col_main:
        # 추상 EVE 별자리
        fig = render_eve_constellation(eve)
        st.plotly_chart(fig, use_container_width=True)

        # 호르몬 시계열
        if len(tracker.history) > 1:
            fig2 = render_hormone_timeline(tracker)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("💡 컨트롤로 시간 진행하면 호르몬 변화 추적")

    with col_mind:
        st.markdown("### 🧠 지금 머릿속")

        # 속마음
        voice = state['mind']['inner_voice']
        if voice:
            st.markdown(f"💭 **속마음**: _{voice}_")

        # DMN 모드
        dmn = state['mind']['dmn_mode']
        if dmn:
            st.caption(f"DMN 모드: `{dmn}`")

        # 활성 카테고리
        st.markdown("**활성 카테고리:**")
        for cat in state['mind']['active_categories'][:8]:
            level = cat['level']
            pct = int(level * 100)
            st.markdown(f"- **{cat['name']}** ({level:.2f})")

        st.divider()

        # 자기 방 객체 또는 NPC
        st.markdown("### 🏠 공간")
        if state['space']['npcs']:
            st.markdown("**누가 있나:**")
            for npc in state['space']['npcs']:
                emoji = {'친구': '👫', '가족': '👨‍👩‍👧', '낯선': '🤔', '지인': '👤'}.get(
                    npc['relationship'], '👤')
                mood_emoji = {'슬픔': '😢', '기쁨': '😊', '아픔': '🤒',
                            '화남': '😠', '보통': '😐'}.get(npc['mood'], '😐')
                st.markdown(f"- {emoji} **{npc['name']}** {mood_emoji} "
                          f"(친밀도 {npc['intimacy']:.2f})")
        else:
            st.markdown("**객체:**")
            for obj in state['space']['objects']:
                st.caption(f"• {obj['name']} → {', '.join(obj['actions'])}")

        # 최근 행동
        if state['recent_actions']:
            st.divider()
            st.markdown("### 📜 최근 행동")
            for ac in state['recent_actions'][-5:]:
                check = '✓' if ac.get('success') else '✗'
                st.caption(f"{check} {ac.get('action')}({ac.get('target') or '-'})")

    # ===== 채팅 =====
    st.divider()
    st.markdown("### 💬 EVE와 대화")

    # 대화 history
    chat_container = st.container(height=300)
    with chat_container:
        for entry in st.session_state.dialogue_log[-10:]:
            with st.chat_message("user"):
                st.write(entry['user'])
            with st.chat_message("assistant"):
                st.write(entry['eve'])
                st.caption(f"_{entry['feeling']} · {entry['pattern']}_")

    # 입력
    user_input = st.chat_input("EVE에게 말을 걸어보세요...")
    if user_input:
        try:
            r = eve.say(user_input)
            st.session_state.dialogue_log.append({
                'user': user_input,
                'eve': r['response'],
                'feeling': r['feeling'],
                'pattern': r['pattern'],
            })
            tracker.snapshot(eve)
            st.rerun()
        except Exception as e:
            st.error(f"대화 에러: {e}")

    # 푸터
    st.caption(
        f"EVE v37.7 · 26 호르몬 · {state['protected']['sa_categories']} 카테고리 · "
        f"tick {state['time']['tick_count']}"
    )


if __name__ == '__main__':
    main()
