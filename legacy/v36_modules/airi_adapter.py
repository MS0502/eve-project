"""
EVE v38 — AIRI Adapter
======================

EVE 백엔드 상태 → AIRI 프론트엔드 메시지 프로토콜 변환.

송출 데이터 종류:
1. character_state — 표정/감정/색상 (호르몬 → 매핑)
2. speech — EVE 발화 (응답 + 속마음)
3. environment — 위치/시간 (자기 방 / 외출 중)
4. mind_state — focus + 활성 카테고리 (선택적 표시)
5. relationship — 김민석 친밀도

수신 데이터:
- user_message — 사용자 채팅
- user_action — UI 인터랙션 (보내기, 환경 전환 등)

표정 매핑 (호르몬 → AIRI 표정 코드):
- 도파민↑ + 옥시토신↑ → "happy"
- 코르티솔↑ → "stressed"
- 멜라토닌↑ → "sleepy"
- 노르에피네프린↑ → "alert"  
- GABA↑ + 세로토닌↑ → "calm"
- 옥시토신↑ + 진짜친구방문 → "loving"
- 평형 → "neutral"
"""

from typing import Dict, Any, Optional, List
import time as time_module


# ============================================================
# 호르몬 → 표정/감정 매핑
# ============================================================

# AIRI/Live2D/VRM 표준 표정 카테고리
EXPRESSION_CATEGORIES = [
    'neutral', 'happy', 'loving', 'calm', 'sleepy',
    'sad', 'stressed', 'alert', 'curious', 'surprised',
]

# ★ v38.2 — 일반 표정 → VRM BlendShape 매핑
# VRoid Studio 표준 표정 7종 (Neutral/Joy/Fun/Sorrow/Angry/Surprised + Extra)
VRM_BLENDSHAPE_MAP = {
    'neutral':   ('Neutral',   1.0),
    'happy':     ('Joy',       1.0),
    'loving':    ('Fun',       1.0),
    'calm':      ('Neutral',   0.6),
    'sleepy':    ('Sorrow',    0.7),
    'sad':       ('Sorrow',    1.0),
    'stressed':  ('Angry',     0.85),
    'alert':     ('Surprised', 0.6),
    'curious':   ('Fun',       0.5),
    'surprised': ('Surprised', 1.0),
}


def to_vrm_blendshape(expression: str, intensity: float = 1.0) -> Dict[str, Any]:
    """일반 표정명 + 강도 → VRM BlendShape (Joy/Fun/Sorrow/Angry/Surprised/Neutral)"""
    name, base_weight = VRM_BLENDSHAPE_MAP.get(expression, ('Neutral', 1.0))
    final_weight = float(base_weight) * float(intensity)
    return {
        'name': name,
        'weight': max(0.0, min(1.0, final_weight)),
    }


# 호르몬 → 표정 점수 함수
# 각 표정은 호르몬 조합으로 점수 산출. 가장 높은 점수가 현재 표정.

def compute_expression(hormones: Dict[str, float]) -> Dict[str, Any]:
    """
    26 호르몬 → AIRI 표정 매핑.
    
    Returns: {'expression': str, 'intensity': 0-1, 'all_scores': dict}
    """
    da = hormones.get('dopamine', 0.3)
    ot = hormones.get('oxytocin', 0.3)
    cort = hormones.get('cortisol', 0.3)
    ne = hormones.get('norepinephrine', 0.3)
    mel = hormones.get('melatonin', 0.1)
    ser = hormones.get('serotonin', 0.4)
    end = hormones.get('endorphin', 0.3)
    gaba = hormones.get('gaba', 0.4)
    ade = hormones.get('adenosine', 0.2)
    glut = hormones.get('glutamate', 0.4)
    his = hormones.get('histamine', 0.3)

    scores = {}

    # neutral — 모든 호르몬이 평형 근처 (엄격 — 큰 편차 있으면 X)
    deviations = [abs(h - 0.4) for h in [da, ot, cort, ne, mel, ser]]
    max_dev = max(deviations)
    avg_dev = sum(deviations) / len(deviations)
    # 큰 편차 페널티 (가장 큰 편차가 0.3 넘으면 neutral 거의 0)
    scores['neutral'] = max(0, 0.5 - avg_dev * 1.2 - max(0, max_dev - 0.3) * 1.5)

    # happy — DA + OT 둘 다 높음
    scores['happy'] = (da - 0.4) * 0.8 + (ot - 0.3) * 0.6 + (end - 0.3) * 0.4
    scores['happy'] = max(0, scores['happy'])

    # loving — OT 매우 높음 (사용자 방문 직후) — happy보다 강하게
    scores['loving'] = max(0, (ot - 0.45) * 1.6 + (end - 0.4) * 0.5
                           - max(0, da - 0.65) * 0.3)  # DA 너무 높으면 happy

    # calm — 세로토닌 + GABA, cort 낮음 (강화)
    scores['calm'] = max(0, (ser - 0.35) * 1.2 + (gaba - 0.4) * 0.8
                         + max(0, 0.3 - cort) * 0.5)

    # sleepy — 멜라토닌 + 아데노신
    scores['sleepy'] = max(0, mel * 0.9 + (ade - 0.3) * 0.5)

    # sad — 세로토닌 낮음 + 코르티솔 약간
    scores['sad'] = max(0, (0.3 - ser) * 0.8 + (cort - 0.4) * 0.3)

    # stressed — 코르티솔 ↑
    scores['stressed'] = max(0, (cort - 0.5) * 1.0 + (ne - 0.5) * 0.5)

    # alert — 노르에피네프린 + 히스타민
    scores['alert'] = max(0, (ne - 0.4) * 0.7 + (his - 0.3) * 0.5)

    # curious — DA + NE 적당, 멜라토닌 낮음, 글루타메이트 높음
    scores['curious'] = max(0, (da - 0.3) * 0.4 + (ne - 0.3) * 0.4
                            + (glut - 0.4) * 0.4 - mel * 0.3)

    # surprised — NE 급증 (단기, baseline 차이 크면)
    scores['surprised'] = max(0, (ne - 0.6) * 1.5)

    # 가장 높은 점수
    best_expr = max(scores.items(), key=lambda x: x[1])
    expr_name, expr_score = best_expr
    intensity = min(1.0, expr_score)

    return {
        'expression': expr_name,
        'intensity': float(intensity),
        'all_scores': {k: round(float(v), 3) for k, v in scores.items()},
    }


# ============================================================
# EVE 색상 (호르몬 가중 RGB)
# ============================================================
def compute_eve_color(hormones: Dict[str, float]) -> Dict[str, int]:
    """EVE 종합 색상 (이미 dashboard_data에 있지만 AIRI용 단순화)"""
    HORMONE_RGB = {
        'dopamine':       (255, 200, 60),
        'oxytocin':       (255, 130, 180),
        'serotonin':      (140, 220, 140),
        'cortisol':       (80, 130, 200),
        'norepinephrine': (255, 80, 80),
        'melatonin':      (40, 40, 90),
        'endorphin':      (255, 220, 100),
        'gaba':           (180, 180, 220),
    }
    weighted = []
    for name, rgb in HORMONE_RGB.items():
        level = hormones.get(name, 0)
        if level > 0.05:
            weighted.append((rgb, level))

    if not weighted:
        return {'r': 150, 'g': 150, 'b': 150}
    total_w = sum(w for _, w in weighted)
    r = sum(c[0] * w for c, w in weighted) / total_w
    g = sum(c[1] * w for c, w in weighted) / total_w
    b = sum(c[2] * w for c, w in weighted) / total_w
    return {'r': int(r), 'g': int(g), 'b': int(b)}


# ============================================================
# 메인 어댑터
# ============================================================

class AIRIAdapter:
    """
    EVE 인스턴스 ↔ AIRI 메시지 변환.
    
    EVE → AIRI: state_message (주기적 push)
    AIRI → EVE: user_message / user_action (수신)
    """

    PROTOCOL_VERSION = "v38.1"

    def __init__(self, eve):
        self.eve = eve
        self.last_state_hash = None  # state diff용
        self.message_count = 0

    # ============= EVE → AIRI 송출 =============
    def build_state_message(self) -> Dict[str, Any]:
        """
        현재 EVE 상태 → AIRI 메시지.
        AIRI 클라이언트가 받아 캐릭터 표정/색/말풍선 갱신.
        """
        eve = self.eve

        # 호르몬
        hormones = {name: float(h.level) for name, h in eve.hs.hormones.items()}

        # 표정
        expr_info = compute_expression(hormones)

        # 색상
        color = compute_eve_color(hormones)

        # 통합 자아 (안전하게)
        try:
            snap = eve.self_view.snapshot()
        except Exception:
            snap = None

        # 환경
        location = None
        is_simulation = False
        if hasattr(eve, '_env_adapter') and eve._env_adapter is not None:
            env = eve._env_adapter.current_env
            if env is not None:
                location = env.name
                is_simulation = getattr(env, 'is_simulation', False)

        # 시간/시간대
        sim_hour = getattr(eve.hs, 'sim_hour', 12.0)
        period = self._sim_hour_period(sim_hour)

        # 친밀도
        intimacy = 0.0
        visit_count = 0
        if hasattr(eve, 'user_presence'):
            intimacy = eve.user_presence.get_intimacy()
            visit_count = eve.user_presence.visit_count

        # feeling
        feeling = ""
        try:
            feeling = eve.nl.feeling_to_text() or ""
        except Exception:
            pass

        # 속마음
        inner_voice = None
        try:
            inner_voice = eve.nl.inner_voice(eve.dmn)
        except Exception:
            pass

        # 최근 행동
        last_action = None
        last_target = None
        if hasattr(eve, '_env_adapter') and eve._env_adapter is not None and eve._env_adapter.current_env is not None:
            history = getattr(eve._env_adapter.current_env, 'action_history', [])
            if history:
                last = history[-1]
                if isinstance(last, dict):
                    last_action = last.get('action')
                    last_target = last.get('target')

        # focus
        focus = None
        if hasattr(eve, 'wm'):
            focus = eve.wm.get_focus()

        # active 카테고리 top 5
        active_cats = []
        if hasattr(eve, 'sa'):
            items = list(eve.sa.activations.items())
            items.sort(key=lambda x: -x[1])
            active_cats = [{'name': c, 'level': round(float(v), 3)} for c, v in items[:5]]

        msg = {
            'type': 'state',
            'protocol': self.PROTOCOL_VERSION,
            'timestamp': time_module.time(),
            'eve_time': float(getattr(eve, 'time', 0)),
            'sim_hour': float(sim_hour),
            'period': period,
            
            # 캐릭터 시각 표현
            'expression': expr_info['expression'],
            'expression_intensity': expr_info['intensity'],
            'vrm_blendshape': to_vrm_blendshape(
                expr_info['expression'], expr_info['intensity']
            ),
            'color': color,
            
            # 감정/feeling
            'feeling': feeling,
            'mood': self._compute_mood_label(hormones),
            
            # 환경
            'location': location,
            'is_in_simulation': is_simulation,
            'last_action': last_action,
            'last_action_target': last_target,
            
            # 정신
            'focus': focus,
            'active_categories': active_cats,
            'inner_voice': inner_voice,
            'dmn_mode': getattr(eve.dmn, 'current_mode', None) if hasattr(eve, 'dmn') else None,
            
            # 사용자 관계
            'user_intimacy': float(intimacy),
            'user_visit_count': int(visit_count),
            
            # 호르몬 (UI 표시용)
            'hormones': {k: round(v, 3) for k, v in hormones.items()},
            'expression_scores': expr_info['all_scores'],
        }

        self.message_count += 1
        return msg

    def build_speech_message(self, text: str, is_response: bool = True,
                            feeling: str = "", pattern: str = "") -> Dict[str, Any]:
        """EVE 발화 → AIRI 메시지 (말풍선 + TTS)"""
        return {
            'type': 'speech',
            'protocol': self.PROTOCOL_VERSION,
            'timestamp': time_module.time(),
            'text': text,
            'is_response': is_response,  # True=사용자 응답, False=자발 발화
            'feeling': feeling,
            'pattern': pattern,
        }

    def build_proactive_message(self, message: str, reason: str = "") -> Dict[str, Any]:
        """EVE 자발 발화 (proactive)"""
        return {
            'type': 'proactive',
            'protocol': self.PROTOCOL_VERSION,
            'timestamp': time_module.time(),
            'text': message,
            'reason': reason,
        }

    # ============= AIRI → EVE 처리 =============
    def handle_user_message(self, text: str) -> Dict[str, Any]:
        """사용자 채팅 메시지 → EVE.say() → 응답"""
        eve = self.eve
        try:
            r = eve.say(text)
            return {
                'success': True,
                'response': r['response'],
                'feeling': r.get('feeling', ''),
                'pattern': r.get('pattern', ''),
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }

    def handle_user_action(self, action: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """UI 액션 (환경 전환, 시간 진행 등)"""
        eve = self.eve
        params = params or {}

        try:
            if action == 'tick':
                eve.tick(dt=params.get('dt', 0.5))
                return {'success': True}

            if action == 'live_hours':
                hours = params.get('hours', 1.0)
                actions = eve.live_in_room(hours=hours,
                                          sim_hour_per_step=params.get('step', 0.25))
                return {'success': True, 'actions_count': len(actions)}

            if action == 'enter_room':
                eve.enter_room()
                return {'success': True, 'env': '내방'}

            if action == 'enter_environment':
                env_name = params.get('env_name', '거실')
                if env_name == '거실':
                    from social_env import make_social_living_room
                    eve.enter_environment(make_social_living_room())
                elif env_name == '주방':
                    from symbolic_env import make_kitchen_env
                    eve.enter_environment(make_kitchen_env())
                else:
                    return {'success': False, 'error': f'알 수 없는 환경: {env_name}'}
                return {'success': True, 'env': env_name}

            if action == 'save':
                path = params.get('path', '/tmp/eve_save.ckpt')
                info = eve.save(path)
                return {'success': True, 'path': info.get('path', path)}

            if action == 'load':
                path = params.get('path')
                if not path:
                    return {'success': False, 'error': 'path 필요'}
                report = eve.load(path)
                return {'success': True, 'restored': report.get('restored', [])}

            return {'success': False, 'error': f'알 수 없는 액션: {action}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============= 헬퍼 =============
    @staticmethod
    def _sim_hour_period(h: float) -> str:
        h = h % 24
        if h < 6: return '새벽'
        if h < 12: return '오전'
        if h < 18: return '오후'
        if h < 22: return '저녁'
        return '밤'

    @staticmethod
    def _compute_mood_label(hormones: Dict[str, float]) -> str:
        """기분 한 단어 레이블"""
        da = hormones.get('dopamine', 0.3)
        ot = hormones.get('oxytocin', 0.3)
        cort = hormones.get('cortisol', 0.3)
        mel = hormones.get('melatonin', 0.1)

        if mel > 0.6:
            return 'sleepy'
        if cort > 0.65:
            return 'stressed'
        if da > 0.6 and ot > 0.5:
            return 'happy'
        if ot > 0.6:
            return 'warm'
        if da > 0.55:
            return 'energetic'
        if cort < 0.3 and da > 0.4:
            return 'calm'
        return 'neutral'
