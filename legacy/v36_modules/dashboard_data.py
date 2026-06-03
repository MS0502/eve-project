"""
EVE v37.7 - Dashboard Data Layer
================================

EVE 인스턴스의 현재 상태를 시각화 친화적 dict로 변환.
UI 독립 — Streamlit/Gradio/Flask 어디서든 사용 가능.

핵심 시각 요소:
1. 호르몬 → 색상 + 강도 (8 주요 + 18 기타)
2. 자기 방 공간 — 객체 위치 + EVE 위치
3. 활성 카테고리 — focus + 주변 (별자리)
4. 일과 — 최근 행동 시퀀스 (timeline)
5. 속마음 — DMN 모드 + 자발 활성 → 자연어
"""

from typing import Dict, List, Tuple, Any, Optional
import math


# ============================================================
# 색상 매핑 — 호르몬에 의미부여
# ============================================================

# 주요 호르몬 → RGB (0-255)
HORMONE_COLORS = {
    'dopamine':       (255, 200, 60),   # 따뜻한 노랑 (보상)
    'oxytocin':       (255, 130, 180),  # 핑크 (사랑)
    'serotonin':      (140, 220, 140),  # 연두 (안정)
    'cortisol':       (80, 130, 200),   # 차가운 파랑 (스트레스)
    'norepinephrine': (255, 80, 80),    # 빨강 (각성)
    'melatonin':      (40, 40, 90),     # 짙은 어둠 (밤)
    'endorphin':      (255, 220, 100),  # 황금 (쾌감)
    'gaba':           (180, 180, 220),  # 부드러운 보라 (이완)
    'glutamate':      (255, 150, 80),   # 주황 (흥분)
    'acetylcholine':  (200, 230, 230),  # 옅은 청록 (학습)
    'adenosine':      (90, 90, 110),    # 회색 (피로)
    'histamine':      (200, 100, 50),   # 갈빛 (경계)
    'bdnf':           (100, 200, 150),  # 푸른 녹 (성장)
}
DEFAULT_COLOR = (150, 150, 150)


def hormone_color(name: str) -> Tuple[int, int, int]:
    return HORMONE_COLORS.get(name, DEFAULT_COLOR)


def blend_colors(weighted: List[Tuple[Tuple[int, int, int], float]]) -> Tuple[int, int, int]:
    """가중 색상 평균 — EVE 종합 색상"""
    if not weighted:
        return DEFAULT_COLOR
    total_w = sum(w for _, w in weighted)
    if total_w < 1e-6:
        return DEFAULT_COLOR
    r = sum(c[0] * w for c, w in weighted) / total_w
    g = sum(c[1] * w for c, w in weighted) / total_w
    b = sum(c[2] * w for c, w in weighted) / total_w
    return (int(r), int(g), int(b))


# ============================================================
# 메인 — EVE 상태 → 대시보드 dict
# ============================================================

def get_eve_state(eve) -> Dict[str, Any]:
    """
    EVE 인스턴스 → 대시보드용 dict.
    
    Returns:
        {
            'time': {...},          # sim_hour, real_time, tick_count
            'feeling': str,
            'mood': dict,           # valence/arousal/dominance
            'hormones': {...},      # 26개 + 색상
            'eve_color': (r,g,b),   # 종합 색상
            'space': {...},         # 자기 방/시뮬 환경
            'mind': {...},          # focus, DMN, 속마음
            'thoughts': [...],      # 최근 자발 활성
            'recent_actions': [...],
            'recent_dialogue': [...],
        }
    """
    return {
        'time': _time_state(eve),
        'feeling': eve.nl.feeling_to_text() if hasattr(eve, 'nl') else None,
        'mood': eve.hs.compute_mood() if hasattr(eve.hs, 'compute_mood') else {},
        'hormones': _hormones_state(eve),
        'eve_color': _compute_eve_color(eve),
        'space': _space_state(eve),
        'mind': _mind_state(eve),
        'thoughts': _thoughts_state(eve),
        'recent_actions': _recent_actions(eve),
        'recent_dialogue': _recent_dialogue(eve),
        'protected': _protected_state(eve),
    }


def _time_state(eve) -> Dict[str, Any]:
    sim_hour = getattr(eve.hs, 'sim_hour', 12.0)
    h_int = int(sim_hour)
    m_int = int((sim_hour - h_int) * 60)
    # 시간대
    if 0 <= sim_hour < 6:
        period = '새벽'
    elif sim_hour < 12:
        period = '오전'
    elif sim_hour < 18:
        period = '오후'
    elif sim_hour < 22:
        period = '저녁'
    else:
        period = '밤'
    return {
        'sim_hour': sim_hour,
        'sim_hour_str': f"{h_int:02d}:{m_int:02d}",
        'period': period,
        'real_time': getattr(eve, 'time', 0.0),
        'tick_count': getattr(eve, 'tick_count', 0),
    }


def _hormones_state(eve) -> Dict[str, Dict[str, Any]]:
    """26 호르몬 — 레벨 + 색상"""
    out = {}
    for name, h in eve.hs.hormones.items():
        out[name] = {
            'level': float(h.level),
            'baseline': float(h.baseline),
            'color': hormone_color(name),
            'is_elevated': h.level > h.baseline + 0.10,
            'is_depleted': h.level < h.baseline - 0.10,
        }
    return out


def _compute_eve_color(eve) -> Tuple[int, int, int]:
    """
    EVE 종합 색상 — 주요 호르몬 가중 블렌드.
    DA/OT/SER (긍정) 따뜻, CORT/NE (스트레스) 차가움, MEL (밤) 어둠.
    """
    weighted = []
    primary_hormones = ['dopamine', 'oxytocin', 'serotonin', 'cortisol',
                        'norepinephrine', 'melatonin', 'endorphin', 'gaba']
    for name in primary_hormones:
        h = eve.hs.hormones.get(name)
        if h is not None:
            weight = h.level
            if weight > 0.05:
                weighted.append((hormone_color(name), weight))
    return blend_colors(weighted)


def _space_state(eve) -> Dict[str, Any]:
    """현재 환경 — 자기 방 또는 시뮬"""
    if (not hasattr(eve, '_env_adapter')
        or eve._env_adapter is None
        or eve._env_adapter.current_env is None):
        return {
            'in_env': False,
            'env_name': None,
            'is_simulation': None,
        }
    env = eve._env_adapter.current_env
    objs = []
    for obj in env.objects.values():
        objs.append({
            'name': obj.name,
            'location': obj.location,
            'actions': sorted(obj.available_actions),
        })
    npcs = []
    if hasattr(env, 'npcs'):
        for npc in env.npcs.values():
            npcs.append({
                'name': npc.name,
                'relationship': npc.relationship,
                'mood': npc.mood,
                'intimacy': float(npc.intimacy),
                'location': npc.location,
            })
    return {
        'in_env': True,
        'env_name': env.name,
        'is_simulation': getattr(env, 'is_simulation', True),
        'current_location': env.current_location,
        'objects': objs,
        'npcs': npcs,
        'env_state': dict(env.state),
    }


def _mind_state(eve) -> Dict[str, Any]:
    """현재 머릿속 — focus, DMN, 활성 카테고리"""
    # focus
    focus = eve.wm.get_focus() if hasattr(eve, 'wm') else None

    # DMN
    dmn_mode = None
    dmn_intent = None
    if hasattr(eve, 'dmn'):
        dmn_mode = getattr(eve.dmn, 'current_mode', None)
        dmn_intent = getattr(eve.dmn, 'current_intent', None)

    # 활성 카테고리 top 8
    active = []
    if hasattr(eve, 'sa'):
        items = list(eve.sa.activations.items())
        items.sort(key=lambda x: -x[1])
        active = [{'name': c, 'level': float(v)} for c, v in items[:12]]

    # 속마음 (자발 활성 자연어)
    inner_voice = None
    if hasattr(eve, 'nl') and hasattr(eve, 'dmn'):
        try:
            inner_voice = eve.nl.inner_voice(eve.dmn)
        except Exception:
            inner_voice = None

    return {
        'focus': focus,
        'dmn_mode': dmn_mode,
        'dmn_intent': dmn_intent,
        'active_categories': active,
        'inner_voice': inner_voice,
        'wm_size': len(eve.wm.slots) if hasattr(eve.wm, 'slots') else 0,
    }


def _thoughts_state(eve) -> List[Dict]:
    """최근 자발 활성 (DMN wandering)"""
    out = []
    if hasattr(eve, 'dmn') and hasattr(eve.dmn, 'wandering_history'):
        for time_, cat, mode in eve.dmn.wandering_history[-10:]:
            out.append({
                'time': float(time_),
                'category': cat,
                'mode': mode,
            })
    return out


def _recent_actions(eve) -> List[Dict]:
    """최근 환경 행동"""
    out = []
    if (hasattr(eve, '_env_adapter') and eve._env_adapter is not None
        and eve._env_adapter.current_env is not None):
        env = eve._env_adapter.current_env
        for ac in list(env.action_history)[-15:]:
            out.append(ac)
    return out


def _recent_dialogue(eve) -> List[Dict]:
    """최근 대화 history"""
    out = []
    if hasattr(eve, 'dialogue_history'):
        for d in eve.dialogue_history[-10:]:
            out.append({
                'time': d.get('time'),
                'input': d.get('input'),
                'response': d.get('response'),
                'feeling': d.get('feeling'),
            })
    return out


def _protected_state(eve) -> Dict:
    """보호 카테고리 등 메타"""
    return {
        'protected_count': len(eve.protected_categories) if hasattr(eve, 'protected_categories') else 0,
        'beliefs_count': len(eve.nl.beliefs) if hasattr(eve, 'nl') and hasattr(eve.nl, 'beliefs') else 0,
        'sa_categories': len(eve.sa.neighbors) if hasattr(eve, 'sa') else 0,
    }


# ============================================================
# 추상 EVE 시각화 좌표 (별자리)
# ============================================================

def get_eve_visualization_data(eve) -> Dict[str, Any]:
    """
    추상 EVE 시각화용 좌표/색/크기.
    중심 = EVE 본체 (호르몬 종합 색)
    주변 = 활성 카테고리 (자발 활성 = 점, 강도 = 크기)
    
    Returns:
        {
            'center': {'x': 0, 'y': 0, 'color': (r,g,b), 'size': float},
            'nodes': [{'name','x','y','color','size','is_focus'}, ...],
            'connections': [...],  # focus와 활성 카테고리 사이 선
        }
    """
    state = get_eve_state(eve)

    # 중심
    eve_color = state['eve_color']
    mood = state['mood']
    arousal = mood.get('arousal', 0.5)
    center_size = 30 + 30 * arousal  # 각성 ↑ → 큼

    # 활성 카테고리 → 별자리 노드
    active = state['mind']['active_categories']
    focus = state['mind']['focus']

    nodes = []
    n = len(active)
    if n > 0:
        for i, item in enumerate(active):
            # 각도 균등 분포 (결정론)
            angle = (i / n) * 2 * math.pi
            radius = 100 + 50 * (1 - item['level'])  # 활성 강할수록 가까이
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            size = 8 + 25 * item['level']
            is_focus = (item['name'] == focus)
            # 색 — 카테고리 그룹에 따라
            color = _category_color(item['name'])
            nodes.append({
                'name': item['name'],
                'x': x,
                'y': y,
                'color': color,
                'size': size,
                'is_focus': is_focus,
                'level': item['level'],
            })

    # 연결선 — focus와 활성 카테고리들 사이
    connections = []
    if focus:
        for nd in nodes:
            if not nd['is_focus']:
                connections.append({
                    'from': focus,
                    'to': nd['name'],
                    'strength': nd['level'],
                })

    return {
        'center': {
            'x': 0, 'y': 0,
            'color': eve_color,
            'size': center_size,
            'feeling': state['feeling'],
            'period': state['time']['period'],
        },
        'nodes': nodes,
        'connections': connections,
        'time_label': state['time']['sim_hour_str'],
    }


# 카테고리 → 그룹별 색
CATEGORY_GROUP_COLORS = {
    'reward':   (255, 200, 60),   # 황
    'social':   (255, 130, 180),  # 핑크
    'threat':   (200, 50, 50),    # 빨강
    'attention': (180, 220, 255), # 옅은 파랑
    'memory':   (200, 180, 220),  # 연보라
    'rest':     (90, 90, 110),    # 회색
    'curiosity': (140, 220, 140), # 연두
    'aversion': (100, 80, 80),    # 갈색
}


def _category_color(category: str) -> Tuple[int, int, int]:
    """카테고리 → 색 (그룹 기반)"""
    try:
        from spreading_activation import SpreadingActivation
        group = SpreadingActivation.CATEGORY_GROUPS.get(category)
        if group:
            return CATEGORY_GROUP_COLORS.get(group, (180, 180, 180))
    except Exception:
        pass
    return (180, 180, 180)


# ============================================================
# 호르몬 추세 (시간 따라 변화 추적용)
# ============================================================

class HormoneTracker:
    """호르몬 시계열 추적 — 대시보드 차트용"""
    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.history: List[Dict[str, float]] = []  # [{name: level, ...}, ...]
        self.timestamps: List[float] = []

    def snapshot(self, eve):
        sample = {name: float(h.level) for name, h in eve.hs.hormones.items()}
        self.history.append(sample)
        self.timestamps.append(getattr(eve.hs, 'sim_hour', 0.0))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]

    def series(self, hormone_name: str) -> Tuple[List[float], List[float]]:
        ts = list(self.timestamps)
        vals = [s.get(hormone_name, 0.0) for s in self.history]
        return ts, vals

    def all_series(self, names: List[str] = None) -> Dict[str, Tuple[List[float], List[float]]]:
        if names is None:
            names = ['dopamine', 'oxytocin', 'serotonin', 'cortisol',
                    'norepinephrine', 'melatonin']
        return {n: self.series(n) for n in names}
