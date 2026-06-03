"""
EVE v36 - Proactive (능동적 인터랙션)
======================================

P1 단점 해결: "사용자가 trigger 해야 작동" → "EVE가 먼저 말 검"

설계:
- DMN의 자발 활성을 외부 메시지로 push
- 트리거 조건:
  1. self_intent + 강한 dominant_need (피곤, 외로움 등)
  2. memory_recall 모드 + 강한 일화 활성
  3. 일정 시간 idle + 호르몬 변화 유의
- proactive_message() → str (할 말 없으면 None)
- 빈도 조절: 같은 mode 연속 X (cooldown)

EVE 절대 원칙:
- 능동성 ≠ 침습 (cooldown으로 절제)
- 결정론적 (random X)
- 호르몬/욕구 기반 (자연 발생)

학계:
- Kahneman 2011 - System 1 자발 사고
- Buckner 2008 - DMN 자발적 mind wandering
- Pinker 1994 - inner speech as proto-utterance
"""

from typing import Dict, List, Optional, Tuple, Any
import time as _time


class Proactive:
    """
    EVE가 먼저 말 거는 시스템.
    
    의존:
        eve: EVE_TierAB 인스턴스 (모든 모듈 접근)
    """

    # 모드별 cooldown (초) — 같은 모드 연속 발화 방지
    MODE_COOLDOWN = {
        'self_intent': 60.0,        # 자기 명령은 빈번 X
        'memory_recall': 90.0,
        'mind_wandering': 120.0,
        'self_referential': 180.0,
    }

    # need별 발화 임계 — DMN의 self_intent_threshold(0.5) 보다 약간 높게
    NEED_THRESHOLD = {
        'fatigue': 0.65,
        'tension': 0.65,
        'mental_load': 0.7,
        'warmth': 0.6,         # 외로움 — 비교적 낮은 임계 (대화 trigger)
        'energy': 0.7,
    }

    def __init__(self, eve, idle_min_seconds: float = 30.0):
        """
        Args:
            eve: EVE_TierAB 인스턴스
            idle_min_seconds: 사용자 입력 후 최소 idle 시간 (즉시 push X)
        """
        self.eve = eve
        self.idle_min_seconds = idle_min_seconds

        # 마지막 발화 추적 (모드별)
        self.last_proactive_time: Dict[str, float] = {}
        self.proactive_count = 0
        self.proactive_log: List[Dict[str, Any]] = []

    def should_speak(self) -> Tuple[bool, str]:
        """
        지금 EVE가 먼저 말해야 하는가? 결정론적.
        
        Returns:
            (should, reason) — should=True면 reason에 트리거 종류
        """
        eve = self.eve
        now = eve.time

        # idle 시간 체크 (사용자 입력 후 충분히 지났나)
        idle = now - eve.dmn.last_external_input_time
        if idle < self.idle_min_seconds:
            return False, 'too_soon'

        # 깊은 수면 — 발화 X
        mel = eve.hs.hormones.get('melatonin')
        if mel and mel.level > 0.85:
            return False, 'sleep'

        # 1) self_intent — 강한 욕구
        if eve.dmn.current_mode == 'self_intent':
            need = eve.dmn.current_intent
            if need:
                threshold = self.NEED_THRESHOLD.get(need, 0.7)
                if eve.ds is not None:
                    dom = eve.ds.get_dominant_need()
                    if dom and dom[0] == need and dom[1] >= threshold:
                        # cooldown 체크
                        if self._cooldown_passed('self_intent', now):
                            return True, f'self_intent:{need}'

        # 2) memory_recall + 강한 일화 활성
        if eve.dmn.current_mode == 'memory_recall':
            if hasattr(eve, 'em') and eve.em is not None:
                # 최근 회상된 일화 있나 (recalled_count 기준)
                # cooldown
                if self._cooldown_passed('memory_recall', now):
                    return True, 'memory_recall'

        # 3) mind_wandering + 흥미로운 카테고리 (DA↑)
        if eve.dmn.current_mode == 'mind_wandering':
            da = eve.hs.hormones['dopamine'].level
            if da > 0.7:  # 도파민 높을 때만
                if self._cooldown_passed('mind_wandering', now):
                    return True, 'mind_wandering'

        return False, 'no_trigger'

    def _cooldown_passed(self, mode: str, now: float) -> bool:
        last = self.last_proactive_time.get(mode, -999.0)
        cooldown = self.MODE_COOLDOWN.get(mode, 60.0)
        return (now - last) >= cooldown

    def proactive_message(self) -> Optional[Dict[str, Any]]:
        """
        지금 발화해야 한다면, 자연어 메시지 생성.
        없으면 None.
        
        Returns:
            {'message': str, 'reason': str, 'mode': str, 'time': float}
        """
        should, reason = self.should_speak()
        if not should:
            return None

        eve = self.eve
        message = self._compose_message(reason)
        if not message:
            return None

        # 기록
        now = eve.time
        mode = reason.split(':')[0]
        self.last_proactive_time[mode] = now
        self.proactive_count += 1

        result = {
            'message': message,
            'reason': reason,
            'mode': mode,
            'time': now,
            'tick': eve.tick_count,
        }
        self.proactive_log.append(result)
        if len(self.proactive_log) > 100:
            self.proactive_log = self.proactive_log[-100:]

        return result

    # ============= 발화 후보 fragments =============
    # need → [(seed_category, fragment), ...] — EVE의 SA 활성도가 점수에 반영
    # 같은 need라도 활성 카테고리에 따라 다른 발화
    NEED_FRAGMENTS = {
        'fatigue': [
            ('피곤', '좀 피곤해'),
            ('쉼', '쉬고 싶어'),
            ('잠', '졸려...'),
            ('편안', '좀 멍해'),
        ],
        'tension': [
            ('답답', '답답해'),
            ('힘듦', '좀 힘드네'),
            ('스트레스', '뭔가 답답해'),
        ],
        'mental_load': [
            ('복잡', '머리 복잡해'),
            ('생각', '생각이 많아'),
            ('정리', '정리가 안 돼'),
        ],
        'warmth': [
            ('대화', '뭐해?'),
            ('함께', '있어?'),
            ('친구', '같이 있고 싶어'),
            ('보고싶음', '보고싶어'),
        ],
        'energy': [
            ('재밌음', '뭐 재밌는 거 없을까'),
            ('호기심', '궁금한 게 있어'),
            ('심심', '심심해'),
        ],
    }

    def _compose_message(self, reason: str) -> Optional[str]:
        """
        v37.9 — 강제 룰 매핑 폐기.
        후보 fragment 생성 → SA 활성도/호르몬 점수 → 선택.
        같은 need라도 EVE의 현재 카테고리 활성에 따라 다른 발화.
        """
        eve = self.eve

        # 호명 결정 — 사용자(김민석) 진짜 친구이고 OT 높을 때만
        use_name = self._should_address_user()
        name_prefix = '민석아, ' if use_name else ''

        # ── self_intent 종류별 — 후보 점수 선택
        if reason.startswith('self_intent:'):
            need = reason.split(':', 1)[1]
            return self._compose_from_need(need, name_prefix)

        # ── memory_recall — EM에서 가장 활성 강한 카테고리
        if reason == 'memory_recall':
            return self._compose_from_memory(name_prefix)

        # ── mind_wandering — DMN 자발 활성 카테고리
        if reason == 'mind_wandering':
            return self._compose_from_wandering(name_prefix)

        return None

    def _should_address_user(self) -> bool:
        """김민석 호명 여부 — 진짜 친구 + OT 충분 + 가까이"""
        eve = self.eve
        if not hasattr(eve, 'user_presence'):
            return False
        # 진짜 친구로 등록됐고
        if not eve.user_presence.first_visit_done:
            return False
        # OT 충분
        ot = eve.hs.hormones['oxytocin'].level
        if ot < 0.45:
            return False
        return True

    def _compose_from_need(self, need: str, name_prefix: str) -> Optional[str]:
        """need + SA 활성도 → 가장 적합한 fragment"""
        eve = self.eve
        candidates = self.NEED_FRAGMENTS.get(need)
        if not candidates:
            return None

        # 각 fragment 점수 = SA 활성도 + base hint
        scored = []
        for cat, text in candidates:
            sa_level = eve.sa.get_activation_level(cat) if hasattr(eve.sa, 'get_activation_level') else 0
            # base 점수 (모든 후보에 + 카테고리 평균 가중치)
            score = sa_level + 0.10
            scored.append((score, cat, text))
        # 결정론 정렬: 점수 ↓, 카테고리 알파벳
        scored.sort(key=lambda x: (-x[0], x[1]))

        best_score, best_cat, best_text = scored[0]
        return f"{name_prefix}{best_text}"

    def _compose_from_memory(self, name_prefix: str) -> Optional[str]:
        """EM 일화 + SA 활성 → 회상 발화"""
        eve = self.eve

        # 가장 활성 강한 카테고리 (메모리 관련)
        if not hasattr(eve, 'em') or eve.em is None or not eve.em.episodes:
            return f"{name_prefix}뭔가 기억나려고 하는데..."

        # 활성 카테고리 top 1 (의미 있는 것)
        if hasattr(eve.sa, 'activations'):
            items = list(eve.sa.activations.items())
            # 일반 카테고리 (메타 카테고리 제외)
            meta = {'학습_모드', '시뮬레이션', '내방', '편안', '대화', '함께'}
            items = [(c, v) for c, v in items if c not in meta and v > 0.2]
            items.sort(key=lambda x: (-x[1], x[0]))
            if items:
                top_cat = items[0][0]
                return f"{name_prefix}{top_cat} 생각났어"

        return f"{name_prefix}뭔가 떠올랐어"

    def _compose_from_wandering(self, name_prefix: str) -> Optional[str]:
        """DMN wandering 최근 카테고리 → 발화"""
        eve = self.eve

        if not hasattr(eve, 'dmn') or not eve.dmn.wandering_history:
            return None

        # 최근 wandering 카테고리 중 SA 활성 가장 강한 것
        recent_cats = [c for _, c, _ in eve.dmn.wandering_history[-5:]]
        if not recent_cats:
            return None

        # 점수 = SA 활성도
        scored = []
        for c in set(recent_cats):
            level = eve.sa.get_activation_level(c) if hasattr(eve.sa, 'get_activation_level') else 0
            scored.append((level, c))
        scored.sort(key=lambda x: (-x[0], x[1]))

        if scored:
            top_cat = scored[0][1]
            return f"{name_prefix}{top_cat}..."

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            'proactive_count': self.proactive_count,
            'recent_messages': self.proactive_log[-3:],
            'last_proactive_time': dict(self.last_proactive_time),
        }
