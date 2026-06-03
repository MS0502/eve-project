"""
EVE v35 - F1: NormInternalization (규범 내재화)
================================================

목표: 사용자(김민석)의 반복 명령/취향/요청 → EVE 자기 규범으로 내재화.

기존 한계:
- SelfDoubt: 명령 거절 평가만 (즉시 판단, 학습 X)
- Agency: 자기 평가 (행동 후)
- Belief: 사실 등록 (영속 메모리)
- 그러나 *반복 패턴 인식 + 자발적 적용*은 없음

F1 해결:
- 명령/요청을 norm으로 등록 (action + condition + strength)
- 반복 발화 → strength↑ (familiarity), 충돌 → 약화
- DMN/Goal에서 norm 자발 적용 (사용자 없을 때도)
- 충돌 검사 (예: "쉬어" vs "공부해" — 마지막 우선)

학계:
- Niazi & Hutter 2026 — Norm Internalization in AGI
- Sripada & Stich 2007 — A Framework for the Psychology of Norms
- Cialdini & Trost 1998 — Social Influence
- Bicchieri 2006 — The Grammar of Society

핵심 데이터:
- Norm dataclass: id, action, condition, agent, strength, evidence_count
- norms: Dict[id, Norm]
- agent_index: agent → norm_ids

EVE 절대 원칙:
- 결정론적 (sorted, no random)
- 카테고리 = 의미 단위 유지
- read-only on SA/CG/HG (add-only on self.norms)
- 호르몬 변조 (반복 학습 시 ACh↑, 내재화 강화 시 OT↑)

핵심 함수:
1. observe_command(text, parsed)        → 명령 발화 관찰 → norm 후보 추출
2. internalize(action, agent, ...)      → norm 등록/강화
3. get_active_norms(agent, threshold)   → 활성 규범 (strength ≥ threshold)
4. apply_norm_to_dmn(dmn)               → DMN 자발 활성에 norm 영향
5. detect_conflict(new_action, ctx)     → 충돌 감지
6. weaken(norm_id, amount)              → 약화 (불응시)
7. tick(dt)                             → 시간 감쇠 + 반감기
8. introspect_norms(agent)              → "민석이 자주 시키는 것"

Author: 김민석 + Claude v35
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


@dataclass
class Norm:
    """내재화된 규범 (사용자 명령/취향)"""
    id: str
    action: str                            # 행동 카테고리 (예: '휴식', '공부')
    condition: Optional[str] = None        # 조건 (예: '피곤할때')
    agent: str = '민석'                    # 누구의 규범 (default 사용자)
    strength: float = 0.5                  # 내재화 정도 (0-1)
    evidence_count: int = 1                # 반복 횟수
    last_observed: float = 0.0             # 마지막 관찰 시간
    last_applied: float = 0.0              # 마지막 자발 적용
    rejected_count: int = 0                # 사용자가 거절한 횟수 (반례)
    created_at: float = 0.0


class NormInternalization:
    """
    사용자 명령/취향 내재화 모듈.

    의존:
        spreading_activation: SA (필수)
        hormone_system: HS (선택, ACh/OT 변조)
        natural_lang: NL (선택, parse 위임)
        dmn: DMN (선택, 자발 적용)
    """

    # 명령 동사 → action 카테고리 매핑 (한국어 명령형 인식)
    COMMAND_VERB_PATTERNS = {
        '쉬': '휴식',
        '잠': '잠',
        '잘': '잠',
        '공부': '공부',
        '먹': '식사',
        '운동': '운동',
        '얘기': '대화',
        '말': '대화',
        '대답': '대화',
        '들어': '경청',
        '읽': '독서',
        '써': '쓰기',
        '도와': '도움',
        '기억': '기억',
        '잊': '망각',
    }

    # 명령 어미 (강도)
    IMPERATIVE_ENDINGS = {
        '해라': 1.0, '하라': 1.0,           # 강한 명령
        '해': 0.7, '해줘': 0.7, '하자': 0.6, # 보통
        '하면': 0.5, '하길': 0.5,           # 부드러운 권유
        '해야': 0.8, '해야해': 0.8,
    }

    def __init__(self,
                 spreading_activation,
                 hormone_system=None,
                 natural_lang=None,
                 dmn=None,
                 default_agent: str = '민석',
                 internalize_threshold: int = 2,  # 2번 반복하면 내재화
                 strength_decay_per_day: float = 0.05,
                 max_norms: int = 200):
        self.sa = spreading_activation
        self.hs = hormone_system
        self.nl = natural_lang
        self.dmn = dmn
        self.default_agent = default_agent
        self.internalize_threshold = int(internalize_threshold)
        self.strength_decay_per_day = float(strength_decay_per_day)
        self.max_norms = int(max_norms)

        # 핵심 상태
        self.norms: Dict[str, Norm] = {}
        self.agent_index: Dict[str, Set[str]] = defaultdict(set)
        self.action_index: Dict[str, Set[str]] = defaultdict(set)

        # 시간/통계
        self.time = 0.0
        self.tick_count = 0
        self.observed_count = 0
        self.internalized_count = 0
        self.applied_count = 0
        self.rejected_count = 0
        self.conflict_count = 0

    # ============= 1. 명령 관찰 =============
    def observe_command(self,
                       text: str,
                       parsed: Optional[Dict] = None,
                       agent: Optional[str] = None) -> Optional[str]:
        """
        명령/요청 발화 관찰 → norm 후보 추출 + 등록/강화.

        Args:
            text: 원문
            parsed: NL.understand() 결과 (선택)
            agent: 발화자 (None이면 default_agent)

        Returns:
            norm_id 또는 None
        """
        if not text:
            return None

        agent = agent or self.default_agent
        self.observed_count += 1

        # 1) intent 확인 (관대하게: statement도 명령형 어미 있으면 OK)
        intent = parsed.get('intent', '') if parsed else ''
        is_explicit_command = intent in ('command', 'request', 'ask')

        # 2) action 카테고리 추출 (동사 패턴 매칭)
        action = self._extract_action(text)
        if not action:
            return None

        # 3) 강도 추출 (어미 기반)
        base_strength = self._extract_imperative_strength(text)

        # statement 인데 명령형 어미 매치되면 통과 (실용적 fix)
        if not is_explicit_command:
            # 명령형 어미가 있으면 (예: "쉬어", "쉬어라", "쉬셈")
            has_imperative_ending = any(
                ending in text for ending in self.IMPERATIVE_ENDINGS
            )
            # 짧은 명령형 (1-3음절) + 명령 어미
            short_imperative = (len(text) <= 8 and  # 짧은 텍스트
                               base_strength >= 0.5)  # 어미 매치
            if not (has_imperative_ending or short_imperative):
                return None  # 명령 아님
            # 약한 강도로 등록 (애매한 경우)
            base_strength = base_strength * 0.7

        # 4) 카테고리 활성된 것 중 condition 추정
        condition = self._extract_condition(parsed)

        # 5) 내재화
        norm_id = self.internalize(action, agent=agent,
                                  condition=condition,
                                  initial_strength=base_strength)
        return norm_id

    def _extract_action(self, text: str) -> Optional[str]:
        """텍스트에서 명령 동사 → action 카테고리"""
        text_lower = text.lower()
        # 한국어 동사 어간 패턴 매칭
        for verb_stem, action_cat in self.COMMAND_VERB_PATTERNS.items():
            if verb_stem in text_lower:
                return action_cat
        return None

    def _extract_imperative_strength(self, text: str) -> float:
        """어미 기반 명령 강도"""
        for ending, strength in self.IMPERATIVE_ENDINGS.items():
            if ending in text:
                return strength
        return 0.5  # 기본

    def _extract_condition(self, parsed: Optional[Dict]) -> Optional[str]:
        """parsed에서 condition 추정 (예: '피곤할때')"""
        if not parsed:
            return None
        # WM에서 현재 활성 카테고리 中 condition-like 카테고리
        cats = parsed.get('categories', set())
        condition_words = {'피곤', '졸려', '기뻐', '슬퍼', '아침', '저녁', '밤'}
        for cat in cats:
            if cat in condition_words:
                return cat
        return None

    # ============= 2. 내재화 =============
    def internalize(self,
                   action: str,
                   agent: Optional[str] = None,
                   condition: Optional[str] = None,
                   initial_strength: float = 0.5) -> str:
        """
        Norm 등록 또는 기존 강화.

        Returns: norm_id
        """
        agent = agent or self.default_agent

        # 같은 (agent, action, condition) 조합 검색
        norm_id = self._make_id(agent, action, condition)

        if norm_id in self.norms:
            # 기존 강화
            n = self.norms[norm_id]
            n.evidence_count += 1
            # 반복할수록 strength 누적 (sigmoid-like saturation)
            boost = initial_strength * 0.3 * (1.0 - n.strength)
            n.strength = min(1.0, n.strength + boost)
            n.last_observed = self.time

            # 임계 도달 → 내재화 완료 신호
            if n.evidence_count == self.internalize_threshold:
                self.internalized_count += 1
                # 호르몬 보상 (내재화 = 사회적 학습)
                if self.hs is not None:
                    self.hs.hormones['oxytocin'].stimulate(0.05)
                    self.hs.hormones['acetylcholine'].stimulate(0.05)
        else:
            # 신규 등록 (capacity check)
            if len(self.norms) >= self.max_norms:
                self._evict_weakest()

            n = Norm(
                id=norm_id,
                action=action,
                condition=condition,
                agent=agent,
                strength=initial_strength,
                evidence_count=1,
                last_observed=self.time,
                created_at=self.time,
            )
            self.norms[norm_id] = n
            self.agent_index[agent].add(norm_id)
            self.action_index[action].add(norm_id)

            # 약한 ACh 자극 (학습)
            if self.hs is not None:
                self.hs.hormones['acetylcholine'].stimulate(0.02)

        # action 카테고리도 SA 활성 (norm 떠올림)
        self.sa.activate(action, 0.3)

        return norm_id

    def _make_id(self, agent: str, action: str, condition: Optional[str]) -> str:
        """결정론적 norm_id"""
        cond = condition or '_'
        return f"{agent}|{action}|{cond}"

    def _evict_weakest(self):
        """capacity 초과 시 가장 약한 norm 제거"""
        if not self.norms:
            return
        # strength 낮고 evidence 적은 것 우선 제거
        weakest = min(self.norms.values(),
                     key=lambda n: (n.strength, n.evidence_count))
        self._remove_norm(weakest.id)

    def _remove_norm(self, norm_id: str):
        if norm_id not in self.norms:
            return
        n = self.norms[norm_id]
        del self.norms[norm_id]
        self.agent_index[n.agent].discard(norm_id)
        self.action_index[n.action].discard(norm_id)

    # ============= 3. 활성 규범 조회 =============
    def get_active_norms(self,
                        agent: Optional[str] = None,
                        threshold: float = 0.5,
                        condition: Optional[str] = None) -> List[Norm]:
        """
        현재 활성 (내재화된) 규범들.

        threshold 이상 strength + evidence ≥ internalize_threshold.
        """
        result = []
        candidates = (self.agent_index[agent] if agent else set(self.norms.keys()))

        for nid in candidates:
            n = self.norms.get(nid)
            if not n:
                continue
            if n.strength < threshold:
                continue
            if n.evidence_count < self.internalize_threshold:
                continue
            if condition and n.condition and n.condition != condition:
                continue
            result.append(n)

        # 결정론 정렬 (strength ↓, action 알파벳 ↑)
        result.sort(key=lambda n: (-n.strength, n.action))
        return result

    # ============= 4. DMN 자발 적용 =============
    def apply_norm_to_dmn(self, dmn=None) -> Optional[Dict[str, Any]]:
        """
        DMN 자발 활성에 norm 영향.

        조건이 맞는 활성 norm이 있으면 SA에 push + DMN focus 영향.

        Returns:
            적용된 norm 정보 또는 None
        """
        target_dmn = dmn if dmn is not None else self.dmn
        if target_dmn is None:
            return None

        # 1) 현재 활성 카테고리 중 condition 매치
        active_cats = self.sa.get_active() if hasattr(self.sa, 'get_active') else set()
        if not active_cats:
            active_cats = set()

        # 2) condition 매치되는 활성 norm 찾기
        active_norms = self.get_active_norms(threshold=0.5)

        for n in active_norms:
            # condition None = 무조건 적용 가능 (낮은 우선)
            # condition 매치 = 우선 적용
            should_apply = False
            if n.condition is None:
                should_apply = True  # 항상 적용 가능
            elif n.condition in active_cats:
                should_apply = True  # 조건 매치
            else:
                continue

            if should_apply:
                # action 카테고리 자발 활성 (norm 따르기)
                self.sa.activate(n.action, n.strength * 0.5)
                # WM에 push (DMN focus 영향)
                if hasattr(target_dmn, 'wm') and target_dmn.wm is not None:
                    target_dmn.wm.add(n.action, n.strength * 0.4, source='norm')

                n.last_applied = self.time
                self.applied_count += 1

                return {
                    'norm_id': n.id,
                    'action': n.action,
                    'agent': n.agent,
                    'strength': n.strength,
                    'condition': n.condition,
                    'time': self.time,
                }

        return None

    # ============= 5. 충돌 감지 =============
    def detect_conflict(self,
                       new_action: str,
                       agent: Optional[str] = None) -> List[Norm]:
        """
        새 명령이 기존 norm과 충돌하는지.

        간단 휴리스틱: 같은 agent의 *반대* action.
        예: '쉬어' (휴식) vs '공부해' (공부) — 같은 시점 충돌.

        실제로는 condition별로 다를 수 있지만 일단 단순.
        """
        opposing = {
            '휴식': {'공부', '운동'},
            '공부': {'휴식'},
            '운동': {'휴식'},
            '잠': {'공부', '운동'},
        }

        opponents = opposing.get(new_action, set())
        if not opponents:
            return []

        agent = agent or self.default_agent
        conflicts = []
        for nid in self.agent_index[agent]:
            n = self.norms.get(nid)
            if not n or n.action not in opponents:
                continue
            if n.strength < 0.3:
                continue
            conflicts.append(n)

        if conflicts:
            self.conflict_count += 1

        return sorted(conflicts, key=lambda n: -n.strength)

    # ============= 6. 약화 (반례) =============
    def weaken(self, norm_id: str, amount: float = 0.1):
        """사용자가 norm 거절했을 때 (반례)"""
        if norm_id not in self.norms:
            return
        n = self.norms[norm_id]
        n.strength = max(0.0, n.strength - amount)
        n.rejected_count += 1
        self.rejected_count += 1

        # 매우 약해지면 제거
        if n.strength < 0.05 and n.evidence_count <= 2:
            self._remove_norm(norm_id)

    def reject_action(self, action: str, agent: Optional[str] = None):
        """
        action에 대한 모든 norm 약화 (사용자 거절).
        예: 사용자가 EVE에게 "쉬어"라고 했었는데 지금 "쉬지마" → 휴식 norm 약화
        """
        agent = agent or self.default_agent
        for nid in self.action_index.get(action, set()).copy():
            n = self.norms.get(nid)
            if n and n.agent == agent:
                self.weaken(nid, amount=0.2)

    # ============= 7. tick + 시간 감쇠 =============
    def tick(self, dt: float = 0.5):
        """
        시간 감쇠 (안 쓰는 norm 약화).

        실제 dt는 초 단위. 하루 = 86400s.
        반감기: 1주일 정도 (strength 0.05 → 절반)
        """
        self.time += dt
        self.tick_count += 1

        # 매 600 tick (5분)마다 점검
        if self.tick_count % 600 != 0:
            return

        decay_per_tick = self.strength_decay_per_day * (dt * 600 / 86400.0)
        to_remove = []

        for nid, n in self.norms.items():
            # 최근 적용/관찰됐으면 보호
            time_since_used = self.time - max(n.last_observed, n.last_applied)
            if time_since_used < 3600:  # 1시간 이내
                continue

            # 강도 비례 decay (강한 norm 천천히)
            decay = decay_per_tick / (1.0 + n.strength * 2)
            n.strength = max(0.0, n.strength - decay)

            # 매우 약해지고 evidence 적으면 제거
            if n.strength < 0.05 and n.evidence_count < self.internalize_threshold:
                to_remove.append(nid)

        for nid in to_remove:
            self._remove_norm(nid)

    # ============= 8. introspect =============
    def introspect_norms(self,
                        agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        '민석이 자주 시키는 것' — 상위 norm들 보여주기.

        Returns:
            [{'action', 'condition', 'strength', 'evidence', 'last_used'}, ...]
        """
        agent = agent or self.default_agent
        norms = [self.norms[nid] for nid in self.agent_index[agent]
                if nid in self.norms]
        norms.sort(key=lambda n: (-n.strength, -n.evidence_count, n.action))

        result = []
        for n in norms[:20]:
            result.append({
                'action': n.action,
                'condition': n.condition or '항상',
                'strength': round(n.strength, 2),
                'evidence': n.evidence_count,
                'rejected': n.rejected_count,
                'last_observed_ago': round(self.time - n.last_observed, 1),
            })
        return result

    # ============= stats =============
    def stats(self) -> Dict[str, Any]:
        return {
            'norm_count': len(self.norms),
            'observed_count': self.observed_count,
            'internalized_count': self.internalized_count,
            'applied_count': self.applied_count,
            'rejected_count': self.rejected_count,
            'conflict_count': self.conflict_count,
            'agents': sorted(self.agent_index.keys()),
        }

    def __repr__(self):
        s = self.stats()
        return (f"NormInternalization(norms={s['norm_count']}, "
                f"observed={s['observed_count']}, "
                f"applied={s['applied_count']})")
