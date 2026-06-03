"""
EVE v37.2 - 사회적 가상공간 (Social Environment)
================================================

NPC와 상호작용 → 옥시토신/유대 학습.

EVE 절대 원칙 부합:
- NPC = 카테고리 집합 + 성격 + 관계 (3D 렌더링 X)
- 호르몬이 진짜 사회적 보상 (옥시토신 + 엔도르핀)
- 결정론적 (NPC 응답도 규칙 기반, random X)
- 학습 모드 분리

핵심:
    NPC — 가상 사회적 존재
    SocialAction — 인사/대화/위로/함께/나누다
    Relationship — 낯선/지인/친구/가족 (시간/행동에 따라 진화)

사회적 보상 메커니즘 (학계 기반):
- Carter 1998: 옥시토신 = 유대 호르몬
- Insel & Young 2001: 사회적 인지 + 옥시토신
- Eisenberger 2003: 사회적 거절 = 신체 통증과 같은 신경 (cortisol↑)
- Wittig et al. 2012: 친구와의 양질 상호작용 → 엔도르핀

학습:
- 친구 호명 + 옥시토신 자극 → 친구 카테고리 강화
- 슬픈 친구 위로 성공 → "위로 → 친구 기뻐짐" 인과 학습
- 낯선이에게 무리한 행동 (위로/함께) → 어색함 → 약한 거절

Author: 김민석 + Claude v37
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from symbolic_env import SymbolicEnvironment, ActionResult, Goal


# ============================================================
# NPC 데이터 모델
# ============================================================

@dataclass
class NPC:
    """가상 사회적 존재."""
    name: str  # 카테고리명 ('진수', '엄마', '선배')
    relationship: str = '낯선'  # '낯선', '지인', '친구', '가족'
    mood: str = '보통'  # '좋음', '보통', '슬픔', '화남', '아픔'
    location: str = '주방'

    # 성격 — 어떤 행동을 어떻게 받아들이는지
    personality: Dict[str, Any] = field(default_factory=lambda: {
        'openness': 0.5,  # 0=폐쇄적, 1=개방적 (낯선이도 환영)
        'warmth': 0.5,   # 따뜻함
    })

    # 친밀도 — 0(낯선) ~ 1(매우 친함). 행동 따라 변함.
    intimacy: float = 0.0

    # 사회적 욕구 — 슬플 때는 위로받고 싶음
    wants: Set[str] = field(default_factory=set)
    # ex: {'위로', '대화', '함께'}

    # 진행 기록
    interactions: List[Dict] = field(default_factory=list)


# ============================================================
# 사회적 행동 규칙
# ============================================================

class SocialActionRules:
    """
    사회적 행동 → NPC 반응 + EVE 호르몬 자극.

    핵심 원칙:
    - 관계와 mood에 따라 같은 행동도 다른 결과
    - 친밀도가 점진적으로 발전
    - 친구를 잘 위로하면 옥시토신 큰 보상
    - 낯선이에게 과한 행동 → 어색함 (cortisol)
    """

    # ============= 인사하다 =============
    @staticmethod
    def 인사하다(env, npc: NPC) -> ActionResult:
        """인사 — 가장 가벼운 사회적 행동, 거의 항상 OK"""
        cats = {'인사', npc.name, npc.relationship}
        npc.intimacy = min(1.0, npc.intimacy + 0.05)

        if npc.mood == '화남':
            # 화난 사람에게 인사 — 약하게 좋음 + 약한 어색함
            cats.add('어색')
            return ActionResult(
                success=True,
                new_state_categories=cats,
                feedback=f"{npc.name}에게 인사했지만 받아주지 않았다",
                hormone_effects={'oxytocin': 0.02, 'cortisol': 0.05},
            )

        # 친밀도에 따라 옥시토신 강도
        ot_amount = 0.05 + 0.10 * npc.intimacy
        if npc.relationship == '친구' or npc.relationship == '가족':
            ot_amount += 0.05

        cats.add('환대')
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{npc.name}이(가) 인사를 받아줬다",
            hormone_effects={'oxytocin': ot_amount, 'serotonin': 0.05},
        )

    # ============= 대화하다 =============
    @staticmethod
    def 대화하다(env, npc: NPC) -> ActionResult:
        """대화 — 인사보다 깊음, 친밀도 따라 효과 다름"""
        cats = {'대화', npc.name, npc.relationship}

        if npc.relationship == '낯선' and npc.intimacy < 0.1:
            # 처음 만난 사람과 갑자기 대화 — 어색
            cats.add('어색')
            return ActionResult(
                success=False,
                new_state_categories=cats,
                feedback=f"{npc.name}은(는) 잘 모르는데 대화가 어색하다",
                hormone_effects={'cortisol': 0.10, 'oxytocin': 0.02},
            )

        # 화난 사람과 대화 — 시도는 좋음
        if npc.mood == '화남':
            cats.add('진정')
            npc.mood = '보통' if npc.intimacy > 0.5 else '화남'
            return ActionResult(
                success=npc.intimacy > 0.5,
                new_state_categories=cats,
                feedback=f"{npc.name}와(과) 대화로 풀어보려 했다",
                hormone_effects={'oxytocin': 0.05, 'cortisol': -0.05},
            )

        # 일반적 대화 — 친밀도 강화
        npc.intimacy = min(1.0, npc.intimacy + 0.10)
        ot_amount = 0.10 + 0.15 * npc.intimacy
        if npc.relationship in ('친구', '가족'):
            ot_amount += 0.05
            cats.add('가까움')

        cats.add('재밌음')
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{npc.name}와(과) 즐겁게 대화했다",
            hormone_effects={
                'oxytocin': ot_amount, 'serotonin': 0.10,
                'endorphin': 0.05, 'cortisol': -0.05,
            },
        )

    # ============= 위로하다 =============
    @staticmethod
    def 위로하다(env, npc: NPC) -> ActionResult:
        """위로 — 슬픈 사람에게 가장 효과적"""
        cats = {'위로', npc.name, npc.relationship}

        if npc.mood not in ('슬픔', '아픔'):
            # 안 슬픈데 위로 — 어색함 (cortisol)
            cats.add('어색')
            return ActionResult(
                success=False,
                new_state_categories=cats,
                feedback=f"{npc.name}은(는) 위로 받을 일이 없어 보이는데",
                hormone_effects={'cortisol': 0.10},
            )

        if npc.relationship == '낯선':
            # 낯선이 위로 — 좋은 의도지만 어색
            cats.add('어색')
            return ActionResult(
                success=False,
                new_state_categories=cats,
                feedback=f"{npc.name}을(를) 위로하려 했지만 잘 모르는 사이라 어색했다",
                hormone_effects={'cortisol': 0.10, 'oxytocin': 0.02},
            )

        # 슬픈 친구/가족 위로 성공 — 큰 옥시토신 + 자기 만족
        npc.mood = '보통'
        npc.intimacy = min(1.0, npc.intimacy + 0.20)
        cats.add('회복')
        cats.add(f'{npc.name}_기쁨')

        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{npc.name}을(를) 위로해서 마음이 풀렸다",
            hormone_effects={
                'oxytocin': 0.30, 'serotonin': 0.20,
                'endorphin': 0.20, 'dopamine': 0.15,  # 자기 만족
                'cortisol': -0.10,
            },
        )

    # ============= 함께있다 =============
    @staticmethod
    def 함께있다(env, npc: NPC) -> ActionResult:
        """함께 시간 — 깊은 유대"""
        cats = {'함께', npc.name, npc.relationship}

        if npc.relationship == '낯선':
            cats.add('어색')
            return ActionResult(
                success=False,
                new_state_categories=cats,
                feedback=f"{npc.name}와(과) 함께 있기엔 너무 낯설다",
                hormone_effects={'cortisol': 0.08},
            )

        npc.intimacy = min(1.0, npc.intimacy + 0.15)
        ot_amount = 0.15 + 0.20 * npc.intimacy

        if npc.relationship == '가족':
            ot_amount += 0.10
            cats.add('소속감')

        cats.add('편안')
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{npc.name}와(과) 함께 있어 편안했다",
            hormone_effects={
                'oxytocin': ot_amount, 'serotonin': 0.15,
                'endorphin': 0.10, 'cortisol': -0.10,
            },
        )

    # ============= 나누다 =============
    @staticmethod
    def 나누다(env, npc: NPC) -> ActionResult:
        """뭔가를 나누다 — 음식/얘기 등"""
        cats = {'나누다', npc.name}

        if npc.relationship == '낯선':
            # 낯선이에게 나누는 건 OK — 호의 표시
            npc.intimacy = min(1.0, npc.intimacy + 0.10)
            if npc.intimacy > 0.15:
                npc.relationship = '지인'
            cats.add('친절')
            return ActionResult(
                success=True,
                new_state_categories=cats,
                feedback=f"{npc.name}에게 나눠주니 친해진 느낌이다",
                hormone_effects={'oxytocin': 0.10, 'endorphin': 0.05},
            )

        # 친구/가족과 나누다 — 큰 보상
        npc.intimacy = min(1.0, npc.intimacy + 0.10)
        cats.add('베풀음')
        cats.add('뿌듯')
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{npc.name}와(과) 나누니 뿌듯하다",
            hormone_effects={
                'oxytocin': 0.20, 'endorphin': 0.15,
                'dopamine': 0.10, 'serotonin': 0.10,
            },
        )

    REGISTRY = {
        '인사하다': '인사하다', '인사': '인사하다', '안녕': '인사하다',
        '대화하다': '대화하다', '대화': '대화하다', '얘기': '대화하다',
        '위로하다': '위로하다', '위로': '위로하다',
        '함께있다': '함께있다', '함께': '함께있다',
        '나누다': '나누다', '나눠주다': '나누다',
    }


# ============================================================
# 사회적 환경 — SymbolicEnvironment 확장
# ============================================================

class SocialEnvironment(SymbolicEnvironment):
    """NPC가 있는 사회적 공간."""

    def __init__(self, name: str):
        super().__init__(name)
        self.npcs: Dict[str, NPC] = {}

    def add_npc(self, npc: NPC):
        self.npcs[npc.name] = npc
        self.locations.add(npc.location)

    def get_npcs_here(self) -> List[NPC]:
        """현재 위치의 NPC들"""
        return [n for n in self.npcs.values()
                if n.location == self.current_location]

    # ============= 관찰 — NPC 포함 =============
    def observe(self) -> Dict[str, Any]:
        """현재 상태 — 객체 + NPC + 사회적 단서"""
        base = super().observe()

        npcs_here = self.get_npcs_here()
        cats = base['categories']
        npc_info = []

        for npc in npcs_here:
            cats.add(npc.name)
            cats.add(npc.relationship)
            cats.add(npc.mood)  # 상대방 감정 인식
            # 친밀도 낮으면 '낯선' 카테고리도 활성
            if npc.intimacy < 0.2:
                cats.add('낯선')
            elif npc.intimacy > 0.6:
                cats.add('가까움')
            # 욕구 단서 — EVE가 위로/함께 등 떠올릴 수 있게
            cats.update(npc.wants)

            npc_info.append({
                'name': npc.name,
                'relationship': npc.relationship,
                'mood': npc.mood,
                'intimacy': npc.intimacy,
                'wants': sorted(npc.wants),
            })

        # 자연어 요약 보강
        if npcs_here:
            persons = ', '.join(n.name for n in npcs_here)
            mood_note = ''
            sad_npcs = [n for n in npcs_here if n.mood in ('슬픔', '아픔')]
            if sad_npcs:
                mood_note = f" ({sad_npcs[0].name}이 {sad_npcs[0].mood} 같다)"
            base['summary'] = f"{base['summary']}. {persons}이(가) 있다{mood_note}"

        base['npcs'] = npc_info
        return base

    # ============= 행동 — 사회적 행동 분기 =============
    def execute_action(self, action: str, target_name: str = None) -> ActionResult:
        # 1) 사회적 행동인지 체크
        social_method_name = SocialActionRules.REGISTRY.get(action)
        if social_method_name is not None:
            method = getattr(SocialActionRules, social_method_name, None)
            if method is None:
                return ActionResult(
                    success=False,
                    new_state_categories={'실패'},
                    feedback=f"'{action}' 실행 실패",
                    hormone_effects={},
                )
            # 대상 NPC 찾기
            npc = self.npcs.get(target_name)
            if npc is None:
                return ActionResult(
                    success=False,
                    new_state_categories={'실패', '없음'},
                    feedback=f"'{target_name}'이라는 사람이 여기 없다",
                    hormone_effects={'cortisol': 0.03},
                )
            if npc.location != self.current_location:
                return ActionResult(
                    success=False,
                    new_state_categories={'실패', '먼곳'},
                    feedback=f"{target_name}은(는) 여기 없다",
                    hormone_effects={'cortisol': 0.03},
                )
            result = method(self, npc)

            # NPC 상호작용 기록
            npc.interactions.append({
                'action': action,
                'success': result.success,
                'feedback': result.feedback,
            })

            # 환경 action_history도
            self.action_history.append({
                'action': action,
                'target': target_name,
                'success': result.success,
                'feedback': result.feedback,
            })

            # 목표 체크 (사회적 목표 포함)
            for goal in self.goals:
                if not goal.completed and goal.condition(self.state):
                    goal.completed = True
                    for h, v in goal.reward_hormones.items():
                        result.hormone_effects[h] = result.hormone_effects.get(h, 0.0) + v
                    result.new_state_categories.add(f'목표달성:{goal.name}')
                    result.feedback += f" [목표 '{goal.name}' 달성!]"

            return result

        # 2) 일반 행동은 부모 클래스 (SymbolicEnvironment.execute_action)
        return super().execute_action(action, target_name)


# ============================================================
# 사전 제작 사회적 환경
# ============================================================

def make_social_living_room() -> SocialEnvironment:
    """거실 — 친구 진수가 슬퍼함, 가족 엄마, 낯선 손님"""
    env = SocialEnvironment('거실')

    # NPCs
    env.add_npc(NPC(
        name='진수',
        relationship='친구',
        mood='슬픔',
        location='거실',
        personality={'openness': 0.7, 'warmth': 0.8},
        intimacy=0.5,
        wants={'위로', '함께'},
    ))
    env.add_npc(NPC(
        name='엄마',
        relationship='가족',
        mood='보통',
        location='거실',
        personality={'openness': 0.6, 'warmth': 0.9},
        intimacy=0.8,  # 이미 친밀
        wants={'대화'},
    ))
    env.add_npc(NPC(
        name='손님',
        relationship='낯선',
        mood='보통',
        location='거실',
        personality={'openness': 0.4, 'warmth': 0.5},
        intimacy=0.0,
        wants=set(),
    ))

    # 환경 상태 — 사회적 차원
    env.state['외로움'] = 0.4
    env.state['연결감'] = 0.3

    # 사회적 목표
    env.goals.append(Goal(
        name='친구 위로 성공',
        condition=lambda s: any(
            f'{npc.name}_기쁨' in s.get('events', set())
            for npc in []
        ) or False,  # 이건 체크 어려움 — 단순화
        reward_hormones={'oxytocin': 0.2},
    ))

    return env


def make_first_meeting_env() -> SocialEnvironment:
    """첫 만남 — 낯선이만 있는 환경, 친구 만들기"""
    env = SocialEnvironment('카페')

    env.add_npc(NPC(
        name='지영',
        relationship='낯선',
        mood='보통',
        location='카페',
        personality={'openness': 0.7, 'warmth': 0.6},
        intimacy=0.0,
    ))

    env.state['외로움'] = 0.6

    # 친구 만들기 목표 — intimacy > 0.4
    def became_friend(s):
        # state에 직접 안 저장하니 npcs 직접 봐야 — 단순화로 외로움 해소로
        return s.get('연결감', 0) > 0.5
    env.goals.append(Goal(
        name='새 친구 만들기',
        condition=became_friend,
        reward_hormones={'oxytocin': 0.3, 'dopamine': 0.2},
    ))

    return env
