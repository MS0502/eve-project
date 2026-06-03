"""
EVE v37 - 기호적 가상공간 (Symbolic Environment)
================================================

EVE 절대 원칙 부합:
- 풀 3D 렌더링 X — 의미 레벨 작동
- 카테고리 = 객체/위치/상태/행동
- 호르몬 = 보상 (성공→DA, 실패→CORT)
- 결정론적 (random X)
- 학습 모드 분리 (메인 EVE 안전)

핵심:
    Environment — 공간 (방, 객체, 상태)
    Object — 카테고리 + 속성 + 가능한 행동
    Action — (행위자, 동작, 대상) → 상태 전이
    Goal — 호르몬 보상 트리거 조건

학습 흐름:
    enter() → 카테고리 활성 (관찰)
    act(action, target) → state transition + 호르몬 자극
    SA Hebbian — 행동-결과 연결 강화
    CG learn_causal — 인과 학습
    exit() → 학습 통합 (수면 시 BDNF↑)

학계 기반:
- Sutton & Barto 2018 — RL 기본 (S, A, R, P)
- Botvinick et al. 2019 — Reinforcement learning, fast and slow
- Friston 2010 — Active Inference (행동 = 예측 오차 최소화)
- Lake et al. 2017 — 인과 모델 + symbolic primitive

Author: 김민석 + Claude v37
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class SymObject:
    """기호 객체 — 카테고리 + 속성 + 가능 행동"""
    name: str  # 카테고리 이름 ('사과', '컵')
    location: str  # 어디 있는지 ('주방', '냉장고')
    properties: Dict[str, Any] = field(default_factory=dict)
    # ex: {'edible': True, 'color': '빨강', 'size': '작음'}
    available_actions: Set[str] = field(default_factory=set)
    # ex: {'먹다', '잡다', '보다'}


@dataclass
class Goal:
    """환경 내 목표 — 호르몬 보상 트리거"""
    name: str
    condition: Callable  # (env_state) -> bool
    reward_hormones: Dict[str, float] = field(default_factory=dict)
    # ex: {'dopamine': 0.3, 'endorphin': 0.2}
    completed: bool = False


@dataclass
class ActionResult:
    """행동 결과"""
    success: bool
    new_state_categories: Set[str]  # 행동 후 활성될 카테고리
    feedback: str  # 자연어 피드백 ("사과를 먹었다")
    hormone_effects: Dict[str, float]  # 호르몬 자극


# ============================================================
# 행동 규칙 — 결정론적 상태 전이
# ============================================================

class ActionRules:
    """
    행동 → 상태 전이 규칙 (사전 정의, 결정론).

    NOTE: 이건 환경의 '물리 법칙'이라 결정론이어야 함.
    EVE의 '학습'은 어떤 행동이 어떤 결과를 낳는지 *발견*하는 것.
    """

    @staticmethod
    def 먹다(env, target: SymObject) -> ActionResult:
        """뭔가를 먹는다"""
        if not target.properties.get('edible', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패', '못먹음', target.name},
                feedback=f"{target.name}은(는) 먹을 수 없다",
                hormone_effects={'cortisol': 0.10, 'dopamine': -0.05},
            )
        # 성공 — 객체 제거, 배고픔 해소
        env.remove_object(target.name)
        env.state['배고픔'] = max(0.0, env.state.get('배고픔', 0.5) - 0.5)
        env.state['만족'] = env.state.get('만족', 0.0) + 0.4
        cats = {'먹었다', target.name, '맛있다', '만족', '배부름'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}을(를) 먹었다. 맛있었다.",
            hormone_effects={
                'dopamine': 0.25, 'endorphin': 0.20,
                'serotonin': 0.10, 'cortisol': -0.10,
                'leptin': 0.15, 'ghrelin': -0.20,
            },
        )

    @staticmethod
    def 마시다(env, target: SymObject) -> ActionResult:
        """뭔가를 마신다"""
        if not target.properties.get('drinkable', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패', '못마심', target.name},
                feedback=f"{target.name}은(는) 마실 수 없다",
                hormone_effects={'cortisol': 0.08},
            )
        env.remove_object(target.name)
        env.state['목마름'] = max(0.0, env.state.get('목마름', 0.5) - 0.5)
        cats = {'마셨다', target.name, '시원하다', '만족'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}을(를) 마셨다. 시원했다.",
            hormone_effects={
                'dopamine': 0.15, 'endorphin': 0.10,
                'cortisol': -0.05,
            },
        )

    @staticmethod
    def 잡다(env, target: SymObject) -> ActionResult:
        """뭔가를 잡는다 (들기)"""
        if not target.properties.get('graspable', True):
            return ActionResult(
                success=False,
                new_state_categories={'실패', '못잡음'},
                feedback=f"{target.name}은(는) 잡을 수 없다",
                hormone_effects={'cortisol': 0.05},
            )
        env.state['손에든것'] = target.name
        cats = {'잡았다', target.name, '손'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}을(를) 잡았다",
            hormone_effects={'norepinephrine': 0.05},  # 약한 각성
        )

    @staticmethod
    def 보다(env, target: SymObject) -> ActionResult:
        """뭔가를 본다 (관찰)"""
        cats = {'봤다', target.name}
        # 객체 속성도 활성
        for prop_v in target.properties.values():
            if isinstance(prop_v, str):
                cats.add(prop_v)
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}을(를) 봤다",
            hormone_effects={'norepinephrine': 0.02},  # 주의
        )

    @staticmethod
    def 가다(env, target_location: str) -> ActionResult:
        """다른 위치로 이동"""
        if target_location not in env.locations:
            return ActionResult(
                success=False,
                new_state_categories={'실패', '없는곳'},
                feedback=f"{target_location}이라는 곳은 없다",
                hormone_effects={'cortisol': 0.05},
            )
        env.current_location = target_location
        cats = {'이동', target_location}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target_location}(으)로 갔다",
            hormone_effects={},
        )

    # ============= 방 행동 =============
    @staticmethod
    def 눕다(env, target: 'SymObject') -> ActionResult:
        """침대 등에 눕다 — 휴식, 멜라토닌 ↑"""
        if not target.properties.get('sleepable', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패', '못눕음'},
                feedback=f"{target.name}에는 눕기 어색하다",
                hormone_effects={'cortisol': 0.03},
            )
        env.state['자세'] = '눕다'
        cats = {'눕다', target.name, '편안', '휴식'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}에 누웠다",
            hormone_effects={
                'melatonin': 0.10, 'cortisol': -0.05,
                'adenosine': 0.05,  # 자연 회복
            },
        )

    @staticmethod
    def 앉다(env, target: 'SymObject') -> ActionResult:
        """의자/책상 앞 앉기"""
        if not target.properties.get('sittable', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패'},
                feedback=f"{target.name}에 앉기 어색하다",
                hormone_effects={'cortisol': 0.02},
            )
        env.state['자세'] = '앉다'
        cats = {'앉다', target.name, '준비됨'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}에 앉았다",
            hormone_effects={'norepinephrine': 0.03},  # 약한 각성
        )

    @staticmethod
    def 읽다(env, target: 'SymObject') -> ActionResult:
        """책/일기 읽기 — 도파민 약하게, 호기심"""
        if not target.properties.get('readable', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패'},
                feedback=f"{target.name}은(는) 읽을 수 없다",
                hormone_effects={'cortisol': 0.02},
            )
        cats = {'읽다', target.name, '집중', '재밌음'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}을(를) 읽었다",
            hormone_effects={
                'dopamine': 0.10, 'acetylcholine': 0.05,
                'norepinephrine': 0.03, 'cortisol': -0.03,
            },
        )

    @staticmethod
    def 쓰다(env, target: 'SymObject') -> ActionResult:
        """일기 쓰기 — 감정 정리, 코르티솔 ↓"""
        if not target.properties.get('writable', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패'},
                feedback=f"{target.name}에는 쓸 수 없다",
                hormone_effects={'cortisol': 0.02},
            )
        cats = {'쓰다', target.name, '정리', '뿌듯'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback=f"{target.name}에 썼다",
            hormone_effects={
                'serotonin': 0.10, 'cortisol': -0.10,
                'dopamine': 0.05, 'acetylcholine': 0.05,
            },
        )

    @staticmethod
    def 창밖_보다(env, target: 'SymObject') -> ActionResult:
        """창밖 보기 — 가벼운 자극, 호기심"""
        if target.name != '창문' and not target.properties.get('window', False):
            return ActionResult(
                success=False,
                new_state_categories={'실패'},
                feedback=f"{target.name}은(는) 창문이 아니다",
                hormone_effects={},
            )
        cats = {'창밖_보다', '창문', '하늘', '관찰', '편안'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback="창밖을 봤다",
            hormone_effects={
                'norepinephrine': 0.03, 'serotonin': 0.05,
                'cortisol': -0.03,
            },
        )

    @staticmethod
    def 쉬다(env, target=None) -> ActionResult:
        """현재 자리에서 쉬기 — 가벼운 휴식"""
        cats = {'쉬다', '편안'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback="잠시 쉬었다",
            hormone_effects={
                'cortisol': -0.05, 'melatonin': 0.02,
            },
        )

    @staticmethod
    def 있다(env, target=None) -> ActionResult:
        """그냥 있기 — 무행동 능동 선택"""
        cats = {'있다', '관찰'}
        return ActionResult(
            success=True,
            new_state_categories=cats,
            feedback="그냥 있다",
            hormone_effects={},
        )

    # 행동 이름 → 메서드
    REGISTRY = {
        '먹다': '먹다',
        '먹어': '먹다',
        '먹기': '먹다',
        '마시다': '마시다',
        '마셔': '마시다',
        '잡다': '잡다',
        '잡아': '잡다',
        '들다': '잡다',
        '보다': '보다',
        '관찰': '보다',
        '가다': '가다',
        '이동': '가다',
        # 방 행동
        '눕다': '눕다',
        '눕기': '눕다',
        '앉다': '앉다',
        '앉기': '앉다',
        '읽다': '읽다',
        '읽기': '읽다',
        '쓰다': '쓰다',
        '쓰기': '쓰다',
        '창밖_보다': '창밖_보다',
        '창밖': '창밖_보다',
        '쉬다': '쉬다',
        '쉬기': '쉬다',
        '있다': '있다',
    }


# ============================================================
# Environment
# ============================================================

class SymbolicEnvironment:
    """기호적 환경 — 공간/객체/상태/규칙."""

    def __init__(self, name: str, is_simulation: bool = True):
        """
        Args:
            name: 환경 이름
            is_simulation: 학습 시뮬인지. False = EVE의 진짜 거주지 (자기 방).
                          True (기본) = 시뮬 환경, '학습_모드' 카테고리 활성.
        """
        self.name = name
        self.is_simulation = is_simulation
        self.objects: Dict[str, SymObject] = {}
        self.locations: Set[str] = {name}  # 환경 자체도 위치
        self.current_location: str = name

        # 환경 상태 (욕구/조건)
        self.state: Dict[str, Any] = {
            '배고픔': 0.0, '목마름': 0.0, '만족': 0.0,
            '손에든것': None,
        }

        # 목표
        self.goals: List[Goal] = []

        # 진행 기록
        self.action_history: List[Dict] = []

    # ============= 객체 관리 =============
    def add_object(self, obj: SymObject):
        self.objects[obj.name] = obj
        self.locations.add(obj.location)

    def remove_object(self, name: str):
        self.objects.pop(name, None)

    def get_objects_here(self) -> List[SymObject]:
        """현재 위치의 객체들"""
        return [obj for obj in self.objects.values()
                if obj.location == self.current_location]

    # ============= 관찰 =============
    def observe(self) -> Dict[str, Any]:
        """현재 상태 관찰 — 카테고리 + 자연어 요약"""
        objs_here = self.get_objects_here()
        cats: Set[str] = {self.current_location}
        for obj in objs_here:
            cats.add(obj.name)
            cats.update(obj.available_actions)
            for v in obj.properties.values():
                if isinstance(v, str):
                    cats.add(v)
        # 욕구 상태도 카테고리화
        if self.state.get('배고픔', 0) > 0.5:
            cats.add('배고픔')
        if self.state.get('목마름', 0) > 0.5:
            cats.add('목마름')
        if self.state.get('손에든것'):
            cats.add(self.state['손에든것'])
            cats.add('손에들음')

        # 자연어 요약
        if objs_here:
            obj_str = ', '.join(o.name for o in objs_here)
            summary = f"{self.current_location}에 {obj_str}이(가) 있다"
        else:
            summary = f"{self.current_location}에 아무것도 없다"

        return {
            'categories': cats,
            'objects': [o.name for o in objs_here],
            'location': self.current_location,
            'state': dict(self.state),
            'summary': summary,
        }

    # ============= 행동 =============
    def execute_action(self, action: str, target_name: str = None) -> ActionResult:
        """행동 실행 — 결정론적 상태 전이"""
        # 행동 정규화
        method_name = ActionRules.REGISTRY.get(action)
        if method_name is None:
            return ActionResult(
                success=False,
                new_state_categories={'모르는행동'},
                feedback=f"'{action}'이 무슨 행동인지 모르겠다",
                hormone_effects={},
            )

        method = getattr(ActionRules, method_name, None)
        if method is None:
            return ActionResult(
                success=False,
                new_state_categories={'실패'},
                feedback=f"'{action}' 실행 실패",
                hormone_effects={},
            )

        # 가다(이동)는 target이 location 이름
        if method_name == '가다':
            result = method(self, target_name)
        else:
            # 객체 대상
            target_obj = self.objects.get(target_name)
            if target_obj is None:
                return ActionResult(
                    success=False,
                    new_state_categories={'실패', '없음'},
                    feedback=f"'{target_name}'이(가) 여기 없다",
                    hormone_effects={'cortisol': 0.05},
                )
            # 현재 위치에 있어야
            if target_obj.location != self.current_location:
                return ActionResult(
                    success=False,
                    new_state_categories={'실패', '먼곳'},
                    feedback=f"{target_name}은(는) 여기에 없다",
                    hormone_effects={'cortisol': 0.05},
                )
            result = method(self, target_obj)

        # 기록
        self.action_history.append({
            'action': action,
            'target': target_name,
            'success': result.success,
            'feedback': result.feedback,
        })

        # 목표 체크
        for goal in self.goals:
            if not goal.completed and goal.condition(self.state):
                goal.completed = True
                # 보상 호르몬 추가
                for h, v in goal.reward_hormones.items():
                    result.hormone_effects[h] = result.hormone_effects.get(h, 0.0) + v
                result.new_state_categories.add(f'목표달성:{goal.name}')
                result.feedback += f" [목표 '{goal.name}' 달성!]"

        return result


# ============================================================
# 사전 제작 환경
# ============================================================

def make_kitchen_env() -> SymbolicEnvironment:
    """주방 환경 — 배고픔/목마름 해소"""
    env = SymbolicEnvironment('주방')

    # 객체들
    env.add_object(SymObject(
        name='사과',
        location='주방',
        properties={'edible': True, 'color': '빨강', 'size': '작음'},
        available_actions={'먹다', '잡다', '보다'},
    ))
    env.add_object(SymObject(
        name='물',
        location='주방',
        properties={'drinkable': True, 'color': '투명'},
        available_actions={'마시다', '보다'},
    ))
    env.add_object(SymObject(
        name='의자',
        location='주방',
        properties={'edible': False, 'graspable': False, 'color': '갈색'},
        available_actions={'보다'},
    ))
    env.add_object(SymObject(
        name='책',
        location='거실',  # 다른 위치
        properties={'edible': False, 'color': '파랑'},
        available_actions={'잡다', '보다'},
    ))

    # 초기 상태 — EVE는 배고프고 목마름
    env.state['배고픔'] = 0.7
    env.state['목마름'] = 0.6

    # 목표
    env.goals.append(Goal(
        name='배고픔 해소',
        condition=lambda s: s.get('배고픔', 1.0) < 0.3,
        reward_hormones={'dopamine': 0.3, 'serotonin': 0.2},
    ))
    env.goals.append(Goal(
        name='목마름 해소',
        condition=lambda s: s.get('목마름', 1.0) < 0.3,
        reward_hormones={'dopamine': 0.2, 'oxytocin': 0.1},
    ))

    return env
