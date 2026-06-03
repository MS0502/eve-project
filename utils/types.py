"""
EVE v41 표준 데이터 구조

eve2.md 'Final Architecture' 2장 기준.
모든 모듈이 이 타입으로 주고받는다.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Meaning:
    """입력 텍스트에서 추출된 의미 표현 (모든 사고의 출발점)"""
    intent: str = "statement"          # question / statement / command / emotion
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    emotions: dict[str, float] = field(default_factory=dict)  # {fatigue: 0.8}
    context: dict = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class MeaningNode:
    id: str
    type: str                          # entity / action / emotion / state
    attributes: dict = field(default_factory=dict)


@dataclass
class MeaningEdge:
    source: str                        # node id
    relation: str                      # has / causes / before / ...
    target: str                        # node id


@dataclass
class WorkspaceState:
    """Global Workspace 현재 상태 (System1/2가 읽음)"""
    focus: Optional[str] = None        # 지금 가장 활성화된 노드 id
    working_memory: list = field(default_factory=list)   # 최근 MeaningGraph 들 (최대 10)
    hormone: dict[str, float] = field(default_factory=dict)
    active_goals: list[str] = field(default_factory=list)
    meaning: Optional[Meaning] = None  # 가장 최근 입력의 의미


@dataclass
class ResponsePlan:
    """말하기 전에 정하는 계획 (Generator가 이걸 기반으로 문장 만듦)"""
    intent: str = "statement"
    core_message: str = ""             # 핵심 한 줄
    sub_points: list[str] = field(default_factory=list)  # 0~5개 보조 문장
    emotional_tone: str = "neutral"    # calm / warm / firm / curious
    stage1_hint: Optional[str] = None  # planner가 호르몬 변조한 망설임 (None이면 톤 default)
    memory_refs: list = field(default_factory=list)
    skip_finalize: bool = False        # 라운드 30: 마무리 멘트 생략 (산술/모름/메타 등)


@dataclass
class Experience:
    """학습 단위 (모든 대화는 이걸로 저장)"""
    input_text: str
    meaning: Meaning
    response: str
    success_score: float = 0.0
    timestamp: float = 0.0
