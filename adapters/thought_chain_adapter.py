"""
ThoughtChainAdapter — 라운드 39b

GPT 분석 (라운드 38 후): 진짜 부족한 건 "생각의 흐름"
- 지금: A → 랜덤 → B
- 목표: A → 연관 → B → 연관 → C ("민석 → PT → 피곤")

학계:
- Collins & Loftus 1975 — Spreading Activation Theory
- Raichle 2001 — Default Mode Network (DMN)
- Mason et al 2007 — Wandering minds
- Smallwood & Schooler 2015 — Mind wandering as decoupled cognition
- Christoff et al 2016 — Dynamic framework of mind wandering
- Dixon et al 2014 — DMN cognitive transitions

원칙:
- next_cat ∝ salience[next] × association[curr][next] × recent_spike
- chain length 2~4 (인간 단기 사고 흐름)
- 결정론 (random X — argmax + tiebreak)
- 같은 카테고리 반복 X
"""

from dataclasses import dataclass, field
from typing import Optional


# 사고 체인 길이 (인간 평균 2~4 step)
CHAIN_MIN_LENGTH = 2
CHAIN_MAX_LENGTH = 4

# 다음 후보 평가 가중치
W_SALIENCE = 0.4       # 중요도
W_ASSOCIATION = 0.5    # 연결 강도 (SA)
W_NOVELTY = 0.1        # 최근 안 떠오른 거 보정 (반복 방지)


@dataclass
class ThoughtNode:
    """사고 체인의 한 노드."""
    category: str
    score: float
    via: str = ""              # "association" / "salience" / "seed"
    salience_value: float = 0.0
    association_weight: float = 0.0


@dataclass
class ThoughtChain:
    """A → B → C 사고 흐름."""
    nodes: list = field(default_factory=list)
    
    def categories(self) -> list:
        return [n.category for n in self.nodes]
    
    def __len__(self) -> int:
        return len(self.nodes)


class ThoughtChainAdapter:
    """
    카테고리 사고 흐름 생성.
    
    spreading activation: 현재 카테고리에서 강하게 연결된 다음 카테고리로.
    salience: 더 중요한 거 선호.
    novelty: 최근 안 떠오른 거 가산점 (체인 안 반복 방지).
    """

    def __init__(self, engine=None,
                 min_length: int = CHAIN_MIN_LENGTH,
                 max_length: int = CHAIN_MAX_LENGTH,
                 min_association: float = 0.1):
        self.engine = engine
        self.min_length = min_length
        self.max_length = max_length
        self.min_association = min_association
        # 통계
        self.chains_generated = 0
        self.total_nodes = 0
        self._last_chain: Optional[ThoughtChain] = None

    def generate_chain(self, start_category: str,
                       length: Optional[int] = None) -> ThoughtChain:
        """
        시작 카테고리 → N단계 사고 체인.
        Args:
            start_category: 첫 카테고리
            length: 원하는 길이 (None이면 max_length)
        """
        if length is None:
            length = self.max_length
        length = max(self.min_length, min(self.max_length, length))
        
        chain = ThoughtChain()
        chain.nodes.append(ThoughtNode(
            category=start_category,
            score=1.0,
            via="seed",
        ))
        
        visited = {start_category}
        current = start_category
        
        for _step in range(length - 1):
            next_node = self._pick_next(current, visited)
            if next_node is None:
                break
            chain.nodes.append(next_node)
            visited.add(next_node.category)
            current = next_node.category
        
        self.chains_generated += 1
        self.total_nodes += len(chain.nodes)
        self._last_chain = chain
        return chain

    def _pick_next(self, current: str, visited: set) -> Optional[ThoughtNode]:
        """
        현재 카테고리에서 다음 카테고리 결정 (deterministic argmax).
        후보 = SA에서 current와 연결된 카테고리들 + Salience top.
        score = W_S × salience + W_A × association + W_N × novelty
        """
        if self.engine is None:
            return None
        
        candidates = self._gather_candidates(current, visited)
        if not candidates:
            return None
        
        # 각 후보 점수 계산
        scored = []
        for cat in candidates:
            assoc = self._get_association(current, cat)
            sal = self._get_salience(cat)
            # novelty — visited에 있으면 0
            novelty = 0.0 if cat in visited else 1.0
            
            score = (W_SALIENCE * sal +
                     W_ASSOCIATION * assoc +
                     W_NOVELTY * novelty)
            
            scored.append(ThoughtNode(
                category=cat,
                score=score,
                via="association" if assoc > sal else "salience",
                salience_value=sal,
                association_weight=assoc,
            ))
        
        # argmax (결정론) — 같으면 카테고리 이름 alphabet 우선
        scored.sort(key=lambda n: (-n.score, n.category))
        
        # 점수 너무 낮으면 안 잡음 (random walk 방지)
        if scored[0].score < 0.1:
            return None
        
        return scored[0]

    def _gather_candidates(self, current: str, visited: set) -> list:
        """SA + Salience에서 후보 카테고리 모음. 합성 키 제외."""
        cands = set()
        
        # SA 이웃 — current와 연결된 카테고리
        if self.engine.activation_adapter is not None:
            sa = self.engine.activation_adapter.sa
            if hasattr(sa, "weights"):
                for (a, b), w in sa.weights.items():
                    if w < self.min_association:
                        continue
                    if a == current and b not in visited:
                        cands.add(b)
                    elif b == current and a not in visited:
                        cands.add(a)
        
        # Salience top 카테고리 (현재 활성)
        if hasattr(self.engine, "salience_adapter") and self.engine.salience_adapter is not None:
            for (cat, _val) in self.engine.salience_adapter.top(10):
                if cat not in visited:
                    cands.add(cat)
        
        # 라운드 46+53+54: 합성 키 / 의문 부호 / 내부 식별자 / 의문문 어미 / stopword 제외
        BAD_SUFFIXES = ('했어', '하고', '있었어', '있어', '거다', '거야', 
                        '인지', '냐', '느냐', '었는다', '했는다', '있는거',
                        # 라운드 54: 명령형 / 합성 어절
                        '해봐', '해줘', '말해', '말해봐', '해라', '하자')
        BAD_WORDS = ("뭐", "왜", "뭔", "어떻게", "언제", "어디", "누구",
                     "그", "이", "저", "음", "그냥", "어떤", "그건",
                     "이건", "저건", "이거", "저거", "그거",
                     "안녕만해봐", "사는거다",
                     # 라운드 55
                     "뭐해", "뭐하", "뭔지알아", "있어", "없어",
                     "그래", "맞아", "아니")
        clean_cands = []
        for c in cands:
            if not c:
                continue
            if "+" in c or "@" in c:
                continue
            if c.endswith("?") or c.endswith("？"):
                continue
            if "_" in c:
                continue
            if c in BAD_WORDS:
                continue
            if any(c.endswith(suf) for suf in BAD_SUFFIXES):
                continue
            # 라운드 54: 너무 긴 어절 (4어 이상 합성된 거)
            if len(c) > 12:
                continue
            clean_cands.append(c)
        return clean_cands

    def _get_association(self, a: str, b: str) -> float:
        """SA 가중치 (정규화 0~1)."""
        if self.engine.activation_adapter is None:
            return 0.0
        sa = self.engine.activation_adapter.sa
        if not hasattr(sa, "get_weight"):
            return 0.0
        w = sa.get_weight(a, b)
        # SA 가중치는 보통 0~1 범위 (하지만 누적 가능). clamp.
        return max(0.0, min(1.0, w))

    def _get_salience(self, category: str) -> float:
        """Salience 값 (0~1 정규화)."""
        if not hasattr(self.engine, "salience_adapter") or self.engine.salience_adapter is None:
            return 0.0
        v = self.engine.salience_adapter.get(category)
        # Salience ceiling 5이니 1로 정규화
        return max(0.0, min(1.0, v / 5.0))

    # ===== 발화 표현 =====

    def chain_to_phrase(self, chain: ThoughtChain) -> str:
        """
        사고 체인 → 자연어 한 문장.
        예: ["민석", "PT", "피곤"] → "민석 생각하다 보니 PT 떠올라서, 피곤하다는 생각이 났어"
        라운드 46: 합성 키 (+, @, _) cleanup
        """
        if len(chain) < 2:
            return ""
        
        cats = [self._clean_cat(c) for c in chain.categories()]
        # cleanup 후 빈 카테고리/중복 제거
        seen = set()
        unique_cats = []
        for c in cats:
            if c and c not in seen:
                seen.add(c)
                unique_cats.append(c)
        cats = unique_cats
        
        if len(cats) < 2:
            return ""
        
        if len(cats) == 2:
            return f"{cats[0]} 생각하다 보니 {cats[1]}이/가 떠오르네"
        elif len(cats) == 3:
            return f"{cats[0]} 생각하다 {cats[1]} 떠올라서, {cats[2]}까지 생각났어"
        else:  # 4+
            middle = " → ".join(cats[1:-1])
            return f"{cats[0]} 떠올랐는데, {middle} 거쳐서 {cats[-1]}까지 갔네"

    @staticmethod
    def _clean_cat(cat: str) -> str:
        """라운드 46+50: 합성 키 / 내부 식별자 / 어절 반복 정리.
        joy+뜻인다 → joy, 진짜_친구 → 진짜
        '내일내일오늘' → '내일' (반복 어절 제거)
        """
        if not cat:
            return ""
        # VSA 합성 키 분리
        c = cat.split("+")[0].split("@")[0]
        # 내부 식별자 (언더스코어)
        if "_" in c:
            c = c.split("_")[0]
        # 의문 부호 제거
        c = c.rstrip("?？.,!")
        c = c.strip()
        # 라운드 50: 어절 반복 패턴 정리
        # "내일내일오늘" 같이 같은 글자 반복 → 첫 부분만
        if len(c) >= 4:
            half = len(c) // 2
            # 첫 절반과 다음 일부 같은지
            for split in [2, 3]:
                if len(c) >= split * 2:
                    if c[:split] == c[split:split*2]:
                        # 반복 시작 — 첫 부분만 유지
                        c = c[:split]
                        break
        return c

    # ===== 통계 =====
    
    def stats(self) -> dict:
        return {
            "chains_generated": self.chains_generated,
            "total_nodes": self.total_nodes,
            "avg_chain_length": (self.total_nodes / self.chains_generated
                                  if self.chains_generated > 0 else 0),
            "last_chain": (self._last_chain.categories()
                            if self._last_chain else []),
        }
