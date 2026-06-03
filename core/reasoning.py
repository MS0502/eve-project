"""
ReasoningLoop — GPT 라운드 11 핵심.

입력 → 이해 → [사고] → 공감 → 말하기
              ↑↑↑
          여기 빠진 것

흐름:
1. generate_thought_streams(state) — 여러 사고 흐름 동시 생성
2. evaluate_thoughts(streams) — Metacognition으로 점수
3. best — 최고 점수 thought 하나
4. plan에 thought 포함 → Generator가 sub_point로 표현

ThoughtStream 종류:
- emotional: 감정 기반 ("힘든 거 같이 느껴져")
- causal: 인과 추론 ("PT 때문에 피로 쌓인 거 같아")
- temporal: 시간 추론 ("계속 쌓인 거 같아")
- counterfactual: 만약 ("쉬어가면 풀릴 텐데")
- creative: 비유/유추 ("물 안 주면 시드는 나무처럼")

LLM 0, random 0, 결정론.
"""

from dataclasses import dataclass

from utils.types import Meaning, WorkspaceState


@dataclass
class ThoughtStream:
    type: str          # emotional / causal / temporal / counterfactual / creative
    content: str
    score: float = 0.0


class ReasoningLoop:
    """사고 엔진 — 여러 흐름 생성 + 평가 + 최고 thought 선택."""

    def __init__(self,
                 hormone_adapter=None,
                 deep_reasoning_adapter=None,
                 temporal_adapter=None,
                 counterfactual_adapter=None,
                 metacognition_adapter=None):
        self.hormone_adapter = hormone_adapter
        self.dr = deep_reasoning_adapter
        self.tm = temporal_adapter
        self.cf = counterfactual_adapter
        self.mc = metacognition_adapter

    # ===== 메인 entry =====

    def think(self, state: WorkspaceState) -> ThoughtStream | None:
        """state → 핵심 thought 한 개. 없으면 None."""
        meaning = state.meaning
        if meaning is None:
            return None
        streams = self.generate_thought_streams(meaning)
        if not streams:
            return None
        evaluated = self.evaluate_thoughts(streams, meaning)
        return evaluated[0] if evaluated else None

    # ===== 사고 흐름 생성 =====

    def generate_thought_streams(self, meaning: Meaning) -> list[ThoughtStream]:
        streams: list[ThoughtStream] = []

        # 1) emotional — 가장 강한 감정
        if meaning.emotions:
            top_emo, strength = max(meaning.emotions.items(), key=lambda x: x[1])
            content = self._emotional_thought(top_emo, strength)
            if content:
                streams.append(ThoughtStream("emotional", content, 0.4 + 0.4 * strength))

        # 2) causal — DR이 원인 체인 찾음
        if self.dr is not None and meaning.entities:
            target = meaning.entities[0]
            phrase = self.dr.reason_phrase(target)
            if phrase:
                streams.append(ThoughtStream("causal", phrase, 0.55))

        # 3) temporal — "계속 쌓인" 류 시간 추론
        if self.tm is not None and meaning.entities and meaning.emotions:
            seed = meaning.entities[0]
            episodes = self.tm.tm.recall_past(seed, n=5)
            if isinstance(episodes, list) and len(episodes) >= 2:
                # 같은 카테고리 2번 이상 회상 → 누적 추론
                streams.append(ThoughtStream(
                    "temporal",
                    f"이게 단순한 게 아니라 계속 쌓인 거 같아",
                    0.6,
                ))

        # 4) counterfactual — "만약 쉬어가면" 류
        if self.cf is not None and meaning.emotions:
            top_emo = max(meaning.emotions.items(), key=lambda x: x[1])[0]
            if top_emo in ("fatigue", "anxiety", "anger"):
                streams.append(ThoughtStream(
                    "counterfactual",
                    "잠깐 멈춰가면 좀 풀릴 거 같은데",
                    0.5,
                ))

        return streams

    # ===== thought 평가 (Metacognition) =====

    def evaluate_thoughts(self,
                          streams: list[ThoughtStream],
                          meaning: Meaning) -> list[ThoughtStream]:
        """점수 보정 + 정렬. Metacognition confidence 반영."""
        # 호르몬 기반 보정
        if self.hormone_adapter is not None:
            hs = self.hormone_adapter.hs
            cort = hs.hormones["cortisol"].level
            da = hs.hormones["dopamine"].level
            for s in streams:
                # cort 높으면 emotional 가중
                if s.type == "emotional" and cort > 0.6:
                    s.score *= 1.2
                # DA 높으면 causal/creative 가중
                if s.type in ("causal", "creative") and da > 0.5:
                    s.score *= 1.15
                # cort 낮고 DA 높으면 counterfactual 활발
                if s.type == "counterfactual" and cort < 0.4 and da > 0.5:
                    s.score *= 1.1

        # MC confidence 보정 (있으면)
        if self.mc is not None:
            try:
                conf = self.mc.confidence(meaning, "")
                for s in streams:
                    s.score *= (0.5 + 0.5 * conf)
            except Exception:
                pass

        # 정렬
        streams.sort(key=lambda x: x.score, reverse=True)
        return streams

    # ===== helpers =====

    EMOTIONAL_THOUGHTS: dict[str, str] = {
        "fatigue":  "이게 단순 피로보다 더 깊은 느낌이야",
        "sadness":  "혼자 안고 있는 게 너무 무거워 보여",
        "anger":    "그게 진짜 짜증 났겠다",
        "anxiety":  "걱정이 머릿속을 빙빙 도는 거 같아",
        "joy":      "그래서 더 빛나 보이는 거구나",
        "warmth":   "그 따뜻함이 너한테 잘 와닿네",
    }

    def _emotional_thought(self, emo: str, strength: float) -> str | None:
        if strength < 0.5:
            return None
        return self.EMOTIONAL_THOUGHTS.get(emo)
