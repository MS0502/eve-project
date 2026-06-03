"""
System 1 (직관) - Kahneman 2011

eve2.md 3.4:
- 빠르게 후보 생성
- memory 기반
- 감정 반영

v41 첫 버전: WorkspaceState의 meaning에서 감정/엔티티 보고 후보 생성.
나중에 v40의 spreading_activation으로 후보 확장.
"""

from utils.types import WorkspaceState


# 감정 → 핵심 표현 매핑 (응답 후보의 씨앗)
# 키는 understanding.EMOTION_KEYWORDS의 카테고리와 일치해야 함.
EMOTION_RESPONSE_SEED: dict[str, list[str]] = {
    "fatigue":  ["좀 지친 상태 같네", "에너지 많이 빠진 거 같아"],
    "sadness":  ["많이 가라앉아 있네", "마음 무거운 게 느껴져"],
    "joy":      ["좋아 보이는데", "기분 좋은 거 같네"],
    "anger":    ["좀 답답한 거 있어 보이네", "뭔가 걸린 게 있나봐"],
    "anxiety":  ["뭔가 걱정되는 게 있는 모양이네", "마음이 좀 불안정한 거 같아"],
    "warmth":   ["따뜻하게 들리네", "마음이 와닿네"],
}


class System1:
    """
    후보 응답 생성기. 검증/필터는 System2가 함.

    출력 형식: [{"text": str, "score": float, "source": str}, ...]
    """

    def __init__(self, activation_adapter=None, vsa_adapter=None):
        self.activation_adapter = activation_adapter
        self.vsa_adapter = vsa_adapter

    def generate_candidates(self, state: WorkspaceState) -> list[dict]:
        candidates: list[dict] = []
        meaning = state.meaning
        if meaning is None:
            return candidates

        # 1) 감정 기반 후보 (가장 강한 감정 우선)
        if meaning.emotions:
            top_emotion, strength = max(meaning.emotions.items(), key=lambda x: x[1])
            for seed in EMOTION_RESPONSE_SEED.get(top_emotion, []):
                candidates.append({
                    "text": seed,
                    "score": 0.5 + 0.5 * strength,
                    "source": f"emotion:{top_emotion}",
                })

        # 2) 의도 기반 후보
        if meaning.intent == "question":
            candidates.append({
                "text": "잘 모르겠는데 같이 생각해볼까",
                "score": 0.4,
                "source": "intent:question",
            })

        # 3) 라운드 64: 활성 카테고리 기반 후보 — *맥락 검증*
        # 입력 entity와 *겹치는* 카테고리만 채택 (맥락 무관 "X 생각 떠오르네" 차단)
        if self.activation_adapter is not None:
            entities = set(meaning.entities or [])
            for cat, lvl in self.activation_adapter.top_active(n=5):
                if cat in entities:
                    continue
                if cat in (meaning.emotions or {}):
                    continue
                if lvl < 0.3:
                    continue
                # 라운드 64: is_relevant 검사 — cat이 입력 entities와 *연관* 있어야
                # 그냥 활성된 카테고리라고 채택 X (그게 "민석 생각" / "EVE 생각" 병의 원인)
                if not self._is_relevant(cat, entities):
                    continue
                candidates.append({
                    "text": f"{cat} 얘기 같이 들을게",
                    "score": 0.3 + 0.4 * lvl,
                    "source": f"activation:{cat}",
                })

        # 3.5) VSA 합성 후보 (어댑터 있을 때)
        if self.vsa_adapter is not None:
            candidates.extend(self.vsa_adapter.expand_candidates(meaning))

        # 4) 라운드 64: fallback → minimal_ack (맥락 무시 X)
        # 이전: "X 얘기 듣고 있어" — 자동 entity 강제. 어색함.
        # 지금: 짧은 인정만 — "그렇구나" 같은 사람 반응
        if not candidates:
            # seed: 입력 텍스트 hash로 결정론
            text = (meaning.raw_text or "")
            seed = sum(ord(c) for c in text) if text else 0
            ack_pool = ["아 그렇구나", "음 그렇네", "오", "그래?", "응 그렇구나"]
            candidates.append({
                "text": ack_pool[seed % len(ack_pool)],
                "score": 0.25,
                "source": "fallback:minimal_ack",
            })

        return candidates

    def _is_relevant(self, cat: str, entities: set) -> bool:
        """라운드 64: 카테고리가 입력 entity와 의미상 연관 있나.
        
        - cat이 입력 entity의 부분 문자열이거나
        - 입력 entity가 cat의 부분 문자열이면 연관
        - 일반 사용자 지칭 ("나", "민석", "EVE", "이브") 단독은 *연관 X*
          → 사용자가 *그것에 대해* 묻거나 말한 게 아니면 응답에 끌어들이지 않음
        """
        # 일반 사용자/자기 지칭은 entity와 자동 연관 X
        SELF_USER_REFS = {"민석", "EVE", "이브", "나", "너"}
        if cat in SELF_USER_REFS:
            # entity에 해당 ref가 *명시적으로* 있을 때만 연관
            return cat in entities
        # 일반 카테고리 — entity와 substring 연관 검사
        for e in entities:
            if cat in e or e in cat:
                return True
        return False
