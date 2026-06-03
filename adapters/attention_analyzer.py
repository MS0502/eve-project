"""
AttentionAnalyzer — 라운드 48

EVE 3-Layer 구조의 Layer 1 — 어텐션 통합 모듈.

원칙 (GPT 정리):
- 어텐션 = "어디 집중할지" 결정만
- 의미 이해 X (심볼릭만)
- 결정론 (softmax X, random X, 가중 합산만)

학계:
- Itti & Koch 2001 — Saliency Map (visual attention 가중 합산)
- Treisman & Gelade 1980 — Feature Integration Theory
- Posner 1980 — Attention orienting (top-down + bottom-up)

통합 점수:
attention_score(entity) = 
    W_SAL * salience[entity] +
    W_REC * recency[entity] +      ← 새로움
    W_EMO * emotion_match[entity] +
    W_KW  * keyword_boost[entity] +
    W_EMB * embedding_relevance[entity]   ← Layer 2 연결

출력:
- top_entities: 우선순위 점수
- primary_node: ThoughtChain/Planner 시드
- dominant_emotion: 감정 톤
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# === 가중치 (학계 saliency 보통 균등) ===
W_SALIENCE = 0.30        # Salience adapter 값
W_RECENCY = 0.20         # 최근 등장
W_EMOTION = 0.20         # 현재 감정과 매치
W_KEYWORD = 0.20         # 강한 키워드 (민석/EVE/감정어)
W_EMBEDDING = 0.10       # 임베딩 관련성 (Layer 2)

# v3 round35: fastText parallel observation is trace-only.
ATTENTION_FASTTEXT_TRACE_CAP = 1000
ATTENTION_FASTTEXT_OBSERVATION_LAYER = "attention"
ATTENTION_GENERATION_EMBEDDING_SOURCE = "wrapper"


# === 강한 키워드 (Cocktail Party Effect) ===
STRONG_KEYWORDS = {
    "민석", "민석아", "민석이", "EVE", "이브",
    "사랑", "보고싶", "힘들", "외로", "기쁘", "슬프",
}

# 감정 단어 매핑
EMOTION_KEYWORDS = {
    "fatigue": ["힘들", "지쳐", "지쳤", "피곤", "졸려"],
    "joy": ["기쁘", "행복", "좋아", "신나", "뿌듯"],
    "sadness": ["슬프", "외로", "우울", "괴로"],
    "anger": ["화나", "짜증", "싫어", "답답"],
    "fear": ["무서워", "불안", "두려"],
}


@dataclass
class AttentionResult:
    """어텐션 분석 결과."""
    scores: dict = field(default_factory=dict)        # entity → combined score
    top_entities: list = field(default_factory=list)  # [(entity, score), ...]
    primary_node: Optional[str] = None                 # 1순위 시드
    dominant_emotion: Optional[str] = None
    urgency: float = 0.0                               # 0~1


class AttentionAnalyzer:
    """
    EVE Layer 1 — 통합 어텐션.
    
    각 entity의 통합 중요도 점수 계산.
    심볼릭 모듈 (planner, ThoughtChain)에게 *어디 집중할지* 알려줌.
    의미 이해 X (그건 심볼릭이).
    """

    def __init__(self, engine=None,
                 recency_tau: float = 60.0):
        self.engine = engine
        self.recency_tau = recency_tau   # recency decay (초)
        # entity별 마지막 등장 시간 (recency 계산용)
        self.last_seen: dict = {}
        # 통계
        self.analyze_count = 0
        # v3 round35: fastText migration observation only.
        # This trace must never affect attention decisions.
        self.fasttext_trace: list[dict] = []
        self.fasttext_observation_count = 0
        self.fasttext_trace_cap = ATTENTION_FASTTEXT_TRACE_CAP

    def analyze(self, text: str, entities: list,
                emotions: dict = None) -> AttentionResult:
        """
        주어진 입력 → 각 entity의 통합 어텐션 점수.
        
        Args:
            text: 원본 텍스트
            entities: 의미 entity 리스트
            emotions: 추출된 감정 dict {emotion_name: intensity}
        """
        result = AttentionResult()
        self.analyze_count += 1
        now = time.time()
        
        # 각 entity 점수 계산 (결정론, 가중 합산)
        # 라운드 53: 의문 어절 / 합성 키 / stopword 거름 (last_seen에 저장 X)
        BAD_SUFFIXES = ('했어', '하고', '있었어', '있어', '거다', '거야', 
                        '인지', '냐', '느냐', '었는다', '했는다', '있는거',
                        '해봐', '해줘', '말해', '말해봐', '해라', '하자')
        BAD_WORDS = ("뭐", "왜", "뭔", "어떻게", "언제", "어디", "누구",
                     "그", "이", "저", "음", "그냥", "어떤", "인사해봐",
                     "살아있어?", "살아있는거같아?", "안녕만해봐", "사는거다",
                     # 라운드 55
                     "뭐해", "뭐하", "뭔지알아", "있어", "없어", "그래",
                     "맞아", "응응", "아니")
        
        clean_entities = []
        for ent in entities:
            if not isinstance(ent, str) or len(ent) < 1:
                continue
            if any(c in ent for c in '+@_'):
                continue
            if ent.endswith('?') or ent.endswith('？'):
                continue
            if ent in BAD_WORDS:
                continue
            if any(ent.endswith(suf) for suf in BAD_SUFFIXES):
                continue
            # 라운드 54: 너무 긴 어절 (사용자 합성)
            if len(ent) > 12:
                continue
            clean_entities.append(ent)
        
        # last_seen 업데이트 (clean된 것만)
        for ent in clean_entities:
            self.last_seen[ent] = now
        
        if not clean_entities:
            return result
        
        # 현재 emotion (최강)
        current_emotion = None
        if emotions:
            current_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            result.dominant_emotion = current_emotion
        
        scores = {}
        for ent in clean_entities:
            sal = self._salience_score(ent)
            rec = self._recency_score(ent, now)
            emo = self._emotion_match(ent, current_emotion, text)
            kw = self._keyword_boost(ent, text)
            emb = self._embedding_relevance(ent)
            
            combined = (
                W_SALIENCE * sal +
                W_RECENCY * rec +
                W_EMOTION * emo +
                W_KEYWORD * kw +
                W_EMBEDDING * emb
            )
            scores[ent] = combined
        
        # 정렬 (결정론 — 점수 내림, alphabet tiebreak)
        sorted_ents = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        result.scores = scores
        result.top_entities = sorted_ents
        if sorted_ents and sorted_ents[0][1] > 0.0:
            result.primary_node = sorted_ents[0][0]
        
        # urgency = top score 기준
        if sorted_ents:
            result.urgency = min(1.0, sorted_ents[0][1])

        # v3 round35: run fastText in parallel observation mode only.
        # The returned result above is not modified by this trace path.
        self._observe_fasttext_parallel(text, clean_entities, result)
        return result

    # ===== v3 round35 fastText observation helpers =====

    def _fasttext_adapter_loaded(self):
        if self.engine is None:
            return None
        adapter = getattr(self.engine, "fasttext_embedding", None)
        if adapter is None or not hasattr(adapter, "is_loaded"):
            return None
        try:
            if not adapter.is_loaded():
                return None
        except Exception:
            return None
        return adapter

    def _observe_fasttext_parallel(self, text: str, entities: list, result: AttentionResult) -> None:
        """Record fastText attention signals without changing decisions.

        This is the embedding-migration equivalent of AGP observation mode:
        explicit loaded fastText adapters can produce diagnostics, but the
        attention result still comes from the existing PMI+SVD path.
        """
        adapter = self._fasttext_adapter_loaded()
        if adapter is None:
            return
        try:
            ft_scores = {}
            for ent in entities:
                # text_similarity uses the adapter's deterministic whitespace
                # token averaging. OOV entities deterministically score 0.0.
                score = float(adapter.text_similarity(str(ent), str(text or "")))
                ft_scores[str(ent)] = max(0.0, score)
            ft_top = sorted(ft_scores.items(), key=lambda item: (-item[1], item[0]))
            trace = {
                "meaning_layer": ATTENTION_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "in_use_by_generation": ATTENTION_GENERATION_EMBEDDING_SOURCE,
                "self_embedding_result": {
                    "primary_node": result.primary_node,
                    "top_entities": list(result.top_entities),
                    "scores": dict(result.scores),
                    "urgency": float(result.urgency),
                },
                "fasttext_result": {
                    "subset_name": getattr(adapter, "subset_name", None),
                    "loaded": True,
                    "scores": ft_scores,
                    "top_entities": ft_top,
                    "primary_node": ft_top[0][0] if ft_top and ft_top[0][1] > 0.0 else None,
                },
            }
            self._record_fasttext_trace(trace)
        except Exception as exc:  # fail-open for observation-only diagnostics
            self._record_fasttext_trace({
                "meaning_layer": ATTENTION_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "error": str(exc),
                "in_use_by_generation": ATTENTION_GENERATION_EMBEDDING_SOURCE,
            })

    def _record_fasttext_trace(self, trace: dict) -> None:
        self.fasttext_observation_count += 1
        self.fasttext_trace.append(dict(trace))
        if len(self.fasttext_trace) > self.fasttext_trace_cap:
            del self.fasttext_trace[: len(self.fasttext_trace) - self.fasttext_trace_cap]

    # ===== 각 점수 함수 =====

    def _salience_score(self, ent: str) -> float:
        """Salience adapter 값 정규화 (0~1)."""
        if self.engine is None or not hasattr(self.engine, "salience_adapter"):
            return 0.0
        sal = self.engine.salience_adapter
        if sal is None:
            return 0.0
        v = sal.get(ent)
        # ceiling 5.0 → 1.0 정규화
        return min(1.0, v / 5.0)

    def _recency_score(self, ent: str, now: float) -> float:
        """exp 형태 decay — 최근일수록 1, 오래될수록 0."""
        last = self.last_seen.get(ent)
        if last is None:
            return 0.0
        elapsed = now - last
        if elapsed < 0:
            return 0.0
        # 지수 감쇠 (학계 표준)
        import math
        return math.exp(-elapsed / self.recency_tau)

    def _emotion_match(self, ent: str, current_emotion: Optional[str],
                       text: str) -> float:
        """entity가 현재 감정과 관련 있나?"""
        if current_emotion is None:
            return 0.0
        # entity 자체가 감정 단어
        for emo, kws in EMOTION_KEYWORDS.items():
            if any(kw in ent for kw in kws):
                # 현재 감정과 일치하면 더 강
                return 1.0 if emo == current_emotion else 0.5
        # entity가 emotion 키워드 일부 포함
        emo_kws = EMOTION_KEYWORDS.get(current_emotion, [])
        for kw in emo_kws:
            if kw in ent or ent in kw:
                return 0.7
        return 0.0

    def _keyword_boost(self, ent: str, text: str) -> float:
        """강한 키워드 — Cocktail Party Effect."""
        if ent in STRONG_KEYWORDS:
            return 1.0
        # 부분 매칭 (예: "민석이" → "민석" 포함)
        for kw in STRONG_KEYWORDS:
            if kw in ent or ent in kw:
                return 0.7
        return 0.0

    def _embedding_relevance(self, ent: str) -> float:
        """Layer 2 — 임베딩으로 *현재 활성 카테고리*와 관련성."""
        if self.engine is None or not hasattr(self.engine, "self_embedding"):
            return 0.0
        emb = self.engine.self_embedding
        if emb is None or ent not in emb.embeddings:
            return 0.0
        # Salience top과의 평균 유사도
        if not hasattr(self.engine, "salience_adapter"):
            return 0.0
        top = self.engine.salience_adapter.top(3)
        if not top:
            return 0.0
        sims = []
        for cat, _ in top:
            if cat == ent:
                continue
            s = emb.similarity(ent, cat)
            if s is not None:
                sims.append(max(0.0, s))   # 음수 무시
        if not sims:
            return 0.0
        return sum(sims) / len(sims)

    def get_current_focus(self, rotation: int = 0) -> Optional[str]:
        """라운드 48+53: 입력 없는 시점 (자율 발화) 어텐션 시드.
        
        rotation: 0=top1, 1=top2, ... (반복 발화 시 다른 시드 가능)
        last_seen 추적 entity + Salience top → 결정론적 선택.
        의문 어절 / 합성 키 / 동사 활용형 거름.
        """
        now = time.time()
        candidates = []
        
        BAD_SUFFIXES = ('했어', '하고', '있었어', '있어', '거다', '거야', 
                        '인지', '냐', '느냐', '었는다', '했는다', '있는거',
                        '해봐', '해줘', '말해', '말해봐', '해라', '하자')
        BAD_WORDS = ("뭐", "왜", "뭔", "어떻게", "언제", "어디", "누구",
                     "그", "이", "저", "음", "그냥", "안녕만해봐", "사는거다",
                     "뭐해", "뭐하", "있어", "없어", "그래", "맞아", "아니")
        
        def is_clean(ent):
            if not ent:
                return False
            if "+" in ent or "_" in ent or "@" in ent:
                return False
            if ent in BAD_WORDS:
                return False
            if any(ent.endswith(suf) for suf in BAD_SUFFIXES):
                return False
            if len(ent) > 12:
                return False
            return True
        
        # 1. 최근 등장 entity (recency 기반)
        for ent, last in self.last_seen.items():
            if not is_clean(ent):
                continue
            rec = self._recency_score(ent, now)
            sal = self._salience_score(ent)
            kw = self._keyword_boost(ent, "")
            score = (W_RECENCY * rec + W_SALIENCE * sal + W_KEYWORD * kw)
            if score > 0.0:
                candidates.append((ent, score))
        
        # 2. Salience top
        if self.engine and hasattr(self.engine, "salience_adapter"):
            for cat, val in self.engine.salience_adapter.top(10):
                if not is_clean(cat):
                    continue
                if cat in [c for c, _ in candidates]:
                    continue
                score = W_SALIENCE * min(1.0, val / 5.0)
                if score > 0.0:
                    candidates.append((cat, score))
        
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[1], x[0]))
        idx = rotation % len(candidates)
        return candidates[idx][0]

    # ===== 통계 =====

    def stats(self) -> dict:
        return {
            "analyze_count": self.analyze_count,
            "tracked_entities": len(self.last_seen),
            "fasttext_parallel_observation": "fasttext" if self._fasttext_adapter_loaded() is not None else None,
            "fasttext_trace_size": len(self.fasttext_trace),
            "fasttext_observation_count": self.fasttext_observation_count,
            "fasttext_trace_cap": self.fasttext_trace_cap,
            "in_use_by_generation": ATTENTION_GENERATION_EMBEDDING_SOURCE,
        }
