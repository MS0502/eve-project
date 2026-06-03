"""
ConceptMemoryAdapter — 라운드 43

GPT 짚은 결정타: 지금 EVE는 "느낌은 사람, 지식은 아기"
- teaching_adapter = 반응 학습 (X 입력 → Y 응답)
- ConceptMemory = 정의 학습 (X = "뭐다")

학계:
- Tulving 1972 — Semantic Memory (의미 기억)
- Quillian 1968 — semantic networks
- Collins & Quillian 1969 — concept hierarchy

원칙:
- 사용자가 가르친 정의 우선 사용 (개인화)
- 가르침 없으면 SA 기반 연상 (지금 방식)
- 둘 다 있으면 정의 + 연상 (이중 응답 — GPT 추천)
"""

import time
from dataclasses import dataclass
from typing import Optional


# v3 round37: fastText migration observation is trace-only.
CONCEPT_MEMORY_FASTTEXT_TRACE_CAP = 1000
CONCEPT_MEMORY_FASTTEXT_OBSERVATION_LAYER = "concept_memory"
CONCEPT_MEMORY_GENERATION_EMBEDDING_SOURCE = "wrapper"
CONCEPT_MEMORY_OPERATION_QUERY = "query"
CONCEPT_MEMORY_OPERATION_WRITE = "write"
CONCEPT_MEMORY_OPERATION_QUARANTINE = "quarantine"


@dataclass
class Concept:
    key: str            # "PT" / "뭐해"
    definition: str     # "운동 도와주는 트레이닝"
    learned_at: float
    source: str = "user"   # "user" / "self_inferred"
    times_used: int = 0
    related: list = None   # 정의에서 추출한 관련 키워드 (라운드 45)
    
    def __post_init__(self):
        if self.related is None:
            self.related = []


class ConceptMemoryAdapter:
    """
    개념 정의 저장 + 조회.
    "X는 Y야" 가르침 → store. "X가 뭐야?" 질문 → retrieve.
    """

    def __init__(self, engine=None):
        self.engine = engine
        self.concepts: dict = {}   # key (lowercase) → Concept
        self.learn_count = 0
        self.use_count = 0
        # v3 round37: fastText parallel observation only.
        # This must never affect query/write decisions or memory contents.
        self.fasttext_trace: list[dict] = []
        self.fasttext_observation_count = 0
        self.fasttext_trace_cap = CONCEPT_MEMORY_FASTTEXT_TRACE_CAP
        self.fasttext_operation_counts = {
            CONCEPT_MEMORY_OPERATION_QUERY: 0,
            CONCEPT_MEMORY_OPERATION_WRITE: 0,
            CONCEPT_MEMORY_OPERATION_QUARANTINE: 0,
        }

    def learn(self, key: str, definition: str, source: str = "user") -> Concept:
        """개념 정의 저장 + 라운드 45: related 키워드 추출 + SA 연결."""
        if not key or not definition:
            return None
        key = key.strip()
        definition = definition.strip()
        if len(key) < 1 or len(definition) < 2:
            return None
        
        # related 키워드 추출 (라운드 45)
        try:
            from utils.speech_style import extract_keywords
            related = extract_keywords(definition, max_n=5)
        except Exception:
            related = []
        
        c = Concept(
            key=key,
            definition=definition,
            learned_at=time.time(),
            source=source,
            related=related,
        )
        self.concepts[key] = c
        self.learn_count += 1

        # SelfEmbedding은 build 때 학습하지 않고 lazy 준비한다. 새 개념은
        # 즉시 관찰 버퍼에 넣어두되, SVD train은 소비 경로가 필요할 때만 실행한다.
        if self.engine is not None and hasattr(self.engine, "self_embedding"):
            emb = self.engine.self_embedding
            if emb is not None and hasattr(emb, "observe"):
                try:
                    synthetic = f"{key} {definition}"
                    old_suspend = getattr(emb, "_suspend_auto_train", False)
                    emb._suspend_auto_train = True
                    try:
                        emb.observe(synthetic)
                        emb.observe(synthetic)  # min_word_freq=2 충족용, 결정론적 반복
                    finally:
                        emb._suspend_auto_train = old_suspend
                    emb._has_seeded_concepts = True
                except Exception:
                    pass
        
        # SA 가중치에 키워드 연결 추가 (ThoughtChain이 이걸 활용)
        # 라운드 46: 합성 키 / 부적절 카테고리 제외
        if self.engine is not None and self.engine.activation_adapter is not None:
            sa = self.engine.activation_adapter.sa
            if hasattr(sa, "learn_pair"):
                # key/related 둘 다 cleanup — 합성 키 / 의문 부호 / 언더스코어 제외
                key_clean = key.split("+")[0].split("@")[0].rstrip("?？.")
                if "_" in key_clean:
                    key_clean = key_clean.split("_")[0]
                if not key_clean or len(key_clean) < 1:
                    return c
                
                for r in related:
                    r_clean = r.split("+")[0].split("@")[0].rstrip("?？.")
                    if "_" in r_clean:
                        r_clean = r_clean.split("_")[0]
                    if not r_clean or len(r_clean) < 2:
                        continue
                    if r_clean == key_clean:
                        continue
                    try:
                        sa.learn_pair(key_clean, r_clean, strength=0.5)
                    except Exception:
                        pass
        
        self._observe_fasttext_parallel(
            operation=CONCEPT_MEMORY_OPERATION_WRITE,
            text=f"{key} {definition}",
            key=key,
            self_embedding_result={
                "stored": True,
                "key": key,
                "definition": definition,
                "related": list(related),
                "learn_count": int(self.learn_count),
            },
        )
        return c

    def get(self, key: str) -> Optional[Concept]:
        """개념 조회 (사용 카운트 증가)."""
        if not key:
            return None
        key = key.strip()
        c = self.concepts.get(key)
        if c is not None:
            c.times_used += 1
            self.use_count += 1
        self._observe_fasttext_parallel(
            operation=CONCEPT_MEMORY_OPERATION_QUERY,
            text=key,
            key=key,
            self_embedding_result={
                "found": c is not None,
                "key": key,
                "definition": None if c is None else c.definition,
                "use_count": int(self.use_count),
            },
        )
        return c

    def has(self, key: str) -> bool:
        return self.get_silent(key) is not None

    def get_silent(self, key: str) -> Optional[Concept]:
        """조회만 (use_count 안 늘림)."""
        if not key:
            return None
        return self.concepts.get(key.strip())

    def all_keys(self) -> list:
        return list(self.concepts.keys())

    # ===== v3 round37 fastText observation helpers =====

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

    def _observe_fasttext_parallel(self, operation: str, text: str, key: str = None,
                                   self_embedding_result: dict = None) -> None:
        """Record fastText concept-memory diagnostics without changing decisions.

        Concept memory is a semantic write/query module. Round37 therefore keeps
        the existing PMI+SVD/self_embedding path as the only active path and
        records fastText signals only when the adapter has been explicitly
        loaded by tests/operators. No trace data is fed back into storage,
        quarantine, generation, or retrieval decisions.
        """
        adapter = self._fasttext_adapter_loaded()
        if adapter is None:
            return
        op = operation if operation in self.fasttext_operation_counts else str(operation or "unknown")
        try:
            key_text = str(key or "")
            text_value = str(text or "")
            key_known = False
            key_norm = 0.0
            key_text_similarity = 0.0
            vector = None
            if key_text and hasattr(adapter, "get_vector"):
                vector = adapter.get_vector(key_text)
            if vector is not None:
                key_known = True
                try:
                    import numpy as np
                    key_norm = float(np.linalg.norm(vector))
                except Exception:
                    key_norm = 0.0
            if key_text and text_value and hasattr(adapter, "text_similarity"):
                key_text_similarity = float(adapter.text_similarity(key_text, text_value))
            trace = {
                "meaning_layer": CONCEPT_MEMORY_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "operation": op,
                "in_use_by_generation": CONCEPT_MEMORY_GENERATION_EMBEDDING_SOURCE,
                "self_embedding_result": dict(self_embedding_result or {}),
                "fasttext_result": {
                    "subset_name": getattr(adapter, "subset_name", None),
                    "loaded": True,
                    "key": key_text,
                    "key_known": bool(key_known),
                    "key_vector_norm": key_norm,
                    "key_text_similarity": key_text_similarity,
                },
            }
            self._record_fasttext_trace(trace)
        except Exception as exc:  # fail-open for observation-only diagnostics
            self._record_fasttext_trace({
                "meaning_layer": CONCEPT_MEMORY_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "operation": op,
                "error": str(exc),
                "in_use_by_generation": CONCEPT_MEMORY_GENERATION_EMBEDDING_SOURCE,
            })

    def _record_fasttext_trace(self, trace: dict) -> None:
        self.fasttext_observation_count += 1
        operation = str(trace.get("operation") or "unknown")
        if operation not in self.fasttext_operation_counts:
            self.fasttext_operation_counts[operation] = 0
        self.fasttext_operation_counts[operation] += 1
        self.fasttext_trace.append(dict(trace))
        if len(self.fasttext_trace) > self.fasttext_trace_cap:
            del self.fasttext_trace[: len(self.fasttext_trace) - self.fasttext_trace_cap]

    def stats(self) -> dict:
        return {
            "concepts_count": len(self.concepts),
            "learn_count": self.learn_count,
            "use_count": self.use_count,
            "recent_5": [
                (k, c.definition) for k, c in
                sorted(self.concepts.items(),
                       key=lambda kv: -kv[1].learned_at)[:5]
            ],
            "fasttext_parallel_observation": "fasttext" if self._fasttext_adapter_loaded() is not None else None,
            "fasttext_trace_size": len(self.fasttext_trace),
            "fasttext_observation_count": self.fasttext_observation_count,
            "fasttext_trace_cap": self.fasttext_trace_cap,
            "operations_observed": dict(self.fasttext_operation_counts),
            "in_use_by_generation": CONCEPT_MEMORY_GENERATION_EMBEDDING_SOURCE,
        }

    # ===== 가르침 입력에서 정의 추출 =====

    @staticmethod
    def extract_definition(text: str) -> Optional[tuple]:
        """
        "X는 Y야" / "X 뜻은 Y야" / "X는 ~하는 거야" 패턴에서 (X, Y) 추출.
        Returns: (key, definition) 또는 None.
        """
        import re
        text = (text or "").strip()
        if not text:
            return None
        # 의문문은 제외
        if "?" in text or "?" in text:
            return None
        
        patterns = [
            # "X 뜻은 Y야" / "X 의미는 Y야"
            re.compile(r'^(.{1,20}?)\s*(?:뜻|의미)(?:은|는)\s+(.{2,80}?)(?:야|이야|이다|다)\s*\.?\s*$'),
            # "X는 ~하는 말이야" / "X는 ~하는 거야" — 동사 어미 "는" 보존
            re.compile(r'^(.{1,20}?)(?:은|는|란|이란)\s+(.+(?:는|ㄴ|은))\s*(?:말|거|것|건)(?:야|이야|이다)\s*\.?\s*$'),
            # "X는 Y야" — 일반 정의
            re.compile(r'^(.{1,20}?)(?:은|는|란|이란)\s+(.{2,80}?)(?:야|이야|이다)\s*\.?\s*$'),
        ]
        
        for pat in patterns:
            m = pat.match(text)
            if m:
                key = m.group(1).strip()
                definition = m.group(2).strip()
                # key가 너무 짧거나 일반 대명사면 X
                if key in ("나", "내", "너", "니", "그", "이", "저"):
                    return None
                if len(key) < 1:
                    continue
                return (key, definition)
        
        # 라운드 54: 역방향 정의 "안녕이 인사야" — A이/가 B야 (A가 인스턴스, B가 카테고리)
        # 이런 경우 key=B (카테고리), definition=A (예시)
        # 근데 사용자는 *A를 정의*하려는 의도 — A 정의 = "B의 한 종류"
        reverse = re.match(
            r'^([\w가-힣]{1,15})(?:이|가)\s+([\w가-힣]{1,15})(?:야|이야)\s*\.?\s*$',
            text
        )
        if reverse:
            a = reverse.group(1).strip()
            b = reverse.group(2).strip()
            if a not in ("나", "내", "너", "니") and b not in ("나", "내", "너", "니"):
                # "안녕이 인사야" → key="안녕", definition="인사의 한 종류"
                return (a, f"{b}의 한 종류")
        
        return None

