"""
CompositorAdapter — 라운드 49

EVE 3-Layer 구조의 Layer 2.5 — Compositional Reasoning.

학계 토대:
- Mao, Tenenbaum, Wu 2025 — Neuro-Symbolic Concepts (MIT/Stanford)
  "개념은 합성 가능 — 구조적 결합으로 새 개념 생성"
- Margolis & Laurence 1999 — 개념을 사고의 기본 단위로 (인지과학)
- NeSyCoCo 2024 — soft composition of predicates
- Fodor & Pylyshyn 1988 — Systematicity argument (생각의 합성성)

원칙 (GPT 정리):
- Compositor = "여러 의미 → 새 의미" 생성
- 의미 *이해*는 X (그건 심볼릭만)
- "아이디어 생성기" 역할
- 결과를 심볼릭이 검증 + 출력

방법 3가지 결합:
1. 임베딩 산술 (word2vec 스타일) — vec(A) + vec(B) → nearest word
2. PMI co-composition — A, B 공통 이웃 (둘 다와 강한 SA 가중치)
3. 룰 기반 fallback — 단순 사전

[Attention] → [Embedding] → [Compositor] → [Symbolic] → [Style]
                                ↑
                            라운드 49
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from adapters.agp_adapter import AGP_MODE_OBSERVATION, AGP_MODE_VETO


# v3 round36: fastText migration observation is trace-only.
COMPOSITOR_FASTTEXT_TRACE_CAP = 1000
COMPOSITOR_FASTTEXT_OBSERVATION_LAYER = "compositor"
COMPOSITOR_GENERATION_EMBEDDING_SOURCE = "wrapper"


# === 룰 기반 fallback (시작점) ===
COMPOSITION_RULES = {
    # (a, b) → [composed concepts]
    ("운동", "힘들"): ["성장", "버티는 과정"],
    ("운동", "피곤"): ["회복 필요", "쉼"],
    ("PT", "힘들"): ["성장", "버티는 과정"],
    ("보고싶", "이브"): ["애착", "그리움"],
    ("보고싶", "민석"): ["애착", "그리움"],
    ("피곤", "오늘"): ["과로", "휴식"],
    ("힘들", "공부"): ["성장", "노력"],
    ("외로", "혼자"): ["고독", "연결 필요"],
    ("기쁘", "성공"): ["성취", "보람"],
    ("슬프", "이별"): ["상실", "그리움"],
}

# === 합성 결과 데이터 클래스 ===

@dataclass
class CompositionResult:
    """합성 의미 결과."""
    inputs: tuple = ()             # (a, b)
    concepts: list = field(default_factory=list)  # 합성된 개념들
    method: str = "none"           # "embedding" / "pmi" / "rule"
    confidence: float = 0.0        # 0~1


class CompositorAdapter:
    """
    EVE Layer 2.5 — Compositional Reasoning.
    
    두 활성 카테고리 → 새 의미 합성.
    심볼릭이 *검증* 후 응답에 활용.
    """

    def __init__(self, engine=None,
                 min_embed_sim: float = 0.2,
                 min_pmi_strength: float = 0.3):
        self.engine = engine
        self.min_embed_sim = min_embed_sim
        self.min_pmi_strength = min_pmi_strength
        # 통계
        self.compose_count = 0
        self.method_counts = defaultdict(int)
        # 자체 누적 룰 (사용자 사용 누적)
        self.learned_compositions: dict = {}   # (a, b) → [concepts]
        # v3 round8-14: AGP observation/veto trace.
        # Default must remain observation; compositor veto is explicit only.
        self.agp_mode = AGP_MODE_OBSERVATION
        self.agp_trace_limit = 1000
        self.agp_trace: list[dict[str, Any]] = []
        self.agp_observation_count = 0
        self.agp_veto_count = 0
        # v3 round36: fastText parallel observation trace.
        # This must never affect composition output or AGP veto behavior.
        self.fasttext_trace: list[dict[str, Any]] = []
        self.fasttext_observation_count = 0
        self.fasttext_trace_cap = COMPOSITOR_FASTTEXT_TRACE_CAP

    # ===== 메인 API =====

    def compose(self, a: str, b: str,
                top_n: int = 3) -> Optional[CompositionResult]:
        """
        두 카테고리 → 합성 의미.
        
        순서: 임베딩 산술 → PMI co-composition → 룰 fallback → 학습된 룰
        """
        if not a or not b or a == b:
            return None
        
        a, b = self._normalize(a), self._normalize(b)
        if not a or not b:
            return None
        
        # 1. 임베딩 산술 (가장 학계 정당, 자체 학습)
        result = self._compose_embedding(a, b, top_n)
        if result is not None:
            self.method_counts["embedding"] += 1
            self.compose_count += 1
            return result
        
        # 2. PMI co-composition (SA 가중치 활용)
        result = self._compose_pmi(a, b, top_n)
        if result is not None:
            self.method_counts["pmi"] += 1
            self.compose_count += 1
            return result
        
        # 3. 학습된 룰 (사용자 누적)
        learned = self._compose_learned(a, b)
        if learned is not None:
            self.method_counts["learned"] += 1
            self.compose_count += 1
            return learned
        
        # 4. 룰 fallback (사전)
        rule_result = self._compose_rule(a, b)
        if rule_result is not None:
            self.method_counts["rule"] += 1
            self.compose_count += 1
            return rule_result
        
        return None

    # ===== 방법 1: 임베딩 산술 =====

    def _compose_embedding(self, a: str, b: str, top_n: int):
        """
        word2vec 스타일: vec(a) + vec(b) → 가장 가까운 단어들.
        학계: Mikolov 2013 "king - man + woman = queen"의 단순화.
        """
        if self.engine is None or not hasattr(self.engine, "self_embedding"):
            return None
        emb = self.engine.self_embedding
        if emb is None:
            return None
        # Hot path 보호: 일반 chat 중 빈 임베딩이면 SVD 부트스트랩을 돌리지 않는다.
        # 사용자가 ConceptMemory 등으로 실제 관찰 데이터를 넣은 뒤에만 lazy 준비한다.
        if (not getattr(emb, "embeddings", None)
            and getattr(emb, "_has_seeded_concepts", False)
            and getattr(emb, "word_counts", None)):
            if a in emb.word_counts or b in emb.word_counts:
                if hasattr(emb, "ensure_ready"):
                    emb.ensure_ready()
        
        va = emb.get_embedding(a)
        vb = emb.get_embedding(b)
        if va is None or vb is None:
            return None
        
        try:
            import numpy as np
        except ImportError:
            return None
        
        # 합성 벡터 (산술 평균)
        composed = (va + vb) / 2.0
        norm_c = np.linalg.norm(composed)
        if norm_c < 1e-8:
            return None
        composed = composed / norm_c
        
        # 모든 단어와 cosine similarity (a, b 자체 제외)
        sims = []
        for word, vec in emb.embeddings.items():
            if word in (a, b):
                continue
            n = np.linalg.norm(vec)
            if n < 1e-8:
                continue
            sim = float(np.dot(composed, vec) / n)
            if sim > self.min_embed_sim:
                sims.append((word, sim))
        
        if not sims:
            return None
        
        # 결정론 — score 내림 + alphabet
        sims.sort(key=lambda x: (-x[1], x[0]))
        concepts = [w for w, _ in sims[:top_n]]
        confidence = sims[0][1] if sims else 0.0
        
        return CompositionResult(
            inputs=(a, b),
            concepts=concepts,
            method="embedding",
            confidence=confidence,
        )

    # ===== 방법 2: PMI co-composition =====

    def _compose_pmi(self, a: str, b: str, top_n: int):
        """
        a와 b 둘 다와 강한 SA 가중치를 가진 카테고리들.
        (a → X, b → X) 둘 다 강하면 X가 a+b 합성 의미.
        """
        if self.engine is None or self.engine.activation_adapter is None:
            return None
        sa = self.engine.activation_adapter.sa
        if not hasattr(sa, "weights"):
            return None
        
        # a의 이웃들
        a_neighbors = {}
        # b의 이웃들
        b_neighbors = {}
        for (x, y), w in sa.weights.items():
            if w < self.min_pmi_strength:
                continue
            if any(ch in x + y for ch in '+@_'):
                continue
            if x == a:
                a_neighbors[y] = w
            elif y == a:
                a_neighbors[x] = w
            if x == b:
                b_neighbors[y] = w
            elif y == b:
                b_neighbors[x] = w
        
        # 공통 이웃 — 둘 다와 연결
        common = set(a_neighbors.keys()) & set(b_neighbors.keys())
        common.discard(a)
        common.discard(b)
        
        if not common:
            return None
        
        # 공통 강도 = a-X 가중치 * b-X 가중치
        scored = []
        for c in common:
            joint = a_neighbors[c] * b_neighbors[c]
            scored.append((c, joint))
        
        scored.sort(key=lambda x: (-x[1], x[0]))
        concepts = [c for c, _ in scored[:top_n]]
        confidence = scored[0][1] if scored else 0.0
        
        return CompositionResult(
            inputs=(a, b),
            concepts=concepts,
            method="pmi",
            confidence=confidence,
        )

    # ===== 방법 3: 학습된 룰 (사용자 누적) =====

    def _compose_learned(self, a: str, b: str):
        """사용자가 가르친 합성 (수동 학습)."""
        # 정렬된 키로 조회 (대칭)
        key = tuple(sorted([a, b]))
        if key in self.learned_compositions:
            concepts = self.learned_compositions[key]
            return CompositionResult(
                inputs=(a, b),
                concepts=list(concepts),
                method="learned",
                confidence=0.9,
            )
        return None

    # ===== 방법 4: 룰 fallback =====

    def _compose_rule(self, a: str, b: str):
        """사전 룰 — 부분 매칭 허용."""
        for (ka, kb), concepts in COMPOSITION_RULES.items():
            # 부분 매칭 (예: "힘들어" → "힘들" 룰 매치)
            a_match = ka in a or a in ka
            b_match = kb in b or b in kb
            if a_match and b_match:
                return CompositionResult(
                    inputs=(a, b),
                    concepts=list(concepts),
                    method="rule",
                    confidence=0.5,
                )
            # 순서 바꿔서
            a_match2 = kb in a or a in kb
            b_match2 = ka in b or b in ka
            if a_match2 and b_match2:
                return CompositionResult(
                    inputs=(a, b),
                    concepts=list(concepts),
                    method="rule",
                    confidence=0.5,
                )
        return None

    # ===== 사용자 가르침 =====

    def teach_composition(self, a: str, b: str, concepts: list):
        """사용자 직접 합성 가르침."""
        a, b = self._normalize(a), self._normalize(b)
        key = tuple(sorted([a, b]))
        self.learned_compositions[key] = list(concepts)

    # ===== 발화용 ====

    def composition_phrase(self, result: CompositionResult) -> Optional[str]:
        """합성 결과 → 자연어 발화. 라운드 66: 설명형 → 반응형."""
        if result is None or not result.concepts:
            return None
        a, b = result.inputs
        
        # 라운드 66: 입력 자체에 *반응 어미* 적용 시도
        # 합성 의미 ("X랑 Y") 설명 X — 그냥 *입력에 반응*
        from utils.korean_endings import reaction_for
        for word in (a, b):
            r = reaction_for(word)
            if r is not None:
                return self._apply_agp_candidate(r, result)
        
        # 어미 변환 안 되면 — minimal_ack (설명형 X)
        seed = sum(ord(ch) for ch in (a + b)) % 4
        ack_pool = [
            "그렇구나.",
            "음, 그래.",
            "어, 그렇네.",
            "응, 알겠어.",
        ]
        phrase = ack_pool[seed]
        return self._apply_agp_candidate(phrase, result)

    # ===== v3 round8/14: AGP observation + explicit compositor veto =====

    def _apply_agp_candidate(
        self,
        candidate_response: str,
        result: CompositionResult | None = None,
        meaning: Any | None = None,
        activated_categories: list[str] | None = None,
        hormone_state: dict[str, Any] | None = None,
    ) -> str:
        """Return candidate unchanged except in explicit compositor veto mode.

        Veto requires both this compositor and the AGP adapter to be in
        AGP_MODE_VETO. This keeps the default engine behavior unchanged and
        prevents accidental fallback activation.
        """
        agp_result = self._observe_agp_candidate(
            candidate_response,
            result=result,
            meaning=meaning,
            activated_categories=activated_categories,
            hormone_state=hormone_state,
        )
        self._observe_fasttext_parallel(candidate_response, result=result, agp_result=agp_result)
        if self.agp_mode != AGP_MODE_VETO:
            return candidate_response
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if getattr(agp, "mode", AGP_MODE_OBSERVATION) != AGP_MODE_VETO:
            return candidate_response
        if agp_result is None or getattr(agp_result, "passed", True):
            return candidate_response
        surface = self._fallback_surface(agp_result)
        return surface or candidate_response

    def _observe_agp_candidate(
        self,
        candidate_response: str,
        result: CompositionResult | None = None,
        meaning: Any | None = None,
        activated_categories: list[str] | None = None,
        hormone_state: dict[str, Any] | None = None,
    ):
        """Call AGP and record trace. Output mutation is handled separately.

        Round14 policy:
        - observation mode records only
        - veto mode records and may replace output in _apply_agp_candidate
        - speech_hub remains observation-only
        """
        if self.agp_mode not in {AGP_MODE_OBSERVATION, AGP_MODE_VETO}:
            return None
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if agp is None or not hasattr(agp, "verify"):
            return None

        active = list(activated_categories) if activated_categories is not None else self._activated_categories(result)
        meaning_obj = meaning if meaning is not None else self._composition_meaning(result, active)
        hormones = dict(hormone_state) if hormone_state is not None else self._current_hormone_state()

        try:
            agp_result = agp.verify(
                candidate_response=candidate_response,
                meaning=meaning_obj,
                activated_categories=active,
                hormone_state=hormones,
            )
        except Exception as exc:  # observation must not affect runtime output
            agp_result = None
            error = type(exc).__name__
        else:
            error = None

        effective_mode = self._effective_agp_mode()
        if effective_mode == AGP_MODE_VETO:
            self.agp_veto_count += 1
        else:
            self.agp_observation_count += 1
        self.agp_trace.append({
            "mode": effective_mode,
            "candidate_response": candidate_response,
            "meaning": meaning_obj,
            "activated_categories": active,
            "hormone_state": hormones,
            "result": agp_result,
            "error": error,
        })
        if len(self.agp_trace) > self.agp_trace_limit:
            del self.agp_trace[:-self.agp_trace_limit]
        return agp_result

    def _effective_agp_mode(self) -> str:
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if self.agp_mode == AGP_MODE_VETO and getattr(agp, "mode", AGP_MODE_OBSERVATION) == AGP_MODE_VETO:
            return AGP_MODE_VETO
        return AGP_MODE_OBSERVATION

    def _fallback_surface(self, agp_result: Any) -> str | None:
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if agp is not None and hasattr(agp, "fallback_to_surface"):
            try:
                return agp.fallback_to_surface(agp_result)
            except Exception:
                return None
        fallback = getattr(agp_result, "fallback", None)
        return str(fallback) if fallback else None

    def _composition_meaning(
        self,
        result: CompositionResult | None,
        active_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a read-only trace meaning for AGP observation.

        v3 round47: include explicit generation-time categories as the meaning
        bridge. These categories are captured from the compositor's own active
        generation surface; AGP only verifies them.
        """
        categories = list(active_categories) if active_categories is not None else self._activated_categories(result)
        if result is None:
            return {
                "meaning_layer": "composition",
                "meaning_categories": list(categories),
                "meaning_bridge_source": "compositor_generation",
                "meaning_categories_are_explicit": True,
            }
        return {
            "meaning_layer": "composition",
            "inputs": list(result.inputs),
            "concepts": list(result.concepts),
            "method": result.method,
            "confidence": float(result.confidence),
            "meaning_categories": list(categories),
            "meaning_bridge_source": "compositor_generation",
            "meaning_categories_are_explicit": True,
        }

    def _activated_categories(self, result: CompositionResult | None) -> list[str]:
        if result is None:
            return []
        cats = ["composition", result.method]
        cats.extend(str(c) for c in result.concepts)
        return cats

    def _current_hormone_state(self) -> dict[str, Any]:
        if self.engine is None:
            return {}
        hormone_adapter = getattr(self.engine, "hormone_adapter", None)
        if hormone_adapter is None:
            return {}
        if hasattr(hormone_adapter, "snapshot"):
            try:
                snap = hormone_adapter.snapshot()
                if isinstance(snap, dict):
                    return dict(snap)
            except Exception:
                return {}
        state = getattr(hormone_adapter, "state", None)
        if isinstance(state, dict):
            return dict(state)
        return {}

    # ===== v3 round36: fastText parallel observation =====

    def _fasttext_adapter_loaded(self):
        """Return explicitly loaded fastText adapter, or None.

        This helper must not call load(). It mirrors attention_analyzer's
        observation-only migration boundary.
        """
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

    def _observe_fasttext_parallel(
        self,
        candidate_response: str,
        result: CompositionResult | None = None,
        agp_result: Any | None = None,
    ) -> None:
        """Record fastText composition signals without changing output.

        This path is independent from AGP observation/veto. It reads the
        explicitly loaded FasttextEmbeddingAdapter and records diagnostics only.
        """
        adapter = self._fasttext_adapter_loaded()
        if adapter is None:
            return
        try:
            inputs = list(result.inputs) if result is not None else []
            concepts = list(result.concepts) if result is not None else []
            input_similarity = 0.0
            if len(inputs) >= 2 and hasattr(adapter, "similarity"):
                input_similarity = float(adapter.similarity(str(inputs[0]), str(inputs[1])))
            concept_scores = {}
            if hasattr(adapter, "text_similarity"):
                for concept in concepts:
                    concept_scores[str(concept)] = float(
                        adapter.text_similarity(str(concept), str(candidate_response or ""))
                    )
            trace = {
                "meaning_layer": COMPOSITOR_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "in_use_by_generation": COMPOSITOR_GENERATION_EMBEDDING_SOURCE,
                "agp_mode": self._effective_agp_mode(),
                "self_embedding_result": self._composition_meaning(result),
                "fasttext_result": {
                    "subset_name": getattr(adapter, "subset_name", None),
                    "loaded": True,
                    "inputs": inputs,
                    "concepts": concepts,
                    "input_similarity": input_similarity,
                    "concept_scores": concept_scores,
                },
                "agp_result_passed": getattr(agp_result, "passed", None),
            }
            self._record_fasttext_trace(trace)
        except Exception as exc:  # fail-open for observation-only diagnostics
            self._record_fasttext_trace({
                "meaning_layer": COMPOSITOR_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "error": str(exc),
                "in_use_by_generation": COMPOSITOR_GENERATION_EMBEDDING_SOURCE,
                "agp_mode": self._effective_agp_mode(),
            })

    def _record_fasttext_trace(self, trace: dict[str, Any]) -> None:
        self.fasttext_observation_count += 1
        self.fasttext_trace.append(dict(trace))
        if len(self.fasttext_trace) > self.fasttext_trace_cap:
            del self.fasttext_trace[: len(self.fasttext_trace) - self.fasttext_trace_cap]

    # ===== 헬퍼 =====

    @staticmethod
    def _normalize(word: str) -> str:
        """카테고리 정규화 — 합성 키 / 의문 부호 제거."""
        if not word:
            return ""
        w = word.split("+")[0].split("@")[0]
        if "_" in w:
            w = w.split("_")[0]
        w = w.rstrip("?？.,!")
        return w.strip()

    # ===== 통계 =====

    def stats(self) -> dict:
        return {
            "compose_count": self.compose_count,
            "method_counts": dict(self.method_counts),
            "learned_count": len(self.learned_compositions),
            "agp_mode": self._effective_agp_mode(),
            "agp_trace_size": len(self.agp_trace),
            "agp_observation_count": self.agp_observation_count,
            "agp_veto_count": self.agp_veto_count,
            "fasttext_parallel_observation": "fasttext" if self._fasttext_adapter_loaded() is not None else None,
            "fasttext_trace_size": len(self.fasttext_trace),
            "fasttext_observation_count": self.fasttext_observation_count,
            "fasttext_trace_cap": self.fasttext_trace_cap,
            "in_use_by_generation": COMPOSITOR_GENERATION_EMBEDDING_SOURCE,
        }
