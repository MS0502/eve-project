"""
EVE v3 round2 — AGP skeleton.

AGP means Anchored Generation Principle.
This adapter is intentionally not integrated into runtime behavior yet.

v3 rule:
- output categories must be anchored in EVE internal SA activation
- output tone must be compatible with hormone state
- if either condition fails, EVE must use an honest fallback instead of a
  plausible unanchored answer

Implementation is introduced in small, observable steps. v3 round6 adds a
minimal output-side verifier for tests only. v3 round7 adds deterministic
fallback strings to failed AGPResult objects. v3 round14 exposes a minimal
fallback surface mapping for compositor veto mode only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AGP_REASON_ANCHORED = "anchored"
AGP_REASON_UNKNOWN_CATEGORY = "unknown_category"
AGP_REASON_HORMONE_MISMATCH = "hormone_mismatch"
AGP_REASON_NOT_IMPLEMENTED = "not_implemented"

AGP_MODE_OBSERVATION = "observation"
AGP_MODE_VETO = "veto"

AGP_FALLBACK_UNKNOWN = "잘 모르겠어."
AGP_FALLBACK_UNKNOWN_CATEGORY = "그게 뭐야?"
AGP_FALLBACK_UNKNOWN_CATEGORY_STRESSED = "잘 모르겠어."
AGP_FALLBACK_HORMONE_MISMATCH = "음..."
AGP_FALLBACK_HORMONE_MISMATCH_HIGH_CORTISOL = "응..."

# v3 round14: tiny surface pool for explicit veto mode only.
# Keep this pool intentionally small; richer phrasing belongs to future
# Construction Grammar, not AGP case expansion.
AGP_FALLBACK_SURFACE_POOL = (
    AGP_FALLBACK_UNKNOWN,
    AGP_FALLBACK_UNKNOWN_CATEGORY,
    AGP_FALLBACK_HORMONE_MISMATCH,
    AGP_FALLBACK_HORMONE_MISMATCH_HIGH_CORTISOL,
)


AGP_INPUT_REASON_APPRAISAL_ACCEPTED = "appraisal_meaning_accepted"
AGP_INPUT_REASON_NO_MEANING = "no_meaning"
AGP_INPUT_REASON_UNSUPPORTED_MEANING = "unsupported_meaning"
AGP_INPUT_LAYER_APPRAISAL = "appraisal"


@dataclass(frozen=True)
class AGPResult:
    """Result object for Anchored Generation Principle verification."""

    passed: bool
    reason: str
    fallback: str | None = None
    activated_categories: list[str] = field(default_factory=list)
    candidate_categories: list[str] = field(default_factory=list)
    hormone_distance: float = 0.0


@dataclass(frozen=True)
class AGPInputResult:
    """Normalized input-side meaning accepted by AGP.

    v3 round5 scope: this is an interface object only. It does not verify or
    veto responses yet. It lets upstream semantic stabilizers, such as
    AppraisalClassifier, hand AGP a deterministic meaning summary.
    """

    accepted: bool
    reason: str
    meaning_layer: str | None = None
    normalized_meaning: dict[str, Any] = field(default_factory=dict)
    requires_output_verification: bool = True


class AGPAdapter:
    """
    Anchored Generation Principle verification adapter.

    v3 round2 scope:
    - construction is allowed so engine wiring can be tested later
    - verify/helper implementation is deliberately deferred
    - no compositor or speech_hub integration in this round
    """

    def __init__(
        self,
        sa_adapter: Any | None = None,
        hormone_adapter: Any | None = None,
        category_threshold: float = 0.3,
        hormone_threshold: float = 0.5,
        mode: str = AGP_MODE_OBSERVATION,
    ):
        self.sa_adapter = sa_adapter
        self.hormone_adapter = hormone_adapter
        self.category_threshold = self._validate_threshold(
            "category_threshold", category_threshold
        )
        self.hormone_threshold = self._validate_threshold(
            "hormone_threshold", hormone_threshold
        )
        self.mode = self._validate_mode(mode)

    def set_thresholds(
        self,
        *,
        category_threshold: float | None = None,
        hormone_threshold: float | None = None,
    ) -> None:
        """Explicitly update AGP thresholds after validation.

        v3 round13 policy:
        - defaults remain unchanged
        - threshold changes require this explicit keyword-only method
        - trace-analyzer recommendation data is never auto-applied here
        - changing thresholds does not change observation/veto mode
        """
        next_category = self.category_threshold
        next_hormone = self.hormone_threshold
        if category_threshold is not None:
            next_category = self._validate_threshold(
                "category_threshold", category_threshold
            )
        if hormone_threshold is not None:
            next_hormone = self._validate_threshold(
                "hormone_threshold", hormone_threshold
            )

        self.category_threshold = next_category
        self.hormone_threshold = next_hormone

    def _validate_threshold(self, name: str, value: Any) -> float:
        threshold = self._safe_float(value, default=None)
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
        return threshold

    def _validate_mode(self, mode: str) -> str:
        if mode not in {AGP_MODE_OBSERVATION, AGP_MODE_VETO}:
            raise ValueError("AGP mode must be 'observation' or 'veto'")
        return mode

    def set_mode(self, mode: str) -> None:
        """Explicitly switch AGP runtime mode.

        v3 round14 policy:
        - default remains observation
        - veto can only be enabled by an explicit mode change
        - threshold changes never enable veto implicitly
        """
        self.mode = self._validate_mode(mode)

    def fallback_to_surface(self, result: AGPResult | None) -> str | None:
        """Map failed AGP fallback data to a minimal surface form.

        This method must not inspect raw candidate text. It only reads the
        AGPResult reason/fallback data generated by verify().
        """
        if result is None or result.passed:
            return None
        fallback = result.fallback
        if fallback in AGP_FALLBACK_SURFACE_POOL:
            return fallback
        if result.reason == AGP_REASON_UNKNOWN_CATEGORY:
            return AGP_FALLBACK_UNKNOWN
        if result.reason == AGP_REASON_HORMONE_MISMATCH:
            return AGP_FALLBACK_HORMONE_MISMATCH
        return AGP_FALLBACK_UNKNOWN

    def accept_input_meaning(self, meaning: Any) -> AGPInputResult:
        """Accept a deterministic input-side meaning summary for future AGP checks.

        This method is intentionally non-blocking in v3 round5. It normalizes
        AppraisalClassifier output so later rounds can perform AGP checks without
        changing route behavior in this round.
        """
        if meaning is None:
            return AGPInputResult(
                accepted=False,
                reason=AGP_INPUT_REASON_NO_MEANING,
                meaning_layer=None,
                normalized_meaning={},
            )

        if hasattr(meaning, "as_meaning_dict"):
            normalized = dict(meaning.as_meaning_dict())
            layer = getattr(meaning, "meaning_layer", None) or normalized.get("meaning_layer")
        elif isinstance(meaning, dict):
            normalized = dict(meaning)
            layer = normalized.get("meaning_layer")
        else:
            return AGPInputResult(
                accepted=False,
                reason=AGP_INPUT_REASON_UNSUPPORTED_MEANING,
                meaning_layer=None,
                normalized_meaning={},
            )

        if self._is_appraisal_meaning(normalized):
            normalized.setdefault("meaning_layer", layer or AGP_INPUT_LAYER_APPRAISAL)
            return AGPInputResult(
                accepted=True,
                reason=AGP_INPUT_REASON_APPRAISAL_ACCEPTED,
                meaning_layer=normalized["meaning_layer"],
                normalized_meaning=normalized,
            )

        return AGPInputResult(
            accepted=False,
            reason=AGP_INPUT_REASON_UNSUPPORTED_MEANING,
            meaning_layer=layer,
            normalized_meaning=normalized,
        )

    def _is_appraisal_meaning(self, meaning: dict[str, Any]) -> bool:
        """Return True for AppraisalClassifier-compatible meaning dictionaries."""
        required = {"label", "subject_type", "affect_type", "route_override", "via"}
        return required.issubset(set(meaning.keys()))

    def verify(
        self,
        candidate_response: str,
        meaning: Any = None,
        activated_categories: list[str] | tuple[str, ...] | set[str] | dict[str, Any] | None = None,
        hormone_state: dict[str, Any] | None = None,
    ) -> AGPResult:
        """Verify a candidate response against AGP in observation-only mode.

        v3 round6/7 scope:
        - returns AGPResult for test scaffolding
        - fills fallback text on failed AGPResult objects
        - does not mutate SA, hormone state, meaning objects, or inputs
        - does not trigger fallback/veto in runtime generation
        - is not wired into compositor or speech_hub yet

        Checks:
        1. deterministic transformation: guaranteed by v3 governance
        2. category anchor: candidate categories are present in activated_categories
        3. hormone alignment: minimal deterministic tone/state distance
        """
        candidate_categories = self._extract_candidate_categories(candidate_response, meaning)
        active_snapshot = self._activated_category_snapshot(activated_categories)

        anchor_ok, missing_categories = self._check_anchor(candidate_categories, activated_categories)
        if not anchor_ok:
            return AGPResult(
                passed=False,
                reason=AGP_REASON_UNKNOWN_CATEGORY,
                fallback=self._generate_fallback(AGP_REASON_UNKNOWN_CATEGORY, hormone_state),
                activated_categories=active_snapshot,
                candidate_categories=missing_categories or candidate_categories,
                hormone_distance=0.0,
            )

        hormone_distance = self._check_hormone_alignment(candidate_response, hormone_state)
        if hormone_distance > self.hormone_threshold:
            return AGPResult(
                passed=False,
                reason=AGP_REASON_HORMONE_MISMATCH,
                fallback=self._generate_fallback(AGP_REASON_HORMONE_MISMATCH, hormone_state),
                activated_categories=active_snapshot,
                candidate_categories=candidate_categories,
                hormone_distance=hormone_distance,
            )

        return AGPResult(
            passed=True,
            reason=AGP_REASON_ANCHORED,
            fallback=None,
            activated_categories=active_snapshot,
            candidate_categories=candidate_categories,
            hormone_distance=hormone_distance,
        )


    def verify_with_trace(
        self,
        candidate_response: str,
        meaning: Any = None,
        activated_categories: list[str] | tuple[str, ...] | set[str] | dict[str, Any] | None = None,
        hormone_state: dict[str, Any] | None = None,
        engine: Any | None = None,
    ) -> dict[str, Any]:
        """Return AGP verification plus category-extraction trace data.

        v3 round46 scope:
        - read-only probe for H1~H4 root-cause analysis
        - uses the same category extraction and verification path as verify()
        - does not mutate thresholds, SA, hormone state, wrapper telemetry, or runtime output
        - does not fix category extraction or relax thresholds
        """
        candidate_categories = self._extract_candidate_categories(candidate_response, meaning)
        active_input = activated_categories
        if active_input is None:
            active_input = self._engine_activated_categories(engine)
        sa_snapshot = self._activated_category_snapshot(active_input)
        if isinstance(active_input, dict):
            overlap = [
                category
                for category in candidate_categories
                if self._safe_float(active_input.get(category), default=0.0) >= self.category_threshold
            ]
            weak_overlap = [
                category
                for category in candidate_categories
                if category in active_input
                and self._safe_float(active_input.get(category), default=0.0) < self.category_threshold
            ]
        else:
            active_set = set(sa_snapshot)
            overlap = [category for category in candidate_categories if category in active_set]
            weak_overlap = []

        result = self.verify(
            candidate_response=candidate_response,
            meaning=meaning,
            activated_categories=active_input,
            hormone_state=hormone_state,
        )
        trace = {
            "candidate_categories_extracted": list(candidate_categories),
            "sa_activated_categories": list(sa_snapshot),
            "overlap": list(overlap),
            "weak_overlap_below_threshold": list(weak_overlap),
            "candidate_count": len(candidate_categories),
            "sa_count": len(sa_snapshot),
            "overlap_count": len(overlap),
            "weak_overlap_count": len(weak_overlap),
            "thresholds_used": {
                "category_threshold": self.category_threshold,
                "hormone_threshold": self.hormone_threshold,
            },
            "extraction_source": self._extraction_source(meaning),
            "extraction_method": (
                "meaning_bridge:meaning_categories"
                if self._meaning_bridge_categories(meaning)
                else "accept_input_meaning_fields:meaning_layer,label,subject_type,affect_type"
            ),
            "does_not_invent_categories": True,
            "read_only": True,
        }
        return {
            "result": result,
            "trace": trace,
            "decision_matches_regular_verify": True,
            "read_only": True,
        }

    def _engine_activated_categories(self, engine: Any | None) -> dict[str, float] | list[str]:
        """Best-effort read-only snapshot of engine SA activation for tracing only."""
        if engine is None:
            return []
        sources = [
            getattr(engine, "activation_adapter", None),
            getattr(engine, "category_activation", None),
            getattr(engine, "sa_adapter", None),
            self.sa_adapter,
        ]
        for source in sources:
            if source is None:
                continue
            if hasattr(source, "top_active"):
                try:
                    pairs = source.top_active(n=20)
                except TypeError:
                    try:
                        pairs = source.top_active()
                    except Exception:
                        pairs = None
                except Exception:
                    pairs = None
                if pairs:
                    out: dict[str, float] = {}
                    for item in pairs:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            out[str(item[0])] = self._safe_float(item[1], default=0.0) or 0.0
                        else:
                            out[str(item)] = 1.0
                    if out:
                        return out
            for field in ("active_categories", "activated_categories", "categories"):
                value = getattr(source, field, None)
                if isinstance(value, dict):
                    return {str(k): self._safe_float(v, default=0.0) or 0.0 for k, v in value.items()}
                if isinstance(value, (list, tuple, set)):
                    return [str(v) for v in value]
        return []

    def _extract_candidate_categories(self, response: str, meaning: Any = None) -> list[str]:
        """Extract deterministic candidate categories from AGP-compatible meaning.

        v3 round47 policy:
        - prefer explicit meaning bridge categories produced by generation modules
        - never invent categories from raw candidate text
        - fall back to legacy AppraisalClassifier-compatible meaning fields
        """
        bridge_categories = self._meaning_bridge_categories(meaning)
        if bridge_categories:
            return bridge_categories

        input_result = self.accept_input_meaning(meaning)
        if not input_result.accepted:
            return []

        normalized = dict(input_result.normalized_meaning)
        categories: list[str] = []
        for key in ("meaning_layer", "label", "subject_type", "affect_type"):
            value = normalized.get(key)
            if isinstance(value, str) and value:
                categories.append(value)
        return categories

    def _meaning_bridge_categories(self, meaning: Any = None) -> list[str]:
        """Return explicit generation-time categories, if provided.

        The bridge is a transport surface only. AGP must not infer or create
        categories here; it only consumes explicit categories handed over by
        SpeechHub/Compositor at generation time.
        """
        if meaning is None:
            return []
        raw = None
        if isinstance(meaning, dict):
            raw = meaning.get("meaning_categories")
        else:
            raw = getattr(meaning, "meaning_categories", None)
        if not isinstance(raw, (list, tuple, set)):
            return []
        categories: list[str] = []
        seen: set[str] = set()
        for item in raw:
            category = str(item or "").strip()
            if category and category not in seen:
                seen.add(category)
                categories.append(category)
        return categories

    def _extraction_source(self, meaning: Any = None) -> str:
        if self._meaning_bridge_categories(meaning):
            return "meaning_bridge"
        return "text_extraction"

    def _check_anchor(
        self,
        candidate_categories: list[str],
        activated_categories: list[str] | tuple[str, ...] | set[str] | dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """Check whether candidate categories are active in EVE internal SA.

        Empty candidate categories fail closed. This prevents unanchored output
        from being treated as pass in early AGP scaffolding.
        """
        if not candidate_categories:
            return False, []
        if isinstance(activated_categories, dict):
            missing = [
                cat
                for cat in candidate_categories
                if self._safe_float(activated_categories.get(cat), default=0.0)
                < self.category_threshold
            ]
            return not missing, missing

        active = set(activated_categories or [])
        missing = [cat for cat in candidate_categories if cat not in active]
        return not missing, missing

    def _activated_category_snapshot(
        self,
        activated_categories: list[str] | tuple[str, ...] | set[str] | dict[str, Any] | None,
    ) -> list[str]:
        """Return a deterministic list snapshot of categories active at threshold.

        Lists/tuples/sets remain presence-based for backward compatibility.
        Mappings are threshold-aware so round13 can test configurable category
        thresholds without changing runtime routing.
        """
        if isinstance(activated_categories, dict):
            return [
                str(category)
                for category, score in activated_categories.items()
                if self._safe_float(score, default=0.0) >= self.category_threshold
            ]
        return list(activated_categories or [])

    def _check_hormone_alignment(
        self,
        response: str,
        hormone_state: dict[str, Any] | None = None,
    ) -> float:
        """Return a minimal deterministic hormone distance.

        Round6 only needs an observable scaffold. High cortisol combined with
        overly warm/cheerful response markers is treated as a mismatch.
        """
        state = dict(hormone_state or {})
        cortisol = self._safe_float(state.get("cortisol", state.get("stress", 0.0)))
        response_text = response or ""
        warm_markers = ("괜찮아", "좋아", "행복", "따뜻", "신나", "괜찮을 거야")
        has_warm_tone = any(marker in response_text for marker in warm_markers)
        if cortisol >= self.hormone_threshold and has_warm_tone:
            return 1.0
        return 0.0

    def _safe_float(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _generate_fallback(
        self,
        reason: str,
        hormone_state: dict[str, Any] | None = None,
    ) -> str:
        """Return a deterministic, honest fallback for a failed AGP check.

        v3 round7 policy:
        - branch only on AGP failure reason and hormone-state signature
        - never branch on raw candidate response text
        - keep strings short and honest
        - return data only; runtime fallback/veto remains inactive
        """
        state = dict(hormone_state or {})
        cortisol = self._safe_float(state.get("cortisol", state.get("stress", 0.0)))

        if reason == AGP_REASON_UNKNOWN_CATEGORY:
            if cortisol >= self.hormone_threshold:
                return AGP_FALLBACK_UNKNOWN_CATEGORY_STRESSED
            return AGP_FALLBACK_UNKNOWN_CATEGORY

        if reason == AGP_REASON_HORMONE_MISMATCH:
            if cortisol >= self.hormone_threshold:
                return AGP_FALLBACK_HORMONE_MISMATCH_HIGH_CORTISOL
            return AGP_FALLBACK_HORMONE_MISMATCH

        return AGP_FALLBACK_UNKNOWN
