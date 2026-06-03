"""EVE v3 round10 — read-only AGP observation trace analyzer.

This module analyzes AGP observation traces produced by compositor and
speech_hub integration. It is intentionally read-only: it never mutates trace
lists, trace entries, AGP results, adapters, or engine state.

Round10 scope:
- summarize existing observation traces
- expose pass/fail statistics for later threshold analysis
- do not tune thresholds
- do not activate veto or fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from adapters.agp_adapter import (
    AGP_REASON_ANCHORED,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
)


@dataclass(frozen=True)
class AGPTraceAnalyzer:
    """Read-only analyzer for bounded AGP observation traces.

    ``sources`` maps a source name such as ``"compositor"`` or
    ``"speech_hub"`` to an iterable of trace dictionaries. The analyzer keeps
    references only so it can report on live observation traces, but all methods
    treat those traces as immutable input.
    """

    sources: Mapping[str, Iterable[dict[str, Any]]]
    category_threshold: float = 0.3
    hormone_threshold: float = 0.5

    @classmethod
    def from_engine(cls, engine: Any) -> "AGPTraceAnalyzer":
        """Build an analyzer over the standard engine AGP trace sources."""
        sources: dict[str, Iterable[dict[str, Any]]] = {}
        compositor = getattr(engine, "compositor", None)
        speech_hub = getattr(engine, "speech_hub", None)
        if compositor is not None and hasattr(compositor, "agp_trace"):
            sources["compositor"] = compositor.agp_trace
        if speech_hub is not None and hasattr(speech_hub, "agp_trace"):
            sources["speech_hub"] = speech_hub.agp_trace
        agp_adapter = getattr(engine, "agp_adapter", None)
        category_threshold = getattr(agp_adapter, "category_threshold", 0.3)
        hormone_threshold = getattr(agp_adapter, "hormone_threshold", 0.5)
        return cls(
            sources=sources,
            category_threshold=float(category_threshold),
            hormone_threshold=float(hormone_threshold),
        )

    def summarize_by_reason(self) -> dict[str, int]:
        """Return counts by AGP result reason.

        Trace rows without an AGPResult are counted under ``"error"`` when an
        observation error exists, otherwise ``"no_result"``.
        """
        counts: dict[str, int] = {}
        for _, row in self._rows():
            reason = self._reason_for_row(row)
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def summarize_by_layer(self) -> dict[str, dict[str, Any]]:
        """Return pass/fail/reason statistics grouped by meaning layer."""
        summary: dict[str, dict[str, Any]] = {}
        for source, row in self._rows():
            layer = self._layer_for_row(source, row)
            bucket = summary.setdefault(
                layer,
                {"total": 0, "passed": 0, "failed": 0, "reasons": {}},
            )
            bucket["total"] += 1
            if self._passed_for_row(row):
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
            reason = self._reason_for_row(row)
            bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
        return summary

    def pass_rate(self) -> float:
        """Return overall AGP pass rate. Empty traces return 0.0."""
        rows = self._rows()
        total = len(rows)
        if total == 0:
            return 0.0
        passed = sum(1 for _, row in rows if self._passed_for_row(row))
        return passed / total

    def recent_fails(self, n: int = 10) -> list[dict[str, Any]]:
        """Return recent failed rows as shallow copies, newest last.

        Returning copies prevents callers from accidentally mutating underlying
        trace rows while inspecting failures.
        """
        if n <= 0:
            return []
        failed: list[dict[str, Any]] = []
        for source, row in self._rows():
            if not self._passed_for_row(row):
                snapshot = dict(row)
                snapshot.setdefault("source", source)
                snapshot.setdefault("reason", self._reason_for_row(row))
                failed.append(snapshot)
        return failed[-n:]

    def hormone_signature_distribution(self, reason: str | None = None) -> dict[str, int]:
        """Return hormone-state signature counts, optionally filtered by reason.

        This is an analytics-only helper for later threshold work. It reads the
        ``hormone_state`` snapshot stored in trace rows and reduces it to stable
        coarse signatures such as ``"cortisol_high"`` or ``"balanced"``.
        """
        counts: dict[str, int] = {}
        for _, row in self._rows():
            if reason is not None and self._reason_for_row(row) != reason:
                continue
            signature = self._hormone_signature(row.get("hormone_state"))
            counts[signature] = counts.get(signature, 0) + 1
        return counts

    def category_activation_distribution(
        self,
        reason: str = "unknown_category",
    ) -> dict[str, int]:
        """Return a coarse activation histogram for candidate categories.

        The analyzer accepts several trace shapes without mutating them:
        ``category_activations`` / ``activation_scores`` mappings when present,
        or the AGPResult candidate categories compared with the
        ``activated_categories`` snapshot. Missing candidate categories count as
        0.0 activation, which is the useful case for unknown-category failures.
        """
        buckets = {"0.0-0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.8": 0, "0.8-1.0": 0}
        for _, row in self._rows():
            if reason is not None and self._reason_for_row(row) != reason:
                continue
            for value in self._activation_values_for_row(row):
                bucket = self._activation_bucket(value)
                buckets[bucket] = buckets.get(bucket, 0) + 1
        return {key: value for key, value in buckets.items() if value}

    def fail_pattern_correlation(self) -> dict[str, dict[str, int]]:
        """Return reason × layer × hormone-signature counts for failed rows.

        Keys are reason strings. Nested keys use ``"<layer>|<signature>"`` so
        the structure stays simple and deterministic for early-round reports.
        """
        matrix: dict[str, dict[str, int]] = {}
        for source, row in self._rows():
            if self._passed_for_row(row):
                continue
            reason = self._reason_for_row(row)
            layer = self._layer_for_row(source, row)
            signature = self._hormone_signature(row.get("hormone_state"))
            key = f"{layer}|{signature}"
            bucket = matrix.setdefault(reason, {})
            bucket[key] = bucket.get(key, 0) + 1
        return matrix

    def threshold_analysis_report(
        self,
        simulated_thresholds: list[float] | tuple[float, ...] | None = None,
        minimum_samples: int = 10,
    ) -> dict[str, Any]:
        """Return an advisory-only threshold simulation report.

        Round12 policy:
        - produce data, not natural-language recommendations
        - do not mutate AGP adapter thresholds or trace rows
        - do not activate veto, fallback, or route changes
        - with insufficient samples, mark recommendation data as deferred

        The simulation is intentionally conservative and early-round. Category
        thresholds are evaluated against trace activation snapshots when present.
        Hormone-mismatch rows remain hormone-mismatch failures regardless of
        category threshold. Rows without activation evidence fall back to their
        observed AGP result.
        """
        thresholds = self._normalize_thresholds(simulated_thresholds)
        rows = self._rows()
        sample_count = len(rows)

        simulated: dict[str, dict[str, Any]] = {}
        for threshold in thresholds:
            key = f"category_{threshold:g}"
            simulated[key] = self._simulate_category_threshold(rows, threshold)

        recommendation_data = self._recommendation_data(
            simulated=simulated,
            sample_count=sample_count,
            minimum_samples=minimum_samples,
        )

        return {
            "current_thresholds": {
                "category_threshold": float(self.category_threshold),
                "hormone_threshold": float(self.hormone_threshold),
            },
            "current_pass_rate": self.pass_rate(),
            "sample_count": sample_count,
            "simulated": simulated,
            "recommendation_data": recommendation_data,
        }


    def first_pass_report(
        self,
        simulated_thresholds: list[float] | tuple[float, ...] | None = None,
        minimum_samples: int = 10,
    ) -> dict[str, Any]:
        """Return a read-only first-pass AGP trace data report.

        Round17 policy:
        - data dict only; no natural-language recommendations
        - no threshold changes
        - no veto/fallback/mode changes
        - use existing analyzer helpers so later tuning rounds have a stable
          evidence shape to review manually
        """
        sample_count = len(self._rows())
        sufficient = sample_count >= int(minimum_samples)
        threshold_report = self.threshold_analysis_report(
            simulated_thresholds=simulated_thresholds,
            minimum_samples=minimum_samples,
        )
        return {
            "summary": {
                "total_observations": sample_count,
                "pass_rate": self.pass_rate(),
                "fail_counts_by_reason": self._fail_counts_by_reason(),
                "reason_counts": self.summarize_by_reason(),
            },
            "by_layer": self.summarize_by_layer(),
            "fail_patterns": {
                "hormone_signature_distribution": self.hormone_signature_distribution(),
                "category_activation_distribution": self.category_activation_distribution(),
                "fail_pattern_correlation": self.fail_pattern_correlation(),
            },
            "threshold_analysis": threshold_report,
            "data_quality": {
                "sample_count": sample_count,
                "minimum_samples": int(minimum_samples),
                "sample_size_sufficient": sufficient,
                "recommendation": "ready_for_tuning" if sufficient else "collect_more",
            },
        }

    def _fail_counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, row in self._rows():
            if self._passed_for_row(row):
                continue
            reason = self._reason_for_row(row)
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _normalize_thresholds(
        self,
        simulated_thresholds: list[float] | tuple[float, ...] | None,
    ) -> list[float]:
        raw = simulated_thresholds if simulated_thresholds is not None else [0.2, 0.25, 0.3, 0.35]
        normalized: list[float] = []
        for value in raw:
            threshold = self._clamp01(self._safe_float(value))
            if threshold not in normalized:
                normalized.append(threshold)
        return sorted(normalized)

    def _simulate_category_threshold(
        self,
        rows: list[tuple[str, dict[str, Any]]],
        threshold: float,
    ) -> dict[str, Any]:
        passed = 0
        fail_count: dict[str, int] = {}

        for _, row in rows:
            passed_for_row, reason = self._simulate_row_at_threshold(row, threshold)
            if passed_for_row:
                passed += 1
            else:
                fail_count[reason] = fail_count.get(reason, 0) + 1

        total = len(rows)
        failed = total - passed
        return {
            "threshold": threshold,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total) if total else 0.0,
            "fail_count": fail_count,
        }

    def _simulate_row_at_threshold(
        self,
        row: dict[str, Any],
        threshold: float,
    ) -> tuple[bool, str]:
        observed_reason = self._reason_for_row(row)
        if observed_reason == AGP_REASON_HORMONE_MISMATCH:
            return False, AGP_REASON_HORMONE_MISMATCH

        values = self._activation_values_for_row(row)
        if values:
            if all(value >= threshold for value in values):
                return True, AGP_REASON_ANCHORED
            return False, AGP_REASON_UNKNOWN_CATEGORY

        if self._passed_for_row(row):
            return True, AGP_REASON_ANCHORED
        if observed_reason in (AGP_REASON_UNKNOWN_CATEGORY, AGP_REASON_HORMONE_MISMATCH):
            return False, observed_reason
        return False, observed_reason or AGP_REASON_UNKNOWN_CATEGORY

    def _recommendation_data(
        self,
        simulated: dict[str, dict[str, Any]],
        sample_count: int,
        minimum_samples: int,
    ) -> dict[str, Any]:
        if sample_count < minimum_samples:
            return {
                "status": "insufficient_data",
                "category_threshold": None,
                "sample_count": sample_count,
                "minimum_samples": int(minimum_samples),
                "rationale_data": {
                    "simulated_thresholds": list(simulated.keys()),
                    "current_pass_rate": self.pass_rate(),
                },
            }

        best_key = None
        best_score = (-1.0, 0.0)
        for key, stats in simulated.items():
            pass_rate = self._safe_float(stats.get("pass_rate"))
            threshold = self._safe_float(stats.get("threshold"))
            score = (pass_rate, threshold)
            if score > best_score:
                best_key = key
                best_score = score

        best_stats = simulated.get(best_key or "", {})
        return {
            "status": "candidate",
            "category_threshold": best_stats.get("threshold"),
            "sample_count": sample_count,
            "minimum_samples": int(minimum_samples),
            "rationale_data": {
                "selected_key": best_key,
                "selected_pass_rate": best_stats.get("pass_rate", 0.0),
                "selected_fail_count": dict(best_stats.get("fail_count", {})),
                "current_pass_rate": self.pass_rate(),
            },
        }

    def _hormone_signature(self, hormone_state: Any) -> str:
        if not isinstance(hormone_state, Mapping):
            return "unknown"
        cortisol = self._safe_float(hormone_state.get("cortisol", hormone_state.get("stress", 0.0)))
        oxytocin = self._safe_float(hormone_state.get("oxytocin", 0.0))
        dopamine = self._safe_float(hormone_state.get("dopamine", 0.0))
        norepinephrine = self._safe_float(hormone_state.get("norepinephrine", hormone_state.get("noradrenaline", 0.0)))

        if cortisol >= 0.7:
            return "cortisol_high"
        if oxytocin >= 0.7:
            return "oxytocin_high"
        if dopamine >= 0.7 or norepinephrine >= 0.7:
            return "energy_high"
        if cortisol <= 0.3 and oxytocin <= 0.3 and dopamine <= 0.3 and norepinephrine <= 0.3:
            return "low_arousal"
        return "balanced"

    def _activation_values_for_row(self, row: dict[str, Any]) -> list[float]:
        scores = row.get("category_activations", row.get("activation_scores"))
        if isinstance(scores, Mapping):
            return [self._clamp01(self._safe_float(value)) for value in scores.values()]

        result = row.get("result")
        candidate_categories = list(getattr(result, "candidate_categories", []) or [])
        activated = row.get("activated_categories", [])

        if isinstance(activated, Mapping):
            if candidate_categories:
                return [self._clamp01(self._safe_float(activated.get(cat, 0.0))) for cat in candidate_categories]
            return [self._clamp01(self._safe_float(value)) for value in activated.values()]

        active_set = set(activated if isinstance(activated, (list, tuple, set)) else [])
        if candidate_categories:
            return [1.0 if cat in active_set else 0.0 for cat in candidate_categories]
        return []

    def _activation_bucket(self, value: float) -> str:
        x = self._clamp01(value)
        if x < 0.1:
            return "0.0-0.1"
        if x < 0.3:
            return "0.1-0.3"
        if x < 0.5:
            return "0.3-0.5"
        if x < 0.8:
            return "0.5-0.8"
        return "0.8-1.0"

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _rows(self) -> list[tuple[str, dict[str, Any]]]:
        rows: list[tuple[str, dict[str, Any]]] = []
        for source, trace in self.sources.items():
            for row in trace:
                if isinstance(row, dict):
                    rows.append((str(source), row))
        return rows

    def _reason_for_row(self, row: dict[str, Any]) -> str:
        result = row.get("result")
        reason = getattr(result, "reason", None)
        if isinstance(reason, str) and reason:
            return reason
        if row.get("error"):
            return "error"
        return "no_result"

    def _passed_for_row(self, row: dict[str, Any]) -> bool:
        result = row.get("result")
        passed = getattr(result, "passed", False)
        return bool(passed)

    def _layer_for_row(self, source: str, row: dict[str, Any]) -> str:
        meaning = row.get("meaning")
        if isinstance(meaning, dict):
            layer = meaning.get("meaning_layer")
            if isinstance(layer, str) and layer:
                return layer
        return source or AGP_REASON_ANCHORED
