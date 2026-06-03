"""EVE v3 round18 — AGP threshold tuning decision data.

This module converts AGP trace reports into explicit, reviewable decision data.
It is intentionally side-effect free: it never mutates AGPAdapter thresholds,
trace rows, analyzer state, fallback pools, or runtime modes.

Round18 policy:
- data-quality gate may decide ``no_change`` when more trace data is needed
- a ``tuned`` decision requires explicit threshold values from a human/reviewer
- analyzer recommendation data is never auto-applied
- applying a decision still requires a separate explicit ``set_thresholds()`` call
"""

from __future__ import annotations

from typing import Any, Mapping


AGP_TUNING_ACTION_TUNED = "tuned"
AGP_TUNING_ACTION_NO_CHANGE = "no_change"

AGP_TUNING_REASON_INSUFFICIENT_DATA = "insufficient_data"
AGP_TUNING_REASON_MANUAL_PROPOSAL = "manual_proposal"
AGP_TUNING_REASON_MANUAL_NO_CHANGE = "manual_review_no_change"


_VALID_ACTIONS = {AGP_TUNING_ACTION_TUNED, AGP_TUNING_ACTION_NO_CHANGE}


def build_threshold_tuning_decision(
    report: Mapping[str, Any],
    *,
    category_threshold: float | None = None,
    hormone_threshold: float | None = None,
    decided_by: str = "manual_review",
) -> dict[str, Any]:
    """Build a data-only AGP threshold tuning decision.

    The function is pure. It reads a first-pass/threshold analysis report and
    returns decision data. It does not call ``AGPAdapter.set_thresholds``.

    Decision rules:
    - insufficient sample data always returns ``no_change``
    - sufficient data + explicit threshold proposal returns ``tuned``
    - sufficient data + no explicit proposal returns ``no_change``
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")

    threshold_analysis = _mapping(report.get("threshold_analysis"))
    current_thresholds = _mapping(threshold_analysis.get("current_thresholds"))
    data_quality = _mapping(report.get("data_quality"))
    recommendation_data = _mapping(threshold_analysis.get("recommendation_data"))

    old_thresholds = _current_threshold_snapshot(current_thresholds)
    sample_size_sufficient = bool(data_quality.get("sample_size_sufficient", False))
    sample_count = _safe_int(data_quality.get("sample_count", recommendation_data.get("sample_count", 0)))
    minimum_samples = _safe_int(data_quality.get("minimum_samples", recommendation_data.get("minimum_samples", 0)))

    base = {
        "schema": "agp_threshold_tuning_decision_v1",
        "decided_by": str(decided_by or "manual_review"),
        "old_thresholds": dict(old_thresholds),
        "requires_explicit_set_thresholds": True,
        "auto_applied": False,
        "rationale_data": {
            "sample_count": sample_count,
            "minimum_samples": minimum_samples,
            "data_quality": dict(data_quality),
            "recommendation_data": dict(recommendation_data),
            "current_pass_rate": threshold_analysis.get("current_pass_rate"),
        },
    }

    if not sample_size_sufficient:
        return {
            **base,
            "action": AGP_TUNING_ACTION_NO_CHANGE,
            "reason": AGP_TUNING_REASON_INSUFFICIENT_DATA,
            "new_thresholds": dict(old_thresholds),
            "next_step": "collect_more",
        }

    has_explicit_proposal = category_threshold is not None or hormone_threshold is not None
    if not has_explicit_proposal:
        return {
            **base,
            "action": AGP_TUNING_ACTION_NO_CHANGE,
            "reason": AGP_TUNING_REASON_MANUAL_NO_CHANGE,
            "new_thresholds": dict(old_thresholds),
            "next_step": "manual_review_or_collect_more",
        }

    new_thresholds = dict(old_thresholds)
    if category_threshold is not None:
        new_thresholds["category_threshold"] = _validate_threshold("category_threshold", category_threshold)
    if hormone_threshold is not None:
        new_thresholds["hormone_threshold"] = _validate_threshold("hormone_threshold", hormone_threshold)

    return {
        **base,
        "action": AGP_TUNING_ACTION_TUNED,
        "reason": AGP_TUNING_REASON_MANUAL_PROPOSAL,
        "new_thresholds": new_thresholds,
        "next_step": "call_agp_set_thresholds_explicitly",
    }


def validate_threshold_tuning_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate decision data shape without applying it.

    Returns a shallow copy so callers can inspect data without mutating the
    original decision object by accident.
    """
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping")
    snapshot = dict(decision)
    action = snapshot.get("action")
    if action not in _VALID_ACTIONS:
        raise ValueError("decision action must be 'tuned' or 'no_change'")
    if snapshot.get("auto_applied") is not False:
        raise ValueError("threshold tuning decisions must not be auto-applied")
    if snapshot.get("requires_explicit_set_thresholds") is not True:
        raise ValueError("threshold tuning decisions require explicit set_thresholds")

    old_thresholds = _mapping(snapshot.get("old_thresholds"))
    new_thresholds = _mapping(snapshot.get("new_thresholds"))
    _current_threshold_snapshot(old_thresholds)
    _current_threshold_snapshot(new_thresholds)
    return snapshot


def _current_threshold_snapshot(raw: Mapping[str, Any]) -> dict[str, float]:
    return {
        "category_threshold": _validate_threshold(
            "category_threshold", raw.get("category_threshold", 0.3)
        ),
        "hormone_threshold": _validate_threshold(
            "hormone_threshold", raw.get("hormone_threshold", 0.5)
        ),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _validate_threshold(name: str, value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0.0 and 1.0") from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return threshold


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
