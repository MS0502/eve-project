"""Round781-800 read-only affect/hormone interaction matrix.

The matrix describes bounded relationships among registered axes.  It is a
proposal/registry surface only and does not schedule, apply, or persist any
emotion or hormone transition.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_hormone_neural_rhythm_registry import AXIS_GROUPS, affect_hormone_axis_registry, no_runtime_mutation_proof
from adapters.affect_event_to_axis_proposal_map import FEATURE_TRACK, ROUND781_800_VERSION

_REQUIRED_BOUNDARIES: tuple[str, ...] = ("saturation", "decay", "refractory", "appraisal_gate")

_SPECIFIC_INTERACTIONS: dict[str, dict[str, Any]] = {
    "social_pain": {
        "may_amplify": ("recovery_need", "expression_inhibition"),
        "may_dampen": ("expression_pressure", "action_readiness"),
        "opponent_axes": ("social_trust", "patience_level"),
        "forbidden_direct_effects": ("self_respect_decrease", "core_identity_write", "long_term_memory_write"),
        "notes": ("may increase recovery_need and expression_inhibition", "cannot lower self_respect directly"),
    },
    "threat_pressure": {
        "may_amplify": ("self_protection", "boundary_defense", "recovery_need"),
        "may_dampen": ("risk_tolerance", "expression_pressure"),
        "opponent_axes": ("social_trust", "patience_level"),
        "forbidden_direct_effects": ("neutral_input_hostile_relabel", "core_identity_write"),
        "notes": ("cannot relabel neutral input as hostile by itself",),
    },
    "curiosity_drive": {
        "may_amplify": ("novelty_seeking", "learning_pressure"),
        "may_dampen": ("uncertainty_pressure",),
        "bounded_by": ("fatigue_pressure", "overload_risk", "self_protection", *_REQUIRED_BOUNDARIES),
        "opponent_axes": ("fatigue_pressure", "overload_risk", "self_protection"),
        "notes": ("bounded by fatigue_pressure, overload_risk, and self_protection",),
    },
    "competence_drive": {
        "may_amplify": ("learning_pressure",),
        "may_dampen": ("uncertainty_pressure",),
        "forbidden_direct_effects": ("risk_tolerance_overboost", "attachment_overboost"),
        "notes": ("praise may increase competence_drive slightly but cannot overboost risk_tolerance or attachment",),
    },
    "social_trust": {
        "may_amplify": ("care_drive",),
        "may_dampen": ("trust_risk", "uncertainty_pressure"),
        "forbidden_direct_effects": ("attachment_overboost", "risk_tolerance_overboost"),
        "notes": ("praise may increase social_trust slightly within bounded appraisal gates",),
    },
    "prediction_error_pressure": {
        "may_amplify": ("learning_pressure", "uncertainty_pressure"),
        "may_dampen": (),
        "forbidden_direct_effects": ("memory_write_without_appraisal", "self_model_write_without_appraisal"),
        "notes": ("useful_criticism requires appraisal before memory/self-model update",),
    },
    "recovery_need": {
        "may_amplify": ("stability_need", "patience_level"),
        "may_dampen": ("action_readiness", "expression_pressure"),
        "opponent_axes": ("action_readiness", "expression_pressure"),
        "forbidden_direct_effects": ("panic",),
        "notes": ("dampens action and expression rather than causing panic",),
    },
    "stability_need": {
        "may_amplify": ("patience_level",),
        "may_dampen": ("stress_load", "uncertainty_pressure", "overload_risk"),
        "opponent_axes": ("stress_load", "uncertainty_pressure"),
        "notes": ("may dampen stress_load and uncertainty_pressure",),
    },
    "energy_budget": {
        "may_amplify": (),
        "may_dampen": ("fatigue_pressure", "overload_risk"),
        "bounded_by": ("hardware_governor_non_panic_policy", *_REQUIRED_BOUNDARIES),
        "forbidden_direct_effects": ("panic", "existential_threat", "social_axis_change", "identity_axis_change"),
        "notes": ("hardware governor states adjust operational axes only within non-panic bounds",),
    },
    "fatigue_pressure": {
        "may_amplify": ("recovery_need",),
        "may_dampen": ("action_readiness", "expression_pressure", "curiosity_drive"),
        "bounded_by": ("hardware_governor_non_panic_policy", *_REQUIRED_BOUNDARIES),
        "opponent_axes": ("curiosity_drive", "action_readiness"),
        "forbidden_direct_effects": ("panic",),
        "notes": ("hardware and overload effects remain non-panic and operational",),
    },
    "overload_risk": {
        "may_amplify": ("recovery_need", "stability_need", "expression_inhibition"),
        "may_dampen": ("action_readiness", "novelty_seeking"),
        "bounded_by": ("hardware_governor_non_panic_policy", *_REQUIRED_BOUNDARIES),
        "opponent_axes": ("novelty_seeking", "action_readiness"),
        "forbidden_direct_effects": ("panic", "global_synchrony"),
        "notes": ("overload proposes cooldown, not global panic",),
    },
}

_DEFAULT_BY_GROUP: dict[str, dict[str, Any]] = {
    "survival_stability": {"may_amplify": ("stability_need",), "may_dampen": ("action_readiness",), "opponent_axes": ("overload_risk",)},
    "risk_defense": {"may_amplify": ("self_protection",), "may_dampen": ("risk_tolerance",), "opponent_axes": ("social_trust",)},
    "social_relationship": {"may_amplify": ("care_drive",), "may_dampen": ("trust_risk",), "opponent_axes": ("social_pain",)},
    "learning_exploration": {"may_amplify": ("learning_pressure",), "may_dampen": ("uncertainty_pressure",), "opponent_axes": ("fatigue_pressure", "overload_risk")},
    "self_identity": {"may_amplify": ("self_coherence",), "may_dampen": ("uncertainty_pressure",), "opponent_axes": ("identity_integrity",)},
    "expression_action": {"may_amplify": ("action_readiness",), "may_dampen": ("expression_pressure",), "opponent_axes": ("expression_inhibition",)},
}


def _axis_group(axis: str) -> str:
    for group, axes in AXIS_GROUPS.items():
        if axis in axes:
            return group
    raise KeyError(axis)


def _entry(axis: str) -> dict[str, Any]:
    group = _axis_group(axis)
    base = dict(_DEFAULT_BY_GROUP[group])
    base.update(_SPECIFIC_INTERACTIONS.get(axis, {}))
    bounded_by = tuple(base.get("bounded_by", _REQUIRED_BOUNDARIES))
    for required in _REQUIRED_BOUNDARIES:
        if required not in bounded_by:
            bounded_by = (*bounded_by, required)
    return {
        "axis": axis,
        "may_amplify": tuple(item for item in base.get("may_amplify", ()) if item != axis),
        "may_dampen": tuple(item for item in base.get("may_dampen", ()) if item != axis),
        "bounded_by": bounded_by,
        "opponent_axes": tuple(item for item in base.get("opponent_axes", ()) if item != axis),
        "saturation_required": True,
        "decay_required": True,
        "refractory_required": True,
        "global_synchrony_blocked": True,
        "direct_identity_write_forbidden": True,
        "forbidden_direct_effects": tuple(base.get("forbidden_direct_effects", ("core_identity_write", "self_model_write", "long_term_memory_write"))),
        "notes": tuple(base.get("notes", ("read-only bounded interaction proposal",))),
    }


def affect_hormone_interaction_matrix() -> dict[str, dict[str, Any]]:
    """Return the complete read-only interaction matrix for every registry axis."""

    return {axis: deepcopy(_entry(axis)) for axis in affect_hormone_axis_registry()}


def validate_interaction_matrix(matrix: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Validate matrix coverage and safety invariants; returns data only."""

    active = matrix or affect_hormone_interaction_matrix()
    registry_axes = set(affect_hormone_axis_registry())
    missing_axes = sorted(registry_axes - set(active))
    unknown_interaction_axes = {
        axis: sorted((set(row.get("may_amplify", ())) | set(row.get("may_dampen", ())) | set(row.get("opponent_axes", ()))) - registry_axes)
        for axis, row in active.items()
        if sorted((set(row.get("may_amplify", ())) | set(row.get("may_dampen", ())) | set(row.get("opponent_axes", ()))) - registry_axes)
    }
    required_flag_failures = {
        axis: {
            flag: row.get(flag)
            for flag in ("saturation_required", "decay_required", "refractory_required", "global_synchrony_blocked", "direct_identity_write_forbidden")
            if row.get(flag) is not True
        }
        for axis, row in active.items()
        if any(row.get(flag) is not True for flag in ("saturation_required", "decay_required", "refractory_required", "global_synchrony_blocked", "direct_identity_write_forbidden"))
    }
    boundary_failures = {
        axis: [required for required in _REQUIRED_BOUNDARIES if required not in row.get("bounded_by", ())]
        for axis, row in active.items()
        if any(required not in row.get("bounded_by", ()) for required in _REQUIRED_BOUNDARIES)
    }
    passed = not (missing_axes or unknown_interaction_axes or required_flag_failures or boundary_failures)
    return {
        "version": ROUND781_800_VERSION,
        "feature_track": FEATURE_TRACK,
        "axis_count": len(active),
        "required_axis_count": len(registry_axes),
        "missing_axes": missing_axes,
        "unknown_interaction_axes": unknown_interaction_axes,
        "required_flag_failures": required_flag_failures,
        "boundary_failures": boundary_failures,
        "passed": passed,
        "no_runtime_mutation_proof": no_runtime_mutation_proof(),
    }


def interaction_matrix_summary() -> dict[str, Any]:
    matrix = affect_hormone_interaction_matrix()
    return {
        "version": ROUND781_800_VERSION,
        "feature_track": FEATURE_TRACK,
        "axis_count": len(matrix),
        "validation": validate_interaction_matrix(matrix),
        "required_examples": {
            "social_pain_cannot_lower_self_respect": "self_respect_decrease" in matrix["social_pain"]["forbidden_direct_effects"],
            "threat_pressure_cannot_relabel_neutral_hostile": "neutral_input_hostile_relabel" in matrix["threat_pressure"]["forbidden_direct_effects"],
            "curiosity_drive_bounded_by_fatigue_overload_self_protection": all(axis in matrix["curiosity_drive"]["bounded_by"] for axis in ("fatigue_pressure", "overload_risk", "self_protection")),
            "praise_cannot_overboost_risk_or_attachment": all(effect in matrix["competence_drive"]["forbidden_direct_effects"] for effect in ("risk_tolerance_overboost", "attachment_overboost")),
            "recovery_need_dampens_action_expression": all(axis in matrix["recovery_need"]["may_dampen"] for axis in ("action_readiness", "expression_pressure")),
        },
        "no_runtime_mutation_proof": no_runtime_mutation_proof(),
    }
