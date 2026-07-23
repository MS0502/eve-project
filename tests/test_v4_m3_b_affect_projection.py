from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_affect_projection import (
    ALLOWED_DRIVES,
    DRIVE_SPECS,
    NEGATIVE_TARGET_PAIRS,
    AffectProjectionError,
    AxisMapping,
    AxisObservation,
    DriveShadowPrior,
    normalize_observation,
    project_shadow_affect,
)
from scripts.audit.m3_b_affect_projection import (
    EXPECTED_DROP_AXES,
    audit_repository,
    baseline_priors,
    parse_mappings,
    synthetic_observations,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m3_b_affect_projection.py"
AFFECT_PLAN = ROOT / "docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md"


def observation(
    axis: str,
    source_family: str,
    *,
    value: float = 0.75,
    confidence: float = 1.0,
) -> AxisObservation:
    return AxisObservation(
        axis=axis,
        source_family=source_family,
        value=value,
        baseline=0.50,
        floor=0.00,
        ceiling=1.00,
        confidence=confidence,
        source_snapshot_id="m3-b:test-snapshot:v1",
        source_schema_version="eve.m3-b.test-source.v1",
        source_integrity_digest=hashlib.sha256(axis.encode("utf-8")).hexdigest(),
        source_metadata=(("test", "true"),),
    )


def mapping(
    axis: str,
    source_family: str,
    drives: tuple[str, ...],
    *,
    status: str = "MAPPED",
    confidence_ruling: str = "high",
) -> AxisMapping:
    return AxisMapping(
        axis=axis,
        source_family=source_family,
        status=status,
        target_drives=drives,
        appraisals=("test_appraisal",) if status == "MAPPED" else (),
        emotions=("test_emotion",) if status == "MAPPED" else (),
        confidence_ruling=confidence_ruling,
        preservation="Preserve original value and provenance for replay.",
    )


def test_source_plan_catalog_is_exactly_63_59_4_and_all_negative_pairs_are_valid():
    mappings = parse_mappings(AFFECT_PLAN.read_text(encoding="utf-8"))
    assert len(mappings) == len({row.axis for row in mappings}) == 63
    assert sum(row.status == "MAPPED" for row in mappings) == 59
    assert tuple(row.axis for row in mappings if row.status == "PROPOSED-DROP") == EXPECTED_DROP_AXES
    assert sum(row.source_family == "legacy_mutable_hormone" for row in mappings) == 26
    assert sum(row.source_family == "read_only_affect_registry" for row in mappings) == 37
    target_pairs = {(row.axis, drive) for row in mappings for drive in row.target_drives}
    assert len(NEGATIVE_TARGET_PAIRS) == 35
    assert NEGATIVE_TARGET_PAIRS <= target_pairs


def test_observation_normalization_is_baseline_centered_bounded_and_provenance_bearing():
    high = observation("example", "read_only_affect_registry", value=0.75)
    low = observation("example", "read_only_affect_registry", value=0.25)
    assert normalize_observation(high) == (0.5, False)
    assert normalize_observation(low) == (-0.5, False)
    assert high.authority == SHADOW_AUTHORITY
    assert high.acquisition_mode == "caller_supplied_read_only"
    assert len(high.digest) == 64
    with pytest.raises(AffectProjectionError, match="within declared bounds"):
        observation("example", "read_only_affect_registry", value=1.01)
    with pytest.raises(AffectProjectionError, match="confidence"):
        observation("example", "read_only_affect_registry", confidence=1.01)


def test_mixed_polarity_is_explicit_per_axis_drive_pair_and_confidence_is_capped():
    row = mapping(
        "norepinephrine",
        "legacy_mutable_hormone",
        ("energy", "safety"),
        confidence_ruling="medium",
    )
    projection = project_shadow_affect(
        mappings=(row,),
        observations=(observation("norepinephrine", "legacy_mutable_hormone", value=0.75),),
        priors=baseline_priors(),
        elapsed_seconds=60,
    )
    axis = projection.axis_projections[0]
    assert axis.calibrated_confidence == 0.75
    assert [(item.drive, item.polarity, item.contribution) for item in axis.contributions] == [
        ("energy", 1, 0.5),
        ("safety", -1, -0.5),
    ]


def test_proposed_drop_preserves_original_and_provenance_but_has_no_contribution():
    row = mapping(
        "estrogen",
        "legacy_mutable_hormone",
        (),
        status="PROPOSED-DROP",
        confidence_ruling="low",
    )
    source = observation("estrogen", "legacy_mutable_hormone", value=0.90)
    projection = project_shadow_affect(
        mappings=(row,),
        observations=(source,),
        priors=baseline_priors(),
        elapsed_seconds=60,
    )
    axis = projection.axis_projections[0]
    assert axis.status == "PROPOSED-DROP"
    assert axis.original_value == source.value
    assert axis.observation_digest == source.digest
    assert axis.source_integrity_digest == source.source_integrity_digest
    assert axis.contributions == ()
    assert all(drive.contribution_count == 0 for drive in projection.drive_projections)


def test_complete_projection_is_deterministic_bounded_and_has_no_live_authority():
    mappings = parse_mappings(AFFECT_PLAN.read_text(encoding="utf-8"))
    observations = synthetic_observations(mappings)
    priors = baseline_priors()
    first = project_shadow_affect(
        mappings=mappings,
        observations=observations,
        priors=priors,
        elapsed_seconds=120,
        strict=True,
    )
    second = project_shadow_affect(
        mappings=mappings,
        observations=observations,
        priors=priors,
        elapsed_seconds=120,
        strict=True,
    )
    assert first.to_mapping() == second.to_mapping()
    assert first.digest == second.digest
    assert len(first.axis_projections) == 63
    assert len(first.drive_projections) == 8
    assert first.missing_axes == ()
    assert first.proposed_drop_axes == EXPECTED_DROP_AXES
    assert first.authority == SHADOW_AUTHORITY
    assert first.legacy_runtime_authoritative is True
    assert first.persistence_accessed is False
    assert first.event_append_performed is False
    assert first.live_behavior_changed is False
    assert first.cutover_authorized is False
    assert first.m3_authority_open is False
    for drive in first.drive_projections:
        spec = DRIVE_SPECS[drive.drive]
        assert spec.floor <= drive.next_value <= spec.ceiling
        assert abs(drive.next_value - drive.previous_value) <= spec.max_slew_per_second * 120 + 1e-12
        assert drive.named_state_mutated is False
        assert drive.event_emitted is False


def test_missing_input_fails_strict_and_is_visible_in_non_strict_debug_projection():
    rows = (
        mapping("energy_budget", "read_only_affect_registry", ("energy",)),
        mapping("stress_load", "read_only_affect_registry", ("safety",)),
    )
    sources = (observation("energy_budget", "read_only_affect_registry"),)
    with pytest.raises(AffectProjectionError, match="missing axes"):
        project_shadow_affect(
            mappings=rows,
            observations=sources,
            priors=baseline_priors(),
            elapsed_seconds=60,
            strict=True,
        )
    projection = project_shadow_affect(
        mappings=rows,
        observations=sources,
        priors=baseline_priors(),
        elapsed_seconds=60,
        strict=False,
    )
    assert projection.missing_axes == ("stress_load",)
    assert projection.axis_projections[1].missing_input is True
    assert projection.axis_projections[1].calibrated_confidence == 0.0


def test_transition_candidate_is_diagnostic_only_and_pending_identity_prevents_duplicate():
    row = mapping("expression_pressure", "read_only_affect_registry", ("expression",))
    source = observation("expression_pressure", "read_only_affect_registry", value=1.0)
    priors = list(baseline_priors())
    expression_index = ALLOWED_DRIVES.index("expression")
    priors[expression_index] = DriveShadowPrior(
        drive="expression",
        value=0.47,
        named_state="forming",
        state_epoch=7,
        seconds_since_transition=30,
    )
    projection = project_shadow_affect(
        mappings=(row,),
        observations=(source,),
        priors=priors,
        elapsed_seconds=30,
    )
    drive = projection.drive_projections[expression_index]
    assert drive.next_value >= 0.48
    assert drive.named_state_retained == "forming"
    assert drive.state_epoch_retained == 7
    assert drive.named_state_mutated is False
    assert drive.event_emitted is False
    assert drive.candidate is not None
    assert drive.candidate.from_state == "forming"
    assert drive.candidate.to_state == "ready"
    assert drive.candidate.next_state_epoch == 8
    assert drive.candidate.diagnostic_only is True
    assert drive.candidate.event_append_authorized is False

    priors[expression_index] = replace(priors[expression_index], pending_candidate_id="a" * 64)
    duplicate_blocked = project_shadow_affect(
        mappings=(row,),
        observations=(source,),
        priors=priors,
        elapsed_seconds=30,
    )
    blocked_drive = duplicate_blocked.drive_projections[expression_index]
    assert blocked_drive.pending_candidate_retained is True
    assert blocked_drive.candidate is None


def test_audit_report_is_recalculable_deterministic_and_error_free():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["deterministic_replay_equal"] is True
    assert first["mapping_summary"]["axis_count"] == 63
    assert first["mapping_summary"]["status_counts"] == {"MAPPED": 59, "PROPOSED-DROP": 4}
    assert first["projection_summary"]["drive_projection_count"] == 8
    assert first["projection_summary"]["drop_preserved_count"] == 4
    assert first["projection_summary"]["drop_contribution_count"] == 0
    assert first["projection_summary"]["event_emission_count"] == 0
    assert first["projection_summary"]["named_state_mutation_count"] == 0
    assert len(first["projection_summary"]["projection_digest"]) == 64
    assert len(first["report_digest"]) == 64


def test_core_import_has_no_io_persistence_runtime_or_live_mutation_surface():
    tree = ast.parse(CORE_MODULE.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imports & {
        "adapters",
        "hormone_system",
        "language",
        "main",
        "os",
        "pathlib",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not calls & {
        "append_event",
        "connect",
        "mkdir",
        "open",
        "start",
        "write",
        "write_text",
    }
