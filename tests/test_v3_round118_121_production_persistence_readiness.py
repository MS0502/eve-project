from __future__ import annotations

import json
from pathlib import Path

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_mapping_production_readiness import (
    ROUND118_PRODUCTION_PERSISTENCE_READINESS_AUDIT_VERSION,
    ROUND119_MINIMAL_ENABLEMENT_RISK_MATRIX_VERSION,
    ROUND120_FINAL_PRE_ACTIVATION_GATE_PACKAGE_VERSION,
    ROUND121_REQUIRED_BLOCKER_ISOLATION_VERSION,
    build_round118_production_persistence_readiness_audit,
    build_round119_minimal_enablement_risk_matrix_operator_checklist,
    build_round120_final_pre_activation_gate_package,
    build_round121_required_blocker_isolation,
    write_round118_production_persistence_readiness_audit,
    write_round119_minimal_enablement_risk_matrix_operator_checklist,
    write_round120_final_pre_activation_gate_package,
    write_round121_required_blocker_isolation,
)


class _DummyStore:
    def stats(self) -> dict:
        return {"count": 1, "dimension": 300, "read_only": True}


class _DummyWrapper:
    def telemetry(self) -> dict:
        return {"total_calls": 0, "primary_hits": 0, "fallback_uses": 0, "eve_specific_hits": 0}


class _DummySA:
    def __init__(self) -> None:
        self.activations = {"concept_category::lex::민석": 1.0}


class _DummyActivation:
    def __init__(self) -> None:
        self.sa = _DummySA()


class _DummyStats:
    def __init__(self, name: str) -> None:
        self.name = name

    def stats(self) -> dict:
        return {"name": self.name, "read_only": True}


class _DummyEngine:
    def __init__(self) -> None:
        self.lex_concept_mapping = LexConceptMappingAdapter(self)
        self.eve_specific_vector_store = _DummyStore()
        self.self_embedding = _DummyWrapper()
        self.activation_adapter = _DummyActivation()
        self.concept_memory = _DummyStats("concept_memory")
        self.agp = _DummyStats("agp")


def _round117_package() -> dict:
    return {
        "go_no_go_package_version": "v3_round117_operator_go_no_go_package",
        "round": 117,
        "completed_rounds": [113, 114, 115, 116],
        "operator_recommendation": "no_go_for_production_persistence_in_this_pr",
        "decision_blockers": ["broader_validation_blocked_or_partial", "legacy_root_collection_blocker"],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
    }


def test_round118_readiness_audit_returns_no_go_without_enabling_flags(tmp_path: Path) -> None:
    engine = _DummyEngine()
    audit = build_round118_production_persistence_readiness_audit(
        engine=engine,
        source_go_no_go_package=_round117_package(),
        validation_summary=[{"command": "pytest -q tests", "status": "partial"}],
        artifact_roots=[tmp_path],
    )

    assert audit["readiness_audit_version"] == ROUND118_PRODUCTION_PERSISTENCE_READINESS_AUDIT_VERSION
    assert audit["audit_status"] == "production_persistence_not_ready_no_go"
    assert audit["readiness_decision"] == "no_go"
    assert "broader_validation_blocked_or_partial" in audit["readiness_blockers"]
    assert audit["operator_recommendation"] == "NO-GO"
    assert audit["production_persistence_enabled"] is False
    assert audit["runtime_mapping_enabled_default"] is False
    assert audit["enforcement_enabled_default"] is False
    assert audit["vectors_npy_committed"] is False
    assert audit["agp_bypass_used"] is False
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False


def test_round118_detects_forbidden_artifact_boundaries(tmp_path: Path) -> None:
    (tmp_path / "vectors.npy").write_bytes(b"not staged fixture")
    audit = build_round118_production_persistence_readiness_audit(
        engine=_DummyEngine(),
        source_go_no_go_package=_round117_package(),
        artifact_roots=[tmp_path],
    )

    assert "forbidden_artifact_detected" in audit["readiness_blockers"]
    assert audit["forbidden_artifact_paths"] == ["vectors.npy"]
    assert audit["checks"]["forbidden_artifacts_absent"] is False


def test_round119_risk_matrix_and_operator_checklist_keep_activation_blocked() -> None:
    audit = build_round118_production_persistence_readiness_audit(
        engine=_DummyEngine(),
        source_go_no_go_package=_round117_package(),
        validation_summary=[{"command": "pytest -q tests", "status": "partial"}],
    )
    risk = build_round119_minimal_enablement_risk_matrix_operator_checklist(source_readiness_audit=audit)

    assert risk["risk_matrix_version"] == ROUND119_MINIMAL_ENABLEMENT_RISK_MATRIX_VERSION
    assert risk["risk_matrix_status"] == "minimal_enablement_risk_matrix_ready"
    assert risk["operator_recommendation"] == "NO-GO"
    assert "C2" in risk["unsatisfied_required_checklist_items"]
    assert "C5" in risk["unsatisfied_required_checklist_items"]
    assert all(item["activation_allowed_before_resolution"] is False for item in risk["risk_matrix"])
    assert risk["production_persistence_enabled"] is False
    assert risk["runtime_mapping_enabled_default"] is False
    assert risk["enforcement_enabled_default"] is False


def test_round120_final_gate_no_go_without_operator_approval_and_broader_validation() -> None:
    audit = build_round118_production_persistence_readiness_audit(
        engine=_DummyEngine(),
        source_go_no_go_package=_round117_package(),
        validation_summary=[{"command": "pytest -q tests", "status": "partial"}],
    )
    risk = build_round119_minimal_enablement_risk_matrix_operator_checklist(source_readiness_audit=audit)

    gate = build_round120_final_pre_activation_gate_package(
        source_readiness_audit=audit,
        source_risk_matrix=risk,
        explicit_operator_approval_present=False,
    )

    assert gate["gate_package_version"] == ROUND120_FINAL_PRE_ACTIVATION_GATE_PACKAGE_VERSION
    assert gate["gate_status"] == "pre_activation_gate_no_go"
    assert gate["final_recommendation"] == "NO-GO"
    assert "explicit_operator_approval_missing" in gate["hard_blocks"]
    assert "broader_validation_blocked_or_partial" in gate["hard_blocks"]
    assert gate["activation_action_taken"] is False
    assert gate["must_not_activate_without_explicit_operator_approval"] is True
    assert gate["production_persistence_enabled"] is False


def test_round121_isolates_no_go_blockers_without_activation() -> None:
    audit = build_round118_production_persistence_readiness_audit(
        engine=_DummyEngine(),
        source_go_no_go_package=_round117_package(),
        validation_summary=[{"command": "pytest -q tests", "status": "partial"}],
    )
    risk = build_round119_minimal_enablement_risk_matrix_operator_checklist(source_readiness_audit=audit)
    gate = build_round120_final_pre_activation_gate_package(source_readiness_audit=audit, source_risk_matrix=risk)

    isolation = build_round121_required_blocker_isolation(source_gate_package=gate)

    assert isolation["blocker_isolation_version"] == ROUND121_REQUIRED_BLOCKER_ISOLATION_VERSION
    assert isolation["isolation_status"] == "required_blockers_isolated_no_activation"
    assert isolation["final_recommendation_inherited"] == "NO-GO"
    assert {item["blocker"] for item in isolation["required_blockers"]} >= {
        "broader_validation_blocked_or_partial",
        "explicit_operator_approval_missing",
    }
    assert all(item["blocks_real_persistence"] is True for item in isolation["required_blockers"])
    assert isolation["activation_action_taken"] is False
    assert isolation["runtime_mapping_enabled_default"] is False
    assert isolation["enforcement_enabled_default"] is False


def test_round118_121_export_helpers_write_json(tmp_path: Path) -> None:
    audit = build_round118_production_persistence_readiness_audit(
        engine=_DummyEngine(),
        source_go_no_go_package=_round117_package(),
        validation_summary=[{"command": "pytest -q tests", "status": "partial"}],
    )
    risk = build_round119_minimal_enablement_risk_matrix_operator_checklist(source_readiness_audit=audit)
    gate = build_round120_final_pre_activation_gate_package(source_readiness_audit=audit, source_risk_matrix=risk)
    isolation = build_round121_required_blocker_isolation(source_gate_package=gate)

    exports = [
        write_round118_production_persistence_readiness_audit(tmp_path / "round118.json", audit),
        write_round119_minimal_enablement_risk_matrix_operator_checklist(tmp_path / "round119.json", risk),
        write_round120_final_pre_activation_gate_package(tmp_path / "round120.json", gate),
        write_round121_required_blocker_isolation(tmp_path / "round121.json", isolation),
    ]

    assert all(export["json_written"] is True for export in exports)
    assert all(export["runtime_mapping_enabled"] is False for export in exports)
    assert all(export["enforcement_enabled"] is False for export in exports)
    assert json.loads((tmp_path / "round120.json").read_text(encoding="utf-8"))["final_recommendation"] == "NO-GO"
