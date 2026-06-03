"""EVE v3 Round107 — runtime mapping persistence activation dry-run harness."""

from pathlib import Path

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_mapping_persistence_activation_dryrun import (
    ROUND107_ACTIVATION_DRYRUN_VERSION,
    audit_log_schema,
    build_runtime_mapping_persistence_activation_dryrun,
    checkpoint_format,
    future_activation_touch_plan,
    rollback_format,
    state_debug_export_surface,
    write_round107_runtime_mapping_persistence_activation_dryrun,
)
from adapters.state_debug_adapter import StateDebugAdapter


class _DummyStore:
    def __init__(self) -> None:
        self._count = 1

    def stats(self) -> dict:
        return {"count": self._count, "dimension": 300, "read_only": True}


class _DummyWrapper:
    def __init__(self) -> None:
        self.calls = 0

    def telemetry(self) -> dict:
        return {"total_calls": self.calls, "primary_hits": 0, "fallback_uses": 0, "eve_specific_hits": 0}


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


def _source_decision() -> dict:
    return {
        "decision_version": "v3_round106_runtime_mapping_persistence_decision",
        "decision_status": "persistence_ready_but_not_applied",
        "decision_ready": True,
        "mapped_tokens": ["민석"],
    }


def test_round107_defines_checkpoint_rollback_audit_and_touch_formats() -> None:
    checkpoint = checkpoint_format()
    rollback = rollback_format()
    audit = audit_log_schema()
    touch = future_activation_touch_plan()
    debug = state_debug_export_surface()

    assert checkpoint["schema_version"].endswith(".checkpoint.v1")
    assert "runtime_mapping_enabled_before" in checkpoint["required_fields"]
    assert rollback["rollback_actions"][0] == "set LexConceptMappingAdapter.runtime_mapping_enabled=False"
    assert audit["format"] == "jsonl_append_only_for_future_activation_patch"
    assert "adapters/lex_concept_mapping_adapter.py" in [row["path"] for row in touch["files"]]
    assert "main.py" in [row["path"] for row in touch["files"]]
    assert debug["state_key"] == "lex_concept_mapping.runtime_mapping_persistence_activation"
    assert checkpoint["read_only"] is True
    assert rollback["read_only"] is True
    assert audit["read_only"] is True


def test_round107_dry_run_proves_no_runtime_or_agp_vector_category_memory_mutation() -> None:
    engine = _DummyEngine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_audit = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats()
    before_active = dict(engine.activation_adapter.sa.activations)

    report = build_runtime_mapping_persistence_activation_dryrun(engine=engine, source_decision=_source_decision())

    assert report["dry_run_version"] == ROUND107_ACTIVATION_DRYRUN_VERSION
    assert report["activation_status"] == "dry_run_only_not_enabled"
    assert report["mapped_tokens"] == ["민석"]
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled_default"] is False
    assert lcm.runtime_mapping_enabled is False
    assert lcm.enforcement_enabled is False
    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert dict(engine.activation_adapter.sa.activations) == before_active

    checks = report["dry_run_execution"]["mutation_checks"]
    assert checks["runtime_mapping_enabled_default_unchanged"] is True
    assert checks["enforcement_enabled_default_unchanged"] is True
    assert checks["concept_categories_unchanged"] is True
    assert checks["concept_commit_records_unchanged"] is True
    assert checks["eve_specific_vector_store_unchanged"] is True
    assert checks["sa_active_categories_unchanged"] is True
    assert checks["agp_verify_called"] is False
    assert checks["embedding_lookup_called"] is False
    assert checks["vectors_npy_committed"] is False
    assert report["dry_run_execution"]["no_agp_vector_category_memory_mutation"] is True


def test_round107_state_debug_surface_is_exported_without_enabling_mapping() -> None:
    engine = _DummyEngine()
    snapshot = StateDebugAdapter(engine).snapshot_state()
    lcm_state = snapshot["lex_concept_mapping"]

    assert lcm_state["runtime_mapping_persistence_activation_dryrun_version"] == ROUND107_ACTIVATION_DRYRUN_VERSION
    assert lcm_state["runtime_mapping_persistence_activation_dryrun_available"] is True
    assert lcm_state["runtime_mapping_enabled"] is False
    assert lcm_state["enforcement_enabled"] is False
    activation = lcm_state["runtime_mapping_persistence_activation"]
    assert activation["activation_dry_run_available"] is True
    assert activation["runtime_mapping_enabled_default"] is False
    assert activation["enforcement_enabled_default"] is False
    assert activation["actual_persistence_enablement"] is False


def test_round107_export_writes_json_only(tmp_path: Path) -> None:
    report = build_runtime_mapping_persistence_activation_dryrun(source_decision=_source_decision())
    export = write_round107_runtime_mapping_persistence_activation_dryrun(tmp_path / "round107.json", report)

    assert export["export_version"] == ROUND107_ACTIVATION_DRYRUN_VERSION
    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert export["runtime_mapping_persisted"] is False
    assert export["read_only"] is True
