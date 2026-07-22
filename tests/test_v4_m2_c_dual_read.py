from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m2_c_migration import (
    COMPARISON_AUTHORITY,
    DUAL_READ_REPORT_SCHEMA_VERSION,
    LEGACY_SOURCE_SCHEMA_VERSION,
    STATE_SERIALIZATION_SCHEMA_VERSION,
    LegacySidecarIncompatible,
    M2CDualReadError,
    MigrationCandidate,
    StateEvidence,
    assess_legacy_sidecar,
    build_migration_candidate,
    compare_dual_read,
)
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)
from core.shadow_projection import PROJECTION_SCHEMA_VERSION
from core.sqlite_shadow_store import SQLiteShadowStore

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m2_c_migration.py"
EMPTY = {"calls": [], "learned": []}
PAIR = ["alpha", "beta", 0.4]
AFTER_ONE = {"calls": [PAIR], "learned": [PAIR]}


def observed_event(
    sequence: int,
    *,
    before: dict,
    after: dict,
    event_id: str | None = None,
) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=event_id or f"m2c:event:{sequence}",
        event_type=SUCCESS_EVENT_TYPE,
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        sequence=sequence,
        producer=OBSERVER_PRODUCER,
        producer_version=OBSERVER_VERSION,
        correlation_id="m2c:correlation",
        causation_id=None,
        payload={
            "after": after,
            "before": before,
            "legacy_outcome": {"error_type": None, "succeeded": True},
            "target": {
                "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
                "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
                "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
                "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            },
        },
        causal_context={
            "arguments_captured": False,
            "legacy_result_captured": False,
            "observation_phase": "after_the_fact",
            "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
        },
    )


def assessment(snapshot: dict = AFTER_ONE):
    return assess_legacy_sidecar(
        source_label="fixture:activation-sidecar",
        source_bytes=b"detached legacy sidecar fixture",
        source_schema_version=LEGACY_SOURCE_SCHEMA_VERSION,
        decoded_snapshot=snapshot,
    )


def initialized_store(tmp_path: Path) -> SQLiteShadowStore:
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    return store


def state_evidence_parts(snapshot: dict, *, canonical: bool = True) -> tuple[str, str]:
    if canonical:
        snapshot_json = canonical_json_object(snapshot, field="test_state")
    else:
        snapshot_json = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_json = canonical_json_object(
        {
            "canonical_bytes": len(snapshot_json.encode("utf-8")),
            "collection_counts": {
                "calls": len(snapshot["calls"]),
                "learned": len(snapshot["learned"]),
            },
            "hash_algorithm": "sha256",
            "key_domain": ["calls", "learned"],
            "serialization_schema": STATE_SERIALIZATION_SCHEMA_VERSION,
            "state_schema_version": PROJECTION_SCHEMA_VERSION,
            "top_level_key_count": 2,
        },
        field="test_manifest",
    )
    return snapshot_json, manifest_json


def test_assessment_hashes_source_and_normalizes_only_bounded_snapshot():
    result = assessment()
    assert result.compatible is True
    assert result.incompatibilities == ()
    assert result.source_sha256 == hashlib.sha256(
        b"detached legacy sidecar fixture"
    ).hexdigest()
    assert result.state is not None
    assert result.state.snapshot == AFTER_ONE
    assert result.legacy_authority_retained is True
    assert result.runtime_integrated is False


def test_assessment_reports_empty_schema_and_snapshot_incompatibilities():
    result = assess_legacy_sidecar(
        source_label="fixture:bad",
        source_bytes=b"",
        source_schema_version="legacy.unknown.v9",
        decoded_snapshot={"unexpected": []},
    )
    assert result.compatible is False
    assert result.incompatibilities == (
        "empty_source_bytes",
        "invalid_bounded_snapshot",
        "unsupported_source_schema",
    )
    assert result.state is None


def test_state_evidence_revalidates_nested_bounded_state_on_direct_construction():
    malformed = {"calls": [["alpha"]], "learned": []}
    snapshot_json, manifest_json = state_evidence_parts(malformed)
    with pytest.raises(M2CDualReadError, match="invalid bounded state evidence"):
        StateEvidence(
            snapshot_json=snapshot_json,
            snapshot_digest=hashlib.sha256(snapshot_json.encode()).hexdigest(),
            manifest_json=manifest_json,
            manifest_digest=hashlib.sha256(manifest_json.encode()).hexdigest(),
        )


def test_state_evidence_rejects_noncanonical_json_even_with_matching_manifest():
    snapshot_json, manifest_json = state_evidence_parts(AFTER_ONE, canonical=False)
    with pytest.raises(M2CDualReadError, match="canonical serialization"):
        StateEvidence(
            snapshot_json=snapshot_json,
            snapshot_digest=hashlib.sha256(snapshot_json.encode()).hexdigest(),
            manifest_json=manifest_json,
            manifest_digest=hashlib.sha256(manifest_json.encode()).hexdigest(),
        )


def test_migration_candidate_is_content_addressed_comparison_only():
    candidate = build_migration_candidate(assessment())
    assert candidate.authority == COMPARISON_AUTHORITY
    assert candidate.stream_id == ACTIVATION_LEARN_PAIR_TARGET.stream_id
    assert candidate.legacy_state.manifest_digest
    assert candidate.source_schema_version == LEGACY_SOURCE_SCHEMA_VERSION
    assert candidate.writes_performed is False
    assert candidate.runtime_integrated is False
    assert candidate.legacy_authority_retained is True
    assert len(candidate.candidate_digest) == 64


def test_migration_candidate_requires_positive_source_bytes():
    candidate = build_migration_candidate(assessment())
    with pytest.raises(M2CDualReadError, match="source_byte_count must be positive"):
        MigrationCandidate(
            source_label=candidate.source_label,
            source_sha256=candidate.source_sha256,
            source_byte_count=0,
            source_schema_version=candidate.source_schema_version,
            legacy_state=candidate.legacy_state,
            stream_id=candidate.stream_id,
            candidate_digest=candidate.candidate_digest,
        )


def test_incompatible_evidence_cannot_become_migration_candidate():
    bad = assess_legacy_sidecar(
        source_label="fixture:bad",
        source_bytes=b"bad",
        source_schema_version="legacy.unknown.v9",
        decoded_snapshot=AFTER_ONE,
    )
    with pytest.raises(LegacySidecarIncompatible):
        build_migration_candidate(bad)


def test_dual_read_matches_replayed_shadow_without_logical_write(tmp_path: Path):
    store = initialized_store(tmp_path)
    store.append(observed_event(1, before=EMPTY, after=AFTER_ONE))
    before = store.integrity_check()
    events_before = store.events(stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id)

    report = compare_dual_read(
        assessment=assessment(),
        store=store,
        initial_snapshot=EMPTY,
    )

    after = store.integrity_check()
    assert report.schema_version == DUAL_READ_REPORT_SCHEMA_VERSION
    assert report.replay_verified is True
    assert report.matches is True
    assert report.mismatches == ()
    assert report.incompatibilities == ()
    assert report.shadow_event_count == 1
    assert report.shadow_sequence == 1
    assert report.shadow_authority == SHADOW_AUTHORITY
    assert report.comparison_authority == COMPARISON_AUTHORITY
    assert report.legacy_authority_retained is True
    assert report.runtime_integrated is False
    assert report.state_changed is False
    assert report.writes_performed is False
    assert len(report.transition_hash) == 64
    assert report.legacy_state.snapshot_json
    assert report.legacy_state.manifest_json
    assert report.shadow_state is not None
    assert report.shadow_state.snapshot_json
    assert report.shadow_state.manifest_json
    assert before.report_digest == report.shadow_integrity_before_digest
    assert after.report_digest == report.shadow_integrity_after_digest
    assert before.report_digest == after.report_digest
    assert store.events(stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id) == events_before


def test_dual_read_reports_exact_state_mismatch_without_authority_change(tmp_path: Path):
    store = initialized_store(tmp_path)
    store.append(observed_event(1, before=EMPTY, after=AFTER_ONE))
    legacy = {"calls": [PAIR], "learned": []}
    report = compare_dual_read(
        assessment=assessment(legacy),
        store=store,
        initial_snapshot=EMPTY,
    )
    assert report.replay_verified is True
    assert report.matches is False
    assert report.mismatches == ("learned_mismatch",)
    assert report.incompatibilities == ()
    assert report.writes_performed is False
    assert report.legacy_authority_retained is True


def test_external_store_change_is_not_attributed_as_comparison_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = initialized_store(tmp_path)
    store.append(observed_event(1, before=EMPTY, after=AFTER_ONE))
    original = SQLiteShadowStore.integrity_check
    calls = 0

    def changed_on_second_check(self: SQLiteShadowStore):
        nonlocal calls
        calls += 1
        report = original(self)
        if calls == 2:
            return replace(report, report_digest="f" * 64)
        return report

    monkeypatch.setattr(SQLiteShadowStore, "integrity_check", changed_on_second_check)
    report = compare_dual_read(
        assessment=assessment(),
        store=store,
        initial_snapshot=EMPTY,
    )
    assert report.state_changed is True
    assert report.writes_performed is False
    assert report.matches is False
    assert report.incompatibilities == ("shadow_store_changed_during_comparison",)


def test_uninitialized_store_is_visible_incompatibility_not_auto_initialized(tmp_path: Path):
    path = tmp_path / "never-created.sqlite3"
    store = SQLiteShadowStore(path)
    report = compare_dual_read(
        assessment=assessment(EMPTY),
        store=store,
        initial_snapshot=EMPTY,
    )
    assert report.replay_verified is False
    assert report.matches is False
    assert any(
        item.startswith("shadow_integrity_before:sqlite:StoreNotInitialized:")
        for item in report.incompatibilities
    )
    assert any(
        item.startswith("shadow_integrity_after:sqlite:StoreNotInitialized:")
        for item in report.incompatibilities
    )
    assert report.shadow_event_count == 0
    assert report.shadow_sequence is None
    assert path.exists() is False


def test_replay_contract_failure_is_reported_fail_closed(tmp_path: Path):
    store = initialized_store(tmp_path)
    unrelated = EventEnvelope.create(
        event_id="unrelated:1",
        event_type="shadow.unrelated",
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        sequence=1,
        producer="tests.m2c.invalid",
        producer_version="1.0.0",
        correlation_id="m2c:invalid",
        causation_id=None,
        payload={"value": 1},
        causal_context={"phase": "invalid"},
    )
    store.append(unrelated)
    report = compare_dual_read(
        assessment=assessment(EMPTY),
        store=store,
        initial_snapshot=EMPTY,
    )
    assert report.replay_verified is False
    assert report.matches is False
    assert report.mismatches == ()
    assert report.incompatibilities == ("shadow_replay:UnsupportedProjectionEvent",)
    assert report.writes_performed is False


def test_module_has_no_file_discovery_deserialization_or_store_write_calls():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imports & {"pickle", "gzip", "sqlite3", "pathlib", "os", "glob"}

    direct_calls: set[str] = set()
    store_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            direct_calls.add(node.func.id)
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "store"
        ):
            store_calls.add(node.func.attr)

    assert not direct_calls & {"open", "exec", "eval", "compile", "__import__"}
    assert not store_calls & {
        "initialize",
        "append",
        "append_many",
        "write_snapshot",
        "create_backup",
    }
