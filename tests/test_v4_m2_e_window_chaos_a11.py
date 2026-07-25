from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from core.event_kernel import MAX_CANONICAL_JSON_BYTES, EventEnvelope, InvalidEventEnvelope, canonical_json_object
from core.m2_e_window_driver import WindowConfig, WindowState, advance_runtime, freeze_shadow, record_discrete_stimulus
from core.sqlite_shadow_store import SQLiteShadowStore
from core import sqlite_shadow_store_v1 as legacy
from core.sqlite_shadow_store_a11 import (
    CONTENT_REFERENCE_SCHEMA_VERSION,
    EVENT_STORAGE_REFERENCE_SCHEMA_VERSION,
)
from scripts.habitat import m2_e_window_runtime_guarded as runtime


def test_exact_legacy_v1_store_gets_additive_content_table_and_reads_legacy_snapshot(tmp_path: Path):
    path = tmp_path / "legacy.sqlite3"
    old = legacy.SQLiteShadowStore(path)
    old.initialize()
    expected_state = {"calls": [], "learned": []}
    old.write_snapshot(
        snapshot_id="legacy:snapshot:zero",
        stream_id="legacy:stream",
        through_sequence=0,
        state=expected_state,
        state_schema_version="legacy.state.v1",
    )

    store = SQLiteShadowStore(path)
    store.initialize()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name='content_materials'"
        ).fetchone()[0] == 1
    finally:
        connection.close()

    selection = store.latest_valid_snapshot("legacy:stream")
    assert selection.selected is not None
    assert selection.selected.state == expected_state
    assert store.integrity_check().valid is True


def test_large_snapshot_uses_a11_digest_manifest_without_relaxing_event_limit(tmp_path: Path):
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    large_state = runtime._snapshot_for(700)

    with pytest.raises(InvalidEventEnvelope, match="exceeds canonical size limit"):
        canonical_json_object(large_state, field="legacy_inline_snapshot")

    receipt = store.write_snapshot(
        snapshot_id="a11:snapshot:large",
        stream_id="m2e:a11-large",
        through_sequence=0,
        state=large_state,
        state_schema_version="eve.m2e.synthetic-state.v1",
    )
    assert receipt.readback_verified is True

    connection = sqlite3.connect(store.database_path)
    try:
        state_json, state_digest, manifest_json = connection.execute(
            "SELECT state_json,state_digest,manifest_json FROM snapshots WHERE snapshot_id=?",
            ("a11:snapshot:large",),
        ).fetchone()
        reference = json.loads(state_json)
        manifest = json.loads(manifest_json)
        material_json, material_bytes = connection.execute(
            "SELECT material_json,material_bytes FROM content_materials WHERE content_digest=?",
            (state_digest,),
        ).fetchone()
    finally:
        connection.close()

    assert reference["reference_schema_version"] == CONTENT_REFERENCE_SCHEMA_VERSION
    assert reference["content_digest"] == state_digest
    assert manifest["content_schema_version"] == "eve.m2e.synthetic-state.v1"
    assert manifest["collection_counts"] == {"calls": 700, "learned": 700}
    assert manifest["top_level_keys"] == ["calls", "learned"]
    assert len(state_json.encode("utf-8")) < MAX_CANONICAL_JSON_BYTES
    assert material_bytes == len(material_json.encode("utf-8"))
    assert hashlib.sha256(material_json.encode("utf-8")).hexdigest() == state_digest
    assert store.latest_valid_snapshot("m2e:a11-large").selected.state == large_state
    assert store.integrity_check().valid is True

    assert MAX_CANONICAL_JSON_BYTES == 65_536
    with pytest.raises(InvalidEventEnvelope, match="payload exceeds canonical size limit"):
        EventEnvelope.create(
            event_id="oversized:event:1",
            event_type="shadow.test",
            stream_id="shadow:oversized",
            sequence=1,
            producer="tests.a11",
            producer_version="1.0.0",
            correlation_id="oversized:corr",
            payload={"blob": "x" * MAX_CANONICAL_JSON_BYTES},
            causal_context={"phase": "chaos"},
        )


def test_seq_280_uses_content_reference_and_reviewed_resume_reconciles_pending_row(tmp_path: Path):
    runtime._enable_io_failure_reason()
    root = tmp_path / "private"
    paths = runtime._ensure_private_root(root)
    config = WindowConfig()
    store = runtime._store(paths, config)

    legacy._canon(legacy._event_material(runtime._event(279)), "event_material")
    with pytest.raises(InvalidEventEnvelope, match="event_material exceeds canonical size limit"):
        legacy._canon(legacy._event_material(runtime._event(280)), "event_material")

    state = WindowState.create(window_id="m2-e:a11-seq280:v1", local_date="2026-07-25")
    for _ in range(279):
        state = advance_runtime(state, elapsed_seconds=300, local_date="2026-07-25")
        state = record_discrete_stimulus(state, config=config)
    assert state.event_count == 279
    assert state.frozen is False

    store.append_many(tuple(runtime._event(sequence) for sequence in range(1, 281)))

    connection = sqlite3.connect(store.database_path)
    try:
        row_279 = json.loads(connection.execute("SELECT event_json FROM events WHERE sequence=279").fetchone()[0])
        row_280 = json.loads(connection.execute("SELECT event_json FROM events WHERE sequence=280").fetchone()[0])
    finally:
        connection.close()
    assert "payload_json" in row_279
    assert row_280["storage_schema_version"] == EVENT_STORAGE_REFERENCE_SCHEMA_VERSION
    assert row_280["payload_reference"]["reference_schema_version"] == CONTENT_REFERENCE_SCHEMA_VERSION
    assert store.events(stream_id=runtime.BOUNDED_STREAM)[-1] == runtime._event(280)
    assert store.integrity_check().valid is True

    frozen = freeze_shadow(state, "recovery_digest_mismatch")
    runtime._save_state(paths, frozen)
    paths["running"].write_text("unclean\n", encoding="utf-8")

    assert runtime.resume_reviewed(private_root=root, window_id=frozen.window_id) == 0
    resumed = runtime._load_state(paths, window_id=frozen.window_id)
    assert resumed.frozen is False
    assert resumed.event_count == 280
    assert resumed.expected_recovery_digest == resumed.last_recovery_digest

    records = [
        json.loads(line)
        for line in paths["raw"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    resume = next(item for item in records if item["type"] == "freeze_reviewed_resume")
    assert resume["reconciled_pending_commit"] is True
