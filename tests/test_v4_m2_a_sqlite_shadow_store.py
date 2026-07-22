from __future__ import annotations

import ast
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY
from core.sqlite_shadow_store import (
    STORE_SCHEMA_VERSION,
    AppendOnlyViolation,
    BackupPolicyError,
    PersistedEventCorruption,
    SQLiteShadowStore,
    SchemaMismatch,
    ShadowStoragePolicy,
    SnapshotCorruption,
    StoragePolicyExceeded,
    StoreNotInitialized,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/sqlite_shadow_store.py"
EVENT_UPDATE_TRIGGER = (
    "CREATE TRIGGER events_no_update BEFORE UPDATE ON events "
    "BEGIN SELECT RAISE(ABORT,'append-only events'); END"
)


def event(
    sequence: int,
    *,
    event_id: str | None = None,
    cause: str | None = None,
    stream: str = "shadow:test",
) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=event_id or f"event:{sequence}",
        event_type="shadow.test",
        stream_id=stream,
        sequence=sequence,
        producer="tests.m2a",
        producer_version="1.0.0",
        correlation_id="corr:m2a",
        causation_id=cause,
        payload={"delta": sequence},
        causal_context={"phase": "test"},
    )


def test_construction_is_io_free_and_initialize_is_explicit(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    assert not path.exists()
    with pytest.raises(StoreNotInitialized):
        store.events()
    with pytest.raises(StoreNotInitialized):
        store.create_backup(tmp_path / "backups", backup_ordinal=1)
    assert not (tmp_path / "backups").exists()
    report = store.initialize()
    assert path.exists()
    assert report.schema_version == STORE_SCHEMA_VERSION
    assert report.authority == SHADOW_AUTHORITY
    assert report.journal_mode in {"wal", "delete", "truncate", "persist", "memory", "off"}
    assert report.wal_enabled == (report.journal_mode == "wal")


def test_initial_schema_install_rolls_back_as_one_transaction_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "shadow.sqlite3"
    original = SQLiteShadowStore._insert_initial_records

    def fail_after_ddl(self: SQLiteShadowStore, connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected initialization interruption")

    monkeypatch.setattr(SQLiteShadowStore, "_insert_initial_records", fail_after_ddl)
    with pytest.raises(sqlite3.OperationalError):
        SQLiteShadowStore(path).initialize()

    connection = sqlite3.connect(path)
    try:
        objects = connection.execute(
            "SELECT name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        connection.close()
    assert objects == []

    monkeypatch.setattr(SQLiteShadowStore, "_insert_initial_records", original)
    assert SQLiteShadowStore(path).initialize().schema_version == STORE_SCHEMA_VERSION


def test_schema_migration_and_append_only_triggers_are_durable(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    SQLiteShadowStore(path).initialize()
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='store_schema_version'"
        ).fetchone()[0] == STORE_SCHEMA_VERSION
        assert connection.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 1
        for table in ("metadata", "migrations"):
            connection.execute("BEGIN")
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(f"DELETE FROM {table}")
            connection.rollback()
    finally:
        connection.close()


def test_reopen_rejects_missing_or_redefined_append_only_trigger(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER events_no_update")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SchemaMismatch):
        SQLiteShadowStore(path).initialize()
    report = store.integrity_check()
    assert report.valid is False
    assert any(item.startswith("schema:") for item in report.errors)


def test_append_is_atomic_readback_verified_and_hash_chained(tmp_path: Path):
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    first, second = event(1), event(2, cause="event:1")
    receipts = store.append_many((first, second))
    assert [item.before_count for item in receipts] == [0, 1]
    assert [item.after_count for item in receipts] == [1, 2]
    assert receipts[0].before_chain_digest == "0" * 64
    assert receipts[0].after_chain_digest == receipts[1].before_chain_digest
    assert all(item.readback_verified and item.state_changed for item in receipts)
    assert store.events() == (first, second)
    with pytest.raises(ValueError):
        store.events(after_sequence=1)
    assert store.integrity_check().valid is True


def test_duplicate_gap_unknown_cause_and_non_envelope_fail_without_partial_write(tmp_path: Path):
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    store.append(event(1))
    with pytest.raises(AppendOnlyViolation):
        store.append(event(1, event_id="duplicate-sequence"))
    with pytest.raises(AppendOnlyViolation):
        store.append(event(2, event_id="bad-cause", cause="missing"))
    with pytest.raises(AppendOnlyViolation):
        store.append_many((event(2, event_id="ok"), event(4, event_id="gap")))
    with pytest.raises(AppendOnlyViolation):
        store.append_many((object(),))  # type: ignore[arg-type]
    assert [item.event_id for item in store.events()] == ["event:1"]


def test_storage_policy_rejects_new_history_and_never_prunes_old_events(tmp_path: Path):
    store = SQLiteShadowStore(
        tmp_path / "shadow.sqlite3",
        policy=ShadowStoragePolicy(max_event_count=1, max_event_bytes=10_000),
    )
    store.initialize()
    store.append(event(1))
    with pytest.raises(StoragePolicyExceeded):
        store.append(event(2))
    assert store.events() == (event(1),)


def test_snapshot_due_binding_and_readback(tmp_path: Path):
    store = SQLiteShadowStore(
        tmp_path / "shadow.sqlite3",
        policy=ShadowStoragePolicy(snapshot_interval_events=2),
    )
    store.initialize()
    store.append_many((event(1), event(2, cause="event:1")))
    assert store.snapshot_due("shadow:test") is True
    receipt = store.write_snapshot(
        snapshot_id="snapshot:2",
        stream_id="shadow:test",
        through_sequence=2,
        state={"sum": 3},
        state_schema_version="test.state.v1",
    )
    assert receipt.readback_verified is True
    assert store.snapshot_due("shadow:test") is False
    selection = store.latest_valid_snapshot("shadow:test")
    assert selection.selected is not None
    assert selection.selected.state == {"sum": 3}
    with pytest.raises(SnapshotCorruption):
        store.write_snapshot(
            snapshot_id="",
            stream_id="shadow:test",
            through_sequence=2,
            state={"sum": 3},
            state_schema_version="test.state.v1",
        )
    with pytest.raises(SnapshotCorruption):
        store.write_snapshot(
            snapshot_id="snapshot:wrong",
            stream_id="shadow:test",
            through_sequence=1,
            state={"sum": 1},
            state_schema_version="test.state.v1",
        )
    with pytest.raises(SnapshotCorruption):
        store.write_snapshot(
            snapshot_id="snapshot:bad-state",
            stream_id="shadow:test",
            through_sequence=2,
            state={1: "not-a-string-key"},  # type: ignore[dict-item]
            state_schema_version="test.state.v1",
        )


def test_corrupt_newest_snapshot_falls_back_to_previous_valid_snapshot(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    store.append(event(1))
    store.write_snapshot(
        snapshot_id="snapshot:good",
        stream_id="shadow:test",
        through_sequence=1,
        state={"sum": 1},
        state_schema_version="test.state.v1",
    )
    store.write_snapshot(
        snapshot_id="snapshot:bad",
        stream_id="shadow:test",
        through_sequence=1,
        state={"sum": 1},
        state_schema_version="test.state.v1",
    )
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER snapshots_no_update")
        connection.execute(
            "UPDATE snapshots SET state_digest=? WHERE snapshot_id='snapshot:bad'",
            ("0" * 64,),
        )
        connection.execute(
            "CREATE TRIGGER snapshots_no_update BEFORE UPDATE ON snapshots "
            "BEGIN SELECT RAISE(ABORT,'append-only snapshots'); END"
        )
        connection.commit()
    finally:
        connection.close()
    selection = store.latest_valid_snapshot("shadow:test")
    assert selection.selected is not None
    assert selection.selected.snapshot_id == "snapshot:good"
    assert selection.rejected_snapshot_ids == ("snapshot:bad",)
    assert selection.fallback_used is True


@dataclass(frozen=True)
class State:
    total: int


def reduce_state(state: State, envelope: EventEnvelope) -> State:
    return State(state.total + int(envelope.payload["delta"]))


def test_restore_replays_twice_from_valid_snapshot_and_is_reproducible(tmp_path: Path):
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    store.append(event(1))
    store.write_snapshot(
        snapshot_id="snapshot:1",
        stream_id="shadow:test",
        through_sequence=1,
        state={"total": 1},
        state_schema_version="test.state.v1",
    )
    store.append(event(2, cause="event:1"))
    result = store.restore_verified(
        stream_id="shadow:test",
        initial_state=State(0),
        reducer=reduce_state,
        state_to_mapping=lambda state: {"total": state.total},
        state_from_mapping=lambda value: State(int(value["total"])),
    )
    assert result.state == State(3)
    assert result.replayed_event_count == 1
    assert result.state_digest == result.repeated_state_digest
    assert result.verified is True


def test_restore_uses_fresh_canonical_start_state_for_each_replay(tmp_path: Path):
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    store.append(event(1))
    initial = {"total": 0}

    def mutating_reducer(state: dict[str, int], envelope: EventEnvelope) -> dict[str, int]:
        state["total"] += int(envelope.payload["delta"])
        return state

    result = store.restore_verified(
        stream_id="shadow:test",
        initial_state=initial,
        reducer=mutating_reducer,
        state_to_mapping=lambda state: state,
        state_from_mapping=lambda value: {"total": int(value["total"])},
    )
    assert result.state == {"total": 1}
    assert result.state_digest == result.repeated_state_digest
    assert initial == {"total": 0}


def test_reopen_after_uncommitted_external_transaction_preserves_committed_history(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    store.append(event(1))
    connection = sqlite3.connect(path)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "INSERT INTO events(ordinal,event_id,stream_id,sequence,event_json,envelope_digest,event_bytes,previous_chain_digest,chain_digest) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (2, "uncommitted", "shadow:test", 2, "{}", "0" * 64, 2, "0" * 64, "0" * 64),
    )
    connection.close()
    reopened = SQLiteShadowStore(path)
    reopened.initialize()
    assert reopened.events() == (event(1),)
    assert reopened.integrity_check().valid is True


def test_event_corruption_is_visible_to_reads_and_integrity_report(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    store.append(event(1))
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET envelope_digest=? WHERE event_id='event:1'", ("0" * 64,)
        )
        connection.execute(EVENT_UPDATE_TRIGGER)
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(PersistedEventCorruption):
        store.events()
    report = store.integrity_check()
    assert report.valid is False
    assert any(item.startswith("event:") for item in report.errors)


def test_denormalized_event_index_corruption_is_not_accepted_as_valid(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    store.append(event(1))
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET stream_id='shadow:wrong', sequence=7 WHERE event_id='event:1'"
        )
        connection.execute(EVENT_UPDATE_TRIGGER)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PersistedEventCorruption):
        store.events()
    report = store.integrity_check()
    assert report.valid is False
    assert any("index columns disagree" in item for item in report.errors)


def test_filtered_stream_read_and_restore_do_not_hide_index_corruption(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    store.append(event(1))
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET stream_id='shadow:wrong' WHERE event_id='event:1'")
        connection.execute(EVENT_UPDATE_TRIGGER)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PersistedEventCorruption):
        store.events(stream_id="shadow:test")
    with pytest.raises(PersistedEventCorruption):
        store.restore_verified(
            stream_id="shadow:test",
            initial_state=State(0),
            reducer=reduce_state,
            state_to_mapping=lambda state: {"total": state.total},
            state_from_mapping=lambda value: State(int(value["total"])),
        )


def test_verified_backups_are_bounded_without_touching_event_history(tmp_path: Path):
    store = SQLiteShadowStore(
        tmp_path / "shadow.sqlite3",
        policy=ShadowStoragePolicy(max_backups=2),
    )
    store.initialize()
    store.append(event(1))
    backup_dir = tmp_path / "backups"
    first = store.create_backup(backup_dir, backup_ordinal=1)
    store.create_backup(backup_dir, backup_ordinal=2)
    third = store.create_backup(backup_dir, backup_ordinal=3)
    assert first.integrity_verified is True
    assert third.pruned_backup_names == ("shadow-backup-00000001.sqlite3",)
    assert sorted(path.name for path in backup_dir.iterdir()) == [
        "shadow-backup-00000002.sqlite3",
        "shadow-backup-00000003.sqlite3",
    ]
    backup_store = SQLiteShadowStore(third.backup_path)
    backup_store.initialize()
    assert backup_store.integrity_check().valid is True
    assert backup_store.events() == (event(1),)
    assert store.events() == (event(1),)
    with pytest.raises(BackupPolicyError):
        store.create_backup(backup_dir, backup_ordinal=3)
    with pytest.raises(BackupPolicyError):
        store.create_backup(backup_dir, backup_ordinal=1)
    assert not (backup_dir / "shadow-backup-00000001.sqlite3").exists()


def test_backup_rejects_logically_corrupt_source_and_issues_no_receipt(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    store = SQLiteShadowStore(path)
    store.initialize()
    store.append(event(1))
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET stream_id='shadow:wrong' WHERE event_id='event:1'")
        connection.execute(EVENT_UPDATE_TRIGGER)
        connection.commit()
    finally:
        connection.close()

    backup_dir = tmp_path / "backups"
    with pytest.raises(BackupPolicyError, match="source logical integrity"):
        store.create_backup(backup_dir, backup_ordinal=1)
    assert list(backup_dir.iterdir()) == []


def test_backup_candidate_must_pass_logical_verifier_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")
    store.initialize()
    store.append(event(1))
    original = SQLiteShadowStore._verify_database_path

    def reject_candidate(cls, path: Path, policy: ShadowStoragePolicy):
        report = original(path, policy)
        return type(report)(
            False,
            ("injected:logical-corruption",),
            report.event_count,
            report.snapshot_count,
            report.chain_head_digest,
            report.report_digest,
        )

    monkeypatch.setattr(SQLiteShadowStore, "_verify_database_path", classmethod(reject_candidate))
    backup_dir = tmp_path / "backups"
    with pytest.raises(BackupPolicyError, match="backup logical integrity"):
        store.create_backup(backup_dir, backup_ordinal=1)
    assert list(backup_dir.iterdir()) == []


def test_module_has_no_default_activation_legacy_bridge_thread_clock_random_or_pickle_surface():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
        "asyncio",
        "datetime",
        "language",
        "main",
        "pickle",
        "random",
        "secrets",
        "threading",
        "time",
        "uuid",
    }
    assert not calls & {"start", "sleep"}
    assert "dual-read" in source
    assert "cutover" in source
