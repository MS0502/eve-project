from __future__ import annotations

import ast
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from core.authoritative_store import (
    AUTHORITY_FAILURE_EXIT_CODE,
    AUTHORITY_LABEL,
    AUTHORITY_PATH_ENV,
    EVENT_SERIALIZATION_VERSION,
    FAULT_AFTER_ACCEPTED_TAIL_UPDATE,
    FAULT_AFTER_EVENT_ROW_WRITE_BEFORE_COMMIT,
    FAULT_AFTER_EVENT_TRANSACTION_COMMIT,
    FAULT_BEFORE_ACCEPTED_TAIL_UPDATE,
    FAULT_BEFORE_EVENT_APPEND,
    FAULT_DURING_ACCEPTED_TAIL_UPDATE,
    GENESIS_HASH,
    AuthorityAmbiguity,
    AuthorityBusy,
    AuthorityNotOpen,
    AuthorityUnprovable,
    AuthoritativeStore,
    InjectedAuthorityFault,
    authority_path_from_environment,
)
from core.event_kernel import EventEnvelope, InMemoryEventKernel

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core" / "authoritative_store.py"
VERIFY = ROOT / "scripts" / "audit" / "b2_authority_verify.py"
FAULT_WORKER = ROOT / "scripts" / "audit" / "b2_authority_fault_worker.py"
PHYSICAL_GATE = ROOT / "scripts" / "operator" / "b2_authority_physical_gate.py"


def event(sequence: int, *, stream: str = "authority:test") -> EventEnvelope:
    return EventEnvelope.create(
        event_id=f"accepted:{stream}:{sequence}",
        event_type="authority.test",
        stream_id=stream,
        sequence=sequence,
        producer="tests.b2",
        producer_version="1.0.0",
        correlation_id=f"corr:{stream}",
        causation_id=None if sequence == 1 else f"accepted:{stream}:{sequence - 1}",
        payload={"delta": sequence},
        causal_context={"phase": "b2-test"},
    )


def _trigger_sql(connection: sqlite3.Connection, name: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?", (name,)
    ).fetchone()
    assert row is not None
    return str(row[0])


def _mutate_with_trigger_disabled(
    path: Path, *, trigger: str, statement: str, parameters: tuple[object, ...] = ()
) -> None:
    connection = sqlite3.connect(path)
    try:
        trigger_sql = _trigger_sql(connection, trigger)
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(statement, parameters)
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()


def test_construction_is_io_free_and_authority_path_is_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authority = tmp_path / "authority.sqlite3"
    shadow = tmp_path / "shadow.sqlite3"
    store = AuthoritativeStore(authority, shadow_path=shadow)
    assert not authority.exists()
    assert not Path(f"{authority}.writer.lock").exists()
    with pytest.raises(AuthorityNotOpen):
        store.verify()
    assert authority_path_from_environment(
        {AUTHORITY_PATH_ENV: str(authority), "EVE_SQLITE_SHADOW_PATH": str(shadow)}
    ) == authority
    with pytest.raises(Exception, match="must differ"):
        authority_path_from_environment(
            {AUTHORITY_PATH_ENV: str(authority), "EVE_SQLITE_SHADOW_PATH": str(authority)}
        )
    with pytest.raises(ValueError, match="must differ"):
        AuthoritativeStore(authority, shadow_path=authority)
    monkeypatch.setenv("EVE_SQLITE_SHADOW_PATH", str(authority))
    with pytest.raises(ValueError, match="must differ"):
        AuthoritativeStore(authority)


def test_schema_wal_full_sync_and_append_only_authority_are_verified(tmp_path: Path):
    path = tmp_path / "authority.sqlite3"
    store = AuthoritativeStore(path)
    startup = store.open()
    try:
        assert startup.authority == AUTHORITY_LABEL
        assert startup.journal_mode in {"wal", "delete"}
        assert startup.wal_enabled != startup.rollback_fallback
        assert startup.synchronous == "FULL"
        assert startup.accepted_event_count == 0
        assert startup.event_chain_head == GENESIS_HASH
        assert startup.accepted_tail_hash == GENESIS_HASH
        connection = sqlite3.connect(path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            assert tables == {
                "accepted_tail",
                "authority_events",
                "authority_meta",
                "authority_migrations",
                "event_candidate",
            }
            assert "events_shadow" not in tables
            assert connection.execute(
                "SELECT value FROM authority_meta WHERE key='event_serialization_version'"
            ).fetchone()[0] == EVENT_SERIALIZATION_VERSION
        finally:
            connection.close()
    finally:
        store.close()


def test_append_hash_chain_tail_and_event_kernel_replay_are_deterministic(tmp_path: Path):
    path = tmp_path / "authority.sqlite3"
    with AuthoritativeStore(path) as store:
        first = store.append(event(1))
        second = store.append(event(2))
        assert first.ordinal == 1 and second.ordinal == 2
        assert first.prev_hash == GENESIS_HASH
        assert second.prev_hash == first.event_hash
        assert first.event_hash != first.content_hash
        assert first.tail_hash != second.tail_hash
        assert first.candidate_commit_durable and first.accepted_commit_durable
        assert first.readback_verified
        report = store.verify()
        assert report.accepted_event_count == 2
        assert report.event_chain_head == second.event_hash
        assert report.accepted_tail_hash == second.tail_hash

        reducer = lambda state, envelope: state + int(envelope.payload["delta"])
        assert store.replay(0, reducer) == 3
        kernel: InMemoryEventKernel[int] = InMemoryEventKernel()
        for item in store.events():
            kernel.append(item)
        assert kernel.replay(0, reducer) == store.replay(0, reducer)


@pytest.mark.parametrize(
    ("fault_point", "recovered", "accepted"),
    [
        (FAULT_BEFORE_EVENT_APPEND, 0, 0),
        (FAULT_AFTER_EVENT_ROW_WRITE_BEFORE_COMMIT, 0, 0),
        (FAULT_AFTER_EVENT_TRANSACTION_COMMIT, 1, 0),
        (FAULT_BEFORE_ACCEPTED_TAIL_UPDATE, 1, 0),
        (FAULT_DURING_ACCEPTED_TAIL_UPDATE, 1, 0),
        (FAULT_AFTER_ACCEPTED_TAIL_UPDATE, 0, 1),
    ],
)
def test_deterministic_fault_matrix_preserves_only_accepted_history(
    tmp_path: Path, fault_point: str, recovered: int, accepted: int
):
    path = tmp_path / f"{fault_point}.sqlite3"

    def inject(point: str) -> None:
        if point == fault_point:
            raise InjectedAuthorityFault(point)

    store = AuthoritativeStore(path, fault_injector=inject)
    store.open()
    try:
        with pytest.raises(InjectedAuthorityFault, match=fault_point):
            store.append(event(1))
    finally:
        store.close()

    reopened = AuthoritativeStore(path)
    startup = reopened.open()
    try:
        assert startup.recovered_candidate_count == recovered
        assert startup.accepted_event_count == accepted
        assert reopened.verify().candidate_count == 0
        if accepted == 0:
            reopened.append(event(1))
        assert len(reopened.events()) == 1
    finally:
        reopened.close()


def test_process_interruption_restart_recovers_residue_and_preserves_committed_tail(
    tmp_path: Path,
):
    path = tmp_path / "interruption.sqlite3"
    with AuthoritativeStore(path):
        pass
    residue = subprocess.run(
        [
            sys.executable,
            str(FAULT_WORKER),
            "--database",
            str(path),
            "--sequence",
            "1",
            "--fault-point",
            FAULT_AFTER_EVENT_TRANSACTION_COMMIT,
        ],
        cwd=ROOT,
        check=False,
    )
    assert residue.returncode == 93
    recovered = AuthoritativeStore(path)
    assert recovered.open().recovered_candidate_count == 1
    assert recovered.events() == ()
    recovered.close()

    accepted = subprocess.run(
        [
            sys.executable,
            str(FAULT_WORKER),
            "--database",
            str(path),
            "--sequence",
            "1",
            "--fault-point",
            FAULT_AFTER_ACCEPTED_TAIL_UPDATE,
        ],
        cwd=ROOT,
        check=False,
    )
    assert accepted.returncode == 93
    final = AuthoritativeStore(path)
    assert final.open().accepted_event_count == 1
    assert [item.event_id for item in final.events()] == ["b2:worker:1"]
    final.close()


@pytest.mark.parametrize(
    ("table", "trigger", "statement"),
    [
        (
            "authority_events",
            "authority_events_no_update",
            "UPDATE authority_events SET event_bytes=x'00' WHERE ordinal=1",
        ),
        (
            "authority_events",
            "authority_events_no_update",
            "UPDATE authority_events SET event_hash='bad' WHERE ordinal=1",
        ),
        (
            "accepted_tail",
            "accepted_tail_no_update",
            "UPDATE accepted_tail SET tail_hash='bad' WHERE revision=1",
        ),
    ],
)
def test_accepted_corruption_is_not_repaired_and_cli_exits_86(
    tmp_path: Path, table: str, trigger: str, statement: str
):
    path = tmp_path / f"corrupt-{table}-{trigger}.sqlite3"
    with AuthoritativeStore(path) as store:
        store.append(event(1))
    _mutate_with_trigger_disabled(path, trigger=trigger, statement=statement)

    result = subprocess.run(
        [sys.executable, str(VERIFY), "--database", str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == AUTHORITY_FAILURE_EXIT_CODE
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
        if table == "authority_events" and "event_bytes" in statement:
            assert connection.execute(
                "SELECT event_bytes FROM authority_events WHERE ordinal=1"
            ).fetchone()[0] == b"\x00"
    finally:
        connection.close()


def test_accepted_historical_corruption_fails_closed_without_truncation(tmp_path: Path):
    path = tmp_path / "historical.sqlite3"
    with AuthoritativeStore(path) as store:
        store.append(event(1))
        store.append(event(2))
    _mutate_with_trigger_disabled(
        path,
        trigger="authority_events_no_update",
        statement="UPDATE authority_events SET content_hash=? WHERE ordinal=1",
        parameters=("0" * 64,),
    )
    with pytest.raises(AuthorityUnprovable):
        AuthoritativeStore(path).open()
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM authority_events").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM accepted_tail").fetchone()[0] == 2
    finally:
        connection.close()


def test_only_proven_unaccepted_residue_is_removed(tmp_path: Path):
    path = tmp_path / "ambiguous.sqlite3"

    def inject(point: str) -> None:
        if point == FAULT_AFTER_EVENT_TRANSACTION_COMMIT:
            raise InjectedAuthorityFault(point)

    store = AuthoritativeStore(path, fault_injector=inject)
    store.open()
    with pytest.raises(InjectedAuthorityFault):
        store.append(event(1))
    store.close()
    _mutate_with_trigger_disabled(
        path,
        trigger="event_candidate_no_update",
        statement="UPDATE event_candidate SET expected_ordinal=9 WHERE slot=1",
    )
    with pytest.raises(AuthorityAmbiguity):
        AuthoritativeStore(path).open()
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM event_candidate").fetchone()[0] == 1
    finally:
        connection.close()


def test_controlled_rollback_journal_fallback_is_explicit(tmp_path: Path):
    path = tmp_path / "fallback.sqlite3"

    def force_delete(connection: sqlite3.Connection) -> str:
        return str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0])

    store = AuthoritativeStore(path, wal_probe=force_delete)
    startup = store.open()
    try:
        assert startup.journal_mode == "delete"
        assert startup.rollback_fallback is True
        assert startup.wal_enabled is False
        store.append(event(1))
        assert store.verify().journal_mode == "delete"
    finally:
        store.close()

    disabled = AuthoritativeStore(
        tmp_path / "no-fallback.sqlite3",
        wal_probe=force_delete,
        allow_rollback_fallback=False,
    )
    with pytest.raises(AuthorityUnprovable, match="fallback is disabled"):
        disabled.open()


def test_single_authoritative_writer_lock_fails_closed(tmp_path: Path):
    path = tmp_path / "writer.sqlite3"
    first = AuthoritativeStore(path)
    second = AuthoritativeStore(path)
    first.open()
    try:
        with pytest.raises(AuthorityBusy):
            second.open()
    finally:
        first.close()
    second.open()
    second.close()


def test_module_does_not_upgrade_or_import_the_shadow_store():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "core.sqlite_shadow_store" not in imports
    assert "events_shadow" not in source
    assert "EVE_SQLITE_SHADOW_PATH" in source
    assert "EVE_AUTHORITY_PATH" in source


@pytest.mark.skipif(sys.platform != "win32", reason="Windows RSS API contract")
def test_physical_gate_reads_process_rss_on_windows():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.operator.b2_authority_physical_gate import _rss_bytes; "
                "value = _rss_bytes(); assert isinstance(value, int) and value > 0"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
