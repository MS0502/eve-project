from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from core.authoritative_store import (
    AUTHORITY_FAILURE_EXIT_CODE,
    FAULT_AFTER_ACCEPTED_TAIL_UPDATE,
    FAULT_AFTER_EVENT_TRANSACTION_COMMIT,
    AuthorityUnprovable,
    AuthoritativeStore,
    classify_authority_failure,
    retry_authority_operation,
    status,
)
from core.event_kernel import EventEnvelope
from scripts.operator import b5_runtime_probe
from scripts.operator.b5_windows_preflight import (
    _defender_exclusion_state,
    _power_indexes,
)


def _sqlite_error(code: int, name: str) -> sqlite3.OperationalError:
    error = sqlite3.OperationalError(name)
    error.sqlite_errorcode = code  # type: ignore[attr-defined]
    error.sqlite_errorname = name  # type: ignore[attr-defined]
    return error


def _ioerr(subcode: int) -> int:
    return sqlite3.SQLITE_IOERR | (subcode << 8)


def _windows_error(code: int) -> OSError:
    error = OSError(code, "injected Windows file error")
    error.winerror = code  # type: ignore[attr-defined]
    return error


def _event(sequence: int) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=f"b5:transient:{sequence}",
        event_type="authority.b5_transient_test",
        stream_id="authority:b5-transient",
        sequence=sequence,
        producer="tests.b5.transient",
        producer_version="1.0.0",
        correlation_id="corr:b5-transient",
        causation_id=None if sequence == 1 else f"b5:transient:{sequence - 1}",
        payload={"sequence": sequence},
        causal_context={"phase": "b5-transient-test"},
    )


@pytest.mark.parametrize(
    ("error", "source", "extended_code"),
    [
        (_sqlite_error(sqlite3.SQLITE_BUSY, "SQLITE_BUSY"), "sqlite", sqlite3.SQLITE_BUSY),
        (
            _sqlite_error(sqlite3.SQLITE_LOCKED, "SQLITE_LOCKED"),
            "sqlite",
            sqlite3.SQLITE_LOCKED,
        ),
        (_sqlite_error(_ioerr(1), "SQLITE_IOERR_READ"), "sqlite", _ioerr(1)),
        (_sqlite_error(_ioerr(3), "SQLITE_IOERR_WRITE"), "sqlite", _ioerr(3)),
        (_sqlite_error(_ioerr(4), "SQLITE_IOERR_FSYNC"), "sqlite", _ioerr(4)),
        (_sqlite_error(_ioerr(15), "SQLITE_IOERR_LOCK"), "sqlite", _ioerr(15)),
        (_windows_error(32), "windows", None),
        (_windows_error(33), "windows", None),
    ],
)
def test_approved_extended_io_and_windows_lock_codes_are_transient(
    error: BaseException, source: str, extended_code: int | None
) -> None:
    classification = classify_authority_failure(error)
    assert classification.classification == "TRANSIENT"
    assert classification.source == source
    assert classification.sqlite_extended_errcode == extended_code


@pytest.mark.parametrize(
    ("code", "name"),
    [
        (sqlite3.SQLITE_CORRUPT, "SQLITE_CORRUPT"),
        (sqlite3.SQLITE_NOTADB, "SQLITE_NOTADB"),
        (_ioerr(2), "SQLITE_IOERR_SHORT_READ"),
        (_ioerr(32), "SQLITE_IOERR_DATA"),
        (_ioerr(33), "SQLITE_IOERR_CORRUPTFS"),
    ],
)
def test_corruption_extended_codes_are_immediate_integrity_failures(
    code: int, name: str
) -> None:
    classification = classify_authority_failure(_sqlite_error(code, name))
    assert classification.classification == "INTEGRITY"
    assert classification.sqlite_extended_errcode == code


def test_primary_ioerr_without_extended_subtype_is_not_retried() -> None:
    classification = classify_authority_failure(
        _sqlite_error(sqlite3.SQLITE_IOERR, "SQLITE_IOERR")
    )
    assert classification.classification == "UNPROVABLE"
    assert "extended subtype is unavailable" in classification.reason


def test_transient_retry_succeeds_and_logs_the_extended_code() -> None:
    calls: list[int] = []
    sleeps: list[float] = []
    records: list[dict[str, Any]] = []

    def operation(attempt: int) -> str:
        calls.append(attempt)
        if attempt == 1:
            raise _sqlite_error(_ioerr(1), "SQLITE_IOERR_READ")
        return "kept-running"

    result = retry_authority_operation(
        operation,
        operation_name="test_transient_success",
        max_attempts=3,
        backoff_initial=0.01,
        backoff_max=0.02,
        observer=lambda record: records.append(dict(record)),
        sleeper=sleeps.append,
    )
    assert result == "kept-running"
    assert calls == [1, 2]
    assert sleeps == [0.01]
    assert records[0]["classification"] == "TRANSIENT"
    assert records[0]["sqlite_extended_errcode"] == _ioerr(1)
    assert records[0]["retry_scheduled"] is True


def test_transient_retry_exhaustion_maps_to_exit_86() -> None:
    calls: list[int] = []
    sleeps: list[float] = []
    records: list[dict[str, Any]] = []

    def operation(attempt: int) -> None:
        calls.append(attempt)
        raise _windows_error(32)

    with pytest.raises(AuthorityUnprovable) as raised:
        retry_authority_operation(
            operation,
            operation_name="test_transient_exhaustion",
            max_attempts=3,
            backoff_initial=0.01,
            backoff_max=0.02,
            observer=lambda record: records.append(dict(record)),
            sleeper=sleeps.append,
        )
    assert status(raised.value) == AUTHORITY_FAILURE_EXIT_CODE
    assert calls == [1, 2, 3]
    assert sleeps == [0.01, 0.02]
    assert records[-1]["retry_exhausted"] is True
    assert records[-1]["windows_error_code"] == 32


@pytest.mark.parametrize(
    "fault_point",
    [FAULT_AFTER_EVENT_TRANSACTION_COMMIT, FAULT_AFTER_ACCEPTED_TAIL_UPDATE],
)
def test_append_retries_proven_commit_without_duplicate_event(
    tmp_path: Path, fault_point: str
) -> None:
    path = tmp_path / "authority.sqlite3"
    injected = False
    records: list[dict[str, Any]] = []

    def inject(point: str) -> None:
        nonlocal injected
        if point == fault_point and not injected:
            injected = True
            raise _sqlite_error(_ioerr(4), "SQLITE_IOERR_FSYNC")

    store = AuthoritativeStore(
        path,
        fault_injector=inject,
        retry_observer=lambda record: records.append(dict(record)),
        retry_sleeper=lambda _delay: None,
    )
    try:
        store.open()
        receipt = store.append(_event(1))
        store.append(_event(2))
        assert receipt.accepted_commit_durable is True
        assert [event.event_id for event in store.events()] == [
            "b5:transient:1",
            "b5:transient:2",
        ]
        assert store.verify().accepted_event_count == 2
    finally:
        store.close()
    assert injected is True
    assert any(
        record["operation"] == "authoritative_append"
        and record["sqlite_extended_errcode"] == _ioerr(4)
        and record["retry_scheduled"] is True
        for record in records
    )


def test_runtime_probe_survives_transient_store_hash_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "authority.sqlite3"
    with AuthoritativeStore(database) as store:
        store.append(_event(1))
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps(
            {
                "schema": b5_runtime_probe.CONTROL_SCHEMA,
                "action": "verify_once",
            }
        ),
        encoding="utf-8",
    )
    raw_log = tmp_path / "probe.jsonl"
    ready = tmp_path / "ready.json"
    original_sha256 = b5_runtime_probe._sha256
    original_retry = b5_runtime_probe.retry_authority_operation
    calls = 0

    def transient_once(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _windows_error(32)
        return original_sha256(path)

    def retry_without_wait(operation: Any, **kwargs: Any) -> Any:
        return original_retry(operation, sleeper=lambda _delay: None, **kwargs)

    monkeypatch.setattr(b5_runtime_probe, "_sha256", transient_once)
    monkeypatch.setattr(b5_runtime_probe, "retry_authority_operation", retry_without_wait)
    result = b5_runtime_probe.main(
        [
            "--database",
            str(database),
            "--control",
            str(control),
            "--raw-log",
            str(raw_log),
            "--ready",
            str(ready),
        ]
    )
    assert result == 0
    assert ready.is_file()
    evidence = [json.loads(line) for line in raw_log.read_text(encoding="utf-8").splitlines()]
    classified = [item for item in evidence if item["event"] == "authority_failure_classified"]
    assert len(classified) == 1
    assert classified[0]["windows_error_code"] == 32
    assert classified[0]["retry_scheduled"] is True
    assert evidence[-1]["event"] == "probe_ready"


def test_real_accepted_tail_corruption_exits_86_without_retry(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    with AuthoritativeStore(path) as store:
        store.append(_event(1))

    connection = sqlite3.connect(path)
    try:
        trigger_sql = str(
            connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name='accepted_tail_no_update'"
            ).fetchone()[0]
        )
        connection.execute("DROP TRIGGER accepted_tail_no_update")
        connection.execute("UPDATE accepted_tail SET tail_hash=?", ("f" * 64,))
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    sleeps: list[float] = []
    records: list[dict[str, Any]] = []
    corrupted = AuthoritativeStore(
        path,
        retry_observer=lambda record: records.append(dict(record)),
        retry_sleeper=sleeps.append,
    )
    with pytest.raises(AuthorityUnprovable) as raised:
        corrupted.open()
    assert status(raised.value) == AUTHORITY_FAILURE_EXIT_CODE
    assert sleeps == []
    assert len(records) == 1
    assert records[0]["classification"] == "INTEGRITY"
    assert records[0]["attempt"] == 1
    assert records[0]["retry_scheduled"] is False


def test_defender_requires_exact_store_directory_not_broader_parent(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "eve" / "authority"
    exact = _defender_exclusion_state(authority, [str(authority)])
    broad = _defender_exclusion_state(authority, [str(tmp_path)])
    missing = _defender_exclusion_state(authority, [])
    assert exact["exact_match"] is True
    assert exact["broader_parent_matches"] == []
    assert broad["exact_match"] is False
    assert broad["broader_parent_matches"] == [str(tmp_path)]
    assert missing["exact_match"] is False


def test_power_index_parser_keeps_ac_and_dc_fail_closed_across_locales() -> None:
    english = _power_indexes(
        "Current AC Power Setting Index: 0x00000000\n"
        "Current DC Power Setting Index: 0x0000001e\n"
    )
    localized_without_recognized_labels = _power_indexes(
        "localized-one: 0x00000000\nlocalized-two: 0x0000001e\n"
    )
    ambiguous = _power_indexes(
        "localized-one: 0x00000000\n"
        "localized-two: 0x0000001e\n"
        "localized-three: 0x00000000\n"
    )
    assert english == {"ac": [0], "dc": [30]}
    assert localized_without_recognized_labels == {"ac": [0], "dc": [30]}
    assert ambiguous == {"ac": [], "dc": []}


def test_preflight_rechecks_all_required_windows_continuity_inputs() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "operator"
        / "b5_windows_preflight.py"
    ).read_text(encoding="utf-8")
    assert 'fast.get("HiberbootEnabled") == 0' in source
    assert '"STANDBYIDLE"' in source
    assert '"HIBERNATEIDLE"' in source
    assert '"DISKIDLE"' in source
    assert '"LIDACTION"' in source
    assert 'exclusion_state["exact_match"] is True' in source
    assert 'not exclusion_state["broader_parent_matches"]' in source
