from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.m2_e_window_driver import WindowConfig, WindowState, freeze_shadow
from scripts.habitat import m2_e_window_runtime_guarded as runtime

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/habitat/supervisor.sh"
STABLE_RUNTIME = ROOT / "scripts/habitat/m2_e_window_runtime.py"


def _healthy_report(*, event_count: int = 0):
    return SimpleNamespace(
        valid=True,
        errors=(),
        event_count=event_count,
        snapshot_count=0,
        chain_head_digest="a" * 64,
        report_digest="b" * 64,
    )


def _corrupt_report(*, event_count: int = 0):
    return SimpleNamespace(
        valid=False,
        errors=("synthetic-corruption",),
        event_count=event_count,
        snapshot_count=0,
        chain_head_digest="a" * 64,
        report_digest="c" * 64,
    )


class _IntegrityStore:
    def __init__(self, report):
        self.report = report

    def integrity_check(self):
        return self.report


def _raw_records(root: Path) -> list[dict]:
    raw = root / "raw_evidence.jsonl"
    if not raw.exists():
        return []
    return [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]


def test_restore_oserror_records_detail_backs_off_three_times_and_classifies_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = runtime._ensure_private_root(tmp_path / "private")
    store = _IntegrityStore(_healthy_report())
    sleeps: list[int] = []

    def fail_restore(_store):
        raise OSError("injected restore I/O failure")

    monkeypatch.setattr(runtime, "_restore", fail_restore)
    monkeypatch.setattr(runtime.time, "sleep", sleeps.append)

    restored, reason, trigger = runtime._restore_with_io_backoff(
        paths,
        store,
        context="test_restore",
    )

    assert restored is None
    assert reason == runtime.IO_FAILURE == "io_failure"
    assert trigger is not None and len(trigger) == 64
    assert sleeps == [1, 2, 4]
    exceptions = [item for item in _raw_records(paths["root"]) if item["type"] == "habitat_exception"]
    assert len(exceptions) == 4
    assert [item["attempt"] for item in exceptions] == [1, 2, 3, 4]
    assert all(item["exception_type"] == "OSError" for item in exceptions)
    assert all(item["exception_message"] == "injected restore I/O failure" for item in exceptions)
    assert all(len(item["traceback_digest"]) == 64 for item in exceptions)
    assert all(len(item["exception_digest"]) == 64 for item in exceptions)


def test_unrecoverable_corruption_requires_failed_integrity_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = runtime._ensure_private_root(tmp_path / "private")
    store = _IntegrityStore(_corrupt_report())

    monkeypatch.setattr(runtime, "_restore", lambda _store: (_ for _ in ()).throw(OSError("io")))
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)

    restored, reason, trigger = runtime._restore_with_io_backoff(
        paths,
        store,
        context="test_corruption",
    )

    assert restored is None
    assert reason == "unrecoverable_corruption"
    assert trigger is not None


def test_reviewed_resume_refuses_corrupt_store_without_changing_frozen_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime._enable_io_failure_reason()
    root = tmp_path / "private"
    paths = runtime._ensure_private_root(root)
    state = freeze_shadow(
        WindowState.create(window_id="m2-e:test-resume:v1", local_date="2026-07-25"),
        "recovery_digest_mismatch",
    )
    runtime._save_state(paths, state)
    monkeypatch.setattr(runtime, "_store", lambda _paths, _config: _IntegrityStore(_corrupt_report()))

    code = runtime.resume_reviewed(private_root=root, window_id=state.window_id)

    assert code == 22
    reloaded = runtime._load_state(paths, window_id=state.window_id)
    assert reloaded == state
    records = _raw_records(root)
    assert records[-1]["type"] == "freeze_reviewed_resume_denied"
    assert records[-1]["reason"] == "unrecoverable_corruption"
    assert records[-2]["type"] == "freeze_review_snapshot"
    assert records[-1]["freeze_record_digest"] == records[-2]["freeze_record_digest"]


def test_reviewed_resume_reconciles_one_exact_pending_event_and_preserves_freeze_evidence(
    tmp_path: Path,
):
    runtime._enable_io_failure_reason()
    root = tmp_path / "private"
    paths = runtime._ensure_private_root(root)
    config = WindowConfig()
    state = freeze_shadow(
        WindowState.create(window_id="m2-e:test-pending:v1", local_date="2026-07-25"),
        "recovery_digest_mismatch",
    )
    runtime._save_state(paths, state)
    store = runtime._store(paths, config)
    store.append(runtime._event(1))

    code = runtime.resume_reviewed(private_root=root, window_id=state.window_id)

    assert code == 0
    resumed = runtime._load_state(paths, window_id=state.window_id)
    assert resumed.frozen is False
    assert resumed.notification_sent is False
    assert resumed.event_count == 1
    assert resumed.hour_counts == ((0, 1),)
    assert resumed.expected_recovery_digest == resumed.last_recovery_digest
    records = _raw_records(root)
    review = next(item for item in records if item["type"] == "freeze_review_snapshot")
    resume = next(item for item in records if item["type"] == "freeze_reviewed_resume")
    assert resume["freeze_record_digest"] == review["freeze_record_digest"]
    assert resume["reconciled_pending_commit"] is True
    assert resume["before_state_digest"] == state.digest
    assert len(resume["resume_record_digest"]) == 64


def test_reviewed_resume_same_count_requires_recomputed_restore_digest(tmp_path: Path):
    runtime._enable_io_failure_reason()
    root = tmp_path / "private"
    paths = runtime._ensure_private_root(root)
    config = WindowConfig()
    store = runtime._store(paths, config)
    restored = runtime._restore(store)
    state = freeze_shadow(
        WindowState.create(window_id="m2-e:test-same:v1", local_date="2026-07-25"),
        "recovery_digest_mismatch",
    )
    state = runtime.replace(state, expected_recovery_digest=restored.state_digest)
    runtime._save_state(paths, state)

    assert runtime.resume_reviewed(private_root=root, window_id=state.window_id) == 0
    resumed = runtime._load_state(paths, window_id=state.window_id)
    assert resumed.frozen is False
    assert resumed.event_count == 0
    assert resumed.last_recovery_digest == restored.state_digest


def test_supervisor_captures_runtime_streams_and_stable_entrypoint_delegates():
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    stable = STABLE_RUNTIME.read_text(encoding="utf-8")

    assert 'LOG="$PRIVATE_ROOT/supervisor.log"' in supervisor
    assert '2>>"$LOG"' in supervisor
    assert '>>"$LOG" 2>&1' in supervisor
    assert "supervisor_start" in supervisor
    assert "m2_e_window_runtime_guarded" in stable
    assert "raise SystemExit(main())" in stable
    assert "kill -9" not in supervisor
    assert "pkill" not in supervisor
