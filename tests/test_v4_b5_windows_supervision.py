from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.authoritative_store import AuthorityUnprovable, AuthoritativeStore
from core.event_kernel import EventEnvelope
from scripts.operator import b5_runtime_environment as runtime
from scripts.operator.b5_windows_physical_gate import inject_tail_mismatch
from scripts.operator.b5_windows_supervisor import (
    SENTINEL_SCHEMA,
    SupervisorPaths,
    _canonical,
    _sentinel_payload,
    clear_sentinel,
    run_supervisor,
)

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "operator" / "b5_windows_supervisor.py"
PROBE = ROOT / "scripts" / "operator" / "b5_runtime_probe.py"
SERVICE = ROOT / "scripts" / "operator" / "b5_windows_service.ps1"
HOST_POLICY = ROOT / "scripts" / "operator" / "b5_windows_host_policy.ps1"
PREFLIGHT = ROOT / "scripts" / "operator" / "b5_windows_preflight.py"


def _event(sequence: int) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=f"b5:test:{sequence}",
        event_type="authority.b5_test",
        stream_id="authority:b5-test",
        sequence=sequence,
        producer="tests.b5",
        producer_version="1.0.0",
        correlation_id="corr:b5",
        causation_id=None if sequence == 1 else f"b5:test:{sequence - 1}",
        payload={"delta": sequence},
        causal_context={"phase": "b5-test"},
    )


def _store(path: Path, count: int = 1) -> None:
    store = AuthoritativeStore(path)
    try:
        store.open()
        for sequence in range(1, count + 1):
            store.append(_event(sequence))
    finally:
        store.close()


def _receipt(path: Path) -> dict[str, object]:
    python = Path(sys.executable).resolve()
    versions = runtime._installed_versions(python)
    packet: dict[str, object] = {
        "schema": runtime.SCHEMA,
        "authoritative_runtime": False,
        "t0_started": False,
        "repository": {
            "commit_sha": runtime._git("rev-parse", "HEAD"),
            "tree_sha": runtime._git("rev-parse", "HEAD^{tree}"),
            "clean_checkout": True,
        },
        "python": {
            "bootstrap_version": runtime._python_version(python),
            "interpreter": str(python),
            "installed_version": runtime._python_version(python),
        },
        "dependency_source": {
            "path": "requirements-lock.txt",
            "sha256": runtime._sha256(runtime.LOCK),
            "require_hashes": True,
            "install_argv": runtime.INSTALL_COMMAND,
            "requirements_runtime_used": False,
            "requirements_runtime_sha256": runtime._sha256(runtime.RUNTIME_RANGE),
        },
        "installed_distributions": versions,
        "numpy_version": versions["numpy"],
    }
    packet["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    path.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
    return packet


def _paths(tmp_path: Path) -> SupervisorPaths:
    store = tmp_path / "authority.sqlite3"
    _store(store)
    return SupervisorPaths(
        sentinel=tmp_path / "authority-stop.sentinel.json",
        audit_log=tmp_path / "supervisor-audit.jsonl",
        alert_log=tmp_path / "supervisor-alert.jsonl",
        state_file=tmp_path / "supervisor-state.json",
        authority_store=store,
    )


def _run(paths: SupervisorPaths, receipt: Path, code: str, max_restarts: int | None = 2) -> int:
    return run_supervisor(
        paths,
        receipt,
        [sys.executable, "-I", "-c", code],
        backoff_initial=0.001,
        backoff_max=0.002,
        max_restarts=max_restarts,
    )


def _audit(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_normal_exit_is_not_restarted(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    _receipt(receipt)
    assert _run(paths, receipt, "raise SystemExit(0)") == 0
    events = _audit(paths.audit_log)
    assert [event["event"] for event in events] == ["child_started", "child_exited"]
    assert json.loads(paths.state_file.read_text())["status"] == "STOPPED_NORMAL"
    assert not paths.sentinel.exists()


def test_exit_93_restarts_with_backoff_then_stops_normally(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    marker = tmp_path / "crashed"
    _receipt(receipt)
    code = (
        "import os,pathlib;"
        f"p=pathlib.Path({str(marker)!r});"
        "exists=p.exists();"
        "p.write_text('93') if not exists else None;"
        "os._exit(0 if exists else 93)"
    )
    assert _run(paths, receipt, code) == 0
    events = _audit(paths.audit_log)
    assert [event.get("child_exit_code") for event in events if event["event"] == "child_exited"] == [93, 0]
    assert any(event["event"] == "restart_scheduled" for event in events)
    assert not paths.sentinel.exists()


def test_exit_86_latches_sentinel_and_returns_service_success(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    _receipt(receipt)
    assert _run(paths, receipt, "raise SystemExit(86)") == 0
    sentinel, state = _sentinel_payload(paths.sentinel)
    assert state == "valid"
    assert sentinel is not None
    assert sentinel["schema"] == SENTINEL_SCHEMA
    assert sentinel["child_exit_code"] == 86
    assert sentinel["automatic_clear_permitted"] is False
    assert sentinel["store_sha256_before"] == sentinel["store_sha256_after"]
    assert json.loads(paths.state_file.read_text())["status"] == "STOPPED_LATCHED"
    assert any(event["event"] == "authority_stop_latched" for event in _audit(paths.audit_log))
    assert paths.alert_log.exists()


def test_service_restart_with_sentinel_never_launches_child(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    marker = tmp_path / "must-not-exist"
    _receipt(receipt)
    assert _run(paths, receipt, "raise SystemExit(86)") == 0
    assert _run(paths, receipt, f"from pathlib import Path;Path({str(marker)!r}).touch()") == 0
    assert not marker.exists()
    assert _audit(paths.audit_log)[-1]["event"] == "startup_blocked_by_sentinel"


def test_invalid_sentinel_also_blocks_launch(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    marker = tmp_path / "must-not-exist"
    _receipt(receipt)
    paths.sentinel.write_text("not-json", encoding="utf-8")
    assert _run(paths, receipt, f"from pathlib import Path;Path({str(marker)!r}).touch()") == 0
    assert not marker.exists()
    assert json.loads(paths.state_file.read_text())["sentinel_state"].startswith("sentinel_unreadable")


def test_operator_clear_requires_digest_and_archives_sentinel(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    _receipt(receipt)
    assert _run(paths, receipt, "raise SystemExit(86)") == 0
    sentinel, _ = _sentinel_payload(paths.sentinel)
    assert sentinel is not None
    with pytest.raises(Exception, match="digest changed"):
        clear_sentinel(
            paths.sentinel,
            paths.audit_log,
            operator="operator:test",
            reason="test recovery",
            expected_sha256="0" * 64,
        )
    archive = clear_sentinel(
        paths.sentinel,
        paths.audit_log,
        operator="operator:test",
        reason="clean test store restored",
        expected_sha256=str(sentinel["sentinel_sha256"]),
    )
    assert not paths.sentinel.exists()
    assert archive.is_file()
    clear = _audit(paths.audit_log)[-1]
    assert clear["event"] == "operator_sentinel_clear"
    assert clear["automatic"] is False
    assert clear["operator"] == "operator:test"


def test_operator_clear_rejects_invalid_sentinel(tmp_path: Path):
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="invalid sentinel"):
        clear_sentinel(
            sentinel,
            tmp_path / "audit.jsonl",
            operator="operator:test",
            reason="must not clear ambiguity",
            expected_sha256="0" * 64,
        )
    assert sentinel.exists()


def test_runtime_receipt_rejects_range_install_source(tmp_path: Path):
    receipt = tmp_path / "runtime.json"
    packet = _receipt(receipt)
    packet["dependency_source"]["path"] = "requirements-runtime.txt"  # type: ignore[index]
    unsigned = dict(packet)
    unsigned.pop("receipt_sha256")
    packet["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    receipt.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(runtime.RuntimeEnvironmentError, match="hash-pinned lock"):
        runtime.load_and_verify_receipt(receipt)


def test_supervisor_rejects_child_from_different_interpreter(tmp_path: Path):
    paths = _paths(tmp_path)
    receipt = tmp_path / "runtime.json"
    _receipt(receipt)
    with pytest.raises(Exception, match="pinned interpreter"):
        run_supervisor(
            paths,
            receipt,
            [str(tmp_path / "missing-python"), "-c", "pass"],
            backoff_initial=0.001,
            backoff_max=0.002,
            max_restarts=0,
        )


def test_tail_mismatch_is_injected_only_into_copy(tmp_path: Path):
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "injected.sqlite3"
    evidence = tmp_path / "injection.json"
    _store(source, 2)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    packet = inject_tail_mismatch(source, target, evidence)
    assert packet["source_unchanged"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert packet["target"]["integrity_check"] == "ok"
    store = AuthoritativeStore(target)
    try:
        with pytest.raises(AuthorityUnprovable):
            store.open()
    finally:
        store.close()


def test_runtime_probe_returns_86_for_tail_mismatch(tmp_path: Path):
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "injected.sqlite3"
    _store(source)
    inject_tail_mismatch(source, target, tmp_path / "injection.json")
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"schema": "eve.b5-runtime-probe-control.v1", "action": "verify_once"}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--database",
            str(target),
            "--control",
            str(control),
            "--raw-log",
            str(tmp_path / "child.jsonl"),
            "--ready",
            str(tmp_path / "ready.json"),
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 86


def test_service_definition_starts_supervisor_not_eve_directly():
    source = SERVICE.read_text(encoding="utf-8")
    assert "b5_windows_supervisor.py" in source
    assert "b5_runtime_probe.py" in source
    assert "<startmode>Automatic</startmode>" in source
    assert "<onfailure action=\"restart\"" in source
    assert "requirements-runtime.txt" not in source


def test_host_policy_preserves_defender_and_disables_fast_startup():
    source = HOST_POLICY.read_text(encoding="utf-8")
    assert "HiberbootEnabled" in source
    assert "NoAutoRebootWithLoggedOnUsers" in source
    assert "Set-MpPreference" not in source
    assert "Add-MpPreference" not in source
    assert "no B5 exclusion is added" in source


def test_preflight_records_numpy_and_never_treats_unresolved_as_pass():
    source = PREFLIGHT.read_text(encoding="utf-8")
    assert '"numpy": receipt["numpy_version"]' in source
    assert '"UNRESOLVED"' in source
    assert 'all(value in {"PASS", "NOT_APPLICABLE"}' in source
    assert "requirements_runtime_used" in source


def test_b5_does_not_modify_or_promote_shadow_store():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            SUPERVISOR,
            PROBE,
            SERVICE,
            HOST_POLICY,
            PREFLIGHT,
            ROOT / "scripts" / "operator" / "b5_windows_physical_gate.py",
        )
    )
    assert "core/sqlite_shadow_store.py" not in sources
    assert "authority_active_for_runtime\": False" in sources
    assert "#246" not in sources
