from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from core.authoritative_store import AuthorityUnprovable, AuthoritativeStore
from core.event_kernel import EventEnvelope
from scripts.operator import b5_runtime_environment as runtime
from scripts.operator.b5_windows_physical_gate import (
    RESTART_CONTINUITY_SCHEMA,
    inject_tail_mismatch,
    load_restart_continuity_proof,
    prove_restart_continuity,
)
from scripts.operator.b5_windows_preflight import classify_windows_update
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


def _reboot_capture(boot: str, *, status: str = "Running") -> dict[str, object]:
    store_sha256 = "a" * 64
    verification = {
        "accepted_count": 3,
        "event_head": "b" * 64,
        "tail_head": "b" * 64,
    }
    ready: dict[str, object] = {
        "schema": "eve.b5-runtime-probe-ready.v1",
        "database": {
            "sha256": store_sha256,
            "verification": verification,
        },
    }
    ready["receipt_sha256"] = hashlib.sha256(_canonical(ready)).hexdigest()
    return {
        "schema": "eve.b5-windows-physical-capture.v1",
        "label": "reboot-capture",
        "receipt_sha256": "c" * 64,
        "boot": {"last_boot_utc": boot},
        "service": {
            "parsed": {
                "Status": status,
                "StartType": "Automatic",
                "StartMode": "Auto",
            }
        },
        "store": {
            "valid": True,
            "sha256": store_sha256,
            "verification": verification,
        },
        "ready": {"content": ready},
    }


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


def test_restart_continuity_proof_requires_real_reboot_and_matching_store():
    before = _reboot_capture("2026-08-15T01:00:00Z")
    after = _reboot_capture("2026-08-15T02:00:00Z")
    proof = prove_restart_continuity(before, after)
    assert proof["passed"] is True
    assert proof["verdict"] == "ACCEPTED"
    assert proof["reason"] == "restart continuity proven"
    assert proof["before"]["store_verification"] == proof["after"]["store_verification"]

    with pytest.raises(RuntimeError, match="identity did not change"):
        prove_restart_continuity(before, before)

    stopped = _reboot_capture("2026-08-15T02:00:00Z", status="Stopped")
    with pytest.raises(RuntimeError, match="not running after reboot"):
        prove_restart_continuity(before, stopped)

    changed = deepcopy(after)
    changed["store"]["sha256"] = "d" * 64  # type: ignore[index]
    with pytest.raises(RuntimeError, match="event continuity"):
        prove_restart_continuity(before, changed)

    invalid_ready = deepcopy(after)
    invalid_ready["ready"]["content"]["receipt_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(RuntimeError, match="ready receipt differs"):
        prove_restart_continuity(before, invalid_ready)


def test_windows_update_acceptance_is_gate_d_bound_not_registry_bound():
    no_registry_policy = {
        "NoAutoRebootWithLoggedOnUsers": None,
        "AlwaysAutoRebootAtScheduledTime": None,
        "CBSRebootPending": False,
        "WURebootRequired": False,
        "PendingFileRename": False,
    }
    accepted = classify_windows_update(
        no_registry_policy,
        {
            "schema": RESTART_CONTINUITY_SCHEMA,
            "passed": True,
            "verdict": "ACCEPTED",
        },
    )
    assert accepted == {
        "verdict": "ACCEPTED",
        "reason": "restart continuity proven",
        "pending_state_clear": True,
        "registry_policy_required": False,
    }

    unresolved = classify_windows_update(no_registry_policy, {"passed": False})
    assert unresolved["verdict"] == "UNRESOLVED"

    pending = dict(no_registry_policy, PendingFileRename=True)
    still_unresolved = classify_windows_update(
        pending,
        {
            "schema": RESTART_CONTINUITY_SCHEMA,
            "passed": True,
            "verdict": "ACCEPTED",
        },
    )
    assert still_unresolved["verdict"] == "UNRESOLVED"
    assert still_unresolved["pending_state_clear"] is False


def test_restart_continuity_loader_binds_both_raw_capture_files(tmp_path: Path):
    paths = [tmp_path / "before.json", tmp_path / "after.json"]
    packets = [
        _reboot_capture("2026-08-15T01:00:00Z"),
        _reboot_capture("2026-08-15T02:00:00Z"),
    ]
    for path, packet in zip(paths, packets, strict=True):
        unsigned = dict(packet)
        unsigned.pop("receipt_sha256")
        payload = dict(unsigned)
        payload["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    proof = load_restart_continuity_proof(*paths)
    assert proof["passed"] is True
    assert proof["capture_files"]["before"]["sha256"] == hashlib.sha256(
        paths[0].read_bytes()
    ).hexdigest()
    assert proof["capture_files"]["after"]["sha256"] == hashlib.sha256(
        paths[1].read_bytes()
    ).hexdigest()

    paths[1].write_text(
        paths[1].read_text(encoding="utf-8").replace("reboot-capture", "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="receipt differs"):
        load_restart_continuity_proof(*paths)


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
    assert 'value in {"PASS", "ACCEPTED", "NOT_APPLICABLE"}' in source
    assert "--before-reboot-capture" in source
    assert "--after-reboot-capture" in source
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
