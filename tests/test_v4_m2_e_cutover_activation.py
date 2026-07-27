import json
import os
from pathlib import Path

import pytest

from core.m2_e_cutover_activation import (
    EVENT_STORE_ACTIVE_ROLE,
    EVENT_STORE_ROLLBACK_ROLE,
    HUMAN_AUTHORIZATION_DIGEST,
    LEGACY_RUNTIME_AUTHORITY,
    MINIMUM_LEGACY_PARALLEL_DAYS,
    M2ECutoverActivationError,
    active_cutover_authority,
    build_operational_rollback_record,
    resolve_cutover_authority,
    verify_human_authorization_artifact,
)
from scripts.operator.m2_e_cutover_rollback import ROLLBACK_FILENAME, main as rollback_main


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = ROOT / "docs/audit/M2_E_CUTOVER_HUMAN_AUTHORIZATION_RECORD.json"


def _authorization_record():
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def test_a1_human_authorization_digest_is_the_only_activation_source():
    record = _authorization_record()

    assert verify_human_authorization_artifact(record) == HUMAN_AUTHORIZATION_DIGEST


def test_active_cutover_opens_only_v4_native_persistence_and_m3_authority():
    state = active_cutover_authority()

    assert state.authorization_digest == HUMAN_AUTHORIZATION_DIGEST
    assert state.cutover_authorized is True
    assert state.m3_authority_open is True
    assert state.event_store_role == EVENT_STORE_ACTIVE_ROLE
    assert state.retained_shadow_history_authoritative is True
    assert state.legacy_runtime_authority == LEGACY_RUNTIME_AUTHORITY
    assert state.legacy_domain_authority_transfer_authorized is False
    assert state.m3_e_affect_cutover_authorized is False
    assert state.minimum_legacy_parallel_days == MINIMUM_LEGACY_PARALLEL_DAYS == 7
    assert state.legacy_persistence_path_changed is False
    assert state.operational_rollback_active is False


def test_operational_rollback_fails_closed_without_revoking_human_decision():
    record = build_operational_rollback_record(
        requested_by="김민석",
        reason="test_fail_closed_rollback",
    )

    assert record["human_authorization_revoked"] is False
    state = resolve_cutover_authority(record)
    assert state.cutover_authorized is False
    assert state.m3_authority_open is False
    assert state.event_store_role == EVENT_STORE_ROLLBACK_ROLE
    assert state.legacy_runtime_authority == LEGACY_RUNTIME_AUTHORITY
    assert state.legacy_domain_authority_transfer_authorized is False
    assert state.m3_e_affect_cutover_authorized is False
    assert state.legacy_persistence_path_changed is False
    assert state.operational_rollback_active is True


def test_tampered_rollback_record_fails_closed():
    record = build_operational_rollback_record(
        requested_by="김민석",
        reason="test_tamper",
    )
    record["m3_e_affect_cutover_authorized"] = True

    with pytest.raises(M2ECutoverActivationError):
        resolve_cutover_authority(record)


def test_one_command_rollback_writes_private_idempotent_control(tmp_path, capsys):
    private_root = tmp_path / "eve-m2e-window-private"
    args = [
        "--private-root",
        str(private_root),
        "--requested-by",
        "김민석",
        "--reason",
        "test_operator_rollback",
    ]

    assert rollback_main(args) == 0
    first = capsys.readouterr().out.strip()
    assert rollback_main(args) == 0
    second = capsys.readouterr().out.strip()
    assert first == second

    target = private_root / ROLLBACK_FILENAME
    record = json.loads(target.read_text(encoding="utf-8"))
    state = resolve_cutover_authority(record)
    assert state.operational_rollback_active is True
    if os.name != "nt":
        assert target.stat().st_mode & 0o077 == 0
        assert private_root.stat().st_mode & 0o077 == 0
