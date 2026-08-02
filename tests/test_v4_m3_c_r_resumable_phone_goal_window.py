from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from core.m3_c_r_resumable_phone_goal_window import (
    BASELINE_BACKUP_FILENAME,
    CONSUMED_PIN_FILENAME,
    EXPECTED_FORBIDDEN_DIGEST_COUNT,
    EXPECTED_PACKAGE_DIGEST,
    EXPECTED_REVIEW_SUMMARY,
    MINIMUM_AVAILABLE_MIB,
    OPERATOR_RECEIPT_FILENAME,
    PIN_FILENAME,
    PRIVATE_ROOT_BASENAME,
    PUBLIC_REVIEW_FILENAME,
    SEPARATE_RESTORE_FILENAME,
    WORKING_STORE_FILENAME,
    M3CRResumableOperatorError,
    private_paths,
    read_forbidden_digests,
    write_idempotent_canonical,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    ROOT / "core/m3_c_r_contracts.py",
    ROOT / "core/m3_c_r_preflight.py",
    ROOT / "core/m3_c_r_session.py",
    ROOT / "core/m3_c_r_resumable_phone_goal_window.py",
)
STAGES = tuple(
    ROOT / f"scripts/operator/m3_c_r_stage{number}_{name}.py"
    for number, name in (
        (1, "record_review"),
        (2, "capture_pin"),
        (3, "preflight"),
        (4, "execute_window"),
        (5, "record_public_review"),
    )
)


def _engine_calls(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_full_engine"
    )


def test_only_stage_four_constructs_the_engine_once():
    for path in CORE_FILES:
        assert "build_full_engine" not in path.read_text(encoding="utf-8")
    for index, path in enumerate(STAGES, start=1):
        assert _engine_calls(path) == (1 if index == 4 else 0)


def test_no_stage_contains_interactive_terminal_wait():
    forbidden_text = ("read -r", "input(", "sys.stdin", "stdin.read")
    for path in STAGES:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_text:
            assert token not in source
    assert "--reviewed" in STAGES[0].read_text(encoding="utf-8")
    assert "--reviewed" in STAGES[4].read_text(encoding="utf-8")


def test_exact_phone_private_path_contract():
    root = Path("/tmp") / PRIVATE_ROOT_BASENAME
    paths = private_paths(root)
    assert paths["pin"].name == PIN_FILENAME
    assert paths["consumed_pin"].name == CONSUMED_PIN_FILENAME
    assert paths["working_store"].name == WORKING_STORE_FILENAME
    assert paths["baseline_backup"].name == BASELINE_BACKUP_FILENAME
    assert paths["separate_restore"].name == SEPARATE_RESTORE_FILENAME
    assert paths["operator_receipt"].name == OPERATOR_RECEIPT_FILENAME
    assert paths["public_review"].name == PUBLIC_REVIEW_FILENAME


def test_accepted_review_summary_preserves_closed_authority():
    assert EXPECTED_REVIEW_SUMMARY["package_digest"] == EXPECTED_PACKAGE_DIGEST
    assert EXPECTED_REVIEW_SUMMARY["probe_count"] == 4
    assert EXPECTED_REVIEW_SUMMARY["probe_operation_summary"] == [
        "goal_set",
        "tick",
        "goal_set",
        "tick",
    ]
    assert EXPECTED_REVIEW_SUMMARY["forbidden_prior_path_digest_count"] == 34
    assert EXPECTED_REVIEW_SUMMARY[
        "legacy_goal_authority_transfer_authorized"
    ] is False
    assert EXPECTED_REVIEW_SUMMARY["raw_private_text_or_path_output"] is False
    assert MINIMUM_AVAILABLE_MIB == 3072


def test_idempotent_write_reuses_exact_output_and_preserves_conflict(
    tmp_path: Path,
):
    target = tmp_path / "stage.json"
    value = {"a": 1, "closed": True}
    assert write_idempotent_canonical(target, value) is True
    assert write_idempotent_canonical(target, value) is False
    original = target.read_bytes()
    with pytest.raises(M3CRResumableOperatorError, match="conflicting or partial"):
        write_idempotent_canonical(target, {"a": 2, "closed": True})
    assert target.read_bytes() == original
    if os.name != "nt":
        assert target.stat().st_mode & 0o077 == 0


def test_forbidden_digest_reader_requires_exact_unique_set(tmp_path: Path):
    path = tmp_path / "forbidden.json"
    values = [f"{index:064x}" for index in range(EXPECTED_FORBIDDEN_DIGEST_COUNT)]
    path.write_text(json.dumps({"digests": values}), encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o600)
    assert read_forbidden_digests(path) == tuple(sorted(values))

    path.write_text(json.dumps({"digests": values[:-1]}), encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o600)
    with pytest.raises(M3CRResumableOperatorError, match="exactly 34"):
        read_forbidden_digests(path)


def test_stage_four_preserves_partial_state_instead_of_retrying():
    source = STAGES[3].read_text(encoding="utf-8")
    assert "refuse_partial_execution" in source
    assert "existing_completed_operator_receipt" in source
    assert "consume_local_pin" in source
    assert '"engine_load_count": 1' in source
    assert "require_memory_headroom()" in source
