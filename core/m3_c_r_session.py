"""Scoped authorization, resume, and review helpers for M3-C-R."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import core.m3_c_p_private_device_goal_window_authorization_pin as m3_c_p_pin
from core.m3_c_p_private_device_goal_window_authorization_pin import (
    M3CPAuthorizationPinError,
    M3CPOperatorAuthorizationBinding,
    reviewed_operator_pin_session,
)
from core.m3_c_q_local_goal_window_authorization_capture import (
    build_local_reviewed_authorization_pin,
    local_pin_from_mapping,
)
from core.m3_c_r_contracts import (
    EXPECTED_PACKAGE_DIGEST,
    PUBLIC_REVIEW_SCHEMA,
    M3CRResumableOperatorError,
    file_sha256,
    load_canonical_mapping,
    mapping_digest,
    private_paths,
    write_idempotent_canonical,
)


def load_local_pin_for_binding(
    private_root: str | Path,
    binding: M3CPOperatorAuthorizationBinding,
) -> Any:
    paths = private_paths(private_root)
    pin_path = paths["pin"] if paths["pin"].exists() else paths["consumed_pin"]
    pin = local_pin_from_mapping(
        load_canonical_mapping(pin_path, field="local authorization pin")
    )
    if pin != build_local_reviewed_authorization_pin(binding):
        raise M3CRResumableOperatorError(
            "local authorization pin differs from package binding"
        )
    return pin


@contextmanager
def local_reviewed_operator_session(
    private_root: str | Path,
    binding: M3CPOperatorAuthorizationBinding,
) -> Iterator[Any]:
    pin = load_local_pin_for_binding(private_root, binding)
    previous = m3_c_p_pin._ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN
    if previous is not None:
        raise M3CRResumableOperatorError(
            "local authorization pin is already active"
        )
    m3_c_p_pin._ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN = pin
    try:
        with reviewed_operator_pin_session(binding) as active:
            yield active
    except M3CPAuthorizationPinError as exc:
        raise M3CRResumableOperatorError(
            "reviewed operator pin session failed"
        ) from exc
    finally:
        m3_c_p_pin._ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN = previous


def consume_local_pin(private_root: str | Path) -> None:
    paths = private_paths(private_root)
    source, target = paths["pin"], paths["consumed_pin"]
    if target.exists():
        if source.exists():
            raise M3CRResumableOperatorError(
                "both active and consumed local pin paths exist"
            )
        return
    if not source.is_file():
        raise M3CRResumableOperatorError("active local pin is absent")
    try:
        source.rename(target)
    except OSError as exc:
        raise M3CRResumableOperatorError(
            "failed to mark local pin consumed"
        ) from exc
    if os.name != "nt":
        os.chmod(target, 0o600)


def existing_completed_operator_receipt(
    private_root: str | Path,
) -> dict[str, Any] | None:
    paths = private_paths(private_root)
    if not paths["operator_receipt"].exists():
        return None
    receipt = load_canonical_mapping(
        paths["operator_receipt"], field="operator execution receipt"
    )
    if (
        receipt.get("private_path_plaintext_public") is not False
        or receipt.get("raw_private_text_public") is not False
    ):
        raise M3CRResumableOperatorError("operator receipt boundary mismatch")
    consume_local_pin(private_root)
    return receipt


def refuse_partial_execution(private_root: str | Path) -> None:
    paths = private_paths(private_root)
    if paths["operator_receipt"].exists():
        return
    occupied = [
        paths[name].name
        for name in ("working_store", "baseline_backup", "separate_restore")
        if paths[name].exists()
    ]
    if occupied:
        raise M3CRResumableOperatorError(
            "partial canonical execution state preserved; do not rerun stage 4: "
            + ", ".join(occupied)
        )


def public_review_mapping(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    receipt = existing_completed_operator_receipt(private_root)
    if receipt is None:
        raise M3CRResumableOperatorError("operator execution receipt is absent")
    return {
        "human_reviewed": True,
        "legacy_goal_authority_transfer_authorized": False,
        "legacy_migration_authorized": False,
        "m3_e_authority_open": False,
        "operator_execution_receipt_digest": mapping_digest(receipt),
        "package_digest": EXPECTED_PACKAGE_DIGEST,
        "probe_count": 4,
        "raw_private_text_or_path_output": False,
        "schema_version": PUBLIC_REVIEW_SCHEMA,
        "single_use_execution_complete": True,
        "source_operator_receipt_file_sha256": file_sha256(
            paths["operator_receipt"]
        ),
    }


def record_public_review(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    review = public_review_mapping(private_root)
    write_idempotent_canonical(paths["public_review"], review)
    return review
