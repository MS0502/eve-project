"""M2-E human-authorized persistence cutover activation contract.

This module promotes the event-kernel/SQLite substrate only for v4-native
subsystems. It does not transfer any legacy-domain authority and it does not
open M3-E affect authority. Import and construction perform no I/O.

Operational rollback is distinct from revoking the immutable A12 human
authorization: a valid private rollback control record temporarily resolves
this activation back to shadow-only while preserving legacy authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_kernel import canonical_json_object


ACTIVATION_SCHEMA = "eve.m2-e-cutover-activation.v1"
ROLLBACK_SCHEMA = "eve.m2-e-cutover-operational-rollback.v1"
HUMAN_AUTHORIZATION_SCHEMA = "eve.m2-e-cutover-human-authorization-record.v1"
HUMAN_AUTHORIZATION_DIGEST = (
    "3844e4d0a836924eb881048d45d98d89d5041f87d15a836686119a2d8487efbf"
)
SEALED_WINDOW_DIGEST = (
    "5bfd2bae9a60107b5bd647eeec30b602a4d6bca922e467755f17a04c990dafbb"
)
EVENT_STORE_ACTIVE_ROLE = "authoritative_persistence_substrate_for_v4_native_subsystems"
EVENT_STORE_ROLLBACK_ROLE = "shadow_only"
LEGACY_RUNTIME_AUTHORITY = "authoritative_per_domain_until_separate_domain_migration_gate"
MINIMUM_LEGACY_PARALLEL_DAYS = 7
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M2ECutoverActivationError(ValueError):
    """Malformed or scope-escaping cutover activation input."""


def _canonical(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def verify_human_authorization_artifact(value: Mapping[str, Any]) -> str:
    """Verify the immutable A-1 record and return its canonical digest."""
    if not isinstance(value, Mapping):
        raise M2ECutoverActivationError("human authorization must be a mapping")
    record = dict(value)
    digest = _digest(record, "m2_e_cutover_human_authorization")
    boundary = record.get("authority_boundary")
    non_approval = record.get("explicit_non_approval")
    evidence = record.get("evidence_pins")
    if (
        digest != HUMAN_AUTHORIZATION_DIGEST
        or record.get("schema_version") != HUMAN_AUTHORIZATION_SCHEMA
        or not isinstance(boundary, Mapping)
        or boundary.get("cutover_authorized") is not True
        or boundary.get("m3_authority_open") is not True
        or boundary.get("legacy_domain_authority_transfer_authorized") is not False
        or boundary.get("legacy_runtime_authority") != LEGACY_RUNTIME_AUTHORITY
        or boundary.get("m3_e_affect_cutover_authorized") is not False
        or not isinstance(non_approval, Mapping)
        or non_approval.get("legacy_domain_authority_transfer") is not False
        or non_approval.get("m3_e_affect_cutover") is not False
        or not isinstance(evidence, Mapping)
        or evidence.get("seal_digest") != SEALED_WINDOW_DIGEST
        or evidence.get("acceptance_checks_passed") != 12
        or evidence.get("acceptance_checks_required") != 12
        or evidence.get("events") != 288
        or evidence.get("death_recoveries") != 2
        or evidence.get("observed_midnights") != 4
    ):
        raise M2ECutoverActivationError("human authorization artifact does not match accepted scope")
    return digest


@dataclass(frozen=True, slots=True)
class CutoverAuthorityState:
    """Resolved authority state consumed by v4-native persistence callers."""

    authorization_digest: str
    cutover_authorized: bool
    m3_authority_open: bool
    event_store_role: str
    legacy_runtime_authority: str = LEGACY_RUNTIME_AUTHORITY
    legacy_domain_authority_transfer_authorized: bool = False
    m3_e_affect_cutover_authorized: bool = False
    retained_shadow_history_authoritative: bool = True
    minimum_legacy_parallel_days: int = MINIMUM_LEGACY_PARALLEL_DAYS
    legacy_persistence_path_changed: bool = False
    schema_version: str = ACTIVATION_SCHEMA
    operational_rollback_active: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.authorization_digest, str) or _DIGEST.fullmatch(
            self.authorization_digest
        ) is None:
            raise M2ECutoverActivationError("authorization_digest must be SHA-256")
        if self.authorization_digest != HUMAN_AUTHORIZATION_DIGEST:
            raise M2ECutoverActivationError("authority state is not pinned to accepted A-1 digest")
        if self.legacy_runtime_authority != LEGACY_RUNTIME_AUTHORITY:
            raise M2ECutoverActivationError("legacy per-domain authority escaped cutover scope")
        if self.legacy_domain_authority_transfer_authorized is not False:
            raise M2ECutoverActivationError("legacy-domain transfer requires a separate gate")
        if self.m3_e_affect_cutover_authorized is not False:
            raise M2ECutoverActivationError("M3-E affect authority requires its separate gate")
        if self.minimum_legacy_parallel_days != MINIMUM_LEGACY_PARALLEL_DAYS:
            raise M2ECutoverActivationError("seven-day legacy parallel minimum is immutable")
        if self.legacy_persistence_path_changed is not False:
            raise M2ECutoverActivationError("legacy persistence path must remain unchanged")
        if self.operational_rollback_active:
            expected = (False, False, EVENT_STORE_ROLLBACK_ROLE)
        else:
            expected = (True, True, EVENT_STORE_ACTIVE_ROLE)
        actual = (self.cutover_authorized, self.m3_authority_open, self.event_store_role)
        if actual != expected:
            raise M2ECutoverActivationError("resolved authority flags disagree with operating mode")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "authorization_digest": self.authorization_digest,
            "cutover_authorized": self.cutover_authorized,
            "event_store_role": self.event_store_role,
            "legacy_domain_authority_transfer_authorized": self.legacy_domain_authority_transfer_authorized,
            "legacy_persistence_path_changed": self.legacy_persistence_path_changed,
            "legacy_runtime_authority": self.legacy_runtime_authority,
            "m3_authority_open": self.m3_authority_open,
            "m3_e_affect_cutover_authorized": self.m3_e_affect_cutover_authorized,
            "minimum_legacy_parallel_days": self.minimum_legacy_parallel_days,
            "operational_rollback_active": self.operational_rollback_active,
            "retained_shadow_history_authoritative": self.retained_shadow_history_authoritative,
            "schema_version": self.schema_version,
        }


def active_cutover_authority() -> CutoverAuthorityState:
    """Return the human-authorized v4-native cutover state."""
    return CutoverAuthorityState(
        authorization_digest=HUMAN_AUTHORIZATION_DIGEST,
        cutover_authorized=True,
        m3_authority_open=True,
        event_store_role=EVENT_STORE_ACTIVE_ROLE,
    )


def build_operational_rollback_record(*, requested_by: str, reason: str) -> dict[str, Any]:
    """Build a private fail-closed rollback control record.

    This does not revoke or supersede the immutable human authorization.
    """
    if not isinstance(requested_by, str) or not requested_by.strip():
        raise M2ECutoverActivationError("requested_by must be non-empty")
    if not isinstance(reason, str) or not reason.strip():
        raise M2ECutoverActivationError("reason must be non-empty")
    material = {
        "action": "rollback_v4_native_persistence_to_shadow",
        "authorization_digest": HUMAN_AUTHORIZATION_DIGEST,
        "cutover_authorized": False,
        "event_store_role": EVENT_STORE_ROLLBACK_ROLE,
        "human_authorization_revoked": False,
        "legacy_domain_authority_transfer_authorized": False,
        "legacy_persistence_path_changed": False,
        "legacy_runtime_authority": LEGACY_RUNTIME_AUTHORITY,
        "m3_authority_open": False,
        "m3_e_affect_cutover_authorized": False,
        "reason": reason.strip(),
        "requested_by": requested_by.strip(),
        "schema_version": ROLLBACK_SCHEMA,
    }
    return {**material, "rollback_digest": _digest(material, "m2_e_operational_rollback")}


def resolve_cutover_authority(
    rollback_record: Mapping[str, Any] | None = None,
) -> CutoverAuthorityState:
    """Resolve active authority or a verified operational rollback state."""
    if rollback_record is None:
        return active_cutover_authority()
    if not isinstance(rollback_record, Mapping):
        raise M2ECutoverActivationError("rollback record must be a mapping")
    record = dict(rollback_record)
    supplied_digest = record.pop("rollback_digest", None)
    if not isinstance(supplied_digest, str) or _DIGEST.fullmatch(supplied_digest) is None:
        raise M2ECutoverActivationError("rollback_digest must be SHA-256")
    if _digest(record, "m2_e_operational_rollback") != supplied_digest:
        raise M2ECutoverActivationError("rollback digest mismatch")
    if (
        record.get("schema_version") != ROLLBACK_SCHEMA
        or record.get("action") != "rollback_v4_native_persistence_to_shadow"
        or record.get("authorization_digest") != HUMAN_AUTHORIZATION_DIGEST
        or record.get("human_authorization_revoked") is not False
        or record.get("cutover_authorized") is not False
        or record.get("m3_authority_open") is not False
        or record.get("event_store_role") != EVENT_STORE_ROLLBACK_ROLE
        or record.get("legacy_runtime_authority") != LEGACY_RUNTIME_AUTHORITY
        or record.get("legacy_domain_authority_transfer_authorized") is not False
        or record.get("m3_e_affect_cutover_authorized") is not False
        or record.get("legacy_persistence_path_changed") is not False
    ):
        raise M2ECutoverActivationError("rollback record escaped accepted cutover scope")
    return CutoverAuthorityState(
        authorization_digest=HUMAN_AUTHORIZATION_DIGEST,
        cutover_authorized=False,
        m3_authority_open=False,
        event_store_role=EVENT_STORE_ROLLBACK_ROLE,
        operational_rollback_active=True,
    )


def canonical_rollback_json(record: Mapping[str, Any]) -> str:
    """Serialize a rollback record deterministically for private persistence."""
    resolved = resolve_cutover_authority(record)
    if not resolved.operational_rollback_active:
        raise M2ECutoverActivationError("record is not an operational rollback")
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
