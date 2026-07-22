"""M2-C bounded legacy evidence and comparison-only dual read.

Import and construction perform no I/O. Callers explicitly supply detached legacy
bytes, a separately decoded bounded snapshot, and an already initialized M2-A
store. No file discovery, pickle decoding, store initialization, append, snapshot
write, runtime hook, recovery authority, or cutover exists here.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_kernel import SHADOW_AUTHORITY, canonical_json_object
from core.shadow_observer import ACTIVATION_LEARN_PAIR_TARGET
from core.shadow_projection import (
    PROJECTION_SCHEMA_VERSION,
    ActivationLearnPairShadowState,
    ShadowProjectionError,
    compare_activation_learn_pair_equivalence,
    replay_activation_learn_pair,
)
from core.sqlite_shadow_store import SQLiteShadowStore

LEGACY_SOURCE_SCHEMA_VERSION = "legacy.activation-learn-pair.snapshot.v1"
STATE_EVIDENCE_SCHEMA_VERSION = "eve.m2-c-state-evidence.v1"
LEGACY_EVIDENCE_SCHEMA_VERSION = "eve.m2-c-legacy-sidecar-evidence.v1"
MIGRATION_CANDIDATE_SCHEMA_VERSION = "eve.m2-c-migration-candidate.v1"
DUAL_READ_REPORT_SCHEMA_VERSION = "eve.m2-c-dual-read-report.v1"
STATE_SERIALIZATION_SCHEMA_VERSION = "eve.canonical-json-state.v1"
COMPARISON_AUTHORITY = "comparison_only"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M2CDualReadError(ValueError):
    """Base error for malformed or out-of-scope M2-C inputs."""


class LegacySidecarIncompatible(M2CDualReadError):
    """Raised when incompatible evidence is promoted to a candidate."""


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Mapping[str, Any], field: str) -> str:
    return _sha_text(_canon(value, field))


def _require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M2CDualReadError(f"{field} must be a lowercase SHA-256 digest")


def _require_source(label: str, digest: str, count: int, schema: str) -> None:
    if not isinstance(label, str) or not label.strip():
        raise M2CDualReadError("source_label must be non-empty")
    _require_digest(digest, "source_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise M2CDualReadError("source_byte_count must be positive")
    if schema != LEGACY_SOURCE_SCHEMA_VERSION:
        raise M2CDualReadError("unsupported legacy source schema")


def _state_manifest(snapshot_json: str, snapshot: Mapping[str, Any]) -> str:
    return _canon(
        {
            "canonical_bytes": len(snapshot_json.encode("utf-8")),
            "collection_counts": {
                "calls": len(snapshot["calls"]),
                "learned": len(snapshot["learned"]),
            },
            "hash_algorithm": "sha256",
            "key_domain": ["calls", "learned"],
            "serialization_schema": STATE_SERIALIZATION_SCHEMA_VERSION,
            "state_schema_version": PROJECTION_SCHEMA_VERSION,
            "top_level_key_count": 2,
        },
        "m2_c_state_manifest",
    )


@dataclass(frozen=True, slots=True)
class StateEvidence:
    snapshot_json: str
    snapshot_digest: str
    manifest_json: str
    manifest_digest: str
    schema_version: str = STATE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_EVIDENCE_SCHEMA_VERSION:
            raise M2CDualReadError("unsupported state evidence schema")
        _require_digest(self.snapshot_digest, "snapshot_digest")
        _require_digest(self.manifest_digest, "manifest_digest")
        if _sha_text(self.snapshot_json) != self.snapshot_digest:
            raise M2CDualReadError("snapshot digest mismatch")
        try:
            decoded = json.loads(self.snapshot_json)
            if not isinstance(decoded, Mapping):
                raise M2CDualReadError("state evidence must decode to an object")
            state = ActivationLearnPairShadowState.from_initial_snapshot(decoded)
        except (json.JSONDecodeError, ShadowProjectionError, TypeError, ValueError) as exc:
            raise M2CDualReadError("invalid bounded state evidence") from exc
        canonical = _canon(state.snapshot, "m2_c_state")
        if self.snapshot_json != canonical:
            raise M2CDualReadError("snapshot_json must use the canonical serialization")
        expected_manifest = _state_manifest(canonical, state.snapshot)
        if (
            self.manifest_json != expected_manifest
            or self.manifest_digest != _sha_text(expected_manifest)
        ):
            raise M2CDualReadError("state manifest mismatch")

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "StateEvidence":
        try:
            state = ActivationLearnPairShadowState.from_initial_snapshot(snapshot)
        except (ShadowProjectionError, TypeError, ValueError) as exc:
            raise M2CDualReadError("invalid bounded state evidence") from exc
        snapshot_json = _canon(state.snapshot, "m2_c_state")
        manifest_json = _state_manifest(snapshot_json, state.snapshot)
        return cls(
            snapshot_json=snapshot_json,
            snapshot_digest=_sha_text(snapshot_json),
            manifest_json=manifest_json,
            manifest_digest=_sha_text(manifest_json),
        )

    @property
    def snapshot(self) -> dict[str, Any]:
        value = json.loads(self.snapshot_json)
        if not isinstance(value, dict):
            raise M2CDualReadError("state evidence is not an object")
        return value


@dataclass(frozen=True, slots=True)
class LegacySidecarAssessment:
    source_label: str
    source_sha256: str
    source_byte_count: int
    source_schema_version: str
    state: StateEvidence | None
    compatible: bool
    incompatibilities: tuple[str, ...]
    schema_version: str = LEGACY_EVIDENCE_SCHEMA_VERSION
    legacy_authority_retained: bool = True
    runtime_integrated: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LEGACY_EVIDENCE_SCHEMA_VERSION:
            raise M2CDualReadError("unsupported legacy evidence schema")
        if not isinstance(self.source_label, str) or not self.source_label.strip():
            raise M2CDualReadError("source_label must be non-empty")
        _require_digest(self.source_sha256, "source_sha256")
        if (
            isinstance(self.source_byte_count, bool)
            or not isinstance(self.source_byte_count, int)
            or self.source_byte_count < 0
        ):
            raise M2CDualReadError("source_byte_count must be non-negative")
        if not isinstance(self.incompatibilities, tuple):
            raise M2CDualReadError("incompatibilities must be immutable")
        if self.compatible != (not self.incompatibilities):
            raise M2CDualReadError("compatible disagrees with incompatibilities")
        if self.compatible:
            _require_source(
                self.source_label,
                self.source_sha256,
                self.source_byte_count,
                self.source_schema_version,
            )
            if not isinstance(self.state, StateEvidence):
                raise M2CDualReadError("compatible evidence requires bounded state")
        elif self.state is not None:
            raise M2CDualReadError("incompatible evidence cannot expose migration state")
        if self.legacy_authority_retained is not True or self.runtime_integrated is not False:
            raise M2CDualReadError("legacy authority boundary changed")


@dataclass(frozen=True, slots=True)
class MigrationCandidate:
    source_label: str
    source_sha256: str
    source_byte_count: int
    source_schema_version: str
    legacy_state: StateEvidence
    stream_id: str
    candidate_digest: str
    schema_version: str = MIGRATION_CANDIDATE_SCHEMA_VERSION
    authority: str = COMPARISON_AUTHORITY
    writes_performed: bool = False
    runtime_integrated: bool = False
    legacy_authority_retained: bool = True

    def __post_init__(self) -> None:
        _require_source(
            self.source_label,
            self.source_sha256,
            self.source_byte_count,
            self.source_schema_version,
        )
        if not isinstance(self.legacy_state, StateEvidence):
            raise M2CDualReadError("migration candidate requires StateEvidence")
        _require_digest(self.candidate_digest, "candidate_digest")
        if (
            self.schema_version != MIGRATION_CANDIDATE_SCHEMA_VERSION
            or self.stream_id != ACTIVATION_LEARN_PAIR_TARGET.stream_id
        ):
            raise M2CDualReadError("migration candidate scope mismatch")
        if (
            self.authority != COMPARISON_AUTHORITY
            or self.writes_performed is not False
            or self.runtime_integrated is not False
            or self.legacy_authority_retained is not True
        ):
            raise M2CDualReadError("migration candidate changed authority or effects")
        if _digest(_candidate_material(self), "migration_candidate") != self.candidate_digest:
            raise M2CDualReadError("migration candidate digest mismatch")


def _candidate_material(value: MigrationCandidate) -> dict[str, Any]:
    return {
        "authority": value.authority,
        "legacy_manifest_digest": value.legacy_state.manifest_digest,
        "legacy_snapshot_digest": value.legacy_state.snapshot_digest,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "schema_version": value.schema_version,
        "source_byte_count": value.source_byte_count,
        "source_label": value.source_label,
        "source_schema_version": value.source_schema_version,
        "source_sha256": value.source_sha256,
        "stream_id": value.stream_id,
    }


@dataclass(frozen=True, slots=True)
class DualReadReport:
    source_label: str
    source_sha256: str
    source_byte_count: int
    source_schema_version: str
    legacy_state: StateEvidence
    shadow_state: StateEvidence | None
    shadow_event_count: int
    shadow_sequence: int | None
    shadow_integrity_before_digest: str
    shadow_integrity_after_digest: str
    replay_verified: bool
    matches: bool
    mismatches: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    state_changed: bool
    writes_performed: bool
    transition_hash: str
    report_digest: str
    schema_version: str = DUAL_READ_REPORT_SCHEMA_VERSION
    comparison_authority: str = COMPARISON_AUTHORITY
    shadow_authority: str = SHADOW_AUTHORITY
    legacy_authority_retained: bool = True
    runtime_integrated: bool = False

    def __post_init__(self) -> None:
        _require_source(
            self.source_label,
            self.source_sha256,
            self.source_byte_count,
            self.source_schema_version,
        )
        if not isinstance(self.legacy_state, StateEvidence):
            raise M2CDualReadError("dual-read report requires legacy StateEvidence")
        if self.shadow_state is not None and not isinstance(self.shadow_state, StateEvidence):
            raise M2CDualReadError("shadow_state must be StateEvidence or None")
        for field in (
            "shadow_integrity_before_digest",
            "shadow_integrity_after_digest",
            "transition_hash",
            "report_digest",
        ):
            _require_digest(getattr(self, field), field)
        if self.schema_version != DUAL_READ_REPORT_SCHEMA_VERSION:
            raise M2CDualReadError("unsupported dual-read report schema")
        if (
            isinstance(self.shadow_event_count, bool)
            or not isinstance(self.shadow_event_count, int)
            or self.shadow_event_count < 0
        ):
            raise M2CDualReadError("shadow_event_count must be non-negative")
        if self.shadow_sequence is not None and (
            isinstance(self.shadow_sequence, bool)
            or not isinstance(self.shadow_sequence, int)
            or self.shadow_sequence < 0
        ):
            raise M2CDualReadError("shadow_sequence must be non-negative")
        if not isinstance(self.mismatches, tuple) or not isinstance(self.incompatibilities, tuple):
            raise M2CDualReadError("findings must be immutable")
        if self.replay_verified != (self.shadow_state is not None):
            raise M2CDualReadError("replay evidence is incomplete")
        if self.matches != (
            self.replay_verified and not self.mismatches and not self.incompatibilities
        ):
            raise M2CDualReadError("matches disagrees with replay and findings")
        changed = self.shadow_integrity_before_digest != self.shadow_integrity_after_digest
        if self.state_changed != changed:
            raise M2CDualReadError("state_changed disagrees with before/after evidence")
        if self.writes_performed is not False:
            raise M2CDualReadError("comparison-only reports cannot claim writes")
        if (
            self.comparison_authority != COMPARISON_AUTHORITY
            or self.shadow_authority != SHADOW_AUTHORITY
            or self.legacy_authority_retained is not True
            or self.runtime_integrated is not False
        ):
            raise M2CDualReadError("dual-read report changed authority")
        if _digest(_transition_material(self), "dual_read_transition") != self.transition_hash:
            raise M2CDualReadError("transition hash mismatch")
        if _digest(_report_material(self), "dual_read_report") != self.report_digest:
            raise M2CDualReadError("report digest mismatch")


def _transition_material(value: DualReadReport) -> dict[str, Any]:
    return {
        "incompatibilities": list(value.incompatibilities),
        "legacy_snapshot_digest": value.legacy_state.snapshot_digest,
        "mismatches": list(value.mismatches),
        "replay_verified": value.replay_verified,
        "shadow_integrity_after_digest": value.shadow_integrity_after_digest,
        "shadow_integrity_before_digest": value.shadow_integrity_before_digest,
        "shadow_snapshot_digest": (
            None if value.shadow_state is None else value.shadow_state.snapshot_digest
        ),
        "source_sha256": value.source_sha256,
        "state_changed": value.state_changed,
        "stream_id": ACTIVATION_LEARN_PAIR_TARGET.stream_id,
    }


def _report_material(value: DualReadReport) -> dict[str, Any]:
    return {
        "comparison_authority": value.comparison_authority,
        "incompatibilities": list(value.incompatibilities),
        "legacy_authority_retained": value.legacy_authority_retained,
        "legacy_manifest_digest": value.legacy_state.manifest_digest,
        "legacy_snapshot_digest": value.legacy_state.snapshot_digest,
        "matches": value.matches,
        "mismatches": list(value.mismatches),
        "replay_verified": value.replay_verified,
        "runtime_integrated": value.runtime_integrated,
        "schema_version": value.schema_version,
        "shadow_authority": value.shadow_authority,
        "shadow_event_count": value.shadow_event_count,
        "shadow_integrity_after_digest": value.shadow_integrity_after_digest,
        "shadow_integrity_before_digest": value.shadow_integrity_before_digest,
        "shadow_manifest_digest": (
            None if value.shadow_state is None else value.shadow_state.manifest_digest
        ),
        "shadow_sequence": value.shadow_sequence,
        "shadow_snapshot_digest": (
            None if value.shadow_state is None else value.shadow_state.snapshot_digest
        ),
        "source_byte_count": value.source_byte_count,
        "source_label": value.source_label,
        "source_schema_version": value.source_schema_version,
        "source_sha256": value.source_sha256,
        "state_changed": value.state_changed,
        "transition_hash": value.transition_hash,
        "writes_performed": value.writes_performed,
    }


def assess_legacy_sidecar(
    *,
    source_label: str,
    source_bytes: bytes | bytearray | memoryview,
    source_schema_version: str,
    decoded_snapshot: Mapping[str, Any] | None,
) -> LegacySidecarAssessment:
    if not isinstance(source_label, str) or not source_label.strip():
        raise M2CDualReadError("source_label must be non-empty")
    if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
        raise M2CDualReadError("source_bytes must be bytes-like")
    raw = bytes(source_bytes)
    problems: list[str] = []
    if not raw:
        problems.append("empty_source_bytes")
    if source_schema_version != LEGACY_SOURCE_SCHEMA_VERSION:
        problems.append("unsupported_source_schema")
    state: StateEvidence | None = None
    if not isinstance(decoded_snapshot, Mapping):
        problems.append("snapshot_not_mapping")
    else:
        try:
            state = StateEvidence.from_snapshot(decoded_snapshot)
        except (M2CDualReadError, TypeError, ValueError):
            problems.append("invalid_bounded_snapshot")
    findings = tuple(sorted(set(problems)))
    return LegacySidecarAssessment(
        source_label=source_label,
        source_sha256=_sha_bytes(raw),
        source_byte_count=len(raw),
        source_schema_version=source_schema_version,
        state=None if findings else state,
        compatible=not findings,
        incompatibilities=findings,
    )


def build_migration_candidate(assessment: LegacySidecarAssessment) -> MigrationCandidate:
    if not isinstance(assessment, LegacySidecarAssessment):
        raise M2CDualReadError("assessment must be LegacySidecarAssessment")
    if not assessment.compatible or assessment.state is None:
        raise LegacySidecarIncompatible(
            "legacy evidence is incompatible: " + ",".join(assessment.incompatibilities)
        )
    material = {
        "authority": COMPARISON_AUTHORITY,
        "legacy_manifest_digest": assessment.state.manifest_digest,
        "legacy_snapshot_digest": assessment.state.snapshot_digest,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "schema_version": MIGRATION_CANDIDATE_SCHEMA_VERSION,
        "source_byte_count": assessment.source_byte_count,
        "source_label": assessment.source_label,
        "source_schema_version": assessment.source_schema_version,
        "source_sha256": assessment.source_sha256,
        "stream_id": ACTIVATION_LEARN_PAIR_TARGET.stream_id,
    }
    return MigrationCandidate(
        source_label=assessment.source_label,
        source_sha256=assessment.source_sha256,
        source_byte_count=assessment.source_byte_count,
        source_schema_version=assessment.source_schema_version,
        legacy_state=assessment.state,
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        candidate_digest=_digest(material, "migration_candidate"),
    )


def compare_dual_read(
    *,
    assessment: LegacySidecarAssessment,
    store: SQLiteShadowStore,
    initial_snapshot: Mapping[str, Any],
) -> DualReadReport:
    candidate = build_migration_candidate(assessment)
    if not isinstance(store, SQLiteShadowStore):
        raise M2CDualReadError("store must be SQLiteShadowStore")
    try:
        initial = ActivationLearnPairShadowState.from_initial_snapshot(initial_snapshot)
    except (ShadowProjectionError, TypeError, ValueError) as exc:
        raise M2CDualReadError("invalid initial snapshot") from exc

    before = store.integrity_check()
    problems: list[str] = []
    mismatches: tuple[str, ...] = ()
    shadow_state: StateEvidence | None = None
    sequence: int | None = None
    event_count = before.event_count

    if not before.valid:
        problems.extend(f"shadow_integrity_before:{item}" for item in before.errors)
    else:
        try:
            events = store.events(stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id)
            event_count = len(events)
            left = replay_activation_learn_pair(initial, events)
            right = replay_activation_learn_pair(
                ActivationLearnPairShadowState.from_initial_snapshot(initial_snapshot),
                events,
            )
            if left.digest != right.digest or left.snapshot != right.snapshot:
                problems.append("shadow_replay_nondeterministic")
            else:
                shadow_state = StateEvidence.from_snapshot(left.snapshot)
                equivalence = compare_activation_learn_pair_equivalence(
                    left,
                    candidate.legacy_state.snapshot,
                )
                mismatches = equivalence.mismatches
                if equivalence.projected_digest != shadow_state.snapshot_digest:
                    problems.append("shadow_digest_method_mismatch")
                sequence = left.sequence
        except (ShadowProjectionError, M2CDualReadError, TypeError, ValueError) as exc:
            problems.append(f"shadow_replay:{type(exc).__name__}")
            shadow_state = None
            sequence = None
            mismatches = ()

    after = store.integrity_check()
    if not after.valid:
        problems.extend(f"shadow_integrity_after:{item}" for item in after.errors)
    changed = before.report_digest != after.report_digest
    if changed:
        problems.append("shadow_store_changed_during_comparison")
    incompatibilities = tuple(sorted(set(problems)))
    replay_verified = shadow_state is not None
    matches = replay_verified and not mismatches and not incompatibilities

    transition_material = {
        "incompatibilities": list(incompatibilities),
        "legacy_snapshot_digest": candidate.legacy_state.snapshot_digest,
        "mismatches": list(mismatches),
        "replay_verified": replay_verified,
        "shadow_integrity_after_digest": after.report_digest,
        "shadow_integrity_before_digest": before.report_digest,
        "shadow_snapshot_digest": (
            None if shadow_state is None else shadow_state.snapshot_digest
        ),
        "source_sha256": candidate.source_sha256,
        "state_changed": changed,
        "stream_id": ACTIVATION_LEARN_PAIR_TARGET.stream_id,
    }
    transition_hash = _digest(transition_material, "dual_read_transition")
    report_material = {
        "comparison_authority": COMPARISON_AUTHORITY,
        "incompatibilities": list(incompatibilities),
        "legacy_authority_retained": True,
        "legacy_manifest_digest": candidate.legacy_state.manifest_digest,
        "legacy_snapshot_digest": candidate.legacy_state.snapshot_digest,
        "matches": matches,
        "mismatches": list(mismatches),
        "replay_verified": replay_verified,
        "runtime_integrated": False,
        "schema_version": DUAL_READ_REPORT_SCHEMA_VERSION,
        "shadow_authority": SHADOW_AUTHORITY,
        "shadow_event_count": event_count,
        "shadow_integrity_after_digest": after.report_digest,
        "shadow_integrity_before_digest": before.report_digest,
        "shadow_manifest_digest": (
            None if shadow_state is None else shadow_state.manifest_digest
        ),
        "shadow_sequence": sequence,
        "shadow_snapshot_digest": (
            None if shadow_state is None else shadow_state.snapshot_digest
        ),
        "source_byte_count": candidate.source_byte_count,
        "source_label": candidate.source_label,
        "source_schema_version": candidate.source_schema_version,
        "source_sha256": candidate.source_sha256,
        "state_changed": changed,
        "transition_hash": transition_hash,
        "writes_performed": False,
    }
    return DualReadReport(
        source_label=candidate.source_label,
        source_sha256=candidate.source_sha256,
        source_byte_count=candidate.source_byte_count,
        source_schema_version=candidate.source_schema_version,
        legacy_state=candidate.legacy_state,
        shadow_state=shadow_state,
        shadow_event_count=event_count,
        shadow_sequence=sequence,
        shadow_integrity_before_digest=before.report_digest,
        shadow_integrity_after_digest=after.report_digest,
        replay_verified=replay_verified,
        matches=matches,
        mismatches=mismatches,
        incompatibilities=incompatibilities,
        state_changed=changed,
        writes_performed=False,
        transition_hash=transition_hash,
        report_digest=_digest(report_material, "dual_read_report"),
    )
