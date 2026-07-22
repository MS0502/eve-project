"""M2-C bounded legacy-sidecar evidence and dual-read comparison.

Import and construction perform no I/O. Callers must explicitly provide detached
legacy source bytes, a separately decoded bounded snapshot, and an already
initialized M2-A SQLite shadow store. This module never discovers files, decodes
pickle, initializes storage, appends events, writes snapshots, installs runtime
hooks, or changes legacy authority.
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
LEGACY_EVIDENCE_SCHEMA_VERSION = "eve.m2-c-legacy-sidecar-evidence.v1"
MIGRATION_CANDIDATE_SCHEMA_VERSION = "eve.m2-c-migration-candidate.v1"
DUAL_READ_REPORT_SCHEMA_VERSION = "eve.m2-c-dual-read-report.v1"
COMPARISON_AUTHORITY = "comparison_only"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M2CDualReadError(ValueError):
    """Base error for malformed or out-of-scope M2-C comparison inputs."""


class LegacySidecarIncompatible(M2CDualReadError):
    """Raised when incompatible legacy evidence is used as a migration candidate."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Mapping[str, Any], *, field: str) -> str:
    return canonical_json_object(value, field=field)


def _digest(value: Mapping[str, Any], *, field: str) -> str:
    return _sha_text(_canonical(value, field=field))


def _require_digest(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M2CDualReadError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class LegacySidecarAssessment:
    source_label: str
    source_sha256: str
    source_byte_count: int
    source_schema_version: str
    normalized_snapshot_json: str | None
    normalized_snapshot_digest: str | None
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
        _require_digest(self.source_sha256, field="source_sha256")
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
            if self.source_byte_count == 0:
                raise M2CDualReadError("compatible evidence cannot have empty source bytes")
            if self.source_schema_version != LEGACY_SOURCE_SCHEMA_VERSION:
                raise M2CDualReadError("compatible evidence has unsupported source schema")
            if not isinstance(self.normalized_snapshot_json, str):
                raise M2CDualReadError("compatible evidence requires normalized snapshot JSON")
            if not isinstance(self.normalized_snapshot_digest, str):
                raise M2CDualReadError("compatible evidence requires normalized snapshot digest")
            _require_digest(
                self.normalized_snapshot_digest,
                field="normalized_snapshot_digest",
            )
            if _sha_text(self.normalized_snapshot_json) != self.normalized_snapshot_digest:
                raise M2CDualReadError("normalized snapshot digest mismatch")
        elif self.normalized_snapshot_json is not None or self.normalized_snapshot_digest is not None:
            raise M2CDualReadError("incompatible evidence cannot expose a migration snapshot")
        if self.legacy_authority_retained is not True or self.runtime_integrated is not False:
            raise M2CDualReadError("legacy authority boundary changed")

    @property
    def normalized_snapshot(self) -> dict[str, Any] | None:
        if self.normalized_snapshot_json is None:
            return None
        value = json.loads(self.normalized_snapshot_json)
        if not isinstance(value, dict):
            raise M2CDualReadError("normalized snapshot is not an object")
        return value


@dataclass(frozen=True, slots=True)
class MigrationCandidate:
    source_label: str
    source_sha256: str
    legacy_snapshot_json: str
    legacy_snapshot_digest: str
    projection_schema_version: str
    stream_id: str
    candidate_digest: str
    schema_version: str = MIGRATION_CANDIDATE_SCHEMA_VERSION
    authority: str = COMPARISON_AUTHORITY
    writes_performed: bool = False
    runtime_integrated: bool = False
    legacy_authority_retained: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != MIGRATION_CANDIDATE_SCHEMA_VERSION:
            raise M2CDualReadError("unsupported migration candidate schema")
        for field in ("source_sha256", "legacy_snapshot_digest", "candidate_digest"):
            _require_digest(getattr(self, field), field=field)
        if _sha_text(self.legacy_snapshot_json) != self.legacy_snapshot_digest:
            raise M2CDualReadError("legacy snapshot digest mismatch")
        if self.projection_schema_version != PROJECTION_SCHEMA_VERSION:
            raise M2CDualReadError("migration candidate projection schema mismatch")
        if self.stream_id != ACTIVATION_LEARN_PAIR_TARGET.stream_id:
            raise M2CDualReadError("migration candidate stream is out of scope")
        if (
            self.authority != COMPARISON_AUTHORITY
            or self.writes_performed is not False
            or self.runtime_integrated is not False
            or self.legacy_authority_retained is not True
        ):
            raise M2CDualReadError("migration candidate changed authority or effects")
        material = {
            "authority": self.authority,
            "legacy_snapshot_digest": self.legacy_snapshot_digest,
            "projection_schema_version": self.projection_schema_version,
            "schema_version": self.schema_version,
            "source_label": self.source_label,
            "source_sha256": self.source_sha256,
            "stream_id": self.stream_id,
        }
        if _digest(material, field="migration_candidate") != self.candidate_digest:
            raise M2CDualReadError("migration candidate digest mismatch")


@dataclass(frozen=True, slots=True)
class DualReadReport:
    source_label: str
    source_sha256: str
    legacy_snapshot_digest: str
    shadow_snapshot_digest: str | None
    shadow_event_count: int
    shadow_sequence: int | None
    shadow_integrity_before_digest: str
    shadow_integrity_after_digest: str
    matches: bool
    mismatches: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    writes_performed: bool
    report_digest: str
    schema_version: str = DUAL_READ_REPORT_SCHEMA_VERSION
    comparison_authority: str = COMPARISON_AUTHORITY
    shadow_authority: str = SHADOW_AUTHORITY
    legacy_authority_retained: bool = True
    runtime_integrated: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != DUAL_READ_REPORT_SCHEMA_VERSION:
            raise M2CDualReadError("unsupported dual-read report schema")
        for field in (
            "source_sha256",
            "legacy_snapshot_digest",
            "shadow_integrity_before_digest",
            "shadow_integrity_after_digest",
            "report_digest",
        ):
            _require_digest(getattr(self, field), field=field)
        if self.shadow_snapshot_digest is not None:
            _require_digest(self.shadow_snapshot_digest, field="shadow_snapshot_digest")
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
            raise M2CDualReadError("dual-read findings must be immutable")
        if self.matches != (not self.mismatches and not self.incompatibilities):
            raise M2CDualReadError("matches disagrees with findings")
        if self.writes_performed != (
            self.shadow_integrity_before_digest != self.shadow_integrity_after_digest
        ):
            raise M2CDualReadError("writes_performed disagrees with integrity evidence")
        if (
            self.comparison_authority != COMPARISON_AUTHORITY
            or self.shadow_authority != SHADOW_AUTHORITY
            or self.legacy_authority_retained is not True
            or self.runtime_integrated is not False
        ):
            raise M2CDualReadError("dual-read report changed authority")
        material = {
            "comparison_authority": self.comparison_authority,
            "incompatibilities": list(self.incompatibilities),
            "legacy_authority_retained": self.legacy_authority_retained,
            "legacy_snapshot_digest": self.legacy_snapshot_digest,
            "matches": self.matches,
            "mismatches": list(self.mismatches),
            "runtime_integrated": self.runtime_integrated,
            "schema_version": self.schema_version,
            "shadow_authority": self.shadow_authority,
            "shadow_event_count": self.shadow_event_count,
            "shadow_integrity_after_digest": self.shadow_integrity_after_digest,
            "shadow_integrity_before_digest": self.shadow_integrity_before_digest,
            "shadow_sequence": self.shadow_sequence,
            "shadow_snapshot_digest": self.shadow_snapshot_digest,
            "source_label": self.source_label,
            "source_sha256": self.source_sha256,
            "writes_performed": self.writes_performed,
        }
        if _digest(material, field="dual_read_report") != self.report_digest:
            raise M2CDualReadError("dual-read report digest mismatch")


def assess_legacy_sidecar(
    *,
    source_label: str,
    source_bytes: bytes | bytearray | memoryview,
    source_schema_version: str,
    decoded_snapshot: Mapping[str, Any] | None,
) -> LegacySidecarAssessment:
    """Bind caller-supplied legacy bytes to a separately decoded bounded snapshot.

    The bytes are hashed only. This function deliberately performs no pickle,
    archive, filesystem, or arbitrary-object deserialization.
    """

    if not isinstance(source_label, str) or not source_label.strip():
        raise M2CDualReadError("source_label must be non-empty")
    if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
        raise M2CDualReadError("source_bytes must be bytes-like")
    raw = bytes(source_bytes)
    incompatibilities: list[str] = []
    if not raw:
        incompatibilities.append("empty_source_bytes")
    if source_schema_version != LEGACY_SOURCE_SCHEMA_VERSION:
        incompatibilities.append("unsupported_source_schema")

    normalized_json: str | None = None
    normalized_digest: str | None = None
    if not isinstance(decoded_snapshot, Mapping):
        incompatibilities.append("snapshot_not_mapping")
    else:
        try:
            state = ActivationLearnPairShadowState.from_initial_snapshot(decoded_snapshot)
            normalized_json = _canonical(state.snapshot, field="legacy_snapshot")
            normalized_digest = _sha_text(normalized_json)
        except (ShadowProjectionError, TypeError, ValueError):
            incompatibilities.append("invalid_bounded_snapshot")

    findings = tuple(sorted(set(incompatibilities)))
    if findings:
        normalized_json = None
        normalized_digest = None
    return LegacySidecarAssessment(
        source_label=source_label,
        source_sha256=_sha_bytes(raw),
        source_byte_count=len(raw),
        source_schema_version=source_schema_version,
        normalized_snapshot_json=normalized_json,
        normalized_snapshot_digest=normalized_digest,
        compatible=not findings,
        incompatibilities=findings,
    )


def build_migration_candidate(assessment: LegacySidecarAssessment) -> MigrationCandidate:
    """Create immutable comparison-only migration input from compatible evidence."""

    if not isinstance(assessment, LegacySidecarAssessment):
        raise M2CDualReadError("assessment must be LegacySidecarAssessment")
    if not assessment.compatible:
        raise LegacySidecarIncompatible(
            "legacy evidence is incompatible: " + ",".join(assessment.incompatibilities)
        )
    if assessment.normalized_snapshot_json is None or assessment.normalized_snapshot_digest is None:
        raise LegacySidecarIncompatible("compatible evidence is missing normalized snapshot")
    material = {
        "authority": COMPARISON_AUTHORITY,
        "legacy_snapshot_digest": assessment.normalized_snapshot_digest,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "schema_version": MIGRATION_CANDIDATE_SCHEMA_VERSION,
        "source_label": assessment.source_label,
        "source_sha256": assessment.source_sha256,
        "stream_id": ACTIVATION_LEARN_PAIR_TARGET.stream_id,
    }
    return MigrationCandidate(
        source_label=assessment.source_label,
        source_sha256=assessment.source_sha256,
        legacy_snapshot_json=assessment.normalized_snapshot_json,
        legacy_snapshot_digest=assessment.normalized_snapshot_digest,
        projection_schema_version=PROJECTION_SCHEMA_VERSION,
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        candidate_digest=_digest(material, field="migration_candidate"),
    )


def compare_dual_read(
    *,
    assessment: LegacySidecarAssessment,
    store: SQLiteShadowStore,
    initial_snapshot: Mapping[str, Any],
) -> DualReadReport:
    """Compare detached legacy evidence with M2-A replay without changing authority."""

    candidate = build_migration_candidate(assessment)
    if not isinstance(store, SQLiteShadowStore):
        raise M2CDualReadError("store must be SQLiteShadowStore")
    initial_state = ActivationLearnPairShadowState.from_initial_snapshot(initial_snapshot)

    before = store.integrity_check()
    incompatibilities: list[str] = []
    mismatches: tuple[str, ...] = ()
    shadow_snapshot_digest: str | None = None
    shadow_sequence: int | None = None
    shadow_event_count = before.event_count

    if not before.valid:
        incompatibilities.extend(f"shadow_integrity_before:{item}" for item in before.errors)
    else:
        try:
            events = store.events(stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id)
            shadow_event_count = len(events)
            state = replay_activation_learn_pair(initial_state, events)
            decoded = json.loads(candidate.legacy_snapshot_json)
            if not isinstance(decoded, dict):
                raise M2CDualReadError("migration candidate snapshot is not an object")
            equivalence = compare_activation_learn_pair_equivalence(state, decoded)
            mismatches = equivalence.mismatches
            shadow_snapshot_digest = equivalence.projected_digest
            shadow_sequence = state.sequence
        except (ShadowProjectionError, M2CDualReadError, TypeError, ValueError) as exc:
            incompatibilities.append(f"shadow_replay:{type(exc).__name__}")

    after = store.integrity_check()
    if not after.valid:
        incompatibilities.extend(f"shadow_integrity_after:{item}" for item in after.errors)
    writes_performed = before.report_digest != after.report_digest
    if writes_performed:
        incompatibilities.append("shadow_store_changed_during_comparison")

    exact_incompatibilities = tuple(sorted(set(incompatibilities)))
    material = {
        "comparison_authority": COMPARISON_AUTHORITY,
        "incompatibilities": list(exact_incompatibilities),
        "legacy_authority_retained": True,
        "legacy_snapshot_digest": candidate.legacy_snapshot_digest,
        "matches": not mismatches and not exact_incompatibilities,
        "mismatches": list(mismatches),
        "runtime_integrated": False,
        "schema_version": DUAL_READ_REPORT_SCHEMA_VERSION,
        "shadow_authority": SHADOW_AUTHORITY,
        "shadow_event_count": shadow_event_count,
        "shadow_integrity_after_digest": after.report_digest,
        "shadow_integrity_before_digest": before.report_digest,
        "shadow_sequence": shadow_sequence,
        "shadow_snapshot_digest": shadow_snapshot_digest,
        "source_label": candidate.source_label,
        "source_sha256": candidate.source_sha256,
        "writes_performed": writes_performed,
    }
    return DualReadReport(
        source_label=candidate.source_label,
        source_sha256=candidate.source_sha256,
        legacy_snapshot_digest=candidate.legacy_snapshot_digest,
        shadow_snapshot_digest=shadow_snapshot_digest,
        shadow_event_count=shadow_event_count,
        shadow_sequence=shadow_sequence,
        shadow_integrity_before_digest=before.report_digest,
        shadow_integrity_after_digest=after.report_digest,
        matches=not mismatches and not exact_incompatibilities,
        mismatches=mismatches,
        incompatibilities=exact_incompatibilities,
        writes_performed=writes_performed,
        report_digest=_digest(material, field="dual_read_report"),
    )
