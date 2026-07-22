"""Bounded M2-E persistence-cutover decision and evidence contracts.

This module does not perform cutover, discover a database, install a runtime hook,
or change defaults. It validates the accepted M2-D machine packet, produces an
immutable technical-candidate packet, validates a separately reviewed human
cutover decision, and constructs bounded post-cutover observation evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.event_kernel import canonical_json_object

M2E_REFERENCE_SCHEMA = "eve.m2-e-accepted-m2-d-reference.v1"
M2E_CANDIDATE_SCHEMA = "eve.m2-e-cutover-candidate.v1"
M2E_DECISION_SCHEMA = "eve.m2-e-human-cutover-decision.v1"
M2E_AUTHORIZATION_SCHEMA = "eve.m2-e-cutover-authorization.v1"
M2E_OBSERVATION_SCHEMA = "eve.m2-e-post-cutover-observation.v1"

CANDIDATE_AUTHORITY = "candidate_only"
EVENT_STORE_SHADOW_AUTHORITY = "shadow_only"
EVENT_STORE_BOUNDED_AUTHORITY = "authoritative_bounded_stream"
LEGACY_AUTHORITY = "authoritative"
LEGACY_EVIDENCE_MODE = "read_only_evidence"
HUMAN_REVIEW_REQUIRED = "required_not_performed"

ACCEPTED_M2_D_PR = 165
ACCEPTED_M2_D_HEAD = "ccf477d33b99c99302328dab1ff8e3292d9c4e91"
ACCEPTED_M2_D_WORKFLOW = 29916248120
ACCEPTED_M2_D_ARTIFACT = (
    "exact-head-validation-ccf477d33b99c99302328dab1ff8e3292d9c4e91"
)
ACCEPTED_M2_D_ARTIFACT_SHA256 = (
    "c669e31928cb329dc80ee170c46cbc078a14edb78f9eb9b0311997d180e4f004"
)
ACCEPTED_M2_D_PACKET_DIGEST = (
    "8064f61c7dfea68a263918b764eb357f0055deb73f7df5dae24fae2a00f7e3d2"
)
ACCEPTED_M2_D_MERGE_SHA = "c59095ccf75419e40107ec03fd20761ee946543d"

BOUNDED_STREAM = "shadow:legacy.activation.learn_pair"
BOUNDED_STATE_SCHEMA = "eve.shadow-projection.activation-learn-pair.v1"
REQUIRED_M2_D_SCENARIOS = (
    "snapshot_restore",
    "full_replay_equivalence",
    "corrupt_snapshot_fallback",
    "corrupt_event_fail_closed",
    "forced_termination",
    "rollback_rehearsal",
)
REQUIRED_CANDIDATE_CHECKS = (
    "accepted_m2_d_packet_digest",
    "accepted_m2_d_scenarios_complete",
    "accepted_m2_d_machine_passed",
    "accepted_m2_d_authority_boundary",
    "bounded_stream_exact",
    "external_human_decision_required",
    "post_cutover_window_defined",
    "rollback_remains_required",
    "no_automatic_cutover",
    "no_runtime_or_default_change",
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$")


class M2ECutoverError(ValueError):
    """Malformed, inconsistent, or out-of-scope M2-E evidence."""


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Mapping[str, Any], field: str) -> str:
    return _sha_text(_canon(value, field))


def _require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M2ECutoverError(f"{field} must be a lowercase SHA-256 digest")


def _require_commit(value: str, field: str) -> None:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise M2ECutoverError(f"{field} must be a lowercase 40-character commit SHA")


def _require_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise M2ECutoverError(f"{field} must be a canonical identifier")


def _require_positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M2ECutoverError(f"{field} must be a positive integer")


def _canonical_mapping(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise M2ECutoverError(f"{field} must be an object")
    try:
        return json.loads(_canon(value, field))
    except (TypeError, ValueError) as exc:
        raise M2ECutoverError(f"{field} is not canonical JSON data") from exc


@dataclass(frozen=True, slots=True)
class AcceptedM2DReference:
    pr: int = ACCEPTED_M2_D_PR
    head: str = ACCEPTED_M2_D_HEAD
    workflow: int = ACCEPTED_M2_D_WORKFLOW
    artifact: str = ACCEPTED_M2_D_ARTIFACT
    artifact_sha256: str = ACCEPTED_M2_D_ARTIFACT_SHA256
    packet_digest: str = ACCEPTED_M2_D_PACKET_DIGEST
    merge_sha: str = ACCEPTED_M2_D_MERGE_SHA
    schema_version: str = M2E_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        fixed = (
            (self.pr, ACCEPTED_M2_D_PR),
            (self.head, ACCEPTED_M2_D_HEAD),
            (self.workflow, ACCEPTED_M2_D_WORKFLOW),
            (self.artifact, ACCEPTED_M2_D_ARTIFACT),
            (self.artifact_sha256, ACCEPTED_M2_D_ARTIFACT_SHA256),
            (self.packet_digest, ACCEPTED_M2_D_PACKET_DIGEST),
            (self.merge_sha, ACCEPTED_M2_D_MERGE_SHA),
            (self.schema_version, M2E_REFERENCE_SCHEMA),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2ECutoverError("M2-D prerequisite reference is not the accepted pin")
        _require_commit(self.head, "head")
        _require_commit(self.merge_sha, "merge_sha")
        _require_positive_int(self.workflow, "workflow")
        _require_digest(self.artifact_sha256, "artifact_sha256")
        _require_digest(self.packet_digest, "packet_digest")

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "head": self.head,
            "merge_sha": self.merge_sha,
            "packet_digest": self.packet_digest,
            "pr": self.pr,
            "schema_version": self.schema_version,
            "workflow": self.workflow,
        }


@dataclass(frozen=True, slots=True)
class CutoverCandidatePacket:
    prerequisite: AcceptedM2DReference
    checks_json: str
    observations_json: str
    transition_hash: str
    packet_digest: str
    machine_passed: bool
    eligible_for_human_review: bool
    schema_version: str = M2E_CANDIDATE_SCHEMA
    human_review_status: str = HUMAN_REVIEW_REQUIRED
    human_accepted: bool = False
    authority: str = CANDIDATE_AUTHORITY
    event_store_authority: str = EVENT_STORE_SHADOW_AUTHORITY
    legacy_authority: str = LEGACY_AUTHORITY
    legacy_sidecars_read_only: bool = False
    cutover_authorized: bool = False
    authoritative_recovery: bool = False
    post_cutover_observation_complete: bool = False
    rollback_available: bool = True
    runtime_integrated: bool = False
    production_defaults_changed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prerequisite, AcceptedM2DReference):
            raise M2ECutoverError("candidate requires the accepted M2-D reference")
        try:
            checks = json.loads(self.checks_json)
            observations = json.loads(self.observations_json)
        except json.JSONDecodeError as exc:
            raise M2ECutoverError("candidate evidence JSON is malformed") from exc
        if tuple(sorted(checks)) != tuple(sorted(REQUIRED_CANDIDATE_CHECKS)):
            raise M2ECutoverError("candidate checks are incomplete")
        if not all(isinstance(value, bool) for value in checks.values()):
            raise M2ECutoverError("candidate checks must be boolean")
        if not isinstance(observations, dict):
            raise M2ECutoverError("candidate observations must be an object")
        if _canon(checks, "m2_e_candidate_checks") != self.checks_json:
            raise M2ECutoverError("candidate checks must be canonical JSON")
        if _canon(observations, "m2_e_candidate_observations") != self.observations_json:
            raise M2ECutoverError("candidate observations must be canonical JSON")
        computed = all(checks.values())
        if self.machine_passed != computed or self.eligible_for_human_review != computed:
            raise M2ECutoverError("candidate status disagrees with raw checks")
        fixed = (
            (self.schema_version, M2E_CANDIDATE_SCHEMA),
            (self.human_review_status, HUMAN_REVIEW_REQUIRED),
            (self.human_accepted, False),
            (self.authority, CANDIDATE_AUTHORITY),
            (self.event_store_authority, EVENT_STORE_SHADOW_AUTHORITY),
            (self.legacy_authority, LEGACY_AUTHORITY),
            (self.legacy_sidecars_read_only, False),
            (self.cutover_authorized, False),
            (self.authoritative_recovery, False),
            (self.post_cutover_observation_complete, False),
            (self.rollback_available, True),
            (self.runtime_integrated, False),
            (self.production_defaults_changed, False),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2ECutoverError("technical candidate cannot self-promote")
        _require_digest(self.transition_hash, "transition_hash")
        _require_digest(self.packet_digest, "packet_digest")
        if _digest(self.transition_material, "m2_e_candidate_transition") != self.transition_hash:
            raise M2ECutoverError("candidate transition hash mismatch")
        if _digest(self.packet_material, "m2_e_candidate_packet") != self.packet_digest:
            raise M2ECutoverError("candidate packet digest mismatch")

    @property
    def checks(self) -> dict[str, bool]:
        return json.loads(self.checks_json)

    @property
    def observations(self) -> dict[str, Any]:
        return json.loads(self.observations_json)

    @property
    def transition_material(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "observations": self.observations,
            "prerequisite": self.prerequisite.canonical_record,
        }

    @property
    def packet_material(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "authoritative_recovery": self.authoritative_recovery,
            "cutover_authorized": self.cutover_authorized,
            "eligible_for_human_review": self.eligible_for_human_review,
            "event_store_authority": self.event_store_authority,
            "human_accepted": self.human_accepted,
            "human_review_status": self.human_review_status,
            "legacy_authority": self.legacy_authority,
            "legacy_sidecars_read_only": self.legacy_sidecars_read_only,
            "machine_passed": self.machine_passed,
            "post_cutover_observation_complete": self.post_cutover_observation_complete,
            "production_defaults_changed": self.production_defaults_changed,
            "rollback_available": self.rollback_available,
            "runtime_integrated": self.runtime_integrated,
            "schema_version": self.schema_version,
            "transition_hash": self.transition_hash,
            **self.transition_material,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.packet_material, "packet_digest": self.packet_digest}

    @classmethod
    def create(
        cls,
        *,
        prerequisite: AcceptedM2DReference,
        checks: Mapping[str, bool],
        observations: Mapping[str, Any],
    ) -> "CutoverCandidatePacket":
        checks_json = _canon(checks, "m2_e_candidate_checks")
        observations_json = _canon(observations, "m2_e_candidate_observations")
        transition_material = {
            "checks": json.loads(checks_json),
            "observations": json.loads(observations_json),
            "prerequisite": prerequisite.canonical_record,
        }
        transition_hash = _digest(transition_material, "m2_e_candidate_transition")
        machine_passed = all(checks.values())
        packet_material = {
            "authority": CANDIDATE_AUTHORITY,
            "authoritative_recovery": False,
            "cutover_authorized": False,
            "eligible_for_human_review": machine_passed,
            "event_store_authority": EVENT_STORE_SHADOW_AUTHORITY,
            "human_accepted": False,
            "human_review_status": HUMAN_REVIEW_REQUIRED,
            "legacy_authority": LEGACY_AUTHORITY,
            "legacy_sidecars_read_only": False,
            "machine_passed": machine_passed,
            "post_cutover_observation_complete": False,
            "production_defaults_changed": False,
            "rollback_available": True,
            "runtime_integrated": False,
            "schema_version": M2E_CANDIDATE_SCHEMA,
            "transition_hash": transition_hash,
            **transition_material,
        }
        return cls(
            prerequisite=prerequisite,
            checks_json=checks_json,
            observations_json=observations_json,
            transition_hash=transition_hash,
            packet_digest=_digest(packet_material, "m2_e_candidate_packet"),
            machine_passed=machine_passed,
            eligible_for_human_review=machine_passed,
        )


@dataclass(frozen=True, slots=True)
class HumanCutoverDecision:
    decision_id: str
    candidate_head: str
    workflow: int
    artifact: str
    artifact_sha256: str
    candidate_packet_digest: str
    reviewer: str
    decision_status: str
    human_accepted: bool
    cutover_authorized: bool
    event_store_authority: str
    legacy_sidecars_mode: str
    rollback_required: bool
    decision_digest: str
    schema_version: str = M2E_DECISION_SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.decision_id, "decision_id")
        _require_identifier(self.reviewer, "reviewer")
        _require_commit(self.candidate_head, "candidate_head")
        _require_positive_int(self.workflow, "workflow")
        _require_digest(self.artifact_sha256, "artifact_sha256")
        _require_digest(self.candidate_packet_digest, "candidate_packet_digest")
        _require_digest(self.decision_digest, "decision_digest")
        if self.schema_version != M2E_DECISION_SCHEMA:
            raise M2ECutoverError("unknown human decision schema")
        if self.artifact != f"exact-head-validation-{self.candidate_head}":
            raise M2ECutoverError("decision artifact does not match candidate head")
        accepted = self.decision_status == "accepted"
        rejected = self.decision_status == "rejected"
        if not accepted and not rejected:
            raise M2ECutoverError("decision status must be accepted or rejected")
        expected = (
            self.human_accepted,
            self.cutover_authorized,
            self.event_store_authority,
            self.legacy_sidecars_mode,
            self.rollback_required,
        )
        if accepted:
            required = (
                True,
                True,
                EVENT_STORE_BOUNDED_AUTHORITY,
                LEGACY_EVIDENCE_MODE,
                True,
            )
        else:
            required = (
                False,
                False,
                EVENT_STORE_SHADOW_AUTHORITY,
                LEGACY_AUTHORITY,
                True,
            )
        if expected != required:
            raise M2ECutoverError("human decision fields disagree with decision status")
        if _digest(self.decision_material, "m2_e_human_decision") != self.decision_digest:
            raise M2ECutoverError("human decision digest mismatch")

    @property
    def decision_material(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "candidate_head": self.candidate_head,
            "candidate_packet_digest": self.candidate_packet_digest,
            "cutover_authorized": self.cutover_authorized,
            "decision_id": self.decision_id,
            "decision_status": self.decision_status,
            "event_store_authority": self.event_store_authority,
            "human_accepted": self.human_accepted,
            "legacy_sidecars_mode": self.legacy_sidecars_mode,
            "reviewer": self.reviewer,
            "rollback_required": self.rollback_required,
            "schema_version": self.schema_version,
            "workflow": self.workflow,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.decision_material, "decision_digest": self.decision_digest}

    @classmethod
    def create(
        cls,
        *,
        decision_id: str,
        candidate_head: str,
        workflow: int,
        artifact_sha256: str,
        candidate_packet_digest: str,
        reviewer: str,
        accepted: bool,
    ) -> "HumanCutoverDecision":
        status = "accepted" if accepted else "rejected"
        material = {
            "artifact": f"exact-head-validation-{candidate_head}",
            "artifact_sha256": artifact_sha256,
            "candidate_head": candidate_head,
            "candidate_packet_digest": candidate_packet_digest,
            "cutover_authorized": accepted,
            "decision_id": decision_id,
            "decision_status": status,
            "event_store_authority": (
                EVENT_STORE_BOUNDED_AUTHORITY if accepted else EVENT_STORE_SHADOW_AUTHORITY
            ),
            "human_accepted": accepted,
            "legacy_sidecars_mode": LEGACY_EVIDENCE_MODE if accepted else LEGACY_AUTHORITY,
            "reviewer": reviewer,
            "rollback_required": True,
            "schema_version": M2E_DECISION_SCHEMA,
            "workflow": workflow,
        }
        return cls(
            decision_id=decision_id,
            candidate_head=candidate_head,
            workflow=workflow,
            artifact=material["artifact"],
            artifact_sha256=artifact_sha256,
            candidate_packet_digest=candidate_packet_digest,
            reviewer=reviewer,
            decision_status=status,
            human_accepted=accepted,
            cutover_authorized=accepted,
            event_store_authority=material["event_store_authority"],
            legacy_sidecars_mode=material["legacy_sidecars_mode"],
            rollback_required=True,
            decision_digest=_digest(material, "m2_e_human_decision"),
        )


@dataclass(frozen=True, slots=True)
class CutoverAuthorization:
    candidate_packet_digest: str
    human_decision_digest: str
    stream_id: str
    state_schema: str
    authorization_digest: str
    schema_version: str = M2E_AUTHORIZATION_SCHEMA
    event_store_authority: str = EVENT_STORE_BOUNDED_AUTHORITY
    legacy_sidecars_mode: str = LEGACY_EVIDENCE_MODE
    authoritative_recovery: bool = True
    rollback_available: bool = True
    production_defaults_changed: bool = False
    runtime_integrated: bool = False

    def __post_init__(self) -> None:
        for field in (
            "candidate_packet_digest",
            "human_decision_digest",
            "authorization_digest",
        ):
            _require_digest(getattr(self, field), field)
        fixed = (
            (self.schema_version, M2E_AUTHORIZATION_SCHEMA),
            (self.stream_id, BOUNDED_STREAM),
            (self.state_schema, BOUNDED_STATE_SCHEMA),
            (self.event_store_authority, EVENT_STORE_BOUNDED_AUTHORITY),
            (self.legacy_sidecars_mode, LEGACY_EVIDENCE_MODE),
            (self.authoritative_recovery, True),
            (self.rollback_available, True),
            (self.production_defaults_changed, False),
            (self.runtime_integrated, False),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2ECutoverError("authorization escaped the bounded cutover scope")
        if _digest(self.authorization_material, "m2_e_cutover_authorization") != self.authorization_digest:
            raise M2ECutoverError("authorization digest mismatch")

    @property
    def authorization_material(self) -> dict[str, Any]:
        return {
            "authoritative_recovery": self.authoritative_recovery,
            "candidate_packet_digest": self.candidate_packet_digest,
            "event_store_authority": self.event_store_authority,
            "human_decision_digest": self.human_decision_digest,
            "legacy_sidecars_mode": self.legacy_sidecars_mode,
            "production_defaults_changed": self.production_defaults_changed,
            "rollback_available": self.rollback_available,
            "runtime_integrated": self.runtime_integrated,
            "schema_version": self.schema_version,
            "state_schema": self.state_schema,
            "stream_id": self.stream_id,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.authorization_material, "authorization_digest": self.authorization_digest}


@dataclass(frozen=True, slots=True)
class PostCutoverObservationEvidence:
    authorization: CutoverAuthorization
    event_count: int
    before_state_json: str
    authoritative_state_json: str
    replay_state_json: str
    rollback_state_json: str
    checks_json: str
    transition_hash: str
    evidence_digest: str
    schema_version: str = M2E_OBSERVATION_SCHEMA
    stream_id: str = BOUNDED_STREAM
    state_schema: str = BOUNDED_STATE_SCHEMA
    legacy_sidecars_mode: str = LEGACY_EVIDENCE_MODE
    event_store_authority: str = EVENT_STORE_BOUNDED_AUTHORITY
    rollback_available: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.authorization, CutoverAuthorization):
            raise M2ECutoverError("post-cutover evidence requires authorization")
        _require_positive_int(self.event_count, "event_count")
        decoded = []
        for field in (
            "before_state_json",
            "authoritative_state_json",
            "replay_state_json",
            "rollback_state_json",
        ):
            try:
                value = json.loads(getattr(self, field))
            except json.JSONDecodeError as exc:
                raise M2ECutoverError(f"{field} is malformed") from exc
            if not isinstance(value, dict):
                raise M2ECutoverError(f"{field} must encode an object")
            if _canon(value, field) != getattr(self, field):
                raise M2ECutoverError(f"{field} must be canonical JSON")
            decoded.append(value)
        try:
            checks = json.loads(self.checks_json)
        except json.JSONDecodeError as exc:
            raise M2ECutoverError("post-cutover checks are malformed") from exc
        required = {
            "authoritative_state_changed",
            "event_store_replay_equivalent",
            "legacy_sidecars_read_only",
            "rollback_restores_pre_cutover_state",
            "rollback_available",
        }
        if set(checks) != required or not all(isinstance(value, bool) for value in checks.values()):
            raise M2ECutoverError("post-cutover checks are incomplete")
        if _canon(checks, "m2_e_post_cutover_checks") != self.checks_json:
            raise M2ECutoverError("post-cutover checks must be canonical JSON")
        if not all(checks.values()):
            raise M2ECutoverError("post-cutover observation did not pass")
        fixed = (
            (self.schema_version, M2E_OBSERVATION_SCHEMA),
            (self.stream_id, BOUNDED_STREAM),
            (self.state_schema, BOUNDED_STATE_SCHEMA),
            (self.legacy_sidecars_mode, LEGACY_EVIDENCE_MODE),
            (self.event_store_authority, EVENT_STORE_BOUNDED_AUTHORITY),
            (self.rollback_available, True),
        )
        if any(actual != expected for actual, expected in fixed):
            raise M2ECutoverError("post-cutover evidence escaped bounded scope")
        _require_digest(self.transition_hash, "transition_hash")
        _require_digest(self.evidence_digest, "evidence_digest")
        if _digest(self.transition_material, "m2_e_post_cutover_transition") != self.transition_hash:
            raise M2ECutoverError("post-cutover transition hash mismatch")
        if _digest(self.evidence_material, "m2_e_post_cutover_evidence") != self.evidence_digest:
            raise M2ECutoverError("post-cutover evidence digest mismatch")

    @property
    def checks(self) -> dict[str, bool]:
        return json.loads(self.checks_json)

    @property
    def transition_material(self) -> dict[str, Any]:
        return {
            "authoritative_state": json.loads(self.authoritative_state_json),
            "before_state": json.loads(self.before_state_json),
            "checks": self.checks,
            "event_count": self.event_count,
            "replay_state": json.loads(self.replay_state_json),
            "rollback_state": json.loads(self.rollback_state_json),
        }

    @property
    def evidence_material(self) -> dict[str, Any]:
        return {
            "authorization": self.authorization.canonical_record,
            "event_store_authority": self.event_store_authority,
            "legacy_sidecars_mode": self.legacy_sidecars_mode,
            "rollback_available": self.rollback_available,
            "schema_version": self.schema_version,
            "state_schema": self.state_schema,
            "stream_id": self.stream_id,
            "transition_hash": self.transition_hash,
            **self.transition_material,
        }

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.evidence_material, "evidence_digest": self.evidence_digest}


def evaluate_cutover_candidate(m2_d_packet: Mapping[str, Any]) -> CutoverCandidatePacket:
    packet = _canonical_mapping(m2_d_packet, "m2_d_packet")
    packet_digest = packet.get("packet_digest")
    scenarios = packet.get("scenarios")
    window = packet.get("window")
    scenario_ids = tuple(
        value.get("scenario_id") for value in scenarios
    ) if isinstance(scenarios, list) and all(isinstance(value, dict) for value in scenarios) else ()
    record_without_digest = dict(packet)
    record_without_digest.pop("packet_digest", None)
    internally_valid_digest = (
        isinstance(packet_digest, str)
        and _DIGEST.fullmatch(packet_digest) is not None
        and _digest(record_without_digest, "m2_d_rehearsal_packet") == packet_digest
    )
    authority_boundary = (
        packet.get("authority") == "rehearsal_only"
        and packet.get("shadow_authority") == EVENT_STORE_SHADOW_AUTHORITY
        and packet.get("legacy_authority_retained") is True
        and packet.get("runtime_integrated") is False
        and packet.get("production_dual_read") is False
        and packet.get("authoritative_recovery") is False
        and packet.get("cutover_authorized") is False
        and packet.get("human_accepted") is False
    )
    bounded_stream = (
        isinstance(window, dict)
        and window.get("stream_id") == BOUNDED_STREAM
        and window.get("state_schema_version") == BOUNDED_STATE_SCHEMA
    )
    checks = {
        "accepted_m2_d_packet_digest": (
            internally_valid_digest and packet_digest == ACCEPTED_M2_D_PACKET_DIGEST
        ),
        "accepted_m2_d_scenarios_complete": (
            scenario_ids == REQUIRED_M2_D_SCENARIOS
            and packet.get("passed_count") == len(REQUIRED_M2_D_SCENARIOS)
            and packet.get("failed_count") == 0
        ),
        "accepted_m2_d_machine_passed": (
            packet.get("machine_passed") is True
            and packet.get("eligible_for_human_review") is True
        ),
        "accepted_m2_d_authority_boundary": authority_boundary,
        "bounded_stream_exact": bounded_stream,
        "external_human_decision_required": True,
        "post_cutover_window_defined": True,
        "rollback_remains_required": True,
        "no_automatic_cutover": True,
        "no_runtime_or_default_change": True,
    }
    observations = {
        "accepted_m2_d_reference": AcceptedM2DReference().canonical_record,
        "bounded_state_schema": BOUNDED_STATE_SCHEMA,
        "bounded_stream": BOUNDED_STREAM,
        "candidate_effects": {
            "authoritative_recovery": False,
            "cutover_authorized": False,
            "event_store_authority": EVENT_STORE_SHADOW_AUTHORITY,
            "legacy_authority": LEGACY_AUTHORITY,
            "legacy_sidecars_read_only": False,
            "production_defaults_changed": False,
            "runtime_integrated": False,
        },
        "m2_d_packet_digest": packet_digest,
        "m2_d_scenario_ids": list(scenario_ids),
        "post_cutover_requirements": [
            "separate immutable human decision pinned to exact M2-E head/workflow/artifact SHA-256",
            "event-store replay equals bounded authoritative state",
            "legacy sidecars become read-only evidence only after accepted decision",
            "rollback restores the pre-cutover bounded state",
            "post-cutover observation evidence is independently recalculable",
        ],
    }
    return CutoverCandidatePacket.create(
        prerequisite=AcceptedM2DReference(),
        checks=checks,
        observations=observations,
    )


def authorize_cutover(
    candidate: CutoverCandidatePacket,
    decision: HumanCutoverDecision,
) -> CutoverAuthorization:
    if not isinstance(candidate, CutoverCandidatePacket) or not candidate.machine_passed:
        raise M2ECutoverError("cutover candidate is not technically eligible")
    if not isinstance(decision, HumanCutoverDecision):
        raise M2ECutoverError("cutover requires an immutable human decision")
    if decision.candidate_packet_digest != candidate.packet_digest:
        raise M2ECutoverError("human decision does not pin this candidate packet")
    if not decision.human_accepted or not decision.cutover_authorized:
        raise M2ECutoverError("human decision did not authorize cutover")
    material = {
        "authoritative_recovery": True,
        "candidate_packet_digest": candidate.packet_digest,
        "event_store_authority": EVENT_STORE_BOUNDED_AUTHORITY,
        "human_decision_digest": decision.decision_digest,
        "legacy_sidecars_mode": LEGACY_EVIDENCE_MODE,
        "production_defaults_changed": False,
        "rollback_available": True,
        "runtime_integrated": False,
        "schema_version": M2E_AUTHORIZATION_SCHEMA,
        "state_schema": BOUNDED_STATE_SCHEMA,
        "stream_id": BOUNDED_STREAM,
    }
    return CutoverAuthorization(
        candidate_packet_digest=candidate.packet_digest,
        human_decision_digest=decision.decision_digest,
        stream_id=BOUNDED_STREAM,
        state_schema=BOUNDED_STATE_SCHEMA,
        authorization_digest=_digest(material, "m2_e_cutover_authorization"),
    )


def record_post_cutover_observation(
    *,
    authorization: CutoverAuthorization,
    event_count: int,
    before_state: Mapping[str, Any],
    authoritative_state: Mapping[str, Any],
    replay_state: Mapping[str, Any],
    rollback_state: Mapping[str, Any],
) -> PostCutoverObservationEvidence:
    before = _canonical_mapping(before_state, "before_state")
    authoritative = _canonical_mapping(authoritative_state, "authoritative_state")
    replay = _canonical_mapping(replay_state, "replay_state")
    rollback = _canonical_mapping(rollback_state, "rollback_state")
    checks = {
        "authoritative_state_changed": authoritative != before,
        "event_store_replay_equivalent": replay == authoritative,
        "legacy_sidecars_read_only": authorization.legacy_sidecars_mode == LEGACY_EVIDENCE_MODE,
        "rollback_restores_pre_cutover_state": rollback == before,
        "rollback_available": authorization.rollback_available,
    }
    checks_json = _canon(checks, "m2_e_post_cutover_checks")
    transition_material = {
        "authoritative_state": authoritative,
        "before_state": before,
        "checks": json.loads(checks_json),
        "event_count": event_count,
        "replay_state": replay,
        "rollback_state": rollback,
    }
    transition_hash = _digest(transition_material, "m2_e_post_cutover_transition")
    evidence_material = {
        "authorization": authorization.canonical_record,
        "event_store_authority": EVENT_STORE_BOUNDED_AUTHORITY,
        "legacy_sidecars_mode": LEGACY_EVIDENCE_MODE,
        "rollback_available": True,
        "schema_version": M2E_OBSERVATION_SCHEMA,
        "state_schema": BOUNDED_STATE_SCHEMA,
        "stream_id": BOUNDED_STREAM,
        "transition_hash": transition_hash,
        **transition_material,
    }
    return PostCutoverObservationEvidence(
        authorization=authorization,
        event_count=event_count,
        before_state_json=_canon(before, "before_state"),
        authoritative_state_json=_canon(authoritative, "authoritative_state"),
        replay_state_json=_canon(replay, "replay_state"),
        rollback_state_json=_canon(rollback, "rollback_state"),
        checks_json=checks_json,
        transition_hash=transition_hash,
        evidence_digest=_digest(evidence_material, "m2_e_post_cutover_evidence"),
    )


def scenario_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact scenario order from a candidate observation record."""
    observations = record.get("observations")
    if not isinstance(observations, Mapping):
        raise M2ECutoverError("record has no observations object")
    values = observations.get("m2_d_scenario_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise M2ECutoverError("record has no scenario sequence")
    if not all(isinstance(value, str) for value in values):
        raise M2ECutoverError("scenario sequence is malformed")
    return tuple(values)
