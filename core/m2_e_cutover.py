"""Bounded M2-E cutover-decision and post-cutover evidence contracts.

Import and construction perform no I/O. The module validates the accepted M2-D
packet, binds a technical candidate to an exact head/workflow, validates a
separate human decision, and records recalculable bounded observations. It does
not discover a database, install a runtime hook, change defaults, or create
human acceptance.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.event_kernel import canonical_json_object
from core.m2_c_migration import StateEvidence
from core.sqlite_shadow_store import IntegrityReport

M2E_REFERENCE_SCHEMA = "eve.m2-e-accepted-m2-d-reference.v1"
M2E_CANDIDATE_SCHEMA = "eve.m2-e-cutover-candidate.v1"
M2E_DECISION_SCHEMA = "eve.m2-e-human-cutover-decision.v1"
M2E_AUTHORIZATION_SCHEMA = "eve.m2-e-cutover-authorization.v1"
M2E_STORE_OBSERVATION_SCHEMA = "eve.m2-e-store-observation.v1"
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
ACCEPTED_M2_D_ARTIFACT = f"exact-head-validation-{ACCEPTED_M2_D_HEAD}"
ACCEPTED_M2_D_ARTIFACT_SHA256 = "c669e31928cb329dc80ee170c46cbc078a14edb78f9eb9b0311997d180e4f004"
ACCEPTED_M2_D_PACKET_DIGEST = "8064f61c7dfea68a263918b764eb357f0055deb73f7df5dae24fae2a00f7e3d2"
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
REQUIRED_CANDIDATE_CHECKS = {
    "accepted_m2_d_packet_digest",
    "accepted_m2_d_scenarios_complete",
    "accepted_m2_d_machine_passed",
    "accepted_m2_d_authority_boundary",
    "bounded_stream_exact",
    "exact_candidate_pin_present",
    "external_human_decision_required",
    "post_cutover_window_defined",
    "rollback_remains_required",
    "no_automatic_cutover",
    "no_runtime_or_default_change",
}
REQUIRED_OBSERVATION_CHECKS = {
    "store_integrity_valid_before",
    "store_integrity_valid_after",
    "event_count_advanced",
    "authoritative_state_changed",
    "event_store_replay_equivalent",
    "legacy_sidecars_read_only",
    "rollback_restores_pre_cutover_state",
    "rollback_available",
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$")


class M2ECutoverError(ValueError):
    pass


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canon(value, field).encode("utf-8")).hexdigest()


def _require_digest(value: Any, field: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise M2ECutoverError(f"{field} must be a lowercase SHA-256 digest")


def _require_commit(value: Any, field: str) -> None:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise M2ECutoverError(f"{field} must be a lowercase 40-character commit SHA")


def _require_positive_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M2ECutoverError(f"{field} must be a positive integer")


def _require_identifier(value: Any, field: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise M2ECutoverError(f"{field} must be a canonical identifier")


def _decode(text: str, field: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise M2ECutoverError(f"{field} is malformed") from exc
    if not isinstance(value, dict) or _canon(value, field) != text:
        raise M2ECutoverError(f"{field} must be a canonical object")
    return value


def _accepted_reference() -> dict[str, Any]:
    return {
        "artifact": ACCEPTED_M2_D_ARTIFACT,
        "artifact_sha256": ACCEPTED_M2_D_ARTIFACT_SHA256,
        "head": ACCEPTED_M2_D_HEAD,
        "merge_sha": ACCEPTED_M2_D_MERGE_SHA,
        "packet_digest": ACCEPTED_M2_D_PACKET_DIGEST,
        "pr": ACCEPTED_M2_D_PR,
        "schema_version": M2E_REFERENCE_SCHEMA,
        "workflow": ACCEPTED_M2_D_WORKFLOW,
    }


def _state_record(value: StateEvidence) -> dict[str, Any]:
    if not isinstance(value, StateEvidence):
        raise M2ECutoverError("expected StateEvidence")
    return {
        "manifest_digest": value.manifest_digest,
        "manifest_json": value.manifest_json,
        "schema_version": value.schema_version,
        "snapshot_digest": value.snapshot_digest,
        "snapshot_json": value.snapshot_json,
    }


@dataclass(frozen=True, slots=True)
class CutoverCandidatePacket:
    record_json: str
    packet_digest: str

    def __post_init__(self) -> None:
        record = _decode(self.record_json, "m2_e_candidate_record")
        _require_digest(self.packet_digest, "packet_digest")
        if _digest(record, "m2_e_candidate_packet") != self.packet_digest:
            raise M2ECutoverError("candidate packet digest mismatch")
        _require_commit(record.get("candidate_head"), "candidate_head")
        _require_positive_int(record.get("workflow"), "workflow")
        if record.get("artifact") != f"exact-head-validation-{record['candidate_head']}":
            raise M2ECutoverError("candidate artifact does not match candidate head")
        checks = record.get("checks")
        if not isinstance(checks, dict) or set(checks) != REQUIRED_CANDIDATE_CHECKS:
            raise M2ECutoverError("candidate checks are incomplete")
        if not all(isinstance(value, bool) for value in checks.values()):
            raise M2ECutoverError("candidate checks must be boolean")
        machine_passed = all(checks.values())
        immutable = (
            record.get("schema_version") == M2E_CANDIDATE_SCHEMA
            and record.get("prerequisite") == _accepted_reference()
            and record.get("machine_passed") is machine_passed
            and record.get("eligible_for_human_review") is machine_passed
            and record.get("human_review_status") == HUMAN_REVIEW_REQUIRED
            and record.get("human_accepted") is False
            and record.get("authority") == CANDIDATE_AUTHORITY
            and record.get("event_store_authority") == EVENT_STORE_SHADOW_AUTHORITY
            and record.get("legacy_authority") == LEGACY_AUTHORITY
            and record.get("legacy_sidecars_read_only") is False
            and record.get("cutover_authorized") is False
            and record.get("authoritative_recovery") is False
            and record.get("post_cutover_observation_complete") is False
            and record.get("rollback_available") is True
            and record.get("runtime_integrated") is False
            and record.get("production_defaults_changed") is False
        )
        if not immutable:
            raise M2ECutoverError("technical candidate cannot self-promote")

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.record, "packet_digest": self.packet_digest}

    def __getattr__(self, name: str) -> Any:
        if name in self.record:
            return self.record[name]
        raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class HumanCutoverDecision:
    record_json: str
    decision_digest: str

    def __post_init__(self) -> None:
        record = _decode(self.record_json, "m2_e_human_decision")
        _require_digest(self.decision_digest, "decision_digest")
        if _digest(record, "m2_e_human_decision") != self.decision_digest:
            raise M2ECutoverError("human decision digest mismatch")
        _require_identifier(record.get("decision_id"), "decision_id")
        _require_identifier(record.get("reviewer"), "reviewer")
        _require_commit(record.get("candidate_head"), "candidate_head")
        _require_positive_int(record.get("workflow"), "workflow")
        _require_digest(record.get("artifact_sha256"), "artifact_sha256")
        _require_digest(record.get("candidate_packet_digest"), "candidate_packet_digest")
        if record.get("artifact") != f"exact-head-validation-{record['candidate_head']}":
            raise M2ECutoverError("decision artifact does not match candidate head")
        accepted = record.get("decision_status") == "accepted"
        rejected = record.get("decision_status") == "rejected"
        if not accepted and not rejected:
            raise M2ECutoverError("decision status must be accepted or rejected")
        expected = (
            True,
            True,
            EVENT_STORE_BOUNDED_AUTHORITY,
            LEGACY_EVIDENCE_MODE,
        ) if accepted else (
            False,
            False,
            EVENT_STORE_SHADOW_AUTHORITY,
            LEGACY_AUTHORITY,
        )
        actual = (
            record.get("human_accepted"),
            record.get("cutover_authorized"),
            record.get("event_store_authority"),
            record.get("legacy_sidecars_mode"),
        )
        if (
            record.get("schema_version") != M2E_DECISION_SCHEMA
            or record.get("rollback_required") is not True
            or actual != expected
        ):
            raise M2ECutoverError("human decision fields disagree with decision status")

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.record, "decision_digest": self.decision_digest}

    def __getattr__(self, name: str) -> Any:
        if name in self.record:
            return self.record[name]
        raise AttributeError(name)

    @classmethod
    def create(
        cls,
        *,
        candidate: CutoverCandidatePacket,
        decision_id: str,
        artifact_sha256: str,
        reviewer: str,
        accepted: bool,
    ) -> "HumanCutoverDecision":
        if not isinstance(candidate, CutoverCandidatePacket):
            raise M2ECutoverError("decision requires a cutover candidate")
        record = {
            "artifact": candidate.artifact,
            "artifact_sha256": artifact_sha256,
            "candidate_head": candidate.candidate_head,
            "candidate_packet_digest": candidate.packet_digest,
            "cutover_authorized": accepted,
            "decision_id": decision_id,
            "decision_status": "accepted" if accepted else "rejected",
            "event_store_authority": (
                EVENT_STORE_BOUNDED_AUTHORITY if accepted else EVENT_STORE_SHADOW_AUTHORITY
            ),
            "human_accepted": accepted,
            "legacy_sidecars_mode": LEGACY_EVIDENCE_MODE if accepted else LEGACY_AUTHORITY,
            "reviewer": reviewer,
            "rollback_required": True,
            "schema_version": M2E_DECISION_SCHEMA,
            "workflow": candidate.workflow,
        }
        text = _canon(record, "m2_e_human_decision")
        return cls(text, _digest(record, "m2_e_human_decision"))


@dataclass(frozen=True, slots=True)
class CutoverAuthorization:
    record_json: str
    authorization_digest: str

    def __post_init__(self) -> None:
        record = _decode(self.record_json, "m2_e_cutover_authorization")
        _require_digest(self.authorization_digest, "authorization_digest")
        if _digest(record, "m2_e_cutover_authorization") != self.authorization_digest:
            raise M2ECutoverError("authorization digest mismatch")
        _require_commit(record.get("candidate_head"), "candidate_head")
        _require_positive_int(record.get("workflow"), "workflow")
        _require_digest(record.get("artifact_sha256"), "artifact_sha256")
        if not (
            record.get("schema_version") == M2E_AUTHORIZATION_SCHEMA
            and record.get("artifact") == f"exact-head-validation-{record['candidate_head']}"
            and record.get("stream_id") == BOUNDED_STREAM
            and record.get("state_schema") == BOUNDED_STATE_SCHEMA
            and record.get("event_store_authority") == EVENT_STORE_BOUNDED_AUTHORITY
            and record.get("legacy_sidecars_mode") == LEGACY_EVIDENCE_MODE
            and record.get("authoritative_recovery") is True
            and record.get("rollback_available") is True
            and record.get("production_defaults_changed") is False
            and record.get("runtime_integrated") is False
        ):
            raise M2ECutoverError("authorization escaped the bounded cutover scope")

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.record, "authorization_digest": self.authorization_digest}

    def __getattr__(self, name: str) -> Any:
        if name in self.record:
            return self.record[name]
        raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class StoreObservation:
    record_json: str
    observation_digest: str

    def __post_init__(self) -> None:
        record = _decode(self.record_json, "m2_e_store_observation")
        _require_digest(self.observation_digest, "observation_digest")
        if _digest(record, "m2_e_store_observation") != self.observation_digest:
            raise M2ECutoverError("store observation digest mismatch")
        if (
            record.get("schema_version") != M2E_STORE_OBSERVATION_SCHEMA
            or not isinstance(record.get("valid"), bool)
            or not isinstance(record.get("errors"), list)
            or record.get("valid") != (not record.get("errors"))
        ):
            raise M2ECutoverError("store observation is malformed")
        for field in ("event_count", "snapshot_count"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise M2ECutoverError(f"{field} must be non-negative")
        _require_digest(record.get("chain_head_digest"), "chain_head_digest")
        _require_digest(record.get("report_digest"), "report_digest")

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.record, "observation_digest": self.observation_digest}

    def __getattr__(self, name: str) -> Any:
        if name in self.record:
            return self.record[name]
        raise AttributeError(name)

    @classmethod
    def from_integrity_report(cls, report: IntegrityReport) -> "StoreObservation":
        if not isinstance(report, IntegrityReport):
            raise M2ECutoverError("store observation requires an IntegrityReport")
        record = {
            "chain_head_digest": report.chain_head_digest,
            "errors": list(report.errors),
            "event_count": report.event_count,
            "report_digest": report.report_digest,
            "schema_version": M2E_STORE_OBSERVATION_SCHEMA,
            "snapshot_count": report.snapshot_count,
            "valid": report.valid,
        }
        return cls(
            _canon(record, "m2_e_store_observation"),
            _digest(record, "m2_e_store_observation"),
        )


@dataclass(frozen=True, slots=True)
class PostCutoverObservationEvidence:
    record_json: str
    evidence_digest: str

    def __post_init__(self) -> None:
        record = _decode(self.record_json, "m2_e_post_cutover_evidence")
        _require_digest(self.evidence_digest, "evidence_digest")
        if _digest(record, "m2_e_post_cutover_evidence") != self.evidence_digest:
            raise M2ECutoverError("post-cutover evidence digest mismatch")
        _require_identifier(record.get("observation_id"), "observation_id")
        checks = record.get("checks")
        if (
            record.get("schema_version") != M2E_OBSERVATION_SCHEMA
            or record.get("stream_id") != BOUNDED_STREAM
            or record.get("state_schema") != BOUNDED_STATE_SCHEMA
            or record.get("event_store_authority") != EVENT_STORE_BOUNDED_AUTHORITY
            or record.get("legacy_sidecars_mode") != LEGACY_EVIDENCE_MODE
            or record.get("rollback_available") is not True
            or not isinstance(checks, dict)
            or set(checks) != REQUIRED_OBSERVATION_CHECKS
            or not all(checks.values())
        ):
            raise M2ECutoverError("post-cutover observation did not pass")
        delta = record.get("event_count_delta")
        if isinstance(delta, bool) or not isinstance(delta, int) or delta <= 0:
            raise M2ECutoverError("event_count_delta must be positive")

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {**self.record, "evidence_digest": self.evidence_digest}

    def __getattr__(self, name: str) -> Any:
        if name in self.record:
            return self.record[name]
        raise AttributeError(name)


def evaluate_cutover_candidate(
    m2_d_packet: Mapping[str, Any], *, candidate_head: str, workflow: int
) -> CutoverCandidatePacket:
    _require_commit(candidate_head, "candidate_head")
    _require_positive_int(workflow, "workflow")
    packet = json.loads(_canon(m2_d_packet, "m2_d_packet"))
    supplied_digest = packet.get("packet_digest")
    scenarios = packet.get("scenarios")
    window = packet.get("window")
    order = (
        tuple(item.get("scenario_id") for item in scenarios)
        if isinstance(scenarios, list) and all(isinstance(item, dict) for item in scenarios)
        else ()
    )
    without_digest = dict(packet)
    without_digest.pop("packet_digest", None)
    digest_valid = (
        isinstance(supplied_digest, str)
        and _DIGEST.fullmatch(supplied_digest) is not None
        and _digest(without_digest, "m2_d_rehearsal_packet") == supplied_digest
    )
    checks = {
        "accepted_m2_d_packet_digest": digest_valid
        and supplied_digest == ACCEPTED_M2_D_PACKET_DIGEST,
        "accepted_m2_d_scenarios_complete": order == REQUIRED_M2_D_SCENARIOS
        and packet.get("passed_count") == len(REQUIRED_M2_D_SCENARIOS)
        and packet.get("failed_count") == 0,
        "accepted_m2_d_machine_passed": packet.get("machine_passed") is True
        and packet.get("eligible_for_human_review") is True,
        "accepted_m2_d_authority_boundary": packet.get("authority") == "rehearsal_only"
        and packet.get("shadow_authority") == EVENT_STORE_SHADOW_AUTHORITY
        and packet.get("legacy_authority_retained") is True
        and packet.get("runtime_integrated") is False
        and packet.get("production_dual_read") is False
        and packet.get("authoritative_recovery") is False
        and packet.get("cutover_authorized") is False
        and packet.get("human_accepted") is False,
        "bounded_stream_exact": isinstance(window, dict)
        and window.get("stream_id") == BOUNDED_STREAM
        and window.get("state_schema_version") == BOUNDED_STATE_SCHEMA,
        "exact_candidate_pin_present": True,
        "external_human_decision_required": True,
        "post_cutover_window_defined": True,
        "rollback_remains_required": True,
        "no_automatic_cutover": True,
        "no_runtime_or_default_change": True,
    }
    machine_passed = all(checks.values())
    record = {
        "artifact": f"exact-head-validation-{candidate_head}",
        "authority": CANDIDATE_AUTHORITY,
        "authoritative_recovery": False,
        "candidate_head": candidate_head,
        "checks": checks,
        "cutover_authorized": False,
        "eligible_for_human_review": machine_passed,
        "event_store_authority": EVENT_STORE_SHADOW_AUTHORITY,
        "human_accepted": False,
        "human_review_status": HUMAN_REVIEW_REQUIRED,
        "legacy_authority": LEGACY_AUTHORITY,
        "legacy_sidecars_read_only": False,
        "machine_passed": machine_passed,
        "observations": {
            "accepted_m2_d_reference": _accepted_reference(),
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
            "candidate_pin": {
                "artifact": f"exact-head-validation-{candidate_head}",
                "head": candidate_head,
                "workflow": workflow,
            },
            "m2_d_packet_digest": supplied_digest,
            "m2_d_scenario_ids": list(order),
            "post_cutover_requirements": [
                "separate immutable human decision pinned to exact M2-E head/workflow/artifact SHA-256",
                "event-store integrity observations before and after the bounded window",
                "event-store replay equals bounded authoritative state",
                "legacy sidecars become read-only evidence only after accepted decision",
                "rollback restores the pre-cutover bounded state",
                "post-cutover observations remain independently recalculable",
            ],
        },
        "post_cutover_observation_complete": False,
        "prerequisite": _accepted_reference(),
        "production_defaults_changed": False,
        "rollback_available": True,
        "runtime_integrated": False,
        "schema_version": M2E_CANDIDATE_SCHEMA,
        "workflow": workflow,
    }
    text = _canon(record, "m2_e_candidate_record")
    return CutoverCandidatePacket(text, _digest(record, "m2_e_candidate_packet"))


def authorize_cutover(
    candidate: CutoverCandidatePacket, decision: HumanCutoverDecision
) -> CutoverAuthorization:
    if not isinstance(candidate, CutoverCandidatePacket) or not candidate.machine_passed:
        raise M2ECutoverError("cutover candidate is not technically eligible")
    if not isinstance(decision, HumanCutoverDecision):
        raise M2ECutoverError("cutover requires an immutable human decision")
    exact = (
        decision.candidate_head == candidate.candidate_head
        and decision.workflow == candidate.workflow
        and decision.artifact == candidate.artifact
        and decision.candidate_packet_digest == candidate.packet_digest
    )
    if not exact:
        raise M2ECutoverError("human decision does not pin this exact candidate")
    if not decision.human_accepted or not decision.cutover_authorized:
        raise M2ECutoverError("human decision did not authorize cutover")
    record = {
        "artifact": decision.artifact,
        "artifact_sha256": decision.artifact_sha256,
        "authoritative_recovery": True,
        "candidate_head": candidate.candidate_head,
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
        "workflow": candidate.workflow,
    }
    return CutoverAuthorization(
        _canon(record, "m2_e_cutover_authorization"),
        _digest(record, "m2_e_cutover_authorization"),
    )


def record_post_cutover_observation(
    *,
    observation_id: str,
    authorization: CutoverAuthorization,
    store_before: StoreObservation,
    store_after: StoreObservation,
    before_state: StateEvidence,
    authoritative_state: StateEvidence,
    replay_state: StateEvidence,
    rollback_state: StateEvidence,
) -> PostCutoverObservationEvidence:
    if not isinstance(authorization, CutoverAuthorization):
        raise M2ECutoverError("post-cutover observation requires authorization")
    if not isinstance(store_before, StoreObservation) or not isinstance(
        store_after, StoreObservation
    ):
        raise M2ECutoverError("post-cutover observation requires store observations")
    before = _state_record(before_state)
    authoritative = _state_record(authoritative_state)
    replay = _state_record(replay_state)
    rollback = _state_record(rollback_state)
    delta = store_after.event_count - store_before.event_count
    checks = {
        "store_integrity_valid_before": store_before.valid,
        "store_integrity_valid_after": store_after.valid,
        "event_count_advanced": delta > 0,
        "authoritative_state_changed": authoritative_state.snapshot_digest
        != before_state.snapshot_digest,
        "event_store_replay_equivalent": replay_state.snapshot_digest
        == authoritative_state.snapshot_digest
        and replay_state.manifest_digest == authoritative_state.manifest_digest,
        "legacy_sidecars_read_only": authorization.legacy_sidecars_mode
        == LEGACY_EVIDENCE_MODE,
        "rollback_restores_pre_cutover_state": rollback_state.snapshot_digest
        == before_state.snapshot_digest
        and rollback_state.manifest_digest == before_state.manifest_digest,
        "rollback_available": authorization.rollback_available,
    }
    record = {
        "authorization": authorization.canonical_record,
        "authoritative_state": authoritative,
        "before_state": before,
        "checks": checks,
        "event_count_delta": delta,
        "event_store_authority": EVENT_STORE_BOUNDED_AUTHORITY,
        "legacy_sidecars_mode": LEGACY_EVIDENCE_MODE,
        "observation_id": observation_id,
        "replay_state": replay,
        "rollback_available": True,
        "rollback_state": rollback,
        "schema_version": M2E_OBSERVATION_SCHEMA,
        "state_schema": BOUNDED_STATE_SCHEMA,
        "store_after": store_after.canonical_record,
        "store_before": store_before.canonical_record,
        "stream_id": BOUNDED_STREAM,
    }
    return PostCutoverObservationEvidence(
        _canon(record, "m2_e_post_cutover_evidence"),
        _digest(record, "m2_e_post_cutover_evidence"),
    )


def scenario_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    observations = record.get("observations")
    if not isinstance(observations, Mapping):
        raise M2ECutoverError("record has no observations object")
    values = observations.get("m2_d_scenario_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise M2ECutoverError("record has no scenario sequence")
    if not all(isinstance(value, str) for value in values):
        raise M2ECutoverError("scenario sequence is malformed")
    return tuple(values)
