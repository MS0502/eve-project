"""Operator-reviewed trust-root contract for M3-B production provenance.

This module deliberately separates three things:

1. operator-private nonce material used only for local attestation construction/review;
2. a public digest-only launch attestation that is safe to retain as review evidence;
3. an immutable repository-reviewed registration of that exact public attestation.

A public attestation is never production provenance by itself. Runtime code cannot
self-promote it: only an exact attestation digest present in the immutable reviewed
registry can cross this trust-root boundary. C1 leaves that registry empty because
no real phone launch attestation has been reviewed yet.

Importing this module performs no runtime hook installation, persistence, source
polling, event append, observation retention, window transition, or authority grant.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from core.event_kernel import SHADOW_AUTHORITY

PRIMARY_OPERATOR_TRUST_DOMAIN = "eve.operator-attestation.primary.v1"
PRIMARY_OPERATOR_ID = "primary-operator"
BINDING_SCHEMA_VERSION = "eve.m3-b.operator-launch-binding.v1"
PUBLIC_ATTESTATION_SCHEMA_VERSION = "eve.m3-b.operator-public-launch-attestation.v1"
REVIEW_REGISTRATION_SCHEMA_VERSION = "eve.m3-b.operator-reviewed-attestation-registration.v1"
VERIFICATION_SCHEMA_VERSION = "eve.m3-b.operator-reviewed-attestation-verification.v1"
PRIVATE_NONCE_MIN_BYTES = 32
ZERO_DIGEST = "0" * 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REVIEW_ISSUANCE_TOKEN = object()


class OperatorAttestationError(ValueError):
    """Raised when operator-attestation material fails the exact trust contract."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise OperatorAttestationError(f"{field} must be a bounded non-empty string")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None or value == ZERO_DIGEST:
        raise OperatorAttestationError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise OperatorAttestationError(f"{field} must be an exact lowercase 40-hex Git SHA")
    return value


def _logical_tick(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OperatorAttestationError("logical_tick must be a non-negative integer")
    return value


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OperatorAttestationError(f"{field} is not canonical JSON material") from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _private_nonce(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) < PRIVATE_NONCE_MIN_BYTES:
        raise OperatorAttestationError(
            f"private nonce must be bytes with length >= {PRIVATE_NONCE_MIN_BYTES}"
        )
    return value


@dataclass(frozen=True, slots=True)
class OperatorLaunchBinding:
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    logical_tick: int
    fixture_only: bool = False
    trust_domain: str = PRIMARY_OPERATOR_TRUST_DOMAIN
    operator_id: str = PRIMARY_OPERATOR_ID
    schema_version: str = BINDING_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_instance_id",
            "source_instance_id",
            "entrypoint_id",
            "launch_attestation_id",
            "trust_domain",
            "operator_id",
        ):
            _identifier(getattr(self, field_name), field_name)
        _git_sha(self.repository_head_sha, "repository_head_sha")
        _logical_tick(self.logical_tick)
        if type(self.fixture_only) is not bool:
            raise OperatorAttestationError("fixture_only must be boolean")
        if self.trust_domain != PRIMARY_OPERATOR_TRUST_DOMAIN:
            raise OperatorAttestationError("operator launch binding must use the primary trust domain")
        if self.operator_id != PRIMARY_OPERATOR_ID:
            raise OperatorAttestationError("operator launch binding must use the primary operator id")
        if self.schema_version != BINDING_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise OperatorAttestationError("operator launch binding must remain exact shadow-only material")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "entrypoint_id": self.entrypoint_id,
            "fixture_only": self.fixture_only,
            "launch_attestation_id": self.launch_attestation_id,
            "logical_tick": self.logical_tick,
            "operator_id": self.operator_id,
            "repository_head_sha": self.repository_head_sha,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "trust_domain": self.trust_domain,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping(), "operator_launch_binding")


@dataclass(frozen=True, slots=True)
class OperatorPublicLaunchAttestation:
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    logical_tick: int
    private_nonce_commitment_digest: str
    nonce_binding_digest: str
    fixture_only: bool = False
    trust_domain: str = PRIMARY_OPERATOR_TRUST_DOMAIN
    operator_id: str = PRIMARY_OPERATOR_ID
    schema_version: str = PUBLIC_ATTESTATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        binding = self.binding
        _sha256(self.private_nonce_commitment_digest, "private_nonce_commitment_digest")
        _sha256(self.nonce_binding_digest, "nonce_binding_digest")
        if self.schema_version != PUBLIC_ATTESTATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise OperatorAttestationError(
                "operator public launch attestation must remain exact shadow-only evidence"
            )
        if binding.trust_domain != self.trust_domain or binding.operator_id != self.operator_id:
            raise OperatorAttestationError("operator public attestation binding identity mismatch")

    @property
    def binding(self) -> OperatorLaunchBinding:
        return OperatorLaunchBinding(
            runtime_instance_id=self.runtime_instance_id,
            source_instance_id=self.source_instance_id,
            repository_head_sha=self.repository_head_sha,
            entrypoint_id=self.entrypoint_id,
            launch_attestation_id=self.launch_attestation_id,
            logical_tick=self.logical_tick,
            fixture_only=self.fixture_only,
            trust_domain=self.trust_domain,
            operator_id=self.operator_id,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "entrypoint_id": self.entrypoint_id,
            "fixture_only": self.fixture_only,
            "launch_attestation_id": self.launch_attestation_id,
            "logical_tick": self.logical_tick,
            "nonce_binding_digest": self.nonce_binding_digest,
            "operator_id": self.operator_id,
            "private_nonce_commitment_digest": self.private_nonce_commitment_digest,
            "repository_head_sha": self.repository_head_sha,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "trust_domain": self.trust_domain,
        }

    @property
    def attestation_digest(self) -> str:
        return _digest(self.to_mapping(), "operator_public_launch_attestation")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OperatorPublicLaunchAttestation":
        if not isinstance(value, Mapping):
            raise OperatorAttestationError("operator public attestation must be a mapping")
        expected = {
            "authority",
            "entrypoint_id",
            "fixture_only",
            "launch_attestation_id",
            "logical_tick",
            "nonce_binding_digest",
            "operator_id",
            "private_nonce_commitment_digest",
            "repository_head_sha",
            "runtime_instance_id",
            "schema_version",
            "source_instance_id",
            "trust_domain",
        }
        if set(value) != expected:
            raise OperatorAttestationError("operator public attestation keys must match exact schema")
        return cls(**dict(value))


def build_operator_public_launch_attestation(
    binding: OperatorLaunchBinding,
    private_nonce: bytes,
) -> OperatorPublicLaunchAttestation:
    """Operator-local construction; the private nonce is never retained or serialized."""

    if type(binding) is not OperatorLaunchBinding:
        raise OperatorAttestationError("operator attestation requires exact launch binding")
    nonce = _private_nonce(private_nonce)
    commitment = hashlib.sha256(nonce).hexdigest()
    mac = hmac.new(
        nonce,
        _canonical(binding.to_mapping(), "operator_launch_binding").encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return OperatorPublicLaunchAttestation(
        runtime_instance_id=binding.runtime_instance_id,
        source_instance_id=binding.source_instance_id,
        repository_head_sha=binding.repository_head_sha,
        entrypoint_id=binding.entrypoint_id,
        launch_attestation_id=binding.launch_attestation_id,
        logical_tick=binding.logical_tick,
        private_nonce_commitment_digest=commitment,
        nonce_binding_digest=mac,
        fixture_only=binding.fixture_only,
    )


def verify_operator_private_binding(
    attestation: OperatorPublicLaunchAttestation,
    private_nonce: bytes,
) -> str:
    """Recompute the private binding locally and return a digest-only review trace."""

    if type(attestation) is not OperatorPublicLaunchAttestation:
        raise OperatorAttestationError("local verification requires exact public attestation")
    nonce = _private_nonce(private_nonce)
    commitment = hashlib.sha256(nonce).hexdigest()
    if not hmac.compare_digest(commitment, attestation.private_nonce_commitment_digest):
        raise OperatorAttestationError("private nonce commitment does not match attestation")
    expected_mac = hmac.new(
        nonce,
        _canonical(attestation.binding.to_mapping(), "operator_launch_binding").encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_mac, attestation.nonce_binding_digest):
        raise OperatorAttestationError("private nonce binding does not match launch binding")
    return _digest(
        {
            "attestation_digest": attestation.attestation_digest,
            "binding_digest": attestation.binding.binding_digest,
            "nonce_commitment_digest": attestation.private_nonce_commitment_digest,
            "nonce_binding_digest": attestation.nonce_binding_digest,
            "verification": "operator_private_binding_verified",
        },
        "operator_private_binding_verification",
    )


@dataclass(frozen=True, slots=True)
class ReviewedOperatorAttestationRegistration:
    attestation_digest: str
    review_record_digest: str
    local_verification_trace_digest: str
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    logical_tick: int
    private_nonce_commitment_digest: str
    nonce_binding_digest: str
    fixture_only: bool = False
    trust_domain: str = PRIMARY_OPERATOR_TRUST_DOMAIN
    operator_id: str = PRIMARY_OPERATOR_ID
    schema_version: str = REVIEW_REGISTRATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for field_name in (
            "attestation_digest",
            "review_record_digest",
            "local_verification_trace_digest",
            "private_nonce_commitment_digest",
            "nonce_binding_digest",
        ):
            _sha256(getattr(self, field_name), field_name)
        OperatorLaunchBinding(
            runtime_instance_id=self.runtime_instance_id,
            source_instance_id=self.source_instance_id,
            repository_head_sha=self.repository_head_sha,
            entrypoint_id=self.entrypoint_id,
            launch_attestation_id=self.launch_attestation_id,
            logical_tick=self.logical_tick,
            fixture_only=self.fixture_only,
            trust_domain=self.trust_domain,
            operator_id=self.operator_id,
        )
        if self.schema_version != REVIEW_REGISTRATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise OperatorAttestationError(
                "reviewed operator attestation registration must remain exact shadow-only governance"
            )


# Intentionally empty in C1. Only a later PR carrying an actually operator-verified
# phone attestation may add an exact digest registration. Runtime callers cannot mutate
# this mapping or turn an unreviewed public record into production provenance.
REVIEWED_OPERATOR_ATTESTATIONS: Mapping[
    str, ReviewedOperatorAttestationRegistration
] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ReviewedOperatorAttestationVerification:
    attestation_digest: str
    review_record_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    fixture_only: bool
    schema_version: str = VERIFICATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _REVIEW_ISSUANCE_TOKEN:
            raise OperatorAttestationError(
                "reviewed operator attestation verification must be issued by exact registry lookup"
            )
        for field_name in (
            "attestation_digest",
            "review_record_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, field_name), field_name)
        _logical_tick(self.verified_logical_tick)
        if type(self.fixture_only) is not bool:
            raise OperatorAttestationError("fixture_only must be boolean")
        if self.schema_version != VERIFICATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise OperatorAttestationError(
                "reviewed operator attestation verification must remain exact shadow-only evidence"
            )

    @property
    def counts_as_production(self) -> bool:
        return not self.fixture_only


def verify_reviewed_operator_attestation(
    candidate: Mapping[str, Any],
    attestation_material: Mapping[str, Any],
) -> ReviewedOperatorAttestationVerification:
    """Require an exact reviewed public attestation; never accept a self-authored digest."""

    if not isinstance(candidate, Mapping):
        raise OperatorAttestationError("runtime provenance candidate material must be a mapping")
    attestation = OperatorPublicLaunchAttestation.from_mapping(attestation_material)
    required_candidate_fields = {
        "trust_domain",
        "runtime_instance_id",
        "source_instance_id",
        "repository_head_sha",
        "entrypoint_id",
        "launch_attestation_id",
        "launch_attestation_digest",
        "logical_tick",
        "fixture_only",
    }
    if not required_candidate_fields.issubset(candidate):
        raise OperatorAttestationError("runtime provenance candidate lacks attestation binding fields")
    expected = (
        attestation.trust_domain,
        attestation.runtime_instance_id,
        attestation.source_instance_id,
        attestation.repository_head_sha,
        attestation.entrypoint_id,
        attestation.launch_attestation_id,
        attestation.attestation_digest,
        attestation.logical_tick,
        attestation.fixture_only,
    )
    actual = tuple(candidate[field] for field in (
        "trust_domain",
        "runtime_instance_id",
        "source_instance_id",
        "repository_head_sha",
        "entrypoint_id",
        "launch_attestation_id",
        "launch_attestation_digest",
        "logical_tick",
        "fixture_only",
    ))
    if actual != expected:
        raise OperatorAttestationError("reviewed operator attestation does not bind exact runtime candidate")
    registration = REVIEWED_OPERATOR_ATTESTATIONS.get(attestation.attestation_digest)
    if type(registration) is not ReviewedOperatorAttestationRegistration:
        raise OperatorAttestationError("operator launch attestation digest is not repository-reviewed")
    registered = (
        registration.trust_domain,
        registration.operator_id,
        registration.runtime_instance_id,
        registration.source_instance_id,
        registration.repository_head_sha,
        registration.entrypoint_id,
        registration.launch_attestation_id,
        registration.logical_tick,
        registration.private_nonce_commitment_digest,
        registration.nonce_binding_digest,
        registration.fixture_only,
    )
    observed = (
        attestation.trust_domain,
        attestation.operator_id,
        attestation.runtime_instance_id,
        attestation.source_instance_id,
        attestation.repository_head_sha,
        attestation.entrypoint_id,
        attestation.launch_attestation_id,
        attestation.logical_tick,
        attestation.private_nonce_commitment_digest,
        attestation.nonce_binding_digest,
        attestation.fixture_only,
    )
    if registered != observed:
        raise OperatorAttestationError("reviewed operator attestation registration does not match public record")
    verifier_trace_digest = _digest(
        {
            "attestation_digest": attestation.attestation_digest,
            "local_verification_trace_digest": registration.local_verification_trace_digest,
            "review_record_digest": registration.review_record_digest,
            "trust_domain": registration.trust_domain,
            "verification": "repository_reviewed_operator_attestation",
        },
        "reviewed_operator_attestation_verification",
    )
    return ReviewedOperatorAttestationVerification(
        attestation_digest=attestation.attestation_digest,
        review_record_digest=registration.review_record_digest,
        verifier_trace_digest=verifier_trace_digest,
        verified_logical_tick=registration.logical_tick,
        fixture_only=registration.fixture_only,
        _issuance_token=_REVIEW_ISSUANCE_TOKEN,
    )
