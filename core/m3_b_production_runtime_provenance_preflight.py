"""Fail-closed M3-B production-runtime provenance verification preflight.

Repository entrypoint identity, a PID, argv/environment values, a caller-selected
source instance, or ``fixture_only=False`` are not trusted production provenance.
This module defines the only future issuance shape that may turn independently
verified launch attestation material into an immutable runtime provenance proof.

No runtime-provenance verifier is registered by this module. Importing or constructing
its objects performs no runtime hook installation, source polling, persistence, event
append, registry-owner mutation, observation-window transition, or authority grant.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_production_capture_adapter import (
    REGISTERED_PRODUCTION_SOURCE_VERIFIERS,
)

CANDIDATE_SCHEMA_VERSION = "eve.m3-b.production-runtime-provenance-candidate.v1"
VERIFIER_RESULT_SCHEMA_VERSION = "eve.m3-b.production-runtime-provenance-verifier-result.v1"
VERIFIER_REGISTRATION_SCHEMA_VERSION = "eve.m3-b.production-runtime-provenance-verifier-registration.v1"
VERIFICATION_SCHEMA_VERSION = "eve.m3-b.production-runtime-provenance-verification.v1"
CAPABILITY_SCHEMA_VERSION = "eve.m3-b.production-runtime-provenance-capability.v1"
PRODUCTION_ENVIRONMENT = "production"
TEST_FIXTURE_ENVIRONMENT = "test_fixture"
RUNTIME_PROVENANCE_VERIFIER_ABSENT = "PRODUCTION_RUNTIME_PROVENANCE_VERIFIER_ABSENT"
RUNTIME_PROVENANCE_ANCHOR_ABSENT = "PREDICTION_ERROR_PRODUCTION_RUNTIME_PROVENANCE_ANCHOR_ABSENT"
PRODUCTION_SOURCE_VERIFIER_BLOCKER = "REGISTRY_PRODUCTION_SOURCE_VERIFIER_COVERAGE_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
OBSERVATION_WINDOW_BLOCKER = "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED"
SELF_AUTHORED_CLAIMS_NOT_TRUSTED = (
    "python___main___entrypoint",
    "main_py_repl_path",
    "process_id",
    "argv_or_environment_flag",
    "caller_selected_source_instance_id",
    "fixture_only_false",
    "self_hashed_launch_metadata",
)
ZERO_DIGEST = "0" * 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PROVENANCE_ISSUANCE_TOKEN = object()


class ProductionRuntimeProvenanceError(ValueError):
    """Raised when runtime provenance cannot satisfy the exact trust boundary."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ProductionRuntimeProvenanceError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None or value == ZERO_DIGEST:
        raise ProductionRuntimeProvenanceError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise ProductionRuntimeProvenanceError(
            f"{field} must be an exact lowercase 40-hex Git SHA"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionRuntimeProvenanceError(
            f"{field} must be a non-negative integer"
        )
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
        raise ProductionRuntimeProvenanceError(
            f"{field} is not canonical JSON material"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeProvenanceCandidate:
    """Untrusted launch claim; construction never establishes production origin."""

    trust_domain: str
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    launch_attestation_digest: str
    logical_tick: int
    fixture_only: bool = False
    schema_version: str = CANDIDATE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for name in (
            "trust_domain",
            "runtime_instance_id",
            "source_instance_id",
            "entrypoint_id",
            "launch_attestation_id",
        ):
            _identifier(getattr(self, name), name)
        _git_sha(self.repository_head_sha, "repository_head_sha")
        _sha256(self.launch_attestation_digest, "launch_attestation_digest")
        _nonnegative_int(self.logical_tick, "logical_tick")
        if type(self.fixture_only) is not bool:
            raise ProductionRuntimeProvenanceError("fixture_only must be boolean")
        if self.schema_version != CANDIDATE_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise ProductionRuntimeProvenanceError(
                "runtime provenance candidate must remain exact shadow-only untrusted material"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "entrypoint_id": self.entrypoint_id,
            "fixture_only": self.fixture_only,
            "launch_attestation_digest": self.launch_attestation_digest,
            "launch_attestation_id": self.launch_attestation_id,
            "logical_tick": self.logical_tick,
            "repository_head_sha": self.repository_head_sha,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "trust_domain": self.trust_domain,
        }

    @property
    def candidate_digest(self) -> str:
        return _digest(self.to_mapping(), "runtime_provenance_candidate")


@dataclass(frozen=True, slots=True)
class RuntimeProvenanceVerifierResult:
    """Untrusted callable output checked before provenance issuance."""

    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    launch_attestation_digest: str
    candidate_digest: str
    verifier_trace_digest: str
    verified_logical_tick: int
    verification_environment: str = PRODUCTION_ENVIRONMENT
    independent_trust_root_verified: bool = True
    production_launch_verified: bool = True
    non_ci_runtime_verified: bool = True
    fixture_only: bool = False
    schema_version: str = VERIFIER_RESULT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for name in (
            "runtime_instance_id",
            "source_instance_id",
            "entrypoint_id",
            "launch_attestation_id",
        ):
            _identifier(getattr(self, name), name)
        _git_sha(self.repository_head_sha, "repository_head_sha")
        for name in (
            "launch_attestation_digest",
            "candidate_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, name), name)
        _nonnegative_int(self.verified_logical_tick, "verified_logical_tick")
        if type(self.fixture_only) is not bool:
            raise ProductionRuntimeProvenanceError("fixture_only must be boolean")
        if self.verification_environment not in {
            PRODUCTION_ENVIRONMENT,
            TEST_FIXTURE_ENVIRONMENT,
        }:
            raise ProductionRuntimeProvenanceError(
                "unsupported runtime provenance verification environment"
            )
        if self.verification_environment == PRODUCTION_ENVIRONMENT and self.fixture_only:
            raise ProductionRuntimeProvenanceError(
                "production provenance verification cannot be fixture_only"
            )
        if self.verification_environment == TEST_FIXTURE_ENVIRONMENT and not self.fixture_only:
            raise ProductionRuntimeProvenanceError(
                "test_fixture verification must remain fixture_only"
            )
        if not (
            self.independent_trust_root_verified
            and self.production_launch_verified
            and self.non_ci_runtime_verified
        ):
            raise ProductionRuntimeProvenanceError(
                "runtime provenance verifier result requires independent trust, launch, and non-CI proof"
            )
        if self.schema_version != VERIFIER_RESULT_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise ProductionRuntimeProvenanceError(
                "runtime provenance verifier result must remain exact shadow-only evidence"
            )


RuntimeProvenanceVerifierCallable = Callable[
    [RuntimeProvenanceCandidate, Mapping[str, Any]],
    RuntimeProvenanceVerifierResult,
]


@dataclass(frozen=True, slots=True)
class RuntimeProvenanceVerifierRegistration:
    trust_domain: str
    verifier_id: str
    verifier_version: str
    verifier: RuntimeProvenanceVerifierCallable = field(repr=False, compare=False)
    schema_version: str = VERIFIER_REGISTRATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        for name in ("trust_domain", "verifier_id", "verifier_version"):
            _identifier(getattr(self, name), name)
        if not callable(self.verifier):
            raise ProductionRuntimeProvenanceError(
                "runtime provenance verifier registration must be executable"
            )
        if self.schema_version != VERIFIER_REGISTRATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise ProductionRuntimeProvenanceError(
                "runtime provenance verifier registration must remain exact shadow-only machinery"
            )


# Deliberately empty and immutable. A future independently reviewable trust source must
# be implemented before repository code may add a registration here.
REGISTERED_RUNTIME_PROVENANCE_VERIFIERS: Mapping[
    str, RuntimeProvenanceVerifierRegistration
] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ProductionRuntimeProvenanceVerification:
    """Immutable verified provenance issued only by registered verifier execution."""

    trust_domain: str
    runtime_instance_id: str
    source_instance_id: str
    repository_head_sha: str
    entrypoint_id: str
    launch_attestation_id: str
    launch_attestation_digest: str
    candidate_digest: str
    verifier_id: str
    verifier_version: str
    verifier_trace_digest: str
    verified_logical_tick: int
    verification_environment: str = PRODUCTION_ENVIRONMENT
    independent_trust_root_verified: bool = True
    production_launch_verified: bool = True
    non_ci_runtime_verified: bool = True
    fixture_only: bool = False
    schema_version: str = VERIFICATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PROVENANCE_ISSUANCE_TOKEN:
            raise ProductionRuntimeProvenanceError(
                "production runtime provenance must be issued by registered verifier execution"
            )
        for name in (
            "trust_domain",
            "runtime_instance_id",
            "source_instance_id",
            "entrypoint_id",
            "launch_attestation_id",
            "verifier_id",
            "verifier_version",
        ):
            _identifier(getattr(self, name), name)
        _git_sha(self.repository_head_sha, "repository_head_sha")
        for name in (
            "launch_attestation_digest",
            "candidate_digest",
            "verifier_trace_digest",
        ):
            _sha256(getattr(self, name), name)
        _nonnegative_int(self.verified_logical_tick, "verified_logical_tick")
        if type(self.fixture_only) is not bool:
            raise ProductionRuntimeProvenanceError("fixture_only must be boolean")
        if self.verification_environment not in {
            PRODUCTION_ENVIRONMENT,
            TEST_FIXTURE_ENVIRONMENT,
        }:
            raise ProductionRuntimeProvenanceError(
                "unsupported runtime provenance verification environment"
            )
        if self.verification_environment == PRODUCTION_ENVIRONMENT and self.fixture_only:
            raise ProductionRuntimeProvenanceError(
                "production provenance cannot be fixture_only"
            )
        if self.verification_environment == TEST_FIXTURE_ENVIRONMENT and not self.fixture_only:
            raise ProductionRuntimeProvenanceError(
                "test_fixture provenance must remain fixture_only"
            )
        if not (
            self.independent_trust_root_verified
            and self.production_launch_verified
            and self.non_ci_runtime_verified
        ):
            raise ProductionRuntimeProvenanceError(
                "issued provenance requires independent trust, production launch, and non-CI proof"
            )
        if self.schema_version != VERIFICATION_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise ProductionRuntimeProvenanceError(
                "issued runtime provenance must remain exact shadow-only evidence"
            )

    @property
    def counts_as_production(self) -> bool:
        return (
            self.verification_environment == PRODUCTION_ENVIRONMENT
            and not self.fixture_only
            and self.independent_trust_root_verified
            and self.production_launch_verified
            and self.non_ci_runtime_verified
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "candidate_digest": self.candidate_digest,
            "entrypoint_id": self.entrypoint_id,
            "fixture_only": self.fixture_only,
            "independent_trust_root_verified": self.independent_trust_root_verified,
            "launch_attestation_digest": self.launch_attestation_digest,
            "launch_attestation_id": self.launch_attestation_id,
            "non_ci_runtime_verified": self.non_ci_runtime_verified,
            "production_launch_verified": self.production_launch_verified,
            "repository_head_sha": self.repository_head_sha,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
            "source_instance_id": self.source_instance_id,
            "trust_domain": self.trust_domain,
            "verification_environment": self.verification_environment,
            "verified_logical_tick": self.verified_logical_tick,
            "verifier_id": self.verifier_id,
            "verifier_trace_digest": self.verifier_trace_digest,
            "verifier_version": self.verifier_version,
        }

    @property
    def verification_digest(self) -> str:
        return _digest(self.to_mapping(), "production_runtime_provenance_verification")


def execute_registered_runtime_provenance_verifier(
    candidate: RuntimeProvenanceCandidate,
    attestation_material: Mapping[str, Any],
) -> ProductionRuntimeProvenanceVerification:
    """Verify attestation through a registered independent trust-domain verifier."""

    if type(candidate) is not RuntimeProvenanceCandidate:
        raise ProductionRuntimeProvenanceError(
            "runtime provenance verifier requires exact untrusted candidate material"
        )
    if not isinstance(attestation_material, Mapping):
        raise ProductionRuntimeProvenanceError(
            "runtime provenance attestation material must be a mapping"
        )
    material_digest = _digest(
        dict(attestation_material),
        "runtime_provenance_attestation_material",
    )
    if material_digest != candidate.launch_attestation_digest:
        raise ProductionRuntimeProvenanceError(
            "attestation material digest does not match candidate"
        )
    registration = REGISTERED_RUNTIME_PROVENANCE_VERIFIERS.get(candidate.trust_domain)
    if type(registration) is not RuntimeProvenanceVerifierRegistration:
        raise ProductionRuntimeProvenanceError(
            "runtime provenance verifier is not registered for this trust domain"
        )
    if registration.trust_domain != candidate.trust_domain:
        raise ProductionRuntimeProvenanceError(
            "runtime provenance registration trust domain mismatch"
        )
    result = registration.verifier(candidate, dict(attestation_material))
    if type(result) is not RuntimeProvenanceVerifierResult:
        raise ProductionRuntimeProvenanceError(
            "runtime provenance verifier must return exact immutable verifier result"
        )
    expected_environment = (
        TEST_FIXTURE_ENVIRONMENT if candidate.fixture_only else PRODUCTION_ENVIRONMENT
    )
    if (
        result.fixture_only is not candidate.fixture_only
        or result.verification_environment != expected_environment
    ):
        raise ProductionRuntimeProvenanceError(
            "runtime provenance verifier result cannot change candidate fixture classification"
        )
    expected = (
        candidate.runtime_instance_id,
        candidate.source_instance_id,
        candidate.repository_head_sha,
        candidate.entrypoint_id,
        candidate.launch_attestation_id,
        candidate.launch_attestation_digest,
        candidate.candidate_digest,
    )
    actual = (
        result.runtime_instance_id,
        result.source_instance_id,
        result.repository_head_sha,
        result.entrypoint_id,
        result.launch_attestation_id,
        result.launch_attestation_digest,
        result.candidate_digest,
    )
    if actual != expected:
        raise ProductionRuntimeProvenanceError(
            "runtime provenance verifier result does not bind the exact candidate"
        )
    if result.verified_logical_tick < candidate.logical_tick:
        raise ProductionRuntimeProvenanceError(
            "runtime provenance verification cannot precede candidate logical tick"
        )
    verification = ProductionRuntimeProvenanceVerification(
        trust_domain=candidate.trust_domain,
        runtime_instance_id=result.runtime_instance_id,
        source_instance_id=result.source_instance_id,
        repository_head_sha=result.repository_head_sha,
        entrypoint_id=result.entrypoint_id,
        launch_attestation_id=result.launch_attestation_id,
        launch_attestation_digest=result.launch_attestation_digest,
        candidate_digest=result.candidate_digest,
        verifier_id=registration.verifier_id,
        verifier_version=registration.verifier_version,
        verifier_trace_digest=result.verifier_trace_digest,
        verified_logical_tick=result.verified_logical_tick,
        verification_environment=result.verification_environment,
        independent_trust_root_verified=result.independent_trust_root_verified,
        production_launch_verified=result.production_launch_verified,
        non_ci_runtime_verified=result.non_ci_runtime_verified,
        fixture_only=result.fixture_only,
        _issuance_token=_PROVENANCE_ISSUANCE_TOKEN,
    )
    return verification


@dataclass(frozen=True, slots=True)
class ProductionRuntimeProvenanceCapabilityStatus:
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    verifier_registry_immutable: bool = True
    registered_runtime_provenance_verifier_count: int = 0
    verified_production_runtime_anchor_count: int = 0
    registered_production_source_verifier_count: int = 0
    retained_real_observation_count: int = 0
    positive_confidence_real_observation_count: int = 0
    observation_window_eligible: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise ProductionRuntimeProvenanceError(
                "unsupported runtime provenance capability status"
            )
        if self.verifier_registry_immutable is not True:
            raise ProductionRuntimeProvenanceError(
                "runtime provenance verifier registry must remain immutable"
            )
        if self.registered_runtime_provenance_verifier_count != len(
            REGISTERED_RUNTIME_PROVENANCE_VERIFIERS
        ):
            raise ProductionRuntimeProvenanceError(
                "runtime provenance verifier count disagrees with exact registry"
            )
        if self.registered_production_source_verifier_count != len(
            REGISTERED_PRODUCTION_SOURCE_VERIFIERS
        ):
            raise ProductionRuntimeProvenanceError(
                "production source verifier count disagrees with exact registry"
            )
        if self.registered_runtime_provenance_verifier_count != 0:
            raise ProductionRuntimeProvenanceError(
                "preflight must not register a runtime provenance verifier"
            )
        if self.verified_production_runtime_anchor_count != 0:
            raise ProductionRuntimeProvenanceError(
                "preflight cannot fabricate a verified production runtime anchor"
            )
        if self.retained_real_observation_count != 0 or self.positive_confidence_real_observation_count != 0:
            raise ProductionRuntimeProvenanceError(
                "runtime provenance preflight cannot fabricate real observation coverage"
            )
        if any(
            (
                self.observation_window_eligible,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise ProductionRuntimeProvenanceError(
                "runtime provenance preflight cannot open window, completion, or authority"
            )

    @property
    def self_authored_claims_not_trusted(self) -> tuple[str, ...]:
        return SELF_AUTHORED_CLAIMS_NOT_TRUSTED

    @property
    def blockers(self) -> tuple[str, ...]:
        return (
            RUNTIME_PROVENANCE_VERIFIER_ABSENT,
            RUNTIME_PROVENANCE_ANCHOR_ABSENT,
            PRODUCTION_SOURCE_VERIFIER_BLOCKER,
            POSITIVE_CONFIDENCE_BLOCKER,
            OBSERVATION_WINDOW_BLOCKER,
        )


def production_runtime_provenance_capability_status() -> ProductionRuntimeProvenanceCapabilityStatus:
    return ProductionRuntimeProvenanceCapabilityStatus(
        registered_runtime_provenance_verifier_count=len(
            REGISTERED_RUNTIME_PROVENANCE_VERIFIERS
        ),
        registered_production_source_verifier_count=len(
            REGISTERED_PRODUCTION_SOURCE_VERIFIERS
        ),
    )
