"""M1-D lifecycle ownership and disconnected shadow-bridge contracts.

This module is declaration-only. It imports, constructs, calls, patches, and
activates no legacy module. Every bridge remains disconnected, default-disabled,
authority-free, event-free, capability-free, and persistence-free.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from core.event_kernel import canonical_json_object

OWNER_SCHEMA_VERSION = "eve.shadow-lifecycle-owner.v1"
BRIDGE_SCHEMA_VERSION = "eve.shadow-bridge-contract.v1"
REGISTRY_SCHEMA_VERSION = "eve.shadow-bridge-registry.v1"
FAILURE_SCHEMA_VERSION = "eve.shadow-bridge-failure-signal.v1"

BRIDGE_DOMAINS: tuple[str, ...] = ("activity", "chat", "goal", "memory")
REVIEWED_DISPOSITION_DOCUMENT = "docs/audit/M0_D_MODULE_DISPOSITION.md"
REVIEWED_BRIDGE_SOURCES: dict[str, tuple[str, str]] = {
    "activity": ("adapters/agency_adapter.py", "WRAP"),
    "chat": ("language/streaming.py", "REWRITE"),
    "goal": ("adapters/goal_adapter.py", "WRAP"),
    "memory": ("adapters/memory_adapter.py", "WRAP"),
}

NO_AUTHORITY = "none"
DISCONNECTED_STATUS = "declared_disconnected"
DISCONNECTED_MODE = "disconnected"
NO_PERSISTENCE = "none"
RETRY_FORBIDDEN = "forbidden"
SUPPRESSION_FORBIDDEN = "forbidden"
SHADOW_ROLLBACK_ONLY = "shadow_state_only"
PROPAGATE_ORIGINAL = "surface_signal_and_propagate_original"

REQUIRED_ACTIVATION_EVIDENCE: tuple[str, ...] = (
    "explicit_reviewed_integration_pr",
    "initialization_and_shutdown_implementation",
    "failure_propagation_and_interruption_tests",
    "shadow_rollback_implementation",
    "exact_head_validation_green",
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_STAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ShadowLifecycleContractError(ValueError):
    """Raised when a lifecycle or bridge declaration is malformed."""


class UnknownShadowBridge(ShadowLifecycleContractError):
    """Raised when a requested owner, bridge, or domain is not registered."""


def _require_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ShadowLifecycleContractError(f"{field} is not a canonical identifier")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ShadowLifecycleContractError(f"{field} must be canonical text")
    return value


@dataclass(frozen=True, slots=True)
class LifecycleOwnerContract:
    """Immutable responsibility assignment for one disconnected domain."""

    owner_id: str
    domain: str
    initialization_responsibility: str = "explicit_caller_construction_only"
    shutdown_responsibility: str = "release_shadow_resources_only"
    interruption_responsibility: str = "cancel_shadow_work_preserve_legacy"
    failure_responsibility: str = PROPAGATE_ORIGINAL
    provenance_responsibility: str = "retain_source_and_contract_versions"
    rollback_responsibility: str = "restore_registered_shadow_state_only"
    authority: str = NO_AUTHORITY
    schema_version: str = OWNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.owner_id, field="owner_id")
        if self.domain not in BRIDGE_DOMAINS:
            raise ShadowLifecycleContractError("owner domain is outside M1-D scope")
        if self.owner_id != f"m1.lifecycle.{self.domain}":
            raise ShadowLifecycleContractError("owner_id does not match domain")
        if self.schema_version != OWNER_SCHEMA_VERSION:
            raise ShadowLifecycleContractError("unsupported owner schema version")
        required = {
            "initialization_responsibility": "explicit_caller_construction_only",
            "shutdown_responsibility": "release_shadow_resources_only",
            "interruption_responsibility": "cancel_shadow_work_preserve_legacy",
            "failure_responsibility": PROPAGATE_ORIGINAL,
            "provenance_responsibility": "retain_source_and_contract_versions",
            "rollback_responsibility": "restore_registered_shadow_state_only",
            "authority": NO_AUTHORITY,
        }
        for field, expected in required.items():
            if getattr(self, field) != expected:
                raise ShadowLifecycleContractError(
                    f"{field} cannot weaken lifecycle ownership"
                )

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "domain": self.domain,
            "failure_responsibility": self.failure_responsibility,
            "initialization_responsibility": self.initialization_responsibility,
            "interruption_responsibility": self.interruption_responsibility,
            "owner_id": self.owner_id,
            "provenance_responsibility": self.provenance_responsibility,
            "rollback_responsibility": self.rollback_responsibility,
            "schema_version": self.schema_version,
            "shutdown_responsibility": self.shutdown_responsibility,
        }


@dataclass(frozen=True, slots=True)
class ShadowBridgeContract:
    """Immutable declaration of one reviewed but unimplemented bridge."""

    bridge_id: str
    domain: str
    owner_id: str
    source_module_path: str
    source_disposition: str
    source_evidence_document: str = REVIEWED_DISPOSITION_DOCUMENT
    lifecycle_status: str = DISCONNECTED_STATUS
    integration_mode: str = DISCONNECTED_MODE
    default_enabled: bool = False
    authority: str = NO_AUTHORITY
    emitted_event_types: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    persistence_mode: str = NO_PERSISTENCE
    retry_policy: str = RETRY_FORBIDDEN
    suppression_policy: str = SUPPRESSION_FORBIDDEN
    rollback_scope: str = SHADOW_ROLLBACK_ONLY
    future_activation_evidence: tuple[str, ...] = REQUIRED_ACTIVATION_EVIDENCE
    schema_version: str = BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.bridge_id, field="bridge_id")
        _require_identifier(self.owner_id, field="owner_id")
        if self.domain not in BRIDGE_DOMAINS:
            raise ShadowLifecycleContractError("bridge domain is outside M1-D scope")
        if self.bridge_id != f"m1.bridge.{self.domain}":
            raise ShadowLifecycleContractError("bridge_id does not match domain")
        if self.owner_id != f"m1.lifecycle.{self.domain}":
            raise ShadowLifecycleContractError("bridge owner does not match domain")
        expected_source = REVIEWED_BRIDGE_SOURCES[self.domain]
        if (self.source_module_path, self.source_disposition) != expected_source:
            raise ShadowLifecycleContractError(
                "bridge source and disposition must match reviewed M0-D evidence"
            )
        if self.source_evidence_document != REVIEWED_DISPOSITION_DOCUMENT:
            raise ShadowLifecycleContractError(
                "bridge evidence document must remain the reviewed M0-D map"
            )
        _require_text(self.source_module_path, field="source_module_path")
        if self.schema_version != BRIDGE_SCHEMA_VERSION:
            raise ShadowLifecycleContractError("unsupported bridge schema version")
        required = {
            "lifecycle_status": DISCONNECTED_STATUS,
            "integration_mode": DISCONNECTED_MODE,
            "default_enabled": False,
            "authority": NO_AUTHORITY,
            "emitted_event_types": (),
            "required_capabilities": (),
            "persistence_mode": NO_PERSISTENCE,
            "retry_policy": RETRY_FORBIDDEN,
            "suppression_policy": SUPPRESSION_FORBIDDEN,
            "rollback_scope": SHADOW_ROLLBACK_ONLY,
            "future_activation_evidence": REQUIRED_ACTIVATION_EVIDENCE,
        }
        for field, expected in required.items():
            if getattr(self, field) != expected:
                raise ShadowLifecycleContractError(
                    f"{field} cannot activate or weaken a bridge"
                )

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "bridge_id": self.bridge_id,
            "default_enabled": self.default_enabled,
            "domain": self.domain,
            "emitted_event_types": list(self.emitted_event_types),
            "future_activation_evidence": list(self.future_activation_evidence),
            "integration_mode": self.integration_mode,
            "lifecycle_status": self.lifecycle_status,
            "owner_id": self.owner_id,
            "persistence_mode": self.persistence_mode,
            "required_capabilities": list(self.required_capabilities),
            "retry_policy": self.retry_policy,
            "rollback_scope": self.rollback_scope,
            "schema_version": self.schema_version,
            "source_disposition": self.source_disposition,
            "source_evidence_document": self.source_evidence_document,
            "source_module_path": self.source_module_path,
            "suppression_policy": self.suppression_policy,
        }

    @property
    def digest(self) -> str:
        encoded = canonical_json_object(
            self.canonical_record,
            field="shadow_bridge_contract",
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ShadowBridgeRegistry:
    """Immutable complete registry for the four M1-D domains."""

    owners: tuple[LifecycleOwnerContract, ...]
    bridges: tuple[ShadowBridgeContract, ...]
    schema_version: str = REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise ShadowLifecycleContractError("unsupported registry schema version")
        if not isinstance(self.owners, tuple) or not isinstance(self.bridges, tuple):
            raise ShadowLifecycleContractError("registry collections must be tuples")

        owner_ids: set[str] = set()
        owner_domains: set[str] = set()
        for owner in self.owners:
            if not isinstance(owner, LifecycleOwnerContract):
                raise ShadowLifecycleContractError("registry owner is malformed")
            if owner.owner_id in owner_ids or owner.domain in owner_domains:
                raise ShadowLifecycleContractError("duplicate lifecycle owner")
            owner_ids.add(owner.owner_id)
            owner_domains.add(owner.domain)

        bridge_ids: set[str] = set()
        bridge_domains: set[str] = set()
        for bridge in self.bridges:
            if not isinstance(bridge, ShadowBridgeContract):
                raise ShadowLifecycleContractError("registry bridge is malformed")
            if bridge.bridge_id in bridge_ids or bridge.domain in bridge_domains:
                raise ShadowLifecycleContractError("duplicate shadow bridge")
            if bridge.owner_id not in owner_ids:
                raise ShadowLifecycleContractError("bridge owner is not registered")
            bridge_ids.add(bridge.bridge_id)
            bridge_domains.add(bridge.domain)

        required = set(BRIDGE_DOMAINS)
        if owner_domains != required or bridge_domains != required:
            raise ShadowLifecycleContractError(
                "registry must contain exactly activity, chat, goal, and memory"
            )

    def owner(self, owner_id: str) -> LifecycleOwnerContract:
        for owner in self.owners:
            if owner.owner_id == owner_id:
                return owner
        raise UnknownShadowBridge(owner_id)

    def bridge(self, bridge_id: str) -> ShadowBridgeContract:
        for bridge in self.bridges:
            if bridge.bridge_id == bridge_id:
                return bridge
        raise UnknownShadowBridge(bridge_id)

    def bridge_for_domain(self, domain: str) -> ShadowBridgeContract:
        for bridge in self.bridges:
            if bridge.domain == domain:
                return bridge
        raise UnknownShadowBridge(domain)

    @property
    def canonical_record(self) -> dict[str, Any]:
        return {
            "bridges": [
                bridge.canonical_record
                for bridge in sorted(self.bridges, key=lambda item: item.bridge_id)
            ],
            "owners": [
                owner.canonical_record
                for owner in sorted(self.owners, key=lambda item: item.owner_id)
            ],
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        encoded = canonical_json_object(
            self.canonical_record,
            field="shadow_bridge_registry",
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BridgeFailureSignal:
    """Redacted evidence that a future bridge owner must propagate a failure."""

    bridge_id: str
    owner_id: str
    domain: str
    stage: str
    error_type: str
    error_message_digest: str
    handling: str = PROPAGATE_ORIGINAL
    retry_allowed: bool = False
    suppression_allowed: bool = False
    legacy_authority_changed: bool = False
    schema_version: str = FAILURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.bridge_id, field="bridge_id")
        _require_identifier(self.owner_id, field="owner_id")
        if self.domain not in BRIDGE_DOMAINS:
            raise ShadowLifecycleContractError("failure domain is outside M1-D scope")
        if self.bridge_id != f"m1.bridge.{self.domain}":
            raise ShadowLifecycleContractError("failure bridge does not match domain")
        if self.owner_id != f"m1.lifecycle.{self.domain}":
            raise ShadowLifecycleContractError("failure owner does not match domain")
        if not isinstance(self.stage, str) or not _STAGE_PATTERN.fullmatch(self.stage):
            raise ShadowLifecycleContractError("failure stage is not canonical")
        _require_text(self.error_type, field="error_type")
        if (
            not isinstance(self.error_message_digest, str)
            or not _DIGEST_PATTERN.fullmatch(self.error_message_digest)
        ):
            raise ShadowLifecycleContractError("failure digest is malformed")
        required = {
            "handling": PROPAGATE_ORIGINAL,
            "retry_allowed": False,
            "suppression_allowed": False,
            "legacy_authority_changed": False,
            "schema_version": FAILURE_SCHEMA_VERSION,
        }
        for field, expected in required.items():
            if getattr(self, field) != expected:
                raise ShadowLifecycleContractError(
                    f"{field} cannot weaken failure propagation"
                )

    @classmethod
    def capture(
        cls,
        registry: ShadowBridgeRegistry,
        *,
        bridge_id: str,
        stage: str,
        error: BaseException,
    ) -> "BridgeFailureSignal":
        if not isinstance(registry, ShadowBridgeRegistry):
            raise ShadowLifecycleContractError("failure capture requires registry")
        bridge = registry.bridge(bridge_id)
        if not isinstance(error, BaseException):
            raise ShadowLifecycleContractError("failure capture requires exception")
        digest = hashlib.sha256(
            str(error).encode("utf-8", errors="replace")
        ).hexdigest()
        return cls(
            bridge_id=bridge.bridge_id,
            owner_id=bridge.owner_id,
            domain=bridge.domain,
            stage=stage,
            error_type=type(error).__name__,
            error_message_digest=digest,
        )


DEFAULT_BRIDGE_REGISTRY = ShadowBridgeRegistry(
    owners=tuple(
        LifecycleOwnerContract(
            owner_id=f"m1.lifecycle.{domain}",
            domain=domain,
        )
        for domain in BRIDGE_DOMAINS
    ),
    bridges=tuple(
        ShadowBridgeContract(
            bridge_id=f"m1.bridge.{domain}",
            domain=domain,
            owner_id=f"m1.lifecycle.{domain}",
            source_module_path=REVIEWED_BRIDGE_SOURCES[domain][0],
            source_disposition=REVIEWED_BRIDGE_SOURCES[domain][1],
        )
        for domain in BRIDGE_DOMAINS
    ),
)
