from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.shadow_lifecycle import (
    BRIDGE_DOMAINS,
    BRIDGE_SCHEMA_VERSION,
    DEFAULT_BRIDGE_REGISTRY,
    DISCONNECTED_MODE,
    DISCONNECTED_STATUS,
    FAILURE_SCHEMA_VERSION,
    NO_AUTHORITY,
    NO_PERSISTENCE,
    OWNER_SCHEMA_VERSION,
    PROPAGATE_ORIGINAL,
    REGISTRY_SCHEMA_VERSION,
    REQUIRED_ACTIVATION_EVIDENCE,
    RETRY_FORBIDDEN,
    SHADOW_ROLLBACK_ONLY,
    SUPPRESSION_FORBIDDEN,
    BridgeFailureSignal,
    LifecycleOwnerContract,
    ShadowBridgeContract,
    ShadowBridgeRegistry,
    ShadowLifecycleContractError,
    UnknownShadowBridge,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_PATH = REPO_ROOT / "core/shadow_lifecycle.py"
DISPOSITION_PATH = REPO_ROOT / "docs/audit/M0_D_MODULE_DISPOSITION.md"
EXPECTED_BRIDGES = {
    "activity": ("adapters/agency_adapter.py", "WRAP"),
    "chat": ("language/streaming.py", "REWRITE"),
    "goal": ("adapters/goal_adapter.py", "WRAP"),
    "memory": ("adapters/memory_adapter.py", "WRAP"),
}


def test_registry_contains_exact_reviewed_domains_sources_and_dispositions():
    registry = DEFAULT_BRIDGE_REGISTRY
    assert registry.schema_version == REGISTRY_SCHEMA_VERSION
    assert tuple(sorted(BRIDGE_DOMAINS)) == ("activity", "chat", "goal", "memory")
    assert {owner.domain for owner in registry.owners} == set(EXPECTED_BRIDGES)
    assert {bridge.domain for bridge in registry.bridges} == set(EXPECTED_BRIDGES)

    document = DISPOSITION_PATH.read_text(encoding="utf-8")
    for domain, (source_path, disposition) in EXPECTED_BRIDGES.items():
        owner = registry.owner(f"m1.lifecycle.{domain}")
        bridge = registry.bridge(f"m1.bridge.{domain}")
        assert owner.domain == bridge.domain == domain
        assert bridge.owner_id == owner.owner_id
        assert registry.bridge_for_domain(domain) is bridge
        assert bridge.source_module_path == source_path
        assert bridge.source_disposition == disposition
        prefix = f"| `{source_path}` | `{disposition}` |"
        rows = [line for line in document.splitlines() if line.startswith(prefix)]
        assert rows
        assert any("| `NO` |" in row for row in rows)


def test_every_bridge_is_disconnected_disabled_and_authority_free():
    for bridge in DEFAULT_BRIDGE_REGISTRY.bridges:
        assert bridge.schema_version == BRIDGE_SCHEMA_VERSION
        assert bridge.lifecycle_status == DISCONNECTED_STATUS
        assert bridge.integration_mode == DISCONNECTED_MODE
        assert bridge.default_enabled is False
        assert bridge.authority == NO_AUTHORITY
        assert bridge.emitted_event_types == ()
        assert bridge.required_capabilities == ()
        assert bridge.persistence_mode == NO_PERSISTENCE
        assert bridge.retry_policy == RETRY_FORBIDDEN
        assert bridge.suppression_policy == SUPPRESSION_FORBIDDEN
        assert bridge.rollback_scope == SHADOW_ROLLBACK_ONLY
        assert bridge.future_activation_evidence == REQUIRED_ACTIVATION_EVIDENCE


def test_every_owner_has_complete_non_authoritative_responsibilities():
    for owner in DEFAULT_BRIDGE_REGISTRY.owners:
        assert owner.schema_version == OWNER_SCHEMA_VERSION
        assert owner.authority == NO_AUTHORITY
        assert owner.initialization_responsibility == "explicit_caller_construction_only"
        assert owner.shutdown_responsibility == "release_shadow_resources_only"
        assert owner.interruption_responsibility == "cancel_shadow_work_preserve_legacy"
        assert owner.failure_responsibility == PROPAGATE_ORIGINAL
        assert owner.provenance_responsibility == "retain_source_and_contract_versions"
        assert owner.rollback_responsibility == "restore_registered_shadow_state_only"


def test_registry_and_contract_digests_are_deterministic_and_detached():
    registry = DEFAULT_BRIDGE_REGISTRY
    assert registry.digest == registry.digest
    assert len(registry.digest) == 64
    assert all(len(bridge.digest) == 64 for bridge in registry.bridges)
    detached = registry.canonical_record
    detached["bridges"][0]["default_enabled"] = True
    assert registry.canonical_record["bridges"][0]["default_enabled"] is False
    with pytest.raises(FrozenInstanceError):
        registry.schema_version = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registry.bridges[0].default_enabled = True  # type: ignore[misc]


def test_future_activation_evidence_is_exact_and_immutable():
    assert REQUIRED_ACTIVATION_EVIDENCE == (
        "explicit_reviewed_integration_pr",
        "initialization_and_shutdown_implementation",
        "failure_propagation_and_interruption_tests",
        "shadow_rollback_implementation",
        "exact_head_validation_green",
    )
    assert isinstance(REQUIRED_ACTIVATION_EVIDENCE, tuple)
    assert all(
        bridge.future_activation_evidence is REQUIRED_ACTIVATION_EVIDENCE
        for bridge in DEFAULT_BRIDGE_REGISTRY.bridges
    )


def test_unknown_registry_lookups_fail_closed():
    with pytest.raises(UnknownShadowBridge):
        DEFAULT_BRIDGE_REGISTRY.owner("m1.lifecycle.unknown")
    with pytest.raises(UnknownShadowBridge):
        DEFAULT_BRIDGE_REGISTRY.bridge("m1.bridge.unknown")
    with pytest.raises(UnknownShadowBridge):
        DEFAULT_BRIDGE_REGISTRY.bridge_for_domain("unknown")


def test_owner_and_bridge_reject_authority_activation_and_bad_paths():
    with pytest.raises(ShadowLifecycleContractError):
        LifecycleOwnerContract(
            owner_id="m1.lifecycle.chat", domain="chat", authority="runtime"
        )
    with pytest.raises(ShadowLifecycleContractError):
        LifecycleOwnerContract(owner_id="m1.lifecycle.memory", domain="goal")
    with pytest.raises(ShadowLifecycleContractError):
        ShadowBridgeContract(
            bridge_id="m1.bridge.chat",
            domain="chat",
            owner_id="m1.lifecycle.chat",
            source_module_path="language/streaming.py",
            source_disposition="REWRITE",
            default_enabled=True,
        )
    with pytest.raises(ShadowLifecycleContractError):
        ShadowBridgeContract(
            bridge_id="m1.bridge.chat",
            domain="chat",
            owner_id="m1.lifecycle.chat",
            source_module_path="language/streaming.py",
            source_disposition="REWRITE",
            emitted_event_types=("shadow.chat",),
        )
    with pytest.raises(ShadowLifecycleContractError):
        ShadowBridgeContract(
            bridge_id="m1.bridge.memory",
            domain="memory",
            owner_id="m1.lifecycle.memory",
            source_module_path="../memory.py",
            source_disposition="WRAP",
        )


def test_registry_rejects_missing_duplicate_and_cross_domain_owner():
    owners = DEFAULT_BRIDGE_REGISTRY.owners
    bridges = DEFAULT_BRIDGE_REGISTRY.bridges
    with pytest.raises(ShadowLifecycleContractError, match="exactly"):
        ShadowBridgeRegistry(owners=owners[:-1], bridges=bridges[:-1])
    with pytest.raises(ShadowLifecycleContractError, match="duplicate"):
        ShadowBridgeRegistry(owners=owners + (owners[0],), bridges=bridges)
    with pytest.raises(ShadowLifecycleContractError):
        replace(bridges[0], owner_id="m1.lifecycle.chat")


def test_failure_signal_is_redacted_and_preserves_propagation_policy():
    signal = BridgeFailureSignal.capture(
        DEFAULT_BRIDGE_REGISTRY,
        bridge_id="m1.bridge.memory",
        stage="initialization",
        error=RuntimeError("private bridge message"),
    )
    assert signal.schema_version == FAILURE_SCHEMA_VERSION
    assert signal.bridge_id == "m1.bridge.memory"
    assert signal.owner_id == "m1.lifecycle.memory"
    assert signal.domain == "memory"
    assert signal.error_type == "RuntimeError"
    assert signal.error_message_digest == hashlib.sha256(
        b"private bridge message"
    ).hexdigest()
    assert signal.handling == PROPAGATE_ORIGINAL
    assert signal.retry_allowed is False
    assert signal.suppression_allowed is False
    assert signal.legacy_authority_changed is False
    assert "private bridge message" not in repr(signal)


def test_failure_signal_rejects_unknown_bad_stage_and_nonexception():
    with pytest.raises(UnknownShadowBridge):
        BridgeFailureSignal.capture(
            DEFAULT_BRIDGE_REGISTRY,
            bridge_id="m1.bridge.unknown",
            stage="startup",
            error=RuntimeError("failure"),
        )
    with pytest.raises(ShadowLifecycleContractError):
        BridgeFailureSignal.capture(
            DEFAULT_BRIDGE_REGISTRY,
            bridge_id="m1.bridge.chat",
            stage="Bad Stage",
            error=RuntimeError("failure"),
        )
    with pytest.raises(ShadowLifecycleContractError):
        BridgeFailureSignal.capture(
            DEFAULT_BRIDGE_REGISTRY,
            bridge_id="m1.bridge.chat",
            stage="startup",
            error="not an exception",  # type: ignore[arg-type]
        )


def test_failure_signal_cannot_enable_retry_suppression_or_authority_change():
    signal = BridgeFailureSignal.capture(
        DEFAULT_BRIDGE_REGISTRY,
        bridge_id="m1.bridge.goal",
        stage="shutdown",
        error=ValueError("visible"),
    )
    for field, value in (
        ("retry_allowed", True),
        ("suppression_allowed", True),
        ("legacy_authority_changed", True),
        ("handling", "swallow"),
    ):
        with pytest.raises(ShadowLifecycleContractError):
            replace(signal, **{field: value})


def test_module_has_no_runtime_import_io_clock_thread_or_activation_surface():
    tree = ast.parse(LIFECYCLE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert not imported_roots & {
        "adapters", "asyncio", "datetime", "language", "main", "pathlib",
        "pickle", "random", "secrets", "sqlite3", "threading", "time", "uuid",
    }
    assert not called_names & {
        "connect", "emit", "load", "observe_call", "open", "save", "sleep",
        "start", "write_bytes", "write_text",
    }
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
