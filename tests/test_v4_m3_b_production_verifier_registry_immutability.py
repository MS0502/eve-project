from __future__ import annotations

from types import MappingProxyType

from core.m3_b_registry_production_capture_adapter import (
    REGISTERED_PRODUCTION_SOURCE_VERIFIERS,
)


def test_repository_production_verifier_registry_is_empty_and_runtime_immutable():
    assert type(REGISTERED_PRODUCTION_SOURCE_VERIFIERS) is MappingProxyType
    assert len(REGISTERED_PRODUCTION_SOURCE_VERIFIERS) == 0
    assert not hasattr(REGISTERED_PRODUCTION_SOURCE_VERIFIERS, "__setitem__")
