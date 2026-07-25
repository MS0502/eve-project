from __future__ import annotations

import pytest

from core.m3_b_registry_production_capture_adapter import (
    REGISTERED_PRODUCTION_SOURCE_VERIFIERS,
    ProductionSourceVerifierRegistration,
)


def test_repository_production_verifier_registry_is_empty_and_runtime_immutable():
    assert len(REGISTERED_PRODUCTION_SOURCE_VERIFIERS) == 0
    with pytest.raises(TypeError):
        REGISTERED_PRODUCTION_SOURCE_VERIFIERS[
            "eve:m3-b:registry-source:prediction_error_pressure:v1"
        ] = ProductionSourceVerifierRegistration(  # type: ignore[index]
            source_contract_id="eve:m3-b:registry-source:prediction_error_pressure:v1",
            verifier_id="forbidden.runtime.injected-verifier",
            verifier_version="v1",
            verifier=lambda evidence, source_material: None,  # type: ignore[arg-type,return-value]
        )
