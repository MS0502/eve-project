import pytest

from adapters.agp_adapter import (
    AGPAdapter,
    AGPResult,
    AGP_FALLBACK_HORMONE_MISMATCH,
    AGP_FALLBACK_UNKNOWN,
    AGP_FALLBACK_UNKNOWN_CATEGORY,
    AGP_REASON_ANCHORED,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
)


def test_v3_round2_agp_result_dataclass_shape():
    result = AGPResult(
        passed=True,
        reason=AGP_REASON_ANCHORED,
        fallback=None,
        activated_categories=["greeting", "minseok"],
        candidate_categories=["greeting"],
        hormone_distance=0.12,
    )

    assert result.passed is True
    assert result.reason == "anchored"
    assert result.fallback is None
    assert result.activated_categories == ["greeting", "minseok"]
    assert result.candidate_categories == ["greeting"]
    assert result.hormone_distance == pytest.approx(0.12)


def test_v3_round2_agp_adapter_initializes_without_runtime_integration():
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)

    assert adapter.sa_adapter is None
    assert adapter.hormone_adapter is None
    assert adapter.category_threshold == pytest.approx(0.3)
    assert adapter.hormone_threshold == pytest.approx(0.5)


def test_v3_round2_agp_verify_fails_closed_without_runtime_integration():
    adapter = AGPAdapter()

    result = adapter.verify("민석아, 이건 아직 검증 전이야.", meaning=None)

    assert result.passed is False
    assert result.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert result.fallback == AGP_FALLBACK_UNKNOWN_CATEGORY
    assert result.candidate_categories == []


def test_v3_round2_agp_fallback_constants_are_honest_not_plausible_generation():
    adapter = AGPAdapter()

    assert adapter._generate_fallback(AGP_REASON_UNKNOWN_CATEGORY) == AGP_FALLBACK_UNKNOWN_CATEGORY
    assert adapter._generate_fallback(AGP_REASON_HORMONE_MISMATCH) == AGP_FALLBACK_HORMONE_MISMATCH
    assert adapter._generate_fallback("other") == AGP_FALLBACK_UNKNOWN

    # AGP must fall back to honest uncertainty rather than generating a fake answer.
    assert "모르" in AGP_FALLBACK_UNKNOWN
    assert "뭐" in AGP_FALLBACK_UNKNOWN_CATEGORY
    assert AGP_REASON_UNKNOWN_CATEGORY == "unknown_category"
    assert AGP_REASON_HORMONE_MISMATCH == "hormone_mismatch"
