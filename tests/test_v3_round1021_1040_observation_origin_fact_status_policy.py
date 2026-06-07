from __future__ import annotations

import pytest

from adapters.observation_origin_fact_status_policy import (
    FALSE_FLAGS,
    SUPPORTED_FACT_STATUSES,
    SUPPORTED_ORIGINS,
    build_origin_fact_status_record,
    validate_origin_fact_status_record,
)


@pytest.mark.parametrize("origin", SUPPORTED_ORIGINS)
def test_valid_origin_with_correct_fact_status(origin):
    # Mapping valid statuses for the origins to test generic valid combinations
    mapping = {
        "external_visual": "observed_external",
        "external_audio": "observed_external",
        "screen_visual": "observed_external",
        "tool_state": "observed_external",
        "internal_state": "observed_external",  # Valid for the record, but isn't external asserted
        "virtual_world_visual": "observed_internal_virtual",
        "memory_replay": "reconstructed_memory",
        "imagination": "imagined_candidate",
        "simulation": "simulated_future",
        "dream_dmn": "symbolic_visualization",
    }

    fact_status = mapping[origin]

    # internal_state observed_external is technically blocked by external_fact_assertion_guard
    # except it is not in the whitelist for observed_external origin.
    # We will adjust mapping specifically to pass the strict guard.
    if origin == "internal_state":
        # Internal state with observed_external fails external_fact_assertion_guard
        # So we skip this exact generic test for internal_state with observed_external.
        # But we can test it fails below.
        pass
    else:
        record = build_origin_fact_status_record(origin, fact_status)
        assert record["origin_fact_status_passed"] is True
        assert validate_origin_fact_status_record(record)["validation_passed"] is True


def test_internal_state_observed_external_fails_guard():
    record = build_origin_fact_status_record("internal_state", "observed_external")
    assert record["origin_fact_status_passed"] is False
    assert "external_fact_assertion_guard_failed_fact_status_mismatch" in record["blocked_reasons"]


def test_external_fact_assertion_guard_blocks_internal_origins():
    internal_origins = ["virtual_world_visual", "memory_replay", "imagination", "simulation", "dream_dmn"]
    for origin in internal_origins:
        record = build_origin_fact_status_record(origin, "observed_external")
        assert record["origin_fact_status_passed"] is False
        assert "external_fact_assertion_guard_failed_internal_origin_cannot_be_external_fact" in record["blocked_reasons"]
        assert record["external_fact_asserted"] is False


@pytest.mark.parametrize("origin,expected_fact_status,guard_error", [
    ("virtual_world_visual", "observed_internal_virtual", "virtual_world_reality_boundary_guard_failed"),
    ("memory_replay", "reconstructed_memory", "memory_replay_provenance_guard_failed"),
    ("imagination", "imagined_candidate", "imagination_memory_contamination_guard_failed"),
    ("simulation", "simulated_future", "simulation_result_boundary_guard_failed"),
    ("dream_dmn", "symbolic_visualization", "dream_dmn_boundary_guard_failed"),
])
def test_specific_boundary_guards(origin, expected_fact_status, guard_error):
    # Valid combination
    valid_record = build_origin_fact_status_record(origin, expected_fact_status)
    assert valid_record["origin_fact_status_passed"] is True

    # Invalid combination to trigger the guard
    invalid_record = build_origin_fact_status_record(origin, "invalid_status_for_test")
    assert invalid_record["origin_fact_status_passed"] is False
    assert guard_error in invalid_record["blocked_reasons"]


def test_unsupported_origins_and_fact_statuses_fail_closed():
    record = build_origin_fact_status_record("unknown_origin", "unknown_status")
    assert record["origin_fact_status_passed"] is False
    assert "unsupported_origin_unknown_origin" in record["blocked_reasons"]
    assert "unsupported_fact_status_unknown_status" in record["blocked_reasons"]


@pytest.mark.parametrize("flag", FALSE_FLAGS.keys())
def test_hard_false_flags_are_always_false(flag):
    record = build_origin_fact_status_record("external_visual", "observed_external")
    assert record[flag] is False

    record_internal = build_origin_fact_status_record("imagination", "imagined_candidate")
    assert record_internal[flag] is False


def test_external_fact_asserted_is_only_true_for_permitted_combinations():
    valid1 = build_origin_fact_status_record("external_visual", "observed_external")
    valid2 = build_origin_fact_status_record("external_audio", "observed_external")
    valid3 = build_origin_fact_status_record("screen_visual", "observed_external")

    assert valid1["external_fact_asserted"] is True
    assert valid2["external_fact_asserted"] is True
    assert valid3["external_fact_asserted"] is True

    # Tool state can be observed_external but does not assert an external sensory fact by default
    tool_state = build_origin_fact_status_record("tool_state", "observed_external")
    assert tool_state["external_fact_asserted"] is False

    invalid1 = build_origin_fact_status_record("virtual_world_visual", "observed_internal_virtual")
    assert invalid1["external_fact_asserted"] is False
