import json
import sys

from adapters.cross_modal_binding_preflight_schema import (
    build_cross_modal_binding_preflight,
    validate_cross_modal_binding_preflight,
    build_cross_modal_binding_preflight_schema_summary
)

def main():
    report = {
        "summary": build_cross_modal_binding_preflight_schema_summary(),
        "tests": []
    }

    # Test 1: Empty candidate
    preflight = build_cross_modal_binding_preflight({}, ["temporal_alignment_hypothesis"])
    report["tests"].append({
        "name": "empty_candidate_blocked",
        "passed": preflight["preflight_passed"] is False and preflight["preflight_decision"] == "blocked_unknown_candidate"
    })

    # Test 2: Identity Resolution Required
    valid_candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(valid_candidate, ["temporal_alignment_hypothesis"], {"requires_identity_resolution": True})
    report["tests"].append({
        "name": "identity_resolution_blocked",
        "passed": preflight["preflight_passed"] is False and preflight["preflight_decision"] == "blocked_identity_resolution_required"
    })

    # Test 3: Face/Voice Matching
    preflight = build_cross_modal_binding_preflight(valid_candidate, ["audio_visual_context_hypothesis"], {"attempt_face_voice_match": True})
    report["tests"].append({
        "name": "face_voice_match_blocked",
        "passed": preflight["preflight_passed"] is False and preflight["preflight_decision"] == "blocked_identity_resolution_required"
    })

    # Test 4: Mixed Boundary
    mixed_candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "mixed_boundary_candidate"}
    preflight = build_cross_modal_binding_preflight(mixed_candidate, ["temporal_alignment_hypothesis"])
    report["tests"].append({
        "name": "mixed_boundary_blocked",
        "passed": preflight["preflight_passed"] is False and preflight["preflight_decision"] == "blocked_mixed_boundary"
    })

    # Test 5: Eligible for future binding
    preflight = build_cross_modal_binding_preflight(valid_candidate, ["temporal_alignment_hypothesis"])
    report["tests"].append({
        "name": "eligible_for_future_binding",
        "passed": preflight["preflight_passed"] is True and preflight["preflight_decision"] == "eligible_for_future_binding" and preflight["future_binding_attempt_allowed"] is True
    })

    # Test 6: Validation checks
    valid = validate_cross_modal_binding_preflight(preflight)
    report["tests"].append({
        "name": "validation_check",
        "passed": valid["validation_passed"] is True
    })

    # Output compact JSON to stdout only
    print(json.dumps(report))

if __name__ == "__main__":
    main()
