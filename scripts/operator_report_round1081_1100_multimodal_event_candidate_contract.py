import json
import sys

from adapters.multimodal_event_candidate_contract import (
    build_multimodal_event_candidate,
    validate_multimodal_event_candidate,
    build_multimodal_event_candidate_contract_summary,
    build_multimodal_event_to_origin_fact_status_plan,
    build_multimodal_event_to_memory_candidate_plan,
    build_multimodal_event_to_appraisal_plan,
    build_multimodal_event_to_agp_input_plan
)

def main():
    report = {
        "summary": build_multimodal_event_candidate_contract_summary(),
        "tests": []
    }

    # Test 1: Empty observations
    candidate = build_multimodal_event_candidate([])
    report["tests"].append({
        "name": "empty_observations",
        "passed": candidate["multimodal_event_candidate_passed"] is False and "empty_observations" in candidate["blocked_reasons"]
    })

    # Test 2: External Visual + Auditory -> external_event_candidate
    candidate = build_multimodal_event_candidate([
        {"modality": "visual_external", "source": "camera"},
        {"modality": "auditory", "source": "microphone"}
    ])
    report["tests"].append({
        "name": "external_visual_and_auditory",
        "passed": candidate["multimodal_event_candidate_passed"] is True and candidate["candidate_type"] == "external_event_candidate"
    })

    # Test 3: Virtual Visual + DMN Inner Voice -> dmn_symbolic_event_candidate
    candidate = build_multimodal_event_candidate([
        {"modality": "visual_internal_virtual", "source": "virtual_room_view"},
        {"modality": "auditory", "source": "dmn_inner_voice_candidate"}
    ])
    report["tests"].append({
        "name": "virtual_visual_and_dmn",
        "passed": candidate["multimodal_event_candidate_passed"] is True and candidate["candidate_type"] == "dmn_symbolic_event_candidate"
    })

    # Test 4: Virtual Visual + External Auditory -> mixed_boundary_candidate
    candidate = build_multimodal_event_candidate([
        {"modality": "visual_internal_virtual", "source": "virtual_room_view"},
        {"modality": "auditory", "source": "microphone"}
    ])
    report["tests"].append({
        "name": "virtual_visual_and_external_auditory_mixed_boundary",
        "passed": candidate["multimodal_event_candidate_passed"] is True and candidate["candidate_type"] == "mixed_boundary_candidate" and "mixed_boundary_uncertainty" in candidate["uncertainty_flags"]
    })

    # Test 5: Validation checks
    valid = validate_multimodal_event_candidate(candidate)
    report["tests"].append({
        "name": "validation_check",
        "passed": valid["validation_passed"] is True
    })

    # Output compact JSON to stdout only
    print(json.dumps(report))

if __name__ == "__main__":
    main()
