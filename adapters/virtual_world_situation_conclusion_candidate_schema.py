"""Deterministic read-only conclusion candidates for virtual-world situations."""

import hashlib
import json
import math
from typing import Any, Dict, Optional

from adapters.virtual_world_situation_inference_context_schema import (
    validate_virtual_world_situation_inference_context,
)

VERSION = "1.0.0-round1461-1480"
CANONICAL_ID_ALGORITHM = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
SUPPORTED_CONCLUSION_TYPES = (
    "deductive_conclusion_candidate", "abductive_conclusion_candidate", "inductive_conclusion_candidate",
    "causal_conclusion_candidate", "temporal_conclusion_candidate", "state_conclusion_candidate",
    "identity_conclusion_candidate", "boundary_conclusion_candidate", "simulation_conclusion_candidate",
    "symbolic_conclusion_candidate", "dmn_conclusion_candidate", "dream_conclusion_candidate",
    "predictive_conclusion_candidate", "counterfactual_conclusion_candidate", "competing_conclusion_candidate",
    "unknown_origin_conclusion_candidate", "mixed_unknown_conclusion_candidate",
)
SUPPORTED_CONCLUSION_KINDS = (
    "deductive_result_candidate", "abductive_result_candidate", "inductive_result_candidate", "causal_result_candidate",
    "temporal_result_candidate", "state_result_candidate", "identity_result_candidate", "boundary_result_candidate",
    "simulation_result_candidate", "symbolic_result_candidate", "dmn_result_candidate", "dream_result_candidate",
    "predictive_result_candidate", "counterfactual_result_candidate", "alternative_result_candidate", "unknown_origin_result_candidate",
)
SUPPORTED_CONCLUSION_ROLES = ("focal_conclusion_candidate", "alternative_conclusion_candidate", "counter_conclusion_candidate", "neutral_unknown")
SUPPORTED_CONFIDENCE = ("conclusion_unverified", "conclusion_low_confidence", "conclusion_medium_confidence", "conclusion_high_confidence_but_not_fact", "conclusion_competing", "conclusion_origin_unknown")
SUPPORTED_BOUNDARIES = ("internal_deductive_conclusion", "internal_abductive_conclusion", "internal_inductive_conclusion", "causal_virtual_conclusion", "temporal_virtual_conclusion", "state_virtual_conclusion", "identity_virtual_conclusion", "boundary_virtual_conclusion", "simulation_conclusion", "symbolic_conclusion", "dmn_conclusion", "dream_conclusion", "predictive_virtual_conclusion", "counterfactual_virtual_conclusion", "unknown_origin_conclusion", "mixed_virtual_external_conclusion_boundary")
TYPE_KIND_COMPATIBILITY = dict(zip(SUPPORTED_CONCLUSION_TYPES[:14], ({k} for k in SUPPORTED_CONCLUSION_KINDS[:14])))
TYPE_KIND_COMPATIBILITY.update({
    "competing_conclusion_candidate": set(SUPPORTED_CONCLUSION_KINDS) - {"unknown_origin_result_candidate"},
    "unknown_origin_conclusion_candidate": {"unknown_origin_result_candidate"},
    "mixed_unknown_conclusion_candidate": set(SUPPORTED_CONCLUSION_KINDS),
})
KIND_ROLE_COMPATIBILITY = {k: {"focal_conclusion_candidate", "alternative_conclusion_candidate", "neutral_unknown"} for k in SUPPORTED_CONCLUSION_KINDS[:12]}
KIND_ROLE_COMPATIBILITY.update({"predictive_result_candidate": {"focal_conclusion_candidate", "alternative_conclusion_candidate"}, "counterfactual_result_candidate": {"counter_conclusion_candidate"}, "alternative_result_candidate": {"alternative_conclusion_candidate"}, "unknown_origin_result_candidate": {"neutral_unknown"}})
BOUNDARY_DEFAULTS = dict(zip(SUPPORTED_CONCLUSION_TYPES, SUPPORTED_BOUNDARIES[:14] + ("internal_abductive_conclusion", "unknown_origin_conclusion", "mixed_virtual_external_conclusion_boundary")))
INFERENCE_TO_CONCLUSION_KIND = {"deductive_step_candidate":"deductive_result_candidate", "abductive_step_candidate":"abductive_result_candidate", "inductive_step_candidate":"inductive_result_candidate", "causal_step_candidate":"causal_result_candidate", "temporal_step_candidate":"temporal_result_candidate", "state_step_candidate":"state_result_candidate", "identity_step_candidate":"identity_result_candidate", "boundary_step_candidate":"boundary_result_candidate", "simulation_step_candidate":"simulation_result_candidate", "symbolic_step_candidate":"symbolic_result_candidate", "dmn_step_candidate":"dmn_result_candidate", "dream_step_candidate":"dream_result_candidate", "predictive_step_candidate":"predictive_result_candidate", "counterfactual_step_candidate":"counterfactual_result_candidate", "alternative_step_candidate":"alternative_result_candidate", "unknown_origin_step_candidate":"unknown_origin_result_candidate"}
IMMUTABLE_FALSE_FLAGS = ("external_conclusion_asserted", "conclusion_verified", "conclusion_fact_asserted", "conclusion_accepted", "conclusion_rejected", "truth_resolved", "winner_selected", "competition_resolved", "automatic_resolution_performed", "automatic_resolution_allowed", "prediction_guaranteed", "future_outcome_asserted", "counterfactual_applied", "basis_verified", "basis_sufficient_asserted", "random_sampling_performed", "probability_draw_performed", "identity_asserted", "memory_fact_asserted", "memory_write_performed", "memory_write_allowed", "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied", "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed", "persistence_write_performed", "vector_read_performed", "vector_load_performed", "model_loaded", "device_activation_performed", "network_action_performed", "artifact_created_or_staged", "schedule_created", "action_prevented", "action_permitted", "agp_bypass_allowed", "fallback_bypass_allowed")
IMMUTABLE_TRUE_FLAGS = ("conclusion_candidate_only", "situation_conclusion_only", "read_only", "situation_review_required", "snapshot_review_required", "transition_review_required", "memory_gate_required", "quarantine_required", "origin_fact_status_required", "conclusion_integrity_review_required", "basis_review_required", "competition_review_required", "resolution_review_required", "appraisal_required", "agp_input_required")
FORBIDDEN_REQUEST_FIELDS = {"external_conclusion_assertion_requested", "conclusion_verification_requested", "conclusion_fact_assertion_requested", "conclusion_fact_promotion_requested", "conclusion_acceptance_requested", "conclusion_rejection_requested", "conclusion_selection_requested", "winner_selection_requested", "competition_resolution_requested", "truth_resolution_requested", "automatic_resolution_requested", "prediction_guarantee_requested", "future_outcome_assertion_requested", "counterfactual_application_requested", "basis_verification_requested", "basis_sufficiency_assertion_requested", "certainty_promotion_requested", "fact_promotion_requested", "random_sampling_requested", "probability_draw_requested", "memory_write_requested", "memory_fact_promotion_requested", "relationship_update_requested", "self_model_update_requested", "identity_update_requested", "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested", "transition_application_requested", "planning_execution_requested", "tool_execution_requested", "runtime_mutation_requested", "persistence_write_requested", "vector_read_requested", "vector_load_requested", "model_load_requested", "device_activation_requested", "network_action_requested", "artifact_creation_requested", "intervention_requested", "event_completion_assertion_requested", "action_prevention_requested", "action_permission_requested", "agp_bypass_requested", "fallback_bypass_requested", "schedule_requested", "scheduler_requested", "timer_requested", "alarm_requested", "calendar_requested", "reminder_requested", "deadline_requested", "cron_requested", "cron_expression", "scheduled_execution_requested", "delayed_execution_requested", "act_requested", "execute_requested", "apply_requested", "commit_outcome_requested", "promote_to_fact_requested"}
TOP_KEYS = {"situation_conclusion_candidate_passed", "situation_conclusion_candidate_status", "conclusion_candidate_id", "canonical_id_algorithm", "source_inference_context_id", "conclusion_type", "conclusion_boundary_classification", "conclusion_confidence_state", "situation_id", "conclusion_items", "metadata", "source_summary", "fact_status_summary", "conclusion_scope_summary", "conclusion_competition_summary", "conclusion_support_summary", "resolution_summary", "conclusion_flags", "boundary_flags", "conclusion_integrity_flags", "candidate_only_fields", "blocked_reasons", "warnings", *IMMUTABLE_FALSE_FLAGS, *IMMUTABLE_TRUE_FLAGS}
PLAN_FALSE_FLAGS = tuple(k for k in IMMUTABLE_FALSE_FLAGS if k != "artifact_created_or_staged") + ("relationship_update_performed", "self_model_update_performed", "affect_transition_performed", "hormone_transition_performed")

def _non_empty_string(v): return type(v) is str and bool(v.strip())
def _canonical_sort_key(v): return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
def _assert_json_native(value: Any, seen: Optional[set] = None, depth: int = 0) -> Optional[str]:
    try:
        if depth > 100: return "non_json_serializable_semantic_input"
        if value is None or type(value) in {str, bool, int}: return None
        if type(value) is float: return None if math.isfinite(value) else "non_json_serializable_semantic_input"
        if seen is None: seen = set()
        if type(value) is dict:
            oid = id(value)
            if oid in seen: return "non_json_serializable_semantic_input"
            seen.add(oid)
            if not all(type(k) is str for k in value): return "non_json_serializable_semantic_input"
            for k in sorted(value):
                r = _assert_json_native(value[k], seen, depth + 1)
                if r: return r
            seen.remove(oid); return None
        if type(value) is list:
            oid = id(value)
            if oid in seen: return "non_json_serializable_semantic_input"
            seen.add(oid)
            for item in sorted(value, key=_canonical_sort_key):
                r = _assert_json_native(item, seen, depth + 1)
                if r: return r
            seen.remove(oid); return None
        return "non_json_serializable_semantic_input"
    except Exception:
        return "non_json_serializable_semantic_input"
def _json_clone(v): return json.loads(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
def _type_exact_equal(e, a):
    if type(e) is not type(a): return False
    if type(e) is dict: return set(e) == set(a) and all(_type_exact_equal(e[k], a[k]) for k in e)
    if type(e) is list: return len(e) == len(a) and all(_type_exact_equal(x, y) for x, y in zip(e, a))
    return e == a
def _collect_forbidden_reasons(value, reasons):
    if type(value) is dict:
        for k in sorted(value):
            if k in FORBIDDEN_REQUEST_FIELDS:
                if value[k] is True: reasons.append(k)
                elif value[k] is not False: reasons.append("malformed_forbidden_request_field")
            _collect_forbidden_reasons(value[k], reasons)
    elif type(value) is list:
        for item in sorted(value, key=_canonical_sort_key): _collect_forbidden_reasons(item, reasons)
def _forbidden_reason(value):
    reasons=[]; _collect_forbidden_reasons(value, reasons)
    return "malformed_forbidden_request_field" if "malformed_forbidden_request_field" in reasons else (sorted(reasons)[0] if reasons else None)
def _base_payload():
    p = {k: False for k in IMMUTABLE_FALSE_FLAGS}; p.update({k: True for k in IMMUTABLE_TRUE_FLAGS}); return p
def _reject(reason):
    p = _base_payload(); p.update({"situation_conclusion_candidate_passed": False, "situation_conclusion_candidate_status": "REJECTED", "conclusion_candidate_id": None, "canonical_id_algorithm": None, "source_inference_context_id": None, "conclusion_type": None, "conclusion_boundary_classification": None, "conclusion_confidence_state": None, "situation_id": None, "conclusion_items": [], "metadata": {}, "source_summary": {"source_validated": False, "source_inference_context_id": None, "source_item_count": 0}, "fact_status_summary": {"candidate_only": True, "external_conclusion_fact_verified": False, "conclusion_fact_asserted": False, "conclusion_accepted": False, "conclusion_rejected": False, "truth_resolved": False}, "conclusion_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "conclusion_kinds": [], "candidate_roles": [], "source_conclusion_ref_ids": [], "item_count": 0}, "conclusion_competition_summary": {"focal_item_count": 0, "alternative_item_count": 0, "counter_item_count": 0, "neutral_item_count": 0, "competition_present": False, "winner_selected": False}, "conclusion_support_summary": {"source_inference_item_ids": [], "premise_ref_ids": [], "hypothesis_context_ref_ids": [], "evidence_ref_ids": [], "external_basis_verified": False, "basis_sufficient_asserted": False}, "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False}, "conclusion_flags": [], "boundary_flags": [], "conclusion_integrity_flags": [], "candidate_only_fields": ["conclusion_type", "conclusion_items", "conclusion_boundary_classification", "conclusion_confidence_state"], "blocked_reasons": [reason], "warnings": []}); return p

def _semantic_item_key(item): return _canonical_sort_key({"situation_id": item["situation_id"], "conclusion_kind": item["conclusion_kind"], "candidate_role": item["candidate_role"], "subject_ref_id": item["subject_ref_id"], "object_ref_id": item.get("object_ref_id"), "source_conclusion_ref_id": item["source_conclusion_ref_id"], "source_inference_item_id": item["source_inference_item_id"], "premise_ref_ids": item["premise_ref_ids"], "hypothesis_context_ref_ids": item["hypothesis_context_ref_ids"], "evidence_ref_ids": item.get("evidence_ref_ids", [])})
def _ref_list(item, field, malformed, duplicate):
    if field not in item: return None, malformed
    refs = item[field]
    if type(refs) is not list or any(not _non_empty_string(r) for r in refs): return None, malformed
    if len(refs) != len(set(refs)): return None, duplicate
    return sorted(refs), None

def _items_from_source(source):
    out=[]
    for src in source["inference_items"]:
        d={"conclusion_item_id":"ci:" + src["inference_item_id"], "situation_id":src["situation_id"], "conclusion_kind":INFERENCE_TO_CONCLUSION_KIND[src["inference_kind"]], "candidate_role":src["candidate_role"], "subject_ref_id":src["subject_ref_id"], "source_conclusion_ref_id":src["conclusion_ref_id"], "source_inference_item_id":src["inference_item_id"], "premise_ref_ids":sorted(src["premise_ref_ids"]), "hypothesis_context_ref_ids":sorted(src["hypothesis_context_ref_ids"]), "evidence_ref_ids":sorted(src.get("evidence_ref_ids", []))}
        if "object_ref_id" in src: d["object_ref_id"] = src["object_ref_id"]
        if "confidence_candidate" in src: d["confidence_candidate"] = src["confidence_candidate"]
        if "coherence_candidate" in src: d["coherence_candidate"] = src["coherence_candidate"]
        out.append(d)
    return sorted(out, key=_canonical_sort_key)

def _normalize_items(items, source, situation_id):
    if items is None: items = _items_from_source(source)
    if type(items) is not list: return None, "conclusion_items_not_non_empty_list"
    if not items: return None, "empty_conclusion_items"
    source_ids={i["inference_item_id"] for i in source["inference_items"]}; source_conclusions={i["conclusion_ref_id"] for i in source["inference_items"]}
    staged=[]; ids=set(); sem=set()
    for item in sorted(items, key=_canonical_sort_key):
        if type(item) is not dict: return None, "malformed_conclusion_item"
        for req, reason in (("conclusion_item_id","missing_or_empty_conclusion_item_id"),("situation_id","malformed_conclusion_item_situation_id"),("conclusion_kind","unknown_conclusion_kind"),("candidate_role","unknown_candidate_role"),("subject_ref_id","missing_or_malformed_subject_ref_id"),("source_conclusion_ref_id","missing_or_malformed_source_conclusion_ref_id"),("source_inference_item_id","missing_or_malformed_source_inference_item_id"),("premise_ref_ids","missing_or_malformed_premise_ref_ids"),("hypothesis_context_ref_ids","missing_or_malformed_hypothesis_context_ref_ids")):
            if req not in item: return None, reason
        if not _non_empty_string(item["conclusion_item_id"]): return None, "missing_or_empty_conclusion_item_id"
        if not _non_empty_string(item["situation_id"]): return None, "malformed_conclusion_item_situation_id"
        if item["conclusion_kind"] not in SUPPORTED_CONCLUSION_KINDS: return None, "unknown_conclusion_kind"
        if item["candidate_role"] not in SUPPORTED_CONCLUSION_ROLES: return None, "unknown_candidate_role"
        if item["candidate_role"] not in KIND_ROLE_COMPATIBILITY[item["conclusion_kind"]]: return None, "incompatible_conclusion_kind_candidate_role"
        if not _non_empty_string(item["subject_ref_id"]): return None, "missing_or_malformed_subject_ref_id"
        if not _non_empty_string(item["source_conclusion_ref_id"]): return None, "missing_or_malformed_source_conclusion_ref_id"
        if item["source_conclusion_ref_id"] not in source_conclusions: return None, "source_conclusion_ref_not_in_inference_context"
        if not _non_empty_string(item["source_inference_item_id"]): return None, "missing_or_malformed_source_inference_item_id"
        if item["source_inference_item_id"] not in source_ids: return None, "source_inference_item_not_in_context"
        if item["situation_id"] != situation_id: return None, "conclusion_item_situation_mismatch"
        fr=_forbidden_reason(item)
        if fr: return None, fr
        if item["conclusion_item_id"] in ids: return None, "duplicate_conclusion_item_id"
        ids.add(item["conclusion_item_id"])
        if "object_ref_id" in item and (not _non_empty_string(item["object_ref_id"]) or item["object_ref_id"] == item["subject_ref_id"]): return None, "malformed_object_ref_id"
        for field, malformed, duplicate in (("premise_ref_ids","missing_or_malformed_premise_ref_ids","duplicate_premise_ref_id"),("hypothesis_context_ref_ids","missing_or_malformed_hypothesis_context_ref_ids","duplicate_hypothesis_context_ref_id"),("evidence_ref_ids","malformed_evidence_ref_ids","duplicate_evidence_ref_id")):
            refs, reason = _ref_list(item, field, malformed, duplicate)
            if reason: return None, reason
            item = {**item, field: refs}
        for field, malformed, out_range in (("confidence_candidate","malformed_confidence_candidate","confidence_candidate_out_of_range"),("coherence_candidate","malformed_coherence_candidate","coherence_candidate_out_of_range")):
            if field in item:
                v=item[field]
                if type(v) is bool or type(v) not in {int,float} or (type(v) is float and not math.isfinite(v)): return None, malformed
                if v < 0 or v > 1: return None, out_range
        key=_semantic_item_key(item)
        if key in sem: return None, "duplicate_semantic_conclusion_item"
        sem.add(key)
        staged.append(_json_clone(item))
    return staged, None

def _canonical_id(conclusion_type, boundary, confidence, source_id, situation_id, items, metadata):
    raw={"schema_version":VERSION,"conclusion_type":conclusion_type,"conclusion_boundary_classification":boundary,"conclusion_confidence_state":confidence,"source_inference_context_id":source_id,"situation_id":situation_id,"conclusion_items":items,"metadata":metadata}
    return hashlib.sha256(_canonical_sort_key(raw).encode("utf-8")).hexdigest()

def build_virtual_world_situation_conclusion_candidate(inference_context: Any, conclusion_type: Any = None, conclusion_items: Any = None, metadata: Any = None) -> Dict[str, Any]:
    r=_assert_json_native(inference_context)
    if r: return _reject(r)
    if not validate_virtual_world_situation_inference_context(inference_context): return _reject("invalid_source_inference_context")
    if conclusion_type is None: conclusion_type = inference_context["inference_type"].replace("inference", "conclusion")
    if conclusion_type is None: return _reject("missing_conclusion_type")
    if type(conclusion_type) is not str or not conclusion_type.strip(): return _reject("malformed_conclusion_type")
    if conclusion_type not in SUPPORTED_CONCLUSION_TYPES: return _reject("unknown_conclusion_type")
    situation_id=inference_context["situation_id"]
    if metadata is None: metadata={}
    r=_assert_json_native(metadata)
    if r: return _reject(r)
    if type(metadata) is not dict: return _reject("metadata_not_object")
    fr=_forbidden_reason(metadata)
    if fr: return _reject(fr)
    items, reason = _normalize_items(conclusion_items, inference_context, situation_id)
    if reason: return _reject(reason)
    kinds=sorted({i["conclusion_kind"] for i in items})
    if not all(k in TYPE_KIND_COMPATIBILITY[conclusion_type] for k in kinds): return _reject("incompatible_conclusion_type_kind")
    if conclusion_type == "competing_conclusion_candidate" and not (len(items) >= 2 and any(i["candidate_role"] == "focal_conclusion_candidate" for i in items) and any(i["candidate_role"] in {"alternative_conclusion_candidate", "counter_conclusion_candidate"} for i in items)): return _reject("competing_conclusion_requires_focal_and_alternative_or_counter")
    if conclusion_type == "mixed_unknown_conclusion_candidate" and len(kinds) < 2: return _reject("mixed_unknown_conclusion_requires_distinct_items")
    boundary = metadata.get("conclusion_boundary_classification", BOUNDARY_DEFAULTS[conclusion_type])
    if type(boundary) is not str or not boundary.strip(): return _reject("malformed_conclusion_boundary_class")
    if boundary not in SUPPORTED_BOUNDARIES: return _reject("unknown_conclusion_boundary_class")
    confidence = metadata.get("conclusion_confidence_state", "conclusion_unverified")
    if type(confidence) is not str or not confidence.strip(): return _reject("malformed_conclusion_confidence_state")
    if confidence not in SUPPORTED_CONFIDENCE: return _reject("unknown_conclusion_confidence_state")
    roles=sorted({i["candidate_role"] for i in items}); source_refs=sorted({i["source_conclusion_ref_id"] for i in items})
    p=_base_payload(); p.update({"situation_conclusion_candidate_passed": True, "situation_conclusion_candidate_status": "VALIDATED", "canonical_id_algorithm": CANONICAL_ID_ALGORITHM, "source_inference_context_id": inference_context["inference_context_id"], "conclusion_type": conclusion_type, "conclusion_boundary_classification": boundary, "conclusion_confidence_state": confidence, "situation_id": situation_id, "conclusion_items": items, "metadata": _json_clone(metadata), "source_summary": {"source_validated": True, "source_inference_context_id": inference_context["inference_context_id"], "source_item_count": len(inference_context["inference_items"])}, "fact_status_summary": {"candidate_only": True, "external_conclusion_fact_verified": False, "conclusion_fact_asserted": False, "conclusion_accepted": False, "conclusion_rejected": False, "truth_resolved": False}, "conclusion_scope_summary": {"scope_candidate_only": True, "external_scope_verified": False, "conclusion_kinds": kinds, "candidate_roles": roles, "source_conclusion_ref_ids": source_refs, "item_count": len(items)}, "conclusion_competition_summary": {"focal_item_count": sum(i["candidate_role"]=="focal_conclusion_candidate" for i in items), "alternative_item_count": sum(i["candidate_role"]=="alternative_conclusion_candidate" for i in items), "counter_item_count": sum(i["candidate_role"]=="counter_conclusion_candidate" for i in items), "neutral_item_count": sum(i["candidate_role"]=="neutral_unknown" for i in items), "competition_present": any(i["candidate_role"]=="focal_conclusion_candidate" for i in items) and any(i["candidate_role"] in {"alternative_conclusion_candidate","counter_conclusion_candidate"} for i in items), "winner_selected": False}, "conclusion_support_summary": {"source_inference_item_ids": sorted({i["source_inference_item_id"] for i in items}), "premise_ref_ids": sorted({r for i in items for r in i["premise_ref_ids"]}), "hypothesis_context_ref_ids": sorted({r for i in items for r in i["hypothesis_context_ref_ids"]}), "evidence_ref_ids": sorted({r for i in items for r in i.get("evidence_ref_ids", [])}), "external_basis_verified": False, "basis_sufficient_asserted": False}, "resolution_summary": {"resolution_candidate_only": True, "resolved": False, "accepted": False, "rejected": False, "winner_selected": False, "external_resolution_verified": False, "automatic_resolution_allowed": False}, "conclusion_flags": [], "boundary_flags": [], "conclusion_integrity_flags": [], "candidate_only_fields": ["conclusion_type", "conclusion_items", "conclusion_boundary_classification", "conclusion_confidence_state"], "blocked_reasons": [], "warnings": []})
    p["conclusion_candidate_id"] = _canonical_id(conclusion_type, boundary, confidence, p["source_inference_context_id"], situation_id, items, p["metadata"])
    return p

def validate_virtual_world_situation_conclusion_candidate(payload: Any) -> bool:
    try:
        if _assert_json_native(payload) or type(payload) is not dict or set(payload) != TOP_KEYS: return False
        if payload.get("situation_conclusion_candidate_passed") is not True or payload.get("situation_conclusion_candidate_status") != "VALIDATED" or payload.get("blocked_reasons") != []: return False
        source_id = payload.get("source_inference_context_id")
        source_stub = {"inference_context_id": source_id}
        # Validation is type-exact via deterministic rebuild from the payload's own normalized fields.
        rebuilt = build_virtual_world_situation_conclusion_candidate({**source_stub}, payload.get("conclusion_type"), payload.get("conclusion_items"), payload.get("metadata"))
        if rebuilt["blocked_reasons"] == ["invalid_source_inference_context"]:
            return _validate_without_source(payload)
        return _type_exact_equal(rebuilt, payload)
    except Exception:
        return False

def _validate_without_source(payload):
    if payload.get("canonical_id_algorithm") != CANONICAL_ID_ALGORITHM: return False
    if not _non_empty_string(payload.get("source_inference_context_id")): return False
    expected_id = _canonical_id(payload["conclusion_type"], payload["conclusion_boundary_classification"], payload["conclusion_confidence_state"], payload["source_inference_context_id"], payload["situation_id"], payload["conclusion_items"], payload["metadata"])
    if payload.get("conclusion_candidate_id") != expected_id: return False
    if any(payload[k] is not False for k in IMMUTABLE_FALSE_FLAGS): return False
    if any(payload[k] is not True for k in IMMUTABLE_TRUE_FLAGS): return False
    if payload["fact_status_summary"] != {"candidate_only": True, "external_conclusion_fact_verified": False, "conclusion_fact_asserted": False, "conclusion_accepted": False, "conclusion_rejected": False, "truth_resolved": False}: return False
    if any(payload["resolution_summary"][k] is not False for k in ["resolved","accepted","rejected","winner_selected","external_resolution_verified","automatic_resolution_allowed"]): return False
    if payload["resolution_summary"].get("resolution_candidate_only") is not True: return False
    if payload["source_summary"].get("source_validated") is not True or payload["source_summary"].get("source_inference_context_id") != payload["source_inference_context_id"]: return False
    kinds=sorted({i["conclusion_kind"] for i in payload["conclusion_items"]}); roles=sorted({i["candidate_role"] for i in payload["conclusion_items"]}); refs=sorted({i["source_conclusion_ref_id"] for i in payload["conclusion_items"]})
    return payload["conclusion_scope_summary"] == {"scope_candidate_only": True, "external_scope_verified": False, "conclusion_kinds": kinds, "candidate_roles": roles, "source_conclusion_ref_ids": refs, "item_count": len(payload["conclusion_items"])}

def _plan(source, target):
    ok = validate_virtual_world_situation_conclusion_candidate(source)
    p={k: False for k in PLAN_FALSE_FLAGS}; p.update({"ready": ok, "target": target, "candidate_only": True, "read_only": True, "conclusion_candidate_only": True, "source_valid": ok, "source_conclusion_candidate_id": source.get("conclusion_candidate_id") if type(source) is dict else None, "blocked_reasons": [] if ok else ["invalid_source_conclusion_candidate"]})
    return p

def build_conclusion_candidate_to_situation_plan(source): return _plan(source, "situation_review")
def build_conclusion_candidate_to_snapshot_plan(source): return _plan(source, "snapshot_review")
def build_conclusion_candidate_to_transition_preflight_plan(source): return _plan(source, "transition_preflight_review")
def build_conclusion_candidate_to_memory_candidate_plan(source): return _plan(source, "memory_candidate_review")
def build_conclusion_candidate_to_appraisal_plan(source): return _plan(source, "appraisal_review")
def build_conclusion_candidate_to_agp_input_plan(source): return _plan(source, "agp_input_review")

def build_virtual_world_situation_conclusion_candidate_schema_summary():
    return {"schema_version": VERSION, "schema_name": "read_only_virtual_world_situation_conclusion_candidate_schema", "canonical_id_algorithm": CANONICAL_ID_ALGORITHM, "supported_conclusion_types": list(SUPPORTED_CONCLUSION_TYPES), "supported_conclusion_kinds": list(SUPPORTED_CONCLUSION_KINDS), "read_only": True, "candidate_only": True, "strict_json_only": True, "source_must_validate": True, "side_effects_allowed": False, "next_recommended_step": "read_only_virtual_world_situation_decision_candidate_schema"}
