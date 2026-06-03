"""EVE v3 round77 — lexical to concept mapping planning scaffold.

Round77 begins the post-self-learning-v1 axis without enabling concept mapping
at runtime. Lexical vectors (fastText/EveSpecific/PMI+SVD) are evidence only;
they are not AGP anchors. Concept/category mapping must be introduced later via
read-only dry-run, operator proposal, then explicit enforcement.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


class LexConceptMappingAdapter:
    """Read-only planning boundary between lexical vectors and concept anchors.

    This adapter intentionally does not map words to categories in Round77. It
    exposes the contract future patches must satisfy so EveSpecific lexical
    learning cannot silently become an AGP/category shortcut.
    """

    planning_round = 77
    planning_version = "v3_round77_lexical_to_concept_mapping_planning"
    enforcement_enabled = False
    runtime_mapping_enabled = False

    def __init__(self, engine: Any | None = None):
        self.engine = engine
        # Round89: explicit concept commit smoke state. This is intentionally
        # separate from runtime mapping enforcement. A category record exists
        # only after an explicit smoke/commit call and must not be inferred
        # from lexical or EveSpecific vectors.
        self._concept_categories: dict[str, dict[str, Any]] = {}
        self._concept_commit_audit: list[dict[str, Any]] = []
        self._latest_runtime_mapping_round = 96

    def _active_self_learning_policy(self) -> dict[str, Any]:
        learner = getattr(self.engine, "eve_self_learning", None)
        if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
            try:
                snapshot = learner.self_learning_v1_freeze_policy_snapshot()
                return dict(snapshot.get("active_policy", {}) or {})
            except Exception as exc:
                return {"error": exc.__class__.__name__, "read_only": True}
        return {"error": "eve_self_learning_missing", "read_only": True}

    def _wrapper_policy(self) -> dict[str, Any]:
        wrapper = getattr(self.engine, "self_embedding", None)
        if wrapper is not None and hasattr(wrapper, "lookup_route_policy_snapshot"):
            try:
                return dict(wrapper.lookup_route_policy_snapshot())
            except Exception as exc:
                return {"error": exc.__class__.__name__, "read_only": True}
        return {"error": "embedding_wrapper_missing", "read_only": True}

    def _component_roles(self) -> dict[str, str]:
        return {
            "EveVocabTracker": "lexical_observation_only_no_concept_mapping",
            "EveSelfLearningAdapter": "lexical_vector_commit_gate_owner_no_category_creation",
            "EveSpecificVectorStore": "deterministic_lexical_vector_storage_only",
            "EmbeddingWrapper": "lexical_lookup_routing_and_telemetry_only",
            "LexConceptMappingAdapter": "round77_read_only_planning_contract_owner",
            "ConceptMemoryAdapter": "future_concept_store_candidate_no_round77_mutation",
            "FrameAdapter": "future_frame_slot_candidate_no_round77_mutation",
            "HypergraphAdapter": "future_relation_candidate_no_round77_mutation",
            "AGPAdapter": "anchor_validation_from_explicit_categories_and_sa_activation_only",
        }

    def mapping_contract_snapshot(self) -> dict[str, Any]:
        """Return the Round77 lexical→concept boundary contract.

        The snapshot is read-only and does not call lookup, commit, observe,
        concept memory writes, frame writes, hypergraph writes, or AGP verify.
        """
        active_policy = self._active_self_learning_policy()
        wrapper_policy = self._wrapper_policy()
        return {
            "planning_version": self.planning_version,
            "round": self.planning_round,
            "axis": "lexical_to_concept_mapping",
            "status": "planning_only_not_enforced",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "source_baseline": "self_learning_v1_round76",
            "active_self_learning_policy": active_policy,
            "wrapper_route_policy": wrapper_policy,
            "component_roles": self._component_roles(),
            "lexical_sources": [
                {
                    "name": "fastText",
                    "role": "external_lexical_seed_primary",
                    "may_create_agp_anchor": False,
                    "may_create_concept_category": False,
                },
                {
                    "name": "EveSpecificVectorStore",
                    "role": "eve_specific_lexical_vector_evidence",
                    "may_create_agp_anchor": False,
                    "may_create_concept_category": False,
                },
                {
                    "name": "PMI+SVD",
                    "role": "local_fallback_lexical_space",
                    "may_create_agp_anchor": False,
                    "may_create_concept_category": False,
                },
            ],
            "concept_anchor_sources": [
                {
                    "name": "SAActivation",
                    "role": "required_agp_activation_evidence",
                    "required_for_agp_anchor": True,
                },
                {
                    "name": "ExplicitMeaningCategory",
                    "role": "candidate_category_name_or_id_provided_by_mapping_layer",
                    "required_for_agp_anchor": True,
                },
                {
                    "name": "Frame/ConceptMemory/Hypergraph",
                    "role": "future_concept_evidence_after_dry_run_proposal_enforcement",
                    "required_in_round77": False,
                },
            ],
            "hard_boundaries": {
                "lexical_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
                "no_auto_category_creation": True,
                "no_auto_memory_promotion": True,
                "no_quarantine_bypass": True,
                "no_runtime_generation_change": True,
            },
            "future_change_rule": {
                "round78": "candidate_schema_dry_run_only",
                "round79": "mapping_dry_run_no_category_mutation",
                "round80": "operator_proposal_report",
                "round81_plus": "explicit_enforcement_only_after_tests_and_operator_review",
                "round83": "operator_acceptance_fixture_category_creation_dry_run_only",
                "round84": "concept_memory_frame_evidence_dry_run_only",
                "round85": "sa_activation_path_dry_run_only",
                "round86": "agp_bridge_smoke_dry_run_only",
                "round87": "concept_mapping_readiness_dashboard",
                "round88": "concept_mapping_v0_proposal_freeze",
                "policy_pattern": "dry_run_then_proposal_then_explicit_enforcement",
            },
            "round77_read_only_checks": {
                "does_not_call_observe_text": True,
                "does_not_call_commit_eve_specific_vectors": True,
                "does_not_call_embedding_lookup": True,
                "does_not_write_concept_memory": True,
                "does_not_write_frame_or_hypergraph": True,
                "does_not_call_agp_verify": True,
                "does_not_change_thresholds": True,
                "does_not_change_context_diversity_gate": True,
            },
            "read_only": True,
        }

    def candidate_record_schema(self) -> dict[str, Any]:
        """Return the future mapping candidate record schema without data."""
        return {
            "schema_version": "v3_round77_lexical_concept_candidate_schema_plan",
            "status": "schema_only_no_candidates_persisted",
            "required_fields": [
                "lexical_token",
                "lexical_route",
                "lexical_evidence",
                "proposed_concept_category",
                "concept_evidence",
                "sa_activation_requirement",
                "agp_anchor_policy",
                "operator_decision",
            ],
            "lexical_evidence_fields": [
                "observed_count",
                "context_diverse",
                "known_context_words",
                "eve_specific_vector_present",
                "wrapper_route",
            ],
            "concept_evidence_fields": [
                "explicit_category_id",
                "frame_slots",
                "concept_memory_reference",
                "hypergraph_relation_reference",
            ],
            "forbidden_fields": [
                "auto_created_category",
                "seed_vector_anchor_pass",
                "eve_specific_vector_anchor_pass",
                "implicit_agp_pass_without_sa",
            ],
            "default_operator_decision": "undecided",
            "read_only": True,
        }

    def plan_for_tokens(self, tokens: Iterable[str]) -> dict[str, Any]:
        """Return planning-only rows for lexical tokens.

        This is not a mapper. It deliberately returns ``mapping_status`` as
        blocked/planning until a future explicit mapping round provides concept
        evidence and operator approval.
        """
        rows: list[dict[str, Any]] = []
        for token in sorted({str(t or "").strip() for t in tokens if str(t or "").strip()}):
            rows.append(
                {
                    "lexical_token": token,
                    "mapping_status": "not_mapped_round77_planning_only",
                    "may_use_lexical_vector_as_evidence": True,
                    "may_use_lexical_vector_as_agp_anchor": False,
                    "requires_explicit_concept_category": True,
                    "requires_sa_activation_for_agp": True,
                    "operator_decision": "undecided",
                }
            )
        return {
            "planning_version": self.planning_version,
            "token_count": len(rows),
            "candidate_rows": rows,
            "runtime_mapping_enabled": False,
            "read_only": True,
        }


    def _resolve_learner(self) -> Any | None:
        return getattr(self.engine, "eve_self_learning", None)

    def _resolve_store(self) -> Any | None:
        return getattr(self.engine, "eve_specific_vector_store", None)

    def _normalize_tokens(self, tokens: Iterable[str] | None) -> list[str]:
        if tokens is None:
            learner = self._resolve_learner()
            if learner is not None and hasattr(learner, "_target_words"):
                try:
                    return list(learner._target_words(None))
                except Exception:
                    pass
            return []
        return sorted({str(t or "").strip() for t in tokens if str(t or "").strip()})

    def candidate_schema_dry_run(self, tokens: Iterable[str] | None = None) -> dict[str, Any]:
        """Round78 read-only lexical→concept candidate row dry-run.

        The method converts lexical observation evidence into candidate rows but
        deliberately leaves concept/category fields unset. It does not create
        categories, write concept memory, change AGP anchors, perform wrapper
        lookup, commit vectors, or append self-learning audit records.
        """
        learner = self._resolve_learner()
        store = self._resolve_store()
        target_tokens = self._normalize_tokens(tokens)
        quality: dict[str, Any] = {}
        if learner is not None and hasattr(learner, "observation_evidence_quality_summary"):
            try:
                quality = learner.observation_evidence_quality_summary(words=target_tokens)
            except Exception as exc:
                quality = {"error": exc.__class__.__name__, "candidate_summaries": []}
        quality_by_word = {
            str(item.get("word", "") or ""): dict(item)
            for item in list(quality.get("candidate_summaries", []) or [])
            if str(item.get("word", "") or "").strip()
        }
        store_vocab: set[str] = set()
        if store is not None and hasattr(store, "stored_vocab"):
            try:
                store_vocab = {str(w) for w in store.stored_vocab()}
            except Exception:
                store_vocab = set()
        rows: list[dict[str, Any]] = []
        for token in target_tokens:
            item = quality_by_word.get(token, {})
            observed_count = int(item.get("observed_count", 0) or 0)
            context_diverse = bool(item.get("context_diverse", False))
            eve_specific_candidate = bool(item.get("is_eve_specific_candidate", False))
            eve_specific_vector_present = token in store_vocab
            threshold_ready = bool(item.get("meets_current_observation_threshold", False))
            lexical_evidence_ready = bool(
                eve_specific_candidate
                and threshold_ready
                and context_diverse
            )
            candidate_status = (
                "lexical_evidence_ready_concept_missing"
                if lexical_evidence_ready
                else "blocked_insufficient_lexical_evidence"
            )
            rows.append({
                "lexical_token": token,
                "candidate_id": f"lex::{token}",
                "candidate_status": candidate_status,
                "mapping_stage": "round78_candidate_schema_dry_run_only",
                "lexical_evidence_ready": lexical_evidence_ready,
                "lexical_route_estimate": (
                    "eve_specific_vector_store"
                    if eve_specific_vector_present
                    else "lexical_observation_pending_or_seed_fallback"
                ),
                "lexical_evidence": {
                    "observed_count": observed_count,
                    "meets_current_observation_threshold": threshold_ready,
                    "context_diverse": context_diverse,
                    "unique_context_count": int(item.get("unique_context_count", 0) or 0),
                    "unique_text_count": int(item.get("unique_text_count", 0) or 0),
                    "unique_peer_token_count": int(item.get("unique_peer_token_count", 0) or 0),
                    "eve_specific_candidate": eve_specific_candidate,
                    "eve_specific_vector_present": eve_specific_vector_present,
                    "evidence_status": item.get("evidence_status"),
                    "peer_token_sample": list(item.get("peer_token_sample", []) or []),
                },
                "proposed_concept_category": None,
                "concept_evidence": {
                    "explicit_category_id": None,
                    "frame_slots": [],
                    "concept_memory_reference": None,
                    "hypergraph_relation_reference": None,
                },
                "sa_activation_requirement": "required_before_agp_anchor",
                "agp_anchor_policy": "lexical_vector_is_evidence_only_explicit_category_and_sa_required",
                "operator_decision": "undecided",
                "category_created": False,
                "agp_anchor_created": False,
            })
        return {
            "dry_run_version": "v3_round78_lexical_concept_candidate_schema_dry_run",
            "round": 78,
            "source_planning_version": self.planning_version,
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(rows),
            "candidate_rows": rows,
            "source_evidence_quality_version": quality.get("summary_version"),
            "quality_read_only": bool(quality.get("read_only", True)),
            "policy": {
                "lexical_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "no_auto_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
            },
            "read_only": True,
        }

    def candidate_evidence_quality_report(self, tokens: Iterable[str] | None = None) -> dict[str, Any]:
        """Round79 read-only quality report over Round78 candidate rows."""
        dry_run = self.candidate_schema_dry_run(tokens=tokens)
        rows = list(dry_run.get("candidate_rows", []) or [])
        status_counts: dict[str, int] = {}
        evidence_status_counts: dict[str, int] = {}
        ready_tokens: list[str] = []
        blocked_tokens: list[str] = []
        vector_present_tokens: list[str] = []
        missing_concept_category_tokens: list[str] = []
        for row in rows:
            token = str(row.get("lexical_token", "") or "")
            status = str(row.get("candidate_status", "unknown") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            evidence = dict(row.get("lexical_evidence", {}) or {})
            estatus = str(evidence.get("evidence_status", "unknown") or "unknown")
            evidence_status_counts[estatus] = evidence_status_counts.get(estatus, 0) + 1
            if bool(row.get("lexical_evidence_ready", False)):
                ready_tokens.append(token)
            else:
                blocked_tokens.append(token)
            if bool(evidence.get("eve_specific_vector_present", False)):
                vector_present_tokens.append(token)
            if row.get("proposed_concept_category") is None:
                missing_concept_category_tokens.append(token)
        return {
            "report_version": "v3_round79_lexical_concept_candidate_evidence_quality_report",
            "round": 79,
            "source_dry_run_version": dry_run.get("dry_run_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(rows),
            "lexical_evidence_ready_count": len(ready_tokens),
            "blocked_candidate_count": len(blocked_tokens),
            "eve_specific_vector_present_count": len(vector_present_tokens),
            "concept_category_created_count": 0,
            "agp_anchor_created_count": 0,
            "missing_explicit_concept_category_count": len(missing_concept_category_tokens),
            "ready_tokens": sorted([t for t in ready_tokens if t]),
            "blocked_tokens": sorted([t for t in blocked_tokens if t]),
            "eve_specific_vector_present_tokens": sorted([t for t in vector_present_tokens if t]),
            "missing_concept_category_tokens": sorted([t for t in missing_concept_category_tokens if t]),
            "candidate_status_counts": dict(sorted(status_counts.items())),
            "lexical_evidence_status_counts": dict(sorted(evidence_status_counts.items())),
            "operator_recommendation": (
                "proceed_to_concept_proposal_report"
                if ready_tokens
                else "collect_more_lexical_evidence_before_concept_proposal"
            ),
            "candidate_rows": deepcopy(rows),
            "policy": {
                "read_only_report_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
            },
            "read_only": True,
        }

    def concept_proposal_report(self, tokens: Iterable[str] | None = None) -> dict[str, Any]:
        """Round80 read-only operator proposal over lexical evidence-ready rows.

        This method promotes *nothing*. It turns Round79 lexical-evidence-ready
        rows into operator-review proposal records. Proposed concept IDs are
        labels for review only; they are not created categories, concept-memory
        entries, frame slots, hypergraph nodes, SA activations, or AGP anchors.
        """
        evidence_report = self.candidate_evidence_quality_report(tokens=tokens)
        rows = list(evidence_report.get("candidate_rows", []) or [])
        proposal_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        for row in rows:
            token = str(row.get("lexical_token", "") or "")
            evidence = dict(row.get("lexical_evidence", {}) or {})
            if bool(row.get("lexical_evidence_ready", False)):
                proposal_id = f"concept_proposal::lex::{token}"
                proposal_rows.append({
                    "lexical_token": token,
                    "source_candidate_id": row.get("candidate_id"),
                    "proposal_id": proposal_id,
                    "proposal_status": "operator_review_required",
                    "proposed_concept_category": {
                        "category_id": proposal_id,
                        "display_name": token,
                        "source": "lexical_evidence_ready_round80_proposal_only",
                        "created": False,
                        "activated": False,
                    },
                    "lexical_evidence": deepcopy(evidence),
                    "required_before_enforcement": [
                        "operator_acceptance",
                        "explicit_category_creation_patch",
                        "concept_memory_or_frame_evidence",
                        "sa_activation_path_test",
                        "agp_bridge_smoke_test",
                    ],
                    "may_create_category_now": False,
                    "may_create_agp_anchor_now": False,
                    "operator_decision": "pending",
                    "category_created": False,
                    "agp_anchor_created": False,
                })
            else:
                blocked_rows.append({
                    "lexical_token": token,
                    "source_candidate_id": row.get("candidate_id"),
                    "proposal_status": "blocked_before_concept_proposal",
                    "blocked_reason": row.get("candidate_status"),
                    "lexical_evidence": deepcopy(evidence),
                    "category_created": False,
                    "agp_anchor_created": False,
                })
        return {
            "report_version": "v3_round80_concept_proposal_report",
            "round": 80,
            "source_evidence_quality_report_version": evidence_report.get("report_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(rows),
            "proposal_count": len(proposal_rows),
            "blocked_candidate_count": len(blocked_rows),
            "concept_category_created_count": 0,
            "agp_anchor_created_count": 0,
            "proposal_rows": proposal_rows,
            "blocked_rows": blocked_rows,
            "ready_tokens": sorted([str(r.get("lexical_token", "") or "") for r in proposal_rows if r.get("lexical_token")]),
            "blocked_tokens": sorted([str(r.get("lexical_token", "") or "") for r in blocked_rows if r.get("lexical_token")]),
            "operator_recommendation": (
                "review_concept_proposals_before_mapping_gate_dry_run"
                if proposal_rows
                else "collect_more_lexical_evidence_before_concept_proposals"
            ),
            "next_round_recommendation": {
                "round81": "concept_mapping_gate_dry_run",
                "allowed_scope": "dry_run_only_no_category_creation_no_agp_anchor",
            },
            "policy": {
                "operator_proposal_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def concept_mapping_gate_dry_run(self, tokens: Iterable[str] | None = None) -> dict[str, Any]:
        """Round81 read-only dry-run for the future concept mapping gate.

        The dry-run evaluates whether Round80 operator proposal rows would be
        eligible for a future explicit mapping patch. It does not create
        categories, write concept memory, mutate frame/hypergraph state, create
        SA activation, call AGP verify, call embedding lookup, or commit vectors.
        """
        proposal_report = self.concept_proposal_report(tokens=tokens)
        proposal_rows = list(proposal_report.get("proposal_rows", []) or [])
        blocked_rows = list(proposal_report.get("blocked_rows", []) or [])
        gate_rows: list[dict[str, Any]] = []
        pass_tokens: list[str] = []
        block_tokens: list[str] = []
        reason_counts: dict[str, int] = {}

        for row in proposal_rows:
            token = str(row.get("lexical_token", "") or "")
            reasons: list[str] = []
            if row.get("proposal_status") != "operator_accepted":
                reasons.append("operator_acceptance_required")
            category = dict(row.get("proposed_concept_category", {}) or {})
            if bool(category.get("created", False)) is not True:
                reasons.append("explicit_category_creation_required")
            if bool(category.get("activated", False)) is not True:
                reasons.append("sa_activation_path_required")
            required = set(row.get("required_before_enforcement", []) or [])
            if "concept_memory_or_frame_evidence" in required:
                reasons.append("concept_memory_or_frame_evidence_required")
            if "agp_bridge_smoke_test" in required:
                reasons.append("agp_bridge_smoke_test_required")

            would_pass = not reasons
            if would_pass:
                pass_tokens.append(token)
            else:
                block_tokens.append(token)
            for reason in reasons or ["would_pass"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            gate_rows.append({
                "lexical_token": token,
                "source_proposal_id": row.get("proposal_id"),
                "dry_run_status": "would_pass_future_mapping_gate" if would_pass else "would_block_future_mapping_gate",
                "would_pass_mapping_gate": would_pass,
                "blocked_reasons": reasons,
                "lexical_evidence_ready": True,
                "operator_decision": row.get("operator_decision", "pending"),
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
            })

        for row in blocked_rows:
            token = str(row.get("lexical_token", "") or "")
            reason = str(row.get("blocked_reason", "blocked_before_concept_proposal") or "blocked_before_concept_proposal")
            block_tokens.append(token)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            gate_rows.append({
                "lexical_token": token,
                "source_candidate_id": row.get("source_candidate_id"),
                "dry_run_status": "blocked_before_mapping_gate",
                "would_pass_mapping_gate": False,
                "blocked_reasons": [reason],
                "lexical_evidence_ready": False,
                "operator_decision": "not_applicable",
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
            })

        return {
            "dry_run_version": "v3_round81_concept_mapping_gate_dry_run",
            "round": 81,
            "source_concept_proposal_report_version": proposal_report.get("report_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(proposal_report.get("candidate_count", len(gate_rows)) or len(gate_rows)),
            "proposal_count": len(proposal_rows),
            "blocked_candidate_count": len(blocked_rows),
            "would_pass_count": len(pass_tokens),
            "would_block_count": len(block_tokens),
            "would_pass_tokens": sorted([t for t in pass_tokens if t]),
            "would_block_tokens": sorted([t for t in block_tokens if t]),
            "blocked_reason_counts": dict(sorted(reason_counts.items())),
            "gate_rows": gate_rows,
            "operator_recommendation": (
                "do_not_enable_mapping_gate_until_operator_acceptance_and_concept_evidence"
                if gate_rows
                else "collect_concept_proposals_before_mapping_gate_dry_run"
            ),
            "next_round_recommendation": {
                "round82": "concept_mapping_operator_acceptance_fixture_or_gate_proposal_report",
                "allowed_scope": "read_only_or_explicit_test_fixture_no_runtime_enforcement",
            },
            "policy": {
                "dry_run_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def concept_mapping_gate_proposal_report(self, tokens: Iterable[str] | None = None) -> dict[str, Any]:
        """Round82 read-only operator proposal over the Round81 mapping gate dry-run.

        This report consolidates the Round81 blocked reasons into operator
        action items. It deliberately does not accept proposals, create
        categories, write concept memory, create SA activation, call AGP verify,
        enable runtime mapping, or enforce the mapping gate.
        """
        dry_run = self.concept_mapping_gate_dry_run(tokens=tokens)
        gate_rows = list(dry_run.get("gate_rows", []) or [])
        reason_counts = dict(dry_run.get("blocked_reason_counts", {}) or {})

        action_items_by_reason = {
            "operator_acceptance_required": "operator_must_explicitly_accept_the_concept_proposal",
            "explicit_category_creation_required": "future_patch_must_create_an_explicit_category_record",
            "sa_activation_path_required": "future_patch_must_prove_sa_activation_path_for_the_category",
            "concept_memory_or_frame_evidence_required": "future_patch_must_attach_concept_memory_or_frame_evidence",
            "agp_bridge_smoke_test_required": "future_patch_must_add_agp_bridge_smoke_test",
            "blocked_insufficient_lexical_evidence": "collect_more_lexical_observation_evidence_before_proposal",
        }
        action_items = [
            {
                "blocked_reason": reason,
                "count": int(count or 0),
                "operator_action": action_items_by_reason.get(reason, "review_blocked_reason_before_enforcement"),
                "required_before_enforcement": True,
            }
            for reason, count in sorted(reason_counts.items())
        ]

        proposal_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        for row in gate_rows:
            token = str(row.get("lexical_token", "") or "")
            entry = {
                "lexical_token": token,
                "source_dry_run_status": row.get("dry_run_status"),
                "would_pass_mapping_gate": bool(row.get("would_pass_mapping_gate", False)),
                "blocked_reasons": list(row.get("blocked_reasons", []) or []),
                "operator_decision_status": "operator_review_required" if row.get("lexical_evidence_ready") else "not_ready_for_operator_acceptance",
                "may_enable_mapping_now": False,
                "may_create_category_now": False,
                "may_create_agp_anchor_now": False,
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
            }
            if bool(row.get("lexical_evidence_ready", False)):
                proposal_rows.append(entry)
            else:
                blocked_rows.append(entry)

        return {
            "report_version": "v3_round82_concept_mapping_gate_proposal_report",
            "round": 82,
            "source_gate_dry_run_version": dry_run.get("dry_run_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(dry_run.get("candidate_count", len(gate_rows)) or len(gate_rows)),
            "proposal_count": int(dry_run.get("proposal_count", len(proposal_rows)) or len(proposal_rows)),
            "blocked_candidate_count": int(dry_run.get("blocked_candidate_count", len(blocked_rows)) or len(blocked_rows)),
            "would_pass_count": int(dry_run.get("would_pass_count", 0) or 0),
            "would_block_count": int(dry_run.get("would_block_count", len(gate_rows)) or len(gate_rows)),
            "operator_action_item_count": len(action_items),
            "gate_proposal_rows": proposal_rows,
            "blocked_rows": blocked_rows,
            "blocked_reason_counts": dict(sorted(reason_counts.items())),
            "operator_action_items": action_items,
            "operator_recommendation": (
                "do_not_enable_runtime_mapping_until_all_action_items_are_resolved"
                if action_items
                else "mapping_gate_has_no_known_blocks_but_still_requires_explicit_enforcement_patch"
            ),
            "next_round_recommendation": {
                "round83": "operator_acceptance_fixture_or_explicit_category_creation_dry_run",
                "allowed_scope": "still_no_runtime_mapping_until_category_and_agp_bridge_tests_exist",
            },
            "policy": {
                "operator_proposal_report_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def operator_acceptance_category_creation_dry_run(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round83 read-only operator-acceptance fixture + category creation dry-run.

        This method models what would happen if an operator explicitly accepted
        selected Round80 concept proposals. Acceptance is a test fixture only: it
        is not persisted as an operator decision. Explicit category creation is
        represented as a ``would_create`` plan only; no category, concept memory,
        frame/hypergraph edge, SA activation, AGP anchor, wrapper lookup, or
        vector commit is created.
        """
        gate_report = self.concept_mapping_gate_proposal_report(tokens=tokens)
        proposal_rows = list(gate_report.get("gate_proposal_rows", []) or [])
        blocked_source_rows = list(gate_report.get("blocked_rows", []) or [])
        if accepted_tokens is None:
            accepted_set = {
                str(row.get("lexical_token", "") or "")
                for row in proposal_rows
                if str(row.get("lexical_token", "") or "").strip()
            }
        else:
            accepted_set = {
                str(t or "").strip()
                for t in accepted_tokens
                if str(t or "").strip()
            }

        accepted_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        category_dry_run_rows: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}

        for row in proposal_rows:
            token = str(row.get("lexical_token", "") or "")
            source_reasons = list(row.get("blocked_reasons", []) or [])
            if token in accepted_set:
                remaining_reasons = [
                    reason
                    for reason in source_reasons
                    if reason != "operator_acceptance_required"
                ]
                # Round83 resolves the operator acceptance fixture, but category
                # creation is still a dry-run artifact, not runtime state. Keep a
                # distinct block reason so later enforcement cannot mistake this
                # for a real category record.
                if "explicit_category_creation_required" in remaining_reasons:
                    remaining_reasons = [
                        "explicit_category_creation_dry_run_only"
                        if reason == "explicit_category_creation_required"
                        else reason
                        for reason in remaining_reasons
                    ]
                category_id = f"concept_category::lex::{token}"
                category_plan = {
                    "category_id": category_id,
                    "display_name": token,
                    "source": "round83_operator_acceptance_fixture_category_creation_dry_run",
                    "would_create": True,
                    "created": False,
                    "persisted": False,
                    "activated": False,
                    "sa_activation_created": False,
                    "agp_anchor_created": False,
                }
                accepted_entry = {
                    "lexical_token": token,
                    "operator_acceptance_fixture": True,
                    "operator_decision_status": "operator_accepted_fixture_not_persisted",
                    "source_gate_status": row.get("source_dry_run_status"),
                    "source_blocked_reasons": source_reasons,
                    "resolved_reasons": ["operator_acceptance_required"],
                    "remaining_blocked_reasons": remaining_reasons,
                    "category_creation_dry_run": category_plan,
                    "would_create_explicit_category": True,
                    "category_created": False,
                    "concept_memory_mutation": False,
                    "frame_hypergraph_mutation": False,
                    "sa_activation_created": False,
                    "agp_anchor_created": False,
                    "agp_verify_called": False,
                    "would_pass_mapping_gate_after_fixture": False,
                }
                accepted_rows.append(accepted_entry)
                category_dry_run_rows.append({
                    "lexical_token": token,
                    "category_id": category_id,
                    "dry_run_status": "would_create_explicit_category_record_later",
                    "would_create": True,
                    "created": False,
                    "requires_future_concept_memory_or_frame_evidence": True,
                    "requires_future_sa_activation_path": True,
                    "requires_future_agp_bridge_smoke_test": True,
                })
                for reason in remaining_reasons or ["would_pass_after_fixture"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            else:
                reasons = source_reasons or ["operator_acceptance_required"]
                blocked_rows.append({
                    "lexical_token": token,
                    "operator_acceptance_fixture": False,
                    "operator_decision_status": "operator_acceptance_required",
                    "blocked_reasons": reasons,
                    "category_created": False,
                    "agp_anchor_created": False,
                })
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for row in blocked_source_rows:
            token = str(row.get("lexical_token", "") or "")
            reasons = list(row.get("blocked_reasons", []) or []) or ["blocked_before_concept_proposal"]
            blocked_rows.append({
                "lexical_token": token,
                "operator_acceptance_fixture": False,
                "operator_decision_status": "not_ready_for_operator_acceptance",
                "blocked_reasons": reasons,
                "category_created": False,
                "agp_anchor_created": False,
            })
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            "dry_run_version": "v3_round83_operator_acceptance_category_creation_dry_run",
            "round": 83,
            "source_gate_proposal_report_version": gate_report.get("report_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(gate_report.get("candidate_count", 0) or 0),
            "proposal_count": int(gate_report.get("proposal_count", len(proposal_rows)) or len(proposal_rows)),
            "accepted_fixture_count": len(accepted_rows),
            "category_creation_dry_run_count": len(category_dry_run_rows),
            "blocked_count": len(blocked_rows),
            "would_pass_mapping_gate_count": 0,
            "accepted_tokens": sorted([str(r.get("lexical_token", "") or "") for r in accepted_rows if r.get("lexical_token")]),
            "blocked_tokens": sorted([str(r.get("lexical_token", "") or "") for r in blocked_rows if r.get("lexical_token")]),
            "remaining_blocked_reason_counts": dict(sorted(reason_counts.items())),
            "accepted_rows": accepted_rows,
            "category_creation_dry_run_rows": category_dry_run_rows,
            "blocked_rows": blocked_rows,
            "operator_recommendation": (
                "proceed_to_concept_memory_or_frame_evidence_dry_run"
                if accepted_rows
                else "operator_must_accept_at_least_one_concept_proposal_before_category_dry_run"
            ),
            "next_round_recommendation": {
                "round84": "concept_memory_or_frame_evidence_dry_run",
                "allowed_scope": "read_only_evidence_attachment_no_runtime_mapping_no_agp_anchor",
            },
            "policy": {
                "operator_acceptance_fixture_only": True,
                "operator_decision_not_persisted": True,
                "explicit_category_creation_dry_run_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def concept_memory_frame_evidence_dry_run(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round84 read-only concept-memory/frame evidence attachment dry-run.

        This consumes the Round83 operator-acceptance/category-plan fixture and
        attaches future concept-memory and frame-evidence plans. It creates no
        category, writes no concept memory, mutates no frame/hypergraph state,
        creates no SA activation, calls no AGP verify, performs no embedding
        lookup, and commits no vectors.
        """
        source = self.operator_acceptance_category_creation_dry_run(
            tokens=tokens,
            accepted_tokens=accepted_tokens,
        )
        evidence_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = deepcopy(list(source.get("blocked_rows", []) or []))
        reason_counts: dict[str, int] = {}

        for row in list(source.get("accepted_rows", []) or []):
            token = str(row.get("lexical_token", "") or "")
            category_plan = deepcopy(dict(row.get("category_creation_dry_run", {}) or {}))
            source_reasons = list(row.get("remaining_blocked_reasons", []) or [])
            remaining: list[str] = []
            for reason in source_reasons:
                if reason == "concept_memory_or_frame_evidence_required":
                    remaining.append("concept_memory_or_frame_evidence_dry_run_only")
                else:
                    remaining.append(reason)
            concept_evidence_plan = {
                "evidence_plan_id": f"concept_evidence::lex::{token}",
                "lexical_token": token,
                "category_id": category_plan.get("category_id"),
                "would_attach_concept_memory_reference": True,
                "concept_memory_reference": f"concept_memory_candidate::lex::{token}",
                "would_attach_frame_slots": True,
                "frame_slots": [
                    {"slot": "lexical_label", "value": token, "source": "round84_dry_run"},
                    {"slot": "source_candidate", "value": f"lex::{token}", "source": "round84_dry_run"},
                ],
                "would_attach_hypergraph_relation": False,
                "concept_memory_persisted": False,
                "frame_persisted": False,
                "hypergraph_mutation": False,
            }
            evidence_rows.append({
                "lexical_token": token,
                "source_category_plan": category_plan,
                "concept_evidence_dry_run": concept_evidence_plan,
                "operator_acceptance_fixture": bool(row.get("operator_acceptance_fixture", False)),
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
                "remaining_blocked_reasons": remaining,
                "would_pass_mapping_gate_after_evidence_fixture": False,
            })
            for reason in remaining or ["would_pass_after_evidence_fixture"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for row in blocked_rows:
            for reason in list(row.get("blocked_reasons", []) or ["blocked_before_evidence_dry_run"]):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

        return {
            "dry_run_version": "v3_round84_concept_memory_frame_evidence_dry_run",
            "round": 84,
            "source_round83_dry_run_version": source.get("dry_run_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(source.get("candidate_count", 0) or 0),
            "accepted_fixture_count": int(source.get("accepted_fixture_count", 0) or 0),
            "evidence_dry_run_count": len(evidence_rows),
            "blocked_count": len(blocked_rows),
            "would_pass_mapping_gate_count": 0,
            "evidence_tokens": sorted([str(r.get("lexical_token", "") or "") for r in evidence_rows if r.get("lexical_token")]),
            "blocked_tokens": sorted([str(r.get("lexical_token", "") or "") for r in blocked_rows if r.get("lexical_token")]),
            "remaining_blocked_reason_counts": dict(sorted(reason_counts.items())),
            "evidence_rows": evidence_rows,
            "blocked_rows": blocked_rows,
            "operator_recommendation": (
                "proceed_to_sa_activation_path_dry_run" if evidence_rows else "operator_acceptance_required_before_evidence_dry_run"
            ),
            "next_round_recommendation": {
                "round85": "sa_activation_path_dry_run",
                "allowed_scope": "read_only_path_plan_no_sa_mutation_no_agp_anchor",
            },
            "policy": {
                "concept_memory_frame_evidence_dry_run_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }

    def sa_activation_path_dry_run(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round85 read-only SA activation path dry-run.

        Builds a future SA activation plan for accepted/evidence-backed category
        plans. No SA state is touched and no AGP verification is called.
        """
        source = self.concept_memory_frame_evidence_dry_run(tokens=tokens, accepted_tokens=accepted_tokens)
        sa_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = deepcopy(list(source.get("blocked_rows", []) or []))
        reason_counts: dict[str, int] = {}
        for row in list(source.get("evidence_rows", []) or []):
            token = str(row.get("lexical_token", "") or "")
            category_id = dict(row.get("source_category_plan", {}) or {}).get("category_id")
            source_reasons = list(row.get("remaining_blocked_reasons", []) or [])
            remaining: list[str] = []
            for reason in source_reasons:
                if reason == "sa_activation_path_required":
                    remaining.append("sa_activation_path_dry_run_only")
                else:
                    remaining.append(reason)
            sa_plan = {
                "sa_activation_plan_id": f"sa_activation_plan::lex::{token}",
                "category_id": category_id,
                "lexical_token": token,
                "would_activate_category": True,
                "activation_source": "round85_dry_run",
                "activation_created": False,
                "sa_state_mutation": False,
                "required_for_agp_anchor": True,
            }
            sa_rows.append({
                "lexical_token": token,
                "source_evidence_plan": deepcopy(dict(row.get("concept_evidence_dry_run", {}) or {})),
                "sa_activation_path_dry_run": sa_plan,
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
                "remaining_blocked_reasons": remaining,
                "would_pass_mapping_gate_after_sa_fixture": False,
            })
            for reason in remaining or ["would_pass_after_sa_fixture"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for row in blocked_rows:
            for reason in list(row.get("blocked_reasons", []) or ["blocked_before_sa_path_dry_run"]):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        return {
            "dry_run_version": "v3_round85_sa_activation_path_dry_run",
            "round": 85,
            "source_round84_dry_run_version": source.get("dry_run_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(source.get("candidate_count", 0) or 0),
            "sa_activation_path_dry_run_count": len(sa_rows),
            "blocked_count": len(blocked_rows),
            "would_pass_mapping_gate_count": 0,
            "sa_path_tokens": sorted([str(r.get("lexical_token", "") or "") for r in sa_rows if r.get("lexical_token")]),
            "blocked_tokens": sorted([str(r.get("lexical_token", "") or "") for r in blocked_rows if r.get("lexical_token")]),
            "remaining_blocked_reason_counts": dict(sorted(reason_counts.items())),
            "sa_activation_path_rows": sa_rows,
            "blocked_rows": blocked_rows,
            "operator_recommendation": (
                "proceed_to_agp_bridge_smoke_dry_run" if sa_rows else "concept_evidence_dry_run_required_before_sa_path"
            ),
            "next_round_recommendation": {
                "round86": "agp_bridge_smoke_dry_run",
                "allowed_scope": "read_only_bridge_plan_no_agp_verify_call_no_anchor_creation",
            },
            "policy": {
                "sa_activation_path_dry_run_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }

    def agp_bridge_smoke_dry_run(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round86 read-only AGP bridge smoke dry-run.

        Models the AGP bridge preconditions but does not call AGP.verify and does
        not create an AGP anchor. It reports whether the accepted token would be
        ready for a future explicit concept commit if the dry-run artifacts were
        persisted by a separate enforcement patch.
        """
        source = self.sa_activation_path_dry_run(tokens=tokens, accepted_tokens=accepted_tokens)
        bridge_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = deepcopy(list(source.get("blocked_rows", []) or []))
        reason_counts: dict[str, int] = {}
        bridge_ready_tokens: list[str] = []
        for row in list(source.get("sa_activation_path_rows", []) or []):
            token = str(row.get("lexical_token", "") or "")
            source_reasons = list(row.get("remaining_blocked_reasons", []) or [])
            remaining: list[str] = []
            for reason in source_reasons:
                if reason == "agp_bridge_smoke_test_required":
                    remaining.append("agp_bridge_smoke_test_dry_run_only")
                else:
                    remaining.append(reason)
            dry_run_only_reasons = {
                "explicit_category_creation_dry_run_only",
                "concept_memory_or_frame_evidence_dry_run_only",
                "sa_activation_path_dry_run_only",
                "agp_bridge_smoke_test_dry_run_only",
            }
            would_pass_if_persisted = bool(remaining) and set(remaining).issubset(dry_run_only_reasons)
            if would_pass_if_persisted:
                bridge_ready_tokens.append(token)
            bridge_plan = {
                "agp_bridge_plan_id": f"agp_bridge_plan::lex::{token}",
                "lexical_token": token,
                "requires_explicit_category": True,
                "requires_sa_activation": True,
                "uses_lexical_vector_as_anchor": False,
                "uses_eve_specific_vector_as_anchor": False,
                "would_call_agp_verify_after_persistence": True,
                "agp_verify_called_now": False,
                "agp_anchor_created": False,
                "would_pass_if_persisted": would_pass_if_persisted,
            }
            bridge_rows.append({
                "lexical_token": token,
                "source_sa_activation_path": deepcopy(dict(row.get("sa_activation_path_dry_run", {}) or {})),
                "agp_bridge_smoke_dry_run": bridge_plan,
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
                "remaining_blocked_reasons": remaining,
                "would_pass_mapping_gate_after_bridge_fixture": False,
                "would_pass_if_all_dry_run_artifacts_persisted": would_pass_if_persisted,
            })
            for reason in remaining or ["would_pass_if_all_dry_run_artifacts_persisted"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for row in blocked_rows:
            for reason in list(row.get("blocked_reasons", []) or ["blocked_before_agp_bridge_dry_run"]):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        return {
            "dry_run_version": "v3_round86_agp_bridge_smoke_dry_run",
            "round": 86,
            "source_round85_dry_run_version": source.get("dry_run_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(source.get("candidate_count", 0) or 0),
            "agp_bridge_dry_run_count": len(bridge_rows),
            "bridge_ready_if_persisted_count": len(bridge_ready_tokens),
            "blocked_count": len(blocked_rows),
            "would_pass_mapping_gate_count": 0,
            "bridge_ready_if_persisted_tokens": sorted([t for t in bridge_ready_tokens if t]),
            "blocked_tokens": sorted([str(r.get("lexical_token", "") or "") for r in blocked_rows if r.get("lexical_token")]),
            "remaining_blocked_reason_counts": dict(sorted(reason_counts.items())),
            "agp_bridge_rows": bridge_rows,
            "blocked_rows": blocked_rows,
            "operator_recommendation": (
                "proceed_to_concept_mapping_readiness_dashboard" if bridge_rows else "sa_activation_path_dry_run_required_before_agp_bridge"
            ),
            "next_round_recommendation": {
                "round87": "concept_mapping_readiness_dashboard",
                "allowed_scope": "read_only_dashboard_no_enforcement",
            },
            "policy": {
                "agp_bridge_smoke_dry_run_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }

    def concept_mapping_readiness_dashboard(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round87 read-only dashboard over Round84-86 concept mapping plans."""
        r84 = self.concept_memory_frame_evidence_dry_run(tokens=tokens, accepted_tokens=accepted_tokens)
        r85 = self.sa_activation_path_dry_run(tokens=tokens, accepted_tokens=accepted_tokens)
        r86 = self.agp_bridge_smoke_dry_run(tokens=tokens, accepted_tokens=accepted_tokens)
        ready_tokens = list(r86.get("bridge_ready_if_persisted_tokens", []) or [])
        dashboard_rows: list[dict[str, Any]] = []
        for row in list(r86.get("agp_bridge_rows", []) or []):
            token = str(row.get("lexical_token", "") or "")
            dashboard_rows.append({
                "lexical_token": token,
                "operator_acceptance_fixture_present": token in set(r84.get("evidence_tokens", []) or []),
                "concept_memory_frame_evidence_dry_run_present": token in set(r84.get("evidence_tokens", []) or []),
                "sa_activation_path_dry_run_present": token in set(r85.get("sa_path_tokens", []) or []),
                "agp_bridge_dry_run_present": True,
                "ready_for_explicit_concept_commit_smoke": token in set(ready_tokens),
                "remaining_blocked_reasons": list(row.get("remaining_blocked_reasons", []) or []),
                "category_created": False,
                "concept_memory_mutation": False,
                "frame_hypergraph_mutation": False,
                "sa_activation_created": False,
                "agp_anchor_created": False,
                "agp_verify_called": False,
            })
        return {
            "dashboard_version": "v3_round87_concept_mapping_readiness_dashboard",
            "round": 87,
            "source_round84_version": r84.get("dry_run_version"),
            "source_round85_version": r85.get("dry_run_version"),
            "source_round86_version": r86.get("dry_run_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(r86.get("candidate_count", 0) or 0),
            "ready_for_explicit_concept_commit_smoke_count": len(ready_tokens),
            "ready_for_explicit_concept_commit_smoke_tokens": sorted([str(t) for t in ready_tokens]),
            "blocked_count": int(r86.get("blocked_count", 0) or 0),
            "dashboard_rows": dashboard_rows,
            "stage_counts": {
                "concept_memory_frame_evidence_dry_run_count": int(r84.get("evidence_dry_run_count", 0) or 0),
                "sa_activation_path_dry_run_count": int(r85.get("sa_activation_path_dry_run_count", 0) or 0),
                "agp_bridge_dry_run_count": int(r86.get("agp_bridge_dry_run_count", 0) or 0),
                "bridge_ready_if_persisted_count": int(r86.get("bridge_ready_if_persisted_count", 0) or 0),
            },
            "operator_recommendation": (
                "freeze_concept_mapping_v0_proposal_before_explicit_commit_smoke"
                if ready_tokens
                else "resolve_readiness_blocks_before_freeze"
            ),
            "next_round_recommendation": {
                "round88": "concept_mapping_v0_proposal_freeze",
                "allowed_scope": "read_only_freeze_no_mutation",
            },
            "policy": {
                "readiness_dashboard_only": True,
                "no_runtime_mapping": True,
                "no_enforcement": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_wrapper_lookup": True,
                "no_vector_commit": True,
            },
            "read_only": True,
        }

    def concept_mapping_v0_proposal_freeze(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round88 read-only freeze of the concept mapping v0 proposal.

        This freezes the operator-review proposal state for the next explicit
        concept commit smoke. It intentionally does not perform that commit.
        """
        dashboard = self.concept_mapping_readiness_dashboard(tokens=tokens, accepted_tokens=accepted_tokens)
        ready_tokens = list(dashboard.get("ready_for_explicit_concept_commit_smoke_tokens", []) or [])
        return {
            "freeze_version": "v3_round88_concept_mapping_v0_proposal_freeze",
            "round": 88,
            "source_dashboard_version": dashboard.get("dashboard_version"),
            "status": "frozen_for_operator_review_no_runtime_mapping",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(dashboard.get("candidate_count", 0) or 0),
            "explicit_concept_commit_candidate_count": len(ready_tokens),
            "explicit_concept_commit_candidate_tokens": sorted([str(t) for t in ready_tokens]),
            "blocked_count": int(dashboard.get("blocked_count", 0) or 0),
            "frozen_policy": {
                "lexical_vector_is_evidence_only": True,
                "operator_acceptance_fixture_not_persisted": True,
                "category_creation_was_dry_run_only": True,
                "concept_memory_frame_evidence_was_dry_run_only": True,
                "sa_activation_path_was_dry_run_only": True,
                "agp_bridge_was_dry_run_only": True,
                "no_runtime_mapping": True,
                "no_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_anchor_creation": True,
                "no_agp_verify_call": True,
                "no_vector_commit": True,
            },
            "dashboard": deepcopy(dashboard),
            "next_round_recommendation": {
                "round89": "explicit_concept_commit_smoke_single_round_only",
                "allowed_scope": "mutation_round_requires_split_full_suite_and_agp_bridge_test",
            },
            "read_only": True,
        }


    def _commit_ready_tokens(self, tokens: Iterable[str] | None = None) -> list[str]:
        freeze = self.concept_mapping_v0_proposal_freeze(
            tokens=tokens,
            accepted_tokens=tokens,
        ) if tokens is not None else self.concept_mapping_v0_proposal_freeze()
        return [str(t) for t in list(freeze.get("explicit_concept_commit_candidate_tokens", []) or []) if str(t).strip()]

    def _concept_memory_stats(self) -> dict[str, Any]:
        cm = getattr(self.engine, "concept_memory", None)
        if cm is None or not hasattr(cm, "stats"):
            return {"concepts_count": 0, "learn_count": 0}
        try:
            data = cm.stats()
            return dict(data) if isinstance(data, dict) else {}
        except Exception as exc:
            return {"error": exc.__class__.__name__}

    def _hypergraph_stats(self) -> dict[str, Any]:
        hg = getattr(self.engine, "hypergraph_adapter", None)
        if hg is None or not hasattr(hg, "stats"):
            return {}
        try:
            data = hg.stats()
            return dict(data) if isinstance(data, dict) else {}
        except Exception as exc:
            return {"error": exc.__class__.__name__}

    def _activate_category(self, category_id: str, token: str) -> dict[str, Any]:
        activation = getattr(self.engine, "activation_adapter", None)
        activated = False
        wm_added = False
        if activation is not None:
            try:
                if hasattr(activation, "sa") and hasattr(activation.sa, "activate"):
                    activation.sa.activate(category_id, strength=0.7)
                    activated = True
            except Exception:
                activated = False
            try:
                if hasattr(activation, "wm") and hasattr(activation.wm, "add"):
                    activation.wm.add(category_id, salience=0.7)
                    wm_added = True
            except Exception:
                wm_added = False
        return {
            "category_id": category_id,
            "lexical_token": token,
            "activation_strength": 0.7,
            "sa_activation_created": activated,
            "working_memory_added": wm_added,
            "required_for_agp_anchor": True,
        }

    def _persist_concept_memory(self, token: str, category_id: str) -> dict[str, Any]:
        cm = getattr(self.engine, "concept_memory", None)
        definition = f"{token} = operator-accepted EVE internal concept category from lexical evidence"
        if cm is None or not hasattr(cm, "learn"):
            return {
                "concept_memory_persisted": False,
                "concept_memory_key": token,
                "definition": definition,
                "error": "concept_memory_missing",
            }
        try:
            concept = cm.learn(token, definition, source="round89_explicit_concept_commit_smoke")
            return {
                "concept_memory_persisted": concept is not None,
                "concept_memory_key": token,
                "definition": definition,
                "source": "round89_explicit_concept_commit_smoke",
            }
        except Exception as exc:
            return {
                "concept_memory_persisted": False,
                "concept_memory_key": token,
                "definition": definition,
                "error": exc.__class__.__name__,
            }

    def _persist_frame_hypergraph(self, token: str, category_id: str) -> dict[str, Any]:
        hg = getattr(self.engine, "hypergraph_adapter", None)
        frame_slots = [
            {"slot": "lexical_label", "value": token, "source": "round89_commit"},
            {"slot": "category_id", "value": category_id, "source": "round89_commit"},
            {"slot": "evidence_layer", "value": "lexical_to_concept", "source": "round89_commit"},
        ]
        edge_id = None
        if hg is not None and hasattr(hg, "add_edge"):
            try:
                edge_id = hg.add_edge(
                    {category_id, f"lex::{token}", "concept_mapping_v0"},
                    roles={
                        category_id: "concept_category",
                        f"lex::{token}": "lexical_evidence",
                        "concept_mapping_v0": "mapping_source",
                    },
                    weight=0.6,
                )
            except Exception:
                edge_id = None
        return {
            "frame_slots": frame_slots,
            "hypergraph_edge_id": edge_id,
            "frame_hypergraph_mutation": edge_id is not None,
        }

    def explicit_concept_commit_smoke(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round89 explicit concept commit smoke.

        This is the first minimal concept-layer mutation. It is still not
        runtime mapping enforcement: it only persists explicit operator-accepted
        categories for tokens that were ready in the Round88 v0 freeze. Lexical
        and EveSpecific vectors remain evidence only and are never used as AGP
        anchors.
        """
        freeze = self.concept_mapping_v0_proposal_freeze(tokens=tokens, accepted_tokens=accepted_tokens)
        ready_tokens = [str(t) for t in list(freeze.get("explicit_concept_commit_candidate_tokens", []) or []) if str(t).strip()]
        before_cm = self._concept_memory_stats()
        before_hg = self._hypergraph_stats()
        committed_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        agp_pass_tokens: list[str] = []
        agp_fail_tokens: list[str] = []

        for token in ready_tokens:
            category_id = f"concept_category::lex::{token}"
            if category_id in self._concept_categories:
                blocked_rows.append({
                    "lexical_token": token,
                    "category_id": category_id,
                    "blocked_reason": "category_already_exists",
                })
                continue
            concept_memory = self._persist_concept_memory(token, category_id)
            frame_record = self._persist_frame_hypergraph(token, category_id)
            activation_record = self._activate_category(category_id, token)
            activated_categories = {category_id: 1.0 if activation_record.get("sa_activation_created") else 0.0}
            agp_result_obj = None
            agp = getattr(self.engine, "agp_adapter", None)
            if agp is not None and hasattr(agp, "verify"):
                try:
                    agp_result_obj = agp.verify(
                        candidate_response=f"{token} 개념을 확인했어.",
                        meaning={"meaning_categories": [category_id]},
                        activated_categories=activated_categories,
                        hormone_state={},
                    )
                except Exception as exc:
                    agp_result_obj = {"passed": False, "reason": exc.__class__.__name__}
            agp_passed = bool(getattr(agp_result_obj, "passed", False)) if agp_result_obj is not None else False
            agp_reason = getattr(agp_result_obj, "reason", "agp_adapter_missing") if agp_result_obj is not None else "agp_adapter_missing"
            if agp_passed:
                agp_pass_tokens.append(token)
            else:
                agp_fail_tokens.append(token)
            record = {
                "lexical_token": token,
                "category_id": category_id,
                "category_created": True,
                "runtime_mapping_enabled": False,
                "enforcement_enabled": False,
                "source_freeze_version": freeze.get("freeze_version"),
                "concept_memory": concept_memory,
                "frame_hypergraph": frame_record,
                "sa_activation": activation_record,
                "agp_bridge": {
                    "agp_verify_called": True,
                    "agp_passed": agp_passed,
                    "agp_reason": agp_reason,
                    "candidate_categories": [category_id],
                    "activated_categories": [category_id] if activation_record.get("sa_activation_created") else [],
                    "uses_lexical_vector_as_anchor": False,
                    "uses_eve_specific_vector_as_anchor": False,
                    "uses_seed_vector_as_anchor": False,
                    "anchor_source": "explicit_category_plus_sa_activation",
                },
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
            }
            self._concept_categories[category_id] = deepcopy(record)
            self._concept_commit_audit.append({
                "event": "explicit_concept_commit_smoke",
                "round": 89,
                "lexical_token": token,
                "category_id": category_id,
                "agp_passed": agp_passed,
                "category_created": True,
            })
            committed_rows.append(deepcopy(record))

        for token in sorted(set(self._normalize_tokens(tokens)) - set(ready_tokens)):
            blocked_rows.append({
                "lexical_token": token,
                "blocked_reason": "not_ready_in_round88_freeze",
                "category_created": False,
            })

        after_cm = self._concept_memory_stats()
        after_hg = self._hypergraph_stats()
        return {
            "commit_version": "v3_round89_explicit_concept_commit_smoke",
            "round": 89,
            "source_freeze_version": freeze.get("freeze_version"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": int(freeze.get("candidate_count", 0) or 0),
            "ready_token_count": len(ready_tokens),
            "created_count": len(committed_rows),
            "blocked_count": len(blocked_rows),
            "created_tokens": sorted([str(r.get("lexical_token")) for r in committed_rows]),
            "blocked_tokens": sorted([str(r.get("lexical_token")) for r in blocked_rows if r.get("lexical_token")]),
            "agp_bridge_pass_count": len(agp_pass_tokens),
            "agp_bridge_fail_count": len(agp_fail_tokens),
            "agp_bridge_pass_tokens": sorted(agp_pass_tokens),
            "agp_bridge_fail_tokens": sorted(agp_fail_tokens),
            "concept_memory_delta": int(after_cm.get("concepts_count", 0) or 0) - int(before_cm.get("concepts_count", 0) or 0),
            "hypergraph_stats_before": before_hg,
            "hypergraph_stats_after": after_hg,
            "committed_rows": committed_rows,
            "blocked_rows": blocked_rows,
            "policy": {
                "explicit_commit_call_required": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
                "no_fasttext_seed_mutation": True,
                "no_eve_specific_vector_commit": True,
                "no_auto_promotion": True,
            },
            "read_only": False,
        }



    def _vector_store_stats(self) -> dict[str, Any]:
        store = getattr(self.engine, "eve_specific_vector_store", None)
        if store is None or not hasattr(store, "stats"):
            return {}
        try:
            data = store.stats()
            return dict(data) if isinstance(data, dict) else {}
        except Exception as exc:
            return {"error": exc.__class__.__name__}

    def _wrapper_telemetry_snapshot(self) -> dict[str, Any]:
        wrapper = getattr(self.engine, "self_embedding", None)
        if wrapper is None or not hasattr(wrapper, "telemetry"):
            return {}
        try:
            data = wrapper.telemetry()
            return dict(data) if isinstance(data, dict) else {}
        except Exception as exc:
            return {"error": exc.__class__.__name__}

    def _active_categories_snapshot(self) -> list[str]:
        activation = getattr(self.engine, "activation_adapter", None)
        if activation is None or not hasattr(activation, "all_active_set"):
            return []
        try:
            return sorted(list(activation.all_active_set(threshold=0.0)))
        except Exception:
            return []

    def _concept_memory_has(self, token: str) -> bool:
        cm = getattr(self.engine, "concept_memory", None)
        if cm is None:
            return False
        try:
            if hasattr(cm, "has"):
                return bool(cm.has(token))
            if hasattr(cm, "get_silent"):
                return cm.get_silent(token) is not None
        except Exception:
            return False
        return False

    def concept_commit_delta_replay_report(
        self,
        source_commit_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Round90 read-only replay/delta report for explicit concept commit.

        The report inspects the Round89 explicit concept commit result already
        present in this adapter and replays AGP bridge checks without creating
        categories, concept-memory rows, SA activations, lexical vectors, or
        runtime mapping enforcement.
        """
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()
        source = dict(source_commit_report or {})
        active_set = set(before_active)
        replay_rows: list[dict[str, Any]] = []
        agp_pass_tokens: list[str] = []
        negative_guard_tokens: list[str] = []
        concept_memory_found_tokens: list[str] = []
        sa_found_tokens: list[str] = []

        agp = getattr(self.engine, "agp_adapter", None)
        for category_id, record in sorted(before_categories.items()):
            token = str(record.get("lexical_token") or "")
            cm_found = self._concept_memory_has(token)
            if cm_found:
                concept_memory_found_tokens.append(token)
            sa_found = category_id in active_set
            if sa_found:
                sa_found_tokens.append(token)

            agp_passed = False
            agp_reason = "agp_adapter_missing"
            negative_passed = False
            negative_reason = "agp_adapter_missing"
            if agp is not None and hasattr(agp, "verify"):
                try:
                    result = agp.verify(
                        candidate_response=f"{token} 개념을 확인했어.",
                        meaning={"meaning_categories": [category_id]},
                        activated_categories={category_id: 1.0 if sa_found else 0.0},
                        hormone_state={},
                    )
                    agp_passed = bool(getattr(result, "passed", False))
                    agp_reason = str(getattr(result, "reason", "unknown"))
                except Exception as exc:
                    agp_passed = False
                    agp_reason = exc.__class__.__name__
                try:
                    negative = agp.verify(
                        candidate_response=f"{token} 개념을 확인했어.",
                        meaning={"meaning_categories": [category_id]},
                        activated_categories=[],
                        hormone_state={},
                    )
                    negative_passed = bool(getattr(negative, "passed", False))
                    negative_reason = str(getattr(negative, "reason", "unknown"))
                except Exception as exc:
                    negative_passed = False
                    negative_reason = exc.__class__.__name__
            if agp_passed:
                agp_pass_tokens.append(token)
            if not negative_passed:
                negative_guard_tokens.append(token)

            frame = dict(record.get("frame_hypergraph", {}) or {})
            replay_rows.append({
                "lexical_token": token,
                "category_id": category_id,
                "category_record_present": True,
                "concept_memory_present": cm_found,
                "frame_hypergraph_edge_present": frame.get("hypergraph_edge_id") is not None,
                "frame_slot_count": len(list(frame.get("frame_slots", []) or [])),
                "sa_activation_present": sa_found,
                "agp_replay_called": agp is not None and hasattr(agp, "verify"),
                "agp_replay_passed": agp_passed,
                "agp_replay_reason": agp_reason,
                "agp_negative_without_sa_passed": negative_passed,
                "agp_negative_without_sa_reason": negative_reason,
                "uses_lexical_vector_as_anchor": False,
                "uses_eve_specific_vector_as_anchor": False,
                "uses_seed_vector_as_anchor": False,
                "anchor_source": "explicit_category_plus_sa_activation",
                "read_only_replay": True,
            })

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()
        source_created_tokens = [str(t) for t in list(source.get("created_tokens", []) or []) if str(t).strip()]
        return {
            "report_version": "v3_round90_concept_commit_delta_replay_report",
            "round": 90,
            "source_commit_version": source.get("commit_version", "v3_round89_explicit_concept_commit_smoke"),
            "source_commit_round": source.get("round", 89),
            "source_created_tokens": sorted(source_created_tokens),
            "source_created_count": int(source.get("created_count", len(source_created_tokens)) or 0),
            "source_concept_memory_delta": int(source.get("concept_memory_delta", 0) or 0),
            "source_agp_bridge_pass_count": int(source.get("agp_bridge_pass_count", 0) or 0),
            "category_count": len(after_categories),
            "category_tokens": sorted([str(row.get("lexical_token")) for row in after_categories.values()]),
            "concept_commit_audit_count": len(after_audit),
            "replay_row_count": len(replay_rows),
            "concept_memory_replay_found_count": len(set(concept_memory_found_tokens)),
            "concept_memory_replay_found_tokens": sorted(set(concept_memory_found_tokens)),
            "sa_activation_replay_found_count": len(set(sa_found_tokens)),
            "sa_activation_replay_found_tokens": sorted(set(sa_found_tokens)),
            "agp_replay_pass_count": len(set(agp_pass_tokens)),
            "agp_replay_pass_tokens": sorted(set(agp_pass_tokens)),
            "agp_negative_without_sa_guard_count": len(set(negative_guard_tokens)),
            "agp_negative_without_sa_guard_tokens": sorted(set(negative_guard_tokens)),
            "rows": replay_rows,
            "delta_consistency": {
                "source_created_count_matches_category_count": int(source.get("created_count", len(after_categories)) or 0) == len(after_categories),
                "source_concept_memory_delta_matches_replay": int(source.get("concept_memory_delta", len(set(concept_memory_found_tokens))) or 0) == len(set(concept_memory_found_tokens)),
                "source_agp_pass_count_matches_replay": int(source.get("agp_bridge_pass_count", len(set(agp_pass_tokens))) or 0) == len(set(agp_pass_tokens)),
                "category_tokens_match_source_created_tokens": not source_created_tokens or sorted(set(source_created_tokens)) == sorted([str(row.get("lexical_token")) for row in after_categories.values()]),
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_replay": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_replay": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_replay": before_store == after_store,
                "wrapper_telemetry_unchanged_during_replay": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_replay": before_active == after_active,
                "runtime_mapping_enabled": False,
                "enforcement_enabled": False,
                "category_created_during_replay": False,
                "concept_memory_mutation_during_replay": False,
                "frame_hypergraph_mutation_during_replay": False,
                "sa_activation_created_during_replay": False,
                "eve_specific_vector_commit_called": False,
                "fasttext_seed_mutation": False,
            },
            "policy": {
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
                "replay_is_read_only": True,
            },
            "read_only": True,
        }


    def concept_commit_replay_export_checkpoint_summary(
        self,
        source_commit_report: dict[str, Any] | None = None,
        source_replay_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Round91 read-only checkpoint/export summary for lexical→concept v0.

        Round91 consolidates the Round77~90 lexical→concept path into an
        operator-facing checkpoint. It may replay the existing explicit concept
        commit evidence, but it must not create categories, write concept
        memory/frame rows, create SA activation, commit vectors, call runtime
        mapping, or enable enforcement.
        """
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()

        replay = deepcopy(source_replay_report) if isinstance(source_replay_report, dict) else self.concept_commit_delta_replay_report(source_commit_report=source_commit_report)

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        category_tokens = sorted([str(row.get("lexical_token")) for row in after_categories.values()])
        replay_rows = list(replay.get("rows", []) or [])
        replay_pass_tokens = sorted(set(str(t) for t in list(replay.get("agp_replay_pass_tokens", []) or []) if str(t).strip()))
        negative_guard_tokens = sorted(set(str(t) for t in list(replay.get("agp_negative_without_sa_guard_tokens", []) or []) if str(t).strip()))
        required_artifacts = [
            "LEXICAL_CONCEPT_MAPPING_PLAN_R77.json",
            "LEXICAL_CONCEPT_CANDIDATE_DRY_RUN_R78.json",
            "LEXICAL_CONCEPT_CANDIDATE_EVIDENCE_QUALITY_R79.json",
            "LEXICAL_CONCEPT_PROPOSAL_R80.json",
            "LEXICAL_CONCEPT_MAPPING_GATE_DRY_RUN_R81.json",
            "LEXICAL_CONCEPT_MAPPING_GATE_PROPOSAL_R82.json",
            "LEXICAL_CONCEPT_OPERATOR_ACCEPTANCE_CATEGORY_DRY_RUN_R83.json",
            "LEXICAL_CONCEPT_MEMORY_FRAME_EVIDENCE_DRY_RUN_R84.json",
            "LEXICAL_CONCEPT_SA_ACTIVATION_PATH_DRY_RUN_R85.json",
            "LEXICAL_CONCEPT_AGP_BRIDGE_SMOKE_DRY_RUN_R86.json",
            "LEXICAL_CONCEPT_MAPPING_READINESS_DASHBOARD_R87.json",
            "LEXICAL_CONCEPT_MAPPING_V0_PROPOSAL_FREEZE_R88.json",
            "EVE_CONCEPT_COMMIT_SMOKE_R89.json",
            "EVE_CONCEPT_COMMIT_DELTA_REPLAY_R90.json",
        ]
        next_round_ready = bool(
            category_tokens
            and int(replay.get("concept_memory_replay_found_count", 0) or 0) >= len(category_tokens)
            and int(replay.get("sa_activation_replay_found_count", 0) or 0) >= len(category_tokens)
            and int(replay.get("agp_replay_pass_count", 0) or 0) >= len(category_tokens)
            and int(replay.get("agp_negative_without_sa_guard_count", 0) or 0) >= len(category_tokens)
        )
        return {
            "checkpoint_version": "v3_round91_concept_commit_replay_export_v0_checkpoint",
            "round": 91,
            "axis": "lexical_to_concept_mapping_v0_checkpoint",
            "source_rounds": [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
            "source_replay_version": replay.get("report_version", "v3_round90_concept_commit_delta_replay_report"),
            "source_commit_version": replay.get("source_commit_version", "v3_round89_explicit_concept_commit_smoke"),
            "status": "checkpoint_ready_runtime_mapping_disabled" if next_round_ready else "checkpoint_incomplete_runtime_mapping_disabled",
            "operator_summary": {
                "category_count": len(after_categories),
                "category_tokens": category_tokens,
                "concept_commit_audit_count": len(after_audit),
                "replay_row_count": len(replay_rows),
                "concept_memory_replay_found_count": int(replay.get("concept_memory_replay_found_count", 0) or 0),
                "sa_activation_replay_found_count": int(replay.get("sa_activation_replay_found_count", 0) or 0),
                "agp_replay_pass_count": int(replay.get("agp_replay_pass_count", 0) or 0),
                "agp_negative_without_sa_guard_count": int(replay.get("agp_negative_without_sa_guard_count", 0) or 0),
                "explicit_concept_commit_tokens": category_tokens,
                "next_recommended_round": "round92_runtime_mapping_gate_dry_run" if next_round_ready else "collect_more_concept_commit_evidence_before_runtime_mapping_dry_run",
            },
            "artifact_manifest": {
                "required_artifacts": required_artifacts,
                "artifact_count": len(required_artifacts),
                "operator_export_artifact": "EVE_CONCEPT_COMMIT_REPLAY_EXPORT_CHECKPOINT_R91.json",
            },
            "replay_rows": deepcopy(replay_rows),
            "replay_guards": {
                "agp_replay_pass_tokens": replay_pass_tokens,
                "agp_negative_without_sa_guard_tokens": negative_guard_tokens,
                "explicit_category_without_sa_fails_agp": bool(negative_guard_tokens),
                "lexical_vector_used_as_anchor": False,
                "eve_specific_vector_used_as_anchor": False,
                "seed_vector_used_as_anchor": False,
                "agp_anchor_source": "explicit_category_plus_sa_activation_only",
            },
            "runtime_policy": {
                "runtime_mapping_enabled": False,
                "enforcement_enabled": False,
                "auto_category_creation_enabled": False,
                "broad_lexical_to_concept_mapping_enabled": False,
                "next_round_is_dry_run_only": True,
                "explicit_operator_gate_required_before_runtime_mapping": True,
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_checkpoint": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_checkpoint": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_checkpoint": before_store == after_store,
                "wrapper_telemetry_unchanged_during_checkpoint": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_checkpoint": before_active == after_active,
                "category_created_during_checkpoint": False,
                "concept_memory_mutation_during_checkpoint": False,
                "frame_hypergraph_mutation_during_checkpoint": False,
                "sa_activation_created_during_checkpoint": False,
                "eve_specific_vector_commit_called": False,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
            },
            "policy": {
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "checkpoint_is_read_only": True,
            },
            "read_only": True,
        }



    def runtime_mapping_gate_dry_run(
        self,
        tokens: Iterable[str] | None = None,
        source_checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Round92 read-only runtime lexical→concept mapping gate dry-run.

        This evaluates which lexical tokens *would* map to already committed
        concept categories if runtime mapping were enabled by a future explicit
        patch. It does not enable runtime mapping, does not create categories,
        does not mutate concept memory/frame/hypergraph/SA, does not call AGP
        verify, and does not perform embedding lookup.
        """
        self._latest_runtime_mapping_round = 94
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()

        checkpoint = deepcopy(source_checkpoint) if isinstance(source_checkpoint, dict) else None
        checkpoint_summary: dict[str, Any] = {}
        replay_guards: dict[str, Any] = {}
        if checkpoint:
            checkpoint_summary = dict(checkpoint.get("operator_summary", {}) or {})
            replay_guards = dict(checkpoint.get("replay_guards", {}) or {})
        else:
            # Build a local, read-only checkpoint summary when the caller did not
            # provide one. This path is still not runtime mapping; it only reuses
            # the existing checkpoint/report surface.
            checkpoint = self.concept_commit_replay_export_checkpoint_summary()
            checkpoint_summary = dict(checkpoint.get("operator_summary", {}) or {})
            replay_guards = dict(checkpoint.get("replay_guards", {}) or {})

        category_tokens = sorted({str(t) for t in list(checkpoint_summary.get("category_tokens", []) or []) if str(t).strip()})
        requested_tokens = self._normalize_tokens(tokens) if tokens is not None else category_tokens
        # Keep dry-run deterministic and useful for operator review. Include
        # category tokens even if a caller asks about extra tokens only.
        eval_tokens = sorted(set(requested_tokens) | set(category_tokens))

        agp_pass_tokens = set(str(t) for t in list(replay_guards.get("agp_replay_pass_tokens", []) or []) if str(t).strip())
        negative_guard_tokens = set(str(t) for t in list(replay_guards.get("agp_negative_without_sa_guard_tokens", []) or []) if str(t).strip())
        active_set = set(before_active)
        audit_category_ids = {str(r.get("category_id")) for r in before_audit if str(r.get("category_id", "")).strip()}

        rows: list[dict[str, Any]] = []
        would_map_tokens: list[str] = []
        blocked_tokens: list[str] = []
        reason_counts: dict[str, int] = {}

        for token in eval_tokens:
            category_id = f"concept_category::lex::{token}"
            category_record = dict(before_categories.get(category_id, {}) or {})
            category_exists = bool(category_record)
            concept_memory_present = self._concept_memory_has(token)
            sa_activation_present = category_id in active_set
            agp_replay_passed = token in agp_pass_tokens
            negative_without_sa_guard = token in negative_guard_tokens
            commit_audit_present = category_id in audit_category_ids

            blocked_reasons: list[str] = []
            if not category_exists:
                blocked_reasons.append("explicit_category_missing")
            if not commit_audit_present:
                blocked_reasons.append("concept_commit_audit_missing")
            if not concept_memory_present:
                blocked_reasons.append("concept_memory_missing")
            if not sa_activation_present:
                blocked_reasons.append("sa_activation_missing")
            if not agp_replay_passed:
                blocked_reasons.append("agp_replay_pass_missing")
            if not negative_without_sa_guard:
                blocked_reasons.append("agp_negative_without_sa_guard_missing")

            would_map = not blocked_reasons
            if would_map:
                would_map_tokens.append(token)
                gate_status = "would_map_if_runtime_mapping_enabled"
            else:
                blocked_tokens.append(token)
                gate_status = "blocked_from_runtime_mapping"
                for reason in blocked_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

            rows.append({
                "lexical_token": token,
                "category_id": category_id,
                "gate_status": gate_status,
                "would_map_if_runtime_mapping_enabled": would_map,
                "runtime_mapping_applied_now": False,
                "runtime_mapping_enabled_now": False,
                "enforcement_enabled_now": False,
                "category_record_present": category_exists,
                "concept_commit_audit_present": commit_audit_present,
                "concept_memory_present": concept_memory_present,
                "sa_activation_present": sa_activation_present,
                "agp_replay_passed": agp_replay_passed,
                "agp_negative_without_sa_guard_present": negative_without_sa_guard,
                "blocked_reasons": blocked_reasons,
                "mapping_target_category_id": category_id if would_map else None,
                "uses_lexical_vector_as_anchor": False,
                "uses_eve_specific_vector_as_anchor": False,
                "uses_seed_vector_as_anchor": False,
                "anchor_source_required_later": "explicit_category_plus_sa_activation_only",
            })

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        return {
            "dry_run_version": "v3_round92_runtime_mapping_gate_dry_run",
            "round": 92,
            "source_checkpoint_version": (checkpoint or {}).get("checkpoint_version", "v3_round91_concept_commit_replay_export_v0_checkpoint"),
            "source_replay_version": (checkpoint or {}).get("source_replay_version", "v3_round90_concept_commit_delta_replay_report"),
            "source_commit_version": (checkpoint or {}).get("source_commit_version", "v3_round89_explicit_concept_commit_smoke"),
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(eval_tokens),
            "category_count": len(before_categories),
            "would_map_count": len(would_map_tokens),
            "would_block_count": len(blocked_tokens),
            "would_map_tokens": sorted(would_map_tokens),
            "blocked_tokens": sorted(blocked_tokens),
            "blocked_reason_counts": dict(sorted(reason_counts.items())),
            "rows": rows,
            "operator_summary": {
                "next_recommended_round": "round93_runtime_mapping_proposal_report" if would_map_tokens else "collect_more_concept_commit_evidence_before_runtime_mapping_proposal",
                "runtime_mapping_can_be_considered": bool(would_map_tokens),
                "runtime_mapping_should_be_enabled_now": False,
                "explicit_operator_gate_required_before_enforcement": True,
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_dry_run": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_dry_run": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_dry_run": before_store == after_store,
                "wrapper_telemetry_unchanged_during_dry_run": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_dry_run": before_active == after_active,
                "category_created_during_dry_run": False,
                "concept_memory_mutation_during_dry_run": False,
                "frame_hypergraph_mutation_during_dry_run": False,
                "sa_activation_created_during_dry_run": False,
                "agp_verify_called_during_runtime_mapping_dry_run": False,
                "embedding_lookup_called_during_dry_run": False,
                "eve_specific_vector_commit_called": False,
                "runtime_mapping_applied": False,
            },
            "policy": {
                "runtime_mapping_gate_dry_run_only": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "no_auto_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_sa_activation_creation": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def runtime_mapping_proposal_report(
        self,
        tokens: Iterable[str] | None = None,
        source_dry_run: dict[str, Any] | None = None,
        source_checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Round93 read-only runtime lexical→concept mapping proposal report.

        This consolidates the Round92 dry-run into an operator review surface.
        It does not enable runtime mapping, does not enforce mapping, does not
        create categories, does not mutate concept memory/frame/hypergraph/SA,
        does not call AGP verify, and does not perform embedding lookup.
        """
        self._latest_runtime_mapping_round = 94
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()

        dry_run = deepcopy(source_dry_run) if isinstance(source_dry_run, dict) else None
        if dry_run is None:
            dry_run = self.runtime_mapping_gate_dry_run(tokens=tokens, source_checkpoint=source_checkpoint)

        rows = list(dry_run.get("rows", []) or [])
        proposal_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        action_items: list[dict[str, Any]] = []

        for row in rows:
            lexical_token = str(row.get("lexical_token", "")).strip()
            if not lexical_token:
                continue
            would_map = bool(row.get("would_map_if_runtime_mapping_enabled", False))
            category_id = str(row.get("mapping_target_category_id") or row.get("target_category_id") or row.get("category_id") or f"concept_category::lex::{lexical_token}")
            if would_map:
                proposal_rows.append({
                    "lexical_token": lexical_token,
                    "target_category_id": category_id,
                    "proposal_status": "operator_review_required",
                    "would_map_if_runtime_mapping_enabled": True,
                    "runtime_mapping_enabled_now": False,
                    "enforcement_enabled_now": False,
                    "may_enable_now": False,
                    "requires_explicit_operator_acceptance": True,
                    "requires_separate_enforcement_round": True,
                    "requires_split_full_suite_before_enforcement": True,
                    "anchor_source_required": "explicit_category_plus_sa_activation_only",
                    "uses_lexical_vector_as_anchor": False,
                    "uses_eve_specific_vector_as_anchor": False,
                    "uses_seed_vector_as_anchor": False,
                })
                action_items.append({
                    "lexical_token": lexical_token,
                    "target_category_id": category_id,
                    "action": "operator_review_runtime_mapping_candidate",
                    "decision_options": ["keep_disabled", "approve_future_enforcement_dry_run", "reject_candidate"],
                    "safe_default": "keep_disabled",
                })
            else:
                blocked_rows.append({
                    "lexical_token": lexical_token,
                    "target_category_id": category_id,
                    "proposal_status": "blocked_from_runtime_mapping_proposal",
                    "blocked_reasons": list(row.get("blocked_reasons", []) or []),
                    "would_map_if_runtime_mapping_enabled": False,
                    "runtime_mapping_enabled_now": False,
                })

        would_map_tokens = sorted([r["lexical_token"] for r in proposal_rows])
        blocked_tokens = sorted([r["lexical_token"] for r in blocked_rows])

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        return {
            "proposal_version": "v3_round93_runtime_mapping_proposal_report",
            "round": 93,
            "source_dry_run_version": dry_run.get("dry_run_version", "v3_round92_runtime_mapping_gate_dry_run"),
            "source_checkpoint_version": dry_run.get("source_checkpoint_version", "v3_round91_concept_commit_replay_export_v0_checkpoint"),
            "source_commit_version": dry_run.get("source_commit_version", "v3_round89_explicit_concept_commit_smoke"),
            "axis": "runtime_lexical_to_concept_mapping_proposal",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "proposal_count": len(proposal_rows),
            "blocked_count": len(blocked_rows),
            "candidate_count": len(rows),
            "would_map_tokens": would_map_tokens,
            "blocked_tokens": blocked_tokens,
            "proposal_rows": proposal_rows,
            "blocked_rows": blocked_rows,
            "operator_action_items": action_items,
            "operator_action_item_count": len(action_items),
            "operator_summary": {
                "runtime_mapping_can_be_considered": bool(proposal_rows),
                "runtime_mapping_should_be_enabled_now": False,
                "next_recommended_round": "round94_runtime_mapping_enforcement_dry_run" if proposal_rows else "collect_more_concept_commit_evidence_before_runtime_mapping_enforcement_dry_run",
                "safe_default": "keep_runtime_mapping_disabled",
                "explicit_operator_gate_required_before_enforcement": True,
                "split_full_suite_required_before_enforcement": True,
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_proposal": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_proposal": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_proposal": before_store == after_store,
                "wrapper_telemetry_unchanged_during_proposal": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_proposal": before_active == after_active,
                "category_created_during_proposal": False,
                "concept_memory_mutation_during_proposal": False,
                "frame_hypergraph_mutation_during_proposal": False,
                "sa_activation_created_during_proposal": False,
                "agp_verify_called_during_runtime_mapping_proposal": False,
                "embedding_lookup_called_during_proposal": False,
                "eve_specific_vector_commit_called": False,
                "runtime_mapping_applied": False,
            },
            "policy": {
                "runtime_mapping_proposal_report_only": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "no_auto_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_sa_activation_creation": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def runtime_mapping_enforcement_dry_run(
        self,
        tokens: Iterable[str] | None = None,
        source_proposal: dict[str, Any] | None = None,
        source_dry_run: dict[str, Any] | None = None,
        source_checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Round94 read-only runtime mapping enforcement dry-run.

        This simulates the API/result shape that a future runtime
        lexical→concept mapper would expose after an explicit enforcement round.
        It still does not enable runtime mapping, does not install a mapper,
        does not create categories, does not mutate concept memory/frame/SA,
        does not call AGP verify, and does not perform embedding lookup.
        """
        self._latest_runtime_mapping_round = 96
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()

        proposal = deepcopy(source_proposal) if isinstance(source_proposal, dict) else None
        if proposal is None:
            proposal = self.runtime_mapping_proposal_report(
                tokens=tokens,
                source_dry_run=source_dry_run,
                source_checkpoint=source_checkpoint,
            )

        requested_tokens = self._normalize_tokens(tokens) if tokens is not None else []
        proposal_rows = list(proposal.get("proposal_rows", []) or [])
        blocked_rows = list(proposal.get("blocked_rows", []) or [])
        proposal_tokens = [str(row.get("lexical_token", "")).strip() for row in proposal_rows]
        blocked_tokens_from_proposal = [str(row.get("lexical_token", "")).strip() for row in blocked_rows]
        eval_tokens = sorted({t for t in (requested_tokens + proposal_tokens + blocked_tokens_from_proposal) if t})

        proposal_by_token = {
            str(row.get("lexical_token", "")).strip(): dict(row)
            for row in proposal_rows
            if str(row.get("lexical_token", "")).strip()
        }
        blocked_by_token = {
            str(row.get("lexical_token", "")).strip(): dict(row)
            for row in blocked_rows
            if str(row.get("lexical_token", "")).strip()
        }

        enforcement_rows: list[dict[str, Any]] = []
        would_apply_tokens: list[str] = []
        would_block_tokens: list[str] = []
        reason_counts: dict[str, int] = {}

        for token in eval_tokens:
            proposal_row = proposal_by_token.get(token)
            blocked_row = blocked_by_token.get(token)
            target_category_id = str(
                (proposal_row or {}).get("target_category_id")
                or (blocked_row or {}).get("target_category_id")
                or f"concept_category::lex::{token}"
            )
            category_record = dict(before_categories.get(target_category_id, {}) or {})
            proposal_ready = bool(proposal_row and proposal_row.get("would_map_if_runtime_mapping_enabled", False))
            blocked_reasons: list[str] = []
            if not proposal_ready:
                blocked_reasons.extend(list((blocked_row or {}).get("blocked_reasons", []) or []))
                if not blocked_reasons:
                    blocked_reasons.append("runtime_mapping_proposal_missing")
            if not category_record:
                blocked_reasons.append("target_category_record_missing")
            if not self._concept_memory_has(token):
                blocked_reasons.append("concept_memory_missing")
            if target_category_id not in set(before_active):
                blocked_reasons.append("sa_activation_missing")

            # Deduplicate while preserving deterministic order.
            seen: set[str] = set()
            blocked_reasons = [r for r in blocked_reasons if not (r in seen or seen.add(r))]
            would_apply = not blocked_reasons
            if would_apply:
                would_apply_tokens.append(token)
                status = "would_apply_if_runtime_mapping_enabled"
                simulated_result = {
                    "lexical_token": token,
                    "category_id": target_category_id,
                    "category_label": category_record.get("label", token),
                    "mapping_status": "simulated_runtime_mapping_success",
                    "anchor_source": "explicit_category_plus_sa_activation_only",
                }
            else:
                would_block_tokens.append(token)
                status = "blocked_from_runtime_mapping_enforcement"
                simulated_result = None
                for reason in blocked_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

            enforcement_rows.append({
                "lexical_token": token,
                "target_category_id": target_category_id,
                "enforcement_status": status,
                "would_apply_if_runtime_mapping_enabled": would_apply,
                "runtime_mapping_enabled_now": False,
                "enforcement_enabled_now": False,
                "runtime_mapping_applied_now": False,
                "api_path_dry_run": [
                    "RuntimeLexConceptMapper.resolve(token)",
                    "LexConceptMappingAdapter.runtime_mapping_gate",
                    "ConceptCategoryRegistry.lookup(category_id)",
                    "SAActivation.check(category_id)",
                    "return MappingResult(category_id)",
                ] if would_apply else [],
                "simulated_mapping_result": simulated_result,
                "blocked_reasons": blocked_reasons,
                "uses_lexical_vector_as_anchor": False,
                "uses_eve_specific_vector_as_anchor": False,
                "uses_seed_vector_as_anchor": False,
                "anchor_source_required": "explicit_category_plus_sa_activation_only",
            })

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        return {
            "dry_run_version": "v3_round94_runtime_mapping_enforcement_dry_run",
            "round": 94,
            "source_proposal_version": proposal.get("proposal_version", "v3_round93_runtime_mapping_proposal_report"),
            "source_dry_run_version": proposal.get("source_dry_run_version", "v3_round92_runtime_mapping_gate_dry_run"),
            "axis": "runtime_lexical_to_concept_mapping_enforcement_dry_run",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(eval_tokens),
            "would_apply_count": len(would_apply_tokens),
            "would_block_count": len(would_block_tokens),
            "would_apply_tokens": sorted(would_apply_tokens),
            "would_block_tokens": sorted(would_block_tokens),
            "blocked_reason_counts": dict(sorted(reason_counts.items())),
            "enforcement_rows": enforcement_rows,
            "operator_summary": {
                "runtime_mapping_enforcement_can_be_considered": bool(would_apply_tokens),
                "runtime_mapping_should_be_enabled_now": False,
                "next_recommended_round": "round95_runtime_mapping_operator_acceptance_fixture" if would_apply_tokens else "collect_more_concept_commit_evidence_before_runtime_mapping_operator_acceptance",
                "safe_default": "keep_runtime_mapping_disabled",
                "separate_mutation_round_required_for_enablement": True,
                "split_full_suite_required_before_enablement": True,
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_enforcement_dry_run": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_enforcement_dry_run": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_enforcement_dry_run": before_store == after_store,
                "wrapper_telemetry_unchanged_during_enforcement_dry_run": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_enforcement_dry_run": before_active == after_active,
                "category_created_during_enforcement_dry_run": False,
                "concept_memory_mutation_during_enforcement_dry_run": False,
                "frame_hypergraph_mutation_during_enforcement_dry_run": False,
                "sa_activation_created_during_enforcement_dry_run": False,
                "agp_verify_called_during_runtime_mapping_enforcement_dry_run": False,
                "embedding_lookup_called_during_enforcement_dry_run": False,
                "runtime_mapping_applied": False,
                "runtime_mapping_enabled_now": False,
                "enforcement_enabled_now": False,
                "eve_specific_vector_commit_called": False,
            },
            "policy": {
                "runtime_mapping_enforcement_dry_run_only": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "no_auto_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_sa_activation_creation": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }



    def runtime_mapping_enable_smoke_precheck(
        self,
        source_fixture: dict[str, Any] | None = None,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
        source_enforcement: dict[str, Any] | None = None,
        source_proposal: dict[str, Any] | None = None,
        source_dry_run: dict[str, Any] | None = None,
        source_checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a read-only Round96 pre-mutation enable smoke checklist.

        This is not the activation round. It packages the exact gates, rollback
        conditions, and accepted runtime-mapping candidates that a separate
        controlled mutation round would need before runtime mapping can be
        enabled.
        """
        self._latest_runtime_mapping_round = 96
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()

        fixture = source_fixture
        if not isinstance(fixture, dict):
            fixture = self.runtime_mapping_operator_acceptance_fixture(
                tokens=tokens,
                accepted_tokens=accepted_tokens,
                source_enforcement=source_enforcement,
                source_proposal=source_proposal,
                source_dry_run=source_dry_run,
                source_checkpoint=source_checkpoint,
            )

        accepted_rows = list(fixture.get("accepted_rows", []) or [])
        ready_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        for row in accepted_rows:
            checks = {
                "operator_accepted": bool(row.get("operator_acceptance_fixture", False)),
                "has_target_category_id": bool(row.get("target_category_id")),
                "has_simulated_mapping_result": isinstance(row.get("simulated_mapping_result"), dict),
                "would_apply_after_fixture": bool(row.get("would_apply_runtime_mapping_after_fixture", False)),
                "uses_lexical_vector_as_anchor": bool(row.get("uses_lexical_vector_as_anchor", False)),
                "uses_eve_specific_vector_as_anchor": bool(row.get("uses_eve_specific_vector_as_anchor", False)),
                "uses_seed_vector_as_anchor": bool(row.get("uses_seed_vector_as_anchor", False)),
            }
            reasons: list[str] = []
            if not checks["operator_accepted"]:
                reasons.append("operator_acceptance_fixture_missing")
            if not checks["has_target_category_id"]:
                reasons.append("target_category_id_missing")
            if not checks["has_simulated_mapping_result"]:
                reasons.append("simulated_mapping_result_missing")
            if not checks["would_apply_after_fixture"]:
                reasons.append("would_apply_after_fixture_false")
            if checks["uses_lexical_vector_as_anchor"]:
                reasons.append("lexical_vector_anchor_forbidden")
            if checks["uses_eve_specific_vector_as_anchor"]:
                reasons.append("eve_specific_vector_anchor_forbidden")
            if checks["uses_seed_vector_as_anchor"]:
                reasons.append("seed_vector_anchor_forbidden")

            precheck_row = deepcopy(row)
            precheck_row.update(
                {
                    "precheck_status": "ready_for_separate_enable_smoke_round" if not reasons else "blocked_from_enable_smoke",
                    "precheck_checks": checks,
                    "precheck_blocked_reasons": sorted(reasons),
                    "runtime_mapping_enabled_now": False,
                    "enforcement_enabled_now": False,
                    "runtime_mapping_applied_now": False,
                }
            )
            if reasons:
                blocked_rows.append(precheck_row)
            else:
                ready_rows.append(precheck_row)

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        return {
            "precheck_version": "v3_round96_runtime_mapping_enable_smoke_precheck",
            "round": 96,
            "source_fixture_version": fixture.get("fixture_version", "v3_round95_runtime_mapping_operator_acceptance_fixture"),
            "axis": "runtime_lexical_to_concept_mapping_enable_smoke_precheck",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(accepted_rows),
            "ready_count": len(ready_rows),
            "blocked_count": len(blocked_rows),
            "ready_tokens": sorted(str(row.get("lexical_token")) for row in ready_rows),
            "blocked_tokens": sorted(str(row.get("lexical_token")) for row in blocked_rows),
            "ready_rows": ready_rows,
            "blocked_rows": blocked_rows,
            "pre_mutation_gate": {
                "operator_approval_required_for_actual_enable": True,
                "pre_mutation_checkpoint_required": True,
                "rollback_plan_required": True,
                "focused_tests_required": True,
                "adjacent_tests_required": True,
                "collect_only_required": True,
                "compileall_required": True,
                "split_full_suite_required_before_persistence": True,
            },
            "rollback_plan": {
                "disable_runtime_mapping_flag": "runtime_mapping_enabled=False",
                "disable_enforcement_flag": "enforcement_enabled=False",
                "restore_mapping_table_from_checkpoint": True,
                "discard_unpersisted_runtime_cache": True,
                "keep_operator_fixture_as_read_only_audit": True,
            },
            "operator_recommendation": {
                "may_attempt_separate_enable_smoke": bool(ready_rows) and not blocked_rows,
                "runtime_mapping_should_be_enabled_now": False,
                "safe_default": "do_not_enable_in_precheck_round",
                "next_recommended_round": "round97_controlled_runtime_mapping_enable_smoke" if ready_rows and not blocked_rows else "repair_precheck_blocks_before_enable_smoke",
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_enable_smoke_precheck": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_enable_smoke_precheck": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_enable_smoke_precheck": before_store == after_store,
                "wrapper_telemetry_unchanged_during_enable_smoke_precheck": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_enable_smoke_precheck": before_active == after_active,
                "category_created_during_enable_smoke_precheck": False,
                "concept_memory_mutation_during_enable_smoke_precheck": False,
                "frame_hypergraph_mutation_during_enable_smoke_precheck": False,
                "sa_activation_created_during_enable_smoke_precheck": False,
                "agp_verify_called_during_enable_smoke_precheck": False,
                "embedding_lookup_called_during_enable_smoke_precheck": False,
                "runtime_mapping_applied": False,
                "runtime_mapping_enabled_now": False,
                "enforcement_enabled_now": False,
                "eve_specific_vector_commit_called": False,
            },
            "policy": {
                "enable_smoke_precheck_only": True,
                "actual_enable_deferred_to_separate_round": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def controlled_runtime_mapping_enable_smoke(
        self,
        source_precheck: dict[str, Any] | None = None,
        source_fixture: dict[str, Any] | None = None,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Round97 controlled, non-persistent runtime mapping enable smoke.

        The smoke opens runtime mapping only inside this method, maps the
        Round96-ready operator accepted rows, then rolls the flag and ephemeral
        mapping table back before returning. Enforcement remains disabled.
        """
        self._latest_runtime_mapping_round = 97
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()
        before_runtime_enabled = bool(getattr(self, "runtime_mapping_enabled", False))
        before_enforcement_enabled = bool(getattr(self, "enforcement_enabled", False))

        precheck = source_precheck
        if not isinstance(precheck, dict):
            precheck = self.runtime_mapping_enable_smoke_precheck(
                source_fixture=source_fixture,
                tokens=tokens,
                accepted_tokens=accepted_tokens,
            )

        ready_rows = [deepcopy(row) for row in list(precheck.get("ready_rows", []) or [])]
        smoke_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        ephemeral_mapping_table: dict[str, str] = {}
        runtime_mapping_enabled_during_smoke = False
        rollback_complete = False

        try:
            setattr(self, "runtime_mapping_enabled", True)
            setattr(self, "enforcement_enabled", False)
            runtime_mapping_enabled_during_smoke = bool(getattr(self, "runtime_mapping_enabled", False))
            for row in ready_rows:
                token = str(row.get("lexical_token", "")).strip()
                category_id = str(row.get("mapping_target_category_id") or row.get("target_category_id") or row.get("category_id") or "").strip()
                reasons: list[str] = []
                if not token:
                    reasons.append("lexical_token_missing")
                if not category_id:
                    reasons.append("mapping_target_category_id_missing")
                if category_id and category_id not in before_categories:
                    reasons.append("explicit_category_missing")
                if bool(row.get("uses_lexical_vector_as_anchor", False)):
                    reasons.append("lexical_vector_anchor_forbidden")
                if bool(row.get("uses_eve_specific_vector_as_anchor", False)):
                    reasons.append("eve_specific_vector_anchor_forbidden")
                if bool(row.get("uses_seed_vector_as_anchor", False)):
                    reasons.append("seed_vector_anchor_forbidden")

                smoke = deepcopy(row)
                smoke.update({
                    "round97_smoke": True,
                    "runtime_mapping_enabled_during_smoke": runtime_mapping_enabled_during_smoke,
                    "enforcement_enabled_during_smoke": False,
                    "runtime_mapping_persisted": False,
                    "anchor_source": "explicit_category_plus_sa_activation_only",
                    "agp_verify_called_during_runtime_mapping_enable_smoke": False,
                    "embedding_lookup_called_during_enable_smoke": False,
                    "eve_specific_vector_commit_called": False,
                    "category_created_now": False,
                    "concept_memory_mutation_now": False,
                    "frame_hypergraph_mutation_now": False,
                    "sa_activation_created_now": False,
                })
                if reasons:
                    smoke.update({
                        "smoke_status": "blocked_during_controlled_enable_smoke",
                        "smoke_blocked_reasons": sorted(set(reasons)),
                        "runtime_mapping_applied_during_smoke": False,
                    })
                    blocked_rows.append(smoke)
                    continue

                ephemeral_mapping_table[token] = category_id
                smoke.update({
                    "smoke_status": "controlled_runtime_mapping_success",
                    "smoke_blocked_reasons": [],
                    "runtime_mapping_applied_during_smoke": True,
                    "mapping_result": {
                        "lexical_token": token,
                        "category_id": category_id,
                        "mapping_status": "ephemeral_runtime_mapping_success",
                        "persisted": False,
                        "anchor_source": "explicit_category_plus_sa_activation_only",
                    },
                })
                smoke_rows.append(smoke)
        finally:
            ephemeral_mapping_table.clear()
            setattr(self, "runtime_mapping_enabled", before_runtime_enabled)
            setattr(self, "enforcement_enabled", before_enforcement_enabled)
            rollback_complete = (
                bool(getattr(self, "runtime_mapping_enabled", False)) == before_runtime_enabled
                and bool(getattr(self, "enforcement_enabled", False)) == before_enforcement_enabled
                and not ephemeral_mapping_table
            )

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        return {
            "smoke_version": "v3_round97_controlled_runtime_mapping_enable_smoke",
            "round": 97,
            "source_precheck_version": precheck.get("precheck_version", "v3_round96_runtime_mapping_enable_smoke_precheck"),
            "axis": "controlled_runtime_lexical_to_concept_mapping_enable_smoke",
            "runtime_mapping_enabled_before": before_runtime_enabled,
            "runtime_mapping_enabled_during_smoke": runtime_mapping_enabled_during_smoke,
            "runtime_mapping_enabled_after_rollback": bool(getattr(self, "runtime_mapping_enabled", False)),
            "enforcement_enabled_before": before_enforcement_enabled,
            "enforcement_enabled_during_smoke": False,
            "enforcement_enabled_after_rollback": bool(getattr(self, "enforcement_enabled", False)),
            "candidate_count": len(ready_rows),
            "mapped_count": len(smoke_rows),
            "blocked_count": len(blocked_rows),
            "mapped_tokens": sorted(str(row.get("lexical_token")) for row in smoke_rows),
            "blocked_tokens": sorted(str(row.get("lexical_token")) for row in blocked_rows),
            "smoke_rows": smoke_rows,
            "blocked_rows": blocked_rows,
            "rollback": {
                "ephemeral_mapping_table_cleared": True,
                "runtime_mapping_flag_restored": bool(getattr(self, "runtime_mapping_enabled", False)) == before_runtime_enabled,
                "enforcement_flag_restored": bool(getattr(self, "enforcement_enabled", False)) == before_enforcement_enabled,
                "rollback_complete": rollback_complete,
            },
            "post_smoke_recommendation": {
                "may_attempt_persistence_gate_next_round": bool(smoke_rows) and not blocked_rows and rollback_complete,
                "next_recommended_round": "round98_runtime_mapping_persistence_gate_audit" if bool(smoke_rows) and not blocked_rows and rollback_complete else "repair_round97_smoke_blocks",
                "runtime_mapping_should_remain_disabled_by_default": True,
            },
            "mutation_checks": {
                "category_snapshot_unchanged_during_enable_smoke": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_enable_smoke": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_enable_smoke": before_store == after_store,
                "wrapper_telemetry_unchanged_during_enable_smoke": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_enable_smoke": before_active == after_active,
                "category_created_during_enable_smoke": False,
                "concept_memory_mutation_during_enable_smoke": False,
                "frame_hypergraph_mutation_during_enable_smoke": False,
                "sa_activation_created_during_enable_smoke": False,
                "agp_verify_called_during_enable_smoke": False,
                "embedding_lookup_called_during_enable_smoke": False,
                "runtime_mapping_persisted": False,
                "enforcement_enabled_now": bool(getattr(self, "enforcement_enabled", False)),
                "eve_specific_vector_commit_called": False,
            },
            "policy": {
                "controlled_enable_smoke_only": True,
                "runtime_mapping_ephemeral_only": True,
                "enforcement_still_disabled": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": False,
        }


    def runtime_mapping_persistence_gate_audit(
        self,
        source_smoke: dict[str, Any] | None = None,
        source_precheck: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Round98 read-only audit after the Round97 controlled enable smoke."""
        self._latest_runtime_mapping_round = 98
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()
        smoke = source_smoke if isinstance(source_smoke, dict) else self.controlled_runtime_mapping_enable_smoke(source_precheck=source_precheck)
        mapped_rows = list(smoke.get("smoke_rows", []) or [])
        blocked_rows = list(smoke.get("blocked_rows", []) or [])
        rollback = dict(smoke.get("rollback", {}) or {})
        hard_blocks: list[str] = []
        if blocked_rows:
            hard_blocks.append("round97_blocked_rows_present")
        if not rollback.get("rollback_complete", False):
            hard_blocks.append("round97_rollback_incomplete")
        if bool(getattr(self, "runtime_mapping_enabled", False)):
            hard_blocks.append("runtime_mapping_still_enabled_after_smoke")
        if bool(getattr(self, "enforcement_enabled", False)):
            hard_blocks.append("enforcement_enabled_after_smoke")

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        return {
            "audit_version": "v3_round98_runtime_mapping_persistence_gate_audit",
            "round": 98,
            "source_smoke_version": smoke.get("smoke_version", "v3_round97_controlled_runtime_mapping_enable_smoke"),
            "runtime_mapping_enabled": bool(getattr(self, "runtime_mapping_enabled", False)),
            "enforcement_enabled": bool(getattr(self, "enforcement_enabled", False)),
            "mapped_count": len(mapped_rows),
            "blocked_count": len(blocked_rows),
            "hard_stop": bool(hard_blocks),
            "hard_stop_reasons": hard_blocks,
            "persistence_gate_status": "ready_for_operator_persistence_decision" if mapped_rows and not hard_blocks else "blocked_or_needs_more_evidence",
            "operator_recommendation": {
                "persist_runtime_mapping_now": False,
                "requires_operator_approval": True,
                "requires_split_full_suite_with_medium_vectors": True,
                "default_action": "keep_runtime_mapping_disabled_until_operator_persistence_round",
                "next_recommended_round": "operator_persistence_decision_with_full_validation" if mapped_rows and not hard_blocks else "repair_persistence_gate_blocks",
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_persistence_gate_audit": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_persistence_gate_audit": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_persistence_gate_audit": before_store == after_store,
                "wrapper_telemetry_unchanged_during_persistence_gate_audit": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_persistence_gate_audit": before_active == after_active,
                "category_created_during_persistence_gate_audit": False,
                "concept_memory_mutation_during_persistence_gate_audit": False,
                "frame_hypergraph_mutation_during_persistence_gate_audit": False,
                "sa_activation_created_during_persistence_gate_audit": False,
                "agp_verify_called_during_persistence_gate_audit": False,
                "embedding_lookup_called_during_persistence_gate_audit": False,
                "runtime_mapping_persisted": False,
            },
            "policy": {
                "persistence_gate_audit_only": True,
                "runtime_mapping_still_disabled_by_default": True,
                "enforcement_still_disabled": True,
                "lexical_vector_is_evidence_only": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def concept_commit_records(self) -> list[dict[str, Any]]:
        return deepcopy(self._concept_commit_audit)

    def concept_categories_snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._concept_categories)


    def runtime_mapping_operator_acceptance_fixture(
        self,
        tokens: Iterable[str] | None = None,
        accepted_tokens: Iterable[str] | None = None,
        source_enforcement: dict[str, Any] | None = None,
        source_proposal: dict[str, Any] | None = None,
        source_dry_run: dict[str, Any] | None = None,
        source_checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a read-only Round95 operator acceptance fixture.

        Round95 converts the Round94 enforcement dry-run into an explicit
        operator-accepted fixture shape. It does not enable runtime mapping,
        does not enable enforcement, and does not mutate category/memory/SA/AGP
        state. This artifact is intentionally a proof object for the next
        mutation round rather than the mutation itself.
        """
        self._latest_runtime_mapping_round = 96
        before_categories = self.concept_categories_snapshot()
        before_audit = self.concept_commit_records()
        before_store = self._vector_store_stats()
        before_telemetry = self._wrapper_telemetry_snapshot()
        before_active = self._active_categories_snapshot()

        enforcement = source_enforcement
        if not isinstance(enforcement, dict):
            enforcement = self.runtime_mapping_enforcement_dry_run(
                tokens=tokens,
                source_proposal=source_proposal,
                source_dry_run=source_dry_run,
                source_checkpoint=source_checkpoint,
            )

        enforcement_rows = list(enforcement.get("enforcement_rows", []) or [])
        if tokens is not None:
            wanted = {str(token) for token in tokens}
            enforcement_rows = [
                row for row in enforcement_rows if str(row.get("lexical_token")) in wanted
            ]

        if accepted_tokens is None:
            accepted_set = {
                str(row.get("lexical_token"))
                for row in enforcement_rows
                if bool(row.get("would_apply_if_runtime_mapping_enabled", False))
            }
        else:
            accepted_set = {str(token) for token in accepted_tokens}

        accepted_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        for row in enforcement_rows:
            token = str(row.get("lexical_token"))
            would_apply = bool(row.get("would_apply_if_runtime_mapping_enabled", False))
            if would_apply and token in accepted_set:
                accepted = deepcopy(row)
                accepted.update(
                    {
                        "operator_acceptance_fixture": True,
                        "operator_decision_status": "operator_accepted_fixture_not_persisted",
                        "resolved_reasons": ["operator_acceptance_required"],
                        "remaining_blocked_reasons": [],
                        "would_apply_runtime_mapping_after_fixture": True,
                        "runtime_mapping_applied_now": False,
                        "runtime_mapping_enabled_now": False,
                        "enforcement_enabled_now": False,
                        "category_created_now": False,
                        "concept_memory_mutation_now": False,
                        "frame_hypergraph_mutation_now": False,
                        "sa_activation_created_now": False,
                        "agp_verify_called_now": False,
                        "embedding_lookup_called_now": False,
                    }
                )
                accepted_rows.append(accepted)
            else:
                blocked = deepcopy(row)
                reasons = list(blocked.get("blocked_reasons", []) or [])
                if would_apply and token not in accepted_set:
                    reasons = ["operator_acceptance_required"]
                elif not reasons:
                    reasons = ["not_ready_for_runtime_mapping"]
                blocked.update(
                    {
                        "operator_acceptance_fixture": False,
                        "operator_decision_status": "blocked_or_not_accepted_fixture_not_persisted",
                        "blocked_reasons": sorted(set(str(reason) for reason in reasons)),
                        "would_apply_runtime_mapping_after_fixture": False,
                        "runtime_mapping_applied_now": False,
                        "runtime_mapping_enabled_now": False,
                        "enforcement_enabled_now": False,
                        "category_created_now": False,
                        "concept_memory_mutation_now": False,
                        "frame_hypergraph_mutation_now": False,
                        "sa_activation_created_now": False,
                        "agp_verify_called_now": False,
                        "embedding_lookup_called_now": False,
                    }
                )
                blocked_rows.append(blocked)

        after_categories = self.concept_categories_snapshot()
        after_audit = self.concept_commit_records()
        after_store = self._vector_store_stats()
        after_telemetry = self._wrapper_telemetry_snapshot()
        after_active = self._active_categories_snapshot()

        accepted_tokens_sorted = sorted(str(row.get("lexical_token")) for row in accepted_rows)
        blocked_tokens_sorted = sorted(str(row.get("lexical_token")) for row in blocked_rows)

        return {
            "fixture_version": "v3_round95_runtime_mapping_operator_acceptance_fixture",
            "round": 95,
            "source_enforcement_version": enforcement.get("dry_run_version", "v3_round94_runtime_mapping_enforcement_dry_run"),
            "source_proposal_version": enforcement.get("source_proposal_version", "v3_round93_runtime_mapping_proposal_report"),
            "axis": "runtime_lexical_to_concept_mapping_operator_acceptance_fixture",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "candidate_count": len(enforcement_rows),
            "accepted_count": len(accepted_rows),
            "blocked_count": len(blocked_rows),
            "accepted_tokens": accepted_tokens_sorted,
            "blocked_tokens": blocked_tokens_sorted,
            "accepted_rows": accepted_rows,
            "blocked_rows": blocked_rows,
            "operator_recommendation": {
                "operator_fixture_created": bool(accepted_rows),
                "runtime_mapping_should_be_enabled_now": False,
                "safe_default": "keep_runtime_mapping_disabled_until_separate_enablement_round",
                "next_recommended_round": "round96_runtime_mapping_enable_smoke_with_checkpoint" if accepted_rows else "collect_more_operator_acceptance_evidence",
                "separate_mutation_round_required_for_enablement": True,
                "split_full_suite_required_before_enablement": True,
            },
            "next_round_recommendation": {
                "round": 96,
                "name": "runtime_mapping_enable_smoke_with_checkpoint",
                "requires_operator_approval": True,
                "requires_pre_mutation_checkpoint": True,
                "requires_rollback_report": True,
                "requires_split_full_suite": True,
            },
            "read_only_checks": {
                "category_snapshot_unchanged_during_operator_fixture": before_categories == after_categories,
                "concept_commit_audit_unchanged_during_operator_fixture": before_audit == after_audit,
                "eve_specific_vector_store_unchanged_during_operator_fixture": before_store == after_store,
                "wrapper_telemetry_unchanged_during_operator_fixture": before_telemetry == after_telemetry,
                "sa_active_categories_unchanged_during_operator_fixture": before_active == after_active,
                "category_created_during_operator_fixture": False,
                "concept_memory_mutation_during_operator_fixture": False,
                "frame_hypergraph_mutation_during_operator_fixture": False,
                "sa_activation_created_during_operator_fixture": False,
                "agp_verify_called_during_operator_fixture": False,
                "embedding_lookup_called_during_operator_fixture": False,
                "runtime_mapping_applied": False,
                "runtime_mapping_enabled_now": False,
                "enforcement_enabled_now": False,
                "eve_specific_vector_commit_called": False,
            },
            "policy": {
                "operator_acceptance_fixture_only": True,
                "runtime_mapping_still_disabled": True,
                "enforcement_still_disabled": True,
                "no_auto_category_creation": True,
                "no_concept_memory_mutation": True,
                "no_frame_hypergraph_mutation": True,
                "no_sa_activation_creation": True,
                "no_agp_verify_call": True,
                "no_embedding_lookup": True,
                "lexical_vector_is_evidence_only": True,
                "eve_specific_vector_is_not_concept": True,
                "eve_specific_vector_is_not_agp_anchor": True,
                "seed_vector_is_not_agp_anchor": True,
                "agp_anchor_requires_explicit_category_and_sa_activation": True,
            },
            "read_only": True,
        }


    def stats(self) -> dict[str, Any]:
        """Return compact state-debug stats."""
        return {
            "module": self.__class__.__name__,
            "planning_version": self.planning_version,
            "candidate_dry_run_version": "v3_round78_lexical_concept_candidate_schema_dry_run",
            "evidence_quality_report_version": "v3_round79_lexical_concept_candidate_evidence_quality_report",
            "concept_proposal_report_version": "v3_round80_concept_proposal_report",
            "concept_mapping_gate_dry_run_version": "v3_round81_concept_mapping_gate_dry_run",
            "concept_mapping_gate_proposal_report_version": "v3_round82_concept_mapping_gate_proposal_report",
            "operator_acceptance_category_creation_dry_run_version": "v3_round83_operator_acceptance_category_creation_dry_run",
            "concept_memory_frame_evidence_dry_run_version": "v3_round84_concept_memory_frame_evidence_dry_run",
            "sa_activation_path_dry_run_version": "v3_round85_sa_activation_path_dry_run",
            "agp_bridge_smoke_dry_run_version": "v3_round86_agp_bridge_smoke_dry_run",
            "concept_mapping_readiness_dashboard_version": "v3_round87_concept_mapping_readiness_dashboard",
            "concept_mapping_v0_proposal_freeze_version": "v3_round88_concept_mapping_v0_proposal_freeze",
            "explicit_concept_commit_smoke_version": "v3_round89_explicit_concept_commit_smoke",
            "concept_commit_delta_replay_report_version": "v3_round90_concept_commit_delta_replay_report",
            "concept_commit_replay_export_checkpoint_version": "v3_round91_concept_commit_replay_export_v0_checkpoint",
            "runtime_mapping_gate_dry_run_version": "v3_round92_runtime_mapping_gate_dry_run",
            "runtime_mapping_proposal_report_version": "v3_round93_runtime_mapping_proposal_report",
            "runtime_mapping_enforcement_dry_run_version": "v3_round94_runtime_mapping_enforcement_dry_run",
            "runtime_mapping_operator_acceptance_fixture_version": "v3_round95_runtime_mapping_operator_acceptance_fixture",
            "runtime_mapping_enable_smoke_precheck_version": "v3_round96_runtime_mapping_enable_smoke_precheck",
            "controlled_runtime_mapping_enable_smoke_version": "v3_round97_controlled_runtime_mapping_enable_smoke",
            "runtime_mapping_persistence_gate_audit_version": "v3_round98_runtime_mapping_persistence_gate_audit",
            "runtime_mapping_persistence_activation_dryrun_version": "v3_round107_runtime_mapping_persistence_activation_dryrun",
            "runtime_mapping_persistence_activation_candidate_version": "v3_round108_runtime_mapping_persistence_activation_candidate",
            "runtime_mapping_persistence_approval_fixture_version": "v3_round109_runtime_mapping_persistence_approval_fixture",
            "runtime_mapping_limited_persistence_sandbox_version": "v3_round110_runtime_mapping_limited_persistence_sandbox",
            "runtime_mapping_sandbox_rollback_cleanup_version": "v3_round111_sandbox_rollback_cleanup_verification",
            "runtime_mapping_sandbox_audit_replay_version": "v3_round112_post_sandbox_focused_validation_audit_replay",
            "round": int(getattr(self, "_latest_runtime_mapping_round", 112) or 112),
            "runtime_mapping_enabled": bool(getattr(self, "runtime_mapping_enabled", False)),
            "enforcement_enabled": bool(getattr(self, "enforcement_enabled", False)),
            "snapshots_built": 0,
            "candidate_schema_dry_run_available": True,
            "candidate_evidence_quality_report_available": True,
            "concept_proposal_report_available": True,
            "concept_mapping_gate_dry_run_available": True,
            "concept_mapping_gate_proposal_report_available": True,
            "operator_acceptance_category_creation_dry_run_available": True,
            "concept_memory_frame_evidence_dry_run_available": True,
            "sa_activation_path_dry_run_available": True,
            "agp_bridge_smoke_dry_run_available": True,
            "concept_mapping_readiness_dashboard_available": True,
            "concept_mapping_v0_proposal_freeze_available": True,
            "explicit_concept_commit_smoke_available": True,
            "concept_commit_delta_replay_report_available": True,
            "concept_commit_replay_export_checkpoint_available": True,
            "runtime_mapping_gate_dry_run_available": True,
            "runtime_mapping_proposal_report_available": True,
            "runtime_mapping_enforcement_dry_run_available": True,
            "runtime_mapping_operator_acceptance_fixture_available": True,
            "runtime_mapping_enable_smoke_precheck_available": True,
            "controlled_runtime_mapping_enable_smoke_available": True,
            "runtime_mapping_persistence_gate_audit_available": True,
            "runtime_mapping_persistence_activation_dryrun_available": True,
            "runtime_mapping_persistence_activation_candidate_available": True,
            "runtime_mapping_persistence_approval_fixture_available": True,
            "runtime_mapping_limited_persistence_sandbox_available": True,
            "runtime_mapping_sandbox_rollback_cleanup_available": True,
            "runtime_mapping_sandbox_audit_replay_available": True,
            "runtime_mapping_persistence_activation": {
                "dry_run_version": "v3_round107_runtime_mapping_persistence_activation_dryrun",
                "round": 107,
                "activation_dry_run_available": True,
                "runtime_mapping_enabled_default": False,
                "enforcement_enabled_default": False,
                "actual_persistence_enablement": False,
                "state_debug_export_surface": "lex_concept_mapping.runtime_mapping_persistence_activation",
                "read_only": True,
            },
            "runtime_mapping_persistence_activation_candidate": {
                "candidate_version": "v3_round108_runtime_mapping_persistence_activation_candidate",
                "round": 108,
                "candidate_available": True,
                "operator_approval_required": True,
                "checkpoint_before_mutation_required": True,
                "rollback_verification_required": True,
                "audit_log_required": True,
                "before_after_state_debug_export_required": True,
                "runtime_mapping_enabled_default": False,
                "enforcement_enabled_default": False,
                "actual_persistence_enablement": False,
                "read_only": True,
            },
            "runtime_mapping_persistence_approval_fixture": {
                "fixture_version": "v3_round109_runtime_mapping_persistence_approval_fixture",
                "round": 109,
                "approval_scope": "runtime_mapping_persistence_only",
                "allowed_tokens": ["민석"],
                "test_dry_run_mode_only": True,
                "checkpoint_before_mutation_required": True,
                "audit_log_ordering_required": True,
                "rollback_verification_required": True,
                "state_debug_before_after_rollback_export_required": True,
                "runtime_mapping_enabled_default": False,
                "enforcement_enabled_default": False,
                "actual_persistence_enablement": False,
                "vectors_npy_committed": False,
                "read_only": True,
            },
            "runtime_mapping_limited_persistence_sandbox": {
                "sandbox_version": "v3_round110_runtime_mapping_limited_persistence_sandbox",
                "round": 110,
                "sandbox_available": True,
                "scope": "limited_sandbox_json_only",
                "checkpoint_before_sandbox_mutation_required": True,
                "audit_log_required": True,
                "rollback_required": True,
                "cleanup_required": True,
                "runtime_mapping_enabled_default": False,
                "enforcement_enabled_default": False,
                "production_persistence_enabled": False,
                "vectors_npy_committed": False,
                "read_only": True,
            },
            "runtime_mapping_sandbox_rollback_cleanup": {
                "cleanup_version": "v3_round111_sandbox_rollback_cleanup_verification",
                "round": 111,
                "cleanup_available": True,
                "removes_sandbox_json_state": True,
                "verifies_disabled_flags": True,
                "production_persistence_enabled": False,
                "vectors_npy_committed": False,
                "read_only": True,
            },
            "runtime_mapping_sandbox_audit_replay": {
                "replay_version": "v3_round112_post_sandbox_focused_validation_audit_replay",
                "round": 112,
                "replay_available": True,
                "source_rounds": [110, 111],
                "read_only_replay": True,
                "production_persistence_enabled": False,
                "vectors_npy_committed": False,
            },
            "concept_category_count": len(self._concept_categories),
            "concept_commit_audit_count": len(self._concept_commit_audit),
            "policy": "lexical_vectors_are_evidence_not_agp_anchors_round88_concept_mapping_v0_freeze_read_only_round89_explicit_commit_round90_replay_round91_checkpoint_round92_runtime_mapping_gate_dry_run_round93_runtime_mapping_proposal_disabled_round94_runtime_mapping_enforcement_dry_run_disabled_round95_runtime_mapping_operator_acceptance_fixture_disabled_round96_runtime_mapping_enable_smoke_precheck_disabled_round97_controlled_enable_smoke_ephemeral_round98_persistence_gate_audit_round107_activation_dryrun_disabled_round108_activation_candidate_guarded_disabled_by_default_round109_approval_fixture_rollback_drill_disabled_by_default_round110_limited_persistence_sandbox_json_only_round111_cleanup_verification_round112_audit_replay_read_only",
            "read_only": False,
        }

    def write_mapping_contract_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Write the current contract snapshot as an explicit operator export."""
        snapshot = self.mapping_contract_snapshot()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return {
            "export_version": snapshot["planning_version"],
            "path": str(out),
            "recomputed": False,
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "commit_called": False,
            "lookup_called": False,
            "concept_memory_mutation": False,
            "frame_hypergraph_mutation": False,
            "agp_verify_called": False,
            "policy_changed": False,
        }
