"""
StateDebugAdapter — round73 patch7

One-shot diagnostic view for why an input is taking a given route.
Read-only by default: it parses/classifies and snapshots state without running chat_stream.
"""

from __future__ import annotations

from typing import Any


class StateDebugAdapter:
    def __init__(self, engine=None):
        self.engine = engine
        self.dumps_made = 0
        self.last_dump: dict[str, Any] | None = None

    def diagnose_text(self, text: str) -> dict[str, Any]:
        """Return a deterministic route/state dump for a candidate user input."""
        engine = self.engine
        raw = text or ""
        if engine is None:
            result = {"input": raw, "error": "engine_missing"}
            self._remember(result)
            return result

        meaning = engine.lu.parse(raw)
        result: dict[str, Any] = {
            "input": raw,
            "meaning": {
                "intent": getattr(meaning, "intent", None),
                "entities": list(getattr(meaning, "entities", []) or []),
                "actions": list(getattr(meaning, "actions", []) or []),
                "emotions": dict(getattr(meaning, "emotions", {}) or {}),
            },
            "route": self._route(meaning),
            "state": self.snapshot_state(),
            "risks": [],
        }
        result["risks"] = self._infer_risks(result)
        self._remember(result)
        return result

    def snapshot_state(self) -> dict[str, Any]:
        """Return compact JSON-friendly state relevant to fallback/routing bugs."""
        engine = self.engine
        state: dict[str, Any] = {
            "activation_top": [],
            "salience_top": [],
            "hormone_primary": None,
            "teaching": {},
            "dialogue_context": {},
            "constraints": [],
            "self_embedding": {},
            "fasttext_embedding": {},
            "attention": {},
            "compositor": {},
            "concept_memory": {},
            "situation_responder": {},
            "live_loop": {},
            "safety": {},
            "eve_self_learning": {},
            "lex_concept_mapping": {},
        }
        if engine is None:
            return state

        # Activation top
        try:
            acts = engine.activation_adapter.sa.activations
            top = sorted(acts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            state["activation_top"] = [(str(k), round(float(v), 4)) for k, v in top if v > 0.01]
        except Exception:
            pass

        # Salience top
        try:
            state["salience_top"] = [
                (str(k), round(float(v), 4))
                for k, v in engine.salience_adapter.top(10)
                if v > 0.01
            ]
        except Exception:
            pass

        # Hormone primary
        try:
            hs = engine.hormone_adapter.hs
            primary = hs.primary_hormone() if hasattr(hs, "primary_hormone") else None
            if isinstance(primary, tuple):
                state["hormone_primary"] = [primary[0], round(float(primary[1]), 4)]
            else:
                state["hormone_primary"] = primary
        except Exception:
            pass

        # Teaching learned responses
        try:
            t = engine.teaching_adapter
            state["teaching"] = {
                "learned_count": len(t.learned),
                "last_emitted": getattr(getattr(t, "_last_emitted", None), "response", None),
                "top": [
                    {
                        "response": lr.response,
                        "triggers": sorted(list(lr.triggers)),
                        "strength": round(float(lr.strength), 4),
                        "used": int(lr.times_used),
                    }
                    for lr in sorted(t.learned, key=lambda lr: (-lr.strength, lr.response))[:5]
                ],
            }
        except Exception:
            pass

        # Dialogue pending question
        try:
            dc = engine.dialogue_context
            state["dialogue_context"] = dc.stats()
        except Exception:
            pass

        # User instruction constraints
        try:
            ui = engine.user_instruction
            state["constraints"] = [
                {"type": c.type, "target": c.target, "remaining": c.remaining}
                for c in ui.constraints
                if getattr(c, "remaining", 0) > 0
            ]
        except Exception:
            pass

        # SelfEmbedding / wrapper status
        try:
            se = engine.self_embedding
            se_stats = se.stats() if hasattr(se, "stats") else {}
            is_wrapper = se.__class__.__name__ == "EmbeddingWrapper"
            fallback = getattr(se, "fallback", None) if is_wrapper else se
            state["self_embedding"] = {
                "module": "embedding_wrapper" if is_wrapper else "self_embedding_adapter",
                "method": se_stats.get("method", "PMI+SVD"),
                "active_class": se.__class__.__name__,
                "dimension": int(se_stats.get("dimension", getattr(se, "dim", 50)) or 50),
                "fallback_dimension": int(se_stats.get("fallback_dimension", getattr(fallback, "dim", 50)) or 50),
                "ready": bool(getattr(fallback, "_ready", False)),
                "observations": len(getattr(fallback, "observations", []) or []),
                "vocab": len(getattr(fallback, "vocab", {}) or getattr(fallback, "word_counts", {}) or {}),
                "in_use_by_generation": "wrapper" if is_wrapper else True,
                "primary_class": se_stats.get("primary_class"),
                "fallback_class": se_stats.get("fallback_class"),
                "fallback_count": int(se_stats.get("fallback_count", 0) or 0),
                "rollback_available": bool(se_stats.get("rollback_available", False)) if is_wrapper else False,
                "eve_specific_dimension": int(se_stats.get("eve_specific_dimension", 0) or 0),
                "eve_specific_count": int(se_stats.get("eve_specific_count", 0) or 0),
                "eve_specific_class": se_stats.get("eve_specific_class"),
            }
            state["main_engine"] = {
                "self_embedding_active": se.__class__.__name__,
                "self_embedding_backup_available": hasattr(engine, "self_embedding_backup"),
                "primary_method": "fasttext_extracted_subset" if is_wrapper else None,
                "fallback_method": "PMI+SVD" if is_wrapper else None,
                "fallback_count": int(se_stats.get("fallback_count", 0) or 0),
                "in_use_by_generation": "wrapper" if is_wrapper else "self_embedding",
                "primary_subset_name": se_stats.get("primary_stats", {}).get("subset_name") if isinstance(se_stats.get("primary_stats"), dict) else None,
                "primary_vocab_size": se_stats.get("primary_stats", {}).get("vocab_size") if isinstance(se_stats.get("primary_stats"), dict) else None,
                "rollback_strategy": "engine.self_embedding = engine.self_embedding_backup; engine.fasttext_embedding unload" if is_wrapper else None,
            }
            telemetry = se.telemetry() if is_wrapper and hasattr(se, "telemetry") else {}
            if is_wrapper:
                state["wrapper_telemetry"] = {
                    "primary_hit_rate": float(telemetry.get("primary_hit_rate", 0.0) or 0.0),
                    "fallback_rate": float(telemetry.get("fallback_rate", 0.0) or 0.0),
                    "eve_specific_rate": float(telemetry.get("eve_specific_rate", 0.0) or 0.0),
                    "error_rate": float(telemetry.get("error_rate", 0.0) or 0.0),
                    "total_calls": int(telemetry.get("total_calls", 0) or 0),
                    "primary_hits": int(telemetry.get("primary_hits", 0) or 0),
                    "fallback_uses": int(telemetry.get("fallback_uses", 0) or 0),
                    "eve_specific_hits": int(telemetry.get("eve_specific_hits", 0) or 0),
                    "errors": int(telemetry.get("errors", 0) or 0),
                    "oov_log_size": int(telemetry.get("oov_log_size", 0) or 0),
                    "oov_log_cap": int(telemetry.get("oov_log_cap", 1000) or 1000),
                    "oov_recent_sample": list(telemetry.get("recent_oov_sample", []) or []),
                    "read_only": True,
                }
                state["seed_drift"] = {
                    "baseline_round": 42,
                    "status": "tracking_started",
                    "fasttext_unchanged": True,
                    "evespecific_substrate": "PMI+SVD",
                    "target": "round 200+ avg drift > 0.3 (Draft, v3.1 precise threshold pending)",
                    "read_only": True,
                }
        except Exception:
            pass

        # Round58: EVE-specific self-learning observation status.
        try:
            learner = getattr(engine, "eve_self_learning", None)
            if learner is not None:
                stats = learner.stats() if hasattr(learner, "stats") else {}
                state["eve_self_learning"] = {
                    "module": learner.__class__.__name__,
                    "implementation_phase": stats.get("implementation_phase"),
                    "auto_observe_enabled": bool(stats.get("auto_observe_enabled", False)),
                    "auto_promotion_enabled": bool(stats.get("auto_promotion_enabled", False)),
                    "observe_calls": int(stats.get("observe_calls", 0) or 0),
                    "commit_calls": int(stats.get("commit_calls", 0) or 0),
                    "audit_calls": int(stats.get("audit_calls", 0) or 0),
                    "commit_gate_enabled": bool(stats.get("commit_gate_enabled", False)),
                    "min_observations_for_commit": int(stats.get("min_observations_for_commit", 0) or 0),
                    "min_known_context_words_for_commit": int(stats.get("min_known_context_words_for_commit", 0) or 0),
                    "commit_gate_blocks": int(stats.get("commit_gate_blocks", 0) or 0),
                    "commit_audit_record_count": int(stats.get("commit_audit_record_count", 0) or 0),
                    "commit_audit_export_version": stats.get("commit_audit_export_version"),
                    "commit_audit_dashboard_version": stats.get("commit_audit_dashboard_version"),
                    "commit_threshold_dry_run_version": stats.get("commit_threshold_dry_run_version"),
                    "commit_threshold_readiness_version": stats.get("commit_threshold_readiness_version"),
                    "commit_threshold_readiness": stats.get("commit_threshold_readiness"),
                    "commit_threshold_proposal_version": stats.get("commit_threshold_proposal_version"),
                    "commit_threshold_proposal": stats.get("commit_threshold_proposal"),
                    "commit_threshold_enforcement_version": stats.get("commit_threshold_enforcement_version"),
                    "commit_threshold_policy_snapshot_version": stats.get("commit_threshold_policy_snapshot_version"),
                    "commit_threshold_policy_snapshot": stats.get("commit_threshold_policy_snapshot"),
                    "observation_evidence_quality_version": stats.get("observation_evidence_quality_version"),
                    "observation_evidence_quality": stats.get("observation_evidence_quality"),
                    "context_diversity_gate_dry_run_version": stats.get("context_diversity_gate_dry_run_version"),
                    "context_diversity_gate_dry_run": stats.get("context_diversity_gate_dry_run"),
                    "context_diversity_proposal_version": stats.get("context_diversity_proposal_version"),
                    "context_diversity_proposal": stats.get("context_diversity_proposal"),
                    "context_diversity_gate_enforcement_version": stats.get("context_diversity_gate_enforcement_version"),
                    "context_diversity_gate_enabled": bool(stats.get("context_diversity_gate_enabled", False)),
                    "context_diversity_rollback_drill_version": stats.get("context_diversity_rollback_drill_version"),
                    "context_diversity_rollback_drill": stats.get("context_diversity_rollback_drill"),
                    "context_diversity_blocked_candidate_report_version": stats.get("context_diversity_blocked_candidate_report_version"),
                    "context_diversity_blocked_candidate_report": stats.get("context_diversity_blocked_candidate_report"),
                    "self_learning_v1_freeze_policy_snapshot_version": stats.get("self_learning_v1_freeze_policy_snapshot_version"),
                    "self_learning_v1_freeze_policy_snapshot": stats.get("self_learning_v1_freeze_policy_snapshot"),
                    "active_policy_version": stats.get("active_policy_version"),
                    "commit_audit_dashboard": stats.get("commit_audit_dashboard"),
                    "vectors_created": int(stats.get("vectors_created", 0) or 0),
                    "vectors_skipped": int(stats.get("vectors_skipped", 0) or 0),
                    "tracker_eve_specific_count": int(stats.get("tracker_eve_specific_count", 0) or 0),
                    "vector_store_count": int(stats.get("vector_store_count", 0) or 0),
                    "policy": stats.get("policy"),
                    "read_only": True,
                }
        except Exception:
            pass

        # Round77: lexical → concept mapping planning boundary.
        try:
            lcm = getattr(engine, "lex_concept_mapping", None)
            if lcm is not None:
                stats = lcm.stats() if hasattr(lcm, "stats") else {}
                state["lex_concept_mapping"] = {
                    "module": stats.get("module", lcm.__class__.__name__),
                    "planning_version": stats.get("planning_version"),
                    "candidate_dry_run_version": stats.get("candidate_dry_run_version"),
                    "evidence_quality_report_version": stats.get("evidence_quality_report_version"),
                    "concept_proposal_report_version": stats.get("concept_proposal_report_version"),
                    "concept_mapping_gate_dry_run_version": stats.get("concept_mapping_gate_dry_run_version"),
                    "concept_mapping_gate_proposal_report_version": stats.get("concept_mapping_gate_proposal_report_version"),
                    "operator_acceptance_category_creation_dry_run_version": stats.get("operator_acceptance_category_creation_dry_run_version"),
                    "concept_memory_frame_evidence_dry_run_version": stats.get("concept_memory_frame_evidence_dry_run_version"),
                    "sa_activation_path_dry_run_version": stats.get("sa_activation_path_dry_run_version"),
                    "agp_bridge_smoke_dry_run_version": stats.get("agp_bridge_smoke_dry_run_version"),
                    "concept_mapping_readiness_dashboard_version": stats.get("concept_mapping_readiness_dashboard_version"),
                    "concept_mapping_v0_proposal_freeze_version": stats.get("concept_mapping_v0_proposal_freeze_version"),
                    "explicit_concept_commit_smoke_version": stats.get("explicit_concept_commit_smoke_version"),
                    "concept_commit_delta_replay_report_version": stats.get("concept_commit_delta_replay_report_version"),
                    "concept_commit_replay_export_checkpoint_version": stats.get("concept_commit_replay_export_checkpoint_version"),
                    "runtime_mapping_gate_dry_run_version": stats.get("runtime_mapping_gate_dry_run_version"),
                    "runtime_mapping_proposal_report_version": stats.get("runtime_mapping_proposal_report_version"),
                    "runtime_mapping_enforcement_dry_run_version": stats.get("runtime_mapping_enforcement_dry_run_version"),
                    "runtime_mapping_operator_acceptance_fixture_version": stats.get("runtime_mapping_operator_acceptance_fixture_version"),
                    "runtime_mapping_enable_smoke_precheck_version": stats.get("runtime_mapping_enable_smoke_precheck_version"),
                    "controlled_runtime_mapping_enable_smoke_version": stats.get("controlled_runtime_mapping_enable_smoke_version"),
                    "runtime_mapping_persistence_gate_audit_version": stats.get("runtime_mapping_persistence_gate_audit_version"),
                    "runtime_mapping_persistence_activation_dryrun_version": stats.get("runtime_mapping_persistence_activation_dryrun_version"),
                    "runtime_mapping_persistence_activation_candidate_version": stats.get("runtime_mapping_persistence_activation_candidate_version"),
                    "runtime_mapping_persistence_approval_fixture_version": stats.get("runtime_mapping_persistence_approval_fixture_version"),
                    "runtime_mapping_limited_persistence_sandbox_version": stats.get("runtime_mapping_limited_persistence_sandbox_version"),
                    "runtime_mapping_sandbox_rollback_cleanup_version": stats.get("runtime_mapping_sandbox_rollback_cleanup_version"),
                    "runtime_mapping_sandbox_audit_replay_version": stats.get("runtime_mapping_sandbox_audit_replay_version"),
                    "round": int(stats.get("round", 112) or 112),
                    "runtime_mapping_enabled": bool(stats.get("runtime_mapping_enabled", False)),
                    "enforcement_enabled": bool(stats.get("enforcement_enabled", False)),
                    "candidate_schema_dry_run_available": bool(stats.get("candidate_schema_dry_run_available", False)),
                    "candidate_evidence_quality_report_available": bool(stats.get("candidate_evidence_quality_report_available", False)),
                    "concept_proposal_report_available": bool(stats.get("concept_proposal_report_available", False)),
                    "concept_mapping_gate_dry_run_available": bool(stats.get("concept_mapping_gate_dry_run_available", False)),
                    "concept_mapping_gate_proposal_report_available": bool(stats.get("concept_mapping_gate_proposal_report_available", False)),
                    "operator_acceptance_category_creation_dry_run_available": bool(stats.get("operator_acceptance_category_creation_dry_run_available", False)),
                    "concept_memory_frame_evidence_dry_run_available": bool(stats.get("concept_memory_frame_evidence_dry_run_available", False)),
                    "sa_activation_path_dry_run_available": bool(stats.get("sa_activation_path_dry_run_available", False)),
                    "agp_bridge_smoke_dry_run_available": bool(stats.get("agp_bridge_smoke_dry_run_available", False)),
                    "concept_mapping_readiness_dashboard_available": bool(stats.get("concept_mapping_readiness_dashboard_available", False)),
                    "concept_mapping_v0_proposal_freeze_available": bool(stats.get("concept_mapping_v0_proposal_freeze_available", False)),
                    "explicit_concept_commit_smoke_available": bool(stats.get("explicit_concept_commit_smoke_available", False)),
                    "concept_commit_delta_replay_report_available": bool(stats.get("concept_commit_delta_replay_report_available", False)),
                    "concept_commit_replay_export_checkpoint_available": bool(stats.get("concept_commit_replay_export_checkpoint_available", False)),
                    "runtime_mapping_gate_dry_run_available": bool(stats.get("runtime_mapping_gate_dry_run_available", False)),
                    "runtime_mapping_proposal_report_available": bool(stats.get("runtime_mapping_proposal_report_available", False)),
                    "runtime_mapping_enforcement_dry_run_available": bool(stats.get("runtime_mapping_enforcement_dry_run_available", False)),
                    "runtime_mapping_operator_acceptance_fixture_available": bool(stats.get("runtime_mapping_operator_acceptance_fixture_available", False)),
                    "runtime_mapping_enable_smoke_precheck_available": bool(stats.get("runtime_mapping_enable_smoke_precheck_available", False)),
                    "controlled_runtime_mapping_enable_smoke_available": bool(stats.get("controlled_runtime_mapping_enable_smoke_available", False)),
                    "runtime_mapping_persistence_gate_audit_available": bool(stats.get("runtime_mapping_persistence_gate_audit_available", False)),
                    "runtime_mapping_persistence_activation_dryrun_available": bool(stats.get("runtime_mapping_persistence_activation_dryrun_available", False)),
                    "runtime_mapping_persistence_activation_candidate_available": bool(stats.get("runtime_mapping_persistence_activation_candidate_available", False)),
                    "runtime_mapping_persistence_approval_fixture_available": bool(stats.get("runtime_mapping_persistence_approval_fixture_available", False)),
                    "runtime_mapping_limited_persistence_sandbox_available": bool(stats.get("runtime_mapping_limited_persistence_sandbox_available", False)),
                    "runtime_mapping_sandbox_rollback_cleanup_available": bool(stats.get("runtime_mapping_sandbox_rollback_cleanup_available", False)),
                    "runtime_mapping_sandbox_audit_replay_available": bool(stats.get("runtime_mapping_sandbox_audit_replay_available", False)),
                    "runtime_mapping_persistence_activation": dict(stats.get("runtime_mapping_persistence_activation", {}) or {}),
                    "runtime_mapping_persistence_activation_candidate": dict(stats.get("runtime_mapping_persistence_activation_candidate", {}) or {}),
                    "runtime_mapping_persistence_approval_fixture": dict(stats.get("runtime_mapping_persistence_approval_fixture", {}) or {}),
                    "runtime_mapping_limited_persistence_sandbox": dict(stats.get("runtime_mapping_limited_persistence_sandbox", {}) or {}),
                    "runtime_mapping_sandbox_rollback_cleanup": dict(stats.get("runtime_mapping_sandbox_rollback_cleanup", {}) or {}),
                    "runtime_mapping_sandbox_audit_replay": dict(stats.get("runtime_mapping_sandbox_audit_replay", {}) or {}),
                    "concept_category_count": int(stats.get("concept_category_count", 0) or 0),
                    "concept_commit_audit_count": int(stats.get("concept_commit_audit_count", 0) or 0),
                    "policy": stats.get("policy"),
                    "read_only": bool(stats.get("read_only", False)),
                }
        except Exception:
            pass

        # FasttextEmbedding migration/debug status (v3 round34).
        # Observation only: this must not call load() or change generation routing.
        try:
            fe = getattr(engine, "fasttext_embedding", None)
            if fe is not None:
                stats = fe.stats() if hasattr(fe, "stats") else {}
                state["fasttext_embedding"] = {
                    "module": "fasttext_embedding_adapter",
                    "method": stats.get("method", "fasttext_extracted_subset"),
                    "dimension": int(stats.get("dimension", getattr(fe, "dimension", 300))),
                    "subset_name": stats.get("subset_name", getattr(fe, "subset_name", None)),
                    "loaded": bool(stats.get("loaded", False)),
                    "vocab_size": int(stats.get("vocab_size", 0) or 0),
                    "vectors_shape": stats.get("vectors_shape"),
                    "in_use_by_generation": "wrapper_primary" if getattr(engine, "self_embedding", None).__class__.__name__ == "EmbeddingWrapper" else False,
                    "migration_stage": "final_swap_primary" if getattr(engine, "self_embedding", None).__class__.__name__ == "EmbeddingWrapper" else "state_debug_exposed_only",
                }
        except Exception:
            pass

        # Attention migration/debug status (v3 round35).
        # Read-only: do not trigger fastText load or alter attention decisions.
        try:
            attention = getattr(engine, "attention", None)
            if attention is not None:
                stats = attention.stats() if hasattr(attention, "stats") else {}
                state["attention"] = {
                    "module": "attention_analyzer",
                    "in_use_by_generation": stats.get("in_use_by_generation", "self_embedding"),
                    "parallel_observation": stats.get("fasttext_parallel_observation"),
                    "fasttext_trace_size": int(stats.get("fasttext_trace_size", 0) or 0),
                    "fasttext_observation_count": int(stats.get("fasttext_observation_count", 0) or 0),
                    "fasttext_trace_cap": int(stats.get("fasttext_trace_cap", 1000) or 1000),
                    "migration_stage": "attention_parallel_observation_only",
                }
        except Exception:
            pass

        # Compositor migration/debug status (v3 round36).
        # Read-only: do not trigger fastText load or alter AGP/composition output.
        try:
            compositor = getattr(engine, "compositor", None)
            if compositor is not None:
                stats = compositor.stats() if hasattr(compositor, "stats") else {}
                state["compositor"] = {
                    "module": "compositor_adapter",
                    "in_use_by_generation": stats.get("in_use_by_generation", "self_embedding"),
                    "parallel_observation": stats.get("fasttext_parallel_observation"),
                    "agp_mode": stats.get("agp_mode"),
                    "agp_trace_size": int(stats.get("agp_trace_size", 0) or 0),
                    "fasttext_trace_size": int(stats.get("fasttext_trace_size", 0) or 0),
                    "fasttext_observation_count": int(stats.get("fasttext_observation_count", 0) or 0),
                    "fasttext_trace_cap": int(stats.get("fasttext_trace_cap", 1000) or 1000),
                    "migration_stage": "compositor_parallel_observation_only",
                }
        except Exception:
            pass

        # ConceptMemory migration/debug status (v3 round37).
        # Read-only: do not trigger fastText load or alter query/write decisions.
        try:
            concept_memory = getattr(engine, "concept_memory", None)
            if concept_memory is not None:
                stats = concept_memory.stats() if hasattr(concept_memory, "stats") else {}
                state["concept_memory"] = {
                    "module": "concept_memory_adapter",
                    "in_use_by_generation": stats.get("in_use_by_generation", "self_embedding"),
                    "parallel_observation": stats.get("fasttext_parallel_observation"),
                    "fasttext_trace_size": int(stats.get("fasttext_trace_size", 0) or 0),
                    "fasttext_observation_count": int(stats.get("fasttext_observation_count", 0) or 0),
                    "fasttext_trace_cap": int(stats.get("fasttext_trace_cap", 1000) or 1000),
                    "operations_observed": dict(stats.get("operations_observed", {}) or {}),
                    "concepts_count": int(stats.get("concepts_count", 0) or 0),
                    "learn_count": int(stats.get("learn_count", 0) or 0),
                    "use_count": int(stats.get("use_count", 0) or 0),
                    "migration_stage": "concept_memory_parallel_observation_only",
                }
        except Exception:
            pass

        # SituationResponder migration/debug status (v3 round38).
        # Read-only: do not trigger fastText load or alter user-visible responses.
        try:
            situation_responder = getattr(engine, "situation_responder", None)
            if situation_responder is not None:
                stats = situation_responder.stats() if hasattr(situation_responder, "stats") else {}
                state["situation_responder"] = {
                    "module": "situation_responder",
                    "in_use_by_generation": stats.get("in_use_by_generation", "self_embedding"),
                    "parallel_observation": stats.get("fasttext_parallel_observation"),
                    "fasttext_trace_size": int(stats.get("fasttext_trace_size", 0) or 0),
                    "fasttext_observation_count": int(stats.get("fasttext_observation_count", 0) or 0),
                    "fasttext_trace_cap": int(stats.get("fasttext_trace_cap", 1000) or 1000),
                    "operations_observed": dict(stats.get("operations_observed", {}) or {}),
                    "migration_stage": "situation_responder_parallel_observation_only",
                }
        except Exception:
            pass

        # Streaming migration/debug status (v3 round39).
        # Read-only: do not trigger fastText load or alter chunk output/order/timing.
        try:
            streaming = engine
            if streaming is not None and hasattr(streaming, "stats"):
                stats = streaming.stats()
                state["streaming"] = {
                    "module": "language/streaming",
                    "in_use_by_generation": stats.get("in_use_by_generation", "self_embedding"),
                    "parallel_observation": stats.get("fasttext_parallel_observation"),
                    "fasttext_trace_size": int(stats.get("fasttext_trace_size", 0) or 0),
                    "fasttext_observation_count": int(stats.get("fasttext_observation_count", 0) or 0),
                    "fasttext_trace_cap": int(stats.get("fasttext_trace_cap", 1000) or 1000),
                    "operations_observed": dict(stats.get("operations_observed", {}) or {}),
                    "migration_stage": "streaming_parallel_observation_only",
                }
        except Exception:
            pass

        # Live loop status
        try:
            state["live_loop"] = engine.live_loop.status()
        except Exception:
            pass

        # Safety compact stats
        try:
            s = engine.safety_adapter.stats()
            state["safety"] = {
                "conflicts_detected": s.get("conflicts_detected", 0),
                "exploration_triggers": s.get("exploration_triggers", 0),
                "beliefs_count": s.get("beliefs_count", 0),
            }
        except Exception:
            pass

        return state

    def render(self, dump: dict[str, Any] | None = None) -> str:
        """Human-readable dump for Termux REPL."""
        d = dump or self.last_dump or {}
        route = d.get("route", {})
        meaning = d.get("meaning", {})
        state = d.get("state", {})
        lines = [
            f"input: {d.get('input', '')}",
            f"intent: {meaning.get('intent')} entities={meaning.get('entities', [])} emotions={meaning.get('emotions', {})}",
            f"route: {route.get('situation')} via={route.get('via')} modules={route.get('modules_selected', [])}",
        ]
        matched = route.get("matched_pattern")
        if matched:
            lines.append(f"matched: {matched}")
        lines.append(f"activation_top: {state.get('activation_top', [])}")
        lines.append(f"salience_top: {state.get('salience_top', [])}")
        lines.append(f"hormone_primary: {state.get('hormone_primary')}")
        pending = (state.get("dialogue_context") or {}).get("pending")
        if pending:
            lines.append(f"pending_question: {pending}")
        constraints = state.get("constraints", [])
        if constraints:
            lines.append(f"constraints: {constraints}")
        risks = d.get("risks") or []
        if risks:
            lines.append(f"risks: {risks}")
        return "\n".join(lines)

    def _route(self, meaning) -> dict[str, Any]:
        engine = self.engine
        orch = getattr(engine, "orchestrator_adapter", None)
        if orch is None:
            return {"situation": None, "via": "orchestrator_missing", "modules_selected": []}
        trace = orch.classify_with_trace(meaning)
        situation = trace.get("situation")
        modules = orch.select_modules(situation) if situation else []
        return {
            "situation": situation,
            "via": trace.get("via"),
            "matched_pattern": trace.get("matched_pattern"),
            "match_text": trace.get("match_text"),
            "modules_selected": modules,
        }

    def _infer_risks(self, dump: dict[str, Any]) -> list[str]:
        risks: list[str] = []
        route = dump.get("route", {})
        meaning = dump.get("meaning", {})
        state = dump.get("state", {})
        text = dump.get("input", "") or ""
        situation = route.get("situation")
        via = route.get("via")

        if via in {"fallback", "intent_question", "intent_command"}:
            risks.append("fallback_or_intent_route")
        if situation == "small_talk" and ("?" in text or meaning.get("emotions")):
            risks.append("possible_missed_question_or_emotion")
        if situation == "emotional_share" and any(k in text for k in ("날씨", "좋다", "맑", "화창")):
            risks.append("positive_weather_may_be_false_emotion")
        if situation == "emotional_share" and any(k in text for k in ("노래", "영화", "커피", "사진", "코드", "분위기")):
            if any(k in text for k in ("싫", "별로", "안 좋", "안좋", "마음에 안", "맘에 안")):
                risks.append("negative_object_appraisal_may_be_false_emotion")
            elif any(k in text for k in ("무섭", "무서", "불편", "위험", "찝찝")):
                risks.append("external_threat_or_discomfort_may_be_false_emotion")
            else:
                risks.append("positive_object_appraisal_may_be_false_emotion")
        if situation in {"emotional_share", "task"} and any(k in text for k in ("영화", "공포영화", "의자", "상황", "여기", "코드")):
            if any(k in text for k in ("무섭", "무서", "불편", "위험", "찝찝")):
                risks.append("external_threat_or_discomfort_may_be_false_emotion")
        if (state.get("dialogue_context") or {}).get("pending"):
            risks.append("pending_dialogue_answer_can_intercept_next_input")
        if state.get("constraints"):
            risks.append("active_user_constraints_can_rewrite_output")
        learned = (state.get("teaching") or {}).get("learned_count", 0)
        if learned:
            risks.append("learned_teaching_response_can_inject_output")
        return risks

    def _remember(self, result: dict[str, Any]):
        self.dumps_made += 1
        self.last_dump = result
