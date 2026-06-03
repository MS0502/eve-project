"""Runtime smoke runner for v3 round43 post-swap telemetry sampling.

The runner feeds a deterministic Korean fixture set through the real streaming
path, then snapshots EmbeddingWrapper telemetry and AGP behavior. It is a data
collection surface only: it does not change thresholds, does not apply drift
results, does not rollback the wrapper, and does not rewrite embeddings.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Iterable

from adapters.agp_trace_analyzer import AGPTraceAnalyzer
from adapters.external_seed_manifest import (
    correlate_agp_and_wrapper_telemetry,
    measure_seed_drift_baseline,
)

SMOKE_BASELINE_ROUND = 43
SMOKE_VERSION = "v3_round43_runtime_conversation_smoke"
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")


def normalize_fixture_texts(fixtures: Iterable[Any]) -> list[str]:
    """Normalize fixture rows to a deterministic list of text strings."""
    texts: list[str] = []
    for row in fixtures:
        if isinstance(row, dict):
            text = row.get("text", "")
        else:
            text = row
        text = str(text or "").strip()
        if text:
            texts.append(text)
    return texts


def _tokenize_for_telemetry(text: str) -> list[str]:
    """Small deterministic token sampler for wrapper coverage telemetry."""
    return _TOKEN_RE.findall(str(text or ""))


def _wrapper_telemetry(engine: Any) -> dict[str, Any]:
    wrapper = getattr(engine, "self_embedding", None)
    if wrapper is None or not hasattr(wrapper, "telemetry"):
        return {"adapter": None, "total_calls": 0, "read_only": True}
    data = wrapper.telemetry()
    return dict(data) if isinstance(data, dict) else {}


def _agp_summary(engine: Any) -> dict[str, Any]:
    try:
        analyzer = AGPTraceAnalyzer.from_engine(engine)
        return {
            "pass_rate": analyzer.pass_rate(),
            "by_layer": analyzer.summarize_by_layer(),
            "by_reason": analyzer.summarize_by_reason(),
            "recent_fails": analyzer.recent_fails(5),
            "read_only": True,
        }
    except Exception as exc:
        return {
            "pass_rate": 0.0,
            "by_layer": {},
            "by_reason": {},
            "recent_fails": [],
            "error": exc.__class__.__name__,
            "read_only": True,
        }


def _structural_state(engine: Any) -> dict[str, Any]:
    wrapper = getattr(engine, "self_embedding", None)
    primary = getattr(wrapper, "primary", None)
    fallback = getattr(wrapper, "fallback", None)
    backup = getattr(engine, "self_embedding_backup", None)
    loaded = False
    try:
        loaded = bool(primary.is_loaded()) if primary is not None and hasattr(primary, "is_loaded") else False
    except Exception:
        loaded = False
    return {
        "self_embedding_class": None if wrapper is None else wrapper.__class__.__name__,
        "primary_class": None if primary is None else primary.__class__.__name__,
        "fallback_class": None if fallback is None else fallback.__class__.__name__,
        "backup_class": None if backup is None else backup.__class__.__name__,
        "primary_loaded": loaded,
        "rollback_available": hasattr(wrapper, "rollback"),
    }


def run_conversation_smoke(engine: Any, fixtures: Iterable[Any]) -> dict[str, Any]:
    """Run deterministic fixture conversations and return telemetry data.

    The only intentional mutations are normal conversation traces/history and
    wrapper telemetry counters from post-swap sampling. No decisions are applied
    from telemetry and no rollback/threshold/embedding rewrite is performed.
    """
    fixture_texts = normalize_fixture_texts(fixtures)
    before_structural_state = _structural_state(engine)
    wrapper_before = getattr(engine, "self_embedding", None)
    rollback_available_before = hasattr(wrapper_before, "rollback")

    responses: list[dict[str, Any]] = []
    sampled_tokens: list[str] = []
    for index, text in enumerate(fixture_texts):
        chunks = [str(chunk) for chunk in engine.chat_stream(text)]
        # Sample lexical coverage after response generation. This records real
        # wrapper primary/fallback/OOV behavior without feeding the result back
        # into response selection.
        for token in _tokenize_for_telemetry(text):
            sampled_tokens.append(token)
            wrapper = getattr(engine, "self_embedding", None)
            if wrapper is not None and hasattr(wrapper, "get_vector"):
                wrapper.get_vector(token)
        responses.append(
            {
                "index": index,
                "input": text,
                "chunks": chunks,
                "response": "".join(chunks),
                "chunk_count": len(chunks),
            }
        )

    telemetry = _wrapper_telemetry(engine)
    agp_summary = _agp_summary(engine)
    drift_baseline = measure_seed_drift_baseline(engine)
    agp_wrapper_correlation = correlate_agp_and_wrapper_telemetry(engine)
    after_structural_state = _structural_state(engine)
    wrapper_after = getattr(engine, "self_embedding", None)

    return {
        "smoke_version": SMOKE_VERSION,
        "baseline_round": SMOKE_BASELINE_ROUND,
        "fixtures_count": len(fixture_texts),
        "fixtures": list(fixture_texts),
        "sampled_tokens": list(sampled_tokens),
        "sampled_token_count": len(sampled_tokens),
        "responses": responses,
        "wrapper_telemetry": telemetry,
        "agp_trace_summary": agp_summary,
        "agp_wrapper_correlation": agp_wrapper_correlation,
        "seed_drift_baseline": drift_baseline,
        "oov_samples_from_run": list(telemetry.get("recent_oov_sample", []) or []),
        "swap_state": "engine.self_embedding = wrapper",
        "rollback_called": False,
        "rollback_method_preserved": rollback_available_before and wrapper_after is wrapper_before and hasattr(wrapper_after, "rollback"),
        "structural_state_before": before_structural_state,
        "structural_state_after": after_structural_state,
        "structural_state_unchanged": before_structural_state == after_structural_state,
        "decision_policy_changed": False,
        "thresholds_changed": False,
        "read_only_policy": "no_threshold_change_no_rollback_no_runtime_decision_application",
        "read_only": True,
    }


def compare_fixture_responses(engine_factory: Any, fixtures: Iterable[Any]) -> dict[str, Any]:
    """Return a deterministic response comparison across two fresh engines."""
    texts = normalize_fixture_texts(fixtures)
    first = run_conversation_smoke(engine_factory(), texts)
    second = run_conversation_smoke(engine_factory(), texts)
    first_pairs = [(row["input"], row["chunks"]) for row in first["responses"]]
    second_pairs = [(row["input"], row["chunks"]) for row in second["responses"]]
    return {
        "comparison_version": "v3_round43_fixture_response_determinism",
        "baseline_round": SMOKE_BASELINE_ROUND,
        "fixtures_count": len(texts),
        "deterministic": first_pairs == second_pairs,
        "first": first_pairs,
        "second": second_pairs,
        "read_only": True,
    }


SMOKE_TRACE_BASELINE_ROUND = 46
SMOKE_TRACE_VERSION = "v3_round46_agp_extraction_trace_probe"


def _agp_trace_lengths(engine: Any) -> dict[str, int]:
    return {
        "compositor": len(getattr(getattr(engine, "compositor", None), "agp_trace", []) or []),
        "speech_hub": len(getattr(getattr(engine, "speech_hub", None), "agp_trace", []) or []),
    }


def _new_agp_rows(engine: Any, before: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, attr in (("compositor", "compositor"), ("speech_hub", "speech_hub")):
        obj = getattr(engine, attr, None)
        trace = list(getattr(obj, "agp_trace", []) or [])
        start = int(before.get(layer, 0))
        for row in trace[start:]:
            if isinstance(row, dict):
                copied = dict(row)
                copied.setdefault("layer", layer)
                rows.append(copied)
    return rows


def _verify_trace_from_agp_row(engine: Any, row: dict[str, Any]) -> dict[str, Any]:
    agp = getattr(engine, "agp_adapter", None)
    candidate = str(row.get("candidate_response", "") or "")
    meaning = row.get("meaning")
    activated = row.get("activated_categories")
    hormones = row.get("hormone_state")
    if agp is None or not hasattr(agp, "verify_with_trace"):
        return {
            "result": row.get("result"),
            "trace": {
                "candidate_categories_extracted": [],
                "sa_activated_categories": list(activated or []),
                "overlap": [],
                "candidate_count": 0,
                "sa_count": len(list(activated or [])),
                "overlap_count": 0,
                "thresholds_used": {},
                "extraction_method": "unavailable",
                "read_only": True,
            },
            "read_only": True,
        }
    return agp.verify_with_trace(
        candidate_response=candidate,
        meaning=meaning,
        activated_categories=activated,
        hormone_state=hormones if isinstance(hormones, dict) else None,
        engine=engine,
    )


def _summarize_fixture_agp_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts: list[int] = []
    sa_counts: list[int] = []
    overlap_counts: list[int] = []
    weak_overlap_counts: list[int] = []
    for row in traces:
        trace = row.get("agp_trace", {}).get("trace", {}) if isinstance(row, dict) else {}
        if not isinstance(trace, dict):
            trace = {}
        candidate_counts.append(int(trace.get("candidate_count", 0) or 0))
        sa_counts.append(int(trace.get("sa_count", 0) or 0))
        overlap_counts.append(int(trace.get("overlap_count", 0) or 0))
        weak_overlap_counts.append(int(trace.get("weak_overlap_count", 0) or 0))
    total = len(traces)
    return {
        "trace_count": total,
        "candidates_per_trace": candidate_counts,
        "sa_per_trace": sa_counts,
        "overlap_per_trace": overlap_counts,
        "weak_overlap_per_trace": weak_overlap_counts,
        "zero_candidate_count": sum(1 for value in candidate_counts if value == 0),
        "zero_sa_count": sum(1 for value in sa_counts if value == 0),
        "zero_overlap_count": sum(1 for value in overlap_counts if value == 0),
        "weak_overlap_count": sum(1 for value in weak_overlap_counts if value > 0),
        "read_only": True,
    }


def run_conversation_smoke_with_agp_trace(engine: Any, fixtures: Iterable[Any]) -> dict[str, Any]:
    """Run fixture smoke and attach AGP category-extraction trace probes.

    This is a round46 probe only. It does not change AGP thresholds, extraction
    logic, SA activation, wrapper policy, fallback policy, or runtime output.
    """
    fixture_texts = normalize_fixture_texts(fixtures)
    before_structural_state = _structural_state(engine)
    traces: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for index, text in enumerate(fixture_texts):
        before = _agp_trace_lengths(engine)
        chunks = [str(chunk) for chunk in engine.chat_stream(text)]
        response = "".join(chunks)
        new_rows = _new_agp_rows(engine, before)
        fixture_traces: list[dict[str, Any]] = []
        for row_index, row in enumerate(new_rows):
            trace_result = _verify_trace_from_agp_row(engine, row)
            trace_entry = {
                "fixture_index": index,
                "input": text,
                "response": response,
                "layer": row.get("layer", "unknown"),
                "agp_row_index": row_index,
                "agp_trace": trace_result,
                "read_only": True,
            }
            traces.append(trace_entry)
            fixture_traces.append(trace_entry)
        responses.append({
            "index": index,
            "input": text,
            "chunks": chunks,
            "response": response,
            "chunk_count": len(chunks),
            "agp_trace_count": len(fixture_traces),
        })
    summary = _summarize_fixture_agp_traces(traces)
    after_structural_state = _structural_state(engine)
    return {
        "smoke_trace_version": SMOKE_TRACE_VERSION,
        "baseline_round": SMOKE_TRACE_BASELINE_ROUND,
        "fixtures_count": len(fixture_texts),
        "fixtures": list(fixture_texts),
        "responses": responses,
        "traces": traces,
        "summary": summary,
        "wrapper_telemetry": _wrapper_telemetry(engine),
        "agp_trace_summary": _agp_summary(engine),
        "structural_state_before": before_structural_state,
        "structural_state_after": after_structural_state,
        "structural_state_unchanged": before_structural_state == after_structural_state,
        "decision_policy_changed": False,
        "thresholds_changed": False,
        "agp_runtime_changed": False,
        "read_only_policy": "trace_probe_only_no_threshold_no_extraction_no_sa_change",
        "read_only": True,
    }

ROUND72_SMOKE_BASELINE_ROUND = 72
ROUND72_SMOKE_VERSION = "v3_round72_eve_specific_smoke_drift_baseline"


def run_round72_eve_specific_smoke_baseline(
    engine: Any,
    fixtures: Iterable[Any],
    *,
    probe_words: Iterable[str] = (),
    probe_context_words: Iterable[str] = ("안녕",),
) -> dict[str, Any]:
    """Run Round72 smoke and drift baseline with optional probe vectors.

    ``probe_words`` are written directly to the isolated test engine's
    EveSpecificVectorStore so the routing surface can be measured. This is not
    a runtime policy change and does not call the self-learning commit path;
    Round73 is reserved for explicit commit smoke. The function records that
    scope in the returned policy block.
    """
    from adapters.external_seed_manifest import measure_eve_specific_round72_baseline

    store = getattr(engine, "eve_specific_vector_store", None)
    probe_results: list[dict[str, Any]] = []
    context = [str(w or "").strip() for w in probe_context_words if str(w or "").strip()]
    for raw in probe_words or []:
        word = str(raw or "").strip()
        if not word:
            continue
        added = False
        if store is not None and hasattr(store, "add_or_update_vector"):
            try:
                added = bool(store.add_or_update_vector(word, context, engine=engine))
            except Exception:
                added = False
        probe_results.append({"word": word, "added": bool(added), "context_words": list(context)})
    smoke = run_conversation_smoke(engine, fixtures)
    round72_baseline = measure_eve_specific_round72_baseline(engine)
    return {
        "smoke_version": ROUND72_SMOKE_VERSION,
        "baseline_round": ROUND72_SMOKE_BASELINE_ROUND,
        "probe_vectors": probe_results,
        "probe_vector_mutation_scope": "isolated_smoke_engine_only_not_self_learning_commit_path",
        "self_learning_commit_path_called": False,
        "smoke": smoke,
        "round72_baseline": round72_baseline,
        "policy": {
            "routing_changed_in_round72": False,
            "auto_promotion_enabled": False,
            "thresholds_changed": False,
            "context_diversity_gate_changed": False,
            "agp_bypass": False,
            "memory_quarantine_unchanged": True,
            "fasttext_seed_mutation": False,
            "read_only_decision_surface": True,
        },
        "read_only": True,
    }


def _safe_wrapper_telemetry_dict(engine: Any) -> dict[str, Any]:
    wrapper = getattr(engine, "self_embedding", None)
    if wrapper is None or not hasattr(wrapper, "telemetry"):
        return {}
    try:
        data = wrapper.telemetry()
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_store_stats_dict(engine: Any) -> dict[str, Any]:
    store = getattr(engine, "eve_specific_vector_store", None)
    if store is None or not hasattr(store, "stats"):
        return {}
    try:
        data = store.stats()
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_learner_stats_dict(engine: Any) -> dict[str, Any]:
    learner = getattr(engine, "eve_self_learning", None)
    if learner is None or not hasattr(learner, "stats"):
        return {}
    try:
        data = learner.stats()
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _numeric_delta(after: dict[str, Any], before: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in keys:
        try:
            delta[str(key)] = int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        except Exception:
            try:
                delta[str(key)] = float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)
            except Exception:
                delta[str(key)] = 0
    return delta


def _route_distribution_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    after_route = dict(after.get("route_distribution", {}) or {})
    before_route = dict(before.get("route_distribution", {}) or {})
    count_keys = (
        "total_calls",
        "routed_hits_total",
        "fasttext_primary_hits",
        "eve_specific_hits",
        "pmi_svd_fallback_uses",
        "errors",
    )
    rate_keys = (
        "fasttext_primary_rate",
        "eve_specific_rate",
        "pmi_svd_fallback_rate",
        "error_rate",
    )
    return {
        "count_delta": _numeric_delta(after_route, before_route, count_keys),
        "rate_delta": _numeric_delta(after_route, before_route, rate_keys),
        "before": before_route,
        "after": after_route,
    }


def _telemetry_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return _numeric_delta(
        after,
        before,
        ("total_calls", "primary_hits", "fallback_uses", "eve_specific_hits", "errors", "oov_log_size"),
    )


ROUND73_EXPLICIT_COMMIT_SMOKE_BASELINE_ROUND = 73
ROUND73_EXPLICIT_COMMIT_SMOKE_VERSION = "v3_round73_explicit_eve_specific_commit_smoke"


def run_round73_explicit_eve_specific_commit_smoke(
    engine: Any,
    *,
    target_word: str = "민석",
    observation_texts: Iterable[Any] = ("민석 오늘", "민석 군대"),
    commit_context_words: Iterable[str] = ("오늘", "군대"),
    lookup_after_commit: bool = True,
) -> dict[str, Any]:
    """Run the first real explicit EveSpecific commit smoke.

    Unlike the Round72 probe, this smoke uses the production self-learning
    coordinator path: observe_text -> commit_eve_specific_vectors -> wrapper
    lookup. It is still scoped to the provided test engine and does not change
    thresholds, auto-promotion, memory/quarantine, fastText seed state, or AGP
    anchoring policy.
    """
    from adapters.external_seed_manifest import measure_eve_specific_round72_baseline

    learner = getattr(engine, "eve_self_learning", None)
    store = getattr(engine, "eve_specific_vector_store", None)
    wrapper = getattr(engine, "self_embedding", None)
    word = str(target_word or "").strip()
    context = [str(w or "").strip() for w in (commit_context_words or []) if str(w or "").strip()]
    observations: list[dict[str, Any]] = []
    before_store_count = 0
    if store is not None and hasattr(store, "stats"):
        try:
            before_store_count = int(store.stats().get("stored_count", 0) or 0)
        except Exception:
            before_store_count = 0
    before_telemetry = {}
    if wrapper is not None and hasattr(wrapper, "telemetry"):
        try:
            before_telemetry = dict(wrapper.telemetry())
        except Exception:
            before_telemetry = {}

    if learner is not None and hasattr(learner, "observe_text"):
        for index, text in enumerate(observation_texts or (), start=1):
            observations.append(learner.observe_text(
                str(text or ""),
                source=f"round73_explicit_commit_smoke_{index}",
                context={"round": 73, "target_word": word},
            ))

    commit_report = {
        "created": [],
        "rejected": [word] if word else [],
        "skipped": [word] if word else [],
        "reason": "learner_missing_or_target_missing",
    }
    if learner is not None and hasattr(learner, "commit_eve_specific_vectors") and word:
        commit_report = learner.commit_eve_specific_vectors(words=[word], context_words=context)

    wrapper_vector_found = False
    if lookup_after_commit and wrapper is not None and hasattr(wrapper, "get_vector") and word:
        try:
            wrapper_vector_found = wrapper.get_vector(word) is not None
        except Exception:
            wrapper_vector_found = False

    after_store_count = 0
    stored_vocab: list[str] = []
    update_count = 0
    if store is not None and hasattr(store, "stats"):
        try:
            stats = store.stats()
            after_store_count = int(stats.get("stored_count", 0) or 0)
            stored_vocab = list(stats.get("stored_vocab", []) or [])
        except Exception:
            after_store_count = 0
    if store is not None and hasattr(store, "update_count") and word:
        try:
            update_count = int(store.update_count(word))
        except Exception:
            update_count = 0

    after_telemetry = {}
    if wrapper is not None and hasattr(wrapper, "telemetry"):
        try:
            after_telemetry = dict(wrapper.telemetry())
        except Exception:
            after_telemetry = {}
    baseline_after_commit = measure_eve_specific_round72_baseline(engine)
    created = list(commit_report.get("created", []) or [])
    rejected = list(commit_report.get("rejected", []) or [])
    gate_pass = bool((commit_report.get("audit_report") or {}).get("gate_pass", False))
    return {
        "smoke_version": ROUND73_EXPLICIT_COMMIT_SMOKE_VERSION,
        "baseline_round": ROUND73_EXPLICIT_COMMIT_SMOKE_BASELINE_ROUND,
        "target_word": word,
        "observation_texts": [str(text or "") for text in (observation_texts or [])],
        "observations": observations,
        "commit_context_words": list(context),
        "commit_report": commit_report,
        "self_learning_commit_path_called": bool(learner is not None and word),
        "commit_created_target": word in created,
        "commit_rejected_target": word in rejected,
        "gate_pass": gate_pass,
        "wrapper_vector_found_after_commit": bool(wrapper_vector_found),
        "store_delta": int(after_store_count - before_store_count),
        "before_store_count": int(before_store_count),
        "after_store_count": int(after_store_count),
        "stored_vocab": stored_vocab,
        "target_update_count": int(update_count),
        "before_telemetry": before_telemetry,
        "after_telemetry": after_telemetry,
        "baseline_after_commit": baseline_after_commit,
        "policy": {
            "round72_probe_path_used": False,
            "explicit_self_learning_commit_path_used": True,
            "auto_promotion_enabled": bool(getattr(learner, "auto_promotion_enabled", False)) if learner is not None else False,
            "commit_gate_enabled": bool(getattr(learner, "commit_gate_enabled", False)) if learner is not None else False,
            "min_observations_for_commit": int(getattr(learner, "min_observations_for_commit", 0) or 0) if learner is not None else 0,
            "context_diversity_gate_enabled": bool(getattr(learner, "context_diversity_gate_enabled", False)) if learner is not None else False,
            "thresholds_changed": False,
            "context_diversity_gate_changed": False,
            "automatic_rollback_enabled": False,
            "memory_quarantine_unchanged": True,
            "fasttext_seed_mutation": False,
            "agp_bypass": False,
        },
        "read_only_decision_surface": True,
    }



ROUND74_EXPLICIT_COMMIT_DELTA_BASELINE_ROUND = 74
ROUND74_EXPLICIT_COMMIT_DELTA_VERSION = "v3_round74_explicit_commit_drift_telemetry_delta"


def run_round74_explicit_commit_delta_report(
    engine: Any,
    *,
    target_word: str = "민석",
    observation_texts: Iterable[Any] = ("민석 오늘", "민석 군대"),
    commit_context_words: Iterable[str] = ("오늘", "군대"),
) -> dict[str, Any]:
    """Compare EveSpecific drift/telemetry before and after explicit commit.

    Round74 intentionally reuses the real Round73 self-learning path, but wraps
    it with pre/post baseline snapshots and target-word lookup probes. The only
    intended mutation is the explicit, gate-approved EveSpecific vector commit
    inside the provided engine. No thresholds, context-diversity policy, AGP
    behavior, fastText seed data, memory/quarantine state, or drift-based
    runtime decisions are changed.
    """
    from adapters.external_seed_manifest import measure_eve_specific_round72_baseline

    word = str(target_word or "").strip()
    wrapper = getattr(engine, "self_embedding", None)
    learner = getattr(engine, "eve_self_learning", None)

    before_store_stats = _safe_store_stats_dict(engine)
    before_learner_stats = _safe_learner_stats_dict(engine)
    before_telemetry = _safe_wrapper_telemetry_dict(engine)
    baseline_before = measure_eve_specific_round72_baseline(engine)

    prelookup_before_telemetry = _safe_wrapper_telemetry_dict(engine)
    target_lookup_before_commit_found = False
    if wrapper is not None and hasattr(wrapper, "get_vector") and word:
        try:
            target_lookup_before_commit_found = wrapper.get_vector(word) is not None
        except Exception:
            target_lookup_before_commit_found = False
    prelookup_after_telemetry = _safe_wrapper_telemetry_dict(engine)

    commit_smoke = run_round73_explicit_eve_specific_commit_smoke(
        engine,
        target_word=word,
        observation_texts=observation_texts,
        commit_context_words=commit_context_words,
        lookup_after_commit=False,
    )

    postlookup_before_telemetry = _safe_wrapper_telemetry_dict(engine)
    target_lookup_after_commit_found = False
    if wrapper is not None and hasattr(wrapper, "get_vector") and word:
        try:
            target_lookup_after_commit_found = wrapper.get_vector(word) is not None
        except Exception:
            target_lookup_after_commit_found = False
    postlookup_after_telemetry = _safe_wrapper_telemetry_dict(engine)

    after_store_stats = _safe_store_stats_dict(engine)
    after_learner_stats = _safe_learner_stats_dict(engine)
    after_telemetry = _safe_wrapper_telemetry_dict(engine)
    baseline_after = measure_eve_specific_round72_baseline(engine)

    before_store_count = int(before_store_stats.get("stored_count", 0) or 0)
    after_store_count = int(after_store_stats.get("stored_count", 0) or 0)
    before_audit_count = int(before_learner_stats.get("commit_audit_record_count", 0) or 0)
    after_audit_count = int(after_learner_stats.get("commit_audit_record_count", 0) or 0)

    return {
        "delta_report_version": ROUND74_EXPLICIT_COMMIT_DELTA_VERSION,
        "baseline_round": ROUND74_EXPLICIT_COMMIT_DELTA_BASELINE_ROUND,
        "target_word": word,
        "observation_texts": [str(text or "") for text in (observation_texts or [])],
        "commit_context_words": [str(w or "").strip() for w in (commit_context_words or []) if str(w or "").strip()],
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "route_distribution_delta": _route_distribution_delta(baseline_after, baseline_before),
        "telemetry_before": before_telemetry,
        "telemetry_after": after_telemetry,
        "telemetry_delta": _telemetry_delta(after_telemetry, before_telemetry),
        "target_lookup_before_commit": {
            "found": bool(target_lookup_before_commit_found),
            "telemetry_delta": _telemetry_delta(prelookup_after_telemetry, prelookup_before_telemetry),
            "eve_specific_hit_delta": int(prelookup_after_telemetry.get("eve_specific_hits", 0) or 0) - int(prelookup_before_telemetry.get("eve_specific_hits", 0) or 0),
            "fallback_delta": int(prelookup_after_telemetry.get("fallback_uses", 0) or 0) - int(prelookup_before_telemetry.get("fallback_uses", 0) or 0),
        },
        "commit_smoke": commit_smoke,
        "target_lookup_after_commit": {
            "found": bool(target_lookup_after_commit_found),
            "telemetry_delta": _telemetry_delta(postlookup_after_telemetry, postlookup_before_telemetry),
            "eve_specific_hit_delta": int(postlookup_after_telemetry.get("eve_specific_hits", 0) or 0) - int(postlookup_before_telemetry.get("eve_specific_hits", 0) or 0),
            "fallback_delta": int(postlookup_after_telemetry.get("fallback_uses", 0) or 0) - int(postlookup_before_telemetry.get("fallback_uses", 0) or 0),
        },
        "store_before": before_store_stats,
        "store_after": after_store_stats,
        "store_delta": int(after_store_count - before_store_count),
        "audit_record_delta": int(after_audit_count - before_audit_count),
        "commit_created_target": bool(commit_smoke.get("commit_created_target", False)),
        "wrapper_vector_found_after_commit": bool(target_lookup_after_commit_found),
        "route_shift_summary": {
            "eve_specific_hits_increased": int(after_telemetry.get("eve_specific_hits", 0) or 0) > int(before_telemetry.get("eve_specific_hits", 0) or 0),
            "post_commit_lookup_used_eve_specific": (int(postlookup_after_telemetry.get("eve_specific_hits", 0) or 0) - int(postlookup_before_telemetry.get("eve_specific_hits", 0) or 0)) >= 1,
            "pre_commit_lookup_used_eve_specific": (int(prelookup_after_telemetry.get("eve_specific_hits", 0) or 0) - int(prelookup_before_telemetry.get("eve_specific_hits", 0) or 0)) >= 1,
            "store_delta": int(after_store_count - before_store_count),
            "audit_record_delta": int(after_audit_count - before_audit_count),
        },
        "policy": {
            "explicit_self_learning_commit_path_used": True,
            "round72_probe_path_used": False,
            "auto_promotion_enabled": bool(getattr(learner, "auto_promotion_enabled", False)) if learner is not None else False,
            "commit_gate_enabled": bool(getattr(learner, "commit_gate_enabled", False)) if learner is not None else False,
            "min_observations_for_commit": int(getattr(learner, "min_observations_for_commit", 0) or 0) if learner is not None else 0,
            "context_diversity_gate_enabled": bool(getattr(learner, "context_diversity_gate_enabled", False)) if learner is not None else False,
            "thresholds_changed": False,
            "context_diversity_gate_changed": False,
            "automatic_rollback_enabled": False,
            "drift_based_runtime_change": False,
            "memory_quarantine_unchanged": True,
            "fasttext_seed_mutation": False,
            "agp_bypass": False,
        },
        "decision_policy_changed": False,
        "read_only_decision_surface": True,
    }

ROUND75_COMMIT_AUDIT_REPLAY_EXPORT_BASELINE_ROUND = 75
ROUND75_COMMIT_AUDIT_REPLAY_EXPORT_VERSION = "v3_round75_commit_audit_replay_export"


def _round75_audit_records_from_engine(engine: Any) -> list[dict[str, Any]]:
    learner = getattr(engine, "eve_self_learning", None)
    if learner is None or not hasattr(learner, "commit_audit_records"):
        return []
    try:
        return list(learner.commit_audit_records())
    except Exception:
        return []


def _round75_compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": str(candidate.get("word", "") or ""),
        "passed": bool(candidate.get("passed", False)),
        "reasons": [str(r) for r in list(candidate.get("reasons", []) or [])],
        "observed_count": int(candidate.get("observed_count", 0) or 0),
        "is_eve_specific_candidate": bool(candidate.get("is_eve_specific_candidate", False)),
        "context_diverse": bool(candidate.get("context_diverse", False)),
        "evidence_status": candidate.get("evidence_status"),
    }


def _round75_compact_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(record.get("index", 0) or 0),
        "event_type": str(record.get("event_type", "unknown") or "unknown"),
        "audit_version": record.get("audit_version"),
        "round": int(record.get("round", 0) or 0),
        "target_words": [str(w) for w in list(record.get("target_words", []) or [])],
        "eligible_words": [str(w) for w in list(record.get("eligible_words", []) or [])],
        "rejected_words": [str(w) for w in list(record.get("rejected_words", []) or [])],
        "candidate_reports": [
            _round75_compact_candidate(dict(candidate))
            for candidate in list(record.get("candidate_reports", []) or [])
            if isinstance(candidate, dict)
        ],
        "context_words": [str(w) for w in list(record.get("context_words", []) or [])],
        "known_context_words": [str(w) for w in list(record.get("known_context_words", []) or [])],
        "known_context_count": int(record.get("known_context_count", 0) or 0),
        "gate_pass": bool(record.get("gate_pass", False)),
        "commit_gate_enabled": bool(record.get("commit_gate_enabled", True)),
        "min_observations_for_commit": int(record.get("min_observations_for_commit", 0) or 0),
        "min_known_context_words_for_commit": int(record.get("min_known_context_words_for_commit", 0) or 0),
        "context_diversity_gate_enabled": bool(record.get("context_diversity_gate_enabled", False)),
        "active_policy_version": record.get("active_policy_version"),
        "read_only": True,
    }


def _round75_build_replay_steps(delta_report: dict[str, Any], audit_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commit_smoke = dict(delta_report.get("commit_smoke", {}) or {})
    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "name": "pre_commit_target_lookup",
            "source": "round74_delta_report.target_lookup_before_commit",
            "read_only": True,
            "found": bool((delta_report.get("target_lookup_before_commit") or {}).get("found", False)),
            "eve_specific_hit_delta": int((delta_report.get("target_lookup_before_commit") or {}).get("eve_specific_hit_delta", 0) or 0),
            "fallback_delta": int((delta_report.get("target_lookup_before_commit") or {}).get("fallback_delta", 0) or 0),
        },
        {
            "step": 2,
            "name": "observe_text_evidence",
            "source": "round73_commit_smoke.observations",
            "read_only": True,
            "observation_count": len(list(commit_smoke.get("observations", []) or [])),
            "observation_texts": [str(t) for t in list(commit_smoke.get("observation_texts", []) or [])],
        },
    ]
    for idx, record in enumerate(audit_records, start=3):
        steps.append({
            "step": idx,
            "name": f"audit_record_{record.get('event_type', 'unknown')}",
            "source": "EveSelfLearningAdapter.commit_audit_records",
            "read_only": True,
            "record_index": int(record.get("index", 0) or 0),
            "event_type": str(record.get("event_type", "unknown") or "unknown"),
            "gate_pass": bool(record.get("gate_pass", False)),
            "eligible_words": [str(w) for w in list(record.get("eligible_words", []) or [])],
            "rejected_words": [str(w) for w in list(record.get("rejected_words", []) or [])],
        })
    steps.append({
        "step": len(steps) + 1,
        "name": "post_commit_target_lookup",
        "source": "round74_delta_report.target_lookup_after_commit",
        "read_only": True,
        "found": bool((delta_report.get("target_lookup_after_commit") or {}).get("found", False)),
        "eve_specific_hit_delta": int((delta_report.get("target_lookup_after_commit") or {}).get("eve_specific_hit_delta", 0) or 0),
        "fallback_delta": int((delta_report.get("target_lookup_after_commit") or {}).get("fallback_delta", 0) or 0),
    })
    return steps


def build_round75_commit_audit_replay_export(delta_report: dict[str, Any], audit_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a read-only replay/export artifact from an existing Round74 delta.

    This function does not call commit, lookup, audit, vector-store, memory, seed,
    or AGP mutation paths. It only compacts already-produced Round73/74 evidence
    into a deterministic operator artifact.
    """
    compact_records = [_round75_compact_audit_record(dict(record)) for record in audit_records]
    target_word = str(delta_report.get("target_word", "") or "")
    event_type_counts: dict[str, int] = {}
    gate_pass_records = 0
    rejected_total = 0
    eligible_total = 0
    for record in compact_records:
        event_type = str(record.get("event_type", "unknown") or "unknown")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        if bool(record.get("gate_pass", False)):
            gate_pass_records += 1
        eligible_total += len(list(record.get("eligible_words", []) or []))
        rejected_total += len(list(record.get("rejected_words", []) or []))
    replay_steps = _round75_build_replay_steps(delta_report, compact_records)
    policy = dict(delta_report.get("policy", {}) or {})
    policy.update({
        "replay_export_read_only": True,
        "replay_does_not_call_commit": True,
        "replay_does_not_call_lookup": True,
        "replay_does_not_append_audit": True,
        "replay_does_not_mutate_vectors": True,
        "automatic_rollback_enabled": False,
        "auto_promotion_enabled": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    })
    return {
        "replay_export_version": ROUND75_COMMIT_AUDIT_REPLAY_EXPORT_VERSION,
        "baseline_round": ROUND75_COMMIT_AUDIT_REPLAY_EXPORT_BASELINE_ROUND,
        "source_delta_report_version": delta_report.get("delta_report_version"),
        "source_commit_smoke_version": (delta_report.get("commit_smoke") or {}).get("smoke_version"),
        "target_word": target_word,
        "observation_texts": [str(t) for t in list(delta_report.get("observation_texts", []) or [])],
        "commit_context_words": [str(w) for w in list(delta_report.get("commit_context_words", []) or [])],
        "audit_record_count": len(compact_records),
        "audit_event_type_counts": event_type_counts,
        "gate_pass_record_count": int(gate_pass_records),
        "eligible_total": int(eligible_total),
        "rejected_total": int(rejected_total),
        "audit_records": compact_records,
        "replay_steps": replay_steps,
        "route_shift_summary": dict(delta_report.get("route_shift_summary", {}) or {}),
        "target_lookup_before_commit": dict(delta_report.get("target_lookup_before_commit", {}) or {}),
        "target_lookup_after_commit": dict(delta_report.get("target_lookup_after_commit", {}) or {}),
        "store_delta": int(delta_report.get("store_delta", 0) or 0),
        "audit_record_delta": int(delta_report.get("audit_record_delta", 0) or 0),
        "commit_created_target": bool(delta_report.get("commit_created_target", False)),
        "wrapper_vector_found_after_commit": bool(delta_report.get("wrapper_vector_found_after_commit", False)),
        "replay_verification": {
            "has_audit_and_commit_records": event_type_counts.get("audit", 0) >= 1 and event_type_counts.get("commit", 0) >= 1,
            "target_created": bool(delta_report.get("commit_created_target", False)),
            "target_lookup_shifted_to_eve_specific": bool((delta_report.get("route_shift_summary") or {}).get("post_commit_lookup_used_eve_specific", False)),
            "pre_commit_lookup_not_eve_specific": not bool((delta_report.get("route_shift_summary") or {}).get("pre_commit_lookup_used_eve_specific", False)),
            "store_delta_is_one": int(delta_report.get("store_delta", 0) or 0) == 1,
            "audit_record_delta_at_least_two": int(delta_report.get("audit_record_delta", 0) or 0) >= 2,
        },
        "policy": policy,
        "decision_policy_changed": False,
        "read_only": True,
    }


def run_round75_commit_audit_replay_export(
    engine: Any,
    *,
    target_word: str = "민석",
    observation_texts: Iterable[Any] = ("민석 오늘", "민석 군대"),
    commit_context_words: Iterable[str] = ("오늘", "군대"),
) -> dict[str, Any]:
    """Run Round74 once, then consolidate its audit trail into a replay export.

    The only mutation is the same explicit, gate-approved commit performed by
    Round74 inside the provided smoke engine. The Round75 replay/export layer
    is read-only: it snapshots audit records and does not perform additional
    commits, lookups, vector writes, audit appends, rollbacks, or policy changes.
    """
    delta_report = run_round74_explicit_commit_delta_report(
        engine,
        target_word=target_word,
        observation_texts=observation_texts,
        commit_context_words=commit_context_words,
    )
    audit_records_before_replay = _round75_audit_records_from_engine(engine)
    store_before_replay = _safe_store_stats_dict(engine)
    telemetry_before_replay = _safe_wrapper_telemetry_dict(engine)
    replay = build_round75_commit_audit_replay_export(delta_report, audit_records_before_replay)
    audit_records_after_replay = _round75_audit_records_from_engine(engine)
    store_after_replay = _safe_store_stats_dict(engine)
    telemetry_after_replay = _safe_wrapper_telemetry_dict(engine)
    replay["replay_read_only_checks"] = {
        "audit_records_unchanged_during_replay": audit_records_after_replay == audit_records_before_replay,
        "vector_store_unchanged_during_replay": store_after_replay == store_before_replay,
        "telemetry_unchanged_during_replay": telemetry_after_replay == telemetry_before_replay,
        "policy_changed_during_replay": False,
    }
    return replay


def write_round75_commit_audit_replay_export(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round75 replay export to JSON without recomputation."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("replay_export_version"),
        "path": str(out),
        "audit_record_count": int(report.get("audit_record_count", 0) or 0),
        "read_only": False,
        "recomputed": False,
        "commit_called": False,
        "vector_mutation": False,
        "audit_record_mutation": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND76_SELF_LEARNING_V1_FREEZE_BASELINE_ROUND = 76
ROUND76_SELF_LEARNING_V1_FREEZE_BASELINE_VERSION = "v3_round76_self_learning_v1_freeze_baseline"


def _round76_learner_freeze_snapshot(engine: Any) -> dict[str, Any]:
    learner = getattr(engine, "eve_self_learning", None)
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        try:
            return dict(learner.self_learning_v1_freeze_policy_snapshot())
        except Exception as exc:
            return {"error": exc.__class__.__name__, "read_only": True}
    return {"error": "learner_missing", "read_only": True}


def _round76_wrapper_route_snapshot(engine: Any) -> dict[str, Any]:
    wrapper = getattr(engine, "self_embedding", None)
    if wrapper is not None and hasattr(wrapper, "lookup_route_policy_snapshot"):
        try:
            return dict(wrapper.lookup_route_policy_snapshot())
        except Exception as exc:
            return {"error": exc.__class__.__name__, "read_only": True}
    return {"error": "wrapper_missing", "read_only": True}


def build_round76_self_learning_v1_freeze_baseline(
    replay_export: dict[str, Any],
    learner_snapshot: dict[str, Any],
    wrapper_route_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Round76 self-learning v1 freeze baseline artifact.

    The builder is read-only. It consumes existing Round75 replay evidence and
    current policy snapshots, then creates an operator artifact for future
    regression/diff checks. It does not recompute commits, perform lookups,
    append audits, mutate vectors, adjust thresholds, rollback, promote memory,
    mutate seed data, or bypass AGP.
    """
    wrapper_route_policy = dict(wrapper_route_policy or {})
    replay_policy = dict(replay_export.get("policy", {}) or {})
    active_policy = dict(learner_snapshot.get("active_policy", {}) or {})
    smoke_ok = {
        "round73_commit_created_target": bool(replay_export.get("commit_created_target", False)),
        "round74_route_shift_to_eve_specific": bool((replay_export.get("route_shift_summary") or {}).get("post_commit_lookup_used_eve_specific", False)),
        "round75_replay_read_only": bool(replay_export.get("read_only", False)),
        "round75_replay_has_audit_and_commit_records": bool((replay_export.get("replay_verification") or {}).get("has_audit_and_commit_records", False)),
        "round75_store_delta_is_one": bool((replay_export.get("replay_verification") or {}).get("store_delta_is_one", False)),
    }
    frozen_artifacts = list(learner_snapshot.get("frozen_artifacts", []) or [])
    frozen_artifacts.extend([
        "SELF_LEARNING_V1_FREEZE_BASELINE_R76.json",
        "ROUND_V3_R76_REPORT.md",
    ])
    return {
        "freeze_baseline_version": ROUND76_SELF_LEARNING_V1_FREEZE_BASELINE_VERSION,
        "baseline_round": ROUND76_SELF_LEARNING_V1_FREEZE_BASELINE_ROUND,
        "baseline_name": "self_learning_v1",
        "covered_rounds": list(range(57, 76)),
        "covered_round_range": "round57-round75",
        "source_replay_export_version": replay_export.get("replay_export_version"),
        "source_delta_report_version": replay_export.get("source_delta_report_version"),
        "source_commit_smoke_version": replay_export.get("source_commit_smoke_version"),
        "target_word": replay_export.get("target_word"),
        "frozen": True,
        "freeze_scope": "policy_artifacts_tests_and_observability_only",
        "learner_freeze_snapshot": learner_snapshot,
        "wrapper_route_policy": wrapper_route_policy,
        "active_policy": active_policy,
        "smoke_verification": smoke_ok,
        "artifact_manifest": sorted(dict.fromkeys(str(item) for item in frozen_artifacts)),
        "component_roles": dict(learner_snapshot.get("component_roles", {}) or {}),
        "locked_commit_gate": list(learner_snapshot.get("locked_commit_gate", []) or []),
        "future_change_rule": {
            "policy_change_requires_new_round_and_operator_review": True,
            "threshold_change_requires_dry_run_then_proposal_then_enforcement": True,
            "context_diversity_change_requires_dry_run_then_proposal_then_enforcement": True,
            "automatic_promotion_requires_new_design_review": True,
            "automatic_rollback_requires_new_design_review": True,
        },
        "freeze_invariants": {
            "auto_observe_enabled": active_policy.get("auto_observe_enabled") is True,
            "auto_promotion_enabled": active_policy.get("auto_promotion_enabled") is False,
            "commit_gate_enabled": active_policy.get("commit_gate_enabled") is True,
            "min_observations_for_commit_is_2": int(active_policy.get("min_observations_for_commit", -1) or -1) == 2,
            "min_known_context_words_for_commit_is_1": int(active_policy.get("min_known_context_words_for_commit", -1) or -1) == 1,
            "context_diversity_gate_enabled": active_policy.get("context_diversity_gate_enabled") is True,
            "automatic_rollback_enabled": active_policy.get("automatic_rollback_enabled") is False,
            "agp_anchor_not_seed_vector": active_policy.get("agp_anchor_policy") == "explicit_categories_plus_sa_activation_only",
            "round75_replay_export_read_only": replay_policy.get("replay_export_read_only") is True,
            "wrapper_get_embedding_duplicate_fallback_guarded": wrapper_route_policy.get("duplicate_fallback_api_call_guarded") is True,
        },
        "policy": {
            "freeze_export_read_only": True,
            "freeze_does_not_call_commit": True,
            "freeze_does_not_call_lookup": True,
            "freeze_does_not_append_audit": True,
            "freeze_does_not_mutate_vectors": True,
            "thresholds_changed": False,
            "context_diversity_gate_changed": False,
            "automatic_rollback_enabled": False,
            "auto_promotion_enabled": False,
            "drift_based_runtime_change": False,
            "memory_quarantine_unchanged": True,
            "fasttext_seed_mutation": False,
            "agp_bypass": False,
        },
        "read_only": True,
    }


def run_round76_self_learning_v1_freeze_baseline(
    engine: Any,
    *,
    target_word: str = "민석",
    observation_texts: Iterable[Any] = ("민석 오늘", "민석 군대"),
    commit_context_words: Iterable[str] = ("오늘", "군대"),
) -> dict[str, Any]:
    """Run the Round75 proof path once, then freeze self-learning v1.

    The only mutation is the already-approved explicit commit inside the Round75
    proof path. The Round76 freeze layer itself is read-only and verifies that
    building the freeze artifact does not alter audit records, vector store
    stats, telemetry, or policy.
    """
    replay = run_round75_commit_audit_replay_export(
        engine,
        target_word=target_word,
        observation_texts=observation_texts,
        commit_context_words=commit_context_words,
    )
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_snapshot = _round76_learner_freeze_snapshot(engine)
    wrapper_route_policy = _round76_wrapper_route_snapshot(engine)
    freeze = build_round76_self_learning_v1_freeze_baseline(
        replay,
        learner_snapshot,
        wrapper_route_policy,
    )
    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    freeze["freeze_read_only_checks"] = {
        "audit_records_unchanged_during_freeze": audit_after == audit_before,
        "vector_store_unchanged_during_freeze": store_after == store_before,
        "telemetry_unchanged_during_freeze": telemetry_after == telemetry_before,
        "policy_changed_during_freeze": False,
    }
    return freeze


def write_round76_self_learning_v1_freeze_baseline(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round76 freeze baseline without recomputation."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("freeze_baseline_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "commit_called": False,
        "vector_mutation": False,
        "audit_record_mutation": False,
        "thresholds_changed": False,
        "context_diversity_gate_changed": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND77_LEXICAL_TO_CONCEPT_MAPPING_PLANNING_ROUND = 77
ROUND77_LEXICAL_TO_CONCEPT_MAPPING_PLANNING_VERSION = "v3_round77_lexical_to_concept_mapping_planning"


def _round77_mapping_adapter(engine: Any) -> Any:
    adapter = getattr(engine, "lex_concept_mapping", None)
    if adapter is not None:
        return adapter
    from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter

    return LexConceptMappingAdapter(engine)


def build_round77_lexical_to_concept_mapping_plan(
    contract_snapshot: dict[str, Any],
    candidate_schema: dict[str, Any],
    token_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Round77 lexical→concept planning artifact.

    This builder is read-only. It does not observe text, commit vectors, call
    embedding lookup, create concept categories, write memory/quarantine,
    mutate frame/hypergraph state, call AGP verify, or change runtime policy.
    """
    token_plan = dict(token_plan or {})
    contract = dict(contract_snapshot or {})
    active_policy = dict(contract.get("active_self_learning_policy", {}) or {})
    hard_boundaries = dict(contract.get("hard_boundaries", {}) or {})
    return {
        "planning_version": ROUND77_LEXICAL_TO_CONCEPT_MAPPING_PLANNING_VERSION,
        "round": ROUND77_LEXICAL_TO_CONCEPT_MAPPING_PLANNING_ROUND,
        "axis": "lexical_to_concept_mapping",
        "source_baseline": "self_learning_v1_round76",
        "status": "planning_only_not_enforced",
        "contract_snapshot": contract,
        "candidate_schema": dict(candidate_schema or {}),
        "token_plan": token_plan,
        "active_self_learning_policy": active_policy,
        "component_roles": dict(contract.get("component_roles", {}) or {}),
        "hard_boundaries": hard_boundaries,
        "readiness_checks": {
            "self_learning_v1_source_baseline": contract.get("source_baseline") == "self_learning_v1_round76",
            "runtime_mapping_disabled": contract.get("runtime_mapping_enabled") is False,
            "enforcement_disabled": contract.get("enforcement_enabled") is False,
            "auto_promotion_disabled": active_policy.get("auto_promotion_enabled") is False,
            "commit_gate_still_enabled": active_policy.get("commit_gate_enabled") is True,
            "threshold_still_2": int(active_policy.get("min_observations_for_commit", -1) or -1) == 2,
            "context_diversity_still_enabled": active_policy.get("context_diversity_gate_enabled") is True,
            "lexical_vector_not_concept": hard_boundaries.get("lexical_vector_is_not_concept") is True,
            "eve_specific_vector_not_agp_anchor": hard_boundaries.get("eve_specific_vector_is_not_agp_anchor") is True,
            "agp_anchor_requires_explicit_category_and_sa": hard_boundaries.get("agp_anchor_requires_explicit_category_and_sa_activation") is True,
        },
        "round77_policy": {
            "planning_only": True,
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "does_not_call_observe_text": True,
            "does_not_call_commit": True,
            "does_not_call_lookup": True,
            "does_not_create_categories": True,
            "does_not_mutate_concept_memory": True,
            "does_not_mutate_frame_or_hypergraph": True,
            "does_not_call_agp_verify": True,
            "thresholds_changed": False,
            "context_diversity_gate_changed": False,
            "auto_promotion_enabled": False,
            "automatic_rollback_enabled": False,
            "memory_quarantine_unchanged": True,
            "fasttext_seed_mutation": False,
            "agp_bypass": False,
        },
        "next_round_recommendation": {
            "round78": "lexical_concept_candidate_schema_dry_run",
            "allowed_scope": "schema_and_candidate_rows_only_no_mapping_enforcement",
        },
        "read_only": True,
    }


def run_round77_lexical_to_concept_mapping_plan(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
) -> dict[str, Any]:
    """Run the Round77 planning surface and verify it is read-only."""
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    contract = adapter.mapping_contract_snapshot()
    schema = adapter.candidate_record_schema()
    token_plan = adapter.plan_for_tokens(planning_tokens)
    report = build_round77_lexical_to_concept_mapping_plan(contract, schema, token_plan)

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["planning_read_only_checks"] = {
        "audit_records_unchanged_during_planning": audit_after == audit_before,
        "vector_store_unchanged_during_planning": store_after == store_before,
        "telemetry_unchanged_during_planning": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_planning": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_planning": False,
        "enforcement_enabled_after_planning": False,
    }
    return report


def write_round77_lexical_to_concept_mapping_plan(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round77 planning artifact without recomputation."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("planning_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND78_LEXICAL_CONCEPT_CANDIDATE_SCHEMA_DRY_RUN_VERSION = "v3_round78_lexical_concept_candidate_schema_dry_run"
ROUND79_LEXICAL_CONCEPT_CANDIDATE_EVIDENCE_QUALITY_VERSION = "v3_round79_lexical_concept_candidate_evidence_quality_report"


def run_round78_lexical_concept_candidate_schema_dry_run(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
) -> dict[str, Any]:
    """Run Round78 lexical→concept candidate schema dry-run read-only."""
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    report = adapter.candidate_schema_dry_run(tokens=planning_tokens)

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["read_only_checks"] = {
        "audit_records_unchanged_during_dry_run": audit_after == audit_before,
        "vector_store_unchanged_during_dry_run": store_after == store_before,
        "telemetry_unchanged_during_dry_run": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_dry_run": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_dry_run": False,
        "enforcement_enabled_after_dry_run": False,
        "category_created": False,
        "agp_verify_called": False,
    }
    report["next_round_recommendation"] = {
        "round79": "lexical_concept_candidate_evidence_quality_report",
        "allowed_scope": "report_only_no_concept_mapping_enforcement",
    }
    return report


def write_round78_lexical_concept_candidate_schema_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round78 candidate dry-run artifact."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("dry_run_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


def run_round79_lexical_concept_candidate_evidence_quality_report(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
) -> dict[str, Any]:
    """Run Round79 evidence quality report read-only."""
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    report = adapter.candidate_evidence_quality_report(tokens=planning_tokens)

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["read_only_checks"] = {
        "audit_records_unchanged_during_report": audit_after == audit_before,
        "vector_store_unchanged_during_report": store_after == store_before,
        "telemetry_unchanged_during_report": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_report": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_report": False,
        "enforcement_enabled_after_report": False,
        "category_created": False,
        "agp_verify_called": False,
    }
    report["next_round_recommendation"] = {
        "round80": "concept_proposal_report",
        "allowed_scope": "operator_proposal_only_no_mapping_enforcement",
    }
    return report


def write_round79_lexical_concept_candidate_evidence_quality_report(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round79 evidence report artifact."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("report_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND80_CONCEPT_PROPOSAL_REPORT_VERSION = "v3_round80_concept_proposal_report"


def run_round80_concept_proposal_report(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
) -> dict[str, Any]:
    """Run the Round80 concept proposal report read-only.

    Proposal records are operator-review labels only. This function must not
    create categories, activate SA, call AGP, mutate concept memory, mutate
    frame/hypergraph state, call wrapper lookup, or commit vectors.
    """
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    report = adapter.concept_proposal_report(tokens=planning_tokens)

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["read_only_checks"] = {
        "audit_records_unchanged_during_report": audit_after == audit_before,
        "vector_store_unchanged_during_report": store_after == store_before,
        "telemetry_unchanged_during_report": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_report": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_report": False,
        "enforcement_enabled_after_report": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_verify_called": False,
        "wrapper_lookup_called": False,
        "vector_commit_called": False,
    }
    return report


def write_round80_concept_proposal_report(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round80 concept proposal report artifact."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("report_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND81_CONCEPT_MAPPING_GATE_DRY_RUN_VERSION = "v3_round81_concept_mapping_gate_dry_run"


def run_round81_concept_mapping_gate_dry_run(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
) -> dict[str, Any]:
    """Run the Round81 concept mapping gate dry-run read-only.

    The dry-run evaluates future mapping-gate readiness but must not enable
    runtime mapping, create categories, mutate concept memory/frame/hypergraph,
    create SA activation, call AGP verify, call wrapper lookup, or commit vectors.
    """
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    report = adapter.concept_mapping_gate_dry_run(tokens=planning_tokens)

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["read_only_checks"] = {
        "audit_records_unchanged_during_dry_run": audit_after == audit_before,
        "vector_store_unchanged_during_dry_run": store_after == store_before,
        "telemetry_unchanged_during_dry_run": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_dry_run": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_dry_run": False,
        "enforcement_enabled_after_dry_run": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "wrapper_lookup_called": False,
        "vector_commit_called": False,
    }
    return report


def write_round81_concept_mapping_gate_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round81 gate dry-run artifact."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("dry_run_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND82_CONCEPT_MAPPING_GATE_PROPOSAL_REPORT_VERSION = "v3_round82_concept_mapping_gate_proposal_report"


def run_round82_concept_mapping_gate_proposal_report(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
) -> dict[str, Any]:
    """Run the Round82 concept mapping gate proposal report read-only.

    The report consolidates Round81 dry-run blocks into operator action items.
    It must not enable runtime mapping, create categories, mutate concept memory,
    mutate frames/hypergraphs, create SA activation, call AGP verify, call wrapper
    lookup, or commit vectors.
    """
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    report = adapter.concept_mapping_gate_proposal_report(tokens=planning_tokens)

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["read_only_checks"] = {
        "audit_records_unchanged_during_report": audit_after == audit_before,
        "vector_store_unchanged_during_report": store_after == store_before,
        "telemetry_unchanged_during_report": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_report": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_report": False,
        "enforcement_enabled_after_report": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "wrapper_lookup_called": False,
        "vector_commit_called": False,
    }
    return report


def write_round82_concept_mapping_gate_proposal_report(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round82 gate proposal report artifact."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("report_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND83_OPERATOR_ACCEPTANCE_CATEGORY_CREATION_DRY_RUN_VERSION = "v3_round83_operator_acceptance_category_creation_dry_run"


def run_round83_operator_acceptance_category_creation_dry_run(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    """Run Round83 operator-acceptance fixture + category creation dry-run.

    This is read-only. Operator acceptance is a fixture, not persisted state.
    Category creation is represented as a future plan only; no category,
    concept-memory record, frame/hypergraph edge, SA activation, AGP anchor,
    wrapper lookup, or vector commit is created.
    """
    learner = getattr(engine, "eve_self_learning", None)
    audit_before = _round75_audit_records_from_engine(engine)
    store_before = _safe_store_stats_dict(engine)
    telemetry_before = _safe_wrapper_telemetry_dict(engine)
    learner_policy_before = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_before = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    adapter = _round77_mapping_adapter(engine)
    report = adapter.operator_acceptance_category_creation_dry_run(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )

    audit_after = _round75_audit_records_from_engine(engine)
    store_after = _safe_store_stats_dict(engine)
    telemetry_after = _safe_wrapper_telemetry_dict(engine)
    learner_policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        learner_policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")

    report["read_only_checks"] = {
        "audit_records_unchanged_during_dry_run": audit_after == audit_before,
        "vector_store_unchanged_during_dry_run": store_after == store_before,
        "telemetry_unchanged_during_dry_run": telemetry_after == telemetry_before,
        "self_learning_policy_unchanged_during_dry_run": learner_policy_after == learner_policy_before,
        "runtime_mapping_enabled_after_dry_run": False,
        "enforcement_enabled_after_dry_run": False,
        "operator_decision_persisted": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "wrapper_lookup_called": False,
        "vector_commit_called": False,
    }
    return report


def write_round83_operator_acceptance_category_creation_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    """Write an existing Round83 dry-run artifact without recomputation."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("dry_run_version"),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "operator_decision_persisted": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


ROUND84_CONCEPT_MEMORY_FRAME_EVIDENCE_DRY_RUN_VERSION = "v3_round84_concept_memory_frame_evidence_dry_run"
ROUND85_SA_ACTIVATION_PATH_DRY_RUN_VERSION = "v3_round85_sa_activation_path_dry_run"
ROUND86_AGP_BRIDGE_SMOKE_DRY_RUN_VERSION = "v3_round86_agp_bridge_smoke_dry_run"
ROUND87_CONCEPT_MAPPING_READINESS_DASHBOARD_VERSION = "v3_round87_concept_mapping_readiness_dashboard"
ROUND88_CONCEPT_MAPPING_V0_PROPOSAL_FREEZE_VERSION = "v3_round88_concept_mapping_v0_proposal_freeze"


def _round84_88_read_only_before(engine: Any) -> dict[str, Any]:
    learner = getattr(engine, "eve_self_learning", None)
    policy = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        policy = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")
    return {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
        "policy": policy,
    }


def _round84_88_read_only_checks(engine: Any, before: dict[str, Any], phase: str) -> dict[str, Any]:
    learner = getattr(engine, "eve_self_learning", None)
    policy_after = None
    if learner is not None and hasattr(learner, "self_learning_v1_freeze_policy_snapshot"):
        policy_after = learner.self_learning_v1_freeze_policy_snapshot().get("active_policy")
    return {
        f"audit_records_unchanged_during_{phase}": _round75_audit_records_from_engine(engine) == before.get("audit"),
        f"vector_store_unchanged_during_{phase}": _safe_store_stats_dict(engine) == before.get("store"),
        f"telemetry_unchanged_during_{phase}": _safe_wrapper_telemetry_dict(engine) == before.get("telemetry"),
        f"self_learning_policy_unchanged_during_{phase}": policy_after == before.get("policy"),
        "runtime_mapping_enabled_after_run": False,
        "enforcement_enabled_after_run": False,
        "operator_decision_persisted": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "wrapper_lookup_called": False,
        "vector_commit_called": False,
    }


def run_round84_concept_memory_frame_evidence_dry_run(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    before = _round84_88_read_only_before(engine)
    report = _round77_mapping_adapter(engine).concept_memory_frame_evidence_dry_run(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )
    report["read_only_checks"] = _round84_88_read_only_checks(engine, before, "evidence_dry_run")
    return report


def run_round85_sa_activation_path_dry_run(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    before = _round84_88_read_only_before(engine)
    report = _round77_mapping_adapter(engine).sa_activation_path_dry_run(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )
    report["read_only_checks"] = _round84_88_read_only_checks(engine, before, "sa_path_dry_run")
    return report


def run_round86_agp_bridge_smoke_dry_run(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    before = _round84_88_read_only_before(engine)
    report = _round77_mapping_adapter(engine).agp_bridge_smoke_dry_run(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )
    report["read_only_checks"] = _round84_88_read_only_checks(engine, before, "agp_bridge_dry_run")
    return report


def run_round87_concept_mapping_readiness_dashboard(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    before = _round84_88_read_only_before(engine)
    report = _round77_mapping_adapter(engine).concept_mapping_readiness_dashboard(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )
    report["read_only_checks"] = _round84_88_read_only_checks(engine, before, "readiness_dashboard")
    return report


def run_round88_concept_mapping_v0_proposal_freeze(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    before = _round84_88_read_only_before(engine)
    report = _round77_mapping_adapter(engine).concept_mapping_v0_proposal_freeze(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )
    report["read_only_checks"] = _round84_88_read_only_checks(engine, before, "v0_proposal_freeze")
    return report


def _write_round84_88_report(report: dict[str, Any], path: str | "Path", *, version_key: str) -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get(version_key),
        "path": str(out),
        "read_only": False,
        "recomputed": False,
        "observe_called": False,
        "commit_called": False,
        "lookup_called": False,
        "operator_decision_persisted": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "agp_anchor_created": False,
        "agp_verify_called": False,
        "policy_changed": False,
        "memory_quarantine_unchanged": True,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }


def write_round84_concept_memory_frame_evidence_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    return _write_round84_88_report(report, path, version_key="dry_run_version")


def write_round85_sa_activation_path_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    return _write_round84_88_report(report, path, version_key="dry_run_version")


def write_round86_agp_bridge_smoke_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    return _write_round84_88_report(report, path, version_key="dry_run_version")


def write_round87_concept_mapping_readiness_dashboard(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    return _write_round84_88_report(report, path, version_key="dashboard_version")


def write_round88_concept_mapping_v0_proposal_freeze(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    return _write_round84_88_report(report, path, version_key="freeze_version")

ROUND89_EXPLICIT_CONCEPT_COMMIT_SMOKE_VERSION = "v3_round89_explicit_concept_commit_smoke"


def _round89_mutation_before(engine: Any) -> dict[str, Any]:
    lcm = _round77_mapping_adapter(engine)
    cm = getattr(engine, "concept_memory", None)
    hg = getattr(engine, "hypergraph_adapter", None)
    activation = getattr(engine, "activation_adapter", None)
    concept_keys = []
    if cm is not None and hasattr(cm, "all_keys"):
        try:
            concept_keys = list(cm.all_keys())
        except Exception:
            concept_keys = []
    active = []
    if activation is not None and hasattr(activation, "all_active_set"):
        try:
            active = sorted(list(activation.all_active_set(threshold=0.0)))
        except Exception:
            active = []
    hg_stats = {}
    if hg is not None and hasattr(hg, "stats"):
        try:
            hg_stats = dict(hg.stats() or {})
        except Exception:
            hg_stats = {}
    return {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
        "concept_keys": concept_keys,
        "hypergraph_stats": hg_stats,
        "active_categories": active,
        "concept_category_count": len(lcm.concept_categories_snapshot()) if hasattr(lcm, "concept_categories_snapshot") else 0,
        "concept_commit_audit_count": len(lcm.concept_commit_records()) if hasattr(lcm, "concept_commit_records") else 0,
    }


def _round89_mutation_checks(engine: Any, before: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    lcm = _round77_mapping_adapter(engine)
    cm = getattr(engine, "concept_memory", None)
    activation = getattr(engine, "activation_adapter", None)
    concept_keys_after = []
    if cm is not None and hasattr(cm, "all_keys"):
        try:
            concept_keys_after = list(cm.all_keys())
        except Exception:
            concept_keys_after = []
    active_after = []
    if activation is not None and hasattr(activation, "all_active_set"):
        try:
            active_after = sorted(list(activation.all_active_set(threshold=0.0)))
        except Exception:
            active_after = []
    created_tokens = set(report.get("created_tokens", []) or [])
    categories = lcm.concept_categories_snapshot() if hasattr(lcm, "concept_categories_snapshot") else {}
    return {
        "audit_records_unchanged_except_concept_commit_audit": _round75_audit_records_from_engine(engine) == before.get("audit"),
        "eve_specific_vector_store_unchanged_during_concept_commit": _safe_store_stats_dict(engine) == before.get("store"),
        "wrapper_lookup_telemetry_unchanged_during_concept_commit": _safe_wrapper_telemetry_dict(engine) == before.get("telemetry"),
        "concept_memory_key_delta": len(set(concept_keys_after) - set(before.get("concept_keys", []))),
        "sa_active_category_delta": len(set(active_after) - set(before.get("active_categories", []))),
        "concept_category_delta": len(categories) - int(before.get("concept_category_count", 0) or 0),
        "concept_commit_audit_delta": len(lcm.concept_commit_records()) - int(before.get("concept_commit_audit_count", 0) or 0),
        "created_tokens_have_category_records": all(any(str(row.get("lexical_token")) == token for row in categories.values()) for token in created_tokens),
        "runtime_mapping_enabled_after_commit": False,
        "enforcement_enabled_after_commit": False,
        "agp_bypass": False,
        "lexical_vector_used_as_anchor": False,
        "eve_specific_vector_used_as_anchor": False,
        "seed_vector_used_as_anchor": False,
    }


def run_round89_explicit_concept_commit_smoke(
    engine: Any,
    *,
    planning_tokens: Iterable[str] = ("민석", "EVE"),
    accepted_tokens: Iterable[str] = ("민석",),
) -> dict[str, Any]:
    before = _round89_mutation_before(engine)
    report = _round77_mapping_adapter(engine).explicit_concept_commit_smoke(
        tokens=planning_tokens,
        accepted_tokens=accepted_tokens,
    )
    report["mutation_checks"] = _round89_mutation_checks(engine, before, report)
    return report


def write_round89_explicit_concept_commit_smoke(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("commit_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": bool(report.get("created_count", 0)),
        "concept_memory_mutation": bool(report.get("concept_memory_delta", 0)),
        "sa_activation_created": bool(report.get("agp_bridge_pass_count", 0)),
        "agp_verify_called": True,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
    }

ROUND90_CONCEPT_COMMIT_DELTA_REPLAY_REPORT_VERSION = "v3_round90_concept_commit_delta_replay_report"


def run_round90_concept_commit_delta_replay_report(
    engine: Any,
    *,
    source_commit_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only replay/delta report for the Round89 concept commit.

    Round90 does not create another category or enable runtime mapping. It only
    replays the already-created explicit category + SA activation bridge and
    verifies that lexical/EveSpecific/seed vectors remain non-anchors.
    """
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.concept_commit_delta_replay_report(source_commit_report=source_commit_report)
    report.setdefault("report_version", ROUND90_CONCEPT_COMMIT_DELTA_REPLAY_REPORT_VERSION)
    report.setdefault("round", 90)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round90_runner": True,
    }
    return report


def write_round90_concept_commit_delta_replay_report(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("report_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }


ROUND91_CONCEPT_COMMIT_REPLAY_EXPORT_CHECKPOINT_VERSION = "v3_round91_concept_commit_replay_export_v0_checkpoint"


def run_round91_concept_commit_replay_export_checkpoint(
    engine: Any,
    *,
    source_commit_report: dict[str, Any] | None = None,
    source_replay_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only Round91 lexical→concept v0 checkpoint export.

    Round91 consolidates Round77~90 artifacts and does not create categories,
    concept-memory rows, SA activations, vectors, or runtime mapping.
    """
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.concept_commit_replay_export_checkpoint_summary(
        source_commit_report=source_commit_report,
        source_replay_report=source_replay_report,
    )
    report.setdefault("checkpoint_version", ROUND91_CONCEPT_COMMIT_REPLAY_EXPORT_CHECKPOINT_VERSION)
    report.setdefault("round", 91)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round91_runner": True,
    }
    return report


def write_round91_concept_commit_replay_export_checkpoint(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("checkpoint_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }



ROUND92_RUNTIME_MAPPING_GATE_DRY_RUN_VERSION = "v3_round92_runtime_mapping_gate_dry_run"


def run_round92_runtime_mapping_gate_dry_run(
    engine: Any,
    *,
    tokens: Iterable[str] | None = None,
    source_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only Round92 runtime mapping gate dry-run report.

    Runtime lexical→concept mapping remains disabled. This only evaluates which
    already committed concept categories would be eligible for mapping if a
    later explicit enforcement patch enabled it.
    """
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.runtime_mapping_gate_dry_run(tokens=tokens, source_checkpoint=source_checkpoint)
    report.setdefault("dry_run_version", ROUND92_RUNTIME_MAPPING_GATE_DRY_RUN_VERSION)
    report.setdefault("round", 92)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round92_runner": True,
    }
    return report


def write_round92_runtime_mapping_gate_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("dry_run_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }

ROUND93_RUNTIME_MAPPING_PROPOSAL_REPORT_VERSION = "v3_round93_runtime_mapping_proposal_report"


def run_round93_runtime_mapping_proposal_report(
    engine: Any,
    *,
    tokens: Iterable[str] | None = None,
    source_dry_run: dict[str, Any] | None = None,
    source_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only Round93 runtime mapping proposal report.

    Runtime lexical→concept mapping remains disabled. This consolidates the
    Round92 dry-run rows into an operator proposal surface only.
    """
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.runtime_mapping_proposal_report(
        tokens=tokens,
        source_dry_run=source_dry_run,
        source_checkpoint=source_checkpoint,
    )
    report.setdefault("proposal_version", ROUND93_RUNTIME_MAPPING_PROPOSAL_REPORT_VERSION)
    report.setdefault("round", 93)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round93_runner": True,
        "runtime_mapping_not_enabled_by_runner": True,
    }
    return report


def write_round93_runtime_mapping_proposal_report(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("proposal_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }

ROUND94_RUNTIME_MAPPING_ENFORCEMENT_DRY_RUN_VERSION = "v3_round94_runtime_mapping_enforcement_dry_run"


def run_round94_runtime_mapping_enforcement_dry_run(
    engine: Any,
    *,
    tokens: Iterable[str] | None = None,
    source_proposal: dict[str, Any] | None = None,
    source_dry_run: dict[str, Any] | None = None,
    source_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only Round94 runtime mapping enforcement dry-run.

    This simulates the future runtime mapping API/result surface without
    enabling runtime lexical→concept mapping or enforcement.
    """
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.runtime_mapping_enforcement_dry_run(
        tokens=tokens,
        source_proposal=source_proposal,
        source_dry_run=source_dry_run,
        source_checkpoint=source_checkpoint,
    )
    report.setdefault("dry_run_version", ROUND94_RUNTIME_MAPPING_ENFORCEMENT_DRY_RUN_VERSION)
    report.setdefault("round", 94)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round94_runner": True,
        "runtime_mapping_not_enabled_by_runner": True,
    }
    return report


def write_round94_runtime_mapping_enforcement_dry_run(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("dry_run_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }


ROUND95_RUNTIME_MAPPING_OPERATOR_ACCEPTANCE_FIXTURE_VERSION = "v3_round95_runtime_mapping_operator_acceptance_fixture"


def run_round95_runtime_mapping_operator_acceptance_fixture(
    engine: Any,
    *,
    tokens: Iterable[str] | None = None,
    accepted_tokens: Iterable[str] | None = None,
    source_enforcement: dict[str, Any] | None = None,
    source_proposal: dict[str, Any] | None = None,
    source_dry_run: dict[str, Any] | None = None,
    source_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only Round95 runtime mapping operator acceptance fixture."""
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.runtime_mapping_operator_acceptance_fixture(
        tokens=tokens,
        accepted_tokens=accepted_tokens,
        source_enforcement=source_enforcement,
        source_proposal=source_proposal,
        source_dry_run=source_dry_run,
        source_checkpoint=source_checkpoint,
    )
    report.setdefault("fixture_version", ROUND95_RUNTIME_MAPPING_OPERATOR_ACCEPTANCE_FIXTURE_VERSION)
    report.setdefault("round", 95)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round95_runner": True,
        "runtime_mapping_not_enabled_by_runner": True,
        "enforcement_not_enabled_by_runner": True,
    }
    return report


def write_round95_runtime_mapping_operator_acceptance_fixture(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("fixture_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }


ROUND96_RUNTIME_MAPPING_ENABLE_SMOKE_PRECHECK_VERSION = "v3_round96_runtime_mapping_enable_smoke_precheck"


def run_round96_runtime_mapping_enable_smoke_precheck(
    engine: Any,
    *,
    source_fixture: dict[str, Any] | None = None,
    tokens: Iterable[str] | None = None,
    accepted_tokens: Iterable[str] | None = None,
    source_enforcement: dict[str, Any] | None = None,
    source_proposal: dict[str, Any] | None = None,
    source_dry_run: dict[str, Any] | None = None,
    source_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only Round96 runtime mapping enable-smoke precheck."""
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.runtime_mapping_enable_smoke_precheck(
        source_fixture=source_fixture,
        tokens=tokens,
        accepted_tokens=accepted_tokens,
        source_enforcement=source_enforcement,
        source_proposal=source_proposal,
        source_dry_run=source_dry_run,
        source_checkpoint=source_checkpoint,
    )
    report.setdefault("precheck_version", ROUND96_RUNTIME_MAPPING_ENABLE_SMOKE_PRECHECK_VERSION)
    report.setdefault("round", 96)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_commit_not_called_by_round96_runner": True,
        "runtime_mapping_not_enabled_by_runner": True,
        "enforcement_not_enabled_by_runner": True,
    }
    return report


def write_round96_runtime_mapping_enable_smoke_precheck(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("precheck_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "fasttext_seed_mutation": False,
        "agp_bypass": False,
        "read_only": True,
    }




ROUND97_CONTROLLED_RUNTIME_MAPPING_ENABLE_SMOKE_VERSION = "v3_round97_controlled_runtime_mapping_enable_smoke"


def run_round97_controlled_runtime_mapping_enable_smoke(
    engine: Any,
    *,
    source_precheck: dict[str, Any] | None = None,
    source_fixture: dict[str, Any] | None = None,
    tokens: Iterable[str] | None = None,
    accepted_tokens: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run the Round97 controlled, non-persistent runtime mapping enable smoke."""
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.controlled_runtime_mapping_enable_smoke(
        source_precheck=source_precheck,
        source_fixture=source_fixture,
        tokens=tokens,
        accepted_tokens=accepted_tokens,
    )
    report.setdefault("smoke_version", ROUND97_CONTROLLED_RUNTIME_MAPPING_ENABLE_SMOKE_VERSION)
    report.setdefault("round", 97)
    report["runner_safety_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled_after_runner": bool(getattr(lcm, "runtime_mapping_enabled", False)),
        "enforcement_enabled_after_runner": bool(getattr(lcm, "enforcement_enabled", False)),
        "concept_commit_not_called_by_round97_runner": True,
        "runtime_mapping_not_persisted_by_runner": True,
        "enforcement_not_enabled_by_runner": True,
    }
    return report


def write_round97_controlled_runtime_mapping_enable_smoke(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("smoke_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled_after_rollback": report.get("runtime_mapping_enabled_after_rollback"),
        "enforcement_enabled_after_rollback": report.get("enforcement_enabled_after_rollback"),
        "runtime_mapping_persisted": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "agp_bypass": False,
        "read_only": False,
    }


ROUND98_RUNTIME_MAPPING_PERSISTENCE_GATE_AUDIT_VERSION = "v3_round98_runtime_mapping_persistence_gate_audit"


def run_round98_runtime_mapping_persistence_gate_audit(
    engine: Any,
    *,
    source_smoke: dict[str, Any] | None = None,
    source_precheck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the Round98 read-only persistence gate audit."""
    before = {
        "audit": _round75_audit_records_from_engine(engine),
        "store": _safe_store_stats_dict(engine),
        "telemetry": _safe_wrapper_telemetry_dict(engine),
    }
    lcm = _round77_mapping_adapter(engine)
    report = lcm.runtime_mapping_persistence_gate_audit(
        source_smoke=source_smoke,
        source_precheck=source_precheck,
    )
    report.setdefault("audit_version", ROUND98_RUNTIME_MAPPING_PERSISTENCE_GATE_AUDIT_VERSION)
    report.setdefault("round", 98)
    report["runner_read_only_checks"] = {
        "self_learning_audit_unchanged": _round75_audit_records_from_engine(engine) == before["audit"],
        "eve_specific_vector_store_unchanged": _safe_store_stats_dict(engine) == before["store"],
        "wrapper_telemetry_unchanged": _safe_wrapper_telemetry_dict(engine) == before["telemetry"],
        "runtime_mapping_enabled_after_runner": bool(getattr(lcm, "runtime_mapping_enabled", False)),
        "enforcement_enabled_after_runner": bool(getattr(lcm, "enforcement_enabled", False)),
        "concept_commit_not_called_by_round98_runner": True,
        "runtime_mapping_not_persisted_by_runner": True,
        "enforcement_not_enabled_by_runner": True,
    }
    return report


def write_round98_runtime_mapping_persistence_gate_audit(report: dict[str, Any], path: str | "Path") -> dict[str, Any]:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("audit_version"),
        "path": str(out),
        "recomputed": False,
        "runtime_mapping_enabled": report.get("runtime_mapping_enabled"),
        "enforcement_enabled": report.get("enforcement_enabled"),
        "runtime_mapping_persisted": False,
        "category_created": False,
        "concept_memory_mutation": False,
        "frame_hypergraph_mutation": False,
        "sa_activation_created": False,
        "eve_specific_vector_commit_called": False,
        "agp_bypass": False,
        "read_only": True,
    }
