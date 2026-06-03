"""Round44 first-pass smoke data analyzer.

This module consumes the round43 runtime smoke result and returns structured,
read-only analysis data. It deliberately does not tune thresholds, change
routing, promote subsets, remove fallback, or mutate engine/runtime state.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping

ROUND44_ANALYSIS_VERSION = "v3_round44_first_pass_smoke_data_analysis"
ROUND44_ANALYSIS_ROUND = 44
ROUND45_ANALYSIS_VERSION = "v3_round45_coverage_and_agp_root_cause"
ROUND45_ANALYSIS_ROUND = 45
ROUND44_BASELINE_ROUND = 43
COVERAGE_LOW_THRESHOLD = 0.50
COVERAGE_GOOD_THRESHOLD = 0.70
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")


def _fixture_rows(fixtures: Iterable[Any] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, item in enumerate(fixtures or []):
        if isinstance(item, Mapping):
            text = str(item.get("text", "") or "").strip()
            category = str(item.get("category", "uncategorized") or "uncategorized")
        else:
            text = str(item or "").strip()
            category = "uncategorized"
        if text:
            rows.append({"index": str(index), "category": category, "text": text})
    return rows


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or ""))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _word_from_oov(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("word", "") or "")
    return str(row or "")


def _classify_oov_word(word: str) -> str:
    w = str(word or "")
    lower = w.lower()
    if not w:
        return "empty"
    if lower in {"eve", "e.v.e"}:
        return "project_identity_term"
    if any(marker in w for marker in ("군대", "생활")):
        return "minsok_military_context"
    if any(marker in lower for marker in ("코딩", "code", "coding", "프로젝트", "project")):
        return "project_or_coding_context"
    if any(marker in w for marker in ("어때", "그래", "뭐야", "생각", "왜")):
        return "conversational_question_expression"
    if any(marker in w for marker in ("좋아", "슬퍼", "기뻐", "힘들")):
        return "emotion_expression"
    if re.fullmatch(r"[A-Za-z0-9_]+", w):
        return "latin_or_symbolic_token"
    if re.fullmatch(r"[가-힣]+", w):
        return "korean_general_oov"
    return "mixed_or_other"


def _coverage_status(primary_hit_rate: float) -> str:
    if primary_hit_rate < COVERAGE_LOW_THRESHOLD:
        return "low_coverage"
    if primary_hit_rate < COVERAGE_GOOD_THRESHOLD:
        return "partial_coverage"
    return "good_coverage"


def _telemetry_summary(smoke_result: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = smoke_result.get("wrapper_telemetry", {}) if isinstance(smoke_result, Mapping) else {}
    if not isinstance(telemetry, Mapping):
        telemetry = {}
    total = _safe_int(telemetry.get("total_calls"))
    primary = _safe_int(telemetry.get("primary_hits"))
    fallback = _safe_int(telemetry.get("fallback_uses"))
    errors = _safe_int(telemetry.get("errors"))
    primary_rate = _safe_float(telemetry.get("primary_hit_rate"))
    fallback_rate = _safe_float(telemetry.get("fallback_rate"))
    error_rate = _safe_float(telemetry.get("error_rate"))
    return {
        "total_calls": total,
        "primary_hits": primary,
        "fallback_uses": fallback,
        "errors": errors,
        "primary_hit_rate": primary_rate,
        "fallback_rate": fallback_rate,
        "error_rate": error_rate,
        "coverage_status": _coverage_status(primary_rate),
        "small_5k_coverage_interpretation": (
            "small_5k_under_covers_runtime_fixtures"
            if primary_rate < COVERAGE_LOW_THRESHOLD
            else "small_5k_partial_or_better_runtime_coverage"
        ),
        "no_runtime_error_signal": errors == 0 and error_rate == 0.0,
        "read_only": True,
    }


def _oov_analysis(smoke_result: Mapping[str, Any], fixture_rows: list[dict[str, str]]) -> dict[str, Any]:
    raw_oov = smoke_result.get("oov_samples_from_run", []) if isinstance(smoke_result, Mapping) else []
    if not isinstance(raw_oov, list):
        raw_oov = []
    words = [_word_from_oov(row) for row in raw_oov]
    words = [word for word in words if word]
    groups = Counter(_classify_oov_word(word) for word in words)

    category_tokens: dict[str, set[str]] = defaultdict(set)
    for row in fixture_rows:
        category = row["category"]
        for token in _tokens(row["text"]):
            category_tokens[category].add(token)

    by_category: dict[str, dict[str, Any]] = {}
    word_set = set(words)
    for category in sorted(category_tokens):
        tokens = sorted(category_tokens[category])
        matching = sorted(token for token in tokens if token in word_set)
        by_category[category] = {
            "fixture_token_count": len(tokens),
            "observed_oov_tokens": matching,
            "observed_oov_count": len(matching),
        }

    return {
        "sample_size": len(words),
        "unique_oov_count": len(set(words)),
        "unique_oov_words": sorted(set(words)),
        "group_counts": dict(sorted(groups.items())),
        "by_fixture_category": by_category,
        "top_oov_words": [word for word, _ in Counter(words).most_common(10)],
        "interpretation": "oov_sample_is_bounded_recent_sample_not_full_vocab_audit",
        "read_only": True,
    }


def _fixture_category_summary(fixture_rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row["category"] for row in fixture_rows)
    token_counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    for row in fixture_rows:
        category = row["category"]
        token_counts[category] += len(_tokens(row["text"]))
        if len(examples[category]) < 3:
            examples[category].append(row["text"])
    return {
        "fixture_count": len(fixture_rows),
        "category_counts": dict(sorted(counts.items())),
        "token_counts_by_category": dict(sorted(token_counts.items())),
        "examples_by_category": dict(sorted(examples.items())),
        "read_only": True,
    }


def _agp_analysis(smoke_result: Mapping[str, Any], telemetry: Mapping[str, Any]) -> dict[str, Any]:
    agp = smoke_result.get("agp_trace_summary", {}) if isinstance(smoke_result, Mapping) else {}
    if not isinstance(agp, Mapping):
        agp = {}
    by_reason = agp.get("by_reason", {}) if isinstance(agp.get("by_reason", {}), Mapping) else {}
    by_layer = agp.get("by_layer", {}) if isinstance(agp.get("by_layer", {}), Mapping) else {}
    pass_rate = _safe_float(agp.get("pass_rate"))
    unknown = _safe_int(by_reason.get("unknown_category"))
    total_reason = sum(_safe_int(v) for v in by_reason.values())
    if total_reason == 0:
        for layer in by_layer.values():
            if isinstance(layer, Mapping):
                total_reason += _safe_int(layer.get("total"))
    all_unknown = bool(total_reason and unknown == total_reason)
    return {
        "pass_rate": pass_rate,
        "reason_counts": dict(by_reason),
        "by_layer": deepcopy(dict(by_layer)),
        "unknown_category_count": unknown,
        "total_traces": total_reason,
        "all_failures_unknown_category": all_unknown,
        "first_pass_interpretation": (
            "agp_anchor_coverage_gap_not_fasttext_runtime_error"
            if all_unknown and telemetry.get("no_runtime_error_signal")
            else "needs_more_trace_samples"
        ),
        "no_threshold_change": True,
        "read_only": True,
    }


def analyze_smoke_data(
    smoke_result: Mapping[str, Any],
    fixture_rows: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return a first-pass, data-only analysis of round43 smoke results."""
    rows = _fixture_rows(fixture_rows)
    if not rows:
        rows = [
            {"index": str(i), "category": "uncategorized", "text": str(text)}
            for i, text in enumerate(smoke_result.get("fixtures", []) if isinstance(smoke_result, Mapping) else [])
        ]
    telemetry = _telemetry_summary(smoke_result)
    oov = _oov_analysis(smoke_result, rows)
    agp = _agp_analysis(smoke_result, telemetry)
    category_summary = _fixture_category_summary(rows)
    data_quality = {
        "baseline_round": _safe_int(smoke_result.get("baseline_round", 0)) if isinstance(smoke_result, Mapping) else 0,
        "analysis_round": ROUND44_ANALYSIS_ROUND,
        "baseline_matches_round43": _safe_int(smoke_result.get("baseline_round", 0)) == ROUND44_BASELINE_ROUND if isinstance(smoke_result, Mapping) else False,
        "fixtures_count": len(rows),
        "has_wrapper_telemetry": bool(smoke_result.get("wrapper_telemetry")) if isinstance(smoke_result, Mapping) else False,
        "has_agp_summary": bool(smoke_result.get("agp_trace_summary")) if isinstance(smoke_result, Mapping) else False,
    }
    return {
        "analysis_version": ROUND44_ANALYSIS_VERSION,
        "analysis_round": ROUND44_ANALYSIS_ROUND,
        "baseline_round": data_quality["baseline_round"],
        "data_only": True,
        "decision_policy_changed": False,
        "thresholds_changed": False,
        "subset_promoted": False,
        "fallback_removed": False,
        "telemetry_summary": telemetry,
        "fixture_category_summary": category_summary,
        "oov_analysis": oov,
        "agp_unknown_category_analysis": agp,
        "data_quality": data_quality,
        "manual_review_flags": _manual_review_flags(telemetry, agp),
        "next_round_candidates": _next_round_candidates(telemetry, agp),
        "read_only": True,
    }


def _manual_review_flags(telemetry: Mapping[str, Any], agp: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if _safe_float(telemetry.get("primary_hit_rate")) < COVERAGE_LOW_THRESHOLD:
        flags.append("primary_hit_rate_below_0_50")
    if _safe_float(telemetry.get("fallback_rate")) > 0.50:
        flags.append("fallback_rate_above_0_50")
    if _safe_int(agp.get("unknown_category_count")) > 0:
        flags.append("agp_unknown_category_present")
    if bool(agp.get("all_failures_unknown_category")):
        flags.append("agp_all_failures_unknown_category")
    return flags


def _next_round_candidates(telemetry: Mapping[str, Any], agp: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "round": 45,
            "candidate": "category_coverage_breakdown",
            "reason": "separate lexical coverage from AGP anchor coverage before tuning",
            "auto_apply": False,
        }
    ]
    if _safe_float(telemetry.get("primary_hit_rate")) < COVERAGE_LOW_THRESHOLD:
        candidates.append(
            {
                "round": "45+",
                "candidate": "medium_30k_subset_evaluation_only",
                "reason": "small_5k runtime fixture coverage below 0.50",
                "auto_apply": False,
            }
        )
    if _safe_int(agp.get("unknown_category_count")) > 0:
        candidates.append(
            {
                "round": "45+",
                "candidate": "agp_unknown_category_root_cause_analysis",
                "reason": "post_swap smoke produced unknown_category failures",
                "auto_apply": False,
            }
        )
    return candidates


# --- v3 round45: category coverage + AGP root-cause analysis ---

REQUIRED_COVERAGE_CATEGORIES = ("greeting", "daily", "emotion", "identity", "reasoning")
OPTIONAL_COVERAGE_CATEGORIES = ("minsok",)
AGP_ROOT_CAUSE_HYPOTHESES = (
    "H1: AGP response category extraction returns no usable candidate categories",
    "H2: SA activation is empty or too weak during fixture response time",
    "H3: candidate categories and SA active categories have zero overlap",
    "H4: category threshold is too strict for weak but relevant activation",
)
AGP_ROOT_CAUSE_OPTIONS = (
    "A: inspect AGP category extraction path with code-level trace",
    "B: inspect SA activation propagation during fixture response",
    "C: run one threshold relaxation experiment only after manual approval",
    "D: verify fixture categories against EVE internal category vocabulary",
)


def _empty_category_bucket(category: str) -> dict[str, Any]:
    return {
        "category": category,
        "fixtures": [],
        "fixture_count": 0,
        "tokens": [],
        "token_count": 0,
        "observed_oov_words": [],
        "observed_oov_count": 0,
        "primary_hit_rate_proxy": 1.0,
        "fallback_rate_proxy": 0.0,
        "coverage_status": "good_coverage",
    }


def analyze_category_coverage(
    smoke_result: Mapping[str, Any],
    fixture_rows: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Break down round43 fixture lexical coverage by fixture category.

    This is a read-only, sample-based analyzer. It uses the bounded OOV sample
    from wrapper telemetry, so category rates are proxies rather than a full
    corpus audit.
    """
    rows = _fixture_rows(fixture_rows)
    if not rows:
        rows = [
            {"index": str(i), "category": "uncategorized", "text": str(text)}
            for i, text in enumerate(smoke_result.get("fixtures", []) if isinstance(smoke_result, Mapping) else [])
        ]
    raw_oov = smoke_result.get("oov_samples_from_run", []) if isinstance(smoke_result, Mapping) else []
    if not isinstance(raw_oov, list):
        raw_oov = []
    oov_words = {_word_from_oov(row) for row in raw_oov if _word_from_oov(row)}
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = row["category"]
        bucket = grouped.setdefault(category, _empty_category_bucket(category))
        text = row["text"]
        bucket["fixtures"].append(text)
        bucket["fixture_count"] += 1
        for token in _tokens(text):
            bucket["tokens"].append(token)
    for category, bucket in grouped.items():
        tokens = bucket["tokens"]
        observed = [token for token in tokens if token in oov_words]
        token_count = len(tokens)
        oov_count = len(observed)
        fallback_proxy = (oov_count / token_count) if token_count else 0.0
        primary_proxy = 1.0 - fallback_proxy
        bucket.update(
            {
                "tokens": sorted(set(tokens)),
                "token_count": token_count,
                "observed_oov_words": sorted(set(observed)),
                "observed_oov_count": oov_count,
                "primary_hit_rate_proxy": primary_proxy,
                "fallback_rate_proxy": fallback_proxy,
                "coverage_status": _coverage_status(primary_proxy),
            }
        )
    # Ensure the five required categories are always visible in the result.
    for category in REQUIRED_COVERAGE_CATEGORIES:
        grouped.setdefault(category, _empty_category_bucket(category))
    ranked = sorted(
        grouped.values(),
        key=lambda item: (float(item.get("primary_hit_rate_proxy", 0.0)), item.get("category", "")),
    )
    worst = ranked[0]["category"] if ranked else None
    best = ranked[-1]["category"] if ranked else None
    medium_likely = []
    medium_unlikely = []
    for category, bucket in sorted(grouped.items()):
        words = set(bucket.get("observed_oov_words", []))
        if words & {"EVE", "코딩", "프로젝트", "군대"}:
            medium_unlikely.append(category)
        elif bucket.get("observed_oov_count", 0):
            medium_likely.append(category)
    return {
        "analysis_version": ROUND45_ANALYSIS_VERSION,
        "analysis_round": ROUND45_ANALYSIS_ROUND,
        "baseline_round": _safe_int(smoke_result.get("baseline_round", 43)) if isinstance(smoke_result, Mapping) else 43,
        "by_category": dict(sorted(grouped.items())),
        "required_categories_present": all(category in grouped for category in REQUIRED_COVERAGE_CATEGORIES),
        "best_covered": best,
        "worst_covered": worst,
        "medium_30k_projected_impact": {
            "category_likely_to_improve": sorted(medium_likely),
            "category_unlikely_to_improve": sorted(medium_unlikely),
            "rationale": "Common-Crawl frequency expansion may help general conversational Korean but not project-specific symbols such as EVE or local military/project terms.",
            "auto_extract_medium_30k": False,
        },
        "sample_based_proxy": True,
        "read_only": True,
    }


def _agp_threshold_snapshot(engine: Any | None) -> dict[str, Any]:
    adapter = getattr(engine, "agp_adapter", None) if engine is not None else None
    return {
        "category_threshold": getattr(adapter, "category_threshold", 0.3),
        "hormone_threshold": getattr(adapter, "hormone_threshold", 0.5),
        "source": "engine.agp_adapter" if adapter is not None else "documented_default",
    }


def _sa_snapshot(engine: Any | None) -> list[str]:
    candidates: list[str] = []
    if engine is None:
        return candidates
    for attr in ("activation_adapter", "category_activation", "sa_adapter"):
        obj = getattr(engine, attr, None)
        if obj is None:
            continue
        for field in ("active_categories", "activated_categories", "categories"):
            value = getattr(obj, field, None)
            if isinstance(value, Mapping):
                candidates.extend(str(k) for k, v in value.items() if _safe_float(v) > 0)
            elif isinstance(value, (list, tuple, set)):
                candidates.extend(str(v) for v in value)
    return sorted(set(candidates))[:50]


def _candidate_categories_from_smoke(smoke_result: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    for key in ("responses", "fixture_responses"):
        value = smoke_result.get(key, []) if isinstance(smoke_result, Mapping) else []
        if not isinstance(value, list):
            continue
        for row in value:
            if not isinstance(row, Mapping):
                continue
            for cat_key in ("candidate_categories", "categories", "agp_categories"):
                cats = row.get(cat_key)
                if isinstance(cats, (list, tuple, set)):
                    categories.extend(str(c) for c in cats)
    # The current smoke runner does not extract response categories; keep this
    # explicit rather than inventing categories from surface text.
    return sorted(set(categories))


def analyze_agp_unknown_category_root_cause(
    smoke_result: Mapping[str, Any],
    engine: Any | None = None,
) -> dict[str, Any]:
    """Analyze why post-swap smoke produced AGP unknown_category failures.

    The result is evidence, not a fix. It does not relax thresholds, mutate SA,
    or change AGP behavior.
    """
    telemetry = _telemetry_summary(smoke_result)
    agp = _agp_analysis(smoke_result, telemetry)
    candidate_categories = _candidate_categories_from_smoke(smoke_result)
    sa_categories = _sa_snapshot(engine)
    overlap = sorted(set(candidate_categories) & set(sa_categories))
    evidence = {
        "H1": {
            "supported": len(candidate_categories) == 0,
            "candidate_category_count": len(candidate_categories),
            "note": "smoke result exposes no candidate response categories" if not candidate_categories else "candidate categories present",
        },
        "H2": {
            "supported": len(sa_categories) == 0,
            "sa_active_category_count": len(sa_categories),
            "note": "SA snapshot empty or unavailable" if not sa_categories else "SA snapshot has active categories",
        },
        "H3": {
            "supported": bool(candidate_categories or sa_categories) and len(overlap) == 0,
            "overlap_count": len(overlap),
            "note": "candidate-vs-SA overlap is zero" if len(overlap) == 0 else "candidate-vs-SA overlap exists",
        },
        "H4": {
            "supported": bool(agp.get("all_failures_unknown_category")),
            "thresholds": _agp_threshold_snapshot(engine),
            "note": "threshold could not be ruled out, but no auto relaxation is allowed",
        },
    }
    return {
        "analysis_version": ROUND45_ANALYSIS_VERSION,
        "analysis_round": ROUND45_ANALYSIS_ROUND,
        "baseline_round": _safe_int(smoke_result.get("baseline_round", 43)) if isinstance(smoke_result, Mapping) else 43,
        "agp_thresholds_at_run": _agp_threshold_snapshot(engine),
        "candidate_categories_from_responses": candidate_categories,
        "sa_activated_categories_at_response_time": sa_categories,
        "overlap_candidate_vs_sa": {
            "overlap": overlap,
            "overlap_count": len(overlap),
        },
        "root_cause_hypotheses": list(AGP_ROOT_CAUSE_HYPOTHESES),
        "hypothesis_evidence": evidence,
        "agp_summary": agp,
        "next_round_options": list(AGP_ROOT_CAUSE_OPTIONS),
        "recommendation_priority": "manual_decision",
        "no_threshold_change": True,
        "no_agp_runtime_change": True,
        "read_only": True,
    }


def confirm_problem_separation() -> dict[str, Any]:
    """Separate lexical coverage from AGP anchor failure for manual planning."""
    return {
        "analysis_version": ROUND45_ANALYSIS_VERSION,
        "analysis_round": ROUND45_ANALYSIS_ROUND,
        "problem_1_lexical_coverage": {
            "severity": "medium",
            "blocker": False,
            "reason": "PMI+SVD fallback keeps runtime path available despite low fastText primary coverage",
            "solution_candidate": "medium_30k_evaluation_or_eve_specific_learning_later",
        },
        "problem_2_agp_anchor": {
            "severity": "high",
            "blocker": True,
            "reason": "AGP unknown_category prevents anchored generation from passing",
            "solution_candidate": "AGP root-cause fix after manual decision",
        },
        "priority": "problem_2_first",
        "medium_30k_forbidden_until_agp_blocker_resolved": True,
        "recommendation_priority": "manual_decision",
        "read_only": True,
    }


ROUND46_ANALYSIS_VERSION = "v3_round46_agp_extraction_trace_probe"
ROUND46_ANALYSIS_ROUND = 46


def _strength_from_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "insufficient_data"
    ratio = count / total
    if ratio >= 0.80:
        return "strong"
    if ratio >= 0.50:
        return "moderate"
    return "weak"


def analyze_extraction_traces(traces: Iterable[Any]) -> dict[str, Any]:
    """Return H1~H4 evidence from round46 AGP extraction traces.

    This analyzer is read-only. It reports evidence only and does not apply
    threshold changes, extraction fixes, SA changes, or runtime decisions.
    """
    rows = [row for row in traces or [] if isinstance(row, Mapping)]
    total = len(rows)
    candidate_counts: list[int] = []
    sa_counts: list[int] = []
    overlap_counts: list[int] = []
    weak_overlap_counts: list[int] = []
    thresholds: list[dict[str, Any]] = []
    methods: set[str] = set()
    for row in rows:
        payload = row.get("agp_trace", {})
        trace = payload.get("trace", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(trace, Mapping):
            trace = {}
        candidate_counts.append(_safe_int(trace.get("candidate_count")))
        sa_counts.append(_safe_int(trace.get("sa_count")))
        overlap_counts.append(_safe_int(trace.get("overlap_count")))
        weak_overlap_counts.append(_safe_int(trace.get("weak_overlap_count")))
        threshold_row = trace.get("thresholds_used", {})
        if isinstance(threshold_row, Mapping):
            thresholds.append(dict(threshold_row))
        method = trace.get("extraction_method")
        if method:
            methods.add(str(method))
    zero_candidates = sum(1 for value in candidate_counts if value == 0)
    zero_sa = sum(1 for value in sa_counts if value == 0)
    zero_overlap = sum(1 for value in overlap_counts if value == 0)
    with_candidates = total - zero_candidates
    with_sa = total - zero_sa
    with_overlap = total - zero_overlap
    weak_overlap = sum(1 for value in weak_overlap_counts if value > 0)

    evidence = {
        "H1_evidence": {
            "hypothesis": "candidate extraction returns no categories",
            "fixtures_with_zero_candidates": zero_candidates,
            "fixtures_with_candidates": with_candidates,
            "trace_count": total,
            "strength": _strength_from_ratio(zero_candidates, total),
        },
        "H2_evidence": {
            "hypothesis": "SA activation empty or too weak during fixture response time",
            "fixtures_with_zero_sa_active": zero_sa,
            "fixtures_with_sa_active": with_sa,
            "avg_sa_count": (sum(sa_counts) / total) if total else 0.0,
            "trace_count": total,
            "strength": _strength_from_ratio(zero_sa, total),
        },
        "H3_evidence": {
            "hypothesis": "candidate categories and SA active categories have zero overlap",
            "fixtures_with_zero_overlap": zero_overlap,
            "fixtures_with_overlap": with_overlap,
            "trace_count": total,
            "strength": _strength_from_ratio(zero_overlap, total),
        },
        "H4_evidence": {
            "hypothesis": "category threshold too strict for weak but relevant activation",
            "fixtures_with_weak_overlap": weak_overlap,
            "thresholds_seen": thresholds[:5],
            "trace_count": total,
            "strength": _strength_from_ratio(weak_overlap, total),
        },
    }
    # H3 can be a derivative symptom when H1 extracts zero candidates: zero
    # candidates necessarily means zero overlap. Prefer H1 in that case so the
    # next round fixes the extraction surface before touching thresholds.
    if total > 0 and zero_candidates == total:
        most_likely = "H1"
    else:
        ordered = [
            ("H1", evidence["H1_evidence"]["strength"], zero_candidates),
            ("H2", evidence["H2_evidence"]["strength"], zero_sa),
            ("H3", evidence["H3_evidence"]["strength"], zero_overlap),
            ("H4", evidence["H4_evidence"]["strength"], weak_overlap),
        ]
        strength_rank = {"strong": 3, "moderate": 2, "weak": 1, "insufficient_data": 0}
        ordered.sort(key=lambda item: (strength_rank.get(item[1], 0), item[2], item[0]), reverse=True)
        most_likely = ordered[0][0] if ordered else "insufficient_data"
    return {
        "analysis_version": ROUND46_ANALYSIS_VERSION,
        "analysis_round": ROUND46_ANALYSIS_ROUND,
        "trace_count": total,
        "candidate_count_distribution": candidate_counts,
        "sa_count_distribution": sa_counts,
        "overlap_count_distribution": overlap_counts,
        "weak_overlap_count_distribution": weak_overlap_counts,
        "extraction_methods_seen": sorted(methods),
        **evidence,
        "most_likely_root_cause": most_likely,
        "next_round_options": list(AGP_ROOT_CAUSE_OPTIONS),
        "next_round_recommendation": "manual_decision",
        "no_threshold_change": True,
        "no_agp_runtime_change": True,
        "no_auto_application": True,
        "read_only": True,
    }


# --- v3 round48: post-bridge smoke rerun + residual AGP analysis ---

ROUND48_ANALYSIS_VERSION = "v3_round48_post_bridge_smoke_rerun"
ROUND48_ANALYSIS_ROUND = 48
ROUND48_PRE_BRIDGE_BASELINE = {
    "baseline_round": 43,
    "agp_pass_rate": 0.0,
    "candidate_non_empty_traces": 0,
    "zero_overlap_traces": 14,
    "primary_hit_rate": 0.3542,
    "fallback_rate": 0.6458,
    "veto_fallback_count": 14,
}


def _trace_rows(post_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = post_data.get("traces", []) if isinstance(post_data, Mapping) else []
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _trace_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("agp_trace", {})
    if isinstance(payload, Mapping):
        trace = payload.get("trace", {})
        return trace if isinstance(trace, Mapping) else {}
    return {}


def _trace_result_passed(row: Mapping[str, Any]) -> bool:
    payload = row.get("agp_trace", {})
    result = payload.get("result") if isinstance(payload, Mapping) else None
    return bool(getattr(result, "passed", False))


def _pass_rate_from_traces(rows: list[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _trace_result_passed(row)) / len(rows)


def _source_distribution(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        trace = _trace_payload(row)
        source = str(trace.get("extraction_source", "unknown") or "unknown")
        counts[source] += 1
    return dict(sorted(counts.items()))


def _post_bridge_metrics(post_data: Mapping[str, Any]) -> dict[str, Any]:
    rows = _trace_rows(post_data)
    telemetry = _telemetry_summary(post_data)
    candidate_non_empty = 0
    zero_overlap = 0
    weak_overlap = 0
    low_overlap = 0
    hormone_mismatch = 0
    without_bridge: list[dict[str, Any]] = []
    residual_failures: list[dict[str, Any]] = []
    for row in rows:
        trace = _trace_payload(row)
        candidate_count = _safe_int(trace.get("candidate_count"))
        overlap_count = _safe_int(trace.get("overlap_count"))
        if candidate_count > 0:
            candidate_non_empty += 1
        if overlap_count == 0:
            zero_overlap += 1
        if _safe_int(trace.get("weak_overlap_count")) > 0:
            weak_overlap += 1
        if overlap_count > 0 and candidate_count > overlap_count:
            low_overlap += 1
        if str(trace.get("extraction_source", "")) != "meaning_bridge":
            without_bridge.append({
                "input": row.get("input"),
                "layer": row.get("layer"),
                "extraction_source": trace.get("extraction_source"),
            })
        if not _trace_result_passed(row):
            payload = row.get("agp_trace", {})
            result = payload.get("result") if isinstance(payload, Mapping) else None
            reason = getattr(result, "reason", "unknown")
            if str(reason) == "hormone_mismatch":
                hormone_mismatch += 1
            residual_failures.append({
                "input": row.get("input"),
                "layer": row.get("layer"),
                "reason": str(reason),
                "extraction_source": trace.get("extraction_source"),
            })
    return {
        "trace_count": len(rows),
        "agp_pass_rate": _pass_rate_from_traces(rows),
        "candidate_non_empty_traces": candidate_non_empty,
        "zero_candidate_traces": len(rows) - candidate_non_empty,
        "zero_overlap_traces": zero_overlap,
        "weak_overlap_traces": weak_overlap,
        "low_overlap_warning_traces": low_overlap,
        "hormone_mismatch_traces": hormone_mismatch,
        "extraction_source_distribution": _source_distribution(rows),
        "fixtures_without_bridge": without_bridge,
        "residual_failures": residual_failures,
        "wrapper_telemetry": telemetry,
        "read_only": True,
    }


def compare_pre_post_bridge(
    pre_data: Mapping[str, Any] | None,
    post_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare round43/44 pre-bridge baseline with round48 post-bridge smoke.

    This is a read-only comparison. It does not tune thresholds, expand the
    bridge, change wrapper policy, promote subsets, or apply recommendations.
    """
    pre = dict(ROUND48_PRE_BRIDGE_BASELINE)
    if isinstance(pre_data, Mapping):
        pre.update({k: pre_data[k] for k in pre.keys() if k in pre_data})
        telemetry = pre_data.get("wrapper_telemetry") or pre_data.get("telemetry_summary")
        if isinstance(telemetry, Mapping):
            pre["primary_hit_rate"] = _safe_float(telemetry.get("primary_hit_rate", pre["primary_hit_rate"]))
            pre["fallback_rate"] = _safe_float(telemetry.get("fallback_rate", pre["fallback_rate"]))
        agp = pre_data.get("agp_trace_summary") or pre_data.get("agp_unknown_category_analysis")
        if isinstance(agp, Mapping):
            pre["agp_pass_rate"] = _safe_float(agp.get("pass_rate", pre["agp_pass_rate"]))
    post = _post_bridge_metrics(post_data)
    post_primary = _safe_float(post["wrapper_telemetry"].get("primary_hit_rate"))
    post_fallback = _safe_float(post["wrapper_telemetry"].get("fallback_rate"))
    primary_delta = post_primary - _safe_float(pre.get("primary_hit_rate"))
    fallback_delta = post_fallback - _safe_float(pre.get("fallback_rate"))
    post_veto = len(post["residual_failures"])
    return {
        "analysis_version": ROUND48_ANALYSIS_VERSION,
        "analysis_round": ROUND48_ANALYSIS_ROUND,
        "baseline_rounds": {"pre_bridge": _safe_int(pre.get("baseline_round", 43)), "post_bridge": 48},
        "agp_pass_rate_change": {
            "pre": _safe_float(pre.get("agp_pass_rate")),
            "post": post["agp_pass_rate"],
            "delta": post["agp_pass_rate"] - _safe_float(pre.get("agp_pass_rate")),
        },
        "candidate_category_change": {
            "pre_non_empty_traces": _safe_int(pre.get("candidate_non_empty_traces")),
            "post_non_empty_traces": post["candidate_non_empty_traces"],
            "post_zero_candidate_traces": post["zero_candidate_traces"],
        },
        "overlap_change": {
            "pre_zero_overlap_traces": _safe_int(pre.get("zero_overlap_traces")),
            "post_zero_overlap_traces": post["zero_overlap_traces"],
            "post_weak_overlap_traces": post["weak_overlap_traces"],
        },
        "wrapper_telemetry_comparison": {
            "primary_hit_rate_pre": _safe_float(pre.get("primary_hit_rate")),
            "primary_hit_rate_post": post_primary,
            "primary_hit_rate_delta": primary_delta,
            "fallback_rate_pre": _safe_float(pre.get("fallback_rate")),
            "fallback_rate_post": post_fallback,
            "fallback_rate_delta": fallback_delta,
            "bridge_expected_to_change_lexical_coverage": False,
            "minimal_change_expected": True,
        },
        "response_path_change": {
            "pre_fallback_used_due_to_veto": _safe_int(pre.get("veto_fallback_count")),
            "post_fallback_used_due_to_veto": post_veto,
            "residual_failures": post["residual_failures"],
        },
        "extraction_source_distribution": post["extraction_source_distribution"],
        "residual_fail_fixtures": post["residual_failures"],
        "interpretation_data_dict": "manual_decision",
        "no_threshold_change": True,
        "no_wrapper_change": True,
        "no_subset_change": True,
        "read_only": True,
    }


def identify_residual_issues(post_data: Mapping[str, Any]) -> dict[str, Any]:
    """Identify remaining post-bridge AGP issues without fixing them."""
    post = _post_bridge_metrics(post_data)
    issues: list[str] = []
    if post["fixtures_without_bridge"]:
        issues.append("fixtures_without_bridge")
    if post["low_overlap_warning_traces"]:
        issues.append("low_overlap_warning")
    if post["hormone_mismatch_traces"]:
        issues.append("hormone_mismatch")
    if post["residual_failures"]:
        issues.append("residual_agp_failures")
    if not issues:
        issues.append("no_residual_agp_issue_detected")
    if post["residual_failures"]:
        priority = "manual_decision_residual_agp"
    elif post["fixtures_without_bridge"]:
        priority = "manual_decision_bridge_extension"
    else:
        priority = "manual_decision_stable_then_lexical_coverage"
    return {
        "analysis_version": ROUND48_ANALYSIS_VERSION,
        "analysis_round": ROUND48_ANALYSIS_ROUND,
        "fixtures_without_bridge": post["fixtures_without_bridge"],
        "fixtures_with_low_overlap": [],
        "low_overlap_warning_count": post["low_overlap_warning_traces"],
        "fixtures_with_hormone_mismatch": [
            row for row in post["residual_failures"] if row.get("reason") == "hormone_mismatch"
        ],
        "residual_failures": post["residual_failures"],
        "issue_flags": issues,
        "next_round_options": [
            "A: all fixtures bridged and anchored -> proceed to lexical coverage planning",
            "B: some fixtures lack bridge -> extend bridge to the missing generation path",
            "C: hormone mismatch appears -> run a separate hormone-layer analysis",
        ],
        "recommendation_priority": priority,
        "no_threshold_change": True,
        "no_auto_decision": True,
        "read_only": True,
    }


# --- v3 round49: lexical coverage planning + C strategy decision ---

ROUND49_ANALYSIS_VERSION = "v3_round49_lexical_coverage_planning"
ROUND49_ANALYSIS_ROUND = 49
ROUND49_STRATEGY = "C_hybrid"
ROUND49_STRATEGY_SEQUENCE = "A_first_medium_30k_then_B_continuous_self_learning"
GENERAL_KOREAN_OOV_GROUPS = {
    "conversational_question_expression",
    "emotion_expression",
    "korean_general_oov",
    "minsok_military_context",
    "project_or_coding_context",
}
EVE_SPECIFIC_OOV_GROUPS = {
    "project_identity_term",
    "latin_or_symbolic_token",
}


def _oov_words_from_any(data: Mapping[str, Any] | Iterable[Any] | None) -> list[str]:
    if data is None:
        return []
    if isinstance(data, Mapping):
        for key in ("oov_samples_from_run", "unique_oov_words", "oov_words", "observed_oov_words"):
            value = data.get(key)
            if isinstance(value, list):
                return [_word_from_oov(row) for row in value if _word_from_oov(row)]
        nested = data.get("oov_analysis")
        if isinstance(nested, Mapping):
            return _oov_words_from_any(nested)
        return []
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return [_word_from_oov(row) for row in data if _word_from_oov(row)]
    return []


def classify_oov_by_origin(oov_data: Mapping[str, Any] | Iterable[Any] | None) -> dict[str, Any]:
    """Classify OOV samples as general Korean vs EVE-specific.

    This is a planning-only helper. It does not add words, extract subsets,
    change wrapper thresholds, or promote any lexical resource.
    """
    words = sorted(set(_oov_words_from_any(oov_data)))
    general: list[str] = []
    eve_specific: list[str] = []
    ambiguous: list[str] = []
    by_category: dict[str, list[str]] = defaultdict(list)
    for word in words:
        group = _classify_oov_word(word)
        by_category[group].append(word)
        if group in EVE_SPECIFIC_OOV_GROUPS or word in {"EVE", "민석"}:
            eve_specific.append(word)
        elif group in GENERAL_KOREAN_OOV_GROUPS:
            general.append(word)
        else:
            ambiguous.append(word)
    return {
        "analysis_version": ROUND49_ANALYSIS_VERSION,
        "analysis_round": ROUND49_ANALYSIS_ROUND,
        "general_korean": sorted(general),
        "eve_specific": sorted(eve_specific),
        "ambiguous": sorted(ambiguous),
        "by_category": {k: sorted(v) for k, v in sorted(by_category.items())},
        "read_only": True,
        "no_auto_vocabulary_addition": True,
    }


def project_medium_30k_impact(
    current_data: Mapping[str, Any] | None = None,
    oov_patterns: Mapping[str, Any] | Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Project medium 30k subset impact without extracting or promoting it."""
    data = current_data if isinstance(current_data, Mapping) else {}
    telemetry = _telemetry_summary(data)
    current_primary = _safe_float(telemetry.get("primary_hit_rate")) or _safe_float(data.get("primary_hit_rate", 0.25))
    current_fallback = _safe_float(telemetry.get("fallback_rate")) or _safe_float(data.get("fallback_rate", 0.75))
    classified = classify_oov_by_origin(oov_patterns if oov_patterns is not None else data)
    projected_low = 0.50
    projected_high = 0.65
    return {
        "analysis_version": ROUND49_ANALYSIS_VERSION,
        "analysis_round": ROUND49_ANALYSIS_ROUND,
        "strategy": "A_medium_30k_extraction",
        "current_subset": "cc.ko.300.subset.small.5k",
        "projected_subset": "cc.ko.300.subset.medium.30k",
        "current_primary_hit_rate": current_primary,
        "current_fallback_rate": current_fallback,
        "projected_primary_hit_rate_range": [projected_low, projected_high],
        "projected_fallback_rate_range": [1.0 - projected_high, 1.0 - projected_low],
        "oov_resolvable_by_30k": classified["general_korean"],
        "oov_unresolvable_by_30k": classified["eve_specific"],
        "ambiguous_oov": classified["ambiguous"],
        "extraction_cost": "Colab operator extraction + deterministic manifest/checksums; expected artifact about 36MB for vectors plus metadata",
        "deterministic": True,
        "alignment_with_appendix_d_drift": "low_to_medium_external_seed_dependency_increases",
        "auto_extract_medium_30k": False,
        "read_only": True,
    }


def project_self_learning_impact(current_data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project EVE-specific self-learning expansion without implementing it."""
    data = current_data if isinstance(current_data, Mapping) else {}
    telemetry = _telemetry_summary(data)
    return {
        "analysis_version": ROUND49_ANALYSIS_VERSION,
        "analysis_round": ROUND49_ANALYSIS_ROUND,
        "strategy": "B_continuous_self_learning",
        "method": "deterministic incremental vocabulary/concept observation through the existing PMI+SVD fallback substrate",
        "substrate": "SelfEmbeddingAdapter(PMI+SVD, 50d) fallback path",
        "current_primary_hit_rate": _safe_float(telemetry.get("primary_hit_rate")) or _safe_float(data.get("primary_hit_rate", 0.25)),
        "current_fallback_rate": _safe_float(telemetry.get("fallback_rate")) or _safe_float(data.get("fallback_rate", 0.75)),
        "current_vocab": "EVE-local learned vocabulary, project-specific and stateful",
        "projected_growth_per_round": "tens_to_hundreds_after_policy_and_quarantine_rules_exist",
        "minsok_specific_strength": "high",
        "eve_specific_strength": "high",
        "general_korean_strength": "partial_only_after_runtime_exposure",
        "implementation_complexity": "medium",
        "alignment_with_appendix_d_drift": "high_eve_specific_distribution_moves_away_from_seed",
        "auto_implement_self_learning": False,
        "read_only": True,
    }


def compare_lexical_coverage_strategies(
    projections: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare A/B/C lexical coverage strategies and record C as selected."""
    projections = projections if isinstance(projections, Mapping) else {}
    medium = projections.get("medium_30k") if isinstance(projections.get("medium_30k"), Mapping) else project_medium_30k_impact()
    self_learning = projections.get("self_learning") if isinstance(projections.get("self_learning"), Mapping) else project_self_learning_impact()
    strategy_c = {
        "strategy": ROUND49_STRATEGY,
        "components": ["A_medium_30k_extraction", "B_continuous_self_learning"],
        "sequence": ROUND49_STRATEGY_SEQUENCE,
        "two_phase": "A first for quick general Korean coverage, B continuous for EVE/Minsok-specific drift",
        "selected_by_minsok": True,
        "decision_status": "confirmed_at_round49",
        "round50_next": "medium_30k_extraction_operator_task",
        "round51_plus_next": "self_learning_mechanism_start",
    }
    return {
        "analysis_version": ROUND49_ANALYSIS_VERSION,
        "analysis_round": ROUND49_ANALYSIS_ROUND,
        "strategy_A_medium_30k": dict(medium),
        "strategy_B_self_learning": dict(self_learning),
        "strategy_C_hybrid": strategy_c,
        "selected_strategy": ROUND49_STRATEGY,
        "selected_sequence": ROUND49_STRATEGY_SEQUENCE,
        "manual_decision": True,
        "data_dict_no_recommendation": True,
        "auto_apply": False,
        "no_medium_30k_extraction_this_round": True,
        "no_self_learning_implementation_this_round": True,
        "read_only": True,
    }


# --- v3 round52: post-medium smoke rerun + primary hit-rate measurement ---

ROUND52_ANALYSIS_VERSION = "v3_round52_post_medium_smoke_measurement"
ROUND52_ANALYSIS_ROUND = 52
ROUND52_PRE_BRIDGE_SMALL_5K_PRIMARY = 0.3542
ROUND52_POST_BRIDGE_SMALL_5K_PRIMARY = 0.25
ROUND52_PRE_BRIDGE_SMALL_5K_FALLBACK = 0.6458
ROUND52_POST_BRIDGE_SMALL_5K_FALLBACK = 0.75
ROUND52_PROJECTED_PRIMARY_RANGE = (0.50, 0.65)
ROUND52_GENERAL_KOREAN_OOV = ("어때", "그래", "뭐야", "좋아해", "군대", "코딩")
ROUND52_EVE_SPECIFIC_OOV = ("EVE", "민석")


def _extract_telemetry_rates(data: Mapping[str, Any] | None) -> dict[str, float]:
    source = data if isinstance(data, Mapping) else {}
    telemetry = source.get("wrapper_telemetry") if isinstance(source.get("wrapper_telemetry"), Mapping) else source
    return {
        "primary_hit_rate": _safe_float(telemetry.get("primary_hit_rate")),
        "fallback_rate": _safe_float(telemetry.get("fallback_rate")),
        "error_rate": _safe_float(telemetry.get("error_rate")),
    }


def _post_medium_oov_words(post_medium_data: Mapping[str, Any] | None) -> list[str]:
    data = post_medium_data if isinstance(post_medium_data, Mapping) else {}
    telemetry = data.get("wrapper_telemetry") if isinstance(data.get("wrapper_telemetry"), Mapping) else {}
    recent = telemetry.get("recent_oov_sample") if isinstance(telemetry, Mapping) else None
    if isinstance(recent, list):
        return sorted(set(_word_from_oov(row) for row in recent if _word_from_oov(row)))
    return sorted(set(_oov_words_from_any(data)))


def compare_three_baselines(
    pre_bridge_data: Mapping[str, Any] | None = None,
    post_bridge_data: Mapping[str, Any] | None = None,
    post_medium_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare small-5k pre/post bridge telemetry with post-medium telemetry.

    This is read-only measurement. It does not change wrapper policy, AGP,
    thresholds, subset state, fallback policy, or self-learning state.
    """
    pre_rates = _extract_telemetry_rates(pre_bridge_data)
    post_bridge_rates = _extract_telemetry_rates(post_bridge_data)
    post_medium_rates = _extract_telemetry_rates(post_medium_data)
    pre_primary = pre_rates["primary_hit_rate"] or ROUND52_PRE_BRIDGE_SMALL_5K_PRIMARY
    post_bridge_primary = post_bridge_rates["primary_hit_rate"] or ROUND52_POST_BRIDGE_SMALL_5K_PRIMARY
    post_medium_primary = post_medium_rates["primary_hit_rate"]
    pre_fallback = pre_rates["fallback_rate"] or ROUND52_PRE_BRIDGE_SMALL_5K_FALLBACK
    post_bridge_fallback = post_bridge_rates["fallback_rate"] or ROUND52_POST_BRIDGE_SMALL_5K_FALLBACK
    post_medium_fallback = post_medium_rates["fallback_rate"]

    oov_words = set(_post_medium_oov_words(post_medium_data))
    actual_resolved = [word for word in ROUND52_GENERAL_KOREAN_OOV if word not in oov_words]
    actual_still_general = [word for word in ROUND52_GENERAL_KOREAN_OOV if word in oov_words]
    actual_eve_specific = [word for word in ROUND52_EVE_SPECIFIC_OOV if word in oov_words]
    agp = (post_medium_data or {}).get("agp_trace_summary", {}) if isinstance(post_medium_data, Mapping) else {}
    agp_pass = _safe_float(agp.get("pass_rate")) if isinstance(agp, Mapping) else 0.0
    return {
        "analysis_version": ROUND52_ANALYSIS_VERSION,
        "analysis_round": ROUND52_ANALYSIS_ROUND,
        "primary_hit_rate_trajectory": {
            "pre_bridge_small_5k": pre_primary,
            "post_bridge_small_5k": post_bridge_primary,
            "post_medium_swap_30k": post_medium_primary,
            "post_medium_minus_post_bridge": post_medium_primary - post_bridge_primary,
        },
        "fallback_rate_trajectory": {
            "pre_bridge_small_5k": pre_fallback,
            "post_bridge_small_5k": post_bridge_fallback,
            "post_medium_swap_30k": post_medium_fallback,
            "post_medium_minus_post_bridge": post_medium_fallback - post_bridge_fallback,
        },
        "oov_pattern_change": {
            "round44_oov_general_korean": list(ROUND52_GENERAL_KOREAN_OOV),
            "round44_oov_eve_specific": list(ROUND52_EVE_SPECIFIC_OOV),
            "actual_resolved_in_smoke": actual_resolved,
            "actual_general_korean_still_oov_in_smoke": actual_still_general,
            "actual_eve_specific_still_oov_in_smoke": actual_eve_specific,
            "recent_oov_words": sorted(oov_words),
        },
        "agp_behavior_unchanged": {
            "pass_rate": agp_pass,
            "stability": "verified" if agp_pass == 1.0 else "needs_review",
            "bridge_still_active": True,
        },
        "read_only": True,
        "no_wrapper_change": True,
        "no_agp_change": True,
        "no_self_learning_change": True,
        "no_auto_decision": True,
    }


def validate_round49_projection_accuracy(
    projection_data: Mapping[str, Any] | None = None,
    actual_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate round49 projected medium-30k impact against round52 data."""
    projection = projection_data if isinstance(projection_data, Mapping) else {}
    projected = projection.get("projected_primary_hit_rate_range", ROUND52_PROJECTED_PRIMARY_RANGE)
    if not isinstance(projected, (list, tuple)) or len(projected) != 2:
        projected = ROUND52_PROJECTED_PRIMARY_RANGE
    low = _safe_float(projected[0])
    high = _safe_float(projected[1])
    actual_rates = _extract_telemetry_rates(actual_data)
    actual_primary = actual_rates["primary_hit_rate"]
    within = low <= actual_primary <= high
    if within:
        accuracy = "verified"
    elif actual_primary > high:
        accuracy = "high_unexpected"
    else:
        accuracy = "low_unexpected"
    return {
        "analysis_version": ROUND52_ANALYSIS_VERSION,
        "analysis_round": ROUND52_ANALYSIS_ROUND,
        "projected_range": [low, high],
        "actual_primary_hit_rate": actual_primary,
        "within_projected_range": within,
        "projection_accuracy": accuracy,
        "interpretation_data_dict": "manual_decision",
        "read_only": True,
        "no_auto_adjustment": True,
    }


def identify_remaining_lexical_issues(post_medium_data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Identify remaining lexical coverage issues after medium 30k swap."""
    oov_words = set(_post_medium_oov_words(post_medium_data))
    general_remaining = sorted(word for word in oov_words if word not in ROUND52_EVE_SPECIFIC_OOV)
    eve_specific = sorted(word for word in ROUND52_EVE_SPECIFIC_OOV if word in oov_words)
    return {
        "analysis_version": ROUND52_ANALYSIS_VERSION,
        "analysis_round": ROUND52_ANALYSIS_ROUND,
        "eve_specific_oov_remaining": eve_specific,
        "general_korean_oov_remaining": general_remaining,
        "round44_general_korean_resolved": [word for word in ROUND52_GENERAL_KOREAN_OOV if word not in oov_words],
        "self_learning_b_priority": "high" if eve_specific else "medium",
        "next_round_53_plus_b_design_required": True,
        "next_round_options": [
            "A: design deterministic self-learning intake for EVE/Minsok-specific tokens",
            "B: keep medium 30k as external lexical primary and preserve PMI+SVD fallback",
            "C: rerun smoke after B mechanism exists to measure drift and OOV reduction",
        ],
        "read_only": True,
        "no_self_learning_implementation": True,
        "no_auto_vocabulary_addition": True,
    }
