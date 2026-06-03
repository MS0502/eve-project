"""EVE v3 round58 — controlled continuous EVE-specific observation.

Round58 connects the existing observation/vector-store pieces without enabling
unbounded auto-promotion. It observes lexical tokens from runtime text, records
EVE-specific OOV candidates through ``EveVocabTracker``, and can explicitly
commit deterministic vectors from known fastText context words.

Round59 adds a deterministic commit gate: explicit vector commits must pass
minimum observation and known-fastText-context requirements before any vector
store mutation occurs. Round62 adds read-only dry-runs for stricter future
observation thresholds without changing the active commit policy. Round63 turns
that dry-run evidence into an operator-facing threshold proposal report, still
without changing thresholds automatically. Round64 explicitly applies the
stricter observation threshold: two observations are now required before an
EVE-specific vector commit can pass. Round65 adds a read-only threshold
configuration/rollback snapshot so the active policy and previous manual
rollback target are visible without enabling automatic rollback. Round66 adds
a read-only observation evidence quality summary that checks repeated evidence
and context diversity before any future gate hardening. Round67 adds a read-only
context-diversity gate dry-run so the operator can see which candidates would be
blocked before making context diversity an active commit requirement. Round68
turns that dry-run evidence into an operator-facing proposal report while still
leaving the active gate unchanged. Round69 enforces the context-diversity requirement as an actual commit gate: repeated same-context observations no longer pass explicit vector commit. Round70 adds read-only blocked-candidate reporting and a manual rollback drill so the operator can inspect what would pass if context-diversity enforcement were manually disabled in a later patch. Round76 freezes the Round57-75 self-learning safety pipeline as v1 baseline without adding new learning shortcuts or changing active gates.

Policy boundary:
- observation may run during chat_stream;
- vector creation requires an explicit ``commit_eve_specific_vectors`` call;
- no memory/quarantine promotion and no AGP bypass.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
import json
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")


class EveSelfLearningAdapter:
    """Coordinate deterministic EVE-specific lexical observation and updates."""

    implementation_phase = "round71_self_learning_policy_consolidation"
    consolidation_version = "v3_round71_self_learning_policy_consolidation"
    freeze_baseline_version = "v3_round76_self_learning_v1_freeze_baseline"
    active_policy_version = "v3_round69_context_diversity_gate_enforced"
    threshold_policy_snapshot_version = "v3_round65_threshold_policy_snapshot"
    observation_evidence_quality_version = "v3_round66_observation_evidence_quality_summary"
    context_diversity_gate_dry_run_version = "v3_round67_context_diversity_gate_dry_run"
    context_diversity_proposal_version = "v3_round68_context_diversity_proposal_report"
    context_diversity_gate_enforcement_version = "v3_round69_context_diversity_gate_enforcement"
    context_diversity_rollback_drill_version = "v3_round70_context_diversity_rollback_drill"
    blocked_candidate_report_version = "v3_round70_context_diversity_blocked_candidate_report"
    threshold_policy_history = (
        {
            "round": 58,
            "policy_version": "v3_round58_continuous_observation_explicit_commit_only",
            "min_observations_for_commit": 1,
            "summary": "continuous observation enabled; explicit commit only",
        },
        {
            "round": 59,
            "policy_version": "v3_round59_commit_gate",
            "min_observations_for_commit": 1,
            "summary": "explicit commits require gate audit and known context",
        },
        {
            "round": 62,
            "policy_version": "v3_round62_threshold_dry_run",
            "min_observations_for_commit": 1,
            "summary": "threshold 2/3 evaluated as read-only dry-run",
        },
        {
            "round": 63,
            "policy_version": "v3_round63_threshold_proposal_report",
            "min_observations_for_commit": 1,
            "summary": "operator-facing proposal report only",
        },
        {
            "round": 64,
            "policy_version": "v3_round64_min_observations_2_enforced",
            "min_observations_for_commit": 2,
            "summary": "two observations required before vector commit can pass",
        },
        {
            "round": 65,
            "policy_version": "v3_round65_threshold_policy_snapshot",
            "min_observations_for_commit": 2,
            "summary": "read-only config and manual rollback reference snapshot",
        },
        {
            "round": 66,
            "policy_version": "v3_round66_observation_evidence_quality_summary",
            "min_observations_for_commit": 2,
            "summary": "read-only evidence quality summary for repeated/context-diverse observations",
        },
        {
            "round": 67,
            "policy_version": "v3_round67_context_diversity_gate_dry_run",
            "min_observations_for_commit": 2,
            "summary": "read-only dry-run for requiring context diversity before vector commit",
        },
        {
            "round": 68,
            "policy_version": "v3_round68_context_diversity_proposal_report",
            "min_observations_for_commit": 2,
            "summary": "operator-facing proposal report for context-diversity gate; active gate unchanged",
        },
        {
            "round": 69,
            "policy_version": "v3_round69_context_diversity_gate_enforced",
            "min_observations_for_commit": 2,
            "summary": "context-diversity is now required before explicit vector commit can pass",
        },
        {
            "round": 70,
            "policy_version": "v3_round70_context_diversity_rollback_drill",
            "min_observations_for_commit": 2,
            "summary": "read-only blocked-candidate report and manual rollback drill; active context-diversity gate remains enabled",
        },
    )

    def __init__(
        self,
        engine: Any | None = None,
        tracker: Any | None = None,
        vector_store: Any | None = None,
        *,
        context_cap: int = 1000,
        auto_observe_enabled: bool = True,
        auto_promotion_enabled: bool = False,
        commit_gate_enabled: bool = True,
        min_observations_for_commit: int = 2,
        min_known_context_words_for_commit: int = 1,
        context_diversity_gate_enabled: bool = True,
    ) -> None:
        self.engine = engine
        self.tracker = tracker
        self.vector_store = vector_store
        self.context_cap = max(0, int(context_cap))
        self.auto_observe_enabled = bool(auto_observe_enabled)
        self.auto_promotion_enabled = bool(auto_promotion_enabled)
        self._events: list[dict[str, Any]] = []
        self._observe_calls = 0
        self._commit_calls = 0
        self._vectors_created = 0
        self._vectors_skipped = 0
        self.commit_gate_enabled = bool(commit_gate_enabled)
        self.min_observations_for_commit = max(1, int(min_observations_for_commit))
        self.min_known_context_words_for_commit = max(1, int(min_known_context_words_for_commit))
        self.context_diversity_gate_enabled = bool(context_diversity_gate_enabled)
        self._audit_calls = 0
        self._commit_gate_blocks = 0
        self._commit_audit_records: list[dict[str, Any]] = []

    def _resolve_tracker(self) -> Any | None:
        return self.tracker if self.tracker is not None else getattr(self.engine, "eve_vocab_tracker", None)

    def _resolve_store(self) -> Any | None:
        return self.vector_store if self.vector_store is not None else getattr(self.engine, "eve_specific_vector_store", None)

    def tokenize(self, text: str) -> list[str]:
        """Return deterministic lexical tokens from Korean/ASCII text."""
        return [m.group(0).strip() for m in _TOKEN_RE.finditer(str(text or "")) if m.group(0).strip()]

    def observe_text(self, text: str, *, source: str = "runtime", context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Observe text into the tracker and return a read-only summary.

        The method records all lexical tokens, but only tracker-classified
        fastText-OOV observed tokens become EVE-specific candidates.
        """
        self._observe_calls += 1
        tokens = self.tokenize(text)
        tracker = self._resolve_tracker()
        base_context = {
            "source": str(source),
            "text": str(text or ""),
            "tokens": list(tokens),
        }
        if context:
            base_context.update(deepcopy(context))
        observed: list[str] = []
        eve_specific: list[str] = []
        if tracker is not None:
            for token in tokens:
                tracker.observe_word(token, context=base_context)
                observed.append(token)
            for token in sorted(set(tokens)):
                try:
                    if tracker.is_eve_specific(token):
                        eve_specific.append(token)
                except Exception:
                    continue
        event = {
            "source": str(source),
            "token_count": len(tokens),
            "observed": list(observed),
            "eve_specific_candidates": list(eve_specific),
            "auto_promotion_enabled": self.auto_promotion_enabled,
        }
        self._remember(event)
        return {
            "adapter": self.__class__.__name__,
            "observed_count": len(observed),
            "tokens": list(tokens),
            "eve_specific_candidates": list(eve_specific),
            "auto_observe_enabled": self.auto_observe_enabled,
            "auto_promotion_enabled": self.auto_promotion_enabled,
            "read_only": True,
        }


    def _known_context_words(self, context_words: Iterable[str] | None) -> list[str]:
        """Return context tokens that are known in the immutable fastText seed.

        This is read-only. It mirrors EveSpecificVectorStore's context filter so
        the audit decision and actual vector mutation use the same substrate.
        """
        store = self._resolve_store()
        ctx = [str(w or "").strip() for w in (context_words or []) if str(w or "").strip()]
        if not ctx or store is None:
            return []
        context_vectors = getattr(store, "_context_vectors", None)
        if callable(context_vectors):
            try:
                _vectors, used = context_vectors(ctx)
                return list(used)
            except Exception:
                return []
        return []

    def _target_words(self, words: Iterable[str] | None) -> list[str]:
        tracker = self._resolve_tracker()
        if words is None:
            if tracker is not None and hasattr(tracker, "eve_specific_vocab"):
                return list(tracker.eve_specific_vocab())
            return []
        return sorted({str(w or "").strip() for w in words if str(w or "").strip()})

    def audit_commit_gate(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Read-only Round59 audit for explicit Eve-specific vector commits.

        A candidate is eligible only when it is already observed, classified as
        EVE-specific by the tracker, and has enough known fastText context words
        to derive a deterministic vector. No vector store mutation happens here.
        """
        self._audit_calls += 1
        tracker = self._resolve_tracker()
        store = self._resolve_store()
        target_words = self._target_words(words)
        raw_context = [str(w or "").strip() for w in (context_words or []) if str(w or "").strip()]
        known_context = self._known_context_words(raw_context)
        known_context_count = len(known_context)
        quality = self.observation_evidence_quality_summary(words=target_words)
        quality_by_word = {
            str(item.get("word", "") or ""): item
            for item in list(quality.get("candidate_summaries", []) or [])
            if str(item.get("word", "") or "").strip()
        }
        candidate_reports: list[dict[str, Any]] = []
        eligible: list[str] = []
        rejected: list[str] = []
        for word in target_words:
            reasons: list[str] = []
            observed_count = 0
            is_eve_specific = False
            quality_item = quality_by_word.get(word, {})
            context_diverse = bool(quality_item.get("context_diverse", False))
            evidence_status = quality_item.get("evidence_status")
            if tracker is None:
                reasons.append("tracker_missing")
            else:
                try:
                    observed_count = int(tracker.get_observation_count(word))
                except Exception:
                    observed_count = 0
                try:
                    is_eve_specific = bool(tracker.is_eve_specific(word))
                except Exception:
                    is_eve_specific = False
                if observed_count < self.min_observations_for_commit:
                    reasons.append("insufficient_observations")
                if not is_eve_specific:
                    reasons.append("not_eve_specific_candidate")
            if store is None:
                reasons.append("vector_store_missing")
            if known_context_count < self.min_known_context_words_for_commit:
                reasons.append("insufficient_known_context")
            if self.context_diversity_gate_enabled and not context_diverse:
                reasons.append("insufficient_context_diversity")
            passed = not reasons or not self.commit_gate_enabled
            if passed:
                eligible.append(word)
            else:
                rejected.append(word)
            candidate_reports.append({
                "word": word,
                "passed": bool(passed),
                "reasons": list(reasons),
                "observed_count": int(observed_count),
                "is_eve_specific_candidate": bool(is_eve_specific),
                "context_diverse": bool(context_diverse),
                "evidence_status": evidence_status,
            })
        gate_pass = bool(target_words) and not rejected and bool(eligible)
        report = {
            "adapter": self.__class__.__name__,
            "audit_version": "v3_round59_commit_gate_audit",
            "active_policy_version": self.active_policy_version,
            "round": 59,
            "policy_round": 70,
            "commit_gate_enabled": bool(self.commit_gate_enabled),
            "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
            "context_diversity_gate_enforcement_version": self.context_diversity_gate_enforcement_version,
            "min_observations_for_commit": int(self.min_observations_for_commit),
            "min_known_context_words_for_commit": int(self.min_known_context_words_for_commit),
            "target_words": list(target_words),
            "eligible_words": list(eligible),
            "rejected_words": list(rejected),
            "candidate_reports": candidate_reports,
            "context_words": list(raw_context),
            "known_context_words": list(known_context),
            "known_context_count": int(known_context_count),
            "gate_pass": bool(gate_pass),
            "read_only": True,
        }
        self._record_commit_audit(report, event_type="audit")
        return report

    def _record_commit_audit(self, report: dict[str, Any], *, event_type: str) -> None:
        """Persist a deterministic in-memory audit record without vector mutation."""
        record = {
            "index": len(self._commit_audit_records) + 1,
            "event_type": str(event_type),
            "audit_version": report.get("audit_version"),
            "round": int(report.get("round", 60) or 60),
            "target_words": list(report.get("target_words", []) or []),
            "eligible_words": list(report.get("eligible_words", []) or []),
            "rejected_words": list(report.get("rejected_words", []) or []),
            "candidate_reports": deepcopy(report.get("candidate_reports", []) or []),
            "context_words": list(report.get("context_words", []) or []),
            "known_context_words": list(report.get("known_context_words", []) or []),
            "known_context_count": int(report.get("known_context_count", 0) or 0),
            "gate_pass": bool(report.get("gate_pass", False)),
            "commit_gate_enabled": bool(report.get("commit_gate_enabled", self.commit_gate_enabled)),
            "min_observations_for_commit": int(report.get("min_observations_for_commit", self.min_observations_for_commit) or self.min_observations_for_commit),
            "min_known_context_words_for_commit": int(report.get("min_known_context_words_for_commit", self.min_known_context_words_for_commit) or self.min_known_context_words_for_commit),
            "active_policy_version": report.get("active_policy_version", self.active_policy_version),
            "context_diversity_gate_enabled": bool(report.get("context_diversity_gate_enabled", self.context_diversity_gate_enabled)),
            "context_diversity_gate_enforcement_version": report.get("context_diversity_gate_enforcement_version", self.context_diversity_gate_enforcement_version),
            "read_only": True,
        }
        self._commit_audit_records.append(record)
        if len(self._commit_audit_records) > self.context_cap > 0:
            del self._commit_audit_records[: len(self._commit_audit_records) - self.context_cap]

    def commit_audit_records(self) -> list[dict[str, Any]]:
        """Return deterministic read-only commit-gate audit records."""
        return deepcopy(self._commit_audit_records)

    def export_commit_audit_snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly Round60 snapshot of commit-gate audits."""
        records = self.commit_audit_records()
        rejected_total = sum(len(r.get("rejected_words", []) or []) for r in records)
        eligible_total = sum(len(r.get("eligible_words", []) or []) for r in records)
        return {
            "adapter": self.__class__.__name__,
            "export_version": "v3_round60_commit_audit_export",
            "round": 60,
            "record_count": len(records),
            "eligible_total": int(eligible_total),
            "rejected_total": int(rejected_total),
            "records": records,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
                "memory_quarantine_unchanged": True,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }


    def commit_audit_dashboard_snapshot(self, *, recent_limit: int = 5) -> dict[str, Any]:
        """Return a compact Round61 dashboard over commit-gate audit records.

        This is a read-only observability surface. It aggregates allowed/rejected
        words, rejection reasons, event types, and a bounded recent trend without
        changing vector-store, memory, quarantine, seed, or AGP state.
        """
        records = self.commit_audit_records()
        reason_counts: dict[str, int] = {}
        event_type_counts: dict[str, int] = {}
        word_attempt_counts: dict[str, int] = {}
        eligible_total = 0
        rejected_total = 0
        gate_pass_records = 0
        gate_block_records = 0
        for record in records:
            event_type = str(record.get("event_type", "unknown") or "unknown")
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            eligible_words = list(record.get("eligible_words", []) or [])
            rejected_words = list(record.get("rejected_words", []) or [])
            eligible_total += len(eligible_words)
            rejected_total += len(rejected_words)
            if bool(record.get("gate_pass", False)):
                gate_pass_records += 1
            else:
                gate_block_records += 1
            for word in list(record.get("target_words", []) or []):
                word = str(word or "").strip()
                if word:
                    word_attempt_counts[word] = word_attempt_counts.get(word, 0) + 1
            for candidate in list(record.get("candidate_reports", []) or []):
                for reason in list(candidate.get("reasons", []) or []):
                    reason = str(reason or "").strip()
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        limit = max(0, int(recent_limit))
        recent_records = records[-limit:] if limit else []
        recent_trend = [
            {
                "index": int(r.get("index", 0) or 0),
                "event_type": str(r.get("event_type", "unknown") or "unknown"),
                "gate_pass": bool(r.get("gate_pass", False)),
                "eligible_count": len(r.get("eligible_words", []) or []),
                "rejected_count": len(r.get("rejected_words", []) or []),
                "reasons": sorted({
                    str(reason)
                    for c in list(r.get("candidate_reports", []) or [])
                    for reason in list(c.get("reasons", []) or [])
                    if str(reason or "").strip()
                }),
            }
            for r in recent_records
        ]
        most_attempted_words = [
            {"word": word, "attempts": int(count)}
            for word, count in sorted(word_attempt_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ]
        return {
            "adapter": self.__class__.__name__,
            "dashboard_version": "v3_round61_commit_audit_dashboard",
            "active_policy_version": self.active_policy_version,
            "round": 61,
            "policy_round": 69,
            "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
            "context_diversity_gate_enforcement_version": self.context_diversity_gate_enforcement_version,
            "record_count": len(records),
            "eligible_total": int(eligible_total),
            "rejected_total": int(rejected_total),
            "gate_pass_records": int(gate_pass_records),
            "gate_block_records": int(gate_block_records),
            "event_type_counts": dict(sorted(event_type_counts.items())),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "most_attempted_words": most_attempted_words,
            "recent_trend": recent_trend,
            "recent_limit": int(limit),
            "threshold_readiness_version": "v3_round62_commit_threshold_readiness",
            "threshold_readiness": self.commit_threshold_readiness_snapshot(),
            "threshold_proposal_version": "v3_round63_threshold_proposal_report",
            "threshold_proposal": self.threshold_proposal_report(),
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "memory_quarantine_unchanged": True,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }


    def dry_run_commit_thresholds(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
        observation_thresholds: Iterable[int] = (2, 3),
        known_context_thresholds: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Read-only Round62 dry-run for stricter future commit thresholds.

        The active policy is intentionally unchanged. This method recomputes
        candidate eligibility with hypothetical observation/context thresholds so
        future hardening can be inspected before changing
        ``min_observations_for_commit``. It does not increment audit counters,
        append audit records, create vectors, promote memory, mutate seed data,
        or bypass AGP.
        """
        tracker = self._resolve_tracker()
        store = self._resolve_store()
        target_words = self._target_words(words)
        raw_context = [str(w or "").strip() for w in (context_words or []) if str(w or "").strip()]
        known_context = self._known_context_words(raw_context)
        known_context_count = len(known_context)
        obs_thresholds = sorted({max(1, int(t)) for t in observation_thresholds})
        if not obs_thresholds:
            obs_thresholds = [int(self.min_observations_for_commit)]
        ctx_thresholds = sorted({max(1, int(t)) for t in (known_context_thresholds or [self.min_known_context_words_for_commit])})
        threshold_reports: list[dict[str, Any]] = []
        for obs_threshold in obs_thresholds:
            for ctx_threshold in ctx_thresholds:
                candidate_reports: list[dict[str, Any]] = []
                eligible: list[str] = []
                rejected: list[str] = []
                for word in target_words:
                    reasons: list[str] = []
                    observed_count = 0
                    is_eve_specific = False
                    if tracker is None:
                        reasons.append("tracker_missing")
                    else:
                        try:
                            observed_count = int(tracker.get_observation_count(word))
                        except Exception:
                            observed_count = 0
                        try:
                            is_eve_specific = bool(tracker.is_eve_specific(word))
                        except Exception:
                            is_eve_specific = False
                        if observed_count < obs_threshold:
                            reasons.append("insufficient_observations")
                        if not is_eve_specific:
                            reasons.append("not_eve_specific_candidate")
                    if store is None:
                        reasons.append("vector_store_missing")
                    if known_context_count < ctx_threshold:
                        reasons.append("insufficient_known_context")
                    passed = not reasons
                    if passed:
                        eligible.append(word)
                    else:
                        rejected.append(word)
                    candidate_reports.append({
                        "word": word,
                        "passed": bool(passed),
                        "reasons": list(reasons),
                        "observed_count": int(observed_count),
                        "is_eve_specific_candidate": bool(is_eve_specific),
                        "observation_threshold": int(obs_threshold),
                        "known_context_threshold": int(ctx_threshold),
                    })
                threshold_reports.append({
                    "observation_threshold": int(obs_threshold),
                    "known_context_threshold": int(ctx_threshold),
                    "eligible_words": list(eligible),
                    "rejected_words": list(rejected),
                    "gate_pass": bool(target_words) and not rejected and bool(eligible),
                    "candidate_reports": candidate_reports,
                })
        return {
            "adapter": self.__class__.__name__,
            "dry_run_version": "v3_round62_multi_observation_commit_threshold_dry_run",
            "round": 62,
            "active_min_observations_for_commit": int(self.min_observations_for_commit),
            "active_min_known_context_words_for_commit": int(self.min_known_context_words_for_commit),
            "active_policy_version": self.active_policy_version,
            "active_policy_unchanged": True,
            "target_words": list(target_words),
            "context_words": list(raw_context),
            "known_context_words": list(known_context),
            "known_context_count": int(known_context_count),
            "threshold_reports": threshold_reports,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "memory_quarantine_unchanged": True,
                "agp_bypass": False,
                "no_threshold_change": True,
                "no_vector_mutation": True,
            },
            "read_only": True,
        }

    def commit_threshold_readiness_snapshot(
        self,
        *,
        observation_thresholds: Iterable[int] = (2, 3),
    ) -> dict[str, Any]:
        """Summarize observed audit candidates against stricter observation thresholds.

        This dashboard-oriented view uses existing audit records only. It is not
        a commit decision and it does not create new audit records.
        """
        records = self.commit_audit_records()
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            known_context_count = int(record.get("known_context_count", 0) or 0)
            for candidate in list(record.get("candidate_reports", []) or []):
                word = str(candidate.get("word", "") or "").strip()
                if not word:
                    continue
                prev = latest.get(word)
                observed_count = int(candidate.get("observed_count", 0) or 0)
                snapshot = {
                    "word": word,
                    "observed_count": observed_count,
                    "is_eve_specific_candidate": bool(candidate.get("is_eve_specific_candidate", False)),
                    "known_context_count": known_context_count,
                }
                if prev is None or observed_count >= int(prev.get("observed_count", 0) or 0):
                    latest[word] = snapshot
        thresholds = sorted({max(1, int(t)) for t in observation_thresholds})
        if not thresholds:
            thresholds = [2, 3]
        threshold_summaries: list[dict[str, Any]] = []
        for threshold in thresholds:
            ready: list[str] = []
            blocked: list[str] = []
            for word, snapshot in sorted(latest.items()):
                enough_observations = int(snapshot.get("observed_count", 0) or 0) >= threshold
                if enough_observations and bool(snapshot.get("is_eve_specific_candidate", False)):
                    ready.append(word)
                else:
                    blocked.append(word)
            threshold_summaries.append({
                "observation_threshold": int(threshold),
                "ready_words": list(ready),
                "blocked_words": list(blocked),
                "ready_count": len(ready),
                "blocked_count": len(blocked),
            })
        return {
            "adapter": self.__class__.__name__,
            "readiness_version": "v3_round62_commit_threshold_readiness",
            "round": 62,
            "active_min_observations_for_commit": int(self.min_observations_for_commit),
            "active_policy_version": self.active_policy_version,
            "active_policy_unchanged": True,
            "candidate_count": len(latest),
            "candidate_snapshots": [latest[word] for word in sorted(latest)],
            "threshold_summaries": threshold_summaries,
            "read_only": True,
        }


    def threshold_proposal_report(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
        target_observation_threshold: int = 2,
        known_context_threshold: int | None = None,
    ) -> dict[str, Any]:
        """Return a read-only Round63 proposal for a stricter commit threshold.

        This method converts the Round62 dry-run surface into an operator review
        report. It never changes ``min_observations_for_commit`` and never
        creates vectors, audit records, memory entries, quarantine entries, or
        seed mutations. The report is a decision aid only; applying a stricter
        threshold must be a later explicit round.
        """
        target_threshold = max(1, int(target_observation_threshold))
        ctx_threshold = max(1, int(known_context_threshold or self.min_known_context_words_for_commit))
        before_records = len(self._commit_audit_records)
        dry_run = self.dry_run_commit_thresholds(
            words=words,
            context_words=context_words,
            observation_thresholds=[target_threshold],
            known_context_thresholds=[ctx_threshold],
        )
        after_records = len(self._commit_audit_records)
        selected_report: dict[str, Any] = {
            "observation_threshold": target_threshold,
            "known_context_threshold": ctx_threshold,
            "eligible_words": [],
            "rejected_words": [],
            "candidate_reports": [],
            "gate_pass": False,
        }
        for report in list(dry_run.get("threshold_reports", []) or []):
            if (
                int(report.get("observation_threshold", 0) or 0) == target_threshold
                and int(report.get("known_context_threshold", 0) or 0) == ctx_threshold
            ):
                selected_report = report
                break
        candidate_count = len(selected_report.get("candidate_reports", []) or [])
        eligible_words = list(selected_report.get("eligible_words", []) or [])
        rejected_words = list(selected_report.get("rejected_words", []) or [])
        current_threshold = int(self.min_observations_for_commit)
        if current_threshold >= target_threshold:
            recommendation = "already_at_or_above_target"
        elif candidate_count == 0:
            recommendation = "insufficient_evidence"
        elif rejected_words:
            recommendation = "defer_threshold_change"
        elif eligible_words:
            recommendation = "operator_may_consider_threshold_change"
        else:
            recommendation = "insufficient_evidence"
        return {
            "adapter": self.__class__.__name__,
            "proposal_version": "v3_round63_threshold_proposal_report",
            "round": 63,
            "current_min_observations_for_commit": current_threshold,
            "proposed_min_observations_for_commit": int(target_threshold),
            "current_min_known_context_words_for_commit": int(self.min_known_context_words_for_commit),
            "proposed_min_known_context_words_for_commit": int(ctx_threshold),
            "candidate_count": int(candidate_count),
            "eligible_under_proposal": eligible_words,
            "blocked_under_proposal": rejected_words,
            "eligible_count": len(eligible_words),
            "blocked_count": len(rejected_words),
            "candidate_reports": deepcopy(selected_report.get("candidate_reports", []) or []),
            "recommendation": recommendation,
            "operator_review_required": True,
            "target_already_enforced": bool(current_threshold >= target_threshold),
            "active_policy_unchanged": True,
            "audit_records_unchanged": before_records == after_records,
            "source_dry_run_version": dry_run.get("dry_run_version"),
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "no_threshold_change": True,
                "no_auto_threshold_adjustment": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }

    def observation_evidence_quality_summary(
        self,
        words: Iterable[str] | None = None,
        *,
        context_sample_limit: int = 3,
        peer_token_sample_limit: int = 10,
    ) -> dict[str, Any]:
        """Return a read-only Round66 summary of observation evidence quality.

        The summary checks whether EVE-specific candidates were observed more
        than once and whether their observations came from diverse contexts. It
        is an operator/debug surface only: it does not change commit thresholds,
        append audit records, create vectors, promote memory, mutate seeds, or
        bypass AGP.
        """
        tracker = self._resolve_tracker()
        before_records = len(self._commit_audit_records)
        before_store_count = 0
        store = self._resolve_store()
        try:
            before_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            before_store_count = 0

        tracker_stats = tracker.stats() if tracker is not None and hasattr(tracker, "stats") else {}
        context_records = list(tracker_stats.get("context_snapshot", []) or [])
        target_words = self._target_words(words)
        summaries: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        threshold_ready_count = 0
        context_diverse_count = 0
        repeated_same_context_count = 0
        insufficient_observation_count = 0
        ctx_limit = max(0, int(context_sample_limit))
        peer_limit = max(0, int(peer_token_sample_limit))

        for word in target_words:
            observed_count = 0
            is_eve_specific = False
            if tracker is None:
                status = "tracker_missing"
            else:
                try:
                    observed_count = int(tracker.get_observation_count(word))
                except Exception:
                    observed_count = 0
                try:
                    is_eve_specific = bool(tracker.is_eve_specific(word))
                except Exception:
                    is_eve_specific = False

                word_records = [r for r in context_records if str(r.get("word", "") or "").strip() == word]
                source_set: set[str] = set()
                text_set: set[str] = set()
                peer_tokens: set[str] = set()
                fingerprints: set[tuple[Any, ...]] = set()
                context_samples: list[dict[str, Any]] = []
                for record in word_records:
                    ctx = record.get("context", {}) if isinstance(record, dict) else {}
                    if not isinstance(ctx, dict):
                        ctx = {}
                    source = str(ctx.get("source", "") or "").strip()
                    text = str(ctx.get("text", "") or "").strip()
                    tokens = [str(t or "").strip() for t in list(ctx.get("tokens", []) or []) if str(t or "").strip()]
                    if source:
                        source_set.add(source)
                    if text:
                        text_set.add(text)
                    for token in tokens:
                        if token != word:
                            peer_tokens.add(token)
                    fingerprints.add((source, text, tuple(tokens)))
                    if len(context_samples) < ctx_limit:
                        context_samples.append({
                            "source": source,
                            "text": text,
                            "tokens": list(tokens),
                            "observation_index": int(record.get("observation_index", 0) or 0),
                        })

                unique_context_count = len(fingerprints)
                duplicate_context_count = max(0, len(word_records) - unique_context_count)
                context_diverse = unique_context_count >= 2 or len(text_set) >= 2 or len(peer_tokens) >= 2
                threshold_ready = observed_count >= int(self.min_observations_for_commit)
                if not is_eve_specific:
                    status = "not_eve_specific_candidate"
                elif not threshold_ready:
                    status = "insufficient_observations"
                elif context_diverse:
                    status = "threshold_met_context_diverse"
                elif duplicate_context_count > 0:
                    status = "threshold_met_repeated_same_context"
                else:
                    status = "threshold_met_single_context"

            if tracker is None:
                word_records = []
                source_set = set()
                text_set = set()
                peer_tokens = set()
                unique_context_count = 0
                duplicate_context_count = 0
                context_diverse = False
                threshold_ready = False
                context_samples = []

            if threshold_ready:
                threshold_ready_count += 1
            else:
                insufficient_observation_count += 1
            if context_diverse:
                context_diverse_count += 1
            if duplicate_context_count > 0 and not context_diverse:
                repeated_same_context_count += 1
            status_counts[status] = status_counts.get(status, 0) + 1
            summaries.append({
                "word": word,
                "observed_count": int(observed_count),
                "is_eve_specific_candidate": bool(is_eve_specific),
                "meets_current_observation_threshold": bool(threshold_ready),
                "context_record_count": len(word_records),
                "unique_context_count": int(unique_context_count),
                "unique_source_count": len(source_set),
                "unique_text_count": len(text_set),
                "unique_peer_token_count": len(peer_tokens),
                "context_diverse": bool(context_diverse),
                "duplicate_context_count": int(duplicate_context_count),
                "evidence_status": status,
                "peer_token_sample": sorted(peer_tokens)[:peer_limit],
                "context_samples": context_samples,
            })

        summaries.sort(key=lambda item: (str(item.get("word", ""))))
        try:
            after_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            after_store_count = before_store_count
        return {
            "adapter": self.__class__.__name__,
            "summary_version": self.observation_evidence_quality_version,
            "round": 66,
            "active_policy_version": self.active_policy_version,
            "min_observations_for_commit": int(self.min_observations_for_commit),
            "candidate_count": len(summaries),
            "threshold_ready_count": int(threshold_ready_count),
            "context_diverse_count": int(context_diverse_count),
            "repeated_same_context_count": int(repeated_same_context_count),
            "insufficient_observation_count": int(insufficient_observation_count),
            "evidence_status_counts": dict(sorted(status_counts.items())),
            "candidate_summaries": summaries,
            "audit_records_unchanged": before_records == len(self._commit_audit_records),
            "vector_store_unchanged": before_store_count == after_store_count,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
                "threshold_changed": False,
                "no_auto_threshold_adjustment": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }



    def dry_run_context_diversity_gate(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
        require_context_diversity: bool = True,
    ) -> dict[str, Any]:
        """Read-only Round67 dry-run for a future context-diversity gate.

        The current commit gate remains unchanged. This recomputes candidate
        eligibility as if a new ``context_diverse`` requirement existed, then
        reports which candidates would be newly blocked. It does not append
        audit records, create vectors, adjust thresholds, promote memory,
        mutate fastText seed data, or bypass AGP.
        """
        tracker = self._resolve_tracker()
        store = self._resolve_store()
        target_words = self._target_words(words)
        raw_context = [str(w or "").strip() for w in (context_words or []) if str(w or "").strip()]
        known_context = self._known_context_words(raw_context)
        known_context_count = len(known_context)
        before_records = len(self._commit_audit_records)
        try:
            before_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            before_store_count = 0
        quality = self.observation_evidence_quality_summary(words=target_words)
        quality_by_word = {
            str(item.get("word", "") or ""): item
            for item in list(quality.get("candidate_summaries", []) or [])
            if str(item.get("word", "") or "").strip()
        }
        candidate_reports: list[dict[str, Any]] = []
        eligible: list[str] = []
        blocked: list[str] = []
        newly_blocked_by_context_diversity: list[str] = []
        would_already_be_blocked: list[str] = []
        for word in target_words:
            item = quality_by_word.get(word, {})
            current_reasons: list[str] = []
            dry_run_reasons: list[str] = []
            observed_count = 0
            is_eve_specific = False
            if tracker is None:
                current_reasons.append("tracker_missing")
            else:
                try:
                    observed_count = int(tracker.get_observation_count(word))
                except Exception:
                    observed_count = int(item.get("observed_count", 0) or 0)
                try:
                    is_eve_specific = bool(tracker.is_eve_specific(word))
                except Exception:
                    is_eve_specific = bool(item.get("is_eve_specific_candidate", False))
                if observed_count < self.min_observations_for_commit:
                    current_reasons.append("insufficient_observations")
                if not is_eve_specific:
                    current_reasons.append("not_eve_specific_candidate")
            if store is None:
                current_reasons.append("vector_store_missing")
            if known_context_count < self.min_known_context_words_for_commit:
                current_reasons.append("insufficient_known_context")
            context_diverse = bool(item.get("context_diverse", False))
            if self.context_diversity_gate_enabled and not context_diverse:
                current_reasons.append("insufficient_context_diversity")
            dry_run_reasons = list(current_reasons)
            if require_context_diversity and not context_diverse and "insufficient_context_diversity" not in dry_run_reasons:
                dry_run_reasons.append("insufficient_context_diversity")
            current_gate_pass = not current_reasons
            dry_run_pass = not dry_run_reasons
            if dry_run_pass:
                eligible.append(word)
            else:
                blocked.append(word)
            if current_gate_pass and not dry_run_pass and "insufficient_context_diversity" in dry_run_reasons:
                newly_blocked_by_context_diversity.append(word)
            elif not current_gate_pass:
                would_already_be_blocked.append(word)
            candidate_reports.append({
                "word": word,
                "current_gate_pass": bool(current_gate_pass),
                "dry_run_pass": bool(dry_run_pass),
                "current_reasons": list(current_reasons),
                "dry_run_reasons": list(dry_run_reasons),
                "observed_count": int(observed_count),
                "is_eve_specific_candidate": bool(is_eve_specific),
                "context_diverse": bool(context_diverse),
                "unique_context_count": int(item.get("unique_context_count", 0) or 0),
                "duplicate_context_count": int(item.get("duplicate_context_count", 0) or 0),
                "evidence_status": item.get("evidence_status"),
            })
        try:
            after_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            after_store_count = before_store_count
        return {
            "adapter": self.__class__.__name__,
            "dry_run_version": self.context_diversity_gate_dry_run_version,
            "round": 67,
            "active_policy_version": self.active_policy_version,
            "active_policy_unchanged": True,
            "current_gate_context_diversity_required": bool(self.context_diversity_gate_enabled),
            "dry_run_context_diversity_required": bool(require_context_diversity),
            "min_observations_for_commit": int(self.min_observations_for_commit),
            "min_known_context_words_for_commit": int(self.min_known_context_words_for_commit),
            "target_words": list(target_words),
            "context_words": list(raw_context),
            "known_context_words": list(known_context),
            "known_context_count": int(known_context_count),
            "eligible_words": list(eligible),
            "blocked_words": list(blocked),
            "newly_blocked_by_context_diversity": list(newly_blocked_by_context_diversity),
            "would_already_be_blocked": list(would_already_be_blocked),
            "candidate_reports": candidate_reports,
            "source_evidence_quality_version": quality.get("summary_version"),
            "audit_records_unchanged": before_records == len(self._commit_audit_records),
            "vector_store_unchanged": before_store_count == after_store_count,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "context_diversity_gate_enforced": bool(self.context_diversity_gate_enabled),
                "no_context_diversity_policy_change": True,
                "no_threshold_change": True,
                "no_auto_threshold_adjustment": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }


    def context_diversity_proposal_report(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
        require_context_diversity: bool = True,
    ) -> dict[str, Any]:
        """Return a read-only Round68 proposal for a context-diversity gate.

        This method converts the Round67 context-diversity dry-run into an
        operator review report. It never changes the active commit gate,
        thresholds, vectors, audit records, memory/quarantine state, fastText
        seed data, or AGP behavior. Applying the gate must remain a later
        explicit patch.
        """
        store = self._resolve_store()
        before_records = len(self._commit_audit_records)
        try:
            before_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            before_store_count = 0
        dry_run = self.dry_run_context_diversity_gate(
            words=words,
            context_words=context_words,
            require_context_diversity=require_context_diversity,
        )
        candidate_reports = deepcopy(list(dry_run.get("candidate_reports", []) or []))
        candidate_count = len(candidate_reports)
        current_gate_ready_words = [
            str(item.get("word", "") or "")
            for item in candidate_reports
            if bool(item.get("current_gate_pass", False)) and str(item.get("word", "") or "").strip()
        ]
        eligible_words = list(dry_run.get("eligible_words", []) or [])
        blocked_words = list(dry_run.get("blocked_words", []) or [])
        newly_blocked = list(dry_run.get("newly_blocked_by_context_diversity", []) or [])
        already_blocked = list(dry_run.get("would_already_be_blocked", []) or [])
        try:
            after_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            after_store_count = before_store_count

        if not require_context_diversity:
            recommendation = "context_diversity_not_requested"
        elif candidate_count == 0:
            recommendation = "insufficient_evidence"
        elif newly_blocked:
            recommendation = "defer_context_diversity_gate"
        elif eligible_words and not blocked_words:
            recommendation = "operator_may_consider_context_diversity_gate"
        elif eligible_words:
            recommendation = "operator_may_consider_after_review"
        else:
            recommendation = "insufficient_evidence"

        return {
            "adapter": self.__class__.__name__,
            "proposal_version": self.context_diversity_proposal_version,
            "round": 68,
            "active_policy_version": self.active_policy_version,
            "source_dry_run_version": dry_run.get("dry_run_version"),
            "current_context_diversity_gate_enforced": bool(self.context_diversity_gate_enabled),
            "proposed_context_diversity_gate_enforced": bool(require_context_diversity),
            "active_policy_unchanged": True,
            "candidate_count": int(candidate_count),
            "current_gate_ready_words": current_gate_ready_words,
            "current_gate_ready_count": len(current_gate_ready_words),
            "eligible_under_proposal": eligible_words,
            "blocked_under_proposal": blocked_words,
            "newly_blocked_by_context_diversity": newly_blocked,
            "would_already_be_blocked": already_blocked,
            "eligible_count": len(eligible_words),
            "blocked_count": len(blocked_words),
            "newly_blocked_count": len(newly_blocked),
            "candidate_reports": candidate_reports,
            "recommendation": recommendation,
            "operator_review_required": True,
            "audit_records_unchanged": before_records == len(self._commit_audit_records),
            "vector_store_unchanged": before_store_count == after_store_count,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "min_observations_for_commit": int(self.min_observations_for_commit),
                "context_diversity_gate_enforced": bool(self.context_diversity_gate_enabled),
                "no_context_diversity_policy_change": True,
                "no_threshold_change": True,
                "no_auto_threshold_adjustment": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }


    def context_diversity_rollback_drill(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Read-only Round70 drill for a manual context-diversity rollback.

        The active gate remains enabled. This method recomputes candidate
        eligibility as if a future operator patch manually disabled only the
        context-diversity requirement. It does not change thresholds, append
        audit records, mutate vectors, promote memory/quarantine, mutate the
        fastText seed, or bypass AGP.
        """
        tracker = self._resolve_tracker()
        store = self._resolve_store()
        if words is None:
            blocked_words: set[str] = set()
            for record in self._commit_audit_records:
                for candidate in list(record.get("candidate_reports", []) or []):
                    word = str(candidate.get("word", "") or "").strip()
                    reasons = [str(r) for r in list(candidate.get("reasons", []) or [])]
                    if word and "insufficient_context_diversity" in reasons:
                        blocked_words.add(word)
            target_words = sorted(blocked_words) or self._target_words(None)
        else:
            target_words = self._target_words(words)
        raw_context = [str(w or "").strip() for w in (context_words or []) if str(w or "").strip()]
        known_context = self._known_context_words(raw_context)
        known_context_count = len(known_context)
        before_records = len(self._commit_audit_records)
        try:
            before_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            before_store_count = 0
        quality = self.observation_evidence_quality_summary(words=target_words)
        quality_by_word = {
            str(item.get("word", "") or ""): item
            for item in list(quality.get("candidate_summaries", []) or [])
            if str(item.get("word", "") or "").strip()
        }
        candidate_reports: list[dict[str, Any]] = []
        newly_unblocked: list[str] = []
        still_blocked: list[str] = []
        active_blocked: list[str] = []
        rollback_eligible: list[str] = []
        for word in target_words:
            item = quality_by_word.get(word, {})
            observed_count = 0
            is_eve_specific = False
            active_reasons: list[str] = []
            rollback_reasons: list[str] = []
            if tracker is None:
                active_reasons.append("tracker_missing")
            else:
                try:
                    observed_count = int(tracker.get_observation_count(word))
                except Exception:
                    observed_count = int(item.get("observed_count", 0) or 0)
                try:
                    is_eve_specific = bool(tracker.is_eve_specific(word))
                except Exception:
                    is_eve_specific = bool(item.get("is_eve_specific_candidate", False))
                if observed_count < self.min_observations_for_commit:
                    active_reasons.append("insufficient_observations")
                if not is_eve_specific:
                    active_reasons.append("not_eve_specific_candidate")
            if store is None:
                active_reasons.append("vector_store_missing")
            if known_context_count < self.min_known_context_words_for_commit:
                active_reasons.append("insufficient_known_context")
            context_diverse = bool(item.get("context_diverse", False))
            if self.context_diversity_gate_enabled and not context_diverse:
                active_reasons.append("insufficient_context_diversity")
            rollback_reasons = [r for r in active_reasons if r != "insufficient_context_diversity"]
            active_pass = not active_reasons
            rollback_pass = not rollback_reasons
            if not active_pass:
                active_blocked.append(word)
            if rollback_pass:
                rollback_eligible.append(word)
            else:
                still_blocked.append(word)
            if (not active_pass) and rollback_pass and "insufficient_context_diversity" in active_reasons:
                newly_unblocked.append(word)
            candidate_reports.append({
                "word": word,
                "active_gate_pass": bool(active_pass),
                "rollback_gate_pass": bool(rollback_pass),
                "active_reasons": list(active_reasons),
                "rollback_reasons": list(rollback_reasons),
                "newly_unblocked_by_manual_context_diversity_rollback": bool((not active_pass) and rollback_pass and "insufficient_context_diversity" in active_reasons),
                "observed_count": int(observed_count),
                "is_eve_specific_candidate": bool(is_eve_specific),
                "context_diverse": bool(context_diverse),
                "unique_context_count": int(item.get("unique_context_count", 0) or 0),
                "duplicate_context_count": int(item.get("duplicate_context_count", 0) or 0),
                "evidence_status": item.get("evidence_status"),
            })
        try:
            after_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            after_store_count = before_store_count
        return {
            "adapter": self.__class__.__name__,
            "drill_version": self.context_diversity_rollback_drill_version,
            "round": 70,
            "active_policy_version": self.active_policy_version,
            "active_context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
            "hypothetical_context_diversity_gate_enabled": False,
            "active_policy_unchanged": True,
            "target_words": list(target_words),
            "context_words": list(raw_context),
            "known_context_words": list(known_context),
            "known_context_count": int(known_context_count),
            "active_blocked_words": list(active_blocked),
            "rollback_eligible_words": list(rollback_eligible),
            "newly_unblocked_by_manual_rollback": list(newly_unblocked),
            "still_blocked_after_manual_rollback": list(still_blocked),
            "candidate_reports": candidate_reports,
            "audit_records_unchanged": before_records == len(self._commit_audit_records),
            "vector_store_unchanged": before_store_count == after_store_count,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "min_observations_for_commit": int(self.min_observations_for_commit),
                "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
                "automatic_rollback_enabled": False,
                "manual_rollback_reference_only": True,
                "no_context_diversity_policy_change": True,
                "no_threshold_change": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }

    def context_diversity_blocked_candidate_report(self, *, recent_limit: int = 10) -> dict[str, Any]:
        """Aggregate Round70 audit records blocked by context diversity.

        The report is an operator/debug surface. It reads existing audit records
        and includes a bounded recent list plus a manual rollback drill. It does
        not append audit records, mutate vectors, alter gate configuration,
        promote memory/quarantine, mutate seed data, or bypass AGP.
        """
        records = self.commit_audit_records()
        by_word: dict[str, dict[str, Any]] = {}
        recent: list[dict[str, Any]] = []
        rollback_context_words: list[str] = []
        for record in records:
            event_type = str(record.get("event_type", "unknown") or "unknown")
            index = int(record.get("index", 0) or 0)
            for candidate in list(record.get("candidate_reports", []) or []):
                word = str(candidate.get("word", "") or "").strip()
                reasons = [str(r) for r in list(candidate.get("reasons", []) or []) if str(r or "").strip()]
                if not word or "insufficient_context_diversity" not in reasons:
                    continue
                for ctx_word in list(record.get("known_context_words", []) or record.get("context_words", []) or []):
                    ctx_word = str(ctx_word or "").strip()
                    if ctx_word and ctx_word not in rollback_context_words:
                        rollback_context_words.append(ctx_word)
                entry = by_word.setdefault(word, {
                    "word": word,
                    "block_count": 0,
                    "event_types": {},
                    "latest_record_index": 0,
                    "latest_observed_count": 0,
                    "latest_unique_context_count": 0,
                    "latest_duplicate_context_count": 0,
                    "latest_evidence_status": None,
                    "reasons": set(),
                })
                entry["block_count"] += 1
                entry["event_types"][event_type] = int(entry["event_types"].get(event_type, 0)) + 1
                entry["latest_record_index"] = max(int(entry["latest_record_index"]), index)
                entry["latest_observed_count"] = int(candidate.get("observed_count", entry["latest_observed_count"]) or 0)
                entry["latest_unique_context_count"] = int(candidate.get("unique_context_count", entry["latest_unique_context_count"]) or 0)
                entry["latest_duplicate_context_count"] = int(candidate.get("duplicate_context_count", entry["latest_duplicate_context_count"]) or 0)
                entry["latest_evidence_status"] = candidate.get("evidence_status")
                entry["reasons"].update(reasons)
                recent.append({
                    "record_index": index,
                    "event_type": event_type,
                    "word": word,
                    "reasons": reasons,
                    "observed_count": int(candidate.get("observed_count", 0) or 0),
                    "context_diverse": bool(candidate.get("context_diverse", False)),
                    "evidence_status": candidate.get("evidence_status"),
                })
        blocked_words: list[dict[str, Any]] = []
        for word, entry in sorted(by_word.items(), key=lambda item: (-int(item[1].get("block_count", 0)), item[0])):
            blocked_words.append({
                "word": word,
                "block_count": int(entry["block_count"]),
                "event_types": dict(sorted(entry["event_types"].items())),
                "latest_record_index": int(entry["latest_record_index"]),
                "latest_observed_count": int(entry["latest_observed_count"]),
                "latest_unique_context_count": int(entry["latest_unique_context_count"]),
                "latest_duplicate_context_count": int(entry["latest_duplicate_context_count"]),
                "latest_evidence_status": entry["latest_evidence_status"],
                "reasons": sorted(entry["reasons"]),
            })
        limit = max(0, int(recent_limit))
        recent_block_events = recent[-limit:] if limit else []
        rollback_words = [item["word"] for item in blocked_words]
        rollback_drill = self.context_diversity_rollback_drill(words=rollback_words, context_words=rollback_context_words)
        return {
            "adapter": self.__class__.__name__,
            "report_version": self.blocked_candidate_report_version,
            "round": 70,
            "active_policy_version": self.active_policy_version,
            "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
            "blocked_candidate_count": len(blocked_words),
            "blocked_words": blocked_words,
            "recent_block_events": recent_block_events,
            "recent_limit": int(limit),
            "rollback_context_words": list(rollback_context_words),
            "rollback_drill_version": rollback_drill.get("drill_version"),
            "rollback_drill": rollback_drill,
            "policy": {
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "min_observations_for_commit": int(self.min_observations_for_commit),
                "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
                "automatic_rollback_enabled": False,
                "manual_rollback_reference_only": True,
                "no_context_diversity_policy_change": True,
                "no_threshold_change": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }


    def threshold_policy_snapshot(self) -> dict[str, Any]:
        """Return a read-only Round65 threshold config/rollback snapshot.

        The snapshot makes the active Round64 threshold explicit and records the
        previous manual rollback target. It is not an automatic rollback API and
        it must not change vectors, audit records, memory/quarantine state, AGP,
        seed data, or runtime thresholds.
        """
        current_threshold = int(self.min_observations_for_commit)
        current_context_threshold = int(self.min_known_context_words_for_commit)
        previous_manual_target = {
            "policy_version": "v3_round59_commit_gate",
            "min_observations_for_commit": 1,
            "min_known_context_words_for_commit": 1,
            "reason": "last stable explicit commit gate before Round64 threshold hardening",
        }
        return {
            "adapter": self.__class__.__name__,
            "snapshot_version": self.threshold_policy_snapshot_version,
            "round": 65,
            "active_policy_version": self.active_policy_version,
            "implementation_phase": self.implementation_phase,
            "current_config": {
                "commit_gate_enabled": bool(self.commit_gate_enabled),
                "min_observations_for_commit": current_threshold,
                "min_known_context_words_for_commit": current_context_threshold,
                "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
                "context_diversity_gate_enforcement_version": self.context_diversity_gate_enforcement_version,
                "auto_observe_enabled": bool(self.auto_observe_enabled),
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
            },
            "history": deepcopy(list(self.threshold_policy_history)),
            "rollback_snapshot": {
                "manual_reference_only": True,
                "automatic_rollback_enabled": False,
                "requires_operator_patch": True,
                "previous_manual_target": previous_manual_target,
                "current_min_observations_for_commit": current_threshold,
                "rollback_would_change_min_observations_to": int(previous_manual_target["min_observations_for_commit"]),
                "vector_store_mutation": False,
                "audit_record_mutation": False,
            },
            "safety": {
                "auto_promotion_enabled": bool(self.auto_promotion_enabled),
                "no_auto_rollback": True,
                "no_auto_threshold_adjustment": True,
                "no_vector_mutation": True,
                "memory_quarantine_unchanged": True,
                "fasttext_seed_mutation": False,
                "agp_bypass": False,
                "no_drift_based_runtime_change": True,
            },
            "read_only": True,
        }

    def self_learning_v1_freeze_policy_snapshot(self) -> dict[str, Any]:
        """Return the Round76 self-learning v1 freeze policy snapshot.

        The snapshot is an operator/read-only baseline. It freezes the active
        Round57-75 self-learning safety pipeline for review and future diffing;
        it does not observe text, create vectors, append audit records, adjust
        thresholds, toggle context-diversity enforcement, rollback, promote
        memory, mutate fastText seeds, or bypass AGP.
        """
        tracker = self._resolve_tracker()
        store = self._resolve_store()
        before_records = len(self._commit_audit_records)
        before_store_count = 0
        try:
            before_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            before_store_count = 0
        tracker_stats = tracker.stats() if tracker is not None and hasattr(tracker, "stats") else {}
        store_stats = store.stats() if store is not None and hasattr(store, "stats") else {}
        try:
            after_store_count = int((store.stats() if store is not None and hasattr(store, "stats") else {}).get("stored_count", 0) or 0)
        except Exception:
            after_store_count = before_store_count
        active_policy = {
            "auto_observe_enabled": bool(self.auto_observe_enabled),
            "auto_promotion_enabled": bool(self.auto_promotion_enabled),
            "commit_gate_enabled": bool(self.commit_gate_enabled),
            "min_observations_for_commit": int(self.min_observations_for_commit),
            "min_known_context_words_for_commit": int(self.min_known_context_words_for_commit),
            "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
            "automatic_rollback_enabled": False,
            "drift_based_runtime_change": False,
            "memory_quarantine_mutation": False,
            "fasttext_seed_mutation": False,
            "agp_anchor_policy": "explicit_categories_plus_sa_activation_only",
            "agp_bypass": False,
        }
        return {
            "adapter": self.__class__.__name__,
            "freeze_version": self.freeze_baseline_version,
            "round": 76,
            "frozen_baseline_name": "self_learning_v1",
            "covered_rounds": list(range(57, 76)),
            "covered_round_range": "round57-round75",
            "active_policy_version": self.active_policy_version,
            "implementation_phase": self.implementation_phase,
            "active_policy": active_policy,
            "component_roles": {
                "EveVocabTracker": "lexical_observation_only_no_vector_generation",
                "EveSelfLearningAdapter": "continuous_observation_and_explicit_commit_gate_owner",
                "EveSpecificVectorStore": "deterministic_vector_storage_only",
                "EmbeddingWrapper": "lookup_routing_and_telemetry_only",
                "AGPAdapter": "anchor_validation_from_explicit_categories_and_sa_activation_only",
            },
            "locked_commit_gate": [
                "observed_count >= 2",
                "fastText-OOV / EVE-specific candidate",
                "known_fastText_context_words >= 1",
                "context_diverse == True",
                "explicit_commit_call_required",
            ],
            "frozen_artifacts": [
                "SELF_LEARNING_POLICY_ROUND71.md",
                "EVE_SPECIFIC_BASELINE_R72.json",
                "EVE_SPECIFIC_COMMIT_SMOKE_R73.json",
                "EVE_SPECIFIC_COMMIT_DELTA_R74.json",
                "EVE_SPECIFIC_COMMIT_REPLAY_EXPORT_R75.json",
            ],
            "verification_gates": [
                "round71_split_full_suite",
                "round72_split_full_suite",
                "round73_explicit_commit_smoke",
                "round74_commit_delta_report",
                "round75_replay_export_read_only_checks",
                "round76_freeze_snapshot_read_only_checks",
            ],
            "current_counts": {
                "observe_calls": int(self._observe_calls),
                "commit_calls": int(self._commit_calls),
                "audit_calls": int(self._audit_calls),
                "commit_audit_record_count": len(self._commit_audit_records),
                "vectors_created": int(self._vectors_created),
                "vectors_skipped": int(self._vectors_skipped),
                "tracker_eve_specific_count": int(tracker_stats.get("eve_specific_count", 0) or 0),
                "vector_store_count": int(store_stats.get("stored_count", 0) or 0),
            },
            "next_allowed_rounds": {
                "round77": "post-freeze explicit commit regression or lexical_to_concept_mapping_planning",
                "requires_new_proposal_for_policy_change": True,
                "automatic_promotion_still_forbidden": True,
                "automatic_rollback_still_forbidden": True,
            },
            "read_only_checks": {
                "audit_records_unchanged": before_records == len(self._commit_audit_records),
                "vector_store_unchanged": before_store_count == after_store_count,
                "threshold_changed": False,
                "context_diversity_gate_changed": False,
                "policy_changed": False,
            },
            "read_only": True,
        }

    def write_self_learning_v1_freeze_policy_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Export the Round76 self-learning v1 freeze policy snapshot."""
        out = Path(path)
        snapshot = self.self_learning_v1_freeze_policy_snapshot()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "adapter": self.__class__.__name__,
            "freeze_version": snapshot["freeze_version"],
            "path": str(out),
            "read_only": False,
            "recomputed": False,
            "commit_called": False,
            "vector_mutation": False,
            "audit_record_mutation": False,
            "threshold_changed": False,
            "context_diversity_gate_changed": False,
            "automatic_rollback_enabled": False,
            "memory_quarantine_unchanged": True,
            "fasttext_seed_mutation": False,
            "agp_bypass": False,
        }

    def write_threshold_policy_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Explicitly export the Round65 threshold policy snapshot to JSON.

        This export is an operator artifact only. It does not apply rollback,
        change thresholds, create vectors, append audit records, promote memory,
        mutate the seed, or bypass AGP.
        """
        out = Path(path)
        snapshot = self.threshold_policy_snapshot()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "adapter": self.__class__.__name__,
            "snapshot_version": snapshot["snapshot_version"],
            "path": str(out),
            "read_only": False,
            "threshold_changed": False,
            "automatic_rollback_enabled": False,
            "vector_mutation": False,
            "audit_record_mutation": False,
            "memory_quarantine_unchanged": True,
            "agp_bypass": False,
        }

    def write_commit_audit_export(self, path: str | Path) -> dict[str, Any]:
        """Write the Round60 commit audit snapshot to JSON.

        This is an explicit export operation only; it does not create vectors,
        promote memory, change thresholds, or touch the fastText seed.
        """
        out = Path(path)
        snapshot = self.export_commit_audit_snapshot()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "adapter": self.__class__.__name__,
            "export_version": snapshot["export_version"],
            "path": str(out),
            "record_count": int(snapshot["record_count"]),
            "read_only": False,
            "vector_mutation": False,
            "memory_quarantine_unchanged": True,
            "agp_bypass": False,
        }

    def commit_eve_specific_vectors(
        self,
        words: Iterable[str] | None = None,
        *,
        context_words: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Explicitly create/update vectors for observed EVE-specific words.

        This is the only vector mutation path introduced in round58. It uses
        ``EveSpecificVectorStore.add_or_update_vector`` and therefore inherits
        the deterministic mean-of-known-fastText-context rule.
        """
        self._commit_calls += 1
        store = self._resolve_store()
        audit = self.audit_commit_gate(words, context_words=context_words)
        if store is None:
            return {
                "adapter": self.__class__.__name__,
                "created": [],
                "skipped": list(audit.get("target_words", []) or []),
                "rejected": list(audit.get("rejected_words", []) or []),
                "reason": "vector_store_missing",
                "audit_report": audit,
                "read_only": False,
            }
        target_words = list(audit.get("eligible_words", []) or [])
        rejected_words = list(audit.get("rejected_words", []) or [])
        ctx = [str(w or "").strip() for w in (context_words or []) if str(w or "").strip()]
        created: list[str] = []
        skipped: list[str] = []
        for word in target_words:
            try:
                ok = bool(store.add_or_update_vector(word, ctx))
            except Exception:
                ok = False
            if ok:
                created.append(word)
                self._vectors_created += 1
            else:
                skipped.append(word)
                self._vectors_skipped += 1
        skipped.extend(rejected_words)
        self._commit_gate_blocks += len(rejected_words)
        event = {
            "source": "explicit_commit",
            "target_words": list(audit.get("target_words", []) or []),
            "eligible_words": list(target_words),
            "rejected_words": list(rejected_words),
            "context_words": list(ctx),
            "known_context_words": list(audit.get("known_context_words", []) or []),
            "created": list(created),
            "skipped": list(skipped),
            "audit_version": audit.get("audit_version"),
        }
        self._record_commit_audit(audit, event_type="commit")
        self._remember(event)
        return {
            "adapter": self.__class__.__name__,
            "created": list(created),
            "skipped": list(skipped),
            "rejected": list(rejected_words),
            "context_words": list(ctx),
            "known_context_words": list(audit.get("known_context_words", []) or []),
            "audit_report": audit,
            "auto_promotion_enabled": self.auto_promotion_enabled,
            "commit_gate_enabled": self.commit_gate_enabled,
            "read_only": False,
        }

    def _remember(self, event: dict[str, Any]) -> None:
        if self.context_cap <= 0:
            return
        self._events.append(deepcopy(event))
        if len(self._events) > self.context_cap:
            del self._events[: len(self._events) - self.context_cap]

    def stats(self) -> dict[str, Any]:
        tracker = self._resolve_tracker()
        store = self._resolve_store()
        tracker_stats = tracker.stats() if tracker is not None and hasattr(tracker, "stats") else {}
        store_stats = store.stats() if store is not None and hasattr(store, "stats") else {}
        return {
            "adapter": self.__class__.__name__,
            "implementation_phase": self.implementation_phase,
            "round": 71,
            "latest_round": 71,
            "consolidation_version": self.consolidation_version,
            "freeze_baseline_version": self.freeze_baseline_version,
            "self_learning_v1_freeze_policy_snapshot_version": self.freeze_baseline_version,
            "automatic_rollback_enabled": False,
            "active_policy_version": self.active_policy_version,
            "auto_observe_enabled": self.auto_observe_enabled,
            "auto_promotion_enabled": self.auto_promotion_enabled,
            "observe_calls": int(self._observe_calls),
            "commit_calls": int(self._commit_calls),
            "audit_calls": int(self._audit_calls),
            "commit_gate_enabled": bool(self.commit_gate_enabled),
            "min_observations_for_commit": int(self.min_observations_for_commit),
            "min_known_context_words_for_commit": int(self.min_known_context_words_for_commit),
            "commit_gate_blocks": int(self._commit_gate_blocks),
            "vectors_created": int(self._vectors_created),
            "vectors_skipped": int(self._vectors_skipped),
            "event_log_size": len(self._events),
            "commit_audit_record_count": len(self._commit_audit_records),
            "commit_audit_export_version": "v3_round60_commit_audit_export",
            "commit_audit_dashboard_version": "v3_round61_commit_audit_dashboard",
            "commit_threshold_dry_run_version": "v3_round62_multi_observation_commit_threshold_dry_run",
            "commit_threshold_readiness_version": "v3_round62_commit_threshold_readiness",
            "commit_threshold_readiness": self.commit_threshold_readiness_snapshot(),
            "commit_threshold_proposal_version": "v3_round63_threshold_proposal_report",
            "commit_threshold_proposal": self.threshold_proposal_report(),
            "commit_threshold_enforcement_version": self.active_policy_version,
            "context_diversity_gate_enforcement_version": self.context_diversity_gate_enforcement_version,
            "context_diversity_gate_enabled": bool(self.context_diversity_gate_enabled),
            "context_diversity_rollback_drill_version": self.context_diversity_rollback_drill_version,
            "context_diversity_rollback_drill": self.context_diversity_rollback_drill(),
            "context_diversity_blocked_candidate_report_version": self.blocked_candidate_report_version,
            "context_diversity_blocked_candidate_report": self.context_diversity_blocked_candidate_report(),
            "commit_threshold_policy_snapshot_version": self.threshold_policy_snapshot_version,
            "commit_threshold_policy_snapshot": self.threshold_policy_snapshot(),
            "observation_evidence_quality_version": self.observation_evidence_quality_version,
            "observation_evidence_quality": self.observation_evidence_quality_summary(),
            "context_diversity_gate_dry_run_version": self.context_diversity_gate_dry_run_version,
            "context_diversity_gate_dry_run": self.dry_run_context_diversity_gate(),
            "context_diversity_proposal_version": self.context_diversity_proposal_version,
            "context_diversity_proposal": self.context_diversity_proposal_report(),
            "commit_audit_dashboard": self.commit_audit_dashboard_snapshot(),
            "self_learning_v1_freeze_policy_snapshot": self.self_learning_v1_freeze_policy_snapshot(),
            "recent_events": deepcopy(self._events[-10:]),
            "tracker_available": tracker is not None,
            "vector_store_available": store is not None,
            "tracker_eve_specific_count": int(tracker_stats.get("eve_specific_count", 0) or 0),
            "vector_store_count": int(store_stats.get("stored_count", 0) or 0),
            "policy": "round76_self_learning_v1_frozen_round57_75_observe_continuously_commit_explicitly_threshold_2_context_diversity_enforced_no_auto_promotion",
            "read_only": True,
        }
