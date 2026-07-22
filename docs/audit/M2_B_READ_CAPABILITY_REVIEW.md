# M2-B Read-Capability Technical Review

## Status

- PR: `#162`
- Base: `2577346d0a95eafa0456c4f6816d25c37edf95a2`
- Authority: `audit_only`
- Runtime integration: `false`
- Human accepted: `false`
- PR state required before independent acceptance: `Draft`

M2-A was accepted and squash-merged by PR #161 at main commit `2577346d0a95eafa0456c4f6816d25c37edf95a2`. That acceptance remains limited to the disconnected SQLite `shadow_only` store. It did not authorize runtime activation, dual read, authoritative recovery, cutover, or legacy-authority transfer.

## Mechanical extraction

`script/audit/m2_b_read_capability_manifest.py` is not used. The canonical extractor is:

```text
scripts/audit/m2_b_read_capability_manifest.py
```

It parses tracked Python source without importing or executing runtime modules. It follows concrete tainted argument flows from raw parameters/read calls through project-local call paths to expression or generation sinks. It emits candidate evidence only and cannot grant capability or acceptance.

Exact technical-review surface:

```text
origin candidate report digest:
433860a215f815c9c55f695832b3013c6c8183992d7ab98d5ce4e062369f8bc7

capability-surface digest:
b0ba984be3434a92d50fc209f8e46a987b6f1c92b79dbcb2fe0fa191ee433c38

candidate edges:             137
legacy rewrite edges:        134
false-positive edges:          3
unresolved boundary calls:     0
parse-error blockers:           2
```

The three false positives are two outbound OpenAI-compatible response serialization helpers and one offline autonomous-handoff package restoration report. They are not accepted runtime raw-text read capabilities.

The two syntactically unparseable legacy foundation files are explicitly `DENIED_NO_CAPABILITY`. Import or activation remains blocked until each file is parseable, mapped, and separately reviewed.

## Decision validation

`docs/audit/M2_B_READ_CAPABILITY_DECISION_GROUPS.json` preserves the exact origin report digest and every reviewed edge/finding ID. `scripts/audit/m2_b_decision_groups.py` expands the compact groups and requires exact, non-stale, non-duplicate coverage of the current candidate evidence. Edge IDs already bind source, call path, sink, and evidence locations; parse finding IDs bind path, line, and parse failure.

A valid machine result may set only:

```text
eligible_for_human_review = true
human_accepted = false
authority = audit_only
```

It cannot mark the PR Ready, merge itself, install a runtime reader, connect a raw source to expression, or authorize M2-C and later activation work.

## Remaining gate

Separate independent human acceptance is required after exact-head CI and artifact review. Until then PR #162 remains Draft and all runtime activation, dual-read, authoritative recovery, cutover, model/vector activation, scheduler activation, and legacy-authority transfer remain blocked.
