# M2-B Read-Capability Technical Review

## Status

- PR: `#162`
- Base: `2577346d0a95eafa0456c4f6816d25c37edf95a2`
- Authority: `audit_only`
- Runtime integration: `false`
- Human accepted: `false`
- PR state required before independent acceptance: `Draft`

M2-A was accepted and squash-merged by PR #161 at main commit `2577346d0a95eafa0456c4f6816d25c37edf95a2`. That acceptance remains limited to the disconnected SQLite `shadow_only` store. It did not authorize runtime activation, dual read, authoritative recovery, cutover, or legacy-authority transfer.

## Superseded technical result

The earlier PR head `a11c439bfca1912503cd6a9c33e66e1e239e3ba0` reported `unresolved boundary calls = 0`. Independent review found that result invalid for two reasons:

1. the unresolved-call insertion branch was unreachable because only already-resolved destinations entered `next_calls`;
2. taint was not propagated through calls on tainted receivers, so expressions such as `input_text.strip()` and `json.dumps(body).encode()` could drop taint before a later sink.

The previous candidate, capability-surface, validation, and artifact digests are retained in PR history as superseded evidence only. They must not be used for M2-B acceptance.

## Corrected mechanical extraction

The canonical extractor is:

```text
scripts/audit/m2_b_read_capability_manifest.py
```

It parses tracked Python source without importing or executing runtime modules. The corrected v3 schema:

- propagates taint through positional arguments, keyword arguments, and a tainted method receiver;
- follows resolvable project-local calls;
- treats `say` as an expression sink while avoiding receiver-name and predicate-name sink false positives;
- emits every unresolved tainted call as exact fail-closed review evidence;
- binds source evidence, call path, call-site location, tainted receiver/argument shape, and sink or unresolved boundary into content-addressed IDs;
- cannot grant capability, runtime integration, or human acceptance.

Corrected technical-review surface:

```text
candidate report digest:
aad7a16cd537f723083866a48c31d6ebb1e09a140b42ca9576acc4cb7d57b1ba

capability-surface digest:
46076350ae4a71e01a3c1c0241831871a1ef6fabe4d2d79de1938283463b83df

candidate edges:                         381
legacy rewrite edges:                    378
not-raw-text false-positive edges:         3
unresolved tainted calls:               3,738
  legacy rewrite / exact remap required: 2,340
  exact non-capability boundaries:       1,361
  denied opaque/external boundaries:        37
parse-error blockers:                        2
```

The three edge false positives remain the two outbound OpenAI-compatible response serialization helpers and one offline autonomous-handoff restoration report. They do not establish inbound runtime raw-text read capability.

The two syntactically unparseable legacy foundation files remain explicitly `DENIED_NO_CAPABILITY`. Import or activation remains blocked until each file is parseable, mapped, and separately reviewed.

## Unresolved-call treatment

`unresolved` no longer means unreviewed or silently absent. Every exact finding is assigned one of three fail-closed technical dispositions:

- `LEGACY_REWRITE`: a project-local cognition, memory, learning, state, or dynamically unresolved legacy boundary that must be resolved or rewritten before activation;
- `NOT_CAPABILITY_BOUNDARY`: an exact local construction, deterministic transform, mapping/regex read, or temporary collection operation; this ruling applies only to the listed finding IDs and cannot be generalized by method name;
- `DENIED_NO_CAPABILITY`: an opaque dynamic, filesystem, network, process, archive, audio, queue, or output-adjacent boundary with no approved capability.

No unresolved finding grants runtime raw-text access. M2-C may not use a denied or rewrite-required boundary as an activated read path.

## Decision validation

`docs/audit/M2_B_READ_CAPABILITY_DECISION_GROUPS.json` preserves exact coverage of every edge and finding ID. `scripts/audit/m2_b_decision_groups.py` requires:

- the corrected decision-group schema;
- exact capability-surface digest equality;
- complete, non-stale, non-duplicate edge, unresolved-call, and parse-error coverage;
- reviewed fields and allowed decision vocabularies;
- `human_accepted=false` and `authority=audit_only`.

Corrected technical validation digest:

```text
716ffaa65fc9ad45fd93269c3dd3fa06ede73fe49ec923a58650e407016b34d6
```

A valid machine result may set only:

```text
eligible_for_human_review = true
human_accepted = false
authority = audit_only
```

It cannot mark the PR Ready, merge itself, install a runtime reader, connect a raw source to expression, or authorize runtime activation, authoritative recovery, cutover, scheduler/model/vector activation, or legacy-authority transfer.

## Focused verification

The corrected local focused set is:

```text
30 passed
```

It includes regression evidence for receiver-taint propagation, exact `say` sink handling, sink-name false-positive prevention, unresolved-call emission, exact unresolved decision coverage, invalid decision rejection, and capability-surface digest pinning.

## Remaining gate

A new exact-head workflow run and artifact are required after the corrected patch is committed. Separate independent human acceptance remains required after that exact-head artifact is reviewed. Until then PR #162 remains Draft, M2-B is not accepted, and M2-C plus all runtime activation work remain blocked.
