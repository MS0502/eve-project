# M3-C-L Goal Dual-Read Comparator Preflight

## Status

Pure read-only implementation preflight under the accepted M3-C-K migration
gate. It uses synthetic or immutable fixture observations only.

```text
production runtime hook: false
production/private database access: false
legacy goal mutation: false
event append: false
persistence write: false
legacy goal-domain authority transferred: false
legacy migration authorized: false
action/scheduler/speech authority: false
M3-E authority open: false
```

## Exact prerequisite

```text
PR:                    #235
exact head:            06a6495089fab4bf7e30ffb5a79180c4b748b6d2
exact run:             30618763444
focused:               No focused tests selected
M0 invariance:         byte-identical
M2-B:                  valid; errors 0
full:                  3,332 passed
forward gate:          0 / 0 / 0 / 0
artifact:              exact-head-validation-06a6495089fab4bf7e30ffb5a79180c4b748b6d2
artifact SHA-256:      e55a84e0e3bc7d96f1e8e73fd0b8144a7f57077759c61d1dcce27ccaeb8f11fc
M2-E run:              30618765141
M2-E:                  6/6 passed
merge SHA:             d9c1cf8f615872b6a59ea7e950ccb9ceeb629133
```

PR #235 and every earlier exact-head, phone, retention, readiness, database,
backup, restore, and private-device execution remain immutable reused evidence.
A chat, branch, PR, review, Draft/Ready, shell, or operator-session change is not
an invalidator and never authorizes a rerun.

## Purpose

M3-C-K requires a deterministic comparison surface before any production-origin
shadow tap may exist. M3-C-L implements only that isolated comparison surface.
It does not discover legacy state, call legacy `GoalManagement`, enter runtime
orchestration, open the M3-C-J database, or retain comparison records.

The comparator consumes:

1. one immutable legacy observation fixture;
2. one v4 shadow observation built from genuine M3-C-B selection and M3-C-C
   lifecycle receipts;
3. optionally, one exact versioned mapping rule.

It emits one immutable canonical comparison receipt.

## Legacy observation contract

`LegacyGoalObservation` binds:

- one comparison-input digest;
- one source-observation digest;
- canonical legacy goal code;
- normalized semantic goal identity or explicit absence;
- normalized lifecycle state or explicit absence;
- decision epoch;
- before-state digest;
- after-state digest;
- structural-manifest digest;
- fixed `legacy_authoritative` authority.

`state_changed` is derived only from the before/after digest inequality. It is
not a caller-supplied boolean.

The fixture represents the future order in which legacy executes exactly once
as the sole behavior authority. M3-C-L does not perform that execution itself.

## V4 shadow observation contract

`V4ShadowGoalObservation` accepts only real immutable kernel receipts:

- `GoalSelectionReceipt` from M3-C-B;
- when a candidate is selected, `LifecycleEvaluationReceipt` from M3-C-C.

The comparator derives the v4 semantic goal identity from the selected
candidate's unique scored-candidate record. It derives lifecycle state from the
M3-C-C transition after-state, or from the unchanged lifecycle state when no
transition exists. A caller cannot directly inject either result.

Selection/lifecycle candidate identity, semantic identity, and decision epoch
must match exactly. A selected candidate without a lifecycle receipt fails
closed. A no-candidate selection cannot carry a lifecycle receipt.

Every v4 observation fixes:

```text
authority: shadow_only
production integration: false
persistence write: false
event append: false
legacy mutation: false
action/scheduler/speech: false
legacy authority transfer: false
M3-E: false
```

## Comparison verdicts

The exact closed catalog is:

```text
exact_equivalent
mapped_equivalent
expected_design_difference
unexplained_divergence
legacy_only_behavior
v4_only_behavior
comparison_unavailable
```

### exact_equivalent

Legacy and v4 semantic goal identities and lifecycle states are byte-identical.
No mapping rule is allowed because identity already proves the result.

### mapped_equivalent

A versioned rule matches the complete observed tuple:

```text
legacy goal code
legacy semantic goal
legacy lifecycle state
v4 semantic goal
v4 lifecycle state
```

A partial or stale rule fails closed.

### expected_design_difference

The same exact tuple is covered by a separately reviewed rule whose ruling is
`expected_design_difference` and whose rationale is a canonical internal code.
This ruling is never inferred from similarity or convenience.

### unexplained_divergence

Both paths produced results, they are not identical, and no exact matching rule
exists. This is the fail-closed default for unknown differences.

### legacy_only_behavior / v4_only_behavior

Exactly one side produced a goal result. A mapping rule is forbidden because it
cannot manufacture the missing observation.

### comparison_unavailable

The v4 fixture explicitly records an unavailable evaluation and a canonical
reason code. It carries no kernel receipts and consumes no mapping rule.

## Canonical identity and fidelity

Every observation, mapping rule, and comparison receipt uses sorted compact
UTF-8 JSON plus SHA-256. NaN, raw free-form goal text, noncanonical identifiers,
malformed digests, mixed decision epochs, mismatched source/input identity, and
selection/lifecycle disagreement fail closed.

The comparison receipt retains:

- both observation digests;
- mapping-rule digest when applicable;
- exact normalized identities and states;
- derived legacy actual-state change;
- derived v4 projected-state change;
- comparison availability;
- the derived verdict;
- explicit false flags for every effect and authority.

## No false transfer

A green comparison receipt is not migration evidence by itself. It cannot:

- install a production hook;
- read or write SQLite;
- mutate legacy state;
- append lifecycle or comparison events;
- transfer legacy goal authority;
- authorize migration;
- route actions;
- schedule activity;
- generate speech;
- mutate memory, drives, affect, or hormones;
- open M3-E.

No count or mixture of fixture receipts changes that boundary.

## Failure behavior

| Failure | Result |
|---|---|
| malformed digest or identifier | raise fail-closed comparison error |
| partial semantic/lifecycle pair | refuse observation |
| v4 receipt unavailable but supplied | refuse observation |
| selected candidate lacks lifecycle receipt | refuse observation |
| selection/lifecycle identity or epoch mismatch | refuse observation |
| comparison input or source digest mismatch | refuse comparison |
| supplied mapping rule does not match exact tuple | refuse comparison |
| unknown difference with no rule | `unexplained_divergence` |
| attempted effect or authority flag | refuse construction |

## Focused proof

The focused suite proves:

1. exact seven-verdict catalog;
2. deterministic exact equivalence from real M3-C-B/C receipts;
3. exact-rule-only mapped equivalence;
4. explicit-rule-only expected design difference;
5. unknown difference becomes `unexplained_divergence`;
6. distinct legacy-only, v4-only, no-goal, and unavailable outcomes;
7. before/after-derived state-change fidelity;
8. input/source/rule mismatch refusal;
9. selection/lifecycle identity mismatch refusal;
10. raw-text-like identifier and authority escalation refusal;
11. immutable deterministic receipt identity;
12. zero I/O, persistence, runtime, action, scheduler, or speech surface.

## Promotion boundary

M3-C-L is not a production observer. After exact acceptance, the next separately
reviewed target is **M3-C-M dormant production-origin shadow tap**. That later
slice must remain unreachable by default behind absent exact-reviewed
implementation and authorization pins. It still cannot transfer legacy goal
authority or open M3-E.

## Acceptance ruling

M3-C-L is accepted only as a pure fixture comparator preflight. Legacy remains
the sole goal-domain behavior authority. Production integration, observation,
retention, migration, authority transfer, action, scheduling, speech, and M3-E
remain closed.
