# M3-C-C Pure Goal-Lifecycle Transition-Candidate Kernel

Baseline: `3a09a6ddd2f3b64d5483fd8564be0e645043538f` — PR #219 squash merge.

Status: isolated implementation candidate. This slice consumes immutable M3-C-B
score/selection evidence and derives at most one exact M3-C-A lifecycle edge.
It does not append an event, write persistence, integrate with production,
execute an action, schedule work, generate speech, transfer legacy goal-domain
authority, or open M3-E.

## Authority boundary

```text
M3 authority open:                    true — #215
M3-C-A design complete:               true — #217
M3-C-B pure selection kernel merged:  true — #219
M3-B retained observations:           5/37 — #218
legacy goal-domain authority:         unchanged
production lifecycle integration:     false
event append/persistence write:       false
action/scheduler/speech authority:     false
M3-E authority open:                  false
```

A transition returned by this kernel is an immutable **transition candidate**.
`event_eligible=true` reflects the #217 lifecycle table only. It is not proof
that an event was appended or that persistent lifecycle state advanced.

## Exact lifecycle catalog

The module `core/m3_c_c_goal_lifecycle_kernel.py` fixes the fourteen reviewed
#217 edges:

```text
absent -> proposed
proposed -> validated | rejected | expired
validated -> eligible | rejected
eligible -> selected | withdrawn
selected -> superseded | expired
rejected | expired | withdrawn | superseded -> absent
```

No self-loop or unlisted edge is accepted. One evaluation derives at most one
edge, so a `proposed` candidate cannot become `eligible` in a single logical
step.

## Input contract

`GoalLifecycleState` carries:

```text
candidate_id
semantic_goal_id
decision_epoch
evidence_digest
lifecycle_state
last_transition_id (optional)
```

`LifecycleEvidence` carries one exact M3-C-B `CandidateScore`, non-negative
logical step, freshness, validation status, permanent-selection-failure flag,
terminal acknowledgement, and optional immutable M3-C-B selection receipt.

When a selection receipt is provided, this kernel fail-closes unless:

- decision epochs match;
- the candidate occurs exactly once in the scored set;
- candidate identity and semantic goal match;
- the score matches within fixed `1e-12` absolute tolerance.

## Predicate behavior

```text
absent:
  fresh and score >= 0.20 -> proposed

proposed:
  stale -> expired
  validation failed -> rejected
  validation passed -> validated

validated:
  permanent selection failure -> rejected
  fresh and score >= 0.30 -> eligible

eligible:
  score <= 0.10 -> withdrawn
  matching transition-eligible initial/switch receipt -> selected

selected:
  stale -> expired
  matching switched-selection receipt naming this prior winner -> superseded

terminal:
  explicit acknowledgement -> absent
```

All other evaluations return an immutable no-transition receipt.

## Transition-candidate identity and A9 boundary

A named transition candidate contains its exact before/after states, logical
step, candidate/evidence identities, score, prior transition id, optional
selection-receipt digest, trigger code, and exact predicate/schema versions.
Its `transition_id` is SHA-256 over canonical sorted compact mapping.

This establishes:

- same state + same evidence + same logical step -> same transition id;
- unchanged pending/retained state -> no transition candidate;
- exactly one allowed edge per evaluation;
- no continuous score/drive event;
- no event append or persistent-state mutation in this module;
- a later append gate can reject duplicate transition identity.

Every transition/evaluation output fixes:

```text
event_append_performed=false
persistence_write_performed=false
production_integration_performed=false
action_authorized=false
scheduler_authorized=false
speech_authorized=false
legacy_goal_authority_transferred=false
m3_e_authority_open=false
```

## Focused acceptance

Focused tests cover:

- exact fourteen-edge catalog;
- proposal enter threshold and one-edge-per-step rule;
- all three `proposed` exits;
- both `validated` exits;
- exact matching selection receipt for `eligible -> selected`;
- proposal exit threshold withdrawal;
- selected supersession and expiry;
- all four terminal acknowledgements;
- deterministic unchanged no-transition receipt digest;
- candidate/receipt mismatch fail-closed behavior;
- all downstream authority/effect flags false;
- no I/O, event append, runtime, or legacy import surface;
- exact #217 design-boundary text remains present.

## Deferred gates

Not implemented here:

- event-envelope construction;
- authoritative SQLite/event-kernel append;
- snapshot/replay reducer integration;
- production semantic-candidate discovery;
- legacy goal-domain migration or cutover;
- action, scheduling, or speech activation;
- M3-E affect authority.

The next legitimate slice after acceptance is an event-envelope/reducer
preflight that remains disconnected from live persistence and production.
