# M3-C-D Lifecycle Event-Envelope / Replay-Reducer Preflight

Baseline: `e9e2c4598d7d0042c3c6fd78f61804b23fea163f` — PR #220 squash merge.

This is an in-memory preflight only. It converts an immutable M3-C-C lifecycle
transition candidate into a canonical event-envelope candidate and replays
ordered candidates into an immutable reducer snapshot. It does not import or
call EventKernel, SQLite, a file writer, or any production loop.

This M3-C-D slice is not the separate M3-D project milestone. M3-D remains
closed.

## Authority boundary

```text
M3 authority open:                    true — #215
M3-C-A design merged:                 true — #217
M3-C-B selection kernel merged:       true — #219
M3-C-C lifecycle kernel merged:       true — #220
M3-B retained observations:           5/37 — #218
legacy goal-domain authority:         unchanged
authoritative event append:           false
persistence write:                    false
production lifecycle integration:     false
action/scheduler/speech authority:     false
M3-E authority open:                  false
```

## Event-envelope candidate

`build_event_envelope_candidate()` accepts only an immutable, event-eligible
`GoalLifecycleTransitionCandidate` that does not already claim an append.
It fixes:

```text
schema:     eve.m3-c-d.goal-lifecycle-event-envelope-candidate.v1
event type: m3c.goal_lifecycle_transition
stream:     m3c.goal_lifecycle
producer:   eve.m3-c-d.goal-lifecycle-event-preflight.v1
authority:  candidate_only
```

The event id is deterministically derived from the full transition id:

```text
m3c:goal-lifecycle:<64-character transition id>
```

The payload digest is SHA-256 of the canonical transition mapping. The envelope
digest is SHA-256 of the complete canonical envelope-candidate mapping.

Every candidate fixes:

```text
append_authorized=false
append_performed=false
persistence_write_performed=false
production_integration_performed=false
action_authorized=false
scheduler_authorized=false
speech_authorized=false
legacy_goal_authority_transferred=false
m3_e_authority_open=false
```

## Immutable replay snapshot

`GoalLifecycleReducerSnapshot` contains:

- candidate-id keyed immutable lifecycle states;
- candidate-id keyed last logical steps;
- ordered applied transition ids;
- canonical snapshot digest.

The reducer fail-closes unless:

- transition identity has not already been applied;
- candidate semantic goal, decision epoch, and evidence identity match;
- current lifecycle state equals the transition before-state;
- current last transition id equals the declared prior transition id;
- logical step advances monotonically for that candidate.

On success it derives a new immutable snapshot and a reducer receipt. The
receipt proves only in-memory replay and fixes all external effect/authority
flags to false.

## Deterministic replay proof

`replay_event_candidates_in_memory()` applies an ordered sequence without any
external write. The tests prove:

- identical transition -> identical envelope and envelope digest;
- `absent -> proposed -> validated -> eligible -> selected` ordered replay;
- full replay and checkpoint/resume reach the same snapshot digest;
- repeating the same full replay yields identical reducer receipt digests;
- duplicate transition candidates fail closed;
- reversed/out-of-order candidates fail on before-state;
- wrong prior transition identity fails closed;
- non-monotonic logical step fails closed;
- snapshot mappings are immutable;
- event/reducer candidates cannot assert append or downstream authority;
- no I/O, EventKernel, SQLite, runtime, action, scheduler, speech, or M3-E
  import/call surface exists.

## A9 no-duplicate boundary

This preflight does not emit continuously. A transition candidate already
exists only for a named M3-C-C lifecycle edge. The event id, payload digest,
envelope digest, snapshot digest, and reducer receipt digest are deterministic.
The reducer keeps applied full transition ids and refuses duplicates.

This demonstrates the material needed for a later append gate, but does not
itself authorize or execute the append.

## Deferred gates

Not implemented here:

- conversion into the repository's authoritative EventEnvelope type;
- EventKernel or SQLite append;
- durable snapshot/checkpoint write;
- live replay bootstrap;
- production candidate discovery/lifecycle integration;
- legacy goal-domain migration;
- action, scheduling, speech, or M3-E authority.

A later separately reviewed activation slice must bind the envelope candidate
to the authoritative substrate and prove rollback before any live writer opens.
