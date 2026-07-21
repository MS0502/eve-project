# M1 Extended Controlled Observation Registration

Baseline: `847621bcd61634958ce505108ade491c50ced0d4`

Status: **registered evidence-generation scope only; no human acceptance or authority change**

## Purpose

This package expands the bounded M1 controlled window from one adapter call path
to a mechanism-focused set sufficient for human review. It does not use a
historical-site coverage fraction as an M1 gate.

The package must demonstrate all of the following from raw observations:

1. M0-A mutation forms: attribute assignment, subscript assignment, augmented
   assignment, mutating method call, and direct write.
2. Multiple adapter paths spanning at least `WRAP` and `REWRITE` M0-D
   dispositions.
3. One mutation while an actual `LiveLoop` tick thread is alive.
4. Complete event replay, complete divergence ledger, visible retained-call
   failure, visible observer-only failure, and no event amplification from
   continuous tick/decay.
5. A raw artifact sufficient to independently recalculate every metric claimed
   by the rendered evidence report.

## Reviewed controlled targets

| Target | Module | Source range | M0-D disposition | Stream |
|---|---|---:|---|---|
| `ActivationAdapter.learn_pair` | `adapters/activation_adapter.py` | 103-105 | `WRAP` | `shadow:legacy.activation.learn_pair` |
| `LiveLoop._drain_user_inputs` | `adapters/live_loop.py` | 68-77 | `REWRITE` | `shadow:legacy.live_loop.drain_user_inputs` |
| `PersistenceAdapter.save` | `adapters/persistence_adapter.py` | 54-80 | `REWRITE` | `shadow:legacy.persistence.save` |

These additional targets exist only in the controlled campaign's local registry.
They are not added to the default production observer registry.

## Direct-write boundary

The `PersistenceAdapter.save` scenario performs a real sidecar write under a
fresh temporary root. The retained v40 save callable is replaced only for the
bounded call with a deterministic no-write implementation so the test cannot
write through the legacy persistence backend. An unobserved baseline executes
the same controlled call in a second temporary root. Both roots must be removed
before the campaign returns.

This is direct-write mechanism evidence. It is not persistence activation,
recovery validation, durability validation, or M2 cutover evidence.

## Concurrency boundary

The campaign starts the actual `LiveLoop` thread and blocks its first hormone
update at a deterministic event barrier. A registered activation mutation is
then observed while that thread remains alive. The barrier is released and the
thread is stopped and joined before the campaign returns.

This is a single controlled concurrency probe. It does not authorize scheduling,
production observer installation, or runtime ownership transfer.

## Artifact contract

The generator produces:

- `docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json`
- `docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_EVIDENCE.md`

The Markdown report pins the raw JSON SHA-256. Every mutation-form row names
its changed state field and stores exact before/after values plus a transition
digest. Focused tests regenerate the campaign, require byte-identical canonical
JSON, recalculate event/replay/failure and granularity totals, and require the
committed report to equal the renderer output for that raw hash.

Generated artifact pins:

```text
raw artifact SHA-256: 3618b948cb2e864741412713b5c724632ae9fd72a214479b970d8c4aeeafcaac
source evidence SHA-256: 06984c653ed2a655f45c7cb27d0777b1c93c6aee872f2cb9c7d1f5a898d9af86
```

The one-shot bootstrap workflow was removed in the generated evidence commit.
The final permanent exact-head validation therefore runs against only the six
reviewed package files.

## Gate boundary

This PR must leave all of the following unchanged:

```text
human_accepted: false
v4_2_eligible: false
runtime_integrated: false
production observer installed: false
production persistence enabled: false
legacy runtime authority: unchanged
```

A passing expanded window makes the package eligible for a separate human
approval record. It does not itself close M1, open v4.2, or begin M2.

Coverage of historical mutation sites remains deferred to the A2/M2 dual-read
and cutover gates. Every unobserved site remains tracked debt and is not claimed
safe.
