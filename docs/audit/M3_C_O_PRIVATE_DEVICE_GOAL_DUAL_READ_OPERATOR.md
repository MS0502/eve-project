# M3-C-O Private-Device Goal Dual-Read Operator Preflight

## Status

This slice implements the default-absent single-use operator required after the
accepted M3-C-N bounded-window contract. It does not pin an active operator,
issue a private-device command, access an existing private database, or create a
real observation window.

```text
default-absent operator implementation: true
explicit operator command: true
active exact implementation pin: false
active local authorization digest: false
actual private-device execution: false
default runtime integration: false
existing M3-C-J database access: false
raw private text retained in digest store: false
event append: false
legacy goal authority transfer: false
legacy migration authorization: false
action / scheduler / speech authorization: false
M3-E authority: false
```

## Accepted prerequisite

PR #238 is immutable and reused without rerun.

```text
PR:                    #238
base:                  1ebd3e27ad4582c67b9b2f072ebd58c625af2057
exact head:            b3f599883b9101d7c3b0609fe0680ba4511784d8
exact run:             30637887864
focused:               11 passed
M0 invariance:         byte-identical
M2-B:                  valid; errors 0
full:                  3,369 passed
forward gate:          0 / 0 / 0 / 0
artifact:              exact-head-validation-b3f599883b9101d7c3b0609fe0680ba4511784d8
artifact SHA-256:      8df56600f347e0d73ab07d8251469eb678542de9d4b405ca723d05533229ab0c
M2-E run:              30637887888
M2-E:                  6/6 passed
merge SHA:             9a26f6040679013066425887c3bcee5a2846a025
main after merge:      identical
```

The branch already contains
`docs/audit/M3_C_N_PR238_VALIDATION_REUSE_PIN.json`. Chat, shell, branch, PR,
Draft/Ready, review, or metadata changes do not invalidate #238 or any accepted
phone, retained-sequence, readiness, database, backup, restore, or completed
private-device evidence.

## Architecture

M3-C-O keeps the accepted implementation layers separated:

1. M3-C-M remains the immutable production-origin in-memory shadow tap.
2. M3-C-N remains the immutable bounded policy, path, rollback, record-chain,
   authorization, and receipt contract.
3. M3-C-O adds one explicit single-use operator around those contracts.
4. A later isolated pin slice must bind the accepted M3-C-O implementation head
   and one concrete locally reviewed authorization packet.
5. A still later private-device command may run exactly once.

The operator does not alter `main.build_full_engine()` or the default
`GoalAdapter` constructor. The checked-in engine still creates no tap. Only the
explicit command may build an engine, verify the seam is empty, inject one
bounded collector for one synchronous call sequence, and restore the prior value
in `finally`.

## Local reviewed package

The canonical private package contains:

- one `BoundedDualReadWindowAuthorizationPacket`;
- the accepted 4–16 observation policy;
- one path-bound rollback plan;
- an exact legacy mapping table;
- 4–16 reviewed production probes;
- one local human-review artifact.

Each probe contains one private production operation plus genuine M3-C-B
candidate/drive material. It fixes all replay and authority fields false. Raw
category text remains only in the private package. It is never copied into the
digest-only store or public receipt.

The package verifies all of these bindings before execution:

```text
policy digest
rollback digest
path-binding digest
accepted M3-C-M raw evidence digest
M3-C-M compatibility pin digest
legacy mapping digest
reviewed V4 evaluator digest
local human-review artifact digest
reviewer identity
exact future M3-C-O implementation head
```

## Exact pin boundary

The implementation file contains two checked-in values:

```text
_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = None
_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = None
```

The command calls `active_reviewed_operator_pin()` immediately after clean
checkout verification and before resolving or reading a private path. Therefore
this preflight PR cannot execute a real window.

A later pin PR must keep this implementation immutable, place the reviewed
constants in an isolated adapter, verify a concrete local packet digest, open the
two values only for one synchronous call, and restore them in `finally`. A
descendant launch head must be attested separately from immutable implementation
provenance.

## Path separation and prior-evidence exclusion

The operator uses exactly four active private paths:

1. canonical reviewed package file;
2. new digest-only working store;
3. new empty-baseline backup;
4. new separate restore file.

The caller must also provide the complete sorted set of prior private path
digests. `PrivateDeviceWindowPathBinding` rejects:

- duplicate active paths;
- an empty prior-path digest set;
- unsorted or duplicate prior digests;
- any active path whose digest overlaps a prior path.

The script also requires all paths to be absolute and outside the repository.
The working store, backup, and restore files must be absent. Existing files are
never deleted, repaired, overwritten, or reused to make a rerun possible.

This blocks the existing M3-C-J database, WAL/SHM/journal sidecars, bundle,
journal, backup, and restore evidence without reading any of them.

## Production-origin execution

After all exact authorization, path, package, and baseline checks pass, the
operator performs this bounded sequence:

1. capture the current legacy goal baseline digest;
2. require it to match the reviewed rollback plan;
3. create a new canonical JSONL store with an empty-baseline header;
4. copy and verify the empty-baseline backup;
5. construct the accepted M3-C-M tap with the reviewed mapping and evaluator;
6. wrap it in an in-memory bounded digest collector;
7. inject that collector into the existing `GoalAdapter` seam;
8. execute each reviewed `goal_set` or `tick` through `GoalAdapter._goal_call`;
9. require the legacy callable to remain authoritative and execute exactly once;
10. convert each successful comparison into one M3-C-N digest-only record;
11. append only that record mapping to the new private JSONL store;
12. restore the prior GoalAdapter seam in `finally`;
13. evaluate the complete 4–16 record chain with M3-C-N;
14. append the digest-only window receipt;
15. restore the empty baseline backup into a separate path and verify its hash;
16. return an operator receipt and public-safe window receipt.

There is no retry. If a comparison fails after the authoritative legacy call,
the operator aborts and preserves any partial store as evidence.

## Evaluator boundary

`ReviewedV4GoalEvaluator` is not a generator or new authority. It is a
deterministic projection bound to the exact private probe digests. For each
production observation it:

- resolves exactly one reviewed probe by source-observation digest;
- runs genuine M3-C-B goal selection;
- runs genuine M3-C-C lifecycle evaluation when selected;
- projects the same before/after structural state digests;
- returns an M3-C-L shadow-only observation.

It performs no persistence, event append, action, scheduling, speech, legacy
mutation, authority transfer, migration, or M3-E work.

## Digest-only retention

The working store contains only:

- package, authorization, path, baseline, record, and receipt digests;
- M3-C-N record mappings;
- the M3-C-N window receipt;
- fixed schema and stage identifiers.

It contains no raw category, raw goal text, raw path, mapping contents, candidate
contents, drive samples, review statement, evaluator source, nonce, or existing
private evidence.

The operator receipt records:

```text
explicit operator injection performed: true
default runtime integration performed: false
existing M3-C-J database accessed: false
raw private text retained in store: false
event append performed: false
legacy goal authority transferred: false
legacy migration authorized: false
action / scheduler / speech authorized: false
M3-E authority open: false
```

A clean result only sets `human_gate_review_eligible=true`. It does not authorize
migration or transfer goal authority.

## Single-use and cross-chat duplicate prevention

The first completed or partial attempt leaves at least one target file present.
Every subsequent command refuses before another production probe. A new chat,
shell, process, branch, PR state, or operator session is never a retry
authorization.

The following remain permanently prohibited:

- rerun PR #238 or any accepted prerequisite because work continued;
- rerun phone witness #211;
- rerun retained sequences 1–5;
- rerun controlled-input readiness;
- rerun the completed M3-C-J private-device operator;
- delete or recreate existing private database or sidecar evidence;
- claim synthetic temporary-path tests as real private observations;
- run M3-C-O before a later exact pin and concrete local review;
- publish raw private goal text or paths;
- infer migration authorization from a clean receipt.

## Focused-test boundary

Focused tests use only synthetic in-memory legacy owners and pytest temporary
paths. They prove:

- checked-in pins remain absent;
- canonical package round-trip and exact digest bindings;
- prior-path overlap fails closed;
- four production-origin legacy calls create four digest-only records;
- the legacy callable executes exactly once per probe;
- the GoalAdapter seam is restored;
- raw private categories do not appear in the store;
- baseline backup and separate restore are byte-identical;
- a blocking divergence remains retained but not review-eligible;
- an existing store blocks a cross-chat rerun;
- wrong exact pins and downstream authority fail before a legacy call;
- the script checks authorization before private path reads;
- #238 and all earlier reuse prohibitions remain durable.

These tests are not private-device evidence.

## Promotion boundary

After this preflight is exact-head validated and merged, the next target is an
isolated M3-C-O exact operator pin bound to:

- the accepted final M3-C-O implementation head and artifact;
- one concrete canonical private package digest;
- one local human-review artifact digest;
- the exact path-binding and rollback digests;
- the accepted mapping and evaluator digests.

That pin still must not issue the command. One actual private-device execution
must be separately launched from a clean reviewed descendant checkout. Its
public receipt then requires another human-gate review before any M3-C-K
migration candidate can be proposed.

## Acceptance ruling

M3-C-O is acceptable only as a default-absent explicit operator preflight. It
adds no default runtime behavior, reads no existing private evidence, retains no
raw private material, leaves legacy as the sole goal authority, and keeps every
migration and downstream authority gate closed.
