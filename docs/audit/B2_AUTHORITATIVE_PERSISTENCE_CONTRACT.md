# B2 Authoritative Persistence Contract

## Scope and separation

B2 adds a persistence foundation for accepted EVE event history. It does not
activate semantic workspace, SNN, brian2, a new cognition architecture, or a
raw-text capability edge. It does not upgrade or modify the M2 shadow store.

The stores are deliberately disjoint:

| Property | M2 shadow | B2 authority |
|---|---|---|
| Environment path | `EVE_SQLITE_SHADOW_PATH` | `EVE_AUTHORITY_PATH` |
| Event table | shadow store `events` | `authority_events` |
| Authority | `shadow_only` | `authoritative_accepted_history` |
| Tail metadata | none | append-only `accepted_tail` |
| Crash residue | not authoritative | `event_candidate` only |

`EVE_AUTHORITY_PATH` must resolve to a concrete file and must not resolve to
the shadow path. Import and construction do not create or activate a database.

The existing `EventEnvelope` remains `shadow_only`; B2 does not silently
upgrade the event kernel's M1 contract. Authority is conferred only by a valid
entry in the B2 accepted event chain and the matching accepted-tail chain.
Accepted events are replayed through the unchanged `InMemoryEventKernel`
validation and reducer boundary.

## Schema

The `eve.authoritative-store.v1` schema contains:

- `authority_meta`: immutable store and serialization identity;
- `authority_migrations`: immutable migration history;
- `authority_events`: append-only canonical event bytes, content hash,
  predecessor hash, and deterministic event hash;
- `accepted_tail`: append-only accepted ordinal/event-hash metadata with its
  own hash chain; and
- `event_candidate`: a single-slot, immutable, explicitly unaccepted staging
  row that may be deleted only by the acceptance transaction or proven startup
  residue recovery.

Update and delete triggers protect every accepted table. The candidate table
rejects updates and permits only controlled deletion. A single adjacent
`<database>.writer.lock` OS advisory lock excludes concurrent authoritative
writers and is released by the operating system after process termination.

## Canonical hashing and replay

Canonical JSON is UTF-8, key-sorted, compact, finite, bounded by the existing
event-kernel JSON contract, and tagged `eve.authoritative-event.v1`.

For event ordinal `n`:

1. `content_hash = SHA-256(canonical_event_bytes)`;
2. `event_hash = SHA-256(canonical(schema, n, content_hash, prev_hash))`; and
3. `prev_hash` is the prior `event_hash`, or 64 zeroes at genesis.

Each accepted-tail revision binds the accepted ordinal, accepted event hash,
prior tail hash, and `eve.accepted-tail.v1`. Startup recomputes the complete
event and tail chains, validates canonical bytes and denormalized indexes,
checks per-stream sequences and causation, and runs SQLite `integrity_check`.

## Transaction and durability boundary

Acceptance is a two-commit protocol with an unambiguous authority boundary:

1. Commit one writes a canonical `event_candidate` using `BEGIN IMMEDIATE` and
   `PRAGMA synchronous=FULL`. This row is durable but is not accepted history.
2. Commit two copies that exact candidate into `authority_events`, appends the
   matching `accepted_tail`, and deletes the candidate in one transaction.
3. An event is accepted only after commit two returns and complete readback
   verification succeeds.

A crash before commit one leaves nothing. A crash after commit one but before
commit two can leave only a candidate. A crash during commit two leaves either
the prior accepted state plus the candidate or the new event and tail with no
candidate; SQLite transaction atomicity forbids a half-accepted state.

WAL is requested and its effective journal mode is read back. If WAL is not
available, the only permitted fallback is verified `DELETE` rollback-journal
mode. Fallback can be disabled. Every connection sets and reads back
`synchronous=FULL`; B2 never describes `NORMAL` or `OFF` as durable authority.

## Startup and fail-closed behavior

Startup obtains the writer lock and proves the entire accepted chain before
reads, appends, or cleanup. Accepted event corruption, historical corruption,
accepted-tail corruption, schema drift, SQLite integrity failure, unprovable
candidate data, writer ambiguity, or durability-mode drift raises
`AuthorityUnprovable`. The process entrypoint
`scripts/audit/b2_authority_verify.py` maps these states to exit code **86**.

Accepted rows and tail rows are never automatically repaired, deleted, or
truncated. Startup may remove exactly one candidate only after proving it is
the next unaccepted suffix with valid canonical bytes, hashes, predecessor,
stream sequence, event identifier, and causation. Unprovable residue is
authority ambiguity and exits 86 without mutation.

## Evidence gates

GitHub workflow `b2-authority-smoke.yml` is deterministic CI evidence only. It
runs the focused fault/integrity matrix, process interruption checks, rollback
journal fallback test, single-writer test, and startup CLI. It is not evidence
of Ryzen 7 8840U sustained-load behavior.

`scripts/operator/b2_authority_physical_gate.py` is the separate physical
gate. It requires a clean exact merged checkout, exact `.python-version`, a
healthy installed environment, validation-identity receipt, an actual CPU name
containing `8840U`, at least 1,000 accepted events over at least 60 sustained
seconds, a crash interval no greater than 100 events, and paths outside the
repository. These minimums cannot be reduced with command-line options. Its JSON receipt retains
raw append latencies, RSS samples, restart/crash results, journal/durability
mode, two replay results, final event/tail hashes, database SHA-256, commit/tree
identity, and a receipt SHA-256.

`scripts/operator/b2_authority_establish.py` accepts only a green physical
receipt bound to the exact clean merged head. It creates a new empty authority
database, verifies the genesis chain, and writes a t=0 establishment receipt.
It does not connect the database to cognition or legacy runtime behavior.
