# M3-C-H Dormant Goal-Lifecycle Writer Integration

## Status

This slice adds a bounded goal-lifecycle writer code path but leaves it
unreachable in checked-in production code. It is not an activation candidate,
not an observation window, not a legacy goal-domain migration, and not an M3-E
cutover.

Exact prerequisite reused without rerun:

```text
PR:             #224
exact head:     ba14eb0b1064b6454c95870bd737e49d01608c0d
exact run:      30442498422
focused:        16 passed
full:           3,288 passed
forward gate:   0 / 0 / 0 / 0
artifact:       exact-head-validation-ba14eb0b1064b6454c95870bd737e49d01608c0d
artifact SHA:   2705d825fc827624e71e4a86ba992e9a19f2b90a60d3d4603ac60ab553de86c2
M2-E run:       30442498493
M2-E:           6/6 passed
merge SHA:      b717b676ec84fd157eabf5b0a947f68c1c6617eb
```

Chat, operator-session, PR metadata, review metadata, and Draft/Ready changes are
not validation invalidators. The #211 phone witness and retained sequences 1
through 5 remain immutable and must not be executed again.

## Dormancy mechanism

`core/m3_c_h_dormant_goal_lifecycle_writer.py` checks two private reviewed pins
before constructing `SQLiteShadowStore` or touching a path:

```text
_ACTIVE_REVIEWED_IMPLEMENTATION_HEAD = None
_ACTIVE_REVIEWED_AUTHORIZATION_DIGEST = None
```

Both are intentionally absent in M3-C-H. There is no public setter, environment
variable, file-presence check, import side effect, filesystem scan, default path,
or heuristic fallback. A structurally valid packet is still rejected while the
reviewed pins are absent.

Focused tests temporarily monkeypatch the private pins only to prove that the
latent path is internally coherent. That test-only injection is not retained in
the repository tree and is not an activation artifact.

## Authorization packet contract

The immutable packet binds:

- exact implementation head, exact-head run, focused/full counts, artifact
  SHA-256, forward error count, and M2-E 6/6 pins;
- exact M3-C-E and M3-C-G prerequisite merge and artifact digests;
- exact writer/store schema versions;
- exact stream, event type, producer, producer version, binding authority, and
  `shadow_only` envelope authority;
- exact bounded SQLite storage limits;
- digest of one explicit absolute caller-owned database path;
- rollback schema and procedure;
- explicit human review and bounded-writer-only authorization;
- explicit false values for action, scheduling, speech, legacy goal authority
  transfer, legacy migration, and M3-E.

Any missing, malformed, unreviewed, mismatched, or scope-escaping field fails
closed before store construction.

## Latent single-append protocol

When and only when a later exact reviewed packet is pinned, one call performs:

1. exact packet and implementation-head verification;
2. path and storage-policy equality verification;
3. operational-disable verification;
4. active M2-E authority verification;
5. explicit store construction and initialization;
6. schema and integrity verification;
7. exact stream-head, next-sequence, causation, binding, payload, and source
   round-trip verification;
8. one SQLite append call for one envelope;
9. post-commit readback, event-chain integrity, and direct reducer equivalence;
10. bounded snapshot creation when due;
11. one immutable append receipt.

No batch append, retry, repair, sequence skip, duplicate-as-success, partial
success, action, scheduling, speech, continuous-drive write, or legacy goal
mutation is added.

## Failure and rollback behavior

- missing reviewed pins or packet: refuse before store construction;
- path/policy mismatch: refuse before store construction;
- inactive or rolled-back M2-E: refuse before store construction;
- duplicate/sequence/causation conflict: no new append;
- bounded capacity failure: preserve prior history and remain deterministic;
- post-commit replay, integrity, or snapshot failure: disable the writer,
  preserve the database as evidence, and require separate-path recovery;
- reviewed rollback control: disable future appends without opening, deleting,
  repairing, or transferring authority.

## Checked-in authority facts

```text
reviewed implementation-head pin present:   false
reviewed authorization-digest pin present:  false
writer reachable by default:                false
production authoritative append performed:  false
production integration performed:           false
live runtime writer installed:               false
action/scheduler/speech authorized:          false
legacy goal authority transferred:           false
legacy migration authorized:                 false
M3-E authority open:                         false
phone command required:                      false
```

## Next boundary

M3-C-I is a separate bounded activation-candidate decision. It may proceed only
after M3-C-H exact-head validation and merge, and only with a new immutable
packet containing the final H evidence plus explicit human review. M3-C-H merge
does not authorize pinning either active value, writing a production database,
starting an observation window, or changing legacy goal-domain authority.
