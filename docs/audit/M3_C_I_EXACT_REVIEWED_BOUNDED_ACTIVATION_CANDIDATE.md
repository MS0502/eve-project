# M3-C-I Exact-Reviewed Bounded Goal-Lifecycle Activation Candidate

## Status

This slice pins the previously dormant M3-C-H writer to one exact reviewed
implementation, one immutable authorization packet, one private caller-owned
absolute database-path digest, and one bounded storage policy.

It is an activation candidate for the M3-C lifecycle persistence stream only. It
does not install a runtime startup hook, produce a lifecycle event, begin the
M3-C-J observation window, transfer legacy goal-domain authority, authorize
legacy migration, or open M3-E.

## Exact prerequisite reused without rerun

```text
PR:             #225
exact head:     68efeca10c6819cb74ccc884e3c0c784e0b44c95
exact run:      30444371019
focused:        15 passed
full:           3,303 passed
forward gate:   0 / 0 / 0 / 0
artifact:       exact-head-validation-68efeca10c6819cb74ccc884e3c0c784e0b44c95
artifact SHA:   79f7f6a2034ced8b04dfb3ae3ed69f56cdd6eb6c8f0da3cb740fc900f4ef80be
M2-E run:       30444371035
M2-E:           6/6 passed
merge SHA:      18b70d277bec6a09db834b349a97bea11ff21abf
```

PR #226 then pinned the full #221-#225 prerequisite chain without rerunning any
accepted test, phone witness, or retained sequence.

## Human authorization basis

After receiving the reported M3-C-H boundary, the human project owner explicitly
instructed the agent to merge and continue the next stage autonomously while
preventing duplicate validation across chat changes. That instruction delegates
selection of the concrete private path and bounded policy within the already
fixed M3-C-F/H scope. It does not authorize any scope excluded below.

## Private path rule

The reviewed caller-owned absolute path is private operational material. The
public repository stores only its lexical SHA-256:

```text
cfcc91e8bab89beceff3ce8f5ecbc325705bd33b256e9d47ca8bdb9008833b80
```

No plaintext absolute path, environment variable, home-directory expansion,
filesystem search, fallback, default-path factory, or import-time path access is
checked in. A caller must supply the exact private absolute path on every writer
construction. Any other path fails before `SQLiteShadowStore` construction.

## Bounded storage policy

```text
snapshot_interval_events: 32
max_event_count:           4096
max_event_bytes:           16777216
max_snapshot_count:        128
max_snapshot_bytes:        16777216
max_backups:               3
```

The event and snapshot caps are paired so a full 4,096-event stream can produce
at most 128 scheduled snapshots. Capacity exhaustion remains fail-closed and
does not silently prune immutable lifecycle events.

## Immutable active packet

```text
reviewed implementation head:
68efeca10c6819cb74ccc884e3c0c784e0b44c95

authorization packet digest:
ab050d04f7ae7a6f920e94696d5b0988e4ad5331e9082d5ec61c30548166c111
```

The packet binds the exact #225 implementation validation, exact M3-C-E/G
prerequisites, writer/store/stream/event/producer versions, path digest, bounded
policy, rollback contract, explicit human review, and explicit false flags for
action, scheduling, speech, legacy goal authority transfer, legacy migration,
and M3-E.

## Reachability and I/O boundary

Import and writer construction remain I/O-free. Each append still requires:

1. the exact immutable active packet;
2. the exact private absolute path whose lexical digest is pinned;
3. the exact bounded policy;
4. active M2-E v4-native substrate authority;
5. one exact next-sequence M3-C-E binding.

M3-C-I itself executes no production append and creates no database. CI exercises
only disposable temporary paths with test-only pins. A production-path append,
when separately initiated by an explicit private caller, is classified in its
receipt as a bounded production-authoritative persistence append while still
claiming no runtime integration or legacy behavior authority.

## Rollback

The existing reviewed rollback control remains unchanged:

- disable future appends;
- preserve immutable database/WAL/history as evidence;
- restore only into a separate path;
- verify schema, migration history, chain, snapshots, and reducer replay;
- keep the writer disabled until a new reviewed packet exists.

## Explicit exclusions

```text
production append executed by this PR:       false
runtime startup integration:                  false
action/scheduler/speech authority:            false
continuous drive/value persistence:           false
legacy goal-domain authority transferred:     false
legacy migration authorized:                  false
M3-E authority open:                          false
M3-C-J observation window started:            false
phone command issued or replayed:              false
retained sequences 1-5 replayed:               false
```

## Next boundary

After this exact-head candidate passes its own final registered validation and is
merged, the next separately reviewed slice is M3-C-J. M3-C-J must define and
observe a bounded lifecycle-event window with zero duplicate/conflict acceptance,
zero chain/replay/snapshot divergence, and verified rollback preservation. This
PR does not start that window merely by pinning the packet.
