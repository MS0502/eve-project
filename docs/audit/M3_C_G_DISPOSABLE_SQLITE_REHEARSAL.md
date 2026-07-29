# M3-C-G Disposable SQLite Rehearsal

## Status

Implementation candidate for the first post-M3-C-F slice. This is a synthetic,
disposable SQLite rehearsal only. It is not a dormant production writer, an
activation candidate, an observation window, a legacy goal-domain migration, or
an M3-E cutover.

Exact prerequisite reused without rerun:

```text
PR:             #223
exact head:     395dea54fdbdac5fad7a2fab35fea1466a52a919
exact run:      30439249755
focused:        no focused tests selected
full:           3,272 passed
artifact:       exact-head-validation-395dea54fdbdac5fad7a2fab35fea1466a52a919
artifact SHA:   a5e801992970cdcb2598f60ddee13a6e3957cc70db63335a3cca9a6c48bd3597
M2-E run:       30439249799
M2-E:           6/6 passed
merge SHA:      91470c1adace585995a2a92d39ebd3e330d57342
```

Chat, operator-session, PR metadata, review metadata, and Draft/Ready changes are
not validation invalidators. The exact #221, #222, and #223 validation evidence
is reused. The #211 phone witness and retained sequences 1 through 5 remain
immutable and must not be executed again.

## Implemented boundary

`core/m3_c_g_disposable_sqlite_rehearsal.py` exposes one bounded operation:

```text
run_disposable_sqlite_rehearsal(
    sources,
    rehearsal_root=<explicit caller-created absolute directory>,
    checkpoint_sequence=<strict interior sequence>,
)
```

The operation accepts only a replay-valid synthetic M3-C-D candidate chain and
first resolves the active M2-E v4-native substrate authority through the merged
M3-C-E binding contract. An inactive or operationally rolled-back authority
fails before any path is opened or created.

The caller must create and pass the rehearsal directory. The implementation has
no default database path, rejects relative or absent roots, rejects pre-existing
targets, never uses `:memory:`, never scans for a database, and never reuses a
legacy goal database.

## Rehearsal order

The order is fixed:

1. validate the M3-C-D source chain and active M2-E authority;
2. derive canonical M3-C-E bindings and `shadow_only` v4 `EventEnvelope`s;
3. initialize one concrete SQLite file under the caller-owned temporary root;
4. append exactly one envelope per store call;
5. read back and verify that single envelope immediately;
6. replay every persisted prefix and compare it with direct M3-C-D reducer replay;
7. at the configured interior sequence, write a reducer snapshot and create a verified backup;
8. append and verify the remaining suffix;
9. prove full SQLite replay equals direct reducer replay;
10. prove snapshot-plus-suffix restore equals full replay;
11. copy the checkpoint backup into a distinct restore path;
12. verify schema, migration history, event chain, snapshot, and reducer state on the restored database;
13. return one immutable public-safe rehearsal receipt.

The forward database and restored checkpoint database are both preserved inside
the disposable test directory. Failure after a committed append does not delete
or silently repair the forward database.

## Acceptance predicates

A successful receipt requires all of the following:

```text
concrete SQLite file used:                  true
one event per append call:                  true
pre-commit store readback verified:         true
post-commit stream readback verified:       true
event-chain integrity verified:             true
direct reducer == full SQLite replay:       true
full replay == snapshot-plus-suffix replay: true
checkpoint backup restore verified:         true
forward database preserved:                 true
restored database is separate:              true
disposable SQLite write performed:          true
production authoritative append performed:  false
live writer installed:                      false
production integration performed:           false
action/scheduler/speech authorized:          false
legacy goal authority transferred:          false
M3-E authority open:                        false
writer operationally enabled:               false
```

The envelope authority remains `shadow_only`. The active M2-E decision means the
SQLite substrate is authoritative for v4-native persistence; it does not turn a
synthetic rehearsal append into production goal-domain authority.

## Focused failure proofs

The focused harness covers:

- M2-E operational rollback refusal before file creation;
- invalid checkpoint and invalid caller path refusal;
- pre-existing target refusal without overwrite;
- duplicate event conflict with no partial write;
- unknown causation conflict with no partial write;
- bounded-capacity refusal with prior history intact;
- preservation of the forward database after an injected post-commit replay failure;
- immutable and deterministic receipt evidence;
- absence of default path, production runtime, legacy goal, action, scheduling, and speech hooks.

## Explicit exclusions

This slice does not add or authorize:

- a production database path;
- startup or runtime integration;
- a persistent writer object reachable outside the one-shot rehearsal;
- an environment-variable or file-presence activation path;
- retry, batch append, sequence repair, or duplicate-as-success behavior;
- action execution, scheduling, or speech generation;
- affect, memory, vector, category, or continuous-drive mutation;
- legacy goal-domain authority transfer or deletion;
- M3-E authority;
- a phone command, witness replay, or retained sequence append.

## Next boundary

Only after exact-head acceptance and merge of this slice may M3-C-H be designed
as a separately reviewed dormant writer integration. M3-C-H must remain
unreachable by default, require an absent reviewed authorization packet, and
must not inherit production activation from this rehearsal.
