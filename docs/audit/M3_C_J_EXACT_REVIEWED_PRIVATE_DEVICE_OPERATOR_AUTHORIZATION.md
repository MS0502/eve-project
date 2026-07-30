# M3-C-J exact-reviewed private-device operator authorization

## Accepted prerequisite

This pin reuses PR #230 as the immutable reviewed operator-preflight prerequisite.

```text
PR:             #230
base SHA:       361ed88be399ed7650a946b58e713bc14253384e
exact head:     d8eb3c2d6b576cc313712f831f8b2f1556cdefb2
exact run:      30529376866
focused:        9 passed
full:           3,324 passed
forward gate:   0 / 0 / 0 / 0
artifact SHA:   52b20d6961565175295fad732dbb0a24fced943b85fb114743438b20f6672aaa
M2-E run:       30529377107
M2-E:           6/6 passed
merge SHA:      4488103356b7bf285badba451acafd0768885dac
```

The downloaded artifact SHA-256 was recomputed before this pin was created. PR #230 full-suite ran exactly once on its final registered head. Earlier discovery heads stopped before full-suite.

## Immutable implementation and isolated pin adapter

`core/m3_c_j_private_device_operator.py` remains byte-for-byte identical to the accepted PR #230 tree. This preserves the exact reviewed implementation and its M2-B evidence identifiers rather than shifting historical source-bound IDs merely to add authorization metadata.

The authorization lives in the separate import-side-effect-free adapter `core/m3_c_j_private_device_operator_pin.py`. It owns the reviewed constants, packet verification, synchronous scoped invocation, launch-head receipt binding, and unconditional restoration of the preflight module's absent authorization pins.

The scoped invocation is permitted only because the operator is an explicit single-use process with no runtime integration. It refuses an already-active scope, opens only the two reviewed in-memory pin values, executes one synchronous call, and restores both values in `finally`. It grants no ambient capability to another runtime caller.

## Reviewed packet

```text
operator implementation head:
  d8eb3c2d6b576cc313712f831f8b2f1556cdefb2
operator authorization digest:
  e360c0e669af3ba89a6f552c81c67e3b3d908171665ed20b510a0044003d13a5
required transitions: 4
max window events: 32
production append authorized: true, explicit command only
runtime integration authorized: false
action authorized: false
scheduler authorized: false
speech authorized: false
legacy goal authority transferred: false
legacy migration authorized: false
M3-E authority open: false
```

The packet binds the reviewed PR #230 implementation provenance, active writer/window packet digests, reviewed private database-path digest, and bounded event count. Building or verifying the packet performs no filesystem or SQLite access.

## Launch-head separation

The pinned implementation head is immutable provenance and cannot equal a descendant pin commit without a circular commit-hash dependency. The explicit operator command therefore performs a separate clean-checkout attestation:

1. `--expected-head` must equal the actual clean checkout head;
2. the packet must equal the reviewed PR #230 implementation packet;
3. the immutable preflight call receives its reviewed implementation head;
4. the adapter deterministically replaces only the returned receipt's `repository_head` with the independently verified launch head;
5. the receipt therefore records both `operator_implementation_head` and `repository_head`;
6. private paths are not inspected until packet and launch checks pass.

The exact launch head for a real command must be the accepted final head of this pin PR or a later separately reviewed descendant that explicitly reuses it. A chat, branch, PR metadata, or operator-session change is not authorization.

## Boundary after this pin

This change does not issue the private-device command and does not access the reviewed production database. It only makes the exact packet available to the explicit single-use command.

Before one real command can run, all of the following remain required:

- final exact-head validation and M2-E success for this pin tree;
- one concrete canonical private input file reviewed as new window material;
- caller-owned nonce file with private permissions;
- exact private database path whose public digest already matches the writer packet;
- absent database and SQLite sidecars;
- absent private journal, bundle, review output, backup directory, and restore directory;
- one clean exact checkout of the accepted launch head.

The command must not replay PR #211 phone evidence or retained sequences 1 through 5. A partial or completed attempt must never be silently retried.
