# M3-C-J Private Database Path Rebind

## Status

This slice replaces one unrecoverable caller-owned database-path digest without
modifying the accepted PR #225, #229, #230, or #231 implementation files.

The original path plaintext was intentionally absent from the repository and
could not be recovered from the private device shell history. The failed local
readiness attempts stopped before input parsing or SQLite access:

```text
real private-device command issued: false
production SQLite opened: false
database or sidecar created: false
observation window started: false
single-use attempt consumed: false
```

## Reused immutable evidence

The following exact evidence remains reused without rerun:

- PR #225 dormant writer validation;
- PR #227 and #228 bounded activation prerequisites;
- PR #229 observation evaluator validation;
- PR #230 private-device operator validation;
- PR #231 isolated exact-reviewed operator pin;
- PR #211 real-phone witness;
- retained observation sequences 1 through 5.

A chat, shell session, branch, PR metadata, or failed pre-I/O path lookup is not
a validation invalidator.

## Rebound packet chain

Only the caller-owned database-path digest and the authorization digests derived
from it change:

```text
original database-path digest:
  cfcc91e8bab89beceff3ce8f5ecbc325705bd33b256e9d47ca8bdb9008833b80
rebound database-path digest:
  269c89e0e6d5614e2ca86ae5e68b261f3bb0d67bc12bf2045957052cf82ef715

rebound writer authorization:
  852e20984a9d670ec2a690106984ebc5d0071daae63bac0c6ebf7f7b255bb1d4
rebound window authorization:
  7347ae8a0e9cf8b5c44e519728847e8a2e2cb87bd4ea7fc2baf63880f3f30e69
rebound operator authorization:
  a344de2cb41a2ffcf3923680b57c297ba340127b9f92e11d4a6ead72deffc7bb
```

The path plaintext remains private. The repository stores no default-path
fallback, environment lookup, directory scan, or heuristic recovery.

## Scoped execution

`core/m3_c_j_private_device_path_rebind.py`:

1. reconstructs the immutable accepted writer packet;
2. replaces only its database-path digest;
3. derives the corresponding window and operator packets;
4. verifies all three precomputed authorization digests;
5. rejects any caller path whose lexical digest differs;
6. opens the writer, window, and operator pins only for one synchronous call;
7. restores all five mutable pin values in `finally`.

`scripts/operator/m3_c_j_private_device_window_rebound.py` reuses the accepted
PR #230 command parser and output discipline. It swaps the three operator
functions only inside `main()` and restores them in `finally`.

## Authority boundary

```text
production append authorized: explicit single-use command only
runtime integration authorized: false
action authorized: false
scheduler authorized: false
speech authorized: false
legacy goal authority transferred: false
legacy migration authorized: false
M3-E authority open: false
phone witness replay authorized: false
retained sequence replay authorized: false
silent retry authorized: false
```

This PR does not issue the private-device command and does not create the
database. After exact-head validation and merge, the device still requires one
new canonical private input and the already-created private nonce before the
single command may run.
