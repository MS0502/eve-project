# B5-E LocalSystem safe-directory scope

Date: 2026-08-15

Status: implementation correction; the physical Windows gate remains
`UNRESOLVED` and t=0 has not started.

## Observed blocker

B5-D successfully removed the LocalSystem PATH dependency: the service found
and executed the exact Git path and SHA-256 pinned in runtime receipt v2.  Git
then exited 128 because the immutable checkout is owned by
`DESKTOP-DRIUF0B\Admin` while the service runs as `NT AUTHORITY\SYSTEM`.
Git reported `detected dubious ownership` and suggested a persistent global
`safe.directory` change.

No global or system Git configuration is written.  No registry value is
changed.  The failed starts are diagnostic only, created no sentinel, and did
not change the proof-store SHA-256.

## Correction

Every repository-verifier invocation now has the exact form:

```text
<receipt-pinned-git> -c safe.directory=<exact-checkout> <read-only-git-args>
```

The exception is command-scoped and names only the checkout already bound by
the runtime receipt.  It is not `safe.directory=*`, is not stored in a user,
LocalSystem, global, or system config, and is not inherited by unrelated Git
commands.  The executable path and digest remain pinned by B5-D.

The verifier continues to perform only the existing clean-install, commit, and
tree observations.  Missing or changed Git, a different checkout identity, or
any Git failure remains fail-closed.

## Physical-gate consequence

The B5-D and B5-E failed starts are not gate-d captures.  After this correction
is reviewed and merged, the operator must build a new receipt from the merged
head, repin the stopped service through dry-run plus exact PLAN_SHA256, and
require a stable Running service and valid startup-ready receipt before the
before-reboot capture.

`authority_active_for_runtime` remains false.  This decision performs no
reboot, physical-gate claim, Windows Update or Defender change, authority-store
mutation, or t=0 transition.
