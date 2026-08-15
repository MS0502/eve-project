# B5-D LocalSystem repository-verifier pin

Date: 2026-08-15

Status: implementation correction; the physical Windows gate remains
`UNRESOLVED` and t=0 has not started.

## Observed blocker

The first service start from merged B5-C ran `EveB5Supervisor` as LocalSystem.
WinSW started the pinned Python process, but the supervisor exited 86 before it
could launch the child.  The raw WinSW error was `[WinError 2]`.

The cause was reproduced without changing the service: runtime-receipt
verification passed with the interactive Codex process PATH and failed with
the machine-only PATH.  The workstation has no `git.exe` in the machine PATH;
the only discovered Git belongs to the Codex dependency runtime.  The B5
receipt verifier invoked `git` by name while checking the exact repository
commit and tree, so LocalSystem could not resolve it.

This is an environment-binding defect, not authority-store corruption.  No
sentinel was created, the service remained stopped, and the proof-store SHA-256
did not change.

## Correction

Runtime receipt schema v2 records the repository verifier as an explicit
triple:

- kind: `git`;
- absolute executable path;
- executable SHA-256.

Environment installation resolves the verifier once, uses that exact
executable for the clean-checkout, commit, and tree observations, and seals its
path and digest into the receipt.  Every later load checks that the executable
still exists and has the recorded digest before invoking the same absolute
path.  Repository verification no longer depends on the service account PATH.

Missing, moved, or changed Git remains fail-closed.  No fallback PATH search is
allowed after receipt creation.  The verifier does not write the repository,
authority store, service configuration, registry, Windows Update policy, or
Defender settings.

## Physical-gate consequence

The failed pre-capture service starts are diagnostic evidence only and are not
gate-d observations.  After this correction is reviewed and merged, the
operator must create a new immutable checkout and v2 runtime receipt, repin the
stopped service through the existing dry-run/PLAN_SHA256 process, and obtain a
clean Running/ready state before making the before-reboot capture.

`authority_active_for_runtime` remains false.  No reboot, physical gate, or
t=0 transition is authorized by this document alone.
