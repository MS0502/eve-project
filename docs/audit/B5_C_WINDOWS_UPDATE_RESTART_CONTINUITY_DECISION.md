# B5-C Windows Update restart-continuity decision

Date: 2026-08-15

Status: implementation decision; physical result remains `UNRESOLVED` until the
Ryzen 7 8840U Windows gate-d evidence passes.

This append-only decision supersedes only the earlier requirement that the B5
preflight prove `NoAutoRebootWithLoggedOnUsers=1` and
`AlwaysAutoRebootAtScheduledTime=0`.  The earlier B5-A observations and raw
host-policy records remain unchanged.  No Windows Update registry write is
authorized by this decision.

## Rationale

The 30-day continuity definition includes process survival, authoritative-event
accumulation, and restart continuity.  A Windows Update restart sends the
service a graceful stop and is no stronger than the gate-c intentional child
exit 93.  `EveB5Supervisor` is an Automatic LocalSystem service and does not
depend on interactive logon.  A Windows reboot is therefore a required
restart-continuity scenario, not an automatic continuity violation.

## Fail-closed verdict

The Windows Update preflight item has exactly two operational verdicts:

- `UNRESOLVED` when gate-d before/after captures are absent, incomplete,
  invalid, differently hashed, or fail any continuity check;
- `ACCEPTED` with reason `restart continuity proven` only when gate-d passes
  and every pending-reboot indicator observed after reboot is false.

The presence or absence of the two AU policy values remains recorded as raw
host state but is not an acceptance condition.  The implementation performs no
registry write and does not translate an absent value to zero.

## Gate-d binding

Preflight v2 consumes the immutable before-reboot and after-reboot physical
capture files and independently verifies:

1. both capture receipts and both startup-ready receipts are hash-valid;
2. the observed boot identity changed;
3. the service was Running before reboot and is Running and Automatic after
   reboot;
4. the authority-store SHA-256 and complete verification result are identical
   before and after reboot;
5. both ready records bind the observed store SHA-256 and startup tail-chain
   verification;
6. `CBSRebootPending`, `WURebootRequired`, and `PendingFileRename` are all
   false when the post-reboot preflight is captured.

The final physical-gate verifier recomputes gate-d from the same raw capture
files and requires exact equality with the proof embedded in preflight.  A
green preflight derived from different captures is rejected.

## Operator order

1. Keep `authority_active_for_runtime=false`; this decision does not start
   t=0.
2. Start the pinned B5 service on a clean proof-store copy and capture the
   before-reboot state.
3. Perform an actual Windows reboot to clear the current pending state.
4. Allow the Automatic service to start without interactive-logon dependence
   and capture the after-reboot state.
5. Run preflight v2 with both capture paths.  If any pending indicator remains
   or gate-d differs, retain `UNRESOLVED`.
6. Continue gate-c, exit-86 latch, sentinel restart blocking, and explicit
   operator-clear proof only after the WU item is `ACCEPTED`.

CI cannot substitute for the Windows physical captures.  Unit tests cover the
classification and binding logic only; they do not claim gate-d passed on the
workstation.
