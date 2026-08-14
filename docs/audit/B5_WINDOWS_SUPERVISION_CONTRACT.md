# B5 Windows Supervision and Runtime-Pin Contract

## Scope and non-activation

B5 supplies the Windows supervision boundary and physical evidence needed
before a future authoritative-runtime activation.  It does not set
`authority_active_for_runtime` to true, does not start the 30-day clock, and
does not replace the existing t=0 establishment database.  Physical corruption
tests use a separate copy.  B5 does not modify or promote
`core/sqlite_shadow_store.py`, and does not add semantic workspace, grounding,
GPU/vision, SNN, brian2, cognition, model, vector, or generated-artifact
authority.

## Process hierarchy

The Windows Service Control Manager starts WinSW.  WinSW starts
`scripts/operator/b5_windows_supervisor.py` in the hash-pinned Python
environment.  The supervisor starts the controlled EVE boundary process.  The
service wrapper never starts that process directly.

```text
Windows SCM -> WinSW -> B5 supervisor -> controlled EVE boundary
```

The supervisor accepts these child outcomes:

| Child result | Supervisor action | Service-visible result |
|---|---|---|
| `0` | record normal stop; do not restart | stopped normally |
| `86` | persist sentinel, audit, alert, and stopped state; do not restart | stopped normally after latch |
| any other nonzero result | restart after bounded exponential backoff | supervisor remains active |

Windows Service Recovery is reserved for failure of WinSW or the supervisor
itself.  It is not the child-restart policy and cannot override the sentinel.

## Exit-86 sentinel

The sentinel schema is `eve.b5-authority-stop-sentinel.v1`.  It binds:

- UTC latch timestamp;
- exact child status `86` and process identifiers;
- SHA-256 of the child argument vector;
- absolute authority-store path;
- store SHA-256 immediately before launch and immediately after exit;
- `automatic_clear_permitted=false`; and
- a digest over the canonical sentinel payload.

Every supervisor start checks the sentinel before creating a child.  A valid,
invalid, unreadable, or digest-mismatched sentinel all block child creation.
The stopped state therefore survives a service-manager restart.

`clear-sentinel` is the only release operation.  It requires an operator
identifier, a nonempty reason, and the exact observed sentinel digest.  The
operation atomically moves the active sentinel to a timestamped archive and
appends an `operator_sentinel_clear` record.  It has no force, timeout,
condition-based ignore, or automatic release option.

Supervisor audit and alert logs are append-only JSON lines with a SHA-256 hash
chain.  A corrupt prior record makes the next state transition fail closed.  On
Windows, exit 86 also attempts an Application Event Log error with source
`EVE-B5-Supervisor`; the durable local alert record is authoritative evidence
even if Event Log submission itself reports a failure.

## Runtime environment pin

`scripts/operator/b5_runtime_environment.py install` creates a new environment
outside the repository using the exact `.python-version`, then executes:

```text
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip check
```

The immutable receipt binds the repository commit/tree, lock digest, exact
interpreter, Python version, locked distribution versions, and numpy version.
The current lock requires numpy `2.5.2`.  `requirements-runtime.txt` is recorded
only to prove that it was not used as the installation source.  Supervisor
startup verifies the receipt, installed environment, current interpreter, and
child interpreter before launching the child.

## Windows continuity preflight

The preflight records raw commands and interpreted values for every item.  A
missing or unreadable setting is `UNRESOLVED`; it is never zero or safe by
default.

| Item | Required B5 policy |
|---|---|
| Windows Update | no forced automatic reboot while an operator is logged on; every reboot still relies on the automatic service and full startup verification |
| sleep/hibernate/disk | AC idle sleep, hibernate, and hard-disk-off timeouts are disabled |
| lid close | do nothing on AC power, or prove the physical device has no lid |
| Fast Startup | `HiberbootEnabled=0`, forcing the complete startup path |
| Defender | real-time protection stays enabled; Minseok records one exact outside-repository proof-store directory exclusion; no broader path/process/type exclusion is allowed |
| power plan | active scheme is identified and its plugged-in idle settings meet the no-suspend policy |
| service | `EveB5Supervisor` exists with Automatic start |

`b5_windows_host_policy.ps1 Capture` is read-only and records before/after raw
host state.  It has no apply mode.  Windows Update, Defender, sleep/hibernate,
and lid settings are reviewed by Minseok in the Windows GUI and remain
`UNRESOLVED` until separately observed.  The sole scripted privileged setup and
service-control entrypoint is `b5_windows_service.ps1`; it is dry-run by
default, prints a content-digested plan, and requires both `-Apply` and that
exact reviewed digest.  Its companion `b5_windows_service_rollback.ps1` is also
dry-run by default and refuses drifted service files.

Fast Startup is the one conditional host registry target in the privileged
setup script.  It enforces
`HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power\HiberbootEnabled=0`.
When the recaptured value already exists as zero, the plan records that result
and performs no registry write.  If it drifts, the plan digest changes and the
conditional write requires review.  Shutdown followed by power-on must execute
the complete supervisor and startup-tail verification path rather than resume
a hiberboot image.

`b5_windows_preflight.py` separately evaluates the installed state and records
the pinned numpy version.  Any `UNRESOLVED` item makes the preflight exit 86.
Its current no-exclusion check conflicts with the reviewed exact proof-store
exclusion and must be corrected before physical execution.

`core/authoritative_store.py` currently converts every startup `sqlite3.Error`
to `AuthorityUnprovable`, while `b5_runtime_probe.py` converts broad `OSError`
and persistence errors to exit 86.  The implementation therefore does not yet
separate transient busy/locked/I/O/sharing failures from accepted-history
integrity failure.  This is a B5 blocker: known transient availability failures
require bounded retry and non-86 supervisor backoff, while actual chain/tail
mismatch, malformed accepted history, failed integrity verification, or a
genuinely ambiguous commit outcome continue to require exit 86 and the
sentinel.  Exact requirements and code evidence are in the B5-A review.

The detailed pre-administration review, exact commands, observed values,
rollback paths, GUI-only steps, and user-level Task Scheduler alternative are
recorded in `docs/audit/B5_A_PRIVILEGED_OPERATION_REVIEW.md`.  A limited
per-user logon task preserves exit-86, sentinel, and child-crash semantics once
it starts.  It does not satisfy unattended reboot continuity before logon.
Enabling automatic logon is itself privileged and is not an administrator-free
alternative; the current physical gate also explicitly requires an Automatic
Windows service.  The service path therefore remains required unless a later
review changes the gate contract.

## Physical proof on the Ryzen 7 8840U Windows workstation

CI runs on Ubuntu and is not evidence for the following B5 claims.  The
physical receipt schema `eve.b5-windows-8840u-supervision-gate.v1` embeds the
raw captures and independently validates all four cases:

1. An accepted-tail mismatch is injected into an outside-repository copy.  The
   source physical-gate database and t=0 establishment database are hashed
   before and after and must remain unchanged.  The child must report 86, the
   service must stop, and the sentinel/store hashes/logs must agree.
2. Starting the service again with the sentinel present must leave the service
   stopped and append `startup_blocked_by_sentinel` without any child launch.
3. An intentional child exit 93 must be followed by a backoff record, a second
   child start, successful tail verification, and a running service.
4. A physical OS reboot must change the observed boot identity.  The Automatic
   service must be running afterwards, the supervisor must start its child,
   complete tail-chain verification, and report the identical accepted-event
   count, event head, tail head, and store SHA-256 seen before reboot.
5. After restoring a known clean proof copy, an explicit operator clear must
   archive the sentinel and allow a verified running child again.

Each physical capture retains timestamps, child exit statuses, raw `sc.exe`
and PowerShell service observations, full sentinel/state/ready content,
hash-chained supervisor and alert logs, child log lines, and store SHA-256.  A
green verdict without these observations is invalid.

## Operator ordering

1. Commit and review the B5-A document and PowerShell preparation in a separate
   Draft PR while PR #252 remains Draft.  This step performs no host mutation.
2. Resolve the transient-I/O classifier and its deterministic tests in a
   separate reviewed change.  Update preflight to bind the exact proof-store
   Defender exclusion and AC disk-idle requirement before granting UAC.
3. Minseok completes and records the GUI-only Defender, Windows Update, lid,
   disk, sleep, and hibernate decisions.  Any unreadable item remains
   `UNRESOLVED`.
4. Freeze the final PR tree and create a new lock-installed runtime environment.
5. Run the service script without `-Apply`, review every printed path, command,
   before/after value, rollback target, and the plan SHA-256, then separately
   authorize the exact plan if acceptable.
6. With the reviewed digest, enforce Fast Startup off and install WinSW as
   `EveB5Supervisor`, targeting the supervisor in that exact checkout and
   environment.  Capture GUI-only host settings and run a preliminary
   fail-closed preflight.
7. Capture the clean running service before reboot, physically reboot Windows,
   capture the automatically started service after boot, and run the final
   preflight after reboot so any pending-reboot state must be cleared.
8. Run exit-93, exit-86, repeated-service-start, and explicit-clear proofs.
9. Finalize the physical receipt on the clean exact PR head and attach its
   digest and raw evidence to the Draft PR.
10. Only after the exact-head workflow, M2-E window driver, and physical receipt
   are green may the PR become Ready and be squash-merged.

None of these steps starts t=0 or authorizes runtime authority.
