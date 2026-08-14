# B5-A Privileged Operation Review

## Decision and non-execution boundary

This review fixes the intended Windows B5 administration surface before any
administrator token is granted.  Nothing in B5-A installs a service, creates a
scheduled task, edits the registry, changes Windows Update or Defender, changes
power settings, reboots Windows, activates runtime authority, or starts t=0.

The approved architecture remains:

```text
Windows SCM -> WinSW -> B5 supervisor -> controlled EVE boundary
```

The only scripted privileged entrypoint is
`scripts/operator/b5_windows_service.ps1`.  It is dry-run by default.  An
application attempt needs both `-Apply` and the exact `PLAN_SHA256` emitted by a
separate dry run over unchanged inputs and unchanged host state.  Rollback uses
`scripts/operator/b5_windows_service_rollback.ps1`, which has the same dry-run
and reviewed-digest gate.

Microsoft Defender exclusions and Windows Update settings are deliberately not
mutated by either script.  Minseok handles them in the GUI using the procedures
below.  Unknown or unreadable state is `UNRESOLVED`, never an assumed zero or
pass.

`authority_active_for_runtime` remains false.  The 30-day clock remains day 0.

## Read-only workstation snapshot

The initial values were observed without elevation at
`2026-08-14T12:01:07Z`.  Fast Startup and the four AC power values were
recaptured read-only at `2026-08-14T12:29:32Z` with the same results.  They are
review context, not evidence that a future apply sees the same state.  The
dry-run plan must recapture its inputs immediately before any authorization.

| Item | Observed value | B5 target | Current verdict |
|---|---|---|---|
| `EveB5Supervisor` | service absent | installed, `Automatic`, initially stopped | change required |
| `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power\HiberbootEnabled` | DWORD exists, value `0` | `0` | target currently met; no registry write is planned unless it drifts |
| `C:\ProgramData\EVE` | absent | deployment/evidence parents created only by reviewed apply | change required |
| AC idle sleep | `0x00000000` | never | pass at observation time |
| AC idle hibernate | `0x00000000` | never | pass at observation time |
| AC lid-close action | `0x00000001` (`Sleep`) | `0x00000000` (`Do nothing`) | manual change required |
| AC hard-disk idle | `0x0000001e` (30 seconds) | `0x00000000` (`Never`) | manual change required |
| active power scheme | `381b4222-f694-41f0-9685-ff5bb260df2e` (`Balanced`) | captured scheme with AC sleep, hibernate, disk idle disabled and lid set to do nothing | blocked by lid and disk idle |
| `AutoAdminLogon` | string `0`; default user `Admin`; no `DefaultPassword` value | disabled for the service path | pass; automatic-logon alternative not active |
| Windows Update AU policy key | absent; policy values unreadable because absent | manually reviewed restart posture | `UNRESOLVED` |
| pending file rename | false | false before physical gate | pass at observation time only |
| Defender antivirus / real-time protection | true / true | protection remains on | pass at observation time |
| Defender exclusions | query returned `N/A: Must be an administrator to view exclusions` | exact outside-repository proof-store directory excluded, no broader path | `UNRESOLVED` |

The previously observed Fast Startup value is not used as a rollback guess.
The apply script captures value existence and the exact value again and binds
them into the reviewed plan and rollback state.

## Hiberboot discrepancy resolution

The earlier B5 report observed `HiberbootEnabled=1`; the B5-A snapshot observed
`0`.  This is not a `powercfg`-versus-registry interpretation difference:

- `reg.exe query` returns `REG_DWORD 0x0` for the exact HKLM value;
- PowerShell's Registry Provider returns integer `0`;
- both 64-bit and 32-bit .NET Registry views return DWORD `0`; and
- `powercfg /availableSleepStates` separately reports Fast Startup disabled by
  current system policy.

The System log shows an intervening user-context shutdown initiated for
`DESKTOP-DRIUF0B\Admin` by `RuntimeBroker.exe` at
`2026-08-14T11:15:28.492Z`, followed by OS start at
`2026-08-14T11:26:09.500Z`.  Kernel-Boot event 27 records boot type `0x0`, a
full boot rather than hiberboot.  Registry value-change auditing was not
available, so the actor and exact command that changed `1` to `0` remain
`UNRESOLVED`.

The current authoritative setting is therefore `0`, but the unexplained state
transition is retained as a provenance gap.  With the current value unchanged,
the dry-run plan emits `fast_startup_registry_write_required=false` and performs
no Fast Startup registry write.  If the value drifts, the plan digest changes
and the conditional privileged write reappears for separate review.

## Fixed inputs and unresolved identity-bound inputs

The following setup targets are fixed before administration:

| Field | Exact target |
|---|---|
| service name | `EveB5Supervisor` |
| deployment directory | `C:\ProgramData\EVE\B5\Service` |
| wrapper executable | `C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe` |
| wrapper configuration | `C:\ProgramData\EVE\B5\Service\EveB5Supervisor.xml` |
| service account | WinSW default `LocalSystem` |
| service start mode | `Automatic`, not delayed |
| wrapper/supervisor recovery | restart after 10 seconds; reset failure window after 3,600 seconds |
| Fast Startup registry target | `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power\HiberbootEnabled=0` |
| reviewed WinSW binary | `WinSW-x64-v3.0.0-alpha.11.exe`, SHA-256 `a2daa6a33a9c2b791ae31d9092e7935c339d1e03e89bfb747618ce2f4e819e20` |
| operator shell compatibility | Windows PowerShell 5.1; plan hashing uses `BitConverter` rather than unavailable `.NET 5+ Convert.ToHexString` |

The exact final PR head/tree, clean checkout path, fresh lock-bound runtime
receipt, Python path, proof-store copy, control file, sentinel, and evidence-log
paths are still `UNRESOLVED` because B5-A itself changes the PR tree.  No apply
is permitted while any of them is unresolved.  A later dry run accepts them as
mandatory arguments, resolves every path, hashes WinSW and the generated XML,
and prints them in the plan before asking for authorization.

The intended rollback-state path is outside the service directory so it
survives service removal:

```text
C:\ProgramData\EVE\B5\evidence\b5-admin-rollback-state.json
```

It must be absent before apply.  The setup script refuses to overwrite it.

## Complete privileged-operation inventory

### P1. Persist rollback state

- Exact operation:
  `Write-AtomicJson C:\ProgramData\EVE\B5\evidence\b5-admin-rollback-state.json`
  using UTF-8 without BOM and a same-directory temporary file followed by
  `Move-Item`.
- Before: `C:\ProgramData\EVE` and the rollback file were absent at the
  read-only snapshot.  A future dry run must reconfirm the exact state.
- After: one digest-bound rollback record contains the source plan digest,
  original Fast Startup value/existence, service state, target paths, and
  expected wrapper/configuration hashes.
- Rollback: the rollback record is retained as evidence.  It is not
  automatically deleted.
- Why privileged: the fixed target is under `C:\ProgramData`, and it is the
  recovery prerequisite for all following changes.

### P2. Conditionally disable Fast Startup

- Exact registry path:
  `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power`.
- Exact value: `HiberbootEnabled`, `REG_DWORD`.
- Exact command, included only when the recaptured value is absent or nonzero:

  ```text
  reg.exe add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power" /v HiberbootEnabled /t REG_DWORD /d 0 /f
  ```

- Before: value exists and is `0` in the current read-only snapshot, so the
  current plan contains no registry write.  The dry run recaptures it; if it
  changes, the plan digest changes and the old approval cannot be used.
- After: value exists and is `0`.
- Rollback, only when apply actually changed the value: if the pre-apply value
  existed, restore that exact DWORD with
  `reg.exe add ... /d <captured-value> /f`; if it did not exist, use
  `reg.exe delete ... /v HiberbootEnabled /f`.
- Why necessary: Fast Startup can preserve a hiberboot session across shutdown
  and power-on.  B5 requires the complete supervisor launch and startup
  tail-chain verification path on each tested shutdown-to-start transition.

This remains a required check even when the observed value is already zero.
It does not authorize a redundant HKLM write.  On the current state, WinSW
deployment is the only planned machine change.

### P3. Create the fixed service deployment files

- Exact operations after the plan has resolved and printed every input:

  ```text
  New-Item -ItemType Directory -Path 'C:\ProgramData\EVE\B5\Service' -Force
  Copy-Item -LiteralPath '<reviewed WinSW path>' -Destination 'C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe'
  [IO.File]::WriteAllText('C:\ProgramData\EVE\B5\Service\EveB5Supervisor.xml', <reviewed XML>, UTF8-no-BOM)
  ```

- Before: deployment directory, wrapper, and configuration were absent in the
  current snapshot.  Apply refuses existing wrapper or configuration files.
- After: wrapper SHA-256 equals the reviewed WinSW digest; the XML SHA-256
  equals the value printed in the dry run.  XML starts the B5 supervisor in the
  pinned interpreter; it never starts EVE directly.
- Rollback: only exact hash-matching wrapper/configuration files are removed.
  WinSW logs and physical evidence are retained.  Hash drift blocks deletion
  and requires a new review.
- Why privileged: these are machine-wide deployment files used by Windows SCM.

### P4. Install and configure the Windows service

- Exact commands:

  ```text
  C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe install
  sc.exe config EveB5Supervisor start= auto
  sc.exe failure EveB5Supervisor reset= 3600 actions= restart/10000
  ```

- Before: service absent.
- After: service exists, startup type is `Automatic`, and it remains stopped
  until a separately reviewed `Start` action.  WinSW's default service identity
  is `LocalSystem`; this broad privileged runtime identity is explicit review
  scope, not an incidental default.
- Rollback, when hashes still match:

  ```text
  C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe stopwait
  C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe uninstall
  ```

  The stop command is omitted if already stopped.  The rollback script does
  not use an unreviewed `sc.exe delete` fallback when the wrapper/configuration
  has drifted.
- Why necessary: current physical-gate case (c) requires Automatic service
  start after an actual reboot.  Service Recovery covers WinSW/supervisor
  failure only.  The supervisor maps child exit 86 to a durable latch and then
  exits successfully, so Service Recovery cannot reinterpret it as a crash.

### P5. Later service-control actions used by the physical gate

Each control action receives its own dry run, plan digest, `-Apply`, and
rollback-state file.  Exact commands are:

```text
C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe start
C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe stop
C:\ProgramData\EVE\B5\Service\EveB5Supervisor.exe restart
```

`Start` and `Restart` plans default to `-ExpectedTerminalStatus Running`.  The
repeated-start proof with an exit-86 sentinel must instead set
`-ExpectedTerminalStatus StoppedLatched`; that means the supervisor checks the
sentinel, launches no child, and stops normally.

The plan records the prior service status.  Rollback restores `Running` with
`start` or `Stopped` with `stopwait`; if status is already restored, rollback
is a no-op.  These actions are necessary for exit-93, exit-86, sentinel restart,
reboot, and operator-clear captures.  They do not clear the sentinel.

### P6. Privileged Event Log context

On exit 86, the supervisor may attempt this Windows command:

```text
eventcreate /T ERROR /ID 86 /L APPLICATION /SO EVE-B5-Supervisor /D <bounded alert>
```

It runs inside the already reviewed service identity, not as a separate UAC
setup operation.  Failure is recorded and does not weaken the durable local
alert/sentinel gate.  No Event Log source is pre-registered by B5-A.

## B5 blocker: transient I/O is not separated from integrity failure

Source inspection finds that the current implementation does **not** satisfy
the required distinction.

- `core/authoritative_store.py:468-473` configures SQLite's five-second timeout
  and `busy_timeout=5000`, but defines no outer bounded retry classifier.
- `core/authoritative_store.py:728-731` converts every startup
  `sqlite3.Error`, including `SQLITE_BUSY`, `SQLITE_LOCKED`, and `SQLITE_IOERR`
  families, into `AuthorityUnprovable`.
- `AuthorityBusy` subclasses `AuthorityUnprovable`; a writer-lock `OSError` is
  therefore an exit-86 condition rather than a separately classified
  availability condition.
- `scripts/operator/b5_runtime_probe.py:173-186` maps every caught `OSError`,
  `AuthorityPersistenceError`, or `AuthorityUnprovable` to exit 86.  This also
  includes sharing violations while hashing the database or writing evidence,
  not just accepted-chain mismatch.
- Append paths roll back and re-raise generic SQLite exceptions.  They have no
  stable transient/integrity classifier, bounded retry receipt, or test matrix.
- No authority test injects `SQLITE_BUSY`, `SQLITE_LOCKED`, `SQLITE_IOERR`, or
  Windows sharing/lock violations and proves that exit 86 is absent.

Verdict: `TRANSIENT_IO_CLASSIFICATION_UNRESOLVED`.  UAC apply and the physical
gate remain blocked until a separate reviewed change establishes:

1. Bounded retry with recorded attempt count/backoff for known transient
   SQLite busy/locked cases and Windows sharing/lock violations.
2. No accepted mutation or child readiness while the store is unavailable.
3. After a commit-phase error, reopen and prove whether the event/tail commit
   happened; never blindly append the same event again.  If the durable outcome
   is genuinely ambiguous, exit 86 remains correct.
4. Persistent availability failure exits non-86 so the supervisor applies its
   crash backoff.  The exact transient exit code remains `UNRESOLVED` until that
   change is reviewed.
5. Canonical bytes/hash, tail mismatch, malformed accepted history, failed
   integrity check, or an unprovable commit outcome still exit 86 and latch the
   sentinel.
6. Deterministic tests cover SQLite primary/extended error codes and Windows
   `ERROR_SHARING_VIOLATION`/`ERROR_LOCK_VIOLATION`, plus a physical WAL churn
   proof under the installed Defender posture.

An exact proof-store Defender exclusion reduces scanner interference but does
not replace this classifier; storage, backup, indexing, or other processes can
cause the same transient failure class.

### Explicitly not classified as privileged setup

- A local interactive user can normally request a reboot with
  `shutdown.exe /r /t 0 /d p:4:2 /c "EVE B5 reboot-continuity proof"`.
  B5-A does not run it.  Host policy may still deny the request; that is
  `UNRESOLVED` until the physical gate.
- Runtime-environment creation in a user-writable directory, proof-store copy
  creation, hashing, preflight capture, and sentinel clearing are not intended
  to require elevation.  They must never target the t=0 establishment DB.
- AC sleep/hibernate and lid changes are assigned to Minseok's GUI review.
  Whether this device prompts for elevation when saving the lid setting is
  `UNRESOLVED`; if it does, stop and amend this review before granting access.

## Dry-run and apply command shape

No command in this section was executed by B5-A.  After the final identity-bound
paths exist, the review command is:

```powershell
& '<FINAL_REPO>\scripts\operator\b5_windows_service.ps1' `
  -Action Install `
  -RollbackState 'C:\ProgramData\EVE\B5\evidence\b5-admin-rollback-state.json' `
  -WinSWPath '<REVIEWED_WINSW_PATH>' `
  -WinSWSha256 'a2daa6a33a9c2b791ae31d9092e7935c339d1e03e89bfb747618ce2f4e819e20' `
  -PythonPath '<FINAL_LOCKED_ENV>\Scripts\python.exe' `
  -RepoPath '<FINAL_REPO>' `
  -AuthorityStore '<PROOF_COPY_ONLY>' `
  -RuntimeReceipt '<FINAL_LOCKED_ENV_RECEIPT>' `
  -SentinelPath '<EVIDENCE_ROOT>\authority-stop.sentinel.json' `
  -AuditLog '<EVIDENCE_ROOT>\supervisor-audit.jsonl' `
  -AlertLog '<EVIDENCE_ROOT>\supervisor-alert.jsonl' `
  -StateFile '<EVIDENCE_ROOT>\supervisor-state.json' `
  -ControlFile '<EVIDENCE_ROOT>\control.json' `
  -ChildRawLog '<EVIDENCE_ROOT>\child.jsonl' `
  -ChildReadyFile '<EVIDENCE_ROOT>\ready.json'
```

With no `-Apply`, this prints the complete plan and changes nothing.  A later
application command is identical plus:

```text
-Apply -ExpectedPlanSha256 <EXACT_REVIEWED_PLAN_SHA256>
```

Rollback is reviewed first:

```powershell
& '<FINAL_REPO>\scripts\operator\b5_windows_service_rollback.ps1' `
  -RollbackState 'C:\ProgramData\EVE\B5\evidence\b5-admin-rollback-state.json'
```

Actual rollback additionally requires
`-Apply -ExpectedPlanSha256 <EXACT_REVIEWED_ROLLBACK_PLAN_SHA256>`.

## Administrator-free alternative: per-user Task Scheduler

### Exact design

A task owned by `DESKTOP-DRIUF0B\Admin` can use an `AtLogOn` trigger,
`Interactive` logon type, and `Limited` run level.  Its action is the same
hash-pinned Python/supervisor command used by WinSW.  It stores sentinel, audit,
alert, state, and child evidence under a stable user-writable directory such as
`C:\Users\Admin\AppData\Local\EVE\B5`.

The registration shape is:

```powershell
$action = New-ScheduledTaskAction -Execute '<PINNED_PYTHON>' -Argument '<SUPERVISOR_AND_PROBE_ARGUMENTS>' -WorkingDirectory '<FINAL_REPO>'
$trigger = New-ScheduledTaskTrigger -AtLogOn -User 'DESKTOP-DRIUF0B\Admin'
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId 'DESKTOP-DRIUF0B\Admin' -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'EveB5UserSupervisor' -Action $action -Trigger $trigger -Settings $settings -Principal $principal
```

This command is not part of the privileged setup script and was not run.  Task
Scheduler permits a non-administrator to register and manage a task in that
user's own security context, subject to the local task-folder ACL.  Host-specific
registration success is therefore `UNRESOLVED` until separately exercised.

### Semantic-equivalence verdict

| Requirement | User task verdict | Reason |
|---|---|---|
| child exit 86 stops restart | equivalent after task start | supervisor writes sentinel and returns 0; task-level restart-on-failure does not fire |
| sentinel survives launcher restart | equivalent after task start | any manual or scheduled relaunch checks sentinel before child creation |
| child exit 93 restarts with backoff | equivalent after task start | restart occurs inside the supervisor, independent of SCM or Task Scheduler |
| normal child exit 0 stays stopped | equivalent after task start | supervisor returns normally and the task ends |
| operator-only sentinel clear | equivalent | the same digest-bound `clear-sentinel` operation is used |
| startup tail verification after launcher begins | equivalent | the same pinned probe verifies the same proof-store copy |
| automatic start immediately after reboot | **not equivalent** | `AtLogOn` does not fire before `Admin` logs on |
| current physical-gate case (c), Automatic service start | **not equivalent** | a scheduled task is not the required Windows service observation |
| 30-day unattended continuity | **not established** | a lock screen, failed login, account restriction, or missing autologon prevents launch |

### Automatic-login route

Current `AutoAdminLogon` is `0`.  Enabling Windows automatic logon is not an
administrator-free step.  Microsoft Sysinternals Autologon stores the password
as an LSA secret, still warns that an administrator can retrieve/decrypt it,
and can be bypassed for one boot with Shift.  Direct Winlogon-registry setup is
not approved because it can expose a password in
`HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon`.

For completeness, the rejected privileged delta would be:

| Field | Current value | Enabled value | Rollback |
|---|---|---|---|
| `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\AutoAdminLogon` | string `0` | string `1` | Sysinternals Autologon **Disable**, returning the value to `0` |
| default identity | `DefaultUserName=Admin`, empty domain | same reviewed local identity | retain the original identity values |
| credential | no `DefaultPassword` registry value observed | LSA secret created by the elevated Sysinternals GUI | Autologon **Disable** removes the configured automatic-logon secret |

The only acceptable command shape for a future review would be to launch a
separately hash-pinned `Autologon64.exe` with elevation and enter the credential
in its GUI.  The documented CLI form places the password on a command line and
is not approved.  The binary path/hash are currently `UNRESOLVED`; therefore no
exact autologon apply target exists and this operation is outside
`b5_windows_service.ps1`.

Consequently:

1. A user-level Task Scheduler task alone preserves exit semantics but fails
   unattended reboot continuity.
2. Adding automatic logon reintroduces a privileged operation and a new
   credential/session risk; it is not in the reviewed apply script.
3. Even with automatic logon, the current physical gate's service-specific
   observation is not met.  Adopting that route would require a separate
   contract change and exact physical proof, not an assumption.

The administrator-required service path therefore remains the accepted B5
execution target.  Fast Startup enforcement and machine-wide service
installation/control are the only scripted privileged changes.

Microsoft references:

- [Task Scheduler security contexts](https://learn.microsoft.com/en-us/windows/win32/taskschd/security-contexts-for-running-tasks)
- [Task Scheduler logon trigger](https://learn.microsoft.com/en-us/windows/win32/taskschd/logon-trigger-example--scripting-)
- [Sysinternals Autologon](https://learn.microsoft.com/en-us/sysinternals/downloads/autologon)

## GUI-only Windows Defender procedure

Current real-time protection is on, but the exclusion query was not authorized
for this token and is `UNRESOLVED`.  The accepted GUI target is one exact
outside-repository proof-store directory exclusion, not a repository, runtime,
source-database, or t=0 exclusion.

1. Open **Windows Security**.
2. Select **Virus & threat protection**.
3. Under **Virus & threat protection settings**, select **Manage settings**.
4. Confirm **Real-time protection** remains **On**.
5. Under **Exclusions**, select **Add or remove exclusions** and approve the GUI
   elevation prompt personally.
6. Record the complete before list.  Do not treat the administrator-only
   placeholder as an empty list.
7. Add one **Folder** exclusion for the exact outside-repository B5 proof-store
   directory.  Never exclude the repository, runtime environment, source B2
   database, t=0 database, a parent containing them, a file extension, or the
   Python process.
8. Record before/after screenshots and the literal resolved path.  Real-time
   protection remains on.
9. The current `b5_windows_preflight.py` rejects an authority-directory
   exclusion.  It must be revised by the transient-I/O follow-up to require and
   bind this exact narrow exclusion before the physical gate can pass.

Microsoft warns that an exclusion stops real-time scanning for that scope and
can increase risk: [Virus and threat protection in Windows Security](https://support.microsoft.com/en-us/windows/virus-and-threat-protection-in-the-windows-security-app-1362f4cd-d71a-b52a-0b66-c2820032b65e).

No Defender change appears in either B5 script.

## GUI-only Windows Update procedure

The current AU policy key is absent, so a no-auto-reboot policy is not proven.
Active hours reduce inconvenient restarts but are not a 30-day no-reboot
guarantee.  Until the GUI state is captured and a controlled-reboot policy is
explicitly accepted, this item remains `UNRESOLVED`.

1. Open **Settings > Windows Update > Advanced options**.
2. Expand **Active hours**, select **Manually**, and record the exact start/end
   values.  Choose the widest values the GUI permits around the supervised
   period.
3. Turn off **Get me up to date** if that option is present.
4. Turn on **Notify me when a restart is required to finish updating** if that
   option is present.
5. Review **Restart options** and any scheduled restart.  Cancel or reschedule
   it to an explicitly controlled evidence window.
6. Record every visible setting and pending-restart indication.  If any named
   option is absent, record `UNRESOLVED`; do not translate absence to zero.
7. Windows 11 Home may not expose a hard no-auto-reboot policy in this GUI.
   Active hours alone do not establish the required guarantee.  If the GUI
   cannot express the accepted policy, the 30-day preflight remains blocked
   pending a separately reviewed decision.

Microsoft describes Active hours as a way to reduce inconvenient restarts, not
as an absolute reboot prohibition: [Keep your PC up to date with active hours](https://support.microsoft.com/en-us/windows/deployment/updates-lifecycle/keep-your-pc-up-to-date-with-active-hours).

No Windows Update write appears in either B5 script.

## GUI-only power and lid review

AC idle sleep and hibernate are already `Never`.  AC lid close is currently
`Sleep` and must become `Do nothing`; AC hard-disk idle is 30 seconds and must
become `Never`.

1. Open **Control Panel > System and Security > Power Options**.
2. Select **Choose what closing the lid does**.
3. Set **When I close the lid > Plugged in** to **Do nothing**.
4. Save and capture the visible result.
5. Select **Change plan settings > Change advanced power settings > Hard disk >
   Turn off hard disk after** and set **Plugged in** to **Never** (`0`).
6. Reopen **Settings > System > Power & battery > Screen, sleep, & hibernate
   timeouts** and confirm plugged-in sleep and hibernate remain **Never**.
7. If Windows unexpectedly demands elevation to save these user power settings,
   stop.  That operation
   is `UNRESOLVED` and must be added to a new privileged review before use.

Microsoft documents the lid and timeout UI paths in [Shut down, sleep, or hibernate your PC](https://support.microsoft.com/windows/shut-down-sleep-or-hibernate-your-pc-2941d165-7d0a-a5e8-c5ad-8c972e8e6eff) and [Power settings in Windows 11](https://support.microsoft.com/en-us/windows/experience/power-battery/power-settings-in-windows-11).

The current `b5_windows_preflight.py` does not evaluate `SUB_DISK/DISKIDLE`.
That check must be added before the physical gate can pass; the read-only host
capture now retains the raw disk-idle query.

Fast Startup is not delegated to this GUI sequence; it remains the explicit
`HiberbootEnabled=0` target in the reviewed privileged script.

## Authorization checkpoint

Administrator access must not be granted until all of the following are exact:

- final PR head/tree and clean checkout;
- fresh `requirements-lock.txt --require-hashes` runtime receipt and numpy
  version;
- the transient-I/O classifier and deterministic error-injection tests;
- proof-store copy and all evidence/control paths;
- dry-run output with no `UNRESOLVED` input paths;
- reviewed setup `PLAN_SHA256`;
- separate GUI decisions for exact Defender proof-store exclusion, Windows
  Update, lid behavior, and disk/sleep/hibernate timeouts; and
- confirmation that PR #252 remains Draft.

This document and the scripts are preparation only.  They are not physical
evidence and do not make any B5 gate green.
