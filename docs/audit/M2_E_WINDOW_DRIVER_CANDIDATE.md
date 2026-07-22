# M2-E Observation Window Driver Candidate

## Human review targets — halt / quota / git exclusion

### Halt conditions

Shadow writes freeze, without touching the legacy runtime, on any of:

- recovery digest mismatch;
- unrecoverable event/store/snapshot corruption;
- private companion disk budget breach;
- A9 hourly event-cap breach;
- any tick-sampling event;
- unauthorized effect evidence;
- companion git-exclusion proof failure;
- raw/private artifact boundary failure.

A freeze is fail-visible and terminal for shadow writes. The supervisor remains responsive, `window_status.sh` remains usable, and one `termux-notification` attempt is recorded. There is no automatic retry that can mutate past the halt and no automatic cutover path.

### Fixed quota

M2-C establishes the bounded empirical mapping **one persisted shadow event per accepted discrete `ActivationAdapter.learn_pair` call**. M2-C does not contain a wall-clock call rate. This candidate therefore fixes a conservative scripted operational proposal rather than falsely presenting a measured hourly rate:

```text
scripted discrete stimuli: 12 per cumulative runtime sim-hour
A9 event cap:              12 per cumulative runtime sim-hour
stimulus interval:         300 active-runtime seconds
quota:                     288 persisted events
circadian requirement:     24 cumulative runtime sim-hours
calendar target:           day 5
automatic extension:       through day 7 only when incomplete
```

The quota is exactly one fixed-rate cumulative circadian cycle: `12 × 24 = 288`. Calendar time does not advance the simulated circadian clock. A phone may be powered off or stored for long periods; only active monotonic runtime advances sim-hours. Power cycles are continuity evidence, not missing-runtime fabrication.

### Companion git-exclusion proof

The default raw companion root is outside the repository:

```text
$HOME/.local/share/eve-m2e-window-private
```

It is created with mode `0700`; raw state, SQLite/WAL data, recovery observations, backups, and notification records remain local and private. `.eve-m2e-window/` is a repository-local emergency fallback name and is registered in `.gitignore`. `setup_window.sh` rejects a private root inside the repository and requires:

```text
git check-ignore --no-index .eve-m2e-window/probe
```

Public/review artifacts contain only schemas, checks, canonical digests, and artifact hashes. No real memory payload enters GitHub Actions or the public repository.

## Baseline and authority pins

- baseline after A12 approval-record merge: `50a448961c8333f788b7f78fe4886cbdf7a0694e`
- accepted M2-E technical head: `6af18fa645a19576caa74d2f8fc8a7fee5baa139`
- accepted M2-E packet digest: `fa657687cc3799e6655d5750fc75438c72b6c86e73836ffc6afde2a043f1987d`
- A12 human-decision artifact digest: `1c2575c7ea2b6c0b8717b6f8f49da634c1f6dfa63a4bf151b6d75e2f154a2a6a`
- bounded stream: `shadow:legacy.activation.learn_pair`
- event-store authority: `shadow_only`
- legacy runtime and persistence: authoritative
- cutover authorization: false
- M3 authority: false

This PR creates a driver and evidence surface only. It does not itself start or satisfy the human observation window, authorize cutover, install production dual-read, grant authoritative recovery, or change defaults.

## Tier 1 — CI chaos

`.github/workflows/m2-e-window-driver.yml` runs synthetic stores only on `ubuntu-latest` and `windows-latest`.

### Hard-kill matrix

Each phase runs three repetitions by default:

1. idle;
2. mid-write with an uncommitted event row;
3. mid-snapshot with an uncommitted snapshot row;
4. mid-consolidation before partial backup publication.

Ubuntu uses `SIGKILL`. Windows has no POSIX `SIGKILL`; the workflow uses Python's hard process termination backed by Windows `TerminateProcess`, the platform-equivalent uncatchable kill. Every case must reopen the store, pass logical/SQLite integrity, retain the same event count, and reproduce the exact pre-kill recovery digest.

### Corruption and pressure

- newest corrupt snapshot is rejected and an older valid snapshot is selected;
- corrupt event evidence fails closed;
- a verified backup restores the exact expected digest;
- bounded storage pressure rejects the next append without pruning or changing the recovery digest.

### Portability

Two source artifacts are created independently, one on Ubuntu and one on Windows. Cross jobs perform:

```text
Ubuntu backup → Windows restore → replay → digest equality
Windows backup → Ubuntu restore → replay → digest equality
```

The backup, manifest, verification report, and each evidence file retain SHA-256 values. GitHub Actions additionally content-addresses uploaded artifact archives.

## Tier 2 — phone habitat

### One-shot setup

`scripts/habitat/setup_window.sh`:

- installs Python, Git, Termux:API and repository requirements;
- creates the mode-`0700` private companion tree;
- proves the private root is outside the repository and the fallback is ignored;
- creates a Termux:Boot hook;
- acquires `termux-wake-lock` when available;
- starts a singleton supervisor;
- prints the one-line status.

The Termux:Boot Android app itself remains an operator-installed prerequisite; the repository creates only its standard hook.

### Supervisor and recovery

The phone tier contains **no intentional kill scheduler**. A persistent worker marker distinguishes an actual unclean death or power interruption from normal seal/freeze/shutdown. Before resuming after an unclean exit, the store is reopened, integrity is checked, replay is repeated, and the recovered digest is compared with the last expected digest. Each death increments evidence only together with its recovery result. A mismatch freezes shadow writes.

Termux:Boot startup records a power-cycle continuity observation. A date gap while powered off is not counted as an observed midnight and does not add simulated runtime.

### Scripted activity and A9

The worker emits one discrete synthetic bounded stimulus every 300 active-runtime seconds, up to 12 in each cumulative runtime sim-hour. Timer checks, decay, status checks, watchdog checks, and elapsed-time accounting emit no event. Event payloads remain synthetic and restricted to the accepted `learn_pair` projection contract.

### Watchdog and no-wedge behavior

Every watchdog pass checks:

- SQLite and logical integrity;
- exact recovery digest;
- private companion disk usage against 512 MiB;
- frozen/sealed state.

On a halt, only shadow writes stop. The legacy runtime is never imported or signaled. The supervisor stays alive in a wait state, status remains readable, and notification is attempted once. A sealed window similarly enters a non-mutating wait state.

### Sealer and privacy

A complete seal requires all pre-fixed criteria and at least calendar day 5. If incomplete, execution continues automatically through day 7, then seals as incomplete for review. Sealing creates only a digest record: raw evidence stays in the private companion directory. The terminal state is `awaiting_human_review=true`; no cutover action exists.

## Pre-fixed acceptance criteria

A complete window requires:

- unexplained divergence: `0`;
- every recorded death has a matching recovery digest;
- A9 tick-sampling events: `0`;
- no cumulative runtime sim-hour exceeds 12 events;
- unauthorized effects: `0`;
- quota: at least 288 events;
- actual observed midnights: at least 3;
- cumulative circadian runtime: at least 24 sim-hours;
- shadow not frozen;
- legacy authority retained;
- cutover and M3 authority remain false.

Automatic cutover is prohibited. Seal means stop mutating and wait for separate exact-head evidence review and a separate human decision.
