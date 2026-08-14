# M3-C-S Workstation Retarget and t=0 Preconditions

Status: **operational-record correction and acceptance retarget only**. This document changes no runtime code, creates no persistence authority, does not execute M3-C-S, does not start the 30-day continuity clock, does not start Issue #246, and does not authorize semantic-workspace, grounding, GPU, or vision work.

Decision date: **2026-08-13 (Asia/Seoul)**.

Baseline at decision: `origin/main = de3b15b6d4008555bdcf06e3ed53c62851ab3d8a` (PR #245 squash merge).

## 1. Operational record correction

The phone habitat requirement is retired. EVE's intended live habitat is the incoming workstation:

- CPU/platform: Ryzen 7 8840U class workstation;
- memory: 64 GiB;
- discrete GPU present: RTX 3060;
- GPU capability is **not** part of this acceptance gate and this retarget authorizes no GPU/vision work.

Reason for retirement: the phone filesystem inventory found no authoritative EVE event store and no evidence that EVE had ever been continuously operated there. The discovered SQLite files were milestone/validation-window outputs whose last modifications stopped in the 2026-07-26 through 2026-07-31 window. Therefore there is no authoritative store to migrate, hand off, or place under custody.

The prior store-migration/custody-transfer plan is **retired as not applicable because there is no transfer target**. This must never be restated as "store transfer completed" or equivalent. The next authoritative store is a **first establishment at t=0**, not a migration.

Consequences:

- no authoritative store currently exists;
- the 30-day continuity clock has **not started**;
- PR #245's persistence-integrity requirements remain binding prerequisites, but any operational wording that assumed an existing store custody-transfer target has no target and is closed as not applicable;
- no prior phone SQLite file may be promoted into authority merely to preserve the old plan.

## 2. M3-C-S retarget

Issue #244 / M3-C-S is retargeted from a phone execution witness to a **workstation first-establishment and continuity witness**.

The former five-stage operator discipline is retained only as historical rationale for bounded execution:

1. stage separation;
2. no unnecessary engine construction;
3. at most one engine construction in the stage that actually requires it;
4. completed evidence/receipts are reused rather than replayed;
5. partial/conflicting evidence fails closed rather than being deleted or repaired to force progress.

The old `>=3072 MiB MemAvailable` phone gate, wake-lock/app-switching instructions, phone path requirements, and phone-specific stage execution are **historical rationale, no longer live constraints**. They must not be applied to the workstation as acceptance requirements.

This retarget does not claim that any workstation execution has occurred.

## 3. Mandatory preconditions before t=0

No authoritative event store may be first-established until **all** of the following are true on the exact runtime/workflow lineage intended for t=0.

### 3.1 Persistence integrity is implemented, not merely documented

PR #245 §2 requirements must exist in runtime behavior and tests:

- WAL where supported by the active SQLite/platform configuration;
- `PRAGMA synchronous=FULL` for authoritative persistence;
- a documented fail-closed fallback when WAL is not supported;
- startup verification of the persisted tail/chain against the previously accepted tail metadata before new authoritative mutation;
- tail mismatch, impossible continuation, or unverifiable accepted tail refuses authority; no silent truncate/repair/continue;
- power-loss and forced-termination fault injection;
- snapshot/checkpoint selection followed by verification of every subsequent authoritative tail segment.

### 3.2 Deterministic environment is pinned

Direct inspection of `.github/workflows/exact-head-validation.yml` at baseline `de3b15b6d4008555bdcf06e3ed53c62851ab3d8a` confirms `runs-on: ubuntu-24.04` and confirms that it does **not** currently pin `PYTHONHASHSEED`, `OPENBLAS_NUM_THREADS`, or `OMP_NUM_THREADS`.

The t=0 policy is:

```text
PYTHONHASHSEED=0
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
```

These values must be applied in CI and at runtime before affected Python/numerical-library initialization. Runtime must reject or normalize an incompatible launch before authoritative mutation; it may not silently run an authoritative lineage under an unrecorded environment.

On the accepted Windows workstation, the authoritative runtime environment is
installed only from `requirements-lock.txt` with `pip --require-hashes`.  The
range declaration in `requirements-runtime.txt` is not an authoritative-runtime
installation source.  A durable B5 environment receipt binds the lock digest,
interpreter path, Python version, every locked installed distribution, and the
installed numpy version; the supervisor rejects a different interpreter or
environment before starting its child.

### 3.3 Supervision policy is implemented

The production supervisor policy is:

- start EVE automatically at workstation boot;
- restart after ordinary process crashes;
- **do not restart** after an integrity failure such as tail mismatch or inability to verify the accepted tail;
- integrity failure exits through dedicated status `EVE_EXIT_INTEGRITY_FAILURE = 86`;
- the supervisor keeps EVE stopped on exit 86 and surfaces an operator-visible alert/log condition;
- Windows Service Recovery does not distinguish child exit 86 from an ordinary
  child crash, so WinSW/NSSM starts the B5 supervisor rather than EVE directly;
- the B5 supervisor alone classifies the child exit: `0` stops normally, `86`
  latches a durable sentinel and forbids restart, and every other nonzero status
  restarts with exponential backoff;
- after child exit 86 the supervisor returns service success only after the
  sentinel, hash-chained audit record, local critical alert, and stopped state
  are durable; this prevents Service Recovery from converting the integrity
  failure into a restart;
- every subsequent supervisor start checks the sentinel before child creation
  and stops without launching EVE while the sentinel exists or is invalid;
- sentinel release requires an explicit operator identity, reason, and expected
  sentinel SHA-256.  Release archives rather than deletes the sentinel and
  appends a hash-chained operator-clear record.  There is no automatic clear;
- no supervisor loop may convert integrity failure into repeated restart/repair attempts.

The concrete Windows contract and physical proof procedure are recorded in
`docs/audit/B5_WINDOWS_SUPERVISION_CONTRACT.md`.  The prior systemd example is
not a valid workstation implementation and is not Windows evidence.

### 3.4 Configuration freeze at t=0

The t=0 acceptance record must pin the exact values/versions for persistence integrity behavior, deterministic environment, and supervision behavior. Changing any item in §§3.1-3.3 after t=0 invalidates the current continuity lineage: the authoritative store must be re-established under a new t=0 and the 30-day clock restarts from zero.

A code change unrelated to these gates does not automatically reset the clock; whether it invalidates continuity must be decided from its actual authority/replay effect. Unknown impact is `UNRESOLVED`, never zero.

## 4. What the 30-day continuity clock proves

The accepted definition is **option (i)**:

> **process survival + authoritative event accumulation + restart continuity**.

For one continuous 30-day acceptance lineage, evidence must show that:

- the supervised EVE process remains operational except for permitted ordinary restarts;
- authoritative events continue to accumulate on the same accepted lineage;
- permitted restarts reopen the store only after successful startup tail verification and continue from the accepted tail without silent repair/truncation;
- an integrity-failure exit stops the lineage rather than being counted as successful continuity;
- any t=0 gate change described in §3.4 resets the clock.

Option (ii), which would additionally require deterministic reproduction of full engine computational state, is **not** the current 30-day definition.

Reason: the current exact-head workflow emits/compares audit evidence such as file outputs, M2-D/M2-E `packet_digest`, and M2-B `report_digest`. Direct workflow/script inspection does not show an acceptance digest gate over the complete live engine state such as hormones, embeddings, and SA activation. Therefore existing exact-head evidence must not be represented as proving full engine-state deterministic reproduction.

A future proposal may add such a gate, but if the 30-day definition is expanded to require it after t=0, the definition changed and the continuity clock must restart.

## 5. Exact-head architecture correction

`.github/workflows/exact-head-validation.yml` runs on `ubuntu-24.04`. The repository's accepted exact-head runs therefore execute on GitHub's Ubuntu runner environment, not on the phone/aarch64 habitat. Historical exact-head success is CI evidence; it is not evidence that the same validation ran on the phone.

## 6. Ordering and non-bypass

The live ordering is:

```text
Task A retarget/correction (this document)
  -> Task B: implement PR #245 §2 persistence integrity + deterministic pins
  -> workstation t=0 first establishment
  -> 30-day M3-C-S continuity witness under option (i)
  -> Issue #244 completion
  -> Issue #246 representation/authority audit
  -> minimal grounding contract
  -> semantic-workspace consolidation design
```

Until Issue #244 is completed under this retarget, Issue #246 remains blocked. This document does not begin or partially execute that audit.

## 7. Unknowns

Any fact needed for t=0 that has not been directly demonstrated is `UNRESOLVED`. No missing store, metric, migration cost, lineage fact, or hardware property may be silently interpreted as zero or as successful completion.
