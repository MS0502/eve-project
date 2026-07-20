# EVE v4.1 Forward Regression Gate

Baseline: `main` at `8cd1a0ad0ed8aaa2810da0730c17b6168bd2fb7b`

Status: infrastructure-only. This gate changes no runtime behavior, persistence authority, model/vector activation, production defaults, or frozen PR.

## Purpose

The historical M0-A through M0-D scanners remain pinned to their merged snapshot universes. This gate is separate: it runs the same detector families against the current tracked Python tree and rejects only additions that are absent from the frozen forward baseline and not registered in the same reviewed PR.

The gate therefore enforces **unregistered delta = 0**, not absolute delta = 0. Event-kernel and audit infrastructure may add justified findings, but the introducing PR must register the exact fingerprint and occurrence count with rationale, owner, disposition, and PR number.

## Detector families

- M0-A current-tree mutation and direct-write visitors.
- M0-B current-tree `SILENT_BROAD_EXCEPTION_PATH` visitor.
- M0-D current-tree adaptive/numeric state, method, component, and artifact-I/O visitors.
- A conservative raw-capability taint candidate detector for runtime callables that route raw/source-named values into expression/generation sinks or return them from expression-named callables.

Raw-capability findings are candidates, not proof of an approved or exploitable edge. The complete capability-edge authority remains assigned to M2-B.

## Frozen manifest

`docs/audit/FORWARD_ADDITIONS_MANIFEST.json` contains:

- the exact v4.1 merge SHA;
- the frozen fingerprint multiset for that baseline;
- reviewed post-baseline additions.

Fingerprints exclude line numbers but include path, callable/symbol, detector family, classification, evidence, and stable details. This prevents unrelated line shifts from becoming false additions while preserving duplicate occurrence counts.

A registration is invalid when it is missing review metadata, exceeds the actual delta count, points at mismatched path/category/symbol metadata, or becomes stale after the corresponding finding is removed. Stale registrations fail the gate and must be deleted.

## CLI

```bash
python scripts/audit/forward_regression_gate.py --pretty
```

The command exits nonzero for baseline drift, malformed/stale registration, or unregistered additions.

For review diagnostics only:

```bash
python scripts/audit/forward_regression_gate.py --report-only --pretty
```

For the initial manifest bootstrap or an explicitly reviewed rebaseline proposal:

```bash
python scripts/audit/forward_regression_gate.py \
  --suggest-manifest-for-pr <PR_NUMBER> \
  --output /tmp/forward-additions-manifest.json \
  --pretty
```

The suggestion command does not approve its output. The resulting manifest must be inspected, committed in the same PR, and pass enforced mode.

## Boundaries

This infrastructure does not:

- modify the snapshot-pinned M0 scanners or their canonical outputs;
- authorize any registered finding automatically;
- implement an event kernel or M1-A;
- create persistence, checkpoints, vectors, weights, or runtime state;
- replace human review of capability, provenance, evaluation, rollback, or module disposition contracts.
