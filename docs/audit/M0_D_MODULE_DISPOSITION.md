# M0-D Module Disposition

Baseline: `main` at `fe10cd954bdf445400ea6aa9708dd214ed761114`

Status: recommendations only. No module is deleted, deprecated in code, wrapped, rewritten, or activated by M0-D.

## Regeneration

```bash
python scripts/audit/m0_d_component_inventory.py --pretty
```

Every runtime module selected by the scanner receives exactly one recommendation: `KEEP`, `WRAP`, `REWRITE`, `EXPERIMENTAL`, `DEPRECATE`, or `REMOVE`. A recommendation is not an action. Each entry must retain M0-A/B/C or new M0-D AST evidence.

## Disposition policy

- `KEEP`: retain current evidence and behavior until explicit v4 ownership is assigned.
- `WRAP`: preserve behavior only behind a bounded capability, lifecycle, provenance, validation, or rollback boundary.
- `REWRITE`: preserve the required capability and tests, but replace architecture that directly conflicts with v4.
- `EXPERIMENTAL`: preserve as non-authoritative evidence with no production promotion or default activation.
- `DEPRECATE`: preserve historical or migration evidence while excluding the module from future runtime authority.
- `REMOVE`: recommend deletion only with high-confidence positive evidence and reviewer approval. The scanner does not infer removal from a name, docstring, or lack of reachability.

## Validated disposition totals

Pending first branch validation. Exact totals and full `REMOVE`/`DEPRECATE` lists will be inserted from canonical script output before review.

## High-impact preliminary recommendations

| Module | Recommendation | Evidence basis |
|---|---|---|
| `main.py` | `REWRITE` | M0-A active composition root, automatic LiveLoop start, autosave configuration, and operator persistence boundaries. |
| `language/streaming.py` | `REWRITE` | M0-A active chat mutation/output funnel; v4 requires structural source quarantine and expression isolation. |
| `adapters/live_loop.py` | `REWRITE` | M0-A/B active timed daemon mutation loop, queue/thread lifecycle, proactive output, and autosave. |
| `core/autonomous.py` | `REWRITE` | M0-A active autonomous state transition and output path spanning needs, environment, curiosity, history, and speech. |
| `adapters/persistence_adapter.py` | `REWRITE` | M0-A/C active legacy persistence plus gzip/pickle sidecar conflicts with future SQLite events and validated snapshots. |
| `adapters/hormone_adapter.py` | `REWRITE` | v4 affect migration requirement and M0-C hormone-state evidence. |
| `adapters/allostatic_adapter.py` | `WRAP` | M0-C hormone-to-drive bridge candidate; potentially reusable only as a bounded Vital compatibility projection. |
| `adapters/urge_adapter.py` | `WRAP` | M0-C hormone/drive bridge feeding proactive behavior. |

## Frozen PR recommendations

Recommendations only. M0-D does not close, comment on, rebase, modify, or merge these PRs.

| PR | Recommendation | Evidence to preserve |
|---:|---|---|
| #109 | `REWRITE-AS-V4-CONTRACT` | Strict JSON, canonical IDs, recursive forbidden fields, tamper matrix, and read-only downstream plans. |
| #97 | `CLOSE-PRESERVE-EVIDENCE` | Fail-closed situation cases, entity/relationship fixtures, read-only plans, and deterministic-ID defect history. |
| #86 | `REWRITE-AS-V4-CONTRACT` | Replay origins, confidence/boundary matrices, no-mutation and origin/fact-status tests. |
| #84 | `REWRITE-AS-V4-CONTRACT` | Cross-modal compatibility, identity-resolution prohibition, fail-closed and no-side-effect tests. |
| #82 | `REWRITE-AS-V4-CONTRACT` | Modality/event matrices, mixed-boundary, no-fact/no-identity, quarantine and gate tests. |
| #11 | `ABSORB-INTO-M1` | Approval guard, checkpoint-before-mutation, audit evidence, rollback verification, protected-state tests. |
| #7 | `ABSORB-INTO-M1` | Vector manifest/checksum/shape/dtype gates, ignored-artifact boundary, approval and rollback evidence. |
| #4 | `ABSORB-INTO-M1` | External-seed provenance, artifact verification, no-binary-commit and fail-closed behavior. |
| #1 | `CLOSE-PRESERVE-EVIDENCE` | Package manifest hash, missing-part hard stop, safe extraction and historical blocker reports. |

Actual closing remains a separate post-M0-D reviewer action.

## v4.0 assumptions vs runtime reality

1. **Event-log reproducibility vs distributed direct mutation.** v4 assumes replayable events and causal provenance; active chat, live, autonomous, and persistence funnels mutate state directly across modules.
2. **SQLite events/snapshots vs pickle sidecar/autosave.** v4 targets append-only SQLite plus validated snapshots; current active persistence combines legacy state, gzip/pickle sidecars, explicit save/load, and automatic autosave.
3. **Required affect migration vs missing migration plan.** M0-C inventories hormone and drive evidence but provides no concrete migration plan. This is a high-confidence unresolved governance defect.
4. **Speech is not life vs timed proactive-output convergence.** Current life-loop surfaces center on clocks, hormone decay, DMN/proactive output, and speech without one explicit continuity/lifecycle owner.
5. **Structural source quarantine vs combined chat funnel.** `StreamingEngine.chat_stream` receives input, mutates learning/context/history, and produces expression in one active module boundary; M0 evidence does not demonstrate structural raw-source isolation.
6. **Bounded learned subsystems vs distributed numeric/adaptive state.** Numeric/vector/adaptive operations and artifact formats are distributed without one complete provenance, evaluation, versioning, activation, and rollback contract.

These are v4.1 inputs only. M0-D does not draft new constitutional text.

## UNRESOLVED

- `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT`: reviewer ruling and a separate scope-compliant correction are required before M0-D recommendations can be treated as a complete constitutional basis.
- The two tracked parse-invalid legacy foundation snapshots remain unresolved evidence and are recommended for `DEPRECATE`, not automatic removal.
- Algorithmic module recommendations marked unresolved require reviewer ruling before implementation work.
