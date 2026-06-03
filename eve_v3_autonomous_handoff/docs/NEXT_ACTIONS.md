# NEXT_ACTIONS

## Current position

Latest completed rounds:

- Round97 controlled runtime mapping enable smoke
- Round98 runtime mapping persistence gate audit

Status:

- `runtime_mapping_enabled=False` after rollback
- `enforcement_enabled=False`
- Ephemerally smoke-mapped token: `민석`
- Persistence gate: `ready_for_operator_persistence_decision`
- Persistence applied now: `false`

## Validation boundary

Passed focused/adjacent validation is recorded in `validation/ROUND97_VALIDATION_STATUS.json` and `validation/ROUND98_VALIDATION_STATUS.json`.

Blocked/partial validation is intentionally separated:

- `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` is absent from the code-only package.
- Full medium fastText validation is blocked until that artifact is restored.
- Repository-wide collect-only/compileall still hits legacy root issues unrelated to Round97/98.

## Highest-value next round

Round99 should design an operator persistence decision path, but must not persist runtime mapping unless one of these is true:

1. Medium vectors are restored and split/full validation passes.
2. The operator explicitly accepts a partial-validation persistence experiment.

Required Round99 outputs:

- explicit persistence preconditions
- operator approval schema
- rollback checklist
- validation plan distinguishing focused pass from medium/full blocked
- no AGP bypass
- no vector-as-anchor shortcut

## Round99 update — validation-first gate

Current position after post-merge validation:

- Focused compile check passed.
- Round97/98 focused tests are blocked/partial due absent subset vector files.
- Round92~Round98 adjacent tests are blocked/partial for the same reason.
- Collect-only and repository-wide compile probes still have separated pre-existing legacy root blockers.

Highest-value next round is now:

```text
Round100: medium vector restoration / validation plan
```

Required Round100 outputs:

- Decide how the medium 30k vector artifact is restored or validated outside the code-only package.
- Preserve manifest provenance/checksum rules; do not create fake checksums or seed files.
- Re-run Round97/98 focused and Round92~Round98 adjacent validation after restoration.
- Keep runtime mapping persistence disabled until validation is honestly passed or the operator explicitly approves partial validation.

Deferred until validation is unblocked:

- AGP proof object expansion.
- Runtime mapping persistence approval gate implementation.
- Any persistence mutation or enforcement enablement.

## Round100 update — artifact restoration gate

Current position:

- Round100 restoration/audit helper is implemented and tested.
- Medium, small, and mini vector artifacts are absent in this checkout.
- Runtime mapping focused validation remains blocked until known fastText context vectors are available.

Next required operator action:

1. Obtain the original medium 30k `vectors.npy` outside the PR diff.
2. Run `python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy`.
3. Install the artifact only if the audit reports `acceptable_for_manual_install=true`.
4. Rerun medium and focused runtime mapping validation.

Still deferred:

- AGP proof object expansion.
- Runtime mapping persistence approval gate.
- Any persistence/enforcement mutation.
