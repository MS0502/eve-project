# EVE v3 Round621-640 operator baseline-lock workflow

Round621-640 freezes the green Round601-620 dual-environment baseline into a
repeatable operator-facing validation workflow.  The workflow is meant to be run
both before and after future feature PRs so feature work starts from the same
known no-load/no-persistence state.

## Operator baseline-lock command

Run this compact, read-only command from the repository root:

```bash
python scripts/operator_lock_round621_640_baseline.py
```

The command emits a compact JSON report to stdout.  It does not write an output
file, does not create `_operator_artifacts/`, and does not stage files.  The
operator-facing committed report path for this round is this document:

```text
docs/round621_640_feature_entry_readiness.md
```

The Round621-640 command wraps and preserves the previous stable command:

```bash
python scripts/operator_verify_round601_620_baseline.py
```

## Dual-environment workflow

### 1. Artifact-free mode

Artifact-free mode is a normal checkout with no operator-local subset files
under `_operator_artifacts/subset_medium_30k/`.  In this mode the baseline-lock
command must still pass and must report the default runtime as no-load.

Required invariants:

- `production_persistence_enabled` remains `false`.
- `runtime_mapping_enabled_default` remains `false`.
- `enforcement_enabled` remains `false`.
- `vector_contents_read` remains `false`.
- `vectors_loaded` remains `false`.

### 2. Artifact-present path-reference mode

Artifact-present path-reference mode is an operator Codespaces/local checkout
where ignored subset files may exist under:

```text
_operator_artifacts/subset_medium_30k/vocab.txt
_operator_artifacts/subset_medium_30k/vectors.npy
_operator_artifacts/subset_medium_30k/subset_manifest.json
```

The baseline-lock workflow may only inspect path and git metadata for these
files.  Their presence must not make production readiness green by itself, must
not read vector contents, and must not load runtime vectors.

### 3. No-load default

Default runtime remains no-load in both supported modes.  A path reference is
not a load authorization.  The baseline-lock command is path-metadata-only and
must keep `vector_contents_read=false` and `vectors_loaded=false`.

### 4. Explicit operator-authorized load boundary

Any future command that actually reads vector contents or loads runtime vectors
must be a separate, explicit operator-authorized command with its own policy,
checks, and tests.  Round621-640 does not create that boundary crossing and does
not grant implicit authorization from artifact path presence.

## Forbidden files and staging rules

Do not stage, track, fabricate, download, or commit these paths or path classes:

```text
_operator_artifacts/
vectors.npy
vocab.txt
subset_manifest.json
seeds/subsets/
*.zip
*.part
```

The safety proof command is:

```bash
git status --short
```

The status output must contain no operator-artifact, vector, vocab,
subset-manifest, seed-subset, zip, or part entries.

## Exact validation commands

Run the following validation before reporting feature-entry readiness:

```bash
python -m compileall -q adapters tests main.py scripts
pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_v3_round621_640_operator_baseline_lock.py
python scripts/operator_verify_round601_620_baseline.py
python scripts/operator_lock_round621_640_baseline.py
git status --short
```

## Regression guards added in Round621-640

The focused Round621-640 tests prove that the baseline-lock workflow:

- does not read vector contents;
- does not load runtime vectors;
- does not enable production persistence;
- does not enable runtime mapping by default;
- does not enable enforcement;
- does not stage forbidden artifacts;
- does not create operator artifacts when run without an explicit output path.

## Feature-entry readiness recommendation

Recommended next feature track, exactly one:

```text
read_only_appraisal_classifier_agp_input_stabilization_audit
```

This is the safest next track because it is read-only and aligns with the
existing transition rule to consolidate frozen semantic safety rails under AGP
input stabilization.  It must not enable production persistence, runtime mapping,
enforcement, vector loading, or AGP bypass.

## Round636-640 validation result recorded for this PR

This PR recorded the following green validation in an artifact-free checkout:

```text
python -m compileall -q adapters tests main.py scripts                         PASS
pytest --collect-only -q                                                       PASS, 1428 tests collected
python -m pytest -q                                                            PASS, 1428 passed
python -m pytest -q tests/test_v3_round621_640_operator_baseline_lock.py       PASS, 5 passed
python -m pytest -q tests/test_v3_round601_620_dual_environment_baseline.py     PASS, 5 passed
python scripts/operator_verify_round601_620_baseline.py                        PASS
git status --short                                                             PASS, no forbidden artifact/vector entries staged or tracked
```

The new Round621-640 command also passed locally:

```text
python scripts/operator_lock_round621_640_baseline.py                          PASS
```

The final git safety proof showed only source/doc/test additions and no
forbidden operator artifact entries:

```text
?? docs/round621_640_feature_entry_readiness.md
?? scripts/operator_lock_round621_640_baseline.py
?? tests/test_v3_round621_640_operator_baseline_lock.py
```
