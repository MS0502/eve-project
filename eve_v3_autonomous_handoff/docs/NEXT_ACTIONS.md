# NEXT_ACTIONS

## Current position

Round97 preflight note:

- On 2026-06-02, Codex searched `/workspace/eve-project` and `/workspace` for the latest Round96 package and found no `.zip`, `.tar.gz`, `.tgz`, or expanded source tree.
- Round97 implementation is blocked until either the preferred code-only Round96 package or the legacy split package is visible in `eve_v3_autonomous_handoff/packages/`, verified, and expanded. A checkout update attempt found no configured git remote and no visible uploaded binary files.


Latest known generated package:

- `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip`

Latest completed round:

- Round96 runtime mapping enable-smoke precheck

Status:

- `runtime_mapping_enabled=False`
- `enforcement_enabled=False`
- Ready token: `민석`
- Full suite still needs long-running validation outside the chat runtime.

## Immediate next steps

### Step 1 — Put source package into the repo

Current status: **blocked / still required** until the code-only package files or legacy split files are present in this checkout.

Upload the preferred code-only Round96 package files into `eve_v3_autonomous_handoff/packages/`, then restore and extract them so Codex can edit the real source directly.

Preferred code-only files:

- `eve_v3_round96_code_only_no_medium_vectors.zip`
- `eve_v3_round96_code_only_manifest.json`

Note: this package intentionally excludes `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy`. Use it for Round95~Round96 focused/adjacent validation and Round97 controlled runtime mapping enable smoke. If fastText medium-vector full validation is required, record that validation as blocked/partial.

Legacy split files are still accepted:

- `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01` or `part01`
- `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02` or `part02`
- `eve_v3_round96_split_manifest.json` or `manifest`

Restore commands:

```bash
cd eve_v3_autonomous_handoff/packages
python restore_round96_package.py --verify-only
python restore_round96_package.py
```

Legacy manual split restore if the split package is used:

```bash
cd eve_v3_autonomous_handoff/packages
cat eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01 \
    eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02 \
    > eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip
python restore_round96_package.py
```

Recommended branch:

- `work/autonomous-loop`

### Step 2 — Run validation

Commands to run from the expanded source root:

```bash
python -m compileall adapters/lex_concept_mapping_adapter.py adapters/runtime_smoke_runner.py adapters/state_debug_adapter.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py
pytest -q tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py --disable-warnings --maxfail=1
pytest --collect-only -q
pytest -q --disable-warnings --maxfail=1
```

Expected known results from chat runtime:

- compileall passed
- focused/adjacent tests: 7 passed
- collect-only: 1217 tests collected
- full pytest not completed in chat runtime due time limit

### Step 3 — Round97 controlled runtime mapping enable smoke

Goal:

- Turn on the smallest possible runtime mapping path for the accepted fixture token only.
- Keep rollback and audit explicit.

Required before implementation:

- pre-mutation checkpoint
- rollback report
- invariant checklist
- focused tests
- adjacent tests
- validation JSON

Stop if:

- full validation cannot run
- enabling requires broad refactor
- enabling bypasses AGP
- enabling treats vector evidence as AGP anchor
- tests need weakening

## Continue after Round97

Do not stop merely because Round97 is complete. If Round97 controlled runtime mapping enable smoke passes validation and no hard stop applies, update docs and choose the next highest-value safe round autonomously.

Candidate order after Round97:

1. AGP proof object expansion
2. Progress/Feasibility check for autonomous loop
3. false coherence diagnostics
4. event/frame graph later


## If Codex resumes before the source package is available

Do not attempt to reconstruct or fake the Round96 source tree. Check `eve_v3_autonomous_handoff/packages/` for the preferred code-only package first, then the legacy split package. If both are absent, record the package as still missing, keep runtime mapping disabled, and wait for package files to become visible in this checkout.


## Latest checkout update attempt

Codex attempted to update/inspect the checkout after the operator reported package files were uploaded to PR #1. In this execution environment:

- `.git/config` has no configured remote.
- `git ls-remote https://github.com/MS0502/eve-project.git` failed with `CONNECT tunnel failed, response 403`.
- `eve_v3_autonomous_handoff/packages/` still contains only `README.md` and `restore_round96_package.py`.
- `/workspace` contains no code-only package files, `part01`, `part02`, `manifest`, or long-named Round96 split files.

Next action remains: make either the two code-only package files or the three legacy split files visible in this checkout, then run `python eve_v3_autonomous_handoff/packages/restore_round96_package.py`.


## Code-only package update

The preferred package is now the 7.7MB code-only zip:

- `eve_v3_round96_code_only_no_medium_vectors.zip`
- `eve_v3_round96_code_only_manifest.json`

Codex should use this package before the legacy split workflow when it is visible in the checkout. The medium vector omission must be recorded honestly in validation results if any full validation path requires it.
