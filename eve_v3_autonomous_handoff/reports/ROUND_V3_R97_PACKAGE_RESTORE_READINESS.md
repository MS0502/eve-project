# ROUND V3 R97 PACKAGE RESTORE READINESS

## Goal

Convert the Round97 source-package blocker into an actionable restore workflow for the split Round96 package.

## What changed

- Added `eve_v3_autonomous_handoff/packages/README.md` with exact upload, restore, SHA-256 verification, and extraction instructions.
- Added `eve_v3_autonomous_handoff/packages/restore_round96_package.py` to restore the split zip deterministically.
- Updated autonomous instructions so Codex does not stop at Round97 when validation passes and no hard stop applies.

## Required uploaded inputs

Long filenames:

```text
eve_v3_autonomous_handoff/packages/eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01
eve_v3_autonomous_handoff/packages/eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02
eve_v3_autonomous_handoff/packages/eve_v3_round96_split_manifest.json
```

Short uploaded filenames are also accepted:

```text
eve_v3_autonomous_handoff/packages/part01
eve_v3_autonomous_handoff/packages/part02
eve_v3_autonomous_handoff/packages/manifest
```

## Restore workflow

```bash
python eve_v3_autonomous_handoff/packages/restore_round96_package.py
```

The helper will:

1. require both part files and the manifest,
2. concatenate the part files into `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip`,
3. compare the restored zip SHA-256 to manifest `source_sha256`,
4. test and extract the zip into `eve_v3_autonomous_handoff/packages/round96_source/`.

## Current status

The restore workflow is ready and now accepts both long filenames and the short uploaded names `part01`, `part02`, and `manifest`. However, the binary part files are still not present in this local checkout. Round97 runtime implementation is therefore still blocked until the split files are uploaded.

## Safety state

No EVE runtime source state was mutated.

- `runtime_mapping_enabled`: unchanged / not enabled in this checkout
- `enforcement_enabled`: unchanged / not enabled in this checkout
- category creation: not performed
- concept memory mutation: not performed
- SA activation mutation: not performed
- AGP state mutation: not performed

## Next autonomous behavior

After the split package is uploaded and restored, Codex should:

1. run Round96 validation from `docs/NEXT_ACTIONS.md`,
2. perform Round97 controlled runtime mapping enable smoke with checkpoint, rollback, audit, tests, and validation JSON,
3. if Round97 passes and no hard stop applies, choose the next highest-value safe round and continue without waiting for another operator prompt.
