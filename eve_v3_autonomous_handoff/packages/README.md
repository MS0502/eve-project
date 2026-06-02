# Round96 split package instructions

GitHub web upload limit is 25MB, so upload the Round96 source archive as split files.

## Required files

Upload these files into this directory:

1. `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01`
2. `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02`
3. `eve_v3_round96_split_manifest.json`

## Reconstruct in Linux / macOS / Termux / Codespace

```bash
cd eve_v3_autonomous_handoff/packages
cat eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01 \
    eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02 \
    > eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip

sha256sum eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip
```

Compare the SHA256 with `source_sha256` in `eve_v3_round96_split_manifest.json`.

## Unzip

```bash
unzip eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip -d round96_source
```

Then ask Codex to continue from `eve_v3_autonomous_handoff/docs/NEXT_ACTIONS.md`.
