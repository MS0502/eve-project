# ROUND V3 R97 CODE-ONLY PACKAGE STATUS

## Goal

Switch the Round96 restore path from the problematic legacy split package to the preferred 7.7MB code-only package when present.

## Preferred package inputs

```text
eve_v3_autonomous_handoff/packages/eve_v3_round96_code_only_no_medium_vectors.zip
eve_v3_autonomous_handoff/packages/eve_v3_round96_code_only_manifest.json
```

## Known omission

The code-only package intentionally excludes only:

```text
seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
```

Round95~Round96 focused/adjacent validation and Round97 controlled runtime mapping enable smoke should use this code-only source first. Any full validation path requiring the missing medium vector must be recorded as blocked or partial.

## Implementation update

`eve_v3_autonomous_handoff/packages/restore_round96_package.py` now prefers the code-only package. It still supports the legacy split package if the code-only zip and manifest are absent.

The helper performs:

1. package input discovery,
2. SHA-256 comparison against manifest (`source_sha256`, `zip_sha256`, or `sha256`),
3. zip integrity test,
4. extraction to `eve_v3_autonomous_handoff/packages/round96_source/` unless `--verify-only` is used.

## Current checkout status

The code-only package files are not visible in this execution checkout. The package directory still contains only the README and restore helper, so Round96 extraction and Round97 implementation remain blocked in this environment.

## Safety state

No Round96 source was extracted and no runtime mutation was attempted.

- `runtime_mapping_enabled`: unchanged / not enabled in this checkout
- `enforcement_enabled`: unchanged / not enabled in this checkout
- category creation: not performed
- concept memory mutation: not performed
- SA activation mutation: not performed
- AGP state mutation: not performed
