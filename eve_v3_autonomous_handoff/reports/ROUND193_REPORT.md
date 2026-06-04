# Round193 guarded runtime path selection

Round193 selects exactly one runtime/test path that may call the guarded explicit medium30k loader:

```text
main.build_full_engine(..., operator_medium30k_load_authorized=True, operator_medium30k_validation=green_report)
```

The path is allowed only when the operator validation report is green, the validation command was run with `--attempt-load`, and the caller explicitly passes load authorization. Default `build_full_engine()` remains no-load. Runtime mapping and enforcement requests are not valid reasons to load.
