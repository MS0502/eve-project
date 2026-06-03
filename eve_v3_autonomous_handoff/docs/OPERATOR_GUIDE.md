# OPERATOR_GUIDE

## Round97/98 operator summary

Round97 opened runtime lexical→concept mapping only inside a controlled smoke path for `민석`, then rolled back. Round98 audited the result and did not persist runtime mapping.

Current flags after Round98:

- `runtime_mapping_enabled=False`
- `enforcement_enabled=False`

## What passed

- Round96 package manifest SHA/size validation and zip integrity.
- Round97/98 focused tests.
- Round92~Round98 focused/adjacent tests.
- Focused compileall for `adapters`, `tests`, and `main.py`.

## What is partial / blocked

- Medium fastText validation is blocked because `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` is absent.
- Full collect-only is partial due legacy root tests importing missing `spreading_activation`.
- Full repository compileall is partial due pre-existing syntax errors in legacy root files.

## Operator decision needed before persistence

Do not persist runtime mapping unless you approve one of these paths:

1. Restore the medium vector artifact and require full validation.
2. Explicitly approve a partial-validation persistence experiment.

In both paths, vectors remain evidence only and must not become AGP anchors.
