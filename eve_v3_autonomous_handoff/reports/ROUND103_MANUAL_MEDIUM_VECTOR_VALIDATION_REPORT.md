# Round103 manual medium vector validation

Round103 adds a fail-closed manual validation checkpoint for the medium 30k
vector artifact after the Round102 release-restore workflow.

- Added `adapters/medium_vector_manual_validation.py`.
- Validation is read-only and never installs, creates, or commits `vectors.npy`.
- Fake, missing, wrong-shape, or checksum-mismatched candidates remain blocked.
- Operator validation must pass before runtime mapping persistence review can
  proceed.
