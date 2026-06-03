# Round105 AGP proof object expansion

Round105 expands the AGP proof object for approved runtime mapping candidates.

- Added `adapters/agp_proof_object_expansion.py`.
- Proof rows preserve the anchor boundary: explicit category plus SA activation
  only.
- Seed, EveSpecific, and lexical vectors remain evidence and are not AGP
  anchors.
- The expansion does not call AGP verification or apply runtime mapping.
