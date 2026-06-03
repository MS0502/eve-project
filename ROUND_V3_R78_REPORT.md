# EVE v3 Round78 Report — lexical-concept candidate schema dry-run

## Summary

Round78 adds a read-only lexical → concept candidate schema dry-run. It converts lexical evidence into candidate rows without enabling runtime mapping, category creation, AGP anchor creation, concept-memory mutation, frame/hypergraph mutation, wrapper lookup, or vector commit.

## Added

- `LexConceptMappingAdapter.candidate_schema_dry_run(tokens=None)`
- `run_round78_lexical_concept_candidate_schema_dry_run(...)`
- `write_round78_lexical_concept_candidate_schema_dry_run(...)`
- `LEXICAL_CONCEPT_CANDIDATE_DRY_RUN_R78.json`

## Candidate row boundary

Each row may contain lexical evidence:

- observed count
- threshold readiness
- context diversity
- EveSpecific candidate flag
- EveSpecific vector presence
- evidence status
- peer-token sample

Each row intentionally keeps concept fields unset:

- `proposed_concept_category = None`
- `category_created = False`
- `agp_anchor_created = False`

## Policy

Unchanged:

- `runtime_mapping_enabled = False`
- `enforcement_enabled = False`
- lexical vector is evidence only
- EveSpecific vector is not an AGP anchor
- AGP anchor still requires explicit category + SA activation

## Smoke result

Using the Round73/74 explicit-commit pattern:

- observations: `민석 오늘`, `민석 군대`
- explicit commit target: `민석`
- candidate rows: 2 (`민석`, `EVE`)
- lexical evidence ready tokens: `민석`
- blocked tokens: `EVE`

## Read-only checks

- audit records unchanged during dry-run
- vector store unchanged during dry-run
- telemetry unchanged during dry-run
- self-learning policy unchanged during dry-run
- category creation: false
- AGP verify call: false

## Next

Round79: candidate evidence quality report.
