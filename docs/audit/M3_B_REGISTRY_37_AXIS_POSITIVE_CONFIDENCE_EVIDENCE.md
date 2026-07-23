# M3-B Registry 37-Axis Positive-Confidence Observation Evidence Contract

## 1. Baseline and purpose

- Baseline: `379d912c4e863b2a692d2c20f9f8113dfa7219cd`
- Predecessor: PR #176 combined 63-axis observation packet preflight
- Remaining blocker: `REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE`

PR #173 established the detached 37-axis current-value owner. Its deterministic genesis initializes current values but is not observation evidence and therefore carries confidence `0.0` on every registry axis. PR #176 proved that the combined packet is structurally complete while preserving this blocker.

This artifact defines the only accepted shape for promoting all 37 registry axes to positive-confidence observed current values. It does not collect real production observations and does not start an observation window.

## 2. Constitutional boundary

The contract follows the v4.2 observation and evidence-recalculability requirements:

- every axis record carries origin, source identity, acquisition method, confidence, logical time, verification method, model/rule version, and immutable SHA-256 references;
- green verdicts, registry definitions, defaults, baselines, genesis values, and event proposal metadata are not observation evidence;
- raw observations must be embedded elsewhere or referenced by non-placeholder SHA-256 plus schema version so the claimed value can be independently recalculated;
- all returned state remains detached, immutable, and `shadow_only`.

## 3. Exact per-axis evidence

`RegistryAxisPositiveConfidenceEvidence` requires one record for each canonical registry axis with:

- exact axis name;
- finite current value inside the registry's declared `[min, max]` range;
- confidence in `(0, 1]`;
- observation logical tick;
- unique observation ID;
- source family, source instance, source snapshot, and source schema version;
- source integrity SHA-256;
- raw observation SHA-256;
- acquisition method;
- verification method;
- model or rule version;
- exact observation kind `verified_current_value_observation`;
- exact verification status `verified`;
- recalculable raw reference present.

The registry owner itself cannot be its own evidence source.

## 4. Fail-closed exclusions

A record is rejected when any of the following is true:

- confidence is `0.0` or otherwise outside `(0, 1]`;
- value is non-finite or outside declared bounds;
- axis is unknown;
- source or raw digest is malformed or the all-zero placeholder;
- raw recalculation reference is absent;
- source family is the registry owner family, creating circular provenance;
- the record is marked as genesis-derived, baseline-derived, default-derived, proposal-only, or synthetic;
- observation kind, verification status, or schema version does not match the versioned contract.

An observed value may numerically equal its baseline. Equality is not rejected because the distinction is provenance, not numerical distance. What is forbidden is deriving the value from the baseline and presenting that derivation as an observation.

## 5. Exact 37-axis bundle

`RegistryPositiveConfidenceEvidenceBundle` requires:

- exactly 37 immutable evidence records;
- exact canonical registry order;
- unique axis names and unique observation IDs;
- strictly positive confidence for every axis;
- every observation tick at or before the bundle logical tick;
- target owner instance ID;
- exact expected predecessor owner digest;
- exact next state sequence;
- source-manifest schema version and non-placeholder SHA-256;
- verification authorization ID and acceptance-policy version.

The bundle is rejected if it claims any runtime hook, scheduler, persistence access, event append, live mutation, observation-window state, M3-B completion, M3-C opening, cutover authorization, or M3-E authority.

## 6. Detached owner materialization

`materialize_registry_observed_owner()`:

1. verifies the exact immutable owner and bundle types;
2. binds the bundle to owner identity, predecessor digest, next sequence, and monotonic logical time;
3. rejects observations older than the predecessor owner;
4. verifies that owner axis bounds still match the active registry definitions;
5. returns a new immutable `RegistryAffectOwnerState` with the observed value and confidence for each axis;
6. binds the owner transition digest to the evidence bundle digest;
7. leaves the predecessor owner unchanged.

The returned owner remains disconnected and `shadow_only`. No storage, event log, scheduler, runtime route, or live state is touched.

## 7. Recalculable audit fixture

The audit harness constructs a deterministic contract-only fixture containing all 37 axes. It proves:

- exact 37-axis positive-confidence coverage;
- deterministic bundle and owner digests;
- predecessor owner immutability;
- rejection of zero-confidence, genesis, baseline, default, proposal-only, synthetic, and missing-raw-reference forms;
- a combined fixture packet computes 63 positive-confidence axes and no confidence blocker;
- despite that calculated eligibility, no production observation window is started or satisfied.

The audit fixture is explicitly not production observation evidence. It cannot authorize M3-B completion or any later milestone.

## 8. Authority status after this artifact

The following remain `false`:

- runtime hook installed;
- scheduler installed;
- persistence accessed;
- event appended;
- live affect, drive, named-state, goal, memory, self, or expression mutation;
- observation window started or satisfied;
- M3-B complete;
- M3-C open;
- M3-E authority open;
- cutover authorized.

Legacy runtime and legacy persistence remain authoritative.

## 9. Next required artifact

A later artifact must capture **real, independently recalculable observed current values for all 37 registry axes** and bind them to this contract. Only after that real evidence passes exact-head validation and human review may a separate decision artifact evaluate whether the observation window can start.
