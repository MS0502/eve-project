# Persistence Integrity and Semantic Rebaseline Preconditions

Status: **governance/design pin only**. This artifact creates no runtime capability, performs no persistence activation, does not execute the M3-C phone workflow, does not transfer legacy goal authority, does not open M3-E, and does not itself amend the active EVE v4.2 Constitution.

## 1. Accepted prerequisite and operational boundary

Repository cross-check at creation time confirms PR #243 is merged. Its accepted code head is `a4c8d0ec1a1767b5ccdbc105c40af94a327eb741`; its squash merge commit is `d9491f6b1dd2149338e37bb199274b63636e66f4`; and `main` was directly observed at that same merge commit before this governance branch was created. PR #243 changed exactly 15 files.

This cross-check seals the M3-C-R code-acceptance state. It does **not** satisfy the separate M3-C-S operational witness. Issue #244 remains the next live phone event. Chat, shell, branch, PR metadata, or agent-session changes alone do not invalidate accepted exact-head evidence. An unverified agent report is never promoted to repository authority merely by repetition: a reported SHA is either directly cross-checked or remains in **comparison-pending** state, not revalidation-pending state.

## 2. Persistence-integrity requirements are unconditional

Hardware-endurance telemetry may determine hardware responses such as SSD replacement, write-budget changes, or UPS deployment. It does not gate or waive consistency requirements.

For every authoritative SQLite event store, the following requirements are binding design/acceptance preconditions:

1. WAL is used where the supported SQLite/platform configuration permits it.
2. The authoritative store pins `PRAGMA synchronous=FULL`. A weaker synchronous mode may not be substituted for production authority merely because measured write volume is low.
3. Before startup accepts any new authoritative mutation, persisted event-tail integrity is verified against the store's accepted chain/tail metadata using the versioned canonical digest method for that store.
4. A tail mismatch, impossible chain continuation, or unverifiable accepted tail fails closed. Startup must not silently truncate, rewrite, repair, or continue authoritative mutation past the discrepancy.
5. Power-loss/forced-termination fault injection is a required acceptance test for authoritative persistence and cutover paths. It must exercise transaction interruption, restart verification, last-valid-state recovery, and refusal of corrupted or unverifiable tails.
6. Checkpoint/snapshot restore and event-tail verification must compose: selecting a valid snapshot does not authorize skipping subsequent event-chain verification.
7. Write telemetry (`bytes/day`, checkpoint bytes, WAL/checkpoint amplification, and projected storage endurance where measurable) informs hardware policy only. It cannot downgrade items 1-6.

This section pins policy only. Any runtime change needed to satisfy it must be implemented and validated in its own scoped change.

## 3. Representation/authority audit — mandatory deliverables before workspace implementation

No semantic workspace implementation may begin from an assumption that existing representation layers are already subordinate views. The audit must classify the actual repository state first.

For every relevant representation/module — including category/SA, VSA, frame/hypergraph, hyperbolic or other concept coordinates, lexical vectors/mappings, semantic memory, working memory, episodic references, goal/affect-linked semantic state, and any additional discovered owner — record at minimum:

- module/path and concrete state surface;
- current mutation authority and writer(s);
- current persistence or checkpoint path, if any;
- producer and consumer edges;
- replay/reconstruction source, if any;
- current classification: `AUTHORITY`, `PROJECTION`, `CACHE`, `INDEX`, or `TRANSIENT_VIEW`;
- proposed disposition after consolidation: retain authority, derive/project, demote, absorb, freeze, migrate, or retire;
- provenance/replay implications;
- rollback implications;
- unresolved conflicts or multiple-authority cases.

### 3.1 Migration cost is a required audit field

Every module/disposition row must include a migration-cost estimate before v4.3 consolidation is approved. The estimate must not be a vague adjective alone. It must identify, where applicable:

- estimated files/modules and authoritative state surfaces touched;
- schema or serialization changes;
- required state conversion, backfill, or re-derivation;
- dependent projection invalidation/rebuild scope;
- replay and historical-compatibility work;
- rollback/checkpoint work;
- focused, full-suite, invariance, artifact, and operational evidence burden;
- migration risk and known blockers.

Unknown cost is recorded as `UNRESOLVED`, not silently treated as zero. The purpose of the audit is to expose whether workspace consolidation is larger than prior M1/M2 work before a constitutional or implementation commitment is made.

The audit itself adds **zero runtime capability**.

## 4. Minimal grounding contract precedes workspace design

After the representation/authority audit and before semantic-workspace consolidation design, a minimal grounding contract must be accepted. Implementation of the environment may occur later; the contract is the requirement input to workspace design.

The workspace must be capable of representing and preserving evidence for at least:

1. **Object permanence** — an observed referent may persist across time without treating persistence as an unquestionable fact.
2. **Action consequence** — an action, its pre-state, outcome observation, and prediction/error relation can be represented with provenance.
3. **Occlusion/reappearance** — disappearance from observation and later reappearance can update a referent hypothesis without rewriting source events.
4. **Revision evidence** — contradictory observations can revise, merge, split, or quarantine identity hypotheses while retaining the original observations.

A design that cannot represent these cases is not eligible to become the semantic workspace, even if it can represent labels or text-derived concepts.

Dependency order is therefore:

`representation/authority audit (+ migration cost) -> minimal grounding contract -> workspace consolidation design -> grounding implementation/benchmarks -> later language-capability opening`

## 5. Candidate v4.3 amendment requirements

The following are pinned as required inputs to a later reviewed v4.3 constitutional amendment. They are **not active constitutional clauses merely by appearing in this artifact**.

### A13 — Semantic Authority Consolidation

A new semantic representation must not create an additional authoritative source of truth by default. Every existing representation must be classified as authority, projection, cache, index, or transient view, and workspace introduction must name what is retained, absorbed, demoted, migrated, frozen, or retired.

### A14 — Referent Identity as Hypothesis

Observation events are preserved. Referent identity and same-object judgments are confidence-bearing, provenance-bearing hypotheses rather than authoritative external-world facts. Referent state must be reconstructable as a versioned projection from retained evidence.

Identity revision includes merge and split, not only same/not-same decisions. A merge or split invalidates every dependent projection whose derivation relied on the superseded identity hypothesis and requires re-derivation. The implementation contract must name an explicit propagation-depth/work bound. Dependencies beyond that bound may not remain silently authoritative: they must be marked invalid/stale or quarantined and fail closed until a bounded continuation/rebuild policy resolves them.

### A15 — Learned Artifact Reproducibility and Rollback

Deterministic core state retains its applicable exact/byte-identical replay gates. Learned artifacts are pinned by artifact hash, parent checkpoint, training-event lineage, data manifest, code/runtime provenance, evaluation suite, and declared numerical tolerance where bitwise backend reproduction is not guaranteed.

A lineage is not accepted merely because metadata matches. Re-execution of the same declared lineage must pass the same named evaluation suite within its predeclared tolerance. Failure rejects the candidate artifact; it may not silently replace the previously accepted artifact.

Rollback policy is **versioned checkpoint selection**, not silent retention of future learned state. Learned artifacts that can affect decisions must be versioned. A rollback to a logical point selects the latest compatible accepted learned checkpoint at or before the rollback boundary and replays forward under the recorded lineage. If no compatible checkpoint/artifact lineage exists, rollback fails closed. A rollback must never replay historical state while retaining a learned artifact that only existed after the requested rollback point.

### A16 — Semantic Conservation of Expression

The expression subsystem has no proposition authority. It may not add an entity, predicate, semantic relation, negation, modality, temporal assertion, causal assertion, certainty claim, or other propositional content absent from its authorized semantic inputs.

Korean morphosyntactic obligations are an explicit bounded exception. Particles, inflection, sentence endings, register/honorific realization, and affect-conditioned surface tone may be supplied only from named internal state or grammatical structure. Examples of required sourcing include register/honorific choice from relationship/user-presence state, affective tone from accepted affect state, and tense/aspect/speech-act morphology from the authorized semantic/speech-act structure. These additions may realize existing semantics but may not invent new propositions. If a required source is absent or contradictory, expression fails closed/refuses rather than guessing.

`empty semantic input -> no semantic utterance` remains a minimum sanity gate, not the complete conservation test.

### A17 — Grounded Language Capability Ladder

Language capability is staged rather than permanently capped. No level is open by declaration alone. Every level requires a versioned, falsifiable acceptance artifact with a named benchmark, held-out cases, predeclared pass/fail conditions, provenance checks, and negative tests demonstrating the boundary below.

- **L0 — grounded referent/action vocabulary.** Opens only after the minimal grounding benchmark proves that enabled lexical labels map to non-linguistic observation/action evidence and still preserve correct referent/action distinctions across held-out scene permutations and label-masking/label-substitution negative tests.
- **L1 — environment description and instruction.** Opens only after held-out environment commands/descriptions are grounded to executable or independently checkable semantic structures, with expression semantic-conservation checks and refusal on absent grounding.
- **L2 — relationship and autobiographical language.** Opens only after claims are traceable to accepted episodic/relationship/self-state provenance, contradictory-source tests trigger uncertainty/revision rather than fabrication, and removal of required provenance produces refusal or explicit uncertainty.
- **L3 — externally observed text assimilation.** Closed by default. Any opening requires proof that raw external text remains `provenance=external`, no raw-text-to-expression capability edge exists, accepted claims survive quarantine/verification rules, and held-out adversarial or conflicting-source tests do not promote source wording into internal fact or direct speech.
- **L4 — long-form/abstract grounded language.** Closed by default. Opening requires an accepted benchmark showing cross-episode/abstract semantic composition with retained provenance, revision, and semantic-conservation properties beyond L3.
- **L5 — open-domain generalization.** Closed by default. Opening requires an explicit research acceptance package demonstrating bounded open-domain generalization without bypassing grounding, provenance, quarantine, or expression authority.

L3 and above are explicitly treated as **currently unresolved research problems for this project**, not scheduled capabilities or promises. Their presence in the ladder defines gates and non-bypass conditions; it does not claim that a known implementation path exists.

## 6. Sequence and non-bypass rule

Nothing in this artifact authorizes work to jump ahead of Issue #244's M3-C-S operational witness. The immediate live-runtime sequence remains unchanged. This artifact exists so that persistence integrity and the later representation audit do not begin with missing acceptance criteria when their turn arrives.
