# M2-E Bounded Persistence Cutover Candidate

## Status

- Baseline: `c59095ccf75419e40107ec03fd20761ee946543d`
- M2-D accepted prerequisite: PR #165
- Technical candidate: in progress
- Human acceptance: not performed
- Cutover authority: not granted
- Production integration: none
- Legacy persistence authority: retained

This branch starts the bounded M2-E technical candidate only. It does not make the event store authoritative, change production defaults, install a runtime observer or lifecycle bridge, enable production dual read, or convert legacy sidecars to read-only evidence.

The candidate will define an immutable external human-decision contract, a caller-invoked cutover mechanism for the single accepted `ActivationAdapter.learn_pair` state envelope, a deterministic post-cutover observation window, and rollback evidence. The final exact head, workflow run, artifact name and SHA-256 must exist before any informed human acceptance can be recorded.
