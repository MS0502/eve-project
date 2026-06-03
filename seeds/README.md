# EVE v3 External Seeds

This directory is the controlled entry point for external lexical/concept seed data.

Round21 commits only the manifest structure and validator. It does **not** import,
load, train, or ship any external seed file.

Policy boundary:

- External generation bodies remain forbidden.
- External lexical/concept seeds are initial maps only, never a speaking subject.
- Every seed must record provenance, license, version, checksum, and import round.
- Future seed updates must be deterministic and drift-tracked.
- AGP anchors must remain EVE internal category activation, not seed vector space.
