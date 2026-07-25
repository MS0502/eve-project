# M3-B C1 Operator Attestation Trust Root

## Baseline and reused prerequisite

C1 is rebased onto PR #196 squash merge `5664fc3bc22054c2d39142b3125416aea6089c63`.
The exact-head validation for #196 is reused under the repository validation policy:

```text
exact head:   4944c01df3b0978ae73ea3060abd39bee14e41c1
exact run:    30179233468
focused:      4 passed
full:         3,138 passed
artifact:     exact-head-validation-4944c01df3b0978ae73ea3060abd39bee14e41c1
artifact SHA: 4f803b16343871b6e20517676cd2a0a4ebce7231444fcd4158fb0926320181b8
M2-E run:     30179233476
M2-E:         6/6 jobs passed
merge SHA:    5664fc3bc22054c2d39142b3125416aea6089c63
```

PR #195 remains inherited through #196 merge ancestry and is not rerun merely because C1 moved to a newer reviewed base. A new chat, PR body edit, review, or Draft/Ready transition does not invalidate the reused prerequisite evidence.

## Threat closed by C1

PR #190 correctly rejected PID, argv/environment values, `fixture_only=False`, caller-selected
runtime/source IDs, entrypoint identity, and self-hashed launch metadata as production
provenance. It also deliberately left the production runtime-provenance verifier registry
empty because no independently reviewable trust root existed.

C1 closes the next design gap without fabricating a real launch:

- a runtime may construct public metadata, but it cannot make that metadata reviewed;
- a public digest may be copied, so a digest alone is never a trust decision;
- private operator nonce material must never enter repository/runtime evidence;
- an operator-private nonce binding is verified locally before review;
- only a separately reviewed exact public-attestation digest may later be pinned into the
  immutable repository registry;
- an unreviewed attestation remains unusable even when it looks production-shaped.

## One-operator trust domain

C1 defines exactly one trust-domain identity:

```text
trust domain: eve.operator-attestation.primary.v1
operator id:  primary-operator
```

This is an operator trust boundary, not an EVE identity/reward relationship. It grants no
control over EVE cognition, goals, memory, affect, expression, or cutover authority.

## Private/public split

The operator-local launch binding contains only bounded public launch facts:

- runtime instance ID;
- source instance ID;
- exact deployed repository head SHA;
- entrypoint ID;
- launch-attestation ID;
- logical tick;
- fixture classification.

The operator holds a private nonce of at least 32 bytes outside the repository. The local
attestation command derives:

```text
private_nonce_commitment_digest = SHA256(private_nonce)
nonce_binding_digest             = HMAC-SHA256(private_nonce, canonical_launch_binding)
```

Only those digests and the public launch facts may leave the private companion. The nonce
itself is never serialized by the C1 module or printed by the operator CLI.

The HMAC is **not** treated as publicly self-verifying. Its purpose is to bind a private
operator-held value to the exact launch facts so the operator can recompute the binding
locally during review. Repository/runtime code receives only the digest-only record.

## Review boundary

`REVIEWED_OPERATOR_ATTESTATIONS` is an immutable `MappingProxyType` and is deliberately
empty in C1. A later review PR may add an entry only after the operator locally recomputes
the private binding and records a digest-only review trace.

Runtime-side acceptance requires all of the following:

1. exact public-attestation schema;
2. exact candidate/attestation equality for trust domain, runtime/source IDs, deployed head,
   entrypoint, attestation ID, logical tick, fixture classification, and attestation digest;
3. exact attestation digest present in `REVIEWED_OPERATOR_ATTESTATIONS`;
4. exact registry equality for nonce commitment/binding digests and all launch fields;
5. issuance through the private reviewed-verification token.

Direct construction and `dataclasses.replace(...)` cannot manufacture a reviewed verification.

## Operator CLI

`scripts/operator/m3_b_operator_attestation.py` has two commands:

- `attest`: reads an operator-private nonce outside the repository and prints only the public
  attestation JSON;
- `verify-local`: recomputes the private binding against a public attestation and prints only
  a digest-only review summary.

The CLI generates no nonce, stores no secret, mutates no EVE runtime state, and does not
register an attestation. Secret generation/provisioning is an explicit operator action so
repository code never silently creates a trust credential.

## C1 state after merge

```text
operator attestation trust-root contract:       present
one-operator trust-domain identity:              present
private nonce in repository/runtime evidence:    0
reviewed real operator launch attestations:      0
registered runtime provenance verifiers:         0
verified production runtime anchors:             0
registered production source verifiers:          0/37
retained real observation:                        0/37
positive-confidence real observation:             0/37
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                     false
M3-C open:                                         false
M3-E authority open:                               false
cutover authorized:                                false
```

C1 does **not** claim a real phone attestation exists. It creates the trust-root contract and
operator-local recomputation surface needed to review one honestly.

## C2 entry condition

C2 may proceed only after the operator creates and locally verifies a real phone launch
attestation from the private companion. The public record and digest-only review summary may
then be reviewed and pinned in a later exact-head PR. Only that later PR may integrate an
executable runtime-provenance verifier and a source-contract verifier and attempt the first
retained positive-confidence real observation.

One observation is one observation. It must not be reported as 37/37 production coverage.
No operator attestation, source registration, retained observation, machine-green result, or
observation-window seal may automatically open M3-C/M3-E or authorize cutover.
