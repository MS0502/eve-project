# M3-C-J Controlled Input Readiness Review

## Baseline

This review is based on merged PR #232 and exact current main:

```text
merge/current main: af0b971e366631f9c93eb8c8c0dff87b04828312
rebound database-path digest: 269c89e0e6d5614e2ca86ae5e68b261f3bb0d67bc12bf2045957052cf82ef715
```

The private database path plaintext, private nonce, canonical private input, and
private output root remain outside the repository.

## Reviewed readiness result

The operator created one canonical private input from explicitly controlled new
window material. It does not claim that the supplied drive values were observed
from live EVE state.

The canonical compact JSON readiness result has SHA-256:

```text
61f9fdc30142f7c99ff645ecb4d0ff1263969cc6e88cc9950ad82f20c6f6717f
```

Reviewed identities:

```text
private input digest:
  893694b44997cb0fd1b487e8f031a2a37352ff8aa96a11f00f272b6371172620
private input binding digest:
  e3ec5e68baee57758e24d18ff2cbf6c85866b163fb58b04428ab724ef4388776
selected candidate id:
  8c4428577b1d07713c4de86f0024e452c36eb85f26f14cc987ec35e7845358de
selection receipt digest:
  fcb2dd0aa7c75e19fdb71694d542c38507050fb88ead2bec88af67d9d8c4bd88
```

The deterministic result is `initial_selection`, score
`0.5238636363636363`, transition eligible, with exactly four lifecycle states:

```text
proposed -> validated -> eligible -> selected
```

## Human review and delegation

Execution of the displayed readiness command constituted review approval of the
explicit operator-controlled input values. The project owner then supplied the
single public JSON result and delegated merge plus continuation to the next
bounded stage.

This approval makes one explicit private-device command eligible only after this
audit record is merged. It does not itself issue that command.

## Current boundary

```text
canonical private input created:                 true
nonce permissions reviewed private:              true
single-use paths absent at readiness time:        true
database path digest matched reviewed rebind:     true
live drive observation claimed:                   false
production database accessed:                     false
operator command issued:                          false
real M3-C-J observation window started:           false
#211 phone witness replayed:                      false
retained sequences 1 through 5 replayed:          false
runtime integration authorized:                   false
action/scheduler/speech authority:                false
legacy goal authority transferred:                false
legacy migration authorized:                      false
M3-E authority open:                              false
```

## Validation reuse

PR #232 exact validation remains accepted and is reused. PR #225, #227 through
#232, the #211 witness, and retained sequences 1 through 5 must not be rerun
because of this audit PR, a chat change, a shell change, or a later operator
session.

This documentation tree receives validation only for its own final exact head.
No superseded or intermediate branch head is merge evidence. The full suite must
run at most once on the final PR head.

## Next boundary

After merge, one clean current-main launch may execute
`scripts/operator/m3_c_j_private_device_window_rebound.py` with the already
reviewed private input and nonce. The command must print one public review JSON.
Any partial or completed attempt is immutable: preserve all database, sidecar,
journal, bundle, backup, and restore evidence and do not silently retry.
