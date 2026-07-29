# Legacy Speech Routing — content-bearing short-modifier fix

## Baseline

- exact post-cutover base: `a9f70ef78b06744eba01a0b35c60371b10eaf672` — PR #215 squash merge
- historical witness owner: PR #211
- independent witness review/retention staging: PR #212
- Layer 1 only: legacy instruction routing

## Defect

`UserInstructionAdapter.parse_instruction()` previously treated any utterance matching a short-answer phrase as a complete meta instruction. `StreamingEngine.chat_stream()` consumes a returned instruction by emitting an acknowledgement and returning before normal language understanding. Therefore a content-bearing utterance such as:

```text
지금 떠오르는 생각 하나를 짧게 말해줘.
```

could be reduced to the meta instruction and never route the actual question.

## Ruling

`짧게`, `간단하게`, and `간결하게` are modifiers when non-empty content precedes the modifier in the same utterance. In that case the `short` constraint is registered but `parse_instruction()` returns `None`, so the unchanged original text continues through the normal pipeline.

A meta-only utterance such as:

```text
짧게 말해줘.
```

still returns a `short` instruction and keeps the existing early-return acknowledgement behavior.

## Exact regression boundary

The focused regression uses the two actual immutable witness-session inputs:

```text
지금 떠오르는 생각 하나를 짧게 말해줘.
방금 말한 생각과 연결되는 이유를 한 가지 설명해줘.
```

It stops deliberately at `LanguageUnderstanding.parse` and proves routing reach only. It does not assert or improve conversational wording.

## Explicit non-goals

- no `INTENT_POOLS` addition or modification;
- no `language/speech_hub.py` structure change;
- no SpeechHub fixed-pool teardown — deferred to M6;
- no M3-C goal/content selection implementation;
- no rewrite or rerun of the #211 real-phone witness;
- no M3-E affect authority change;
- no persistence or cutover authority change.

The historical witness remains immutable. This repair changes future routing only.