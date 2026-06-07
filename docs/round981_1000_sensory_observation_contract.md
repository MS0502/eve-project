# Round981-1000 sensory observation contract

## Track

`read_only_sensory_observation_contract_for_multimodal_eve`

## Scope

Round981-1000 adds a pure read-only contract for future sensory observations.
It is design/test-surface only and does not integrate into live runtime paths.
Raw sensory data must not enter memory or thought directly.  The only allowed
outputs are normalized observation data, event-candidate metadata, confidence
metadata, uncertainty/privacy flags, and gated thought/memory candidate plans.

## Supported modalities and sources

- `text`: `user_text_candidate`, `operator_text_candidate`, `system_text_candidate`
- `visual`: `user_uploaded_image`, `camera_frame_candidate`, `screenshot_candidate`, `screen_ui_candidate`
- `auditory`: `voice_message_candidate`, `microphone_frame_candidate`, `speech_transcript_candidate`, `environmental_sound_candidate`
- `internal_state`: `battery`, `temperature`, `ram_pressure`, `storage_pressure`, `network_state`, `process_health`, `error_state`, `latency_state`
- `tool_state`: `file_change`, `notification`, `calendar_event`, `telegram_event`, `github_event`, `test_result`, `search_result`, `app_state`
- `multimodal`: `multimodal_candidate`, `paired_text_visual_candidate`, `paired_text_audio_candidate`, `paired_visual_audio_candidate`

Unknown modalities and unknown sources fail closed.

## Safety summaries

### Visual observations

Visual observations are normalized candidates only.  They do not activate a
camera, persist raw image/video/screen data, perform OCR, load image models, or
assert user emotion as fact.  Face, person identity, expression, posture, and
similar signals remain inference candidates and set privacy flags when people
or identities may be involved.

### Auditory observations

Auditory observations are normalized candidates only.  They do not activate a
microphone, persist raw audio, perform speech-to-text, load audio models, or
assert user emotion as fact.  Voice, speaker identity, and prosody remain
inference candidates and set privacy flags when voice/person identity may be
involved.

### Internal-state observations

Internal hardware observations are operational-only and non-panic.  Battery,
temperature, RAM, storage, network, process-health, error, and latency inputs
cannot target social, self, or identity axes.  They do not directly mutate
affect/hormone state.

### Tool-state observations

Tool-state observations are path-metadata-only in this round.  They must not
read or carry file contents, message bodies, secrets, tokens, vectors, vocab, or
subset artifacts.  Any future content-bearing tool-state path requires a
separate explicit operator-authorized round.

### Multimodal observations

Multimodal observations may carry normalized candidate fields, but cross-modal
binding is deferred.  No binding, identity merge, memory update, AGP bypass, or
fallback bypass is executed in Round981-1000.

## Confidence, uncertainty, and privacy

Confidence is caller-supplied metadata only.  This module performs no model
inference and does not turn inferred features into facts.  Uncertainty flags are
carried forward for appraisal.  Visual/audio person, face, voice, or identity
signals set privacy review flags and remain candidate-only unless confirmed by
later gated processes.

## AGP and memory compatibility

The observation-to-AGP plan preserves appraisal and AGP input requirements.
Category anchors must still come from EVE internal category activation; sensory
vectors or raw sensory text/media cannot become AGP anchors.  The observation-to
memory plan preserves memory gate and quarantine requirements.  Observation
success never means long-term memory write permission or self-model update
permission.

## Affect/hormone and hardware governor compatibility

Sensory observations do not transition affect or hormones.  One observation
cannot produce all-axis affect/hormone activation, and global synchrony remains
blocked.  Hardware observations remain operational-only and non-panic.

## Non-permissions

The contract keeps all of the following false:

- raw image/audio/video/screen persistence
- OCR/STT execution
- camera/microphone activation
- image/audio model loading
- vector read/load and vector content read
- memory write and self-model update
- affect/hormone transition
- AGP/fallback bypass
- persistence/runtime mutation
- artifact creation or staging
- runtime mapping default enablement
- enforcement default enablement

## Operator command

```bash
python scripts/operator_report_round981_1000_sensory_observation_contract.py
```

The command emits compact JSON with schema examples, fail-closed examples,
privacy proofs, non-persistence proofs, AGP/memory compatibility proofs,
affect-ladder/future-preflight compatibility proofs, artifact safety proof, and
exactly one next implementation recommendation.

## Next implementation recommendation

Add a read-only operator-authorized sensory adapter preflight that accepts this
contract output and still performs no OCR, STT, device activation, model load,
memory write, or runtime mutation.
