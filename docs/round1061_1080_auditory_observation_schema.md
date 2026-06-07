# Round1061-1080: Read-Only Auditory Observation Schema

## 1. Overview
EVE v3.1 requires a structured, safe boundary for processing auditory inputs (voice messages, microphone frames, environmental sounds) without immediately triggering state mutations. This round implements the **Read-Only Auditory Observation Schema**, defining a pure-data pipeline for auditory inputs.

This is a schema/design/test-surface round only. **It does not** activate microphones, perform speech-to-text (STT), load audio models, assert voice biometrics, write memory, update self-model, or trigger affect/hormone transitions.

## 2. Supported Auditory Sources
- `voice_message_candidate`
- `microphone_frame_candidate`
- `speech_transcript_candidate`
- `environmental_sound_candidate`
- `notification_sound_candidate`
- `tool_audio_event_candidate`
- `dmn_inner_voice_candidate`

## 3. Supported Features
### Observed Feature Groups
- acoustic_features, speech_presence_features, transcript_metadata_features, prosody_features, environmental_sound_features, notification_sound_features, audio_context_features, safety_relevance_features.

### Inferred Candidate Groups
- speaker_identity_candidate, speaker_role_candidate, speech_content_candidate, prosody_state_candidate, emotion_candidate, intent_candidate, urgency_candidate, environmental_event_candidate, auditory_salience_candidate.

## 4. Key Invariants & Safeguards
1. **No Hardware/Model Activation**: `microphone_activated`, `speech_to_text_performed`, `speaker_recognition_performed`, `audio_model_loaded`, and `voice_biometric_asserted` are strictly `False`.
2. **No Data Persistence**: `raw_data_persisted`, `raw_audio_persisted`, and `raw_video_persisted` are strictly `False`. Raw audio data is not permitted into the schema.
3. **Candidate-Only Enforcement**: Speaker identity, emotion, intent, urgency, prosody, and speech content (unless explicitly provided as transcript metadata) are always marked as candidates and never asserted as facts.
4. **Privacy**: Voice or speech data automatically appends privacy flags (`voice_or_speech_data_requires_privacy_handling`). Private audio metadata triggers additional flags.
5. **No State Mutation**: Memory write, self-model update, affect transition, hormone transition, runtime mutation, vector read/load, and persistence writes are `False`.
6. **Integration Boundaries**:
   - All valid schema outputs properly convert to Sensory Observation Contract candidates.
   - All valid schema outputs properly generate valid Origin / Fact Status Policy records (e.g., `external_audio` / `observed_external`).
   - DMN Inner Voice uniquely generates `dream_dmn` origin and `symbolic_visualization` fact status, rather than external observation.
   - Memory and event candidate plans enforce quarantine and appraisal.

## 5. Next Steps
The recommended next implementation is `read_only_multimodal_sensory_fusion_contract`.
