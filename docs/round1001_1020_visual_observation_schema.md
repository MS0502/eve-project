# Round1001-1020 visual observation schema

Track: `read_only_visual_observation_schema_from_sensory_contract`

## Scope

This round adds a pure, read-only visual observation schema derived from the
Round981-1000 sensory observation contract. The schema accepts only structured
visual observation metadata and candidate feature dictionaries. It does not
connect a camera, process real images, perform OCR, load vision models, perform
face recognition, persist raw visual data, write memory, update the self-model,
mutate affect or hormone state, read/load vectors, mutate runtime state, or
create operator artifacts.

## Supported visual sources

- `user_uploaded_image`
- `camera_frame_candidate`
- `screenshot_candidate`
- `screen_ui_candidate`

Unknown visual sources fail closed.

## Observed feature groups

- `scene_features`
- `object_features`
- `person_presence_features`
- `screen_ui_features`
- `lighting_features`
- `motion_features`
- `safety_relevance_features`

## Inferred candidate groups

- `person_identity_candidate`
- `expression_candidate`
- `posture_candidate`
- `gaze_candidate`
- `activity_candidate`
- `place_candidate`
- `screen_state_candidate`
- `visual_salience_candidate`

All inferred visual groups are candidate-only. They must not assert person
identity, user emotion, user intent, relationship state, or memory facts.

## Gate compatibility

Visual observations require appraisal, AGP input preparation, sensory contract
compatibility, and memory gates. Visual-to-event plans produce candidates only.
Visual-to-memory plans preserve quarantine/appraisal gates and do not allow
long-term memory or self-model writes.

## Privacy and uncertainty policy

Person/face, screenshot, screen/UI, and private-scene observations receive
privacy flags. Caller-supplied confidence values below `0.50` add uncertainty
flags; they do not become fact assertions.

## Non-persistence and no-device policy

The schema keeps raw image/video/screen-recording persistence false. Camera
activation, OCR, vision model loading, and face recognition are false for every
visual observation, including camera-frame and screenshot/screen candidates.

## Affect/hormone and anti-global-synchrony policy

A visual observation cannot directly update affect or hormone state and cannot
produce all-axis affect/hormone activation. Global synchrony remains blocked.

## Operator report

Run:

```bash
python scripts/operator_report_round1001_1020_visual_observation_schema.py
```

The report emits compact JSON containing schema/source/feature summaries,
candidate-only and non-assertion proofs, privacy and uncertainty proofs,
sensory-contract compatibility, visual-to-event and visual-to-memory plans,
affect/hormone compatibility, raw visual non-persistence, no device/model/OCR
proofs, no mutation/vector/artifact proofs, anti-global-synchrony proof, and
exactly one recommended next implementation step.
