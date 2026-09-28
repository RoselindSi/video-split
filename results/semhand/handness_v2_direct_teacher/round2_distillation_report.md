# Handness round-two distillation report

Date: 2026-09-28

## Human review

- Round two: 6/6 tracks complete, 2 `nothand`, 4 `hand`.
- Round-two selected views: 9 `nothand`, 20 `hand`.
- Combined rounds: 15 tracks, 8 `nothand`, 7 `hand`.
- Four non-hand tracks have at least three reviewed views and are eligible for
  an automatic track decision; they occur in two recordings.

The four hand tracks among six second-round teacher/student disagreements show
that direct Qwen negative predictions cannot be used as automatic truth in
this enriched region. Human labels are the only hard targets.

## Frozen evaluation policy

1. Hold out a complete negative recording.
2. Exclude it from checkpoint and threshold selection.
3. Freeze the highest threshold retaining at least 99.5% of validation hands.
4. Fewer than three views fails open.
5. Any above-threshold view rescues a track.
6. Reject only when at least three views are all below threshold.

## Results

| Student | Held-out recording | Threshold | Legacy test hand recall | Eligible non-hand tracks rejected |
|---|---|---:|---:|---:|
| Qwen pooled feature MLP | R0824_120917 | 0.37994 | 159/165 (96.36%) | 0/1 |
| Qwen pooled feature MLP | R0828_105127 | 0.57657 | 162/165 (98.18%) | 0/3 |
| Pixel student, random init | R0824_120917 | 0.32465 | 165/165 (100%) | 0/1 |
| Pixel student, random init | R0828_105127 | 0.33908 | 164/165 (99.39%) | 0/3 |
| Pixel + Qwen global-feature pilot | R0824_120917 | 0.31915 | 160/165 (96.97%) | 0/1 |

## Decision

No distilled head passed. Across the complete two-fold endpoint, both full
students rejected 0/4 eligible held-out body-part tracks. The pixel student
also missed the 99.5% hand-recall floor in one fold; the pooled-feature model
missed it in both. The feature-regression pilot did not justify expansion.

Do not connect these checkpoints to rendering. Rendering a video batch with
them would show a known unsafe model rather than a validated speed replacement.

## Next experiment

Mine substantially more upper-arm, elbow, thigh, knee, torso, and object-plus-
finger negatives from many recordings. Query Qwen only offline, retain human
overrides, and distill spatial supervision such as patch tokens, masks, or
teacher-localized hand regions into a Qwen-free student. Freeze the next model
before evaluating on new recordings with both positive and negative tracks.
