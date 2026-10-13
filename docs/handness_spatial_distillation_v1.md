# Handness spatial distillation v1

Date: 2026-09-28

## Goal

Expand the Handness training set and add spatial supervision without using
ImageNet weights or calling Qwen at inference time.

## Data expansion

- Existing reviewed/distillation manifest: 12,175 views, 229 recordings.
- Added track-package manifest: 2,417 views, 18 recordings, 108 tracks.
- Combined manifest: 14,592 views, 244 recordings, 404 tracks.
- Recording-level splits are preserved. One overlapping recording had a split
  mismatch; 202 views were reconciled to the existing recording split.

The spatial Qwen teacher received both the full context frame and the track
crop. It returned hand presence plus a normalized hand box inside the crop.
All 2,417 responses parsed successfully.

## Supervision policy

- Human track label and Qwen frame presence agree: use for classification and
  spatial supervision.
- Human track label and Qwen frame presence conflict: set the frame to unsure
  and exclude it from training.
- Teacher-only track: exclude until blind human review.

Accepted spatial supervision:

- 1,814 hand views with a Qwen hand box.
- 13 nothand views with an empty heatmap.
- 21 track-label/frame-presence conflicts excluded.
- 569 teacher-only views from 38 tracks held for blind review.

## Student

The student is a randomly initialized ResNet-18. It has two outputs:

1. A hand-presence logit trained from human labels and Qwen soft targets.
2. A 7x7 heatmap trained from Qwen boxes on reviewed-agreement views.

The inference path is Qwen-free. The spatial output is an auxiliary training
signal; the runtime gate consumes the hand-presence probability.

## Pilot result

Three random seeds were trained in parallel and ensembled. A threshold was
selected on validation data to require at least 99.5% human-hand recall.

| Split | Human hand recall | Nothand specificity | Human AUC |
| --- | ---: | ---: | ---: |
| Validation | 99.54% (1286/1292) | 4.35% (3/69) | 0.7626 |
| Test | 99.67% (298/299) | 11.84% (9/76) | 0.8699 |

Frozen threshold: 0.278254.

This is an improvement over the earlier zero-rejection student, but it is not
strong enough to deploy. The model remains fail-open and no rendering pipeline
was changed.

## Next gate

Blind-review the 38 held-out tracks in
`handness_spatial_expand_review_packet`. Merge those labels, retrain, and only
render new videos if track-level evaluation retains every reviewed hand track
while rejecting a useful fraction of reviewed non-hand tracks.

## Blind track review update

The 38 held-out tracks were fully reviewed: 36 hand and 2 nothand, with no
missing or duplicate labels. Applying these labels added 558 trusted hand
boxes and 2 trusted nothand heatmaps. Nine hand-track frames disagreed with the
single-frame teacher and remain excluded. The combined trusted spatial set is
now 2,372 hand views and 15 nothand views.

Retraining the crop-only student produced only a small test improvement at the
99.5% recall operating point: nothand specificity increased from 11.8% to
13.0%. The one newly held-out nothand example, a person's head, was still
classified as hand with probability 0.772.

A dual-view student was then tested because Qwen receives both crop and full
context. At the untuned 0.5 threshold, test nothand specificity increased to
23.4% while hand recall was 99.69%. However, the validation-selected threshold
did not transfer: it reduced test hand recall to 97.82%. The head false positive
also increased to 0.897 because the full frame did not mark which region the
crop came from. This model is rejected for deployment.

Track-level evaluation exposed the sampling limitation directly. The test set
has no human-labeled nothand track with at least three views, so all negative
tracks must fail open. Searching all 5,786 unlabeled teacher-scored views found
zero unreviewed Qwen-negative tracks with at least three sampled views.

The two reviewed singleton negatives were therefore propagated conservatively
to nearby packaged frames, producing a new blind packet with 2 tracks and 10
views. These propagated views require human confirmation before training.
