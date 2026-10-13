# Final-test v1 preregistration

Frozen on 2026-09-12, before any selected recording was annotated, any model
was run on it, or any final-test output was inspected.

## Question

Can a taxonomy-free segmentation pipeline distinguish predictable repetition
inside one continuous interaction instance from a new instance of the same
action after an observable disengagement, release, reset, or restart?

The primary manuscript claim is a pipeline comparison. Annotator consistency
is quality control for the answer key, not the research target.

## Selection

- Population: 471 recordings in the two 10 fps recording corpus parts.
- Excluded: 329 IDs present in prior `data/gold`, plus
  `recording_000003`, whose `segments.json` was opened while checking the remote
  directory structure.
- Eligible: 141 recordings with a non-empty, ffprobe-readable `mid.mp4` of at
  least 30 seconds.
- Selected: 48 recordings by ascending
  `SHA256(salt + NUL + recording_id)`.
- Salt: `taxonomy-free-episode-boundary-final-v1-2026-09-12`.
- Total selected duration: 509.4 minutes.

Selection used no task category, action label, segment, boundary, candidate,
model output, or score. In particular, the sampler never opens `segments.json`.
The selected IDs and technical metadata live in
`data/splits/final_test_v1.json`.

## Correction trail

The first mechanical draw contained `recording_000003`. Before any model or
annotation work, validation noticed that this recording was one of three whose
`segments.json` had been printed during an earlier corpus-layout check. IDs 1
and 2 were already excluded by prior gold; ID 3 was added to the explicit
manual exclusion file and the deterministic draw was rerun. The discarded
first draw is not an experiment and no score was computed from it.

## Frozen hashes

```text
final_test_v1.json
  d886a58154887093ac426e19b3e1bb2f1d138debbe260e077ab401bacef36a75
final_test_v1_excluded_recordings.txt
  a46bf0abadcb8bad3728db29ff0d23841ea167395f325b756e9810efd096f44a
```

The manifest also embeds the exclusion-list hash. A mismatch invalidates the
split until explained and re-frozen before any scoring.

## Test isolation

Before the exhaustive gold is frozen, a method may produce and cache blind
outputs, but nobody may inspect those outputs. The 48 recordings may not enter
training, PCA, scaling, class weighting, early stopping, threshold selection,
duration priors, hyperparameter selection, loss weighting, failure mining, or
architecture choice.

After gold and all method versions are frozen, each method is scored once.
There is no final-test ablation and no rerun after seeing a result.

## Answer key

The answer key is built in two stages. First, the natural test receives an
exhaustive full-timeline pass that records all observable boundary events:

- `new_action`: boundary.
- `same_action_new_instance`: boundary.
- `cannot_determine`: excluded from binary scoring and reported.
- `initial_action_start` and `terminal_action_end`: reported separately.

This first pass sees only raw video and does not show prior segments, method
candidates, scores, or model output. It does not attempt to mark every instant
that is not a boundary.

Second, after every compared method and threshold is frozen, their candidates
are unioned and deduplicated within one second. Each shared candidate receives
one of `new_action`, `same_action_new_instance`, `same_instance`, or
`cannot_determine`. Within `same_instance`, periodic repetition, direction
reversal, regrasp/reposition, spatial phase shift, and other are recorded as
diagnostic subtypes. Missing evidence is never converted into no-boundary.

The timeline pass is implemented by
`src/auditor/boundary/final_timeline_audit.py`. It reads only the frozen
manifest and each selected `mid.mp4`, binds to loopback, saves drafts
atomically, and retains timestamped final snapshots. The separate
shared-candidate audit is created only after all method outputs are frozen.

## Comparisons and metrics

Proposal methods are compared by recall at 0.5/1.0 seconds, proposals per
minute, and recall-versus-budget AUC. Shared-candidate verifiers are compared on
the same union/deduplicated pool, with within-recording pair accuracy primary.
Native pipelines retain their own proposal, verifier, and decoder, with all
thresholds fixed on development data.

The two repetition-primary metrics are `same_action_new_instance` recall on the
exhaustive timeline and `same_instance` false-split rate on the shared candidate
pool. Boundary F1 at 1 second, new-action recall, cuts per minute,
recording-macro scores, and paired recording-bootstrap 95% intervals are always
reported beside them.
