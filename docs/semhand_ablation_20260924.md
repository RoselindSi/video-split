# Semhand sequential ablation, 2026-09-24

## Contract

All three changes use the same ten-video detections, raw tracker ids, Qwen
scores, thresholds, and frame windows. Each step changes one decision layer
and is compared with the immediately preceding output. Raw ids are never
rewritten.

Server artifacts live under:

`/workspace/semantic_ablation_20260924`

## Frozen baseline

- 10 gate files, including one explicitly empty recording.
- 51 reviewed raw tracks and 236 scored context views.
- Verdicts: 45 owner, 3 other, 3 unsure.
- Reasons: 44 context consensus, 3 context reject, 3 too few, 1 base-track
  consensus.

## Step 1: too-few context

Only the insufficient-context branch changed. One or two Qwen views still
cannot admit a track by themselves. They can be rescued when the whole raw
track has `base_p_median >= 0.95` and at least 80% of base observations are
`>= 0.90`.

Result:

- `R0820_151218/tid8`: unsure -> owner, 12 owner frames.
- `R0825_155630/tid15`: unsure -> owner, 37 owner frames.
- `R0825_155630/tid6` remained unsure because its strong-base fraction was
  0.667.
- Net change: 2 tracks, 49 owner frames.

Validation: unit tests cover stable and unstable rescue cases. All changed
context/crop views were inspected; both rescued tracks are wearer hands.

## Step 2: canonical track-id fusion

The offline fusion accepts a link only when a successor starts within three
frames, endpoint or short motion prediction overlaps, normalized center
distance is bounded, strong side evidence does not conflict, and neither end
has a near-tied competitor. Every accepted/rejected candidate is logged.

Result:

- 260 raw ids -> 247 canonical ids; 13 accepted links.
- Hand/not-hand decision changes: 0 frames.
- Owner decision changes: 0 frames.
- Handedness stabilization: 65 frames.
- Canonical semantic propagation added an explicit owner verdict for
  `R0826_102020/tid19`, linked to `tid23`. Its 3 frames were already rendered
  as primary owner frames, so this changed gate completeness but not pixels.

Validation: synthetic tests cover clear handoff, temporal overlap, strong side
conflict, and ambiguous successors. The 13 endpoint pairs were inspected; no
accepted pair showed two different simultaneous hands. The two links in
`R0826_102020` also reproduce the two handoffs reported independently by the
online fragmentation diagnostic.

## Step 3: semantic evidence fusion

The final owner decision now records separate `context_state` and `base_state`.
Context consensus still admits directly. Stable whole-track base evidence can
rescue a Qwen context rejection; without stable base evidence, context reject
still returns other.

Result:

- `R0820_151218/tid17`: other -> owner, 39 owner frames.
- `R0820_151218/tid27`: other -> owner, 17 owner frames.
- `R0820_151218/tid16` remained other because its base track was unstable.
- Net change: 2 tracks, 56 owner frames; all other gates unchanged.

Validation: both changed tracks were inspected over all eight context views.
They enter from the camera-wearer's lower/right edge and are wearer hands;
Qwen's descriptions recognized that motion but its Boolean identity answer was
wrong. Unit tests lock this conflict behavior.

## Scope of the evidence

This is a controlled behavior ablation and visual audit, not a new accuracy
estimate. The ten-video set has no independent track-level negative gold for
all 51 candidates. A held-out, double-annotated identity set is still required
before reporting sensitivity/specificity for the fused policy.

## Final replay checks

- A fresh `R0820_151218` collection reproduced all 1,196 frozen raw-track rows
  exactly before any new decisions were applied.
- The final `R0820_151218` replay changed exactly 68 owner states: `tid8` 12,
  `tid17` 39, and `tid27` 17. `tid16` remained other. Hand/not-hand and side
  fields did not change.
- Old/new frames and all crops for those three tracks were inspected. The
  rescued boxes are camera-wearer hands; the nearby colleagues' hands remain
  other.
- `R0826_102020` replay had no rendered owner, hand/not-hand, or side changes;
  this is the expected no-regression result for its two canonical links.
- Both validation videos obey `max_owner=2` on every frame.

Validation videos:

- `/workspace/semantic_ablation_20260924/final_replay/R0820_151218.mp4`
- `/workspace/semantic_ablation_20260924/final_replay/R0826_102020.mp4`
