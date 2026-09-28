# Handness V2 Direct-Teacher Diagnostic

## Question

The frozen crop-feature student retained two reviewed non-hand canonical tracks:
a thigh/knee track and an upper-arm/elbow track. This diagnostic asks whether
the original Qwen semantic teacher can distinguish those body parts from hands
under the same multi-view, fail-open policy.

## Frozen protocol

- Input: the same 101 full-frame plus crop view pairs from 21 reviewed tracks.
- Model: the already-running qwen3.8-27b vLLM service.
- Prompt: the production hand-versus-non-hand question and JSON prefix.
- Per-view decision threshold: p(hand) = 0.60.
- Track policy: fewer than three views fails open; any accepted view retains
  the track; rejection requires at least three views and all below threshold.
- Labels were not used to choose the threshold or aggregation rule.

## Result

| Metric | Result |
|---|---:|
| Real-hand tracks retained | 19/19 |
| Real-hand tracks deleted | 0/19 |
| Non-hand tracks rejected | 2/2 |
| Tracks failed open for too few views | 1 |

The thigh/knee track had p(hand)=0.0141-0.0474; the upper-arm/elbow track had
p(hand)=0.0006-0.0059. The one one-view hand track failed open and was kept.

This establishes that the teacher has the needed semantic distinction on this
small revealed set. The failure lies in the distilled feature head, not in the
teacher prompt or canonical-track aggregation.

## Hard-negative mining

The three frozen student heads were replayed over 12,175 cached development
views. Mining required:

- no existing human label;
- canonical track identity available;
- old teacher p(hand) <= 0.20;
- student ensemble p(hand) >= 0.60.

This produced 103 disagreement views in nine canonical tracks. Sampling at
most five temporal views per track yielded 30 pairs. The current direct teacher
re-scored them:

- five tracks had at least three views and all were below 0.60;
- four tracks had only one or two eligible views and therefore remain
  fail-open rather than being auto-labeled.

A blind nine-track review packet was generated. Its page contains only the
images and frame numbers; student scores, teacher scores, and mining reasons
are absent.

## Decision

Do not deploy the crop-plus-context head and do not retune a threshold on the
revealed 21-track confirmation set. After the nine mined tracks are reviewed,
add only human-confirmed non-hand tracks to development, keep recordings
disjoint, retrain the crop-feature head, and verify on a new untouched
recording-level lockbox with more non-hand tracks.
