# Handness V2 Crop vs Crop+Context A/B

## Question

Does concatenating a full-frame Qwen vision embedding with the candidate-crop
embedding fix stable upper-arm/elbow and thigh/knee false positives without
reducing real-hand safety?

The 21-track confirmation set was converted to development data after its labels
were revealed. Two recording-held-out development folds were used:

- Fold A: train includes the thigh/knee recording; validation holds out the
  upper-arm/elbow recording.
- Fold B: train includes the upper-arm/elbow recording; validation holds out the
  thigh/knee recording.

Every comparison used the same recording split, human labels, three seeds,
training recipe, 99.5% validation-recall threshold rule, and fail-open track
aggregation. The only changed variable was crop-only versus concatenated
crop-plus-context features.

## Aggregate result

| Fold | Input | Validation specificity | Test recall | Test specificity |
|---|---|---:|---:|---:|
| A | Crop | 31.18% | 96.73% | 19.15% |
| A | Crop + context | 33.33% | 97.06% | 21.28% |
| B | Crop | 27.96% | 98.26% | 17.02% |
| B | Crop + context | 30.11% | 97.17% | 22.34% |

Context produced only a roughly two-point validation-specificity increase. Test
recall was not consistently improved and dropped by 1.09 points in fold B.

## Held-out body-part tracks

| Held-out track | Crop score range | Crop+context score range | Result |
|---|---:|---:|---|
| Upper arm / elbow | 0.785-0.902 | 0.909-0.971 | Both retain as hand |
| Thigh / knee | 0.561-0.931 | 0.753-0.942 | Both retain as hand |

All real-hand tracks in the two held-out recordings remained retained. However,
context increased every body-part track's semantic confidence and did not reject
either false positive.

## Decision

Reject the crop-plus-context head. Do not deploy it and do not tune its threshold
on these recordings. Full-frame scene context by itself does not supply the
missing hand-anatomy boundary and introduces additional carrier/source cues.

The next diagnostic is the original Qwen semantic teacher on the same views. If
the teacher rejects the body parts, mine and review more arm/elbow/thigh/knee
hard negatives and distill that boundary. If the teacher accepts them, revise
the semantic question or add explicit anatomical/keypoint evidence before any
further distillation.
