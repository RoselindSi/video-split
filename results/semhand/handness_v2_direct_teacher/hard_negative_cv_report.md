# Handness V2 Hard-Negative Review and Cross-Validation

## First active-learning review

The first packet contained nine canonical tracks selected where the frozen
student scored at least 0.60 and the direct Qwen teacher scored below 0.20.
Human review found:

| Human label | Tracks | Reviewed views |
|---|---:|---:|
| Hand | 3 | 15 |
| Non-hand | 6 | 15 |

All nine tracks were direct-teacher negatives on the selected views. Therefore
the teacher's negative prediction was correct for six tracks and wrong for
three. This packet is deliberately enriched for disagreement and is not an
unbiased error-rate estimate, but it establishes that direct Qwen cannot
auto-label these candidates safely.

Only the 30 views actually shown to the reviewer received human labels. Labels
were not propagated to another 853 views sharing the same canonical identities.

## Recording-held-out experiment

The six reviewed non-hand tracks came from four recordings. Four folds held out
one complete negative recording at a time. Checkpoint selection and threshold
selection excluded that recording. The threshold was calibrated to retain at
least 99.5% of human hands on the remaining validation recordings.

Two hard-example weights were tested: 4 and 16. Each condition used three seeds
and the original crop-feature MLP. The detector, tracker, canonical fusion,
cached Qwen embedding features, and fail-open rule were unchanged.

Only one held-out recording supplied tracks with the required minimum of three
reviewed views. It contained two eligible non-hand tracks:

| Weight | Eligible tracks rejected | Held-out negative view specificity |
|---|---:|---:|
| 4 | 0/2 | 0/9 |
| 16 | 0/2 | 2/9 |

At weight 16, the two track maxima were 0.926 and 0.777 against a frozen
threshold of 0.650, so any-view rescue retained both. The other four negative
tracks had only one or two reviewed views and correctly failed open.

Across folds and weights, the legacy test-set hand recall was 97.0%-98.2%,
below the 99.5% safety target. Increasing hard-example weight lowered some
negative scores but did not produce track-level rejection and did not preserve
the required hand recall.

## Decision

Do not deploy the current crop-feature head, the crop-plus-context head, or the
direct Qwen teacher as an automatic rejection gate. The first two lack
demonstrated cross-recording body-part discrimination; direct Qwen produced
three false-negative hand tracks in the enriched disagreement set.

The next active-learning packet contains six previously unreviewed tracks and
29 views. Every track has at least three disagreement views, so each label can
contribute directly to the multi-view policy rather than being forced to fail
open.
