# Handness Qwen-feature head V1

Date: 2026-09-26

## Decision

Retain direct Qwen-feature distillation. Reject the ImageNet/ResNet student.
Do not connect either model to the production gate until the pending human
negative review and a new recording-held-out confirmation run are complete.

## Target

The head answers only whether a candidate crop contains visible human-hand
evidence. Ownership and handedness are separate decisions. A crop dominated
by an object is still `hand` when fingers, palm, back of hand, or glove are
visible. `unsure` fails open.

## Data corrections

- Legacy `owner` and `other` safely map to `hand`.
- Legacy `nothand` does not become human truth without V2 review.
- Exact image hashes, not reused stems, define duplicates.
- Splits are grouped by recording.
- The old three-class attempt mixed rendered 1600x900 panoramas with raw
  1920x1520 camera halves. The classifier learned that carrier. V1 therefore
  reads only the canonical 1.6x candidate crop at a fixed Qwen pixel budget.
- No inverse-frequency weighting is used. It previously inflated an enriched
  non-hand prior and damaged precision.

The current manifest has 12,175 unique view pairs from 229 recordings. It has
5,932 reviewed human positives and no reviewed human negatives. A blind packet
of 457 legacy negative/unsure cases is therefore the blocking annotation task.

## Architecture

1. The 27B Qwen handness score remains the soft teacher target.
2. `Qwen3-VL-Embedding-2B` supplies only its frozen vision tower. The language
   model is never constructed and no text is generated.
3. Qwen `pooler_output` tokens for the crop are mean- and max-pooled,
   concatenated, and L2-normalized into a 4,096-dimensional cache vector.
4. The trainable head is `LayerNorm -> Linear(4096,256) -> GELU -> Dropout ->
   Linear(256,1)` with soft binary cross entropy.
5. Three independently initialized heads are averaged. Feature extraction is
   one-time; threshold changes, fusion changes, and re-rendering reuse cache.

Primary code:

- `scripts/extract_handness_qwen_features.py`
- `src/semhand/qwen_feature_head.py`

The older `src/semhand/handness_head.py` is a rejected ImageNet control only.

## Development results

All values below are per-view. The test split is recording-held-out, but it has
already been inspected during development and is not a final confirmation set.

| Model | Test AUC | Test Brier | Hard agreement | Human-hand recall at 0.5 |
|---|---:|---:|---:|---:|
| ImageNet crop control, seed 0 | 0.7572 | 0.0966 | 0.8395 | 0.9809 |
| Qwen features, seed 0 | 0.9586 | 0.0176 | 0.9759 | 0.9936 |
| Qwen features, 3-seed mean | 0.9565 | 0.0173 | 0.9748 | 0.9809 |

The ensemble threshold is provisionally 0.20. It is the highest tested value
that removed no positive item on the validation split. Applied unchanged to
the development test split, it:

- kept 100% of teacher-positive views;
- kept 100% of the 157 reviewed human-hand views;
- rejected 20/70 teacher-negative views (28.6%);
- rejected no teacher-positive view.

This is intentionally a high-precision, low-recall veto. At source level the
remaining weakness is the small C1 subset (`n=16`, AUC 0.688); the larger
legacy-negative subset has AUC 0.880. Aggregate performance must not hide the
C1 limitation.

## Speed

A cold one-GPU benchmark extracted 1,024 crop features in 18.077 seconds,
including model load: 56.6 crops/second. Four GPUs cached all 12,175 crops in
under one minute. Once cached, MLP training and rescoring do not invoke Qwen.

Online integration should batch a few representative views per canonical
track, write their Qwen feature vectors to the track cache once, and replay the
head from cache. It must not run the language model per frame.

## Reviewed V2 result

The blind review completed on 2026-09-27: 457/457 rows, comprising 58 `hand`,
398 `nothand`, and one `unsure`. The `unsure` row is excluded from binary
fitting and scoring and must fail open in deployment.

The original split was invalid for this confirmation task because its 67
validation negatives all came from one recording and 75/76 test negatives came
from one recording. It is retained only as a diagnostic result. A deterministic
replacement (`handness-v2-reviewed-v1`) keeps recordings disjoint and balances
human truth across the splits:

| Split | Human hand | Human non-hand | Non-hand recordings |
|---|---:|---:|---:|
| Train | 4,193 | 220 | 28 |
| Validation | 898 | 86 | 6 |
| Test | 899 | 92 | 5 |

The same architecture and three seeds were retrained without rerunning the Qwen
vision tower. Validation selected threshold 0.316826, the highest observed
threshold retaining at least 99.5% of human hands.

| Split | Human AUC | Recall | Specificity | FN | TN |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9623 | 99.55% | 27.91% | 4/898 | 24/86 |
| Test, threshold frozen | 0.7299 | 99.11% | 9.78% | 8/899 | 9/92 |

This misses the deployment requirement. Six of eight test false negatives are
from `R0826_102020`; the other two are from `R0821_172625`. Nearby frames often
contain high-scoring hand crops, but the legacy review samples lack tracker IDs,
so this does not yet establish that same-track multi-view aggregation rescues
the failures. Do not tune this per-crop model further on the read test split and
do not connect it to the production gate.

The next test is track-level: align these views to cached canonical tracks,
aggregate several representative views with a fail-open rule, and measure
owner-track false removal and non-hand-track rejection on a new recording-held-
out confirmation set.

## Canonical-track aggregation

The legacy reviewed views were aligned by recording, frame, and normalized box
IoU to raw tracker IDs, then mapped through the frozen track-fusion table. Of
1,975 reviewed views, 224 aligned unambiguously. They covered 30 canonical
tracks; 16 had at least three labeled views. Fail-open aggregation rescued all
15 resolvable true-hand tracks, but the only resolvable non-hand track was also
kept. This is evidence for safety, not useful cleaning: the historical packet
was sparse at track level and is not a final confirmation set.

The frozen deployment candidate uses this exact policy:

1. Query only tracks currently proposed as owner hands.
2. Sample up to five temporally distributed views per canonical track.
3. Fewer than three scored views returns `unsure` and keeps the track.
4. Any view at or above 0.316826 keeps the track.
5. A track is rejected as `nothand` only when at least three scored views are
   all below 0.316826.
6. Human `mixed` tracks also stay open and are reported separately as
   tracker/fusion failures.

Primary code:

- `scripts/evaluate_handness_tracks.py`
- `scripts/build_handness_track_review_packet.py`
- `scripts/build_handness_track_feature_manifest.py`
- `scripts/evaluate_handness_track_packet.py`

## New-recording confirmation

The final packet uses seven recordings from 2026-09-10 through 2026-09-14,
which are absent from the 229-recording development manifest. Detection and
tracking ran once; 29 raw tracks fused to 28 canonical tracks. Restricting the
semantic check to owner candidates produced 21 review tracks and 101 views.

The Qwen vision features, three head checkpoints, threshold, minimum-view rule,
and fusion rule were frozen before labels. The unlabeled mechanical run kept 20
tracks by an above-threshold view and failed open on one one-view track. Those
counts are not accuracy measurements.

Blind review completed with 19 `hand` tracks and two `nothand` tracks; there
were no `mixed` or `unsure` labels. Revealing the frozen predictions gave:

| Result | Count |
|---|---:|
| Real-hand tracks retained | 19/19 |
| Real-hand tracks deleted | 0/19 |
| Non-hand tracks rejected | 0/2 |
| Non-hand tracks retained | 2/2 |
| Tracks failed open for too few views | 1 |

All 91 reviewed hand views exceeded the frozen threshold. The one hand track
with only one eligible view was retained by the minimum-view rule, not by a
semantic rejection decision. Thus safety recall was 100% on this set, but the
95% Wilson lower bound is only 83.2% because there are 19 positive tracks.

Both non-hand failures were stable canonical tracks on skin-colored body parts:
one covered thigh/knee and the other upper arm/elbow. Their ten view scores were
all high (`0.786-0.931`), so multi-view aggregation cannot repair them. This is
not track mixing. The feature extractor currently reads only the candidate crop;
it does not encode the full-frame context shown to the annotator. The observed
failure is therefore a semantic hard-negative gap: the head recognizes exposed
skin/limb appearance without reliably requiring hand anatomy.

Do not move the threshold after seeing this lockbox. Track maximum scores happen
to separate the 19 hand tracks (`>=0.988`) from the two body-part tracks
(`<=0.931`), but selecting a threshold from these 21 reviewed tracks would turn
the confirmation set into development data. The next valid experiment is to add
upper-arm, elbow, thigh, and knee crops to training/validation hard negatives,
freeze either a context-aware head or a revised threshold on development data,
then evaluate once on another untouched recording set with substantially more
non-hand tracks.

### Context A/B follow-up

The revealed tracks were converted to development data and used in two
recording-held-out folds: train on thigh/knee and validate on upper arm/elbow,
then reverse them. Crop-only and crop-plus-full-context heads shared identical
splits, labels, training settings, three-seed ensembles, and validation threshold
selection.

Context did not solve the failure. It raised the held-out upper-arm/elbow scores
from `0.785-0.902` to `0.909-0.971`, and the held-out thigh/knee scores from
`0.561-0.931` to `0.753-0.942`. Both tracks remained `hand` under the frozen
fail-open rule. Validation specificity improved by only about two percentage
points, while test real-hand recall was not consistently preserved.

The crop-plus-context head is therefore rejected. The next diagnostic is the
original Qwen semantic teacher on the same 101 views. Further distillation is
justified only if the teacher itself distinguishes the body-part negatives.

### Direct-teacher diagnostic

The original Qwen semantic teacher was then run on the same 101 full-frame plus
crop pairs. The JSON prefix and two-token true/false probability were
preserved, and the track policy remained frozen at a 0.60 per-view threshold,
three-view minimum, any-view rescue, and fail-open on insufficient views.

It retained all 19 real-hand tracks and rejected both non-hand tracks. The
thigh/knee views scored 0.014-0.047; the upper-arm/elbow views scored
0.0006-0.0059. The one one-view hand track failed open and remained kept.
Thus the direct teacher has the required anatomy distinction on this revealed
set; the failure is in student distillation rather than the semantic prompt.

Targeted mining replayed the frozen three-head student over 12,175 development
views and selected unlabeled canonical tracks where the old teacher scored at
most 0.20 but the student scored at least 0.60. There were 103 disagreement
views across nine tracks. Thirty temporally distributed views were re-scored
with the current direct teacher: five tracks were stable teacher negatives and
four had only one or two eligible views, so they remain fail-open. All nine were
packaged for blind human review before any retraining.

### Hard-negative review and grouped CV

Blind review labeled six of the nine tracks as non-hand and three as hand,
covering 15 selected views in each class. Because all nine were direct-teacher
negatives, direct Qwen is useful for candidate mining but unsafe as an automatic
labeler in this enriched disagreement region.

Only the 30 displayed views received human labels. A four-fold experiment held
out one complete negative recording at a time and excluded it from checkpoint
and threshold selection. Hard-example weights 4 and 16 were tested with three
seeds. The only fold with at least three reviewed views per held-out track
contained two eligible non-hand tracks; neither weight rejected either track.
At weight 16, two of nine negative views fell below threshold, but any-view
rescue retained both tracks. Legacy test-set hand recall ranged from 97.0% to
98.2%, below the 99.5% safety target.

The current crop-feature student, crop-plus-context student, and direct Qwen
teacher are therefore not deployment-ready rejection gates. A second
active-learning packet contains six new tracks and 29 views, with at least
three disagreement views per track.

### Second hard-negative round and student decision

The second blind packet labeled all six tracks: two `nothand` tracks across
nine selected views and four `hand` tracks across 20 selected views. Combined
with round one, the reviewed disagreement set contains 15 tracks: eight
`nothand` and seven `hand`. Four non-hand tracks have at least three reviewed
views and are evaluable under the frozen fail-open track rule; they come from
only two recordings.

Two recording-held-out folds trained on one negative recording and held out
the other. Thresholds were selected without the held-out recording at at least
99.5% validation hand recall. The Qwen pooled-feature MLP rejected 0/1 eligible
held-out non-hand tracks in the R0824 fold and 0/3 in the R0828 fold. Frozen
legacy-test hand recall was 96.36% and 98.18%, respectively. Increasing hard-
example weight therefore did not produce a safe rejection gate.

Two Qwen-free pixel students were also tested without ImageNet weights. A
randomly initialized ResNet-18 trained from Qwen probabilities plus human
overrides rejected 0/1 and 0/3 held-out non-hand tracks; legacy-test hand recall
was 100% and 99.39%. A pilot that additionally regressed the cached 4,096-D
Qwen global feature rejected 0/1 and retained only 96.97% of legacy-test hands.

All tested distilled heads failed the primary endpoint: zero of four eligible
body-part non-hand tracks were rejected in held-out recordings. None may be
connected to rendering. The next distillation experiment needs substantially
more recording-diverse body-part negatives and spatial teacher supervision
(patch tokens, masks, or boxes), rather than more weight on the same global
embedding. The fast inference target remains Qwen-free; ImageNet weights remain
excluded.
