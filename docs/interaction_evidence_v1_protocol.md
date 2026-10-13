# Interaction evidence: development pilot v1

Protocol set on 2026-09-13 before running this pilot. Source discussion:
ChatGPT conversation `6aa642a6-4dfc-83ea-af87-5adde3a169dc`,
"解读压缩学习选项", final experiment-order answer.

## Approval and amendments

Approve the sequence of evidence ablations before graph integration, with
these changes:

1. E0 (target and data readiness) and E1 (credible visual baselines) precede
   E2-E4. Do not deliberately weaken a comparator or require a positive result.
2. Same-action new instances, continuous-contact transitions, hard negatives,
   and missing evidence are strata from the beginning, not only E7/E8.
3. Release is neither necessary nor sufficient. Object identity changes and
   direction reversals are evidence, not automatic labels. Instance completion
   and action identity are separate; the same regime can begin a new instance.
4. Gradual transition shape does not establish a boundary; gradual internal
   motion does not establish its absence. Require an explicit relation label.
5. Keep candidate discrimination separate from full-timeline proposal recall.
   A candidate-selected pilot cannot estimate population FPR or missed-event
   recall. Same-instance errors here are conditional false-split rates.
6. All preprocessing and fitting are recording-grouped. No test-time tuning,
   feature sign flipping on held-out labels, or reuse of learned OOF scores
   inside new folds. Report matched pools and missingness controls.
7. Graph ablations later include no graph, matched random graphs, transition
   counts and learned graphs. Learn every graph/prototype on training data;
   test unseen tasks/orderings. A graph's usefulness is not proof of causality.

## Test isolation

The 48 recordings in `data/splits/final_test_v1.json` are forbidden here.
Verify its preregistered SHA256 before loading any pilot feature table. Reject
overlap in either target or feature rows rather than silently training on it.
Do not alter the existing final-test files, inspect annotations/predictions,
or change its release-oriented answer key retrospectively. The broader target
needs a separately versioned future evaluation; this is development only.

## E0: target audit

Use `relation_audit_annotator1_filled.csv` as the only pilot target source:
it explicitly asked the relation question. Retain `same_instance` (0),
`same_action_new_instance` (1) and `new_action` (1) independently in output.
Exclude `cannot_determine` and report it. Initial/terminal events are separate.
Neither old morphology nor prose-to-label migration supplies target labels.
Existing relation annotations are historical development judgments, not a new
blind confirmatory gold set or proof of inter-annotator reliability.

Audit all 415 migrated rows, including unknown and derived labels. Produce a
label-blind reannotation queue for rows lacking an explicit relation judgment.
The queue carries no old label, model score, or suggested evidence. Reviewers
record relation, shape, visibility, contact evidence and negative subtype
separately; none of these evidence fields determines the relation by rule.

## First runnable comparison

Reuse raw development CSVs from `/workspace/tr1/results/hal/c3/`:

- `concentration_features_full.csv`: visual change magnitude/shape descriptors.
- `hand_trajectory_features.csv`: hand kinematics and observation quality.
- `observable_audit/hand_object_pilot.csv`: provenance audit ONLY, never input.

Four arms on identical explicitly labeled, feature-covered events:
`visual_change_shape`, `hand_kinematics`, `visual_plus_hand`, `missingness_only`.
These are a feasibility baseline, not the complete E2/E3 cue experiment and not
a reproduction of P1, ABD, GEBD, or VidParse. Hand motion is NOT hand-object
relative motion. Box changes are NOT object state changes.

Use grouped five-fold CV, fixed logistic regression C=1, balanced class weights,
training-only median imputation/scaling, and fixed threshold 0.5. Every arm
receives observation quality and missingness indicators; the missingness-only
arm receives exactly those controls. An absent entire signal group is an
abstention, not a synthetic zero observation. Predict excluded labels only for
an observability diagnostic, never train on them as negatives.

Report pooled OOF AUROC/AP, recording-macro accuracy, candidate-conditional
TPR/FPR, within-recording pair accuracy, per-relation/subtype support, and paired
recording-bootstrap AUROC differences. Bootstrap intervals are conditional on
fixed OOF predictions, not uncertainty from refitting the entire pipeline.
Record folds, source hashes, configuration, dependencies and per-event output.
No hyperparameter sweep and no model promotion based on this reused pilot.
Below 15 events or 5 recordings per class, mark evidence insufficient even if
an exploratory score is calculable. Unmeasured strata must be shown as missing.

## Readiness gates

The 36-row historical hand-object table has exactly one release and one target
switch in every event, identical object visibility 35/41 and hand visibility
1.0. This matches the scripted synthetic pilot signature; it lacks usable
real-backend provenance. Quarantine it pending validation. Do not deserialize
its pickle objects with `weights_only=False` or treat it as measured evidence.

E2 must start with a real, provenance-recorded detector/observation extractor
and a blind QA subset. E3 requires all three actual cues on a shared development
pool, plus single-cue, missingness-only and leave-one-cue-out controls. E4 needs
adequate independent recordings in each stratum. Progress to graph experiments
only after these gates, or document a negative local-evidence result.

## Run

`python -m src.auditor.boundary.interaction_evidence_pilot --out_dir PATH`
performs E0 without feature files or ML dependencies.

Add `--visual_csv PATH --hand_csv PATH --historical_hoi_csv PATH` to run the
pilot. NumPy and scikit-learn are required for this optional evaluation. Output
directories must not exist: a rerun gets a new directory and preserves history.
