# V2 development expansion to 150 events

Requested 2026-09-14. This changes the size of the development review packet,
not its target definition or any final-test protocol. The original 54-event
packet and historical-model pilot remain immutable.

## Selection policy

- Retain all 54 historical direct-relation events for definition rechecking.
- Select 96 additional events from the same frozen 415-event development pool.
- Verify the final-test manifest hash and reject overlap in all source tables.
- Use only event membership, recording ID, timestamp, and the fixed seed.
  Do not use old label values, new model scores, errors, or feature availability.
- First add a distinct candidate to legacy recordings with one selected event.
  Then add pairs from new recordings in deterministic hash order. Fill any
  remaining slots in order of lowest selected count per recording, with the
  same deterministic tie-break. Maximize neither a model score nor class balance.
- Require at least two seconds between selected candidates within a recording,
  and no more than three events per recording, including retained events.
  Fail explicitly if these constraints prevent reaching the requested size.
- Keep public review aliases stable relative to the original packet. Hide old
  labels, model scores, event names and candidate sources in the review sheet.

Pairs are different candidate times, not assumed opposite labels. The two-second
rule reduces duplicate proposals but does not prove distinct action instances.
This enriches within-recording comparisons; it is not uniform sampling of the
natural event distribution. All later evaluation remains recording-grouped.

## Readiness versus sample count

150 is the number of selected events requiring v2 review, not 150 accepted new
labels. Preserve explicit blank relation fields. Old release-conditioned
labels are not copied into them. The reserve now contains 265 events.

After selection, audit visual and hand-motion cache coverage without changing
the selected set. Missing rows require extraction, not a detector-error label.
Real contact evidence remains a separate E2 requirement. Do not retrain or
report a 150-event v2 AUROC until labels and required inputs are available.
Candidate-level sample size does not imply full-timeline recall coverage.

## Reproduce

Run `python -m src.auditor.boundary.interaction_evidence_review` with the original
pilot report, `--n_events 150`, this file as `--sampling_protocol`, and a new
output directory. Optionally pass both `--visual_csv` and `--hand_csv` for the
post-selection cache audit. Source hashes and selection arguments are saved.
