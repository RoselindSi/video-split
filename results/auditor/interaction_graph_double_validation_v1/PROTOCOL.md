# Whole-timeline graph double annotation protocol v1

Frozen before Reviewer 2 results are available.

## Validation set

- 30 videos from `interaction_graph_mosaic_complement_v1`.
- Reviewer 1 completed the set before this protocol was written.
- Reviewer 2 receives only the videos, a blank whole-timeline annotator, and the rule below.
- Reviewer 2 must not inspect Reviewer 1 annotations or discuss case-level decisions before export.
- Frozen Reviewer 2 packet SHA-256: `b376c29b47885cf4f721ed907532aec4e92aba8e00ac14e345c2b444fcd9c708`.

## Annotation rule

1. Continuing the same interaction instance, including repeated back-and-forth motion: do not cut.
2. The same action type starts again after the prior instance ended: cut at the true restart and reuse the same event node.
3. The goal changes: cut at the true start of the new action and use a different event node.
4. Waiting or blank time is not a separate event. Put the boundary at the observable action start.
5. Mark `repeats_within_segment` only for repetition inside one continuous instance.

This induces the three research relations without asking for taxonomy labels:

- `CONTINUE`: no boundary.
- `SAME_ACTION_NEW_INSTANCE`: boundary with the same event node on both sides.
- `NEW_ACTION`: boundary with different event nodes on the two sides.

## Frozen analysis

Raw event letters and free-text descriptions are not compared. Event letters are local to each reviewer.

Primary metrics:

- Boundary micro precision, recall, and F1 at 0.25, 0.5, and 1.0 seconds.
- Relation agreement at one-to-one matched boundaries within 1.0 second.
- Temporal event-type partition agreement using adjusted Rand index on a 0.5-second grid.
- Agreement and Cohen's kappa for `repeats_within_segment` on the same grid.
- Exact normalized event-path count after replacing local event letters by order of first appearance.

The validation gate is fixed before Reviewer 2 results:

- all 30 manifest videos are present and there are no extras;
- boundary F1 at 1.0 second is at least 0.80;
- matched-boundary relation agreement is at least 0.80;
- median temporal partition ARI is at least 0.80;
- internal-repeat time agreement is at least 0.80.

Passing the gate means the contract is reproducible enough for graph construction. It does not make either reviewer ground truth. Score first, preserve the raw disagreement list, and only then adjudicate disagreements while blinded to model output.

## Scoring command

```bash
python3 -m src.auditor.boundary.interaction_graph_timeline_agreement \
  --manifest results/auditor/interaction_graph_mosaic_complement_v1/manifest.json \
  --reviewer-1 results/auditor/interaction_graph_mosaic_complement_v1/annotations/reviewer_1 \
  --reviewer-2 /path/to/reviewer_2_graph_double_validation_v1_final.json \
  --out-dir results/auditor/interaction_graph_double_validation_v1/agreement
```

Do not change thresholds after opening Reviewer 2's results.
