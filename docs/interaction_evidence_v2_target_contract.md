# Interaction evidence target contract v2: development draft

Prepared 2026-09-14 after the v1 historical-label pilot. This is a draft
definition audit, not a change to final_test_v1, not new gold, and not evidence
that the proposed model works. No old label is automatically accepted as v2.

## Unit of analysis

Judge a local, outcome-directed interaction episode at the candidate time.
Distinguish an episode boundary from motion phases inside an episode. Do not
change granularity from a whole task to a finger movement across examples.
No predefined vocabulary of action classes is supplied: judge the relation
between the episodes on the two sides.

The proposed question is whether the visible context supports a new goal or
outcome-directed episode, including a renewed instance of the same action,
rather than only a change in appearance, speed, direction or contact.
An inferred goal must be supported by observable evidence; do not invent intent.

## Relation field

- CONTINUE: the same episode continues. Repeated cycles, a pause, a regrasp,
  changing direction, or a brief release alone do not establish a new instance.
- SAME_ACTION_NEW_INSTANCE: a completed or terminated episode is followed by
  another instance of the same action. Require evidence of an episode renewal,
  such as completing work on one item and starting the equivalent work on the
  next, even if the hands remain in contact. A repeated motion cycle alone is
  insufficient. Record the evidence supporting renewal, not a required cue.
- NEW_ACTION: a different immediate interaction goal or outcome begins, such
  as opening a container followed by closing it. The hands need not release.
- UNOBSERVABLE: the available context cannot establish the relation. Use the
  visibility/context fields to distinguish occlusion from inadequate temporal
  context or semantic ambiguity. It is not a negative training example.

For a one-sided recording start or terminal end, set boundary_scope to initial
or terminal, leave relation blank, and explain. Internal candidates use
boundary_scope=internal. One-sided events do not enter binary boundary scoring.

## Evidence fields are independent

After assigning the relation, record visibility (visible/partial/not_visible),
context_sufficient (yes/no/uncertain), transition_shape
(sharp/gradual/overlap/no_transition/uncertain), continuous_contact
(yes/no/uncertain), and release_observed (yes/no/uncertain).
These fields never overwrite the relation by a rule. A contact detector's
missing output does not establish a release or human unobservability.

Opening then closing a cap can be NEW_ACTION without a release. Regrasping
the same cap while still unscrewing it can be CONTINUE with a release.
Tightening several fasteners can contain SAME_ACTION_NEW_INSTANCE boundaries
without letting go of the tool, when each fastener's operation visibly ends
and the next begins. Stirring cycles are not automatically separate instances.
Examples establish episode granularity, not an exhaustive action taxonomy.

## Development review packet

The first packet revisits the 54 historically direct relation judgments to
measure definition drift. It is not a random sample of natural video and is
not an independent test. Selection does not use the new pilot's predictions.
Keep earlier labels and the private_key.csv hidden during the review.
Public review IDs hide candidate names that encode old answers such as
gt_boundary or false_near_edge. Provide clips under these aliases only.

The 361 reserve entries are an inventory, not a request to reannotate them all.
After checking the definition, choose further development examples using a
frozen sampling policy, including within-recording boundary/non-boundary pairs,
continuous-contact transitions and releases without boundaries. Do not select
only model errors or discard inconvenient examples after scoring.

Current packet generation does not export clips or modify the running final-test
annotation interface. Neither blank CSVs nor prior relation judgments count as
completed v2 annotations. Validate a separate versioned target file before
training or scoring with this contract. A graph is deferred: its dwell and
self-transition decisions cannot repair a target-definition mismatch.
