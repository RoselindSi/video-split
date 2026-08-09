# The repeated-instance gap: a rule the taxonomy never wrote down

## What happened

48 never-re-checked batch3 events went to a blind annotator with the
`sharp | same | cannot` vocabulary the 36-event audit used. The sheet came back
with a fourth value the annotator added, `interval`, used 30 times, and
`sharp` used **once in 48** — against a sample that is 24 POINT and 24
NO_TRANSITION by construction.

Reading the stated reasons, the sheet is dominated by one structural case:

| stated reason | n | calls |
|---|---:|---|
| the gap between two instances of the SAME repeated action | 21 | interval 20, same 1 |
| an action ends into idle with nothing following | 6 | interval 3, same 3 |
| the relevant hands are off-frame | 4 | cannot 4 |
| everything else | 17 | interval 7, same 9, sharp 1 |

**27 of 48 events, 56%, turn on a question the taxonomy does not answer.**
Cup moved left, then right, then left again with the hands briefly idle
between. Strainer rinsed and replaced, hands leave the sink, strainer taken
out and rinsed again. Slippers into the drawer, then out, then back.

## This is not label noise. It is label undefinedness

The seven subtypes describe what a change looks like. None of them says
whether the pause between two repetitions of one action is a boundary, so an
annotator meeting one has to invent a rule, and a second annotator will invent
a different one. Measuring their agreement would measure the absence of the
rule, not their skill.

## The canonical labels already show the damage

The GT segmentation **does** split repeated instances: it emits two consecutive
segments carrying the same label. Of the 44 events in `pair_labels_v1.csv` that
record both neighbouring segment labels, **8 have `prev == next`**. Their
stored subtypes:

    sharp_visible_transition       3
    same_action_internal_motion    2
    annotation_convention          1
    gradual_phase_transition       1
    ambiguous                      1

One structural situation, five different answers, spread almost evenly. Eight
events is small, but the double-audit sheet reached the same gap independently
from a different population, and neither route was looking for it.

This also explains things already measured. batch3 is where `raw_change_peak`
and `gt_boundary` candidates land, which is where repeated-cycle gaps land, and
batch3 is where every scorer is weakest (0.73–0.76 against dev's 0.84–0.90).
The 37 gradual events re-audited to 25 `same_action` on the same underlying
question — is internal repetition a boundary. And humans agree with the stored
labels at only 0.73–0.75 while agreeing with each other at 0.86.

## The decision, which is a product decision and not an empirical one

**Q1. Is the gap between two instances of one repeated action a boundary?**

- **A — yes, each instance is its own segment.** Consistent with what the GT
  already does. Produces many more boundaries; the idle becomes POINT. Every
  repeated-cycle recording gains segments.
- **B — no, repetitions belong to one action.** The idle is internal motion,
  NO_TRANSITION. Consistent with what this annotator did (20 of 21) and with
  the ontology re-audit of the 37. Contradicts the current GT, so the GT's
  same-label splits become `annotation_convention`.
- **C — it depends on whether the repetition count is part of the task.**
  Defensible and needs a second rule to decide when, which is where an
  undefined case usually comes back.

**Q2. Is "an action ends and nothing follows" a boundary?** Six events, three
called `interval` and three `same` by the same annotator on the same day, which
is what an undefined case looks like from the inside.

**Q3. Does `INTERVAL_TRANSITION` cover the repeated-instance gap?** The
annotator reached for `interval` 30 times without being offered it. If the
answer to Q1 is B, most of those are NO_TRANSITION and `interval` was standing
in for "there is a gap but it is not a new action". If the answer is A, they
are POINT with a short offset. The word the annotator needed does not exist in
either taxonomy.

## What not to do next

**Do not send the second annotator the same sheet.** Two people answering an
undefined question produce an agreement number that describes the rule's
absence. The human-human figure this experiment was built to produce would be
uninterpretable, and it is the number everything downstream depends on.

Write the rule, restate the vocabulary so it has a slot for the repeated-cycle
gap, and then run both annotators on it. The 48 events are a good sample for
that and can be reused unchanged.
