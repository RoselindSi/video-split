# Pair taxonomy — subtype definitions for the second annotator

These are the definitions the **existing** labels were assigned under, taken
from `src/boundary/pair_taxonomy.py`. They are reproduced, not rewritten. The
measurement is whether two people applying *this* rulebook agree; handing you a
clearer rulebook would answer a different question, so if a definition seems
underspecified, **record that in `notes` rather than resolving it yourself** —
an underspecified definition is exactly the finding this exercise is looking
for.

You will see a ~6-second clip centred on a candidate moment, a contact sheet,
and the segment labels around it. You will not see the existing label, the
first reviewer's opinion, or any model output.

---

## What you are judging

Each clip has a candidate moment in the middle. The question is what kind of
relationship holds between the moment **before** it and the moment **after**
it. Not "is this a boundary" — *what kind of thing is this*.

Pick exactly one of seven.

---

### `sharp_visible_transition`
A clear, fast change of interaction or state. The previous action ends and a
different one begins, and there is a moment you could point to.

Recorded example: *previous action ends, hand reaches for and picks up the wrap
roll; interaction target changes.*

### `same_action_internal_motion`
One ongoing action. There is motion — sometimes a lot — but the intent and the
thing being interacted with do not change.

Recorded example: *continuous pressing/smoothing of plastic wrap along the bowl
rim; intent and interaction target unchanged.*

### `gradual_phase_transition`
A real change, but with **no instantaneous switch**. One phase becomes another
over time and any boundary you drew would be arbitrary within a second or two.

Recorded examples: *pulling wrap gradually becomes laying it over the bowl; no
clear instantaneous switch point.* / *holding roll, locating the film edge,
gradually starting to pull; continuous process.*

This is the option most likely to be missing when someone is forced to choose
between the first two. If the honest answer is "it changes, but not at a
moment", this is that answer.

### `camera_or_viewpoint_shift`
The dominant change is **global** — the camera or the head moved. There is no
visible new object interaction; the scene moved, not the action.

Recorded example: *hand leaves frame while the whole view/scene shifts; no
visible new object interaction.*

### `visibility_or_offscreen`
The moment that matters is **not observable**. Hands or the object leave the
frame, or an occlusion covers exactly the part you would need.

Recorded example: *continues crumpling plastic, then hands and object leave
frame; bin and the discard action are never visible.*

### `annotation_convention`
A split that exists because of a labelling rule, not because of anything
visible. Nothing in the video marks this moment; it is where a convention says
a segment ends.

### `ambiguous`
Cannot be resolved from the clip, and not for any of the specific reasons
above.

---

## How to choose between them

Work in this order and take the **first** that applies:

1. Can you not see the relevant moment at all? → `visibility_or_offscreen`
2. Is the dominant change the camera rather than the action? →
   `camera_or_viewpoint_shift`
3. Is there a change, but no moment you could point to? →
   `gradual_phase_transition`
4. Is there a clear moment where the interaction changes? →
   `sharp_visible_transition`
5. Is it one action throughout, however much motion? →
   `same_action_internal_motion`
6. Is there nothing visible marking this moment, and you suspect a rule put it
   here? → `annotation_convention`
7. Otherwise → `ambiguous`

The order matters. An event can be both "the camera moved" and "hard to tell";
this ordering says which to record.

---

## Rules

**Do not consult the existing labels.** They are in a separate key file and
looking at them destroys the measurement.

**`gradual_phase_transition`, `ambiguous` and `annotation_convention` are real
answers, not failures.** The strongest hypothesis about why the first audit
disagreed 12 times out of 33 is that some of those events belong to these
categories and were forced into a two-way choice. If you find yourself
reaching for sharp-or-same because the others feel like giving up, that is the
bias this sheet is trying to avoid.

**Answer one clip at a time and do not revise after moving on.** Drift across a
re-read correlates with viewing order and looks like signal.

**`why_this_subtype` should name the specific thing** — "the hand releases the
lid and reaches for a different object", not "clear transition". One sentence.

**Confidence is about your own certainty**, not about how clear the video is.
`2_lean` is a normal answer; a sheet of all `3_sure` cannot distinguish a
firm judgement from a habit.

---

## What happens to your answers

Three groups, and you cannot tell which clip is in which — that is deliberate:

- One group is the clips where the first reviewer and the existing label
  disagreed. Your call decides which of them the evidence supports.
- One group is fresh clips from the same hard band. **This is the group that
  actually decides the project's direction**: your agreement with the existing
  labels here estimates whether the taxonomy is reproducible at all.
- One group is easier clips the system already handles automatically, as a
  control. If agreement is high there and low in the hard band, the problem is
  specific to the hard cases; if it is low in both, the taxonomy itself is the
  thing to fix.

If agreement comes out around 0.6–0.7 in the second group, the conclusion is
that no model can learn this target as currently defined, and the next work is
on the definitions rather than on the model. That outcome is a useful result,
not a failed exercise.
