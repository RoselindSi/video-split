# REVIEW-band observability audit — filling guidelines

36 clips, each about 4 seconds, centred on a candidate boundary. For each one
you answer **two separate questions**, in a fixed order.

The point of this audit is *not* to produce better labels. It is to find out
**what information a correct decision requires**, so we stop building
representations that measure the wrong thing. Two rounds have now failed on
these same events: the global visual model reaches 0.528 here, and hand
trajectories reach 0.513 — chance. Neither result says what is missing.
You do.

---

## What every clip is

All 36 are events the frozen policy sends to a human. Every one of them is
already labelled either **sharp visible transition** (a real boundary: one
action ends, another begins) or **same-action internal motion** (motion that
looks like a boundary but is inside one continuous action — a regrasp, a
repetition, a pause). You are not told which.

Events that were excluded as off-screen, camera motion or ambiguous are **not**
in this set. If a clip looks like one of those, that is itself worth recording
(see answer 4), but expect it to be rare.

---

## The order — do not reorder it

### Step 1. Watch, then decide. Before reading anything else.

Play the clip. Decide: **sharp**, **same**, or **cannot**. Record your
confidence: 1 = guess, 2 = lean, 3 = sure.

Do this *before* you think about what evidence you used. Answering "what would
settle this" first produces a tidy explanation for a decision you never
actually made — that is the main way an audit like this goes wrong.

`cannot` is a real answer, not a failure. If you would not stake anything on
either option, write `cannot` and confidence 1. A band where a careful human
is at chance would mean no model can do better, and that is the single most
useful thing this audit could establish.

### Step 2. Now classify what your decision rested on.

Pick exactly one of the four. Work down the list in order and take the **first**
one that applies — they overlap, and the order encodes what we can act on.

---

## The four answers

### 1 — `1_hand_kinematic`
**The hand motion alone settles it.** If you covered everything but the hands
and watched only how they moved — speed, direction, pauses, reversals, whether
one hand or two — you would still give the same answer.

Take this when the decision came from *how the hands moved*, not from what they
moved toward or why.

- Hand decelerates, stops, then sets off in a clearly different direction →
  sharp, on kinematics alone.
- Hand traces the same small back-and-forth four times → same-action, on
  kinematics alone.

Choosing this says: **the information was there and our features missed it.**
That is a claim about our tracking and feature vocabulary, and it is
actionable, so do not choose it loosely. Ask honestly whether you would have
answered the same way with the objects blurred out.

### 2 — `2_object_relative`
**You needed to know what the hand was interacting with.** Which object, and
whether contact was made, held, or released.

- Hand releases the knife and reaches for the bowl → sharp, but only because
  you saw *two different objects*. The motion alone is one continuous reach.
- Hand keeps hold of the same cloth throughout → same-action, but only because
  you tracked the cloth.

Take this whenever object identity, contact or release did the work — even if
hand motion also pointed the same way. Order matters: if the kinematics alone
were **sufficient**, answer 1; if you needed the object, answer 2.

### 3 — `3_semantic_context`
**You needed to know the purpose, or more time than the window shows.**

- The same reaching motion is a boundary or not depending on whether the
  previous step was finished — and you cannot see the previous step in 4
  seconds.
- Two visually identical motions differ because one completes a task and the
  other starts a new one.

Also take this when you found yourself reasoning "in this kind of task, that
usually means…". That is semantic knowledge, not something visible in the clip.

### 4 — `4_not_resolvable`
**No amount of better perception would settle it from this clip.** Hands
occluded or out of frame at the critical moment, the camera swings, or the
boundary reflects a labelling convention with no visual correlate (a segment
that starts at a fixed interval, a boundary placed at a verbal cue).

Take this when the evidence is *absent*, not when it is present but you find it
hard. "Hard to see" is not answer 4; "cannot be seen" is.

---

## `what_evidence_would_settle_it` — one concrete sentence

Not a category, not a restatement of your answer. Name the specific thing.

Good:
- `whether the left hand let go of the lid before reaching`
- `the 3 seconds before the window — was the pouring already finished`
- `which of the two bowls the hand went into`

Not useful:
- `more context` / `better features` / `object information`

This field is what turns a count into a design decision. If ten rows say "which
object the hand entered", that is a specification. If ten rows say "more
context", it is not.

---

## Rules

**Answer one clip at a time and do not go back and revise** once you have moved
on. Consistency drift across a re-read is worse than a few inconsistent rows,
because it correlates with the order you watched them in.

**Do not look up the true label.** It is not in your sheet, and the stratum each
clip was drawn from is deliberately in a separate file. Knowing a clip came
from the low-detection stratum is enough to push you toward answer 4 before you
have watched it.

**Do not skip clips you find hard.** Those are the population. A sheet with the
hard ones missing describes the easy ones.

**Do not let step 1 and step 2 agree by construction.** You can be at
confidence 1 and still answer 2 — "I could not decide, and what I needed was to
see whether the hand released the cup" is a complete, useful row. `cannot` plus
a clear answer 2 or 3 is one of the most informative combinations here.

**A short honest note beats a confident guess.** If you genuinely cannot tell
which of two answers applies, pick one and say so in `notes`.

---

## What comes out of it

Two independent readings, which is why both columns exist:

- **`your_call` / `confidence`** bound what is achievable. If a careful human
  is near chance on this band, no representation will fix it, and the work
  should shift to the annotation protocol rather than to modelling.
- **The four-way distribution** decides the next line. Mostly 1 → our tracking
  and features are the problem. Mostly 2 → hand–object relational state, which
  is the current hypothesis but is *not yet evidence*. Mostly 3 → longer
  context or task structure. Mostly 4 → the band is partly unlabelable and the
  target itself needs revisiting.

36 clips fixes the direction, not the proportions: a 50% share carries roughly
±16 points of uncertainty at this size. Enough to tell "most" from "few", not
enough to quote a percentage.
