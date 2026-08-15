# Atomic temporal verifier — design

Written after the baseline freeze (`semantic_baseline_freeze.md`) and the verb
supervision audit. Every number below is measured, and the design is shaped by
three of them that closed off the obvious version of this model.

## 0. What the measurements already ruled out

**Finer temporal sampling is not the answer.** 8 → 16 → 32 frames moved
`wrong_verb` accuracy 0.690 → 0.707 → 0.701, `sep` 0.42 → 0.44 → 0.46, and the
verb \|margin\| not at all. Four times the resolution and four times the
compute bought nothing. A model whose contribution is "more frames" or "more
temporal attention over the same pooled representation" is a more expensive
8-frame model.

**Temporal order is not the axis in trouble.** At scale the cross-encoder's
`reorder_label` excess is +0.261 [+0.206, +0.314], above its own `wrong_verb`
+0.221 [+0.092, +0.339] with an interval four times tighter. The frozen 28-pair
arm said the opposite and was wrong. **Verb is the target; order is a
non-regression control.**

**Verb-directed supervision is thin and lands in branch C.** Of 92 verbs, 19
have both a same-cluster positive and a cross-cluster negative; 1 of 33
singletons is covered; 263 positives and 266 negatives in total. That cannot
carry a main objective.

## 1. The hypothesis, in the only form the frame sweep leaves open

If eight frames already carry the motion and thirty-two add nothing, the
missing thing is not resolution — it is **selection**. A verb's evidence lives
in a short stretch of the segment; a global pool over the whole window averages
it against everything else, and averaging more samples of the same window does
not undo that. An object survives pooling because its appearance is present in
most frames, which is exactly why `wrong_object` is the strongest axis
(+0.506, `sep` 1.00) and `wrong_verb` the weakest (`sep` 0.42) in both
architectures.

**H:** a scorer that attends from an atomic claim to individual frames, and is
free to concentrate on a few of them, separates verbs better than one that
scores a pooled window.

## 2. G0 — the gate, before anything is built

H predicts something checkable with the existing reranker and no training: if
verb evidence is localised, then scoring a `wrong_verb` pair on the **best
sub-window** should beat scoring it on the whole segment.

    for each wrong_verb pair, score the original and the counterfactual on
    each of K sub-windows of the segment, and take the pair's decision from
    the sub-window with the largest |margin|

- **max-over-sub-windows clearly beats whole-window** on true accuracy and
  `sep` → evidence is localised, a selector has something to select, build the
  model.
- **no better than whole-window** → even a perfect selector gains nothing, and
  H is false in the form stated. The model as designed should not be built;
  the next question becomes representation, not localisation.

This costs 92 pairs × 2 texts × K sub-windows × 5 pairings. At K=4 that is
under an hour and it can invalidate the whole design before a line of training
code exists. **G0 runs first.**

## 3. Architecture, conditional on G0

Frozen Qwen3-VL video tower, per-frame features, no fine-tuning of the
backbone — the same posture the boundary head used, and the only one the data
volume supports.

    atomic claim  ->  text encoder (frozen)  ->  query
    segment       ->  per-frame features (frozen, 32 frames)
    query x frames -> cross-attention -> per-frame evidence -> claim score

The claim is one `(verb, object, qualifiers)` triple from the frozen
decomposition, not a whole label. A label's score is an aggregation over its
claims, which is what makes `drop_claim` and `add_claim` expressible as
set operations rather than as string-length effects — the failure that killed
the cosine arm.

Trainable parameters: the cross-attention block and the score head only.

## 4. Training data, and the one change that makes it usable

`compound_supervision_v2.jsonl`: 1555 examples over 342 spans and 128
recordings, pool-A recordings excluded by construction (38 spans skipped), so
the evaluation's scenes are not in training.

**The label must not be a function of the text alone.** As emitted, a
paraphrase keeps a plausible verb–object pair and a replacement often does not,
so a model can score `fold mug` low without consulting the video. The measured
version of this concern is mild — the benchmark's analogous `wrong_verb`
construction has a wrong-video null of 0.447, near chance — but the positives
are lexically concentrated in a way the negatives are not: 23 distinct
substitutes with the top five carrying 52%, against 86 substitutes and 14%.

So **every training example is emitted twice**: once with its own video and its
variant's label, once with a video from a different recording and the label NO.
Text is then uninformative about the label on its own, and the model must use
the video to separate them. This is the evaluation's wrong-video null used as
training signal.

**Loss.** Main objective over all five variants (`original`, `drop_claim`,
`reorder`, `replace_verb`, `paraphrase`) plus the wrong-video copies.
Verb-contrastive term as an **auxiliary** loss on the 19 two-sided verbs — that
is branch C, and it is what 263/266 examples can support.

**Do not widen the clusters.** The temptation, given 19 verbs, is to relax the
16 clusters so more verbs get two-sided supervision. The audit already showed
what that costs: keying unclustered verbs under a shared `-1` bucket labelled
79 of 338 cross-cluster substitutions as meaning-preserving. A cluster is the
semantic boundary the model is supposed to learn; widening it teaches that
`wash` and `fold` are the same action, and `sep` is the quantity being raised.
The thing to widen is decomposition coverage — 288 labels, 92 verbs, 33
appearing once.

## 5. Metrics — accuracy first, excess never alone

```
PRIMARY     wrong_verb true accuracy   > 0.701   (the 32-frame reranker)
            wrong_verb sep             > 0.46    (likewise)
SECONDARY   wrong_verb excess = true - null
GUARD       wrong_verb null            not below 0.393
SPLIT       report covered (39 pairs) and uncovered (53) separately
CONSTRAINT  wrong_object, wrong_qualifier, drop_claim, add_claim,
            replace_claim, reorder_label: true must not fall and excess
            lower bounds must not fall below the frozen values
```

**Why the guard.** Excess has moved three times in this project for reasons
other than the scorer improving: `drop_claim`'s length prior, `add_claim`'s
adverse prior, and the frame sweep, where +0.065 of verb excess was entirely
the null falling from 0.447 to 0.393 while accuracy stayed flat. Without the
guard, a model can manufacture excess by becoming more hostile to wrong videos
and change nothing about the judgement being measured.

**Why the split.** Only 11 of the 37 source verbs in the `wrong_verb` arm have
two-sided supervision, covering 39 of 92 pairs — 42%. Reporting the whole arm
dilutes an effect that can only reach a subset, and "no improvement" would not
distinguish a method that fails from one that never reached. If covered
improves and uncovered does not, the bottleneck is coverage, and the next move
is decomposing more labels — starting with the frequent uncovered verbs the
audit named: `flip`, `apply`, `coat`, `secure`, `crumple`, `unroll`, `spread`,
`test`, `reposition`, `stack`.

## 6. Stop conditions

- **G0 shows no localisation gain** → do not build this model.
- **Covered and uncovered both flat** → the auxiliary loss does not transfer;
  the limit is representation, not supervision, and neither more data of this
  kind nor a bigger head is indicated.
- **Excess rises while accuracy does not** → the guard fired; report it as a
  null result, not a gain.

## 7. G0 result — the gate did not pass. Do not build §3.

Reranker-2B, 193 pairs, 3 sub-windows of 8 frames against a whole window of 24,
same run and same batching.

| rule | verb true | verb excess | object true | object excess | verb sep |
|---|---:|---|---:|---|---:|
| whole (24 frames) | 0.717 | +0.296 [+0.180, +0.419] | 0.921 | +0.525 [+0.428, +0.618] | 0.43 |
| max_abs | 0.728 | +0.291 [+0.182, +0.400] | 0.950 | +0.547 [+0.432, +0.656] | 0.47 |
| mean | 0.745 | +0.307 [+0.193, +0.415] | 0.950 | +0.552 [+0.438, +0.656] | 0.37 |

The gate was pre-registered on the difference between the axes. Under
`max_abs`, verb gains +0.011 and **object gains +0.029** — the difference is
−0.018, the wrong sign. Under `mean` both gain +0.029, identical. Verb excess
does not move at all (+0.296 → +0.291 / +0.307).

**`mean` beating `max_abs` on verb (0.745 vs 0.728) is the tell.** If the
evidence were localised, picking the strongest sub-window would beat averaging
three. It does not, which is the signature of variance reduction from three
semi-independent scorings rather than of selection finding anything. At n=92
the SE is 0.047, so +0.028 is 0.6 SE — both axes' gains are inside noise.

**H is false as stated.** A verb's evidence is not a short stretch being
averaged away by a global pool; there is nothing for a claim-conditioned
selector to select. §3 is not built.

### What replaces it

The stop condition said the next question is representation rather than
localisation, and there is a cheaper test of that than any new architecture:
**is the verb axis capacity-limited?** `Qwen3-VL-Reranker-8B` exists and the
sweep tooling already takes `--kinds wrong_verb,wrong_object`, so this is one
run and no new code.

- **8B lifts verb much more than object** → the weakness is representational
  capacity, and the move is a bigger backbone, not a new architecture.
- **both lift proportionally** → verb is not specially hard for the model;
  it is specially hard in this data, and the next question is the 33 singleton
  verbs and the 51 verbs with no supervision at all.
- **neither lifts** → the ceiling is in the frozen decompositions and the
  annotation, which is where this project's other lines have already landed
  twice.

### 8B result — the verb axis was capacity-limited

Reranker-8B, 32 frames, same 193 pairs, same four pairings.

| | 2B @32 | 8B @32 |
|---|---:|---:|
| `wrong_verb` true | 0.701 | **0.902** |
| `wrong_verb` excess | +0.308 | +0.484 [+0.409, +0.560] |
| `wrong_verb` null | 0.393 | 0.418 |
| `wrong_verb` sep | 0.46 | 0.51 |
| `wrong_object` true | 0.911 | **1.000** |
| `wrong_object` excess | +0.537 | +0.603 [+0.512, +0.698] |

Verb error falls from 0.299 to 0.098, a two-thirds reduction, from an
off-the-shelf checkpoint with no training, no new data and no new code. **The
guard passes**: the null went UP, 0.393 → 0.418, so every point of the excess
gain is `true` rising — the pattern the frame sweep failed to produce and the
reason the guard was written.

**Two things this does not say.**

`wrong_object` is at 1.000 over 101 pairs, so that control is now saturated:
it can only fall, and it can no longer resolve any improvement. "8B lifts verb
more than object" is confounded by that ceiling. What is supported is weaker
and still decisive: **both axes were capacity-limited and object hit the top
first.**

`sep` moved only 0.46 → 0.51 while \|margin\| grew roughly fourfold, 0.188 →
0.702. The model did not learn verbs specifically; it became far more decisive
overall and the verb-to-object difficulty ratio survived almost unchanged. More
scale should keep raising verb accuracy, but the structural gap between the two
axes is not what capacity removes.

**Consequence.** Nothing in this document needs to be built to reach verb
0.902. The next reference point is Reranker-8B on the whole benchmark, and any
future model is measured against that rather than against the 2B table in
`semantic_baseline_freeze.md` §1.
