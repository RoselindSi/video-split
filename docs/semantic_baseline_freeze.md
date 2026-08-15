# Semantic verifier baseline — frozen

Benchmark `data/gold/paired_semantic_benchmark.jsonl`, 438 pairs / 103 segments
/ 26 recordings. Evaluator `src/auditor/semantic/paired_null.py` at `9713ae4`.
Scores: `paired_cosine_scores_v4.jsonl` + `cosine_paired_ext_v4.jsonl`,
`reranker_paired_scores_v2.jsonl`. Both arms are 2B, both scored the same 438
pairs against the same 4 wrong-video pairings.

Every pair holds the video fixed and changes the text, so recording identity is
constant within a pair and cancels. `excess` is accuracy minus accuracy under a
video from a **different recording**; the interval resamples recordings.

## 1. Final table — Qwen3-VL-Reranker-2B

| kind | pairs | recs | max/rec | true | null | excess | 95% CI | sep | ties |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| add_claim | 103 | 26 | 12 | 0.922 | 0.649 | +0.273 | [+0.159, +0.375] | 1.29 | 0.06 |
| wrong_object | 101 | 26 | 12 | 0.926 | 0.420 | +0.506 | [+0.390, +0.618] | 1.00 | 0.01 |
| wrong_verb | 92 | 26 | 10 | 0.668 | 0.447 | +0.221 | [+0.092, +0.339] | 0.42 | 0.12 |
| wrong_qualifier | 56 | 19 | 12 | 0.848 | 0.464 | +0.384 | [+0.260, +0.475] | 1.03 | 0.05 |
| replace_claim | 29 | 14 | 4 | 0.776 | 0.388 | +0.388 | [+0.185, +0.596] | 1.71 | 0.03 |
| drop_claim | 29 | 14 | 4 | 0.741 | 0.237 | +0.504 | [+0.379, +0.621] | 0.77 | 0.10 |
| reorder | 28 | 14 | 4 | 0.625 | 0.411 | +0.214 | [−0.055, +0.421] | 0.25 | 0.25 |
| ALL | 438 | 26 | — | 0.820 | 0.470 | +0.349 | [+0.276, +0.415] | 0.93 | 0.07 |

`sep` = mean \|margin\| over the same model's `wrong_object`. Scale-free, so it
survives a change of architecture where a raw margin and a tie rate do not.

The three clause-surgery kinds are capped at **14 recordings** — they need a
label with ≥2 clauses and a temporal constraint. That, not the pair count, is
the ceiling on what `reorder` can resolve.

## 2. cosine vs reranker, same 438 pairs

| kind | cosine excess | reranker excess | cosine true | reranker true |
|---|---|---|---:|---:|
| wrong_object | +0.436 [+0.312, +0.537] | **+0.506** [+0.390, +0.618] | 0.921 | 0.926 |
| wrong_qualifier | +0.393 [+0.250, +0.514] | +0.384 [+0.260, +0.475] | 0.786 | 0.848 |
| replace_claim | +0.397 [+0.193, +0.591] | +0.388 [+0.185, +0.596] | 0.759 | 0.776 |
| **drop_claim** | **−0.034** [−0.111, +0.044] | **+0.504** [+0.379, +0.621] | 0.793 | 0.741 |
| add_claim | +0.439 [+0.312, +0.555] | +0.273 [+0.159, +0.375] | **0.709** | **0.922** |
| wrong_verb | +0.187 [**+0.000**, +0.343] | +0.221 [+0.092, +0.339] | 0.641 | 0.668 |
| **reorder** | **+0.330** [+0.150, +0.490] | +0.214 [**−0.055**, +0.421] | **0.893** | **0.625** |

**Established.** `drop_claim` is the architecture result. Cosine cannot detect
an omitted claim at all — its excess interval is symmetric about zero and its
null (0.828) is entirely length: a slope fitted on all 541 texts predicts 81%
of it from word count alone. The cross-encoder's null collapses to 0.237,
i.e. under a wrong video it *prefers* the text that claims less. Subset is a
relation a dual encoder has no way to express and a joint forward pass does.

**Object, qualifier, replace_claim: both arms strong, reranker marginally
ahead.** Not an architecture result.

**verb is the weakest axis in both, and only the reranker clears zero.** Cosine's
lower bound is exactly +0.000. `sep` is 0.42 / 0.44 — both architectures put the
two texts less than half as far apart as on `wrong_object`.

**reorder: the cross-encoder is WORSE.** Cosine clears zero and the reranker does
not, on true accuracy 0.893 against 0.625. `sep` is the lowest row in both arms
(0.20 / 0.25). Temporal order is unsolved *for the cross-encoder*; the bar for
anything new is cosine's **+0.330**, not the reranker's +0.214.

**`excess` and `true` must be read together.** `add_claim` is the case: cosine's
excess (+0.439) exceeds the reranker's (+0.273) while its accuracy is 0.709
against 0.922. Cosine's `add_claim` null is 0.269 — the overclaimed text is
LONGER, so the length prior backs the wrong side, and an adverse prior inflates
excess. Excess answers "how much of this needed the right video", not "how often
is it right".

## 3. Known defects, frozen as caveats

These are recorded rather than repaired. Fixing one means re-emitting the
benchmark and rescoring both arms, and the numbers above would then describe a
different dataset than the one that produced them.

1. **`e2` leakage.** At least one frozen decomposition uses an entity id as a
   qualifier VALUE, so `e2` entered the qualifier vocabulary and appears in
   counterfactuals (`Hand rinsing under e2`). No annotator would write it, so a
   model rejects it for the wrong reason and `wrong_qualifier` is inflated by an
   unknown amount. Affects a small number of the 56 pairs.
2. **`wrong_qualifier` history is not reproducible.** Until `21e26e3` the
   qualifier pool was an unsorted `set`, so `rng.choice` drew from an order set
   by `PYTHONHASHSEED`: 41 of 56 pairs moved between two runs that changed
   nothing. Numbers published before that commit are one unrepeatable draw.
   Fixed going forward; the pre-fix sample cannot be regenerated.
3. **A borrowed clause is only *probably* false.** `add_claim` and
   `replace_claim` take a clause from another label, rejected if its verb or
   head noun already appears in the target. Nothing verifies it is absent from
   that segment's video — an adjacent action may genuinely be present. This
   biases both kinds toward the null, so a positive result is conservative and a
   null result is not evidence of absence. `note` records what was borrowed.
4. **Scores are not bit-reproducible across benchmark versions.** The same pair
   scored with a different batch composition moves: over pairs the guard
   verified unchanged, reranker accuracy shifted 0.006–0.018 and `drop_claim`'s
   tie rate halved, 0.21 → 0.10. Accuracy differences of that size touch no
   conclusion here; **tie rate is not a usable statistic across runs**.

## 4. Next stage — pre-registered

Two hypotheses for `atomic text query → frame-level features → temporal
evidence`, measured on this frozen benchmark with this evaluator.

- **H1** — `wrong_verb` excess rises above the reranker's +0.221, with a lower
  bound clear of zero. Well powered: 92 pairs over 26 recordings.
- **H2** — `reorder` excess exceeds **cosine's +0.330** with a lower bound clear
  of zero. Underpowered as it stands: 28 pairs over 14 recordings, and the
  spread is the problem rather than the count — `drop_claim` has the same 29 /
  14 / 4 structure and half the interval width (±0.123 against ±0.238), so the
  reorder effect is heterogeneous across recordings, not merely thin. **Widen
  this arm across more recordings before H2 is tested**, or H2 can only detect a
  large effect and will return "inconclusive" independently of the truth.
- **Separation, not ties.** Report `sep` for `wrong_verb` (0.42 reranker / 0.44
  cosine) and `reorder` (0.25 / 0.20). Tie rate was proposed for this and is
  withdrawn: it is an artifact of output quantisation, so a model emitting
  floats scores ~0 ties with no additional competence, and it moved by half on
  unchanged pairs (defect 4).
- **Non-regression controls** — `wrong_object`, `wrong_qualifier`,
  `drop_claim`, `add_claim`, `replace_claim`: excess lower bounds must not fall
  below the frozen values, and `true` must not fall. Both are required, because
  §2 shows a scorer can raise excess while losing accuracy.

## 5. Amendment — the reorder target, measured at scale

Recorded after §1–4 were frozen. The frozen table is not edited; this section
says which of its numbers may not be used and what replaces them.

**The frozen `reorder` excess is unstable across null draws.** The same 28
pairs, rescored under three different sets of wrong-video pairings, gave
**+0.330 / +0.250 / +0.232** (nulls 0.562 / 0.643 / 0.625). At 28 pairs over 14
recordings, redrawing which wrong video each pair receives moves the point
estimate by 0.1. §2's "cosine beats the reranker on temporal order" rests on
that number and is weaker than it reads there.

**At scale, under the identical construction, cosine's reorder excess is
+0.093.** `reorder_label` — one segment, one annotator's label, its `and`
clauses swapped, exactly the frozen construction — over 495 pairs and 141
recordings: true 0.659, null 0.565, **excess +0.093 [+0.028, +0.158]**. It
clears zero and it is 40% of the frozen estimate, with a half-width of ±0.065
against the frozen arm's ±0.238.

The only variable between the two is that the frozen originals were audited
`claim_support=yes` and these were not. Whether the audited subset is genuinely
cleaner or unaudited labels attenuate the effect, both readings say the same
thing about which number to build on.

**Two constructions that do not work.**

| arm | n | recs | excess | 95% CI |
|---|---:|---:|---:|---|
| `reorder_then` — frozen 28, `and` → `then` | 28 | 14 | +0.134 | [−0.048, +0.304] |
| `reorder_span` — two segments joined, order from timestamps | 385 | 116 | −0.019 | [−0.078, +0.035] |

`reorder_span` was designed to be the clean arm — its ground truth is the
segment boundaries rather than the annotator's ordering phrase — and it is at
chance with a tight interval. Two explanations died before this table: all 28
frozen pairs are joined by `and` alone, so no connective leaks; and span
accuracy RISES with duration (0.410 / 0.536 / 0.565), so eight frames over a
long window is not the constraint. It changed the window, the join word and the
audit status at once, and each of the two isolable changes costs excess.

**§2's temporal conclusion is REVERSED at scale.** Both arms on
`reorder_label`, 495 pairs over 141 recordings:

| arm | cosine | reranker |
|---|---|---|
| `reorder_label` excess | +0.093 [+0.028, +0.158] | **+0.261 [+0.206, +0.314]** |
| `reorder_label` true | 0.659 | **0.820** |
| `reorder_span` excess | −0.019 [−0.078, +0.035] | +0.061 [−0.004, +0.130] |
| `reorder_then` excess (n=28) | +0.134 [−0.048, +0.304] | +0.272 [−0.020, +0.516] |

The intervals on `reorder_label` are disjoint. §2 says "reorder: the
cross-encoder is WORSE" and sets H2's bar at cosine's +0.330; both come from 28
pairs over 14 recordings and both are wrong. **The cross-encoder reads order
better, and by a margin no reading of the frozen arm would have found.**

**H2 is restated.** Target arm `reorder_label`. The bar is the reranker's
**+0.261 [+0.206, +0.314]**, with a half-width of ±0.054.

**And H2's motivation weakens.** At scale the cross-encoder's order excess
(+0.261) exceeds its own `wrong_verb` excess (+0.221 [+0.092, +0.339]) and its
interval is four times tighter. Order is not the axis in most trouble — verb
is, in both architectures and on every arm measured. A frame-level temporal
model should be justified on verb first; H2 remains worth testing, but "temporal
order is unsolved" is no longer supported.

`reorder_span` stays dead: near chance for both arms. Its `then` join helps the
reranker and hurts the cosine, so the join word is an encoder-specific effect
rather than a construction defect — both at n=28 with heavily overlapping
intervals, so that is a hint, not a finding.

One caveat on `sep` in this run: `--reference_kind reorder`, so its column is
not comparable with §1's, which normalised on `wrong_object`. In §1's units the
reranker's `reorder_label` separation is about 0.30 — high accuracy carried by
small margins.

**Length is not a monotone prior.** On this longer-text pool (5–19 words, mean
10.7) `corr(words, score)` is **−0.294**, slope −0.0075 per word — opposite in
sign to the +0.0032 measured on the 438-pair pool (mean 5.8 words). Any
correction of the form "subtract k × word count" is wrong; the effect reverses.
