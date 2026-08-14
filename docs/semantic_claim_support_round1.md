# claim_support learnability — round 1 baseline and round 2 replication

Committed so round 2 compares against a file, not a conversation. Every number
below came from one run and nothing about the procedure may change before the
replication.

    gold        data/gold/semantic_ontology_gold_48.csv / .json
    naming      naming_run_pred_v3.jsonl  (chunk 12, no_repeat_ngram_size 0,
                repetition_penalty 1.0, max_new_tokens auto)
    join        naming_run_join.json      (position + bounds)
    events      naming_targets_48_event_map.json

## Coverage — passed

    events with a usable naming feature   48/48
    joined segments                       197
    segments skipped, not shown in sheet    7
    empty predicted names                   0
    recordings dropped, n_pred != n_gt      0  (0/38 misaligned)
    metric transcription check          38/38  rows reproduce the eval's own
                                               verb_acc and obj_f1
    YES vs NO with features              29 vs 6, over 26 recordings

If discrimination fails later it cannot be blamed on the join, the window
recovery, or naming alignment. That was the point of doing this before more
annotation.

## Discrimination — not shown

Noise bar computed BEFORE the features were scored: a random scorer reaches
**AUROC 0.753** at the 97.5th percentile with 29 positives against 6 negatives.

| feature     | AUROC | grouped 95%      |
|-------------|-------|------------------|
| verb_min    | 0.520 | [0.155, 0.667]   |
| verb_mean   | 0.606 | [0.155, 0.778]   |
| obj_min     | 0.684 | [0.455, 0.859]   |
| obj_mean    | 0.710 | [0.286, 0.922]   |
| generic_any | 0.500 | [0.500, 0.500]   |

Nothing clears the bar. The honest statement is:

> object-side naming agreement is the most promising hand-designed signal so
> far, and at 29 YES / 6 NO it has not exceeded the pre-defined random AUROC
> bound.

NOT: object agreement can verify semantic correctness.

## Two things that must not be quoted as findings

**`video prior = 0.601` against `obj_mean = 0.710` is not evidence that
reading the label adds information.** The prior was measured on 186 events
with the target `correct` against everything else; this is 35 events with
`yes` against `no`. Different population, different label, no paired
significance test, and both samples tiny. Directional description only.

**verb agreement is weaker than object agreement** (0.520 / 0.606 against
0.684 / 0.710). If that survives the enrichment round it is worth explaining
— the naming pipeline may recover *which object* more reliably than *what
action* — but on 6 negatives it is a hypothesis, not a result.

## Frozen for the replication

Round 2 reruns this unchanged. No new features, no fusion, no threshold, no
classifier. In particular: obj_mean being the best is NOT a reason to add
object-specific feature engineering, because doing that turns the replication
into a search on the sample it is meant to test.

    event construction        unchanged
    shown_in_sheet filtering  unchanged
    features                  verb_min, verb_mean, obj_min, obj_mean,
                              generic_any — this list and no other
    grouped bootstrap         recording-clustered, same seed
    null procedure            same, recomputed at the new n

Read in this order:

1. how many clean NO the 41 enrichment rows actually produced
2. whether obj_mean is still strongest, and whether it clears the NEW bar

`NO >= 15` — rerun and judge whether a very simple verifier is worth building.
`NO < 15` — open the second sampling frame and do not touch the model.

## One-line status

Coverage passed; discrimination did not. The bottleneck remains
semantic-negative supervision, not model architecture.


---

# Round 2 — the replication. Neither round-1 observation survives.

Same procedure, unchanged: same event construction, same `shown_in_sheet`
filtering, same five features, same recording-clustered bootstrap, same null
recomputed at the new n. The only differences are the 41 enrichment events and
one bug fix in the event-map key lookup.

    gold      semantic_ontology_gold_48.json + semantic_enrichment_gold_41.csv
    89 audited events: yes 46 / partial 25 / no 17 / uncertain 1
    coverage  88/89 with a usable feature, 1 real gap
    contrast  45 YES vs 17 NO over 32 recordings

Noise bar recomputed at the new n: a random scorer reaches **AUROC 0.664** at
the 97.5th percentile, down from 0.759 at 29 vs 6.

| feature     | round 1 | round 2 | grouped 95% (r2) |
|-------------|---------|---------|------------------|
| verb_min    | 0.520   | 0.559   | [0.413, 0.654]   |
| verb_mean   | 0.606   | 0.570   | [0.370, 0.690]   |
| obj_min     | 0.684   | 0.574   | [0.426, 0.764]   |
| obj_mean    | **0.710** | **0.572** | [0.385, 0.750] |
| generic_any | 0.500   | 0.500   | [0.500, 0.500]   |
| noise bar   | 0.759   | 0.664   |                  |

## Both round-1 observations are answered, and both answers are no

**`obj_mean = 0.710` was small-sample fluctuation.** It falls to 0.572 with
roughly three times the negatives. Round 1 said it was "the most promising
hand-designed signal, not yet above the bound"; round 2 says it was noise.

**`verb weaker than object` does not replicate.** Round 1 had 0.520/0.606
against 0.684/0.710 — a visible gap that was recorded as a hypothesis worth
explaining if it survived. It did not: round 2 has all four features inside
0.559–0.574, with no verb/object separation at all.

**`generic_any` is a constant, not a weak feature.** 0.500 with a degenerate
[0.500, 0.500] interval in both rounds: no prediction in this set contains a
generic word. Removing it from FEATURES in a later round is deleting a column
proven constant, not searching.

The four remaining features sit in a 0.015 band and are highly correlated —
this is one observation, not four.

## What this establishes

Naming-derived verb/object agreement does not discriminate `claim_support`.
And the bottleneck is no longer only the negative count: 17 negatives with a
0.664 bar is enough to see an effect of 0.7, and there is none.

The likely reason is upstream and already measured. On these exact windows,
with alignment guaranteed, the naming model scores verb_acc 0.225 and obj_f1
0.266 against the stored labels. A signal that agrees with the label a quarter
of the time cannot separate labels that are right from labels that are wrong —
its disagreement is dominated by its own error, not by the label's.

## Decision

No semantic verifier is trained on these features. More negatives will not fix
a feature family this flat; a better naming model or a different signal might.
Pool B stays unsampled — the gate it was waiting on (NO >= 15) opened, and the
round it opened for came back negative.
