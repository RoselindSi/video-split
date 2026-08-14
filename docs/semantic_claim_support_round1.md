# claim_support learnability, round 1 — frozen baseline

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
