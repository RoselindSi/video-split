# Driver A — pre-registration

**FROZEN before the scorer ran.** The model was still downloading when this was
written; `results/batch4_obs_local.jsonl` existed and carried no scores. The
commit timestamp is the evidence, which is the only reason to write this down
at all.

Two operating points in this project were withdrawn after they breached, and
both were chosen after their numbers were seen — a hand-written AUTO_KEEP rule
and a threshold picked on pooled out-of-fold scores. This file exists so the
same move cannot be made a third time on the semantic arm.

## 1. What is measured

**Within-recording paired accuracy, YES vs NO.**

```
for each recording:
    for each observation with support == yes:
        for each observation with support == no:
            1.0 if score_yes >  score_no
            0.5 if score_yes == score_no
            0.0 otherwise
accuracy = mean over all such pairs
```

361 pairs over the 31 recordings that carry both classes, out of 449 scored
observations over 61 recordings.

**Why the pairing must be within a recording.** Both members share scene,
person, camera and session, so a scorer that only recognises the kitchen earns
nothing. The frozen 89-event gold had 6 of 765 pairs within a recording and 1
recording carrying both classes; 8B scored 0.750 [0.517, 0.912] there against
the 0.664 a random scorer reaches at that class ratio — the bar sat inside the
interval and the number could not be read. batch4 is what makes it readable.

## 2. Frozen analysis decisions

| decision | value | why |
|---|---|---|
| primary endpoint | within-recording paired accuracy, YES vs NO | |
| ties | 0.5 | the reranker emits coarsely quantised logits; calling ties wins or losses moves the answer |
| `partial`, `uncertain` | recorded as gold, **excluded** from the primary | 90 and 18 observations; forcing them into a binary invents a judgement nobody made |
| bootstrap unit | **recording**, not pair | pairs inside one recording are not independent; a pair-level interval is falsely narrow |
| interval | 2.5 / 97.5 percentile, 2000 resamples | |
| decision read off | **CI lower bound**, never the point estimate | the withdrawn operating point had a point estimate of 0.789 and a lower bound of 0.591 |
| secondary, reported | global all-pairs accuracy, labelled CONFOUNDED | the number the old arm reported and could not read; contrast only |

**Resolution of this measurement.** On random scores the same code returned
0.501 [0.396, 0.620] — a half-width near 0.11, driven by the 31 recordings and
not by the 361 pairs. More pairs will not narrow it; more recordings would.
A CI lower bound above 0.55 therefore needs a point estimate near 0.66.

## 3. The capability gate

```
PRIMARY GATE:  recording-bootstrap 95% CI lower bound > 0.55
```

Chosen at the resolution of the measurement, not at a preference. A bound above
0.50 is worth nothing — 0.50 is chance, and with a 0.11 half-width a bound that
merely touches it means a point estimate of 0.61. A bound above 0.60 needs a
point estimate near 0.71, which on natural negatives may not be reachable, and
a gate nobody can pass is a decision to quit taken in advance.

### The three branches, and what each actually changes

```
CI lower > 0.55
    the representation carries readable ordering signal on natural
    within-recording pairs
    -> semantic threshold calibration MAY begin
    -> the full-segment version becomes a refinement, not a prerequisite

CI interval includes 0.50
    NOT evidence the scorer cannot do this. This is the sub-window variant
    (3s halves of a 6s clip) and G0 already showed a short window can lose a
    signal that is present -- with the wrong sign. A null here is
    uninterpretable about the scorer and interpretable about the data.
    -> the 3s data may NOT back any semantic automation certificate
    -> the full-segment version becomes REQUIRED before any semantic claim

CI upper < 0.50
    the only genuinely bad branch, and it is diagnosable rather than fatal.
    Scores running opposite to human judgement points at plumbing, not
    capability.
    -> check, in this order: label attribution (prev/next vs positional),
       score direction/sign, the prev-next mapping in the emit step
```

The middle branch is a real decision, not an absence of one: it moves the
full-segment run from optional to required and forbids calibrating a semantic
threshold on 6s windows.

## 4. Capability is not deployment — the two levels

This is the distinction the code now enforces, and the reason a driver A pass
does **not** produce a shippable threshold.

```
Driver A CAPABILITY gate
    within-recording pairwise CI lower > 0.55
    asks:  does this scorer rank a real YES above a real NO?
         |
         | PASS -> permission to BUILD a threshold risk curve
         v
AUTO_ACCEPT gate
    precision CI lower bound  +  errors accepted  +  minimum coverage
    asks:  is `score >= t` safe to ship unreviewed?
         |
         | PASS -> semantic automation certificate
         v
    --semantic_thr may take effect
```

**A paired ranking can be perfect while every absolute score sits on the wrong
side of any fixed cut.** Pairwise accuracy is invariant to any monotone
rescaling; a threshold is not. So `verify_semantic_certificate` refuses a
certificate whose `kind` is `capability`, whatever the number in it, and
`--self_test` asserts that refusal.

**The three AUTO_ACCEPT numbers are deliberately NOT set here.** They are to be
frozen separately, after driver A passes and **before** the risk curve is
looked at. Setting them now would be guessing; setting them after the curve
would be the withdrawn-operating-point move. `configs/auditor/auto_keep_gate_v1.yaml`
ships its three targets null for the same reason, and a null is not a pass.

## 5. What the certificate binds

A semantic automation certificate is void unless all five match the deployment:

```
scorer             which model produced the scores
window             candidate_6s_half vs full_segment -- NOT interchangeable
gold_fingerprint   which gold the curve was measured against
gate_version       which pre-registered conditions it passed
code_fingerprint   digest of the evaluator; a certificate that survives an
                   edit to how ties are scored certifies an unreproducible
                   number
```

## 6. Provenance of the run this pre-registers

```
observations   449 (window, label, verdict) over 61 recordings
               236 candidates | 240 blind rows | 240 local clips
               skipped: 10 R-side label blank, 5 L-side blank, 4 no blind row
verdicts       no 216 | yes 125 | partial 90 | uncertain 18
pairs          31 recordings with both classes, 361 within-recording pairs
window         candidate_6s_half (3s each side of the candidate time)
label side     positional -- prev_segment_label = left, next_segment_label =
               right, per batch3_sample.py:246-249. `containing_segment_label`
               is unused: at a junction it matches whichever side iteration
               reached first, which is why gt_boundary rows split 65 / 20 / 35
               across C==P, C==N and both.
scorer         Qwen3-VL-Reranker-8B, 32 frames
```

---

# 7. VERDICT — 2026-08-21, after the run

```
WITHIN-RECORDING PAIRED ACCURACY   0.641  [0.550, 0.726]
449 observations | 361 pairs | 31 recordings
global (all pairs, CONFOUNDED)     0.562
```

**The capability gate is NOT met.** The exact lower bound is 0.549858, and it
is not a razor edge: eight further bootstrap seeds put it at 0.541, 0.545,
0.546, 0.542, 0.544, 0.538, 0.542, 0.540. The shipped seed produced the
**highest** of the nine. Every one is below 0.55.

**The signal is real.** The interval excludes 0.50 at every seed, so the scorer
does order a real YES above a real NO inside a recording. It is not strong
enough for the bar this document set for beginning threshold calibration.

## The gap in §3, recorded rather than smoothed over

The three branches do not tile the space. This result cleared neither "lower
bound above 0.55" nor "interval contains 0.50", and §3 named no case for it.

The consequence is forced regardless, and that is why the gap does not become a
judgement call. The PRIMARY GATE was stated as a single inequality — lower
bound `> 0.55` — and it was not met. Everything that is not a pass takes the
conservative action:

```
NO semantic automation certificate may be issued on 3s sub-window data
The full-segment run is REQUIRED before any semantic automation claim
```

Which is branch 2's action, reached through the gate rather than through
branch 2's condition.

**0.55 was not changed, before or after seeing this.** A gate moved to admit
the number that just missed it is not a gate.

## What this does NOT say

It does not say the scorer cannot do this. This is the sub-window variant — 3s
halves of a 6s clip — and G0 already showed a short window can lose a signal
that is present, with the wrong sign. 0.641 is a **lower bound on what the
representation can do**, measured under a handicap of known direction.

## What ships anyway

Ordering. `review_lift()` measures what the score is worth to a person working
a queue, and ordering skips nothing, so it needs no threshold and no
certificate. The boundary arm gives 1.25x — the worst-scoring half meets 62% of
the errors rather than 50%.

Ordering only pays if the reviewer stops early, and stopping early is skipping.
The distinction is who decides: a model saying "this one is fine" is automation
and needs a certificate; a person saying "I have two hours" is spending a
budget, with the residual risk explicit and owned.
