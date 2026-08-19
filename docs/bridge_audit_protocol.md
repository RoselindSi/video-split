# Bridge audit — protocol

> **§1–4 describe the batch as designed. §5 amends it after packets
> 176 / 242 / 250 came back: the semantic arm is STOPPED and the remaining
> packets are span-only. The two numbers below are the state at design time,
> not the state now — batch4 has since produced 31 both-class recordings, and
> the span arm's "zero recordings" has become two of the first three.**

This batch does not estimate an error rate. It builds the two within-recording
contrasts that do not currently exist, and without which neither arm can be
measured at all.

    semantic   765 YES/NO pairs, 6 within a recording (0.8%). 38 recordings
               audited, 1 carrying both classes. An AUROC on that structure
               cannot separate "predicts the label" from "recognises the
               kitchen", and the 8B reranker's 0.750 [0.517, 0.912] contains
               the 0.664 a random scorer reaches at 45 versus 17.
    span       among the 244 spans reorder_span has scored, **zero**
               recordings hold both a confirmed internal join and one a human
               judged NOT to be a boundary. The 0.882 / 0.704 split reported
               earlier compares confirmed against UNAUDITED, not against
               rejected.

Both need the same thing. One batch serves both.

## 1. Sampling — the unit is the recording

Targets come from `bridge_sampler`, written to
`data/gold/bridge_audit_targets.json`. Priority is **3 → 1 → 2**:

| tier | meaning | n |
|---|---|---:|
| 3 | one audit from a contrast on BOTH arms | 11 |
| 1 | one audit from a contrast on one arm | 24 |
| 2 | unaudited, rich in candidates on both arms | 5 |

The quantity to maximise is **recordings that gain a contrast**, not events
audited. Auditing 35 packets is ~280 judgements; auditing one event in each of
280 recordings would buy nothing at all.

**No model score selected anything.** Choosing the events a reranker calls
wrong would fill the NO class with the errors that model detects, and
evaluating it on the result would be circular — blinding the auditor does not
fix contamination that happened at sampling. Candidates were selected on
annotation properties only, and each carries `why_selected`: a rare verb, a
label reused five or more times, a duration over three times the recording's
own median, a multi-clause label. For span targets: the two clauses sharing a
verb or an object head, or one side under two seconds.

That enrichment is visible, so it can be conditioned on later. It also means
**nothing here predicts which way a judgement will go**, which is why the
sampler prints a ceiling and never a projection.

## 2. Semantic arm

Goal: **25–30 recordings holding both YES and NO, and ≥50 within-recording
YES×NO comparisons.**

**Audit all four semantic targets in every packet.** Stopping once a NO appears
is the natural efficiency instinct and it defeats the purpose: a recording with
1 YES and 1 NO contributes **one** pair. Fifty pairs needs most recordings at
2×2, which is four. Twenty-five recordings audited to the first NO gives
twenty-five pairs.

**The NO must be a natural label the pipeline produced.** Not a counterfactual.
The paired benchmark already answers the controlled question — 8B scores 0.933
there — and this arm exists to answer a different one: whether an auditor can
judge `claim_support` on annotation people actually wrote. Mixing a synthetic
negative into this set collapses the two questions.

Schema is unchanged and stays unchanged: `claim_support` ∈ yes / partial / no /
uncertain, plus `granularity`, `major_action_missing`, `action_presence`,
`segment_structure`, `upstream_timing_issue`. The **primary endpoint is
`claim_support` YES vs NO only**. `partial` is recorded as real gold and is not
forced into the binary.

## 3. Span arm

Goal: **20–25 recordings holding both a confirmed and a rejected internal
join.** Today that is zero, so the scarce verdict is **"not a boundary"** — a
confirmed join has no counterpart without it.

**Audit the internal joins of spans reorder_span has already scored.**
Confirming one of those upgrades an existing measurement with no GPU;
confirming a newly proposed boundary needs a scoring pass before it is worth
anything. The packets put scored spans first.

The auditor answers **four** questions about the join between clause A and
clause B. They were one free-text question in the first three packets and every
useful distinction had to be read back out of a notes column.

`boundary_exists` is NOT one of them: it is derived from `join_relation` by
`instance_relation_policy_v2`. Asking for the relation and its consequence in
the same sheet puts the policy layer inside the annotation, and the two would
disagree the first time the policy changed.

```
join_relation          new_action | same_action_new_instance
                       same_instance | cannot_determine

candidate_relation     exact | early | late | not_applicable
                       asked SEPARATELY from existence. 242/span4 is a real
                       boundary whose join is ~1.5s late, outside the 1.0s
                       tolerance; collapsing that into one YES loses the only
                       EARLY/LATE evidence in a class holding ten events.

semantic_phase_split   valid | valid_but_labels_poor | weak_or_incorrect
                       | not_applicable
                       what makes a NO informative. Eight of the first twelve
                       joins were not task boundaries WHILE the phase split was
                       valid -- the annotation identified a real change that
                       sits inside one action. "saw a real change in the wrong
                       place" and "saw nothing" cannot share one label.

release_reset_restart  observed_present | observed_absent
                       | uncertain | not_observable
                       the evidence joint_policy_v1's first precedence block
                       turns on. Only observed_absent is an observation of
                       absence. `uncertain` and `not_observable` both block and
                       are kept apart because they are different information
                       even where the policy treats them alike.
```

The contrast the first three packets produced, and the one the remaining
recordings are there to confirm or break:

| task truth | phase split | reset |
|---|---|---|
| NO | valid | absent |
| YES | valid / possible | present |

Both YES cases carried a disengagement and eight of the ten NOs carried a valid
phase split with no reset. If that pattern holds across 20+ recordings it is
real empirical support for the policy's first precedence block, produced by
auditors who could not see the policy.

The original single question, kept because the mapping is still this:

    new_action                 -> BOUNDARY
    same_action_new_instance   -> BOUNDARY
    same_instance              -> NO BOUNDARY
    cannot_determine           -> REVIEW

and, when it is a boundary, the point or interval.

**A change in motion direction is not a boundary.** Wiping left then wiping
right is one instance of wiping. This project has already produced a
motion-phase boundary set once; `reorder_span` is the arm that would be
destroyed by producing a second.

## 4. Blind

The auditor sees the clip, the candidate label, the surrounding context, and
the internal join time. The auditor does not see: any model score, any current
confirmed/unconfirmed classification, the expected class, or which failure
bucket the example came from. Otherwise this gold is written by the hypothesis
it is meant to test.

## 5. Stop conditions — amended after packets 176 / 242 / 250

### Semantic arm — STOPPED

Not because it failed. Because the structure it existed to create now exists,
produced independently by batch4:

```
31 recordings containing both usable semantic classes
361 within-recording YES x NO comparisons
```

Those samples are for **identifiability and paired evaluation, not population
prevalence**. batch4 is not a representative sample and is not being treated as
one; it happens to contain the within-recording comparison structure this arm
was built to obtain, and that is the whole requirement.

The first three bridge packets returned 1 of 3 on this arm — 176 needed a NO and
returned four YES, 250 returned yes and partial. Continuing to spend half the
human budget hunting SEM-NO has low research value once the contrast exists.

### Span arm — CONTINUE, span only

```
Primary stopping condition
    >= 20 recordings containing BOTH
        at least one join whose relation implies a task boundary
        at least one join judged NOT a task boundary

Secondary evidence target
    retain every decisive audited join needed to characterise
        task boundary
        semantic-phase-only split
        timing misalignment

Stop when the primary RECORDING-level condition is reached, or when the
frozen bridge packet pool is exhausted.
```

The target is stated at recording level, not as a join count, because the
problem being solved is within-recording identifiability. More NOs inside
recordings that already have both buy nothing.

**No re-sampling.** The remaining packets keep the frozen order in
`data/gold/bridge_audit_targets.json`. `--arms span` changes only the evidence
requested per packet; the sampling design and the packet sequence are untouched,
so having seen the first three results introduces no adaptive selection.

**2 of 3 is not an extrapolation.** Three recordings, all tier 3, is a planning
note and not a statistical justification — the other tiers may behave
differently and the interval on 2/3 is enormous. The stopping rule above is
stated in outcomes, so it does not depend on a rate being right.

## 6. Three evaluation sets, never merged

| set | question |
|---|---|
| natural semantic paired, same-recording YES vs NO | can an auditor judge `claim_support` on real annotation? |
| boundary-conditioned span, same-recording confirmed vs rejected | where does `reorder_span`'s 0.730 come from? |
| existing counterfactual benchmark, 7 kinds | controlled semantic sensitivity |

They answer different questions and their populations differ. One AUROC over
all three would be a composition statistic, and this project has already read
one of those as a relation.
