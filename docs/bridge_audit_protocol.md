# Bridge audit — protocol

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

The auditor answers one question about the join between clause A and clause B:

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

## 5. Stop conditions — counted in usable contrasts

```
Semantic   >=25 recordings with BOTH yes and no
           >=50 within-recording yes x no comparisons
Span       >=20 recordings with BOTH a confirmed and a rejected internal join
Quality    >=20% double-audited, or every decisive bridge case double-audited
```

No target number of events. If it takes 80, it takes 80; if it takes 140, 140.
An event count would let the batch finish without the contrasts existing, which
is exactly the state this batch is fixing.

## 6. Three evaluation sets, never merged

| set | question |
|---|---|
| natural semantic paired, same-recording YES vs NO | can an auditor judge `claim_support` on real annotation? |
| boundary-conditioned span, same-recording confirmed vs rejected | where does `reorder_span`'s 0.730 come from? |
| existing counterfactual benchmark, 7 kinds | controlled semantic sensitivity |

They answer different questions and their populations differ. One AUROC over
all three would be a composition statistic, and this project has already read
one of those as a relation.
