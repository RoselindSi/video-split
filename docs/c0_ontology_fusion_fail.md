# C0 — pre-registered parameter-free ontology fusion — **FAIL**

Frozen 2026-08-25. Level C0 of the ablation in
`configs/auditor/ontology_v1_constitution.yaml`. This record exists so the
pooled numbers below can never be quoted on their own.

## What was run

    E = logit(p_detector) + eta_pt * log P(POINT) - eta_nt * log P(NO_TRANSITION)
    eta_pt = eta_nt = 1, eps = 1e-8, nothing fitted

Evaluation: the frozen pool — 3707 candidates over 36 recordings, 2124 true
boundaries, 1316 candidates matched to audited `false_mid_segment`. Morphology
predictions from `morphology_external_oof.jsonl`, trained on recordings
disjoint from these 36. Tolerance 1.0s.

## Result

Pooled:

| | detector | morph only | E_onto |
|---|---|---|---|
| AUROC true vs false_mid | .6763 | .7122 | **.7225** |
| top-5% true rate | .849 | .870 | **.919** |
| top-10% true rate | .841 | .889 | **.903** |
| top-20% true rate | .804 | .880 | **.892** |

Holding recording fixed:

| | detector | morph only | E_onto |
|---|---|---|---|
| same-recording pair accuracy | **.5400** | .4992 | .5099 |
| true-boundary within-recording pctile | **.5382** | .5058 | .5132 |
| false_mid within-recording pctile | **.4678** | .4872 | .4804 |

Term scales:

| | detector logit | ontology term |
|---|---|---|
| std over the pool | 0.716 | 6.288 |
| mean std within a recording | 0.579 | 2.793 |
| range | −0.20 … 4.02 | −12.39 … 12.20 |

The ontology term is larger in magnitude than the detector term on **93.2%**
of candidates.

## Verdict

**FAIL on the core ranking hypothesis.** The pooled improvement is not
accepted as evidence, because it does not survive a recording-controlled
evaluation. The failure C0 was built to attack — `signal_present_not_top`, a
real boundary ranked below an internal motion **in the same video** — got
worse, .5400 → .5099.

## Why the pooled columns look good anyway

Both true boundaries and audited false_mid are unevenly distributed across the
36 recordings. A score that merely knows which RECORDINGS are boundary-rich
raises pooled AUROC and top-k without ever separating two candidates a person
is actually choosing between. This is the same failure pattern as the natural
semantic evaluator, whose AUROC was inflated by recording identity.

## The finding that outlives C0

`morph only` same-recording pair accuracy is **.4992** — chance. The learned
morphology head has no candidate-local within-recording discrimination on this
pool. That reframes three earlier results:

- **Flat confidence** (p_nt .5 → .9 leaving veto composition unchanged) is not
  a calibration problem. There is no local ordering to be confident about.
- **The 27–40% of oracle gain** captured by the morphology veto was measured
  at a GLOBAL threshold, which can ride the same between-recording signal.
  Whether that gain is between-recording is **an inference here, not a
  measurement** — see `recording_shortcut_diag.py`.
- **The oracle stays strong while the learned head cannot reach it** because
  the two hold different kinds of information: the oracle knows *this
  candidate is an internal motion*; the head appears to know *this recording
  looks like such-and-such a regime*.

The detector's own .5400 belongs in the same sentence. On the within-recording
discrimination the auditor needs, **nothing currently in the system is far from
chance**, and its pooled AUROC of .6763 is therefore also substantially a
between-recording effect.

## Two things this result forbids

**No eta tuned on these 36 recordings.** Choosing eta after seeing this table
is selecting on the test set, and the 79 development pairs cannot host the
held-out that would then be required. The scale assumption hidden in eta=1 is
real — 8.8× at pool level — but it is not what makes C0 fail: adding a term
that is at chance within recordings can only inject noise there, so the
optimal eta for the within-recording metric is 0, not a smaller positive
number.

**No move to D (duration priors, segment graph, Viterbi, structured NLL).**
DP optimises the energy it is given. If candidate energy cannot separate a
true reset from an internal motion, global decoding propagates that failure
more systematically rather than repairing it. Level D stays gated.

## Also forbidden by arithmetic, not by policy

Subtracting a recording median, z-scoring within a recording, and taking
within-recording rank percentile are **monotone within a recording**. They
cannot change any within-recording comparison, so none of them can raise
.4992. Recording-relative scoring is a fix for pooled metrics and for the
scale a term carries into a sum. `transform_invariance` prints all four side
by side.

## Next — diagnosis, not training

`src/auditor/boundary/recording_shortcut_diag.py`:

- **A.** ICC / variance decomposition of each score — how much of it is *which
  recording*.
- **B.** Pair accuracy per recording, with a cluster bootstrap over the 36.
  The question is whether `morph only` is ~.5 **everywhere** (no local signal
  in the representation) or **split** — .8 in some recordings, .2 in others
  (local signal with an unstable sign, a different and more repairable
  problem).
- **C.** What the recording-level offset tracks. `true_boundary_density` is
  the one to read first: a strong correlation there is the mechanism itself.

---

## Addendum 2026-08-25 — why the reranker was not written

The next step after C0 was a residual pairwise scorer trained on
within-recording contrasts. It was not written, and the reason is supply.

batch4 is disjoint from the evaluation pool (61 audit recordings and 62
manifest recordings against the 36, after normalising `4` and
`recording_000004` to the same spelling). But it is **117 injected
`gt_boundary` points and 115 `raw_change_peak`**. A reranker only ever sees
detector peaks at inference, so an injected annotation time is an
out-of-distribution positive: training on it produces a model that recognises
annotated instants, scores well in development, and does nothing in
production — a mismatch no development metric reveals.

Restricted to detector peaks on both sides:

| | recordings with both | ≤60s | pairs | pos | neg |
|---|---|---|---|---|---|
| strict | **6** | 1 | 12 | 20 | 48 |
| + mislocalised positives | 8 | 2 | 17 | 32 | 48 |
| + no-action negatives | 8 | 2 | 14 | 20 | 59 |
| everything admitted | 10 | 3 | 19 | 32 | 59 |

Six recordings is not a gate that was narrowly missed. It is an order of
magnitude short of the 40–50 the ranking hypothesis would need, and no
relaxation of the label policy reaches it.

## The number that redirects the work

**52 of 117 injected stored-GT boundaries were audited as NOT boundaries**
(q = 0.444). `initial_action_start` and `terminal_action_end`, 19 rows, are
excluded from q: they are real events at the ends of a recording, outside the
inter-episode definition but not the label being wrong. Folding them in gives
q = 0.617, which is the more flattering error to make.

The evaluation pool's `is_true_boundary` comes from the same stored ground
truth and was never audited, while its `false_mid_segment` negatives were. The
two sides of every pairwise comparison therefore do not have the same label
quality. Positives that are not boundaries are indistinguishable from
negatives, which pulls an observed accuracy toward .5:

    true ≈ (observed − q/2) / (1 − q)
    detector macro    .663 → ≈ .79
    morphology macro  .531 → ≈ .55

An estimate carried from batch4 to the 36, not a measurement on them. It does
not reorder the arms. It changes what is left to win: **a ranker has less room
than it appeared, and the labels are holding more of the gap than any model
change measured so far.** That agrees with three earlier independent results —
relabelling 2.6% of events moved all-clean by +0.025 (p = 0.000), model and
human both score 0.74 against the stored labels while scoring 0.85 against
each other, and the double audit found the outlier was the stored label rather
than the annotator.
