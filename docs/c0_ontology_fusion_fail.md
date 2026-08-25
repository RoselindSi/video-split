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
