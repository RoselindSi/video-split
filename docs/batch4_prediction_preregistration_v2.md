# Pre-registration v2: batch4, decomposed rather than attributed

Written 2026-08-06, still before batch4 is labelled (`temporal_truth` empty).
**v1 (`batch4_prediction_preregistration.md`) is kept unchanged as the record
of what was predicted first.** This version exists because v1's central
argument was wrong, and the error is worth stating rather than quietly
replacing.

## What v1 got wrong

v1 argued that batch4's precision would be governed by prevalence alone,
citing "lift over base rate" of 1.27x on the original pool versus 1.63x on
relabelled batch3, and concluding the model was not worse where precision was
lower.

`precision / prevalence` is not a prevalence-invariant quantity. With

$$\text{PPV} = \frac{\pi\,\text{TPR}}{\pi\,\text{TPR} + (1-\pi)\,\text{FPR}}$$

the ratio $\text{PPV}/\pi$ still contains $\pi$, and it rises as $\pi$ falls
even when TPR and FPR are unchanged. v1 used a prevalence-dependent statistic
to argue that prevalence was the whole story.

Reconstructing the operating point at the P1 threshold of 0.75:

| | positives | negatives | TPR | FPR | PPV |
|---|---|---|---|---|---|
| original 145 pool | 108 | 37 | 0.667 | 0.108 | 0.947 |
| batch3, relabelled | 67 | 87 | 0.478 | 0.149 | 0.711 |

Holding the development TPR/FPR fixed and substituting only batch3's
prevalence of 0.435 predicts PPV = **0.826**. Observed is **0.711**. So
prevalence explains roughly half the drop and the conditional behaviour has
also shifted -- positives pass less often and negatives pass more often.

A third mechanism is confounded with both and was not accounted for at all:
`local_events.csv` and `scored_events_relabel_v1.csv` come from **two separate
fits**. `c3_local_eval` scores either the original pool or batch3, never both
in one fit, so their scores are not on a common scale and the medians (0.734
versus 0.315) mix prevalence, conditional shift and scale.

## A blocking precondition, discovered while checking this

There is no model persistence anywhere in the repository -- no `torch.save`,
no `pickle.dump`, no saved weights for the P1 scorer, the local scorer, the
PCA basis or the scaler. Every score is out-of-fold from a fit performed
inside the evaluation.

**Therefore batch4 cannot currently be run as a held-out test of a frozen
artefact.** Any score it received would come from a model fitted with batch4's
own labels present in the training folds. That measures whether the method
works on batch4, not whether this artefact transfers to it, and only the
second is what a reserved test set is for.

Freezing `configs/c3_selective_policy_v2_relabel_v1_combined.json` freezes the
thresholds and nothing that produces the numbers they are applied to. What
must additionally be frozen, before batch4 is labelled or scored:

- P1 scorer weights and intercept
- local scorer weights and intercept
- the PCA basis and the imputer/scaler statistics
- the reliability column's definition
- feature cache identities (path + hash) and the extraction settings
- the commit hash of the code that produced them

The same fix removes the score-scale confound above: one frozen scorer applied
to both development sub-populations makes their score distributions directly
comparable, which is the only way prevalence and conditional shift can be
separated.

## The prediction, restated as a decomposition

No single-cause claim. On batch4, report four quantities and let them separate
the mechanisms:

**A. Observed prevalence** $\pi_4$ -- unknown until labelling; report the
candidate-source breakdown beside it, since 23.3% of `gt_boundary` candidates
in development carry no visible boundary and a different candidate mix moves
$\pi$ for that reason alone.

**B. Observed conditional performance** TPR and FPR at the frozen operating
point. These are the transfer diagnostics; precision is not, because it moves
with prevalence by construction.

**C. Prevalence-only expected precision**, from the frozen development TPR/FPR
and $\pi_4$:

$$\widehat{\text{PPV}} = \frac{\pi_4\,\text{TPR}_{dev}}{\pi_4\,\text{TPR}_{dev} + (1-\pi_4)\,\text{FPR}_{dev}}$$

**D. Residual transport gap** = observed PPV − $\widehat{\text{PPV}}$.

Fixed in advance:

- TPR and FPR compatible with their development intervals, and observed PPV
  close to $\widehat{\text{PPV}}$ (D near zero) → **prevalence-dominated**. A
  constraint breach here is not evidence about the representation, and the
  response is to change how the decision rule is defined, not the model.
- TPR and FPR clearly degraded → **conditional transport failure**, on top of
  whatever prevalence contributes. This is evidence about the representation
  or about recording-level generalisation.
- Both → report both; the split between them is D.

Also fixed in advance, independent of the above: the nested-selection
diagnostic already breaches on development (primary 1/5 folds below the
precision constraint, 2/5 above the false-reject constraint; secondary 2/5 and
2/5). A batch4 breach of either constraint has prior evidence behind it and
must not trigger retuning.

## What this does not claim

Not that the policy is good. Not that prevalence is the main cause -- v1
claimed that and was wrong. This fixes what each outcome will be taken to
mean, before the data can influence the reading.

## Why the care

batch4 is spent once. 233 unused recordings remain and there is no batch6
after batch5. A breach read as "the model failed" sends the next round into
representation work while the prevalence sensitivity and the missing frozen
scorer both survive for batch5 to rediscover.
