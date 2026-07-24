# Visual auditor (MVP)

Can a strong, *independent* video model reproduce a human's structured audit of
a boundary event? That question gates everything downstream: the boundary error
audit showed ~half of what the F1 metric calls "wrong" is actually annotation
noise or granularity mismatch, so we cannot mine training pairs (reranker
negatives, contrastive positives) straight from the model's own false positives
— a large fraction are visually real, unlabeled sub-actions. Before trusting any
automatic label correction, we first measure **where** a video auditor agrees
with the human, field by field, on a frozen 72-event gold set.

This directory is that measurement — **not** a training pipeline. No second-stage
model is trained here.

## Design: three passes, un-anchored, ATOMIC

A single "here's the clip and the label, is the label right?" prompt lets the
model rationalize whatever annotation it was handed. So the auditor is split so
the visual judgment happens *before* the model ever sees the annotation --
**and** Pass B/C are restricted to questions actually answerable from the video
(see "What went wrong first" below for why):

| Pass | Sees | Produces |
|---|---|---|
| **A — blind** | clip only (no GT, no prediction, no score curve, no error category) | before/after action, object, `semantic_action_changed`, **`motion_change_without_semantic_change`**, candidate boundary time |
| **B — semantic (atomic)** | Pass A + the *original* segment label(s), as context only | `observed_primary_verb`, `observed_secondary_verbs`, `observed_object`, `additional_action_visible` — what the video shows, full stop |
| **C — temporal (atomic)** | Pass A + GT time / model time / adjacent labels (+ optional score plot) | `temporal_truth` — the ONE judgment call in this pass |

Pass A's core job is the distinction the boundary audit flagged as the model's
main failure mode: **motion change ≠ semantic action change**. Repetitive wiping,
a direction reversal, or a regrasp is strong visual motion but *not* an action
boundary.

Model routing (`--model_id_a` / `--model_id_bc`): with no Think/reasoning
checkpoint available, both passes currently run on Instruct models; a bigger
Instruct model (32B vs 8B) did **not** fix the anchoring bias — see below.

## Judgment fields are DERIVED, not asked of the VLM (`derive_fields.py`)

**What went wrong first:** early versions asked Pass B/C to directly emit
`label_support`/`label_completeness`/`label_granularity`/`semantic_relation`/
`object_relation` and `gt_boundary_relation`/`model_boundary_behavior`. Across
two full-72 runs (8B, then 8B with an anti-anchoring prompt rewrite) and a
32B-vs-8B comparison, `gt_boundary_relation`/`model_boundary_behavior` were
consistently the worst-scoring fields (8B: near-random on the missed/weak-
response distinction; 32B on a 15-event subset: 3/15 correct, *worse* than 8B).
The reason is structural, not model capacity: **a VLM watching an 8-second
clip cannot know whether a candidate peak was suppressed by NMS vs never
cleared the decode threshold** — that's decode-mechanics information, not
something visible in the video. The semantic fields showed a different but
equally structural problem: a strong anchoring/leniency bias (rubber-stamping
the label as `supported` even when the blind description disagreed) that an
adversarial-framing prompt rewrite only partially fixed, and fixed by
*trading* false-lenient errors for false-suspicious ones (see "Findings so
far").

So these fields are now **computed deterministically** in `derive_fields.py`
from the atomic VLM observations above + already-known structured context:

- `derive_boundary_fields(temporal_truth, source_category, pred_time, pred_score)`
  — `source_category` (e.g. `missed_weak_signal`, `false_mid_segment`, `late`)
  already encodes which automatic-metric bucket produced this event, which is
  enough to derive `gt_boundary_relation`/`model_boundary_behavior` via a
  lookup table with no VLM guessing about decode mechanics at all.
- `derive_semantic_fields(observed_primary_verb, observed_secondary_verbs, observed_object, additional_action_visible, label_text)`
  — compares the VLM's own observation against the original label text using
  the **frozen naming ontology's** own verb/object normalization
  (`src/analysis/build_ontology.py`: `norm_verb`, `OBJECT_NORM`,
  `STRICT_INVERSE`/`CONTEXTUAL_INVERSE` for verb relations, `GENERIC_VERBS` to
  catch correct-but-coarse parent labels like "manipulate the device"),
  instead of asking the VLM to self-report a parent/child/compatible
  relation. This is a best-effort v1 rule, not a claim of parity with human
  judgment — cases it can't resolve return `unresolved`/`unknown` rather than
  guessing, which is the honest signal for "route to human review."

Derivation runs **per-repeat** (not just once on the consensus), so the
consistency-based confidence signal below still measures agreement on the
actual derived judgment.

## Confidence is consistency, not the model's word

`review_confidence` / `auto_proposal_eligible` are **not** the model's stated
confidence. They come from agreement across `--repeats` fps-jittered runs of
ONE model, plus a blind-vs-conditioned check (does Pass C's `temporal_truth`
agree with Pass A's `semantic_action_changed`) — **plus, optionally, real
cross-model agreement** (`--model_id_bc2`: run a second, differently-sized
model for Pass B/C). Same-model repeats alone cannot catch a *systematic*
bias: it reproduces identically across fps-jittered repeats of one model and
reads as high confidence despite being wrong (observed on the server: hard-
slice(2) was 0/17 in the "high confidence" bucket on the first full run).
Only high-consistency, resolved cases are marked `auto_proposal_eligible` —
and that flag means *"propose a correction for review"*, never an
unconditional overwrite.

## Fusion → Gold v2 fields

`fuse()` in `run_visual_auditor.py` combines the atomic pass-through fields
(e.g. `corrected_primary_verb` ← Pass B's `observed_primary_verb`) with the
derived-consensus judgment fields from `derive_fields.py` into the orthogonal
Gold v2 schema (temporal / semantic / policy heads — see `gold_schema.py`),
plus a further layer of rule-derivation that was already deterministic before
this rewrite, e.g. `temporal_truth=spurious` + `motion_change_without_
semantic_change=yes` → `boundary_contrastive_role=motion_hard_negative`.

## Files

```
gold_schema.py          closed vocabularies + gold/context loaders (source of truth)
prompts.py              the three-pass ATOMIC prompts + JSON reply parsing
derive_fields.py         rule layer: atomic VLM observations -> Gold v2 judgment fields
vision_backends.py      VisionBackend ABC; MockBackend (plumbing test); QwenVLBackend
run_visual_auditor.py   driver: 3 passes × repeats [× 2 models] → consensus → derive → fuse → jsonl + manifest
eval_auditor.py         field-by-field scoring vs gold, incl. the 3 hard-case slices + confidence calibration
export_gold_v2.py       regenerate data/gold/*.jsonl from the xlsx + audit CSV (stdlib only)
```

Gold data lives in `data/gold/`: `audit_72_gold_v2.jsonl` (frozen labels) and
`audit_72_context.jsonl` (the original segment labels the auditor verifies).

## Run

**Smoke test** (no GPU, no video, no model — proves the plumbing and the scorer
end to end; the numbers are random by construction):

```bash
python -m src.auditor.run_visual_auditor --backend mock --repeats 3 \
    --out /tmp/auditor_pred.jsonl
python -m src.auditor.eval_auditor --pred /tmp/auditor_pred.jsonl --show_confusion
```

**Real run** (server, with clips + weights). No Think checkpoint currently on
the box, so both passes run on the same Instruct model; add `--model_id_bc2`
for a second model's worth of real cross-model confidence:

```bash
python -m src.auditor.run_visual_auditor --backend qwen \
    --model_id_a  /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
    --model_id_bc /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
    --media_dir /workspace/tr1/results/boundary/error_audit/media \
    --repeats 3 --score_plot \
    --out /workspace/tr1/results/auditor/auditor_pred.jsonl
python -m src.auditor.eval_auditor \
    --pred /workspace/tr1/results/auditor/auditor_pred.jsonl --show_confusion
```

`--model_id_a` / `--model_id_bc` are just checkpoint ids — swap in whatever is
actually on the box (`--model_id` sets one model for all passes). The clips are
the exact per-event mp4s already rendered for the manual audit, so the auditor
watches what the human watched. For a full 72-event run, shard across GPUs by
splitting `data/gold/audit_72_gold_v2.jsonl` into N files and passing each to
`--gold` with a distinct `CUDA_VISIBLE_DEVICES`, then `cat` the outputs back
together — `eval_auditor.py` only cares about `event_id`, not row order.

## Findings so far (pre-atomic-rewrite experiments)

Three real full/partial runs, all on the server, before this file's atomic
rewrite (i.e. all the numbers below are with Pass B/C asking for judgment
fields directly, not through `derive_fields.py` — kept here as the evidence
trail, not necessarily reproducible with the current code):

1. **8B, original prompts, full 72**: hard-slice(2) (motion-hard-negative →
   `spurious`) was 1/20 = 5% — the model almost never confidently rejected a
   motion-induced false boundary; `label_wrong` detection F1 = 0.067 (rubber-
   stamped 17/21 contradicted labels as `supported`).
2. **8B, anti-anchoring prompt rewrite, full 72**: hard-slice(2) jumped to
   12/20 = 60% (confirmed real, not noise, at full n). But hard-slice(1)
   (true boundary kept `valid`) *fell* from 67.4% to 53.5%, and 13/43 real
   boundaries got mislabeled `motion_hard_negative` — a new, more dangerous
   failure mode for training-pair mining than the original one, since it
   would actively teach a reranker to suppress real boundaries. Net: the bias
   moved from "too lenient" to "too suspicious," not eliminated.
3. **32B vs 8B (same anti-anchoring prompts), 15-event subset**: 32B kept
   more true boundaries valid (6/10 vs 3/10) but got *worse* at rejecting
   motion-hard-negatives (1/5 vs 3/5) and scored 3/15 (vs 8B's 0/15) on
   `model_boundary_behavior` — confirming bigger-Instruct-alone doesn't fix
   this, it just moves where the bias lands. This is the finding that
   motivated the atomic-observation + rule-derivation rewrite above.

Confidence calibration was checked at every stage and never useful: same-
model self-consistency reads "high confidence" even when systematically
wrong (hard-slice(2) was 0/17 in the "high" bucket on run 1), because a
stable bias reproduces identically across fps-jittered repeats of one model.

## What to read in the eval

The headline is **not** overall accuracy — it's the per-field/per-slice
trust map. The three hard-case slices are the whole point:

1. **true boundaries kept `valid`** — want high; a low number means the auditor
   dismisses fast real actions as internal motion.
2. **motion-hard-negatives called `spurious`** — want high; low means it still
   calls repetitive motion a boundary (the same failure as the boundary model).
3. **correct-but-coarse / compound labels wrongly flagged `incorrect`** — want
   **low**; coarse is not wrong.

The intended use: auto-act only on the fields/slices where the auditor is
reliable; route the rest to human review. A field being unreliable is a finding,
not a failure.

## Next (gated on this eval, not started here)

- **Immediate**: re-run the full 72 (or a stratified subset first) with the
  atomic prompts + rule-derivation above, to see whether removing the non-
  visual questions from the VLM's job actually closes the gap the three prior
  experiments couldn't, before drawing any conclusion about model choice.
- Build a properly **stratified** 30-ish-event test set (true fast boundary /
  motion-hard-negative / weak-gradual boundary / wrong semantic label /
  correct-but-coarse / ambiguous-exclude, each with real coverage) instead of
  reusing the first N rows of the 72 for quick checks — the 15-event subset
  used for the 32B comparison happened to contain 0 coarse/compound cases,
  which is exactly the slice most likely to reveal `derive_semantic_fields`'s
  current blind spots (e.g. `too_fine`/`mixed` granularity isn't derived at
  all yet, only `too_coarse` via `GENERIC_VERBS`).
- Expand the frozen 72-event gold set itself with **random** (non-anomaly)
  events so the auditor is calibrated on more than just hard cases.
- Only after the auditor is trusted on the above: audited-pair contrastive
  supervision for the boundary head (true fast boundaries as positives, high-
  motion same-action as hard negatives) with a multi-scale temporal encoder.
  No global slow-latent smoothness prior — this data has many sub-2s actions.
