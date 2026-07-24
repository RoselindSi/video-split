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

## Design: three passes, un-anchored

A single "here's the clip and the label, is the label right?" prompt lets the
model rationalize whatever annotation it was handed. So the auditor is split so
the visual judgment happens *before* the model ever sees the annotation:

| Pass | Sees | Produces |
|---|---|---|
| **A — blind** | clip only (no GT, no prediction, no score curve, no error category) | before/after action, object, `semantic_action_changed`, **`motion_change_without_semantic_change`**, candidate boundary time |
| **B — semantic** | Pass A + the *original* segment label(s) | `label_support`, `label_completeness`, `label_granularity`, corrected verb/object |
| **C — temporal** | Pass A + GT time / model time / adjacent labels (+ optional score plot) | `temporal_truth`, `gt_boundary_relation`, `model_boundary_behavior`, corrected boundary time |

Pass A's core job is the distinction the boundary audit flagged as the model's
main failure mode: **motion change ≠ semantic action change**. Repetitive wiping,
a direction reversal, or a regrasp is strong visual motion but *not* an action
boundary.

Recommended routing (per the design review): run Pass A on an **Instruct** model
(faithful description, less prone to language-prior rationalization) and Passes
B/C on a **Think/reasoning** model (they are comparison tasks). Pass two model
ids on the command line; the driver routes each pass.

## Confidence is consistency, not the model's word

`review_confidence` / `auto_proposal_eligible` are **not** the model's stated
confidence. They come from agreement across `--repeats` fps-jittered runs plus a
blind-vs-conditioned check (does Pass C's `temporal_truth` agree with Pass A's
`semantic_action_changed`). Only high-consistency, resolved cases are marked
`auto_proposal_eligible` — and that flag means *"propose a correction for
review"*, never an unconditional overwrite.

## Fusion → Gold v2 fields

The three passes are fused deterministically into the orthogonal Gold v2 schema
(temporal / semantic / policy heads — see `gold_schema.py`). Fusion rules
(`fuse()` in `run_visual_auditor.py`) mirror the gold set's own logic, e.g.
`temporal_truth=spurious` + `motion_change_without_semantic_change=yes` →
`boundary_contrastive_role=motion_hard_negative`.

## Files

```
gold_schema.py          closed vocabularies + gold/context loaders (source of truth)
prompts.py              the three-pass prompts + JSON reply parsing
vision_backends.py      VisionBackend ABC; MockBackend (plumbing test); QwenVLBackend
run_visual_auditor.py   driver: 3 passes × repeats → consensus → fuse → jsonl + manifest
eval_auditor.py         field-by-field scoring vs gold, incl. the 3 hard-case slices
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

**Real run** (server, with clips + weights). Blind pass on Instruct, reasoning
passes on Think, 3 repeats for the consistency signal, score plot attached to
Pass C:

```bash
python -m src.auditor.run_visual_auditor --backend qwen \
    --model_id_a  Qwen/Qwen3.5-VL-27B-Instruct \
    --model_id_bc Qwen/Qwen3.5-VL-27B-Think \
    --media_dir /workspace/tr1/results/boundary/error_audit/media \
    --repeats 3 --score_plot \
    --out /workspace/tr1/results/auditor/auditor_pred.jsonl
python -m src.auditor.eval_auditor \
    --pred /workspace/tr1/results/auditor/auditor_pred.jsonl --show_confusion
```

`--model_id_a` / `--model_id_bc` are just checkpoint ids — swap in whatever is
actually on the box (`--model_id` sets one model for all passes). The clips are
the exact per-event mp4s already rendered for the manual audit, so the auditor
watches what the human watched.

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

- Expand the gold set with **random** (non-anomaly) events so the auditor is not
  calibrated only on hard cases.
- Only after the auditor is trusted: audited-pair contrastive supervision for the
  boundary head (true fast boundaries as positives, high-motion same-action as
  hard negatives) with a multi-scale temporal encoder. No global slow-latent
  smoothness prior — this data has many sub-2s actions.
