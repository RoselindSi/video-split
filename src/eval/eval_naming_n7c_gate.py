"""N7 follow-up (2) -- explicit has_secondary binary gate, replacing both the
abandoned 1-6 count predictor (N6.1: 5.2% exact, MAE 2.69) and the
uncalibrated per-candidate threshold (N7: secondary precision 14.5%, FP/seg
1.26 even with oracle primary). Downgrades the question from "how many
actions" to the strictly easier "is there a second action at all, yes/no" --
same YES/NO next-token-logit scoring mechanism as N7's candidate scoring, no
free-text generation to parse.

Reuses the EXACT SAME frames N7 scored candidates on (loaded via the
frame_indices N7 already saved per item), so this is a controlled add-on,
not a different input condition.

Reports the full binary-classification picture, not just accuracy (the N7
diagnosis explicitly flagged that the item set is 59 single-action / 25
compound -- an unbalanced-class accuracy number alone is misleading):
  overall accuracy, precision/recall/F1 for the "has secondary" class,
  PR-AUC and pairwise ranking AUC on the gate score, and separately the
  single->compound false-positive rate and compound->single false-negative
  rate (the two error costs are NOT symmetric: a false "yes" reintroduces
  N7's over-selection problem on truly-atomic segments; a false "no"
  silently drops a real secondary action).

Usage (server):
    python -m src.eval.eval_naming_n7c_gate \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --prev_jsonl /tmp/n7_scored.jsonl --out /tmp/n7c_gate.jsonl
"""
import argparse, json, os

import torch
from decord import VideoReader

from src.eval.eval_naming_n7_scored import (
    resolve_first_token_ids, YES_SURFACES, NO_SURFACES,
)
from src.boundary.decode_sweep import pr_auc
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image


def score_gate(proc, model, frames, obj, primary_verb, yes_ids, no_ids):
    content_msg = [{"type": "image", "image": Image.fromarray(f),
                    "max_pixels": 768 * 28 * 28} for f in frames]
    content_msg.append({"type": "text", "text": (
        "The images are frames in temporal order of a short clip of a person "
        f"acting on the {obj}. The MAIN action shown is \"{primary_verb} "
        f"{obj}\". Does this clip ALSO contain another DISTINCT action, "
        "besides that main action? Answer YES or NO.")})
    msgs = [{"role": "user", "content": content_msg}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :]
    yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
    no_logit = torch.logsumexp(logits[no_ids], dim=0)
    return (yes_logit - no_logit).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--prev_jsonl", required=True, help="output of eval_naming_n7_scored.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists
    print_manifest_if_exists(a.prev_jsonl)
    items = [json.loads(l) for l in open(a.prev_jsonl)]

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    yes_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, YES_SURFACES))
    no_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, NO_SURFACES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    vr_cache = {}
    records = []
    for it in items:
        rid = it["recording_id"]
        if rid not in vr_cache:
            vr_cache[rid] = VideoReader(it["video"], num_threads=1)
        vr = vr_cache[rid]
        frames = [vr[i].asnumpy() for i in it["frame_indices"]]
        letters = "ABCDEF"[:len(it["options"])]
        primary_verb = it["options"][letters.index(it["primary_letter"])]
        gate_score = score_gate(proc, model, frames, it["object"], primary_verb, yes_ids, no_ids)
        gt_has_secondary = len(set(it["gt_letters"]) - {it["primary_letter"]}) > 0
        rec = {"recording_id": rid, "segment_idx": it["segment_idx"],
               "object": it["object"], "primary_verb": primary_verb,
               "gate_score": gate_score, "gt_has_secondary": gt_has_secondary}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        records.append(rec)
        print(f"{rid} seg{it['segment_idx']} obj='{it['object']}' primary={primary_verb} "
              f"gate_score={gate_score:.2f} gt_has_secondary={gt_has_secondary}")

    scores = [r["gate_score"] for r in records]
    labels = [int(r["gt_has_secondary"]) for r in records]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    print(f"\n==== N7 has_secondary gate (n={len(records)}, "
          f"compound={n_pos}, single={n_neg}) ====")
    auc = pr_auc(scores, labels)
    pos_s = [s for s, l in zip(scores, labels) if l == 1]
    neg_s = [s for s, l in zip(scores, labels) if l == 0]
    wins = sum(1 for p in pos_s for n in neg_s if p > n)
    ties = sum(1 for p in pos_s for n in neg_s if p == n)
    pairwise_auc = (wins + 0.5 * ties) / max(len(pos_s) * len(neg_s), 1)
    print(f"PR-AUC: {auc:.3f}   pairwise ranking AUC: {pairwise_auc:.3f}")

    best_tau, best_acc, best_stats = None, -1, None
    for tau in [x / 20 for x in range(-40, 41)]:
        tp = fp = tn = fn = 0
        for s, l in zip(scores, labels):
            pred = int(s > tau)
            tp += pred == 1 and l == 1
            fp += pred == 1 and l == 0
            tn += pred == 0 and l == 0
            fn += pred == 0 and l == 1
        acc = (tp + tn) / max(len(labels), 1)
        if acc > best_acc:
            best_acc, best_tau, best_stats = acc, tau, (tp, fp, tn, fn)
    tp, fp, tn, fn = best_stats
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    single_to_compound_fpr = fp / max(n_neg, 1)   # single items wrongly gated "yes"
    compound_to_single_fnr = fn / max(n_pos, 1)   # compound items wrongly gated "no"
    print(f"best-accuracy threshold tau={best_tau:.2f} (CAVEAT: same-set, not held-out)")
    print(f"accuracy={best_acc:.1%}  precision={prec:.1%}  recall={rec:.1%}  F1={f1:.1%}")
    print(f"single->compound false-positive rate (wrongly says 'yes' on an "
          f"atomic segment, reintroduces over-selection): {fp}/{n_neg} = {single_to_compound_fpr:.1%}")
    print(f"compound->single false-negative rate (wrongly says 'no', silently "
          f"drops a real secondary action): {fn}/{n_pos} = {compound_to_single_fnr:.1%}")

    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=[a.prev_jsonl],
                   extra={"n": len(records), "auc": auc, "best_tau": best_tau,
                          "accuracy": best_acc})


if __name__ == "__main__":
    main()
