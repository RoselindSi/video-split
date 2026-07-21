"""END-TO-END naming with PREDICTED primary (evaluation-validity fix).

N8/N9/N10 all anchored the contrastive secondary scoring on the GT primary
verb -- they are ORACLE-PRIMARY CONDITIONAL results ("given the correct
primary, can the contrastive method find the secondary?"), NOT end-to-end
naming performance. This script measures the real pipeline: the primary is
chosen by the actual deployment module (single-choice MCQ over the object-
conditioned candidate set, ~55-56% accuracy per N4/N7), and the secondary
contrastive scoring is re-anchored on that PREDICTED primary.

Two anchor conditions produce the secondary scores:
  predicted_primary == gt_primary : reuse N9's contrastive scores (anchor
      unchanged) -- no model call
  predicted_primary != gt_primary : re-score ALL secondary candidates with
      the PREDICTED primary as the contrastive anchor (the whole candidate
      set's scores change when the anchor changes, not just one) -- model
      calls, only for the ~44% of items where primary was mispredicted

Candidate-set rule: secondary candidates = all options EXCEPT the
predicted_primary. gt_primary is NOT removed -- so when primary is
mispredicted, the true primary can (and should) be recovered as a secondary.
Both ordered-role and unordered-action-set exact are reported to separate
"actions all recognized but roles swapped" from "action genuinely missed".

Threshold conditions (secondary accept tau), to decompose the drop from the
oracle-primary number into its causes:
  oracle-primary  + frozen tau (=N9's tau) : reproduces N9 as the reference
  predicted-primary + frozen tau           : isolates ANCHOR degradation
  predicted-primary + grouped-OOF tau      : deployable; adds threshold shift

Sanity check enforced: ordered full-exact rate <= primary accuracy (ordered
exact requires primary correct); a violation means oracle info leaked.

Usage (server):
    python -m src.eval.eval_naming_e2e_predicted_primary \
        --model_base /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
        --n7_jsonl /workspace/tr1/results/naming/n7_scored.jsonl \
        --n9_jsonl /workspace/tr1/results/naming/n9_full_contrastive.jsonl \
        --out /workspace/tr1/results/naming/n11_e2e_predictions.jsonl \
        --frozen_tau 10.25 --n_folds 5
"""
import argparse, csv, json, os, random, re
from collections import Counter, defaultdict

import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.eval.eval_naming_n7_scored import resolve_first_token_ids, YES_SURFACES, NO_SURFACES
from src.eval.eval_naming_n8_windowed import score_contrastive, AB_A_SURFACES, AB_B_SURFACES

LETTER_RE = re.compile(r"\b([A-F])\b", re.I)


def single_choice_primary(proc, model, frames, options, obj):
    """The deployment primary module: single-choice MCQ over the candidate
    set. Returns the chosen letter."""
    letters = "ABCDEF"[:len(options)]
    content = [{"type": "image", "image": Image.fromarray(f), "max_pixels": 768 * 28 * 28}
               for f in frames]
    opts_str = "\n".join(f"{l}: {v} {obj}" for l, v in zip(letters, options))
    content.append({"type": "text", "text": (
        "The images are frames in temporal order of a short clip of a person "
        f"acting on the {obj}. Which SINGLE option best describes the MAIN "
        f"action shown?\n{opts_str}\nAnswer with exactly one letter: "
        f"{'/'.join(letters)}.")})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=5, do_sample=False)
    out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = LETTER_RE.search(out)
    return (m.group(1).upper() if m else letters[0])


def grouped_tau(records, n_folds, seed):
    """recording-grouped OOF tau. label(candidate) = (letter in gt_letters):
    a real action in the segment that should be captured in the output set."""
    rng = random.Random(seed)
    rids = sorted({r["recording_id"] for r in records})
    rng.shuffle(rids)
    folds = [rids[i::n_folds] for i in range(n_folds)]
    fold_of = {rid: fi for fi, fold in enumerate(folds) for rid in fold}

    def pairs_of(subset):
        out = []
        for r in subset:
            gt_set = set(r["gt_letters"])
            for l, s in r["e2e_secondary_scores"].items():
                out.append((s, int(l in gt_set)))
        return out

    def best_f1(pairs):
        if not pairs:
            return 0.0
        bt, bf = 0.0, -1
        for tau in sorted({s for s, _ in pairs}):
            tp = sum(1 for s, l in pairs if s > tau and l == 1)
            fp = sum(1 for s, l in pairs if s > tau and l == 0)
            fn = sum(1 for s, l in pairs if s <= tau and l == 1)
            p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            if f1 > bf:
                bf, bt = f1, tau
        return bt

    tau_of = {}
    for fi in range(n_folds):
        train = [r for r in records if fold_of[r["recording_id"]] != fi]
        tau = best_f1(pairs_of(train))
        for r in records:
            if fold_of[r["recording_id"]] == fi:
                tau_of[(r["recording_id"], r["segment_idx"])] = tau
    return tau_of, fold_of


def decode_and_score(records, tau_fn, use_oracle_primary):
    """Returns per-record decode dicts + aggregate counters. tau_fn(record) ->
    threshold. use_oracle_primary: if True, primary=gt (reproduces N9);
    else primary=predicted."""
    err = Counter()
    ordered_exact = unordered_exact = 0
    atomic_ordered = atomic_n = compound_ordered = compound_n = 0
    sec_tp = sec_fp = sec_fn = 0            # unconditional
    sec_tp_c = sec_fp_c = sec_fn_c = 0      # conditional on primary correct
    atomic_false_sec = atomic_seg = 0
    per_rows = []
    for r in records:
        gt_letters = set(r["gt_letters"])
        gt_primary = r["gt_primary_letter"]
        gt_secondary = gt_letters - {gt_primary}
        primary = gt_primary if use_oracle_primary else r["pred_primary_letter"]
        primary_correct = (primary == gt_primary)
        scores = (r["oracle_secondary_scores"] if use_oracle_primary else r["e2e_secondary_scores"])
        tau = tau_fn(r)
        pred_secondary = {l for l, s in scores.items() if s > tau}
        pred_action_set = {primary} | pred_secondary
        # secondary role set the ordered metric compares against
        gt_secondary_role = gt_letters - {primary}  # what "secondary" means given this primary
        ord_ok = primary_correct and (pred_secondary == gt_secondary)
        unord_ok = (pred_action_set == gt_letters)
        ordered_exact += int(ord_ok); unordered_exact += int(unord_ok)

        is_compound = bool(gt_secondary)
        if is_compound:
            compound_n += 1; compound_ordered += int(ord_ok)
        else:
            atomic_n += 1; atomic_ordered += int(ord_ok)
            atomic_seg += 1; atomic_false_sec += len(pred_secondary)

        # unconditional secondary P/R/F1 vs gt_secondary_role
        sec_tp += len(pred_secondary & gt_secondary_role)
        sec_fp += len(pred_secondary - gt_secondary_role)
        sec_fn += len(gt_secondary_role - pred_secondary)
        if primary_correct:
            sec_tp_c += len(pred_secondary & gt_secondary)
            sec_fp_c += len(pred_secondary - gt_secondary)
            sec_fn_c += len(gt_secondary - pred_secondary)

        # mutually exclusive error decomposition
        role_swap = (not primary_correct and primary in gt_secondary
                     and gt_primary in pred_secondary and unord_ok)
        if primary_correct and pred_secondary == gt_secondary:
            cat = "primary_correct_secondary_correct"
        elif primary_correct:
            cat = "primary_correct_secondary_wrong"
        elif role_swap:
            cat = "primary_secondary_role_swap"
        elif unord_ok:
            cat = "primary_wrong_action_set_recovered"
        else:
            cat = "primary_wrong_secondary_also_wrong"
        err[cat] += 1
        per_rows.append({"recording_id": r["recording_id"], "segment_idx": r["segment_idx"],
                        "primary_correct": primary_correct, "ordered_exact": ord_ok,
                        "unordered_exact": unord_ok, "error_category": cat, "tau": round(tau, 3)})

    n = len(records)
    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        return p, rc, 2 * p * rc / max(p + rc, 1e-9)
    return {"n": n, "ordered_exact": ordered_exact / n, "unordered_exact": unordered_exact / n,
            "atomic_ordered": atomic_ordered / max(atomic_n, 1),
            "compound_ordered": compound_ordered / max(compound_n, 1),
            "sec_uncond_prf": prf(sec_tp, sec_fp, sec_fn),
            "sec_cond_prf": prf(sec_tp_c, sec_fp_c, sec_fn_c),
            "atomic_false_sec_per_seg": atomic_false_sec / max(atomic_seg, 1),
            "error_decomp": dict(err)}, per_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--n7_jsonl", required=True)
    ap.add_argument("--n9_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen_tau", type=float, default=10.25)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.eval.run_manifest import print_manifest_if_exists, write_manifest
    print_manifest_if_exists(a.n7_jsonl); print_manifest_if_exists(a.n9_jsonl)
    n7 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n7_jsonl))}
    n9 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n9_jsonl))}
    keys = sorted(set(n7) & set(n9))

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map="cuda").eval()
    a_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, AB_A_SURFACES))
    b_ids = torch.tensor(resolve_first_token_ids(proc.tokenizer, AB_B_SURFACES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w")
    vr_cache = {}
    records = []
    n_reused = n_rescored = 0
    primary_correct_n = 0
    per_verb = defaultdict(lambda: [0, 0])   # gt_primary_verb -> [n, correct]

    for k in keys:
        r7, r9 = n7[k], n9[k]
        letters = "ABCDEF"[:len(r7["options"])]
        gt_primary_letter = r9["primary_letter"]
        gt_primary_verb = r7["options"][letters.index(gt_primary_letter)]
        rid = k[0]
        if rid not in vr_cache:
            vr_cache[rid] = VideoReader(r7["video"], num_threads=1)
        vr = vr_cache[rid]
        frames = [vr[i].asnumpy() for i in r7["frame_indices"]]

        pred_primary_letter = single_choice_primary(proc, model, frames, r7["options"], r7["object"])
        pred_primary_verb = r7["options"][letters.index(pred_primary_letter)]
        primary_correct = (pred_primary_letter == gt_primary_letter)
        primary_correct_n += int(primary_correct)
        per_verb[gt_primary_verb][0] += 1; per_verb[gt_primary_verb][1] += int(primary_correct)

        # oracle-anchor secondary scores (N9, excludes gt_primary_letter)
        oracle_secondary_scores = dict(r9["contrastive_scores"])

        # e2e secondary scores: anchored on predicted primary, candidates = all
        # except predicted_primary (gt_primary retained if primary mispredicted)
        if primary_correct:
            e2e_secondary_scores = dict(oracle_secondary_scores)
            n_reused += 1
        else:
            e2e_secondary_scores = {}
            for l, verb in zip(letters, r7["options"]):
                if l == pred_primary_letter:
                    continue
                e2e_secondary_scores[l] = score_contrastive(
                    proc, model, frames, pred_primary_verb, verb, r7["object"], a_ids, b_ids)
            n_rescored += 1

        rec = {"recording_id": rid, "segment_idx": k[1], "object": r7["object"],
               "options": r7["options"], "gt_letters": r9["gt_letters"],
               "gt_primary_letter": gt_primary_letter, "gt_primary_verb": gt_primary_verb,
               "pred_primary_letter": pred_primary_letter, "pred_primary_verb": pred_primary_verb,
               "primary_correct": primary_correct,
               "oracle_secondary_scores": oracle_secondary_scores,
               "e2e_secondary_scores": e2e_secondary_scores}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        records.append(rec)
        print(f"{rid} seg{k[1]}: gt_primary={gt_primary_verb} pred_primary={pred_primary_verb} "
              f"{'OK' if primary_correct else 'WRONG'} "
              f"({'reused' if primary_correct else 'rescored'})")

    n = len(records)
    primary_acc = primary_correct_n / n
    print(f"\n==== END-TO-END naming with predicted primary (n={n}) ====")
    print(f"anchor reuse: {n_reused} (primary correct)  rescored: {n_rescored} (primary wrong)")
    print(f"\nPRIMARY accuracy: {primary_correct_n}/{n} = {primary_acc:.1%}  "
          f"(expect ~55-56% -- if far off, the single-choice module or item set changed)")
    # macro-F1 over primary verbs would need full confusion; report per-verb acc
    print("per-gt-primary-verb accuracy (n>=3 only):")
    for v, (nn, c) in sorted(per_verb.items(), key=lambda kv: -kv[1][0]):
        if nn >= 3:
            print(f"  {v:12s} {c}/{nn} = {c/nn:.1%}")

    frozen = lambda r: a.frozen_tau
    oof_tau, fold_of = grouped_tau(records, a.n_folds, a.seed)
    oof = lambda r: oof_tau[(r["recording_id"], r["segment_idx"])]

    conditions = [
        ("oracle-primary + frozen tau (== N9 reference)", frozen, True),
        ("predicted-primary + frozen tau (anchor degradation)", frozen, False),
        ("predicted-primary + grouped-OOF tau (deployable)", oof, False),
    ]
    all_summ = {}
    all_rows = {}
    for label, tau_fn, oracle in conditions:
        summ, rows = decode_and_score(records, tau_fn, oracle)
        all_summ[label] = summ; all_rows[label] = rows
        print(f"\n--- {label} ---")
        print(f"  ordered full-exact: {summ['ordered_exact']:.1%}   "
              f"unordered action-set exact: {summ['unordered_exact']:.1%}")
        print(f"  atomic ordered-exact: {summ['atomic_ordered']:.1%}   "
              f"compound ordered-exact: {summ['compound_ordered']:.1%}")
        pu = summ["sec_uncond_prf"]; pc = summ["sec_cond_prf"]
        print(f"  secondary P/R/F1 unconditional: {pu[0]:.1%}/{pu[1]:.1%}/{pu[2]:.1%}   "
              f"conditional-on-primary-correct: {pc[0]:.1%}/{pc[1]:.1%}/{pc[2]:.1%}")
        print(f"  atomic false-secondary per seg: {summ['atomic_false_sec_per_seg']:.2f}")
        # SANITY CHECK
        if not oracle and summ["ordered_exact"] > primary_acc + 1e-9:
            print(f"  !! SANITY VIOLATION: ordered exact {summ['ordered_exact']:.1%} > "
                  f"primary acc {primary_acc:.1%} -- oracle info leaked into eval")

    # error decomposition (deployable condition)
    deploy_label = "predicted-primary + grouped-OOF tau (deployable)"
    err = all_summ[deploy_label]["error_decomp"]
    print(f"\n=== error decomposition ({deploy_label}, n={n}) ===")
    for cat in ("primary_correct_secondary_correct", "primary_correct_secondary_wrong",
                "primary_secondary_role_swap", "primary_wrong_action_set_recovered",
                "primary_wrong_secondary_also_wrong"):
        c = err.get(cat, 0)
        print(f"  {cat:38s} {c:4d}  {c/n:.1%}")

    # write outputs
    summ_path = a.out.replace("_predictions.jsonl", "_summary.json")
    json.dump({"primary_accuracy": primary_acc, "n": n, "n_reused": n_reused,
               "n_rescored": n_rescored, "frozen_tau": a.frozen_tau,
               "conditions": all_summ}, open(summ_path, "w"), indent=2)
    err_path = a.out.replace("_predictions.jsonl", "_error_decomposition.csv")
    with open(err_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "segment_idx", "primary_correct", "ordered_exact",
                    "unordered_exact", "error_category", "tau"])
        for row in all_rows[deploy_label]:
            w.writerow([row["recording_id"], row["segment_idx"], row["primary_correct"],
                       row["ordered_exact"], row["unordered_exact"], row["error_category"], row["tau"]])
    write_manifest(a.out, input_paths=[a.n7_jsonl, a.n9_jsonl],
                   extra={"primary_accuracy": primary_acc, "primary_prediction_source": "single_choice_mcq",
                          "oracle_cache_reuse": n_reused, "rescore_count": n_rescored,
                          "frozen_tau": a.frozen_tau, "n_folds": a.n_folds})
    print(f"\nwrote -> {a.out}\n       -> {summ_path}\n       -> {err_path}")
    print("\nNOTE: N8/N9/N10 are hereby oracle-primary conditional results "
          "(given correct primary). This is the end-to-end number. Any "
          "'pipeline frozen / auto-labelable' claim should cite THIS, not N9.")


if __name__ == "__main__":
    main()
