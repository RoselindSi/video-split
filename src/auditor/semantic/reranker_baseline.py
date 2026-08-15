"""Baseline 2: a cross-encoder on the identical 306 pairs, with the same null.

THE HYPOTHESIS IS ARCHITECTURAL, and it is the only one the existing data can
answer without being changed. A dual encoder pools the whole label into one
vector and compares it to one video vector, so `a claim is missing` is a subset
relation it has no way to express -- dropping a claim does not push the text
away from the video, it moves it somewhere else that is still compatible. That
is a structural limit, not a capacity shortfall, and no amount of training a
dual encoder fixes it. A cross-encoder sees the frames and the words in the
same forward pass and can attend from one to the other.

So: the same 306 pairs, the same benchmark file, the same evaluator, and the
same wrong-video null. Nothing about the data moves, which is what makes a
difference in the table attributable to the architecture.

WHAT WOULD COUNT. Not the ALL row, and not drop_claim's accuracy -- a
cross-encoder can carry its own preference for longer, more complete-looking
text just as the cosine did. The claim requires drop_claim's accuracy to
separate from ITS OWN wrong-video null, which under cosine it did not
(0.793 against 0.789). An arm that reaches 0.80 against a null of 0.55 has
shown it is reading the video; one that reaches 0.85 against a null of 0.84
has shown it is reading the sentence.

THE NULL COSTS FORWARD PASSES HERE. The cosine null permuted vectors a
thousand times for free; a cross-encoder has no separable sides, so each
pairing is a full rescoring of all 409 entries. Few pairings are affordable,
which is why the evaluator pools them and takes its precision from the number
of pairs instead. See paired_null.

THE SCORING CALL IS NOT VERIFIED. I have not run this checkpoint, and guessing
signatures from memory has already cost this project four API errors in the
cosine file. So `--inspect` loads the checkpoint and prints what it actually
is -- architecture, module list, score head, whether it is a
sentence-transformers CrossEncoder or a generative yes/no scorer -- and the
scoring mode is a flag chosen from that output rather than a constant compiled
in from memory. `--dry_run` exercises frame extraction, pairing construction,
the recording constraint and the output format with random scores.

Usage:
    python -m src.auditor.semantic.reranker_baseline --inspect \
        --model /workspace/tr1/ckpts/Qwen3-VL-Reranker-2B
    python -m src.auditor.semantic.reranker_baseline \
        --model /workspace/tr1/ckpts/Qwen3-VL-Reranker-2B \
        --benchmark data/gold/paired_semantic_benchmark.jsonl \
        --data /workspace/tr1/results/auditor/naming_run.json \
        --score_mode sentence_transformers --n_pairings 4 \
        --out /workspace/tr1/results/auditor/reranker_paired_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from src.auditor.semantic.cosine_baseline import sample_times, write_frames
from src.auditor.semantic.text_prior_null import permute_across_recordings

INSTRUCTION = ("Given a video segment, judge whether the caption is a correct "
               "and complete description of what happens in it.")


def inspect(path):
    """Read the API off the checkpoint instead of remembering it."""
    print(f"{path}\n")
    for f in sorted(os.listdir(path)):
        if f.endswith((".json", ".txt")) or f in ("1_Pooling", "2_Dense"):
            print(f"  {f}")
    for f in ("config.json", "modules.json", "sentence_bert_config.json",
              "config_sentence_transformers.json"):
        p = os.path.join(path, f)
        if os.path.exists(p):
            print(f"\n--- {f}")
            print(json.dumps(json.load(open(p, encoding="utf-8")),
                             indent=2)[:1800])
    print("\n--- what this decides")
    print("  READ modules.json AND config_sentence_transformers.json, NOT "
          "`architectures`.\n  A trained reranker in this family still says "
          "Qwen3VLForConditionalGeneration\n  there, which on its own reads "
          "as a plain generative model and sends you to write\n  your own "
          "yes/no prompt -- bypassing the score head the checkpoint was "
          "trained with.")
    print("    Transformer + Pooling + Normalize   bi-encoder. NOT a "
          "reranker; this arm would\n"
          "                                        be the cosine arm again.")
    print("    Transformer + LogitScore            sentence-transformers "
          "CrossEncoder:\n"
          "                                        --score_mode "
          "sentence_transformers")
    print("    ForSequenceClassification, 1 label  --score_mode seq_cls")
    print("    none of the above                   --score_mode yes_no, and "
          "the two token ids\n"
          "                                        come from this tokenizer "
          "at run time")

    cst = os.path.join(path, "config_sentence_transformers.json")
    if os.path.exists(cst) and json.load(open(cst)).get(
            "model_type") == "CrossEncoder":
        # THE SIGNATURE, NOT THE RECOLLECTION. The embedding arm of this
        # project lost four rounds to guessed keyword arguments -- encode_video
        # that did not exist, frames that had to be paths, do_sample_frames
        # that routed through processing_kwargs, frames_indices that was
        # required. Whether predict() takes processing_kwargs at all is
        # readable here for free.
        import inspect as _i
        try:
            import sentence_transformers as st
            from sentence_transformers import CrossEncoder
            print(f"\n--- sentence_transformers {st.__version__} "
                  f"(checkpoint was saved by "
                  f"{json.load(open(cst)).get('__version__', {}).get('sentence_transformers')})")
            print(f"  CrossEncoder.predict{_i.signature(CrossEncoder.predict)}")
            # `PairInput` IS THE WHOLE QUESTION. The signature says predict
            # takes one, and whether that is (str, str), a dict per side, or a
            # structured message list decides how a video window is handed
            # over -- which is the same question that took four attempts on
            # the embedding side.
            import sys as _s
            for name, mod in list(_s.modules.items()):
                if name.startswith("sentence_transformers") and \
                        hasattr(mod, "PairInput"):
                    print(f"\n  PairInput, from {name}:\n    "
                          f"{getattr(mod, 'PairInput')}")
                    break
            else:
                print("\n  !! PairInput is not exported by any loaded "
                      "sentence_transformers module")
            doc = (CrossEncoder.predict.__doc__ or "").strip()
            print(f"\n--- CrossEncoder.predict docstring\n{doc[:3500]}")
            if len(doc) > 3500:
                print(f"  ... {len(doc) - 3500} more characters")
        except Exception as e:
            print(f"\n  !! could not read the CrossEncoder API: "
                  f"{type(e).__name__}: {e}")


def answer_ids(proc):
    """The two answer tokens, resolved against THIS tokenizer.

    Hardcoded ids are a silent failure: a wrong id indexes a real logit and
    produces a real-looking score. Leading-space and bare variants are both
    tried and the one that round-trips to a single token wins."""
    tk = getattr(proc, "tokenizer", proc)
    out = []
    for word in ("yes", "no"):
        got = None
        for cand in (f" {word}", word, word.capitalize(), f" {word.capitalize()}"):
            ids = tk.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                got = ids[0]
                break
        if got is None:
            raise SystemExit(f"'{word}' is not a single token in this "
                             f"tokenizer, so a yes/no logit contrast is not "
                             f"defined. Use --score_mode seq_cls.")
        out.append(got)
    return out


def score_batch(model, proc, mode, frames, texts, total_pixels, ids,
                meta=None, pair_order="video_text", prompt_name=None):
    """The one place that touches the model. Returns one score per text.

    THE CALL IS THE ONE THIS REPO ALREADY RUNS. eval_naming_decoupled drives
    Qwen3-VL through apply_chat_template(tokenize=False) -> process_vision_info
    -> proc(text=, images=, videos=, video_metadata=), and that path works on
    this checkpoint family today. Writing a second, tidier-looking path from
    memory is how this project bought four API errors in the cosine file, so
    this mirrors the working one rather than improving on it."""
    import torch
    if mode == "sentence_transformers":
        # READ OFF predict's DOCSTRING AND PairInput, not remembered. A pair is
        # a two-element tuple whose sides may each be a modality dict, so a
        # window is handed over as {"video": [frame paths]}; **kwargs is
        # forwarded to preprocess and forward and names processing_kwargs
        # explicitly, with the same structure the embedding arm already uses.
        #
        # do_sample_frames=False because the frames ARE the sample -- the
        # processor otherwise tries to sample from a list of images and
        # refuses. video_metadata because Qwen3VL writes frame TIMESTAMPS into
        # the prompt from fps, and with pre-sampled frames it cannot infer one
        # and defaults to 24, which flattens a 3s and an 88s segment onto the
        # same apparent duration with the distortion scaling with length.
        vid = {"do_sample_frames": False}
        if meta:
            vid["video_metadata"] = [meta] * len(texts)
        pairs = [(({"video": frames}, t) if pair_order == "video_text"
                  else (t, {"video": frames})) for t in texts]
        kw = {} if prompt_name is None else {"prompt_name": prompt_name}
        out = model.predict(pairs, batch_size=len(pairs),
                            show_progress_bar=False,
                            processing_kwargs={"video": vid}, **kw)
        return [float(x) for x in np.asarray(out).reshape(-1)]

    from qwen_vl_utils import process_vision_info

    prompts, vids_all, meta = [], [], []
    for t in texts:
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": frames, "total_pixels": total_pixels},
            {"type": "text",
             "text": f"{INSTRUCTION}\nCaption: {t}\nAnswer:"}]}]
        prompts.append(proc.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True))
        _im, vv, vkw = process_vision_info(msgs, return_video_kwargs=True)
        fps = vkw.get("fps", 2.0)
        fps = float(fps[0] if isinstance(fps, list) else fps)
        nf = vv[0].shape[0] if hasattr(vv[0], "shape") else len(vv[0])
        vids_all.append(vv[0])
        meta.append({"fps": fps, "total_num_frames": int(nf),
                     "duration": float(nf) / fps})
    inp = proc(text=prompts, images=None, videos=vids_all,
               video_metadata=meta, padding=True,
               return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inp).logits
    if mode == "seq_cls":
        return logits.squeeze(-1).float().cpu().tolist()

    # THE LAST REAL TOKEN, NOT POSITION -1. Batched prompts are padded, and
    # which end carries the padding depends on the tokenizer's padding_side.
    # Reading position -1 is correct under left padding and silently reads a
    # pad token under right padding -- a bug that produces plausible scores.
    # The last position with attention_mask == 1 is correct under either.
    am = inp["attention_mask"]
    last = am.shape[1] - 1 - am.flip(1).argmax(1)
    row = logits[torch.arange(logits.shape[0]), last].float()
    # YES/NO AS A DIFFERENCE, not as P(yes). The absolute yes-logit moves with
    # sequence length and prompt shape; the contrast between the two answer
    # tokens is what stays on one scale across captions of different lengths,
    # which matters here more than anywhere since the length prior is the thing
    # being measured.
    return (row[:, ids[0]] - row[:, ids[1]]).cpu().tolist()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--benchmark",
                    default="data/gold/paired_semantic_benchmark.jsonl")
    ap.add_argument("--data", help="naming_run.json, for the video per recording")
    ap.add_argument("--score_mode",
                    choices=["yes_no", "seq_cls", "sentence_transformers"],
                    default="yes_no",
                    help="chosen from --inspect output, not from memory")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--kinds",
                    help="comma-separated kinds to keep. Every setting in a "
                         "sweep must score the same filtered file, baseline "
                         "included -- batch composition alone moves accuracy")
    ap.add_argument("--frame_dir", default="/tmp/reranker_frames")
    ap.add_argument("--n_pairings", type=int, default=4,
                    help="wrong-video pairings. Each one rescores every entry, "
                         "so this is (n_pairings+1) x 409 forward passes")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28,
                    help="same cap eval_naming_decoupled uses")
    ap.add_argument("--device_map", default="cuda")
    ap.add_argument("--pair_order", default="video_text",
                    choices=["video_text", "text_video"],
                    help="which side is the query. The checkpoint's default "
                         "prompt is \"Retrieve text relevant to the user's "
                         "query.\", which reads as query=video and "
                         "document=caption, but the docstring does not say "
                         "which element the prompt attaches to -- so this is "
                         "a flag rather than a decision compiled in from a "
                         "sentence in a config")
    ap.add_argument("--prompt_name", default=None,
                    help="None uses the checkpoint's default_prompt_name, "
                         "'query'. A different prompt gives a different "
                         "number")
    ap.add_argument("--no_metadata", action="store_true",
                    help="omit video_metadata, letting the processor assume "
                         "fps=24 for every segment. Kept only so the two can "
                         "be compared rather than one silently replacing the "
                         "other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", required=False)
    a = ap.parse_args()

    if a.inspect:
        if not a.model or not os.path.exists(a.model):
            raise SystemExit(f"--inspect needs a checkpoint; {a.model} "
                             f"does not exist")
        inspect(a.model)
        return

    bench = [json.loads(l) for l in open(a.benchmark, encoding="utf-8")
             if l.strip()]
    if a.kinds:
        # A FRAME-COUNT SWEEP CANNOT AFFORD THE WHOLE BENCHMARK, and scoring a
        # subset is not free: batch composition changes with the number of
        # texts per segment, and that alone moved accuracy 0.006-0.018 and
        # halved a tie rate between two earlier runs. So every frame setting
        # in a sweep must score the SAME filtered file, including the baseline
        # setting -- comparing a filtered 16-frame run against the unfiltered
        # 8-frame table would confound frames with batching.
        want = {k.strip() for k in a.kinds.split(",") if k.strip()}
        missing = want - {r["kind"] for r in bench}
        if missing:
            raise SystemExit(f"--kinds names {sorted(missing)}, which this "
                             f"benchmark does not contain; it has "
                             f"{sorted({r['kind'] for r in bench})}")
        bench = [r for r in bench if r["kind"] in want]
        print(f"  --kinds {sorted(want)}: {len(bench)} pairs kept")
    segs, texts = {}, defaultdict(set)
    for p in bench:
        segs[p["segment_uid"]] = p
        texts[p["segment_uid"]].add(p["original"])
        texts[p["segment_uid"]].add(p["counterfactual"])
    uids = sorted(segs)
    n_entries = sum(len(texts[u]) for u in uids)
    print(f"{len(bench)} pairs over {len(uids)} segments; {n_entries} entries "
          f"per pairing")

    rec = np.array([segs[u]["recording_id"] for u in uids], dtype=object)
    rng = np.random.default_rng(a.seed)
    # PAIRING 0 IS THE IDENTITY. It is written to the same file in the same
    # format as the nulls so that the true scores and the null scores cannot
    # come from different runs of the model -- which, with a stochastic frame
    # sampler or a nondeterministic kernel, would put a difference into the
    # excess that is neither the video nor the text.
    pairings = [np.arange(len(uids))]
    for _ in range(a.n_pairings):
        pairings.append(permute_across_recordings(rec, rng))
    print(f"  1 true pairing + {a.n_pairings} wrong-video pairings; "
          f"{n_entries * (a.n_pairings + 1)} scorings")
    for j, p in enumerate(pairings[1:], 1):
        same = int(sum(rec[p[i]] == rec[i] for i in range(len(uids))))
        print(f"    pairing {j}: {same} segments kept their own recording")

    model = tok = ids = None
    if not a.dry_run:
        if not a.model or not os.path.exists(a.model):
            raise SystemExit(f"--model {a.model} does not exist")
        # --data IS ONLY NEEDED FOR ROWS THAT CARRY NO PATH. reorder_label and
        # reorder_span reach recordings naming_run.json never covered, and
        # those benchmarks write the path per row. Demanding --data anyway
        # rejected a file that had everything it needed -- the lookup below
        # already prefers the row's own path.
        no_path = [u for u, p_ in segs.items() if not p_.get("video")]
        if no_path and not a.data:
            raise SystemExit(
                f"{len(no_path)} of {len(segs)} segments carry no `video` "
                f"field, so --data is needed for those. First: {no_path[0]}")
        import torch
        from transformers import AutoProcessor
        if a.score_mode == "sentence_transformers":
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(a.model, trust_remote_code=True)
        elif a.score_mode == "seq_cls":
            from transformers import AutoModelForSequenceClassification as M
            model = M.from_pretrained(a.model, dtype=torch.bfloat16,
                                      device_map=a.device_map,
                                      trust_remote_code=True).eval()
        else:
            # THE CLASS THIS REPO ALREADY LOADS Qwen3-VL WITH. CausalLM is the
            # habitual guess and it does not carry the vision tower.
            from transformers import AutoModelForImageTextToText as M
            model = M.from_pretrained(a.model, dtype=torch.bfloat16,
                                      device_map=a.device_map).eval()
        tok = AutoProcessor.from_pretrained(a.model)
        ids = (None if a.score_mode != "yes_no" else answer_ids(tok))
        if ids:
            print(f"  yes/no token ids from this tokenizer: {ids}")
        os.makedirs(a.frame_dir, exist_ok=True)

    video_of = ({r["recording_id"]: r["video"]
                 for r in json.load(open(a.data, encoding="utf-8"))}
                if a.data else {})
    drng = np.random.default_rng(a.seed + 1)
    rows, done = [], 0
    for j, perm in enumerate(pairings):
        for i, u in enumerate(uids):
            vu = uids[perm[i]]
            tl = sorted(texts[u])
            if a.dry_run:
                sc = drng.normal(size=len(tl)).tolist()
            else:
                vp = segs[vu]
                # THE BENCHMARK'S OWN PATH WINS. naming_run.json only
                # covers the audited recordings; reorder_span reaches 150 more,
                # and looking them up there would silently skip every one of
                # them while the counts above still looked right.
                path = vp.get("video") or video_of.get(vp["recording_id"])
                if not path:
                    print(f"  !! no video for {vp['recording_id']}; {vu} "
                          f"skipped")
                    continue
                frames = write_frames(path,
                                      sample_times(vp["start"], vp["end"],
                                                   a.n_frames),
                                      a.frame_dir, f"{j}_{vu}".replace("/", "_"))
                dur = max(float(vp["end"]) - float(vp["start"]), 1e-3)
                nf = len(frames)
                # THE SAME METADATA SHAPE THE EMBEDDING ARM NEEDED, including
                # frames_indices, which is required and must be an array --
                # that was the fourth API error on the cosine side and there is
                # no reason to rediscover it here.
                meta = (None if a.no_metadata else
                        {"fps": (nf - 1) / dur if nf > 1 else 1.0,
                         "total_num_frames": nf, "duration": dur,
                         "frames_indices": np.arange(nf)})
                sc = []
                for b in range(0, len(tl), a.batch):
                    sc += list(score_batch(model, tok, a.score_mode, frames,
                                           tl[b:b + a.batch],
                                           a.total_pixels, ids, meta,
                                           a.pair_order, a.prompt_name))
                for q in frames:
                    os.remove(q)
            for t, s in zip(tl, sc):
                rows.append({"pairing": j, "segment_uid": u, "video_uid": vu,
                             "text": t, "score": float(s)})
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(uids) * len(pairings)} "
                      f"(segment, pairing) scored", flush=True)

    out = a.out or "reranker_paired_scores.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} scores -> {out}"
          + ("   (DRY RUN, random)" if a.dry_run else ""))
    print(f"  then:\n    python -m src.auditor.semantic.paired_null \\\n"
          f"      --scores {out} --benchmark {a.benchmark} --label reranker")


if __name__ == "__main__":
    main()
