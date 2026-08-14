"""A0.2+A0.3 — Decoupled naming eval.

Feed GT boundaries, ask the model to NAME each segment (naming isolated from
boundary errors). Score per aligned pair with a DETERMINISTIC decomposed metric:
verb-cluster match + content-word(object) F1 + genericity + embedding sim.
An independent LLM-judge is run separately by score_names.py.

Import shim below makes this run under BOTH layouts:
  - server (time-r1): flat  `src/seg_rewards.py`
  - repo   (video-split): nested `src/rewards/seg_rewards.py`

Usage (server):
    python eval_naming_decoupled.py --model_base /workspace/tr1/ckpts/<model> \
        --out /tmp/naming_<model>.jsonl
"""
import argparse, json, os, re, statistics, time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

try:                                            # server flat layout
    from src.seg_rewards import _as_segs, _default_sim_fn
except ImportError:                             # repo nested layout
    from src.rewards.seg_rewards import _as_segs, _default_sim_fn

W = re.compile(r"[a-zA-Z]+")
tok = lambda s: [w.lower() for w in W.findall(s)]
ORD = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh",
       "eighth", "ninth", "tenth", "final", "initial", "re"}
STOP = {"the", "a", "an", "and", "or", "to", "of", "into", "onto", "on", "in",
        "with", "from", "for", "at", "by", "up", "down", "out", "off", "over",
        "then", "all", "it", "its", "this", "that", "these", "those", "each",
        "again", "perform", "performs", "performing"}
GEN = {"object", "objects", "item", "items", "thing", "things", "stuff",
       "something", "task", "tasks", "step", "steps", "part", "parts", "area",
       "surface", "material"}
# verb synonym clusters (seeded from GT verb distribution; extend freely)
CLUSTERS = [
    {"open", "unseal", "uncover", "unwrap", "unzip"},
    {"close", "shut", "seal", "cover", "zip"},
    {"remove", "take", "detach", "extract", "pull", "lift", "withdraw"},
    {"put", "place", "set", "return", "store", "replace", "reposition",
     "position", "insert", "mount", "load", "adjust", "align", "reset",
     "arrange", "straighten"},
    {"inspect", "check", "examine", "look", "observe", "view"},
    {"rotate", "turn", "flip", "spin", "rotation"},
    {"tighten", "screw", "fasten", "secure"},
    {"loosen", "unscrew", "undo", "release"},
    {"stack", "pile", "pack", "repack", "gather"},
    {"fold", "bend", "crease"},
    {"grab", "grasp", "pick", "retrieve", "hold", "grip"},
    {"wipe", "clean", "scrub", "wash", "rinse"},
    {"pour", "empty", "dump", "spread", "tip"},
    {"press", "push", "tap", "operate"},
    {"slip", "slide"}, {"tear", "rip"},
]


def cluster_of(v):
    for i, c in enumerate(CLUSTERS):
        if v in c:
            return i
    return -1


def clusters_in(name):
    return {cluster_of(w) for w in tok(name)} - {-1}


def primary_verb(name):
    for w in tok(name):
        if w not in ORD:
            return w
    return ""


def content(name):
    return {w for w in tok(name) if w not in STOP and w not in ORD}


def verb_match(p, g):
    if clusters_in(p) & clusters_in(g):
        return 1.0
    vp, vg = primary_verb(p), primary_verb(g)
    return 1.0 if (vp and vp == vg) else 0.0


def obj_f1(p, g):
    cp, cg = content(p), content(g)
    if not cp or not cg:
        return 0.0
    inter = len(cp & cg)
    if not inter:
        return 0.0
    pr, rc = inter / len(cp), inter / len(cg)
    return 2 * pr * rc / (pr + rc)


def is_generic(name):
    return 1.0 if (GEN & set(tok(name))) else 0.0


NAME_RE = re.compile(r"<name>(.*?)</name>", re.S | re.I)


def build_prompt(gts):
    lines = "\n".join(f"{i+1}. {s:.2f}-{e:.2f}s" for i, (_, s, e) in enumerate(gts))
    return (f"This video shows a person performing {len(gts)} sub-tasks in sequence. "
            f"They occur in these time spans:\n{lines}\n"
            f"For each span, IN ORDER, give a short name = an imperative verb + the "
            f"specific object (e.g. \"Open the jar lid\", \"Stack the bowls\"). Name the "
            f"actual object, do NOT use generic words like 'object' or 'item'. "
            f"Output exactly {len(gts)} lines, one per span, each as:\n"
            f"<seg><name>NAME</name></seg>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--data", default="/workspace/tr1/data_handtask/train_multiseg_val.json")
    ap.add_argument("--out", default="logs/eval_naming.jsonl")
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="0 = size it per recording as 24*n_segments+256. The "
                         "fixed 512 cannot hold 53 names, let alone 147")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4,
                    help="THE REQUIRED OUTPUT FORMAT REPEATS `<seg><name>` on "
                         "every line, which is itself a short n-gram, so a "
                         "non-zero value here can make the format "
                         "unproducible. Left at 4 so old runs reproduce; pass "
                         "0 when you need the format back")
    ap.add_argument("--repetition_penalty", type=float, default=1.3)
    ap.add_argument("--chunk", type=int, default=0,
                    help="ask for N consecutive segments per call instead of "
                         "the whole recording in one. 0 keeps the original "
                         "single-call behaviour. The failure this addresses "
                         "is a function of LIST LENGTH: with the guards on "
                         "the tags mangle, with them off the names cycle, and "
                         "both start within the first handful of a long list")
    ap.add_argument("--device_map", default="cuda",
                    help="'cuda' for single-GPU; 'auto' to shard a big model across GPUs")
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model_base)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_base, dtype=torch.bfloat16, device_map=a.device_map).eval()
    sim = _default_sim_fn()
    rows = json.load(open(a.data))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "w"); agg = []
    def name_block(block, mnt):
        """One generate for one contiguous run of segments."""
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": r["video"],
             "total_pixels": a.total_pixels},
            {"type": "text", "text": build_prompt(block)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
        if isinstance(vkw.get("fps"), list):
            vkw["fps"] = vkw["fps"][0]
        fps_val = vkw.get("fps", 2.0)
        nf = vids[0].shape[0] if hasattr(vids[0], "shape") else len(vids[0])
        vmeta = [{"fps": float(fps_val), "total_num_frames": int(nf),
                  "duration": float(nf) / float(fps_val)}]
        inp = proc(text=[text], images=imgs, videos=vids,
                   video_metadata=vmeta, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=mnt, do_sample=False,
                                 repetition_penalty=a.repetition_penalty,
                                 no_repeat_ngram_size=a.no_repeat_ngram_size)
        return proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]

    t0 = time.time()
    for ri, r in enumerate(rows):
        gts = _as_segs(r["solution"])
        # BEFORE the generate, not after. One call is minutes on a long video,
        # so a line printed only on completion leaves the run looking hung.
        el = time.time() - t0
        eta = (el / ri * (len(rows) - ri)) if ri else 0.0
        print(f"[{ri + 1}/{len(rows)}] {os.path.basename(r['video'])} "
              f"{len(gts)} segments  elapsed {el / 60:.1f}m"
              + (f"  eta {eta / 60:.0f}m" if ri else ""), flush=True)
        step = a.chunk or len(gts)
        pred_names, chunks, out_parts = [], [], []
        for c0 in range(0, len(gts), step):
            block = gts[c0:c0 + step]
            mnt = a.max_new_tokens or (24 * len(block) + 256)
            o = name_block(block, mnt)
            got = [m.strip() for m in NAME_RE.findall(o)]
            # TRUNCATE OR PAD TO THE BLOCK. A block that ran long has looped
            # past its request and the extra names are not segments; a block
            # that ran short leaves holes, and a hole must stay a hole rather
            # than shifting every later segment onto the wrong name.
            chunks.append({"start_index": c0, "asked": len(block),
                           "returned": len(got)})
            got = got[:len(block)] + [""] * max(0, len(block) - len(got))
            pred_names += got
            out_parts.append(o)
        out = "\n<<<CHUNK>>>\n".join(out_parts)
        n_over = sum(1 for c in chunks if c["returned"] > c["asked"])
        n_under = sum(1 for c in chunks if c["returned"] < c["asked"])
        if n_over or n_under:
            print(f"    {len(chunks)} chunk(s): {n_over} over-produced "
                  f"(looped, truncated), {n_under} under-produced (padded)",
                  flush=True)
        pred_names = [m.strip() for m in NAME_RE.findall(out)]
        # A parse failure and a model that said nothing are different problems
        # and both printed as `pred 0`. The raw text is kept on every row so
        # the next failure is diagnosable without a second hour-long run.
        if not pred_names and out.strip():
            print(f"    !! 0 names parsed from {len(out)} chars of output. "
                  f"First 200: {out.strip()[:200]!r}", flush=True)
        gt_names = [g[0] for g in gts]
        k = min(len(pred_names), len(gt_names))
        vm = [verb_match(pred_names[i], gt_names[i]) for i in range(k)]
        of = [obj_f1(pred_names[i], gt_names[i]) for i in range(k)]
        gr = [is_generic(pred_names[i]) for i in range(k)]
        es = [sim(pred_names[i], [gt_names[i]])[0] for i in range(k)]
        m = {"n_gt": len(gt_names), "n_pred": len(pred_names),
             "count_match": 1.0 if len(pred_names) == len(gt_names) else 0.0,
             "verb_acc": statistics.mean(vm) if vm else 0.0,
             "obj_f1": statistics.mean(of) if of else 0.0,
             "generic_rate": statistics.mean(gr) if gr else 0.0,
             "emb_sim": statistics.mean(es) if es else 0.0}
        agg.append(m)
        print("   ", os.path.basename(r["video"]), "gt", m["n_gt"], "pred", m["n_pred"],
              "verb", round(m["verb_acc"], 2), "obj", round(m["obj_f1"], 2),
              "gen", round(m["generic_rate"], 2), "sim", round(m["emb_sim"], 2))
        fout.write(json.dumps({"video": r["video"], **m,
                               "chunks": chunks,
                               "pred_names": pred_names,
                               "gt_names": gt_names, "raw": out}) + "\n")
        fout.flush()
    print("\n==== NAMING (decoupled, n=%d) ====" % len(agg))
    for k in ["count_match", "verb_acc", "obj_f1", "generic_rate", "emb_sim"]:
        print(k.ljust(14), round(statistics.mean([m[k] for m in agg]), 3))


if __name__ == "__main__":
    main()
