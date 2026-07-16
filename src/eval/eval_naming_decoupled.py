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
import argparse, json, os, re, statistics
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
    ap.add_argument("--max_new_tokens", type=int, default=512)
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
    for r in rows:
        gts = _as_segs(r["solution"])
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": r["video"], "total_pixels": a.total_pixels},
            {"type": "text", "text": build_prompt(gts)}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(msgs, return_video_kwargs=True)
        if isinstance(vkw.get("fps"), list):
            vkw["fps"] = vkw["fps"][0]
        fps_val = vkw.get("fps", 2.0)
        nf = vids[0].shape[0] if hasattr(vids[0], "shape") else len(vids[0])
        vmeta = [{"fps": float(fps_val), "total_num_frames": int(nf),
                  "duration": float(nf) / float(fps_val)}]
        inp = proc(text=[text], images=imgs, videos=vids, video_metadata=vmeta,
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            # repetition_penalty/no_repeat_ngram_size: greedy decoding (do_sample=False)
            # on long structured lists (many segments) can degenerate into repeating
            # the same line verbatim until max_new_tokens is hit -- observed directly
            # on a 147-segment recording (36 identical predicted names). This does not
            # fix visual grounding, only stops the decode-time repetition failure mode.
            gen = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.3, no_repeat_ngram_size=4)
        out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        pred_names = [m.strip() for m in NAME_RE.findall(out)]
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
        print(os.path.basename(r["video"]), "gt", m["n_gt"], "pred", m["n_pred"],
              "verb", round(m["verb_acc"], 2), "obj", round(m["obj_f1"], 2),
              "gen", round(m["generic_rate"], 2), "sim", round(m["emb_sim"], 2))
        fout.write(json.dumps({"video": r["video"], **m, "pred_names": pred_names,
                               "gt_names": gt_names, "raw": out}) + "\n")
        fout.flush()
    print("\n==== NAMING (decoupled, n=%d) ====" % len(agg))
    for k in ["count_match", "verb_acc", "obj_f1", "generic_rate", "emb_sim"]:
        print(k.ljust(14), round(statistics.mean([m[k] for m in agg]), 3))


if __name__ == "__main__":
    main()
