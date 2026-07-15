"""Recompute naming metrics from a saved naming-eval jsonl (no VLM re-gen needed),
plus an optional neutral LLM-judge.

The deterministic metric (verb-cluster match + object F1 + genericity + emb sim)
is a pure function of the stored pred/gt names, so metric changes can be re-scored
without re-running the VLM. The judge uses an ON-DISK model of a DIFFERENT family
than the model under test (e.g. Time-R1 / Qwen2.5-VL judging Qwen3 outputs) so it
is not self-judging, and needs no network.

Usage (server):
    python score_names.py --jsonl /tmp/naming_<model>.jsonl \
        --judge_model /workspace/tr1/ckpts/Time-R1-7B
"""
import argparse, json, re, statistics

try:                                            # server flat layout
    from src.seg_rewards import _default_sim_fn
except ImportError:                             # repo nested layout
    from src.rewards.seg_rewards import _default_sim_fn

# keep metric defs in sync with eval_naming_decoupled.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--judge_model", default=None,
                    help="on-disk model of a DIFFERENT family than the model under test")
    a = ap.parse_args()
    sim = _default_sim_fn()
    recs = [json.loads(l) for l in open(a.jsonl)]

    judge = None
    if a.judge_model:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        jp = AutoProcessor.from_pretrained(a.judge_model)
        jm = AutoModelForImageTextToText.from_pretrained(
            a.judge_model, dtype=torch.bfloat16, device_map="cuda").eval()

        def judge(pred, gt):
            q = (f"Two labels for the same short video sub-task. Ignore wording, order "
                 f"and repetition-count details; judge ONLY whether they describe the "
                 f"same ACTION on the same OBJECT.\nReference: {gt}\nPrediction: {pred}\n"
                 f"Answer one word, YES or NO:")
            msgs = [{"role": "user", "content": [{"type": "text", "text": q}]}]
            t = jp.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = jp(text=[t], return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = jm.generate(**inp, max_new_tokens=5, do_sample=False)
            o = jp.batch_decode(g[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].lower()
            return 1.0 if "yes" in o else 0.0

    agg = []
    for rec in recs:
        pn, gn = rec["pred_names"], rec["gt_names"]
        k = min(len(pn), len(gn))
        m = {"verb_acc": statistics.mean([verb_match(pn[i], gn[i]) for i in range(k)]) if k else 0,
             "obj_f1": statistics.mean([obj_f1(pn[i], gn[i]) for i in range(k)]) if k else 0,
             "generic_rate": statistics.mean([is_generic(pn[i]) for i in range(k)]) if k else 0,
             "emb_sim": statistics.mean([sim(pn[i], [gn[i]])[0] for i in range(k)]) if k else 0}
        if judge:
            m["judge_acc"] = statistics.mean([judge(pn[i], gn[i]) for i in range(k)]) if k else 0
        agg.append(m)
    print("==== %s (n=%d) ====" % (a.jsonl.split("/")[-1], len(agg)))
    keys = ["verb_acc", "obj_f1", "generic_rate", "emb_sim"] + (["judge_acc"] if judge else [])
    for kk in keys:
        print(kk.ljust(13), round(statistics.mean([m[kk] for m in agg]), 3))


if __name__ == "__main__":
    main()
