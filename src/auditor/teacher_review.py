"""A VLM second opinion on the candidates the student wants to admit.

The student's last safety gap is narrow and specific. v4 proposes 74 events
for automatic admission; 58 are real sharp transitions and 16 are not. To
reach 0.95 precision with one error of buffer, a reviewer must challenge 14 of
those 16 while keeping most of the 58. That is a demanding target, not "better
than the student on average", and this file exists to measure whether it is
met rather than to assume it.

TWO PASSES, AND THE FIRST ONE IS BLIND.

  Pass 1  the model sees frames and the annotation guideline. It does NOT see
          the student's decision, any score, the policy's verdict, the gold
          label, the taxonomy subtype, the event id, or the segment labels --
          those name the actions on either side and would hand over the
          answer. It reports structured observations and an independent call.

  Pass 2  only for events that might be admitted. Now it sees the student's
          decision and its own pass-1 output, and is asked to find the
          STRONGEST case against admission before deciding. The prompt never
          asks it to explain why the student is right; that phrasing produces
          agreement rather than review.

Calling one model twice is NOT independent cross-validation and is not
described as such anywhere in the output. It is a consistency-and-challenge
check: pass 2 can catch what pass 1 asserted without evidence, and the two
disagreeing is itself a routing signal.

THE MODEL IS NOT ASKED FOR THE SEVEN-WAY TAXONOMY. `annotation_convention`
means "the dataset's rule cut here and nothing visible happened", which is not
a visual fact and cannot be read off frames -- demanding it invites invention.
The question is narrower and answerable: is there sufficient, clearly visible
evidence of a sharp transition? Negative reasons are collected separately.

FRAMES ARE SENT INLINE AS BASE64. Uploading them to a reachable URL would put
the user's recordings somewhere they were never meant to be, for no benefit.

THE API KEY IS READ FROM ARK_API_KEY AND NEVER WRITTEN ANYWHERE. It is not
logged, not echoed into results, not stored in the output JSON, and not
accepted as a command-line argument -- an argument would land in shell history
and in the process list.

Usage:
    export ARK_API_KEY=...        # never committed, never passed as a flag
    python -m src.auditor.teacher_review \
        --decisions .../policy_decisions_v4.primary_transportability_frontier.csv \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --data .../recseg_train.json --data .../recseg_val.json \
        --n_per_class 16 \
        --out /workspace/tr1/results/hal/c3/teacher_review.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np

from src.boundary.pair_taxonomy import load_pair_labels

SHARP = "sharp_visible_transition"
MODEL = "doubao-seed-2-1-turbo-260628"

GUIDELINE = """You are reviewing a candidate moment in an egocentric (head-mounted camera) video of a person doing a manual task.

A SHARP VISIBLE TRANSITION means one action ends and a different one begins, and there is a moment you could point to. Typical visible evidence:
- the hand releases an object and reaches for a DIFFERENT object
- contact with an object ends and later restarts after a hand-free interval
- the target of the manipulation changes
- a discrete state change happens (a lid closes, an item is dropped in a bin)

It is NOT a sharp visible transition when:
- the same object is held continuously and only the grip, direction or posture changes (repetition, regrasp, reversal)
- something really does change but gradually, with no instant you could point to
- the dominant change is the CAMERA or the head moving, not the hands
- the critical moment is off-frame, occluded, or otherwise not visible
- nothing visible happens at all at this moment

The candidate moment is the MIDDLE of the clip. Frames are given in time order."""

BLIND_SCHEMA = """Reply with ONLY a JSON object, no prose around it:
{
  "evidence_sufficient": true|false,
  "hand_visibility": "clear"|"partial"|"absent",
  "active_object_visibility": "clear"|"partial"|"absent",
  "same_object_before_after": "yes"|"no"|"unknown",
  "contact_transition": "none"|"release"|"recontact"|"release_and_recontact",
  "object_switch_observed": true|false,
  "target_switch_observed": true|false,
  "discrete_state_change": true|false,
  "camera_motion_dominant": true|false,
  "transition_type": "sharp"|"gradual"|"continuous"|"not_observable",
  "blind_decision": "approve_visible_sharp"|"reject_visible_sharp"|"not_observable"|"uncertain",
  "negative_reason": "same_action_continuous"|"gradual_change"|"camera_dominant"|"visibility_insufficient"|"no_discrete_change"|"other"|null,
  "evidence": ["one short sentence naming what you actually SAW, per item"]
}
Every entry in "evidence" must describe something visible in these frames. Do not infer what probably happened off-screen."""

CHALLENGE_SCHEMA = """An automatic system has PROPOSED admitting this candidate as a real sharp transition, without further human review. Your job is to look for the strongest reason that proposal is WRONG, and only then decide.

Consider specifically whether this is instead:
- same-action internal motion (same object held throughout)
- a gradual change with no single instant
- camera or viewpoint motion rather than a hand-object change
- off-frame or occluded at the critical moment
- a cut with no visible discrete change at all

Reply with ONLY a JSON object:
{
  "strongest_counterevidence": ["the best case against admission, one item per line"],
  "counterevidence_is_decisive": true|false,
  "final_review": "approve"|"challenge"|"abstain",
  "approve_for_direct_admission": true|false
}"""


def event_time(eid):
    m = re.search(r"_t(\d+(?:\.\d+)?)$", eid)
    return float(m.group(1)) if m else None


def eye_slice(frames, eye="left"):
    ax = 2 if frames.ndim == 4 else 1
    W = frames.shape[ax]
    sl = slice(0, W // 2) if eye == "left" else slice(W // 2, W)
    return frames[:, :, sl] if frames.ndim == 4 else frames[:, sl]


def frames_b64(video, t, half_window, n, eye="left", long_side=640):
    """n frames centred on t, as inline base64 JPEGs. Inline rather than
    uploaded: putting the recordings on a reachable URL would move the user's
    data somewhere it was never meant to go, and buys nothing."""
    from decord import VideoReader
    from PIL import Image
    vr = VideoReader(video, num_threads=1)
    fps, total = vr.get_avg_fps(), len(vr)
    want = t + np.linspace(-half_window, half_window, n)
    idx = np.clip(np.round(want * fps).astype(int), 0, total - 1)
    arr = eye_slice(vr.get_batch(idx.tolist()).asnumpy(), eye)
    out = []
    for k in range(arr.shape[0]):
        im = Image.fromarray(np.ascontiguousarray(arr[k]))
        if max(im.size) > long_side:
            s = long_side / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        out.append("data:image/jpeg;base64,"
                   + base64.b64encode(buf.getvalue()).decode())
    return out, [float(x - t) for x in want]


def parse_json(text):
    """The model is asked for bare JSON; a fenced block or leading prose is
    still salvageable, and silently returning None for those would be reported
    as the model failing rather than as parsing failing."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def call(client, images, rel_t, instruction, temperature=0.0):
    content = []
    for url, dt in zip(images, rel_t):
        content.append({"type": "text", "text": f"[t{dt:+.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instruction})
    r = client.chat.completions.create(
        model=MODEL, temperature=temperature,
        messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--n_per_class", type=int, default=16)
    ap.add_argument("--wide_s", type=float, default=3.0)
    ap.add_argument("--narrow_s", type=float, default=1.5)
    ap.add_argument("--n_wide", type=int, default=7)
    ap.add_argument("--n_narrow", type=int, default=5)
    ap.add_argument("--eye", default="left")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="build the sample and the prompts, call nothing")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = os.environ.get("ARK_API_KEY")
    if not key and not a.dry_run:
        raise SystemExit(
            "ARK_API_KEY is not set. It is read from the environment only -- "
            "never a flag, which would land in shell history and in the "
            "process list, and never a file.")

    labels = {}
    for p in a.pair_labels:
        for e, v in load_pair_labels(p).items():
            labels[e] = v["temporal_pair_subtype"]
    video = {}
    for p in a.data:
        for r in json.load(open(p, encoding="utf-8")):
            video[r["recording_id"]] = r["video"]

    keeps = []
    with open(a.decisions, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("decision") != "AUTO_KEEP":
                continue
            sub = labels.get(r["event_id"])
            if sub is None:
                continue
            keeps.append({"event_id": r["event_id"],
                          "recording_id": r["recording_id"],
                          "subtype": sub, "correct": sub == SHARP,
                          "score": r.get("score") or r.get("fused_score") or ""})
    good = [k for k in keeps if k["correct"]]
    bad = [k for k in keeps if not k["correct"]]
    print(f"student proposes {len(keeps)} admissions: {len(good)} correct, "
          f"{len(bad)} wrong")
    print(f"  the wrong ones: {dict(Counter(k['subtype'] for k in bad))}")
    if not bad:
        raise SystemExit("no false keeps in this decisions file -- nothing for "
                         "a reviewer to catch")
    n = min(a.n_per_class, len(bad))
    print(f"\n  To reach 0.95 with one error of buffer on all {len(keeps)} "
          f"proposals, a reviewer must challenge "
          f"{max(0, len(bad) - max(0, int(len(good) / 0.95) - len(good) - 1))} "
          f"of {len(bad)} while keeping most of the {len(good)}. This pilot "
          f"measures both rates on {n} + {n}.")

    # matched controls: the true keeps nearest in score, so a reviewer cannot
    # succeed by exploiting an obvious score gap between the classes
    rng = random.Random(a.seed)
    picked_bad = bad if len(bad) <= n else rng.sample(bad, n)
    fnum = lambda x: float(x) if x not in ("", None) else float("nan")
    pool = sorted(good, key=lambda g: fnum(g["score"]))
    picked_good, used = [], set()
    for b in picked_bad:
        cand = [g for g in pool if g["event_id"] not in used]
        if not cand:
            break
        j = int(np.argmin([abs(fnum(g["score"]) - fnum(b["score"]))
                           if np.isfinite(fnum(g["score"])) else 1e9
                           for g in cand]))
        picked_good.append(cand[j])
        used.add(cand[j]["event_id"])
    sample = [dict(x, arm="false_keep") for x in picked_bad] + \
             [dict(x, arm="true_keep") for x in picked_good]
    rng.shuffle(sample)
    print(f"  sampled {len(picked_bad)} false keeps + {len(picked_good)} "
          f"score-matched true keeps, shuffled")

    if a.dry_run:
        print("\n--dry_run: no API call made. Prompt sizes:")
        print(f"  guideline {len(GUIDELINE)} chars, blind schema "
              f"{len(BLIND_SCHEMA)}, challenge schema {len(CHALLENGE_SCHEMA)}")
        for s in sample[:3]:
            print(f"  would review {s['event_id']}  arm={s['arm']}  "
                  f"(subtype withheld from the prompt)")
        return

    from volcenginesdkarkruntime import Ark
    client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3",
                 api_key=key)

    results = []
    for i, s in enumerate(sample):
        vp = video.get(s["recording_id"])
        t = event_time(s["event_id"])
        if vp is None or t is None:
            print(f"  !! {s['event_id']}: no video or no timestamp, skipped")
            continue
        wide, wt = frames_b64(vp, t, a.wide_s, a.n_wide, a.eye)
        narrow, nt = frames_b64(vp, t, a.narrow_s, a.n_narrow, a.eye)
        imgs, rel = wide + narrow, wt + nt
        try:
            b_raw = call(client, imgs, rel, GUIDELINE + "\n\n" + BLIND_SCHEMA)
        except Exception as ex:
            print(f"  !! {s['event_id']}: pass 1 {type(ex).__name__}: "
                  f"{str(ex)[:120]}")
            continue
        blind = parse_json(b_raw)
        rec = {**s, "blind": blind,
               "blind_unparsed": None if blind else (b_raw or "")[:400]}
        # pass 2 sees the student's proposal and pass 1, and is asked for the
        # case AGAINST admission first
        ctx = (f"\n\nThe automatic system's own confidence score for this "
               f"candidate is {s['score'] or 'unavailable'}.\n"
               f"Your own first-pass observations were:\n"
               f"{json.dumps(blind, ensure_ascii=False) if blind else '(unparsed)'}")
        try:
            c_raw = call(client, imgs, rel,
                         GUIDELINE + ctx + "\n\n" + CHALLENGE_SCHEMA)
        except Exception as ex:
            print(f"  !! {s['event_id']}: pass 2 {type(ex).__name__}: "
                  f"{str(ex)[:120]}")
            c_raw = None
        chal = parse_json(c_raw)
        rec["challenge"] = chal
        rec["challenge_unparsed"] = None if chal else (c_raw or "")[:400]
        results.append(rec)
        bd = (blind or {}).get("blind_decision", "?")
        fr = (chal or {}).get("final_review", "?")
        print(f"  [{i+1}/{len(sample)}] {s['event_id'][:44]:<44} "
              f"{s['arm']:<11} blind={bd:<24} review={fr}", flush=True)

    # ---------------------------------------------------------------- report
    def arm(x):
        return [r for r in results if r["arm"] == x]

    fk, tk = arm("false_keep"), arm("true_keep")
    ch = lambda r: (r.get("challenge") or {}).get("final_review")
    n_ch_bad = sum(1 for r in fk if ch(r) in ("challenge", "abstain"))
    n_ap_good = sum(1 for r in tk if ch(r) == "approve")
    print(f"\n{'=' * 72}\nPRE-REGISTERED CRITERIA\n{'=' * 72}")
    c1 = len(fk) and n_ch_bad >= 12
    c2 = len(tk) and n_ap_good >= 12
    print(f"  {'PASS' if c1 else 'FAIL'}  challenge >=12 of the false keeps -- "
          f"{n_ch_bad}/{len(fk)}")
    print(f"  {'PASS' if c2 else 'FAIL'}  approve >=12 of the true keeps -- "
          f"{n_ap_good}/{len(tk)}")
    print("  Both are needed. A reviewer that challenges everything passes the "
          "first and is useless; one that approves everything passes the "
          "second and is the\n  student again.")

    print(f"\n  observability discipline (an answer can be right for an "
          f"invented reason):")
    inv = [r for r in results
           if (r.get("blind") or {}).get("camera_motion_dominant") is False
           and (r.get("blind") or {}).get("contact_transition") in
           ("release", "recontact", "release_and_recontact")
           and r["subtype"] in ("same_action_internal_motion",
                                "camera_or_viewpoint_shift")]
    print(f"    reports a release/recontact on a same-action or camera event: "
           f"{len(inv)}")
    off = [r for r in results if r["subtype"] == "visibility_or_offscreen"]
    off_ab = [r for r in off
              if (r.get("blind") or {}).get("blind_decision") == "not_observable"
              or ch(r) == "abstain"]
    print(f"    abstains on offscreen events: {len(off_ab)}/{len(off)}")
    bad_parse = sum(1 for r in results
                    if r["blind_unparsed"] or r["challenge_unparsed"])
    if bad_parse:
        print(f"    !! {bad_parse} response(s) did not parse as JSON; they are "
              f"kept verbatim in the output and counted as neither")

    print(f"\n  blind vs challenge agreement: "
          f"{sum(1 for r in results if (r.get('blind') or {}).get('blind_decision') == 'approve_visible_sharp') } "
          f"blind approvals, "
          f"{sum(1 for r in results if ch(r) == 'approve')} after challenge")
    print("  Two calls to ONE model are not independent cross-validation. This "
          "is a consistency-and-challenge check, and the pair disagreeing is "
          "itself a routing signal.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    blob = {"model": MODEL, "n": len(results),
            "criteria": {"challenge_false_keeps": [n_ch_bad, len(fk)],
                         "approve_true_keeps": [n_ap_good, len(tk)]},
            "results": results}
    txt = json.dumps(blob, ensure_ascii=False, indent=2, default=str)
    if key and key in txt:
        raise SystemExit("refusing to write: the API key appears in the output")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
