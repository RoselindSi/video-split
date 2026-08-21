"""Is the 3s window the handicap? A DIAGNOSTIC, and nothing downstream of one.

Driver A returned 0.641 [0.550, 0.726] and did not clear its capability gate.
That was measured on 3s halves of a 6s clip, and G0 has already shown a short
window can lose a signal that is present. The open question is whether more
context recovers it -- which decides whether rebuilding batch4's exact segments
is worth being first in the queue when the packed machine returns.

WHAT THIS MAY NEVER DO, stated first because the temptation is structural:

    it is NOT driver A          different events, different gold, no
                                within-recording pair structure
    it does NOT touch the       the capability gate is 0.55 on driver A's
    capability gate             statistic and this is not that statistic
    it may NOT back a           it emits kind: diagnostic, which
    semantic certificate        verify_semantic_certificate refuses
    it changes NO product       AUTO_KEEP stays off, AUTO_ACCEPT stays off,
    state whatever it says      auditor_v1 keeps routing everything to REVIEW

The single decision it can move: whether the exact full-segment driver A is
first priority when full recordings are available again.

THE STATISTIC, frozen before the run:

    per event      delta = score(long span) - score(short 3s window)
    primary        D = mean(delta | YES) - mean(delta | NO)
    interval       bootstrap over RECORDINGS, not over events

Each event is its own control, so scene, person, camera and label are identical
across the two arms and cancel in `delta`. What does not cancel is a systematic
difference in how window length affects different recordings, which is why the
interval resamples recordings.

THE LONG ARM IS A PROXY WITH NEIGHBOUR CONTAMINATION. The span runs from the
start of the previous segment to the end of the next, so it carries material
the label does not describe. The usual expectation is that this hurts a YES
more than a NO, which would make the test conservative -- but that is an
expectation, not a guarantee, since a neighbouring segment can happen to
support the label too. Read it as a proxy whose bias is uncertain in size and
probably conservative in direction, not as a bound.

THE SHORT ARM IS THE CENTRE 3s OF THE SPAN. The candidate's position inside
the span is not recoverable -- the renderer wrote no manifest and the gold
carries no segment times -- and the span is prev+containing+next, so its
geometric centre falls inside the containing segment unless the segments are
very unequal. That is an approximation and it is the reason this is a
diagnostic.

Usage:
    python -m src.auditor.semantic.window_effect --emit \
        --events /tmp/window_events.json --out obs.jsonl
    python -m src.auditor.semantic.window_effect --score \
        --observations obs.jsonl --model ckpts/reranker-8b --out scores.jsonl
    python -m src.auditor.semantic.window_effect --evaluate \
        --observations obs.jsonl --scores scores.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time

import numpy as np

SHORT_S = 3.0     # driver A's sub-window length, so the arms differ only in it
N_FRAMES = 32


def duration(video):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video], capture_output=True)
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        return None


def emit(a):
    ev = json.load(open(a.events, encoding="utf-8"))
    if a.media_root:
        for d in ev.values():
            hit = glob.glob(os.path.join(a.media_root, "**",
                                         os.path.basename(d["video"])),
                            recursive=True)
            if not hit:
                raise SystemExit(f"{os.path.basename(d['video'])} not under "
                                 f"{a.media_root}")
            d["video"] = hit[0]
    obs, skip = [], 0
    for eid, d in sorted(ev.items()):
        dur = duration(d["video"])
        if not dur or dur <= SHORT_S:
            skip += 1
            continue
        mid = dur / 2.0
        rid = re.match(r"(recording_\d+)", eid).group(1)
        for arm, s0, s1 in (("short", mid - SHORT_S / 2, mid + SHORT_S / 2),
                            ("long", 0.0, dur)):
            obs.append({"obs_id": f"{eid}#{arm}", "event_id": eid,
                        "recording_id": rid, "arm": arm,
                        "support": d["support"], "label": d["label"],
                        "video": d["video"], "start": s0, "end": s1,
                        "span_duration": dur})
    with open(a.out, "w", encoding="utf-8") as f:
        for o in obs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n_ev = len({o["event_id"] for o in obs})
    print(f"{n_ev} events x 2 arms = {len(obs)} observations"
          + (f"  ({skip} skipped: span shorter than {SHORT_S}s)" if skip else ""))
    import collections
    c = collections.Counter(o["support"] for o in obs if o["arm"] == "long")
    print(f"  {dict(c)} over "
          f"{len({o['recording_id'] for o in obs})} recordings")
    sp = [o["span_duration"] for o in obs if o["arm"] == "long"]
    print(f"  span length: median {np.median(sp):.0f}s, "
          f"range {min(sp):.0f}-{max(sp):.0f}s  vs short arm {SHORT_S}s")
    print(f"\nwrote {a.out}")


def score(a):
    from sentence_transformers import CrossEncoder
    from transformers import AutoProcessor

    from src.auditor.semantic.cosine_baseline import sample_times, write_frames
    from src.auditor.semantic.reranker_baseline import score_batch

    obs = [json.loads(l) for l in open(a.observations, encoding="utf-8")
           if l.strip()]
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {json.loads(l)["obs_id"] for l in open(a.out) if l.strip()}
        obs = [o for o in obs if o["obs_id"] not in done]
    print(f"{len(obs)} observations to score"
          + (f"  ({len(done)} already done)" if done else ""))

    model = CrossEncoder(a.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(a.model)
    os.makedirs(a.frame_dir, exist_ok=True)
    fout = open(a.out, "a" if done else "w", encoding="utf-8")
    t0 = time.time()
    for i, o in enumerate(obs):
        frames = write_frames(o["video"],
                              sample_times(o["start"], o["end"], a.n_frames),
                              a.frame_dir, o["obs_id"].replace("/", "_"))
        dur = max(float(o["end"]) - float(o["start"]), 1e-3)
        nf = len(frames)
        meta = {"fps": (nf - 1) / dur if nf > 1 else 1.0,
                "total_num_frames": nf, "duration": dur,
                "frames_indices": np.arange(nf)}
        s = score_batch(model, proc, "sentence_transformers", frames,
                        [o["label"]], a.total_pixels, None, meta)[0]
        import src.auditor.semantic.cosine_baseline as _cb
        fout.write(json.dumps({"obs_id": o["obs_id"], "score": float(s),
                               "frame_extractor": _cb.EXTRACTOR}) + "\n")
        fout.flush()
        for q in frames:
            os.remove(q)
        n = i + 1
        if n == 1 or n % 25 == 0 or n == len(obs):
            el = time.time() - t0
            print(f"    {n}/{len(obs)}  {el / n:.1f}s/obs  "
                  f"eta {(el / n) * (len(obs) - n) / 60:.1f}m", flush=True)
    fout.close()
    print(f"\nwrote {a.out}")


def evaluate(a):
    obs = [json.loads(l) for l in open(a.observations, encoding="utf-8")
           if l.strip()]
    sc = {json.loads(l)["obs_id"]: json.loads(l)["score"]
          for l in open(a.scores, encoding="utf-8") if l.strip()}

    by = {}
    for o in obs:
        if o["obs_id"] in sc:
            by.setdefault(o["event_id"], {})[o["arm"]] = (
                sc[o["obs_id"]], o["support"], o["recording_id"])
    full = {k: v for k, v in by.items() if "short" in v and "long" in v}
    print(f"{len(full)} events scored in BOTH arms "
          f"(of {len(by)} with any score)")

    rows = [(v["long"][2], v["long"][1], v["long"][0] - v["short"][0])
            for v in full.values()]
    yes = [d for _, s, d in rows if s == "yes"]
    no = [d for _, s, d in rows if s == "no"]
    if not yes or not no:
        raise SystemExit("need both classes")
    D = float(np.mean(yes) - np.mean(no))

    print(f"\n  delta = score(long span) - score(short {SHORT_S}s window)")
    print(f"    YES  n={len(yes):<4} mean delta {np.mean(yes):+.4f}")
    print(f"    NO   n={len(no):<4} mean delta {np.mean(no):+.4f}")

    recs = sorted({r for r, _, _ in rows})
    idx = {r: [(s, d) for rr, s, d in rows if rr == r] for r in recs}
    rng = np.random.default_rng(a.seed)
    boot = []
    for _ in range(a.n_boot):
        take = rng.choice(len(recs), len(recs), replace=True)
        v = [x for i in take for x in idx[recs[i]]]
        y = [d for s, d in v if s == "yes"]
        n = [d for s, d in v if s == "no"]
        if y and n:
            boot.append(float(np.mean(y) - np.mean(n)))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n  D = mean(delta|YES) - mean(delta|NO)   {D:+.4f}  "
          f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"    {len(recs)} recordings resampled; the interval is over "
          f"RECORDINGS, not events.")

    # THE THREE READINGS, frozen before the run. Printed rather than left to
    # whoever reads the number, because the second one is the one a reader
    # under time pressure will convert into the third.
    print()
    if lo > 0:
        print("  READING: CI lower bound above 0. There is evidence that more "
              "context improves\n  YES/NO separation, and the long arm carries "
              "neighbour contamination while\n  showing it. Rebuilding "
              "batch4's exact segments and running the real\n  full-segment "
              "driver A is FIRST PRIORITY when full recordings return.")
    elif hi < 0:
        print("  READING: CI upper bound below 0. This three-segment span is "
              "clearly worse than\n  the short window. Because the span is "
              "contaminated by neighbouring segments,\n  this does NOT support "
              "'an exact full segment would be worse' -- it says this\n  proxy "
              "is worse, and the proxy is not the thing.")
    else:
        print("  READING: INCONCLUSIVE. The interval spans 0. This is not "
              "evidence that a longer\n  window fails to help, and it unlocks "
              "nothing. A formal full-segment driver A\n  is still required "
              "once full recordings exist.")
    print("\n  NOTHING HERE CHANGES PRODUCT STATE. AUTO_KEEP off, AUTO_ACCEPT "
          "off, auditor_v1\n  routes everything to REVIEW. This run emits "
          "kind: diagnostic, which\n  verify_semantic_certificate refuses as "
          "backing for any threshold.")

    if a.out:
        json.dump({"kind": "diagnostic", "not_a_capability_gate": True,
                   "may_back_a_threshold": False,
                   "D": D, "lo": float(lo), "hi": float(hi),
                   "n_events": len(full), "n_recordings": len(recs),
                   "mean_delta_yes": float(np.mean(yes)),
                   "mean_delta_no": float(np.mean(no)),
                   "short_window_s": SHORT_S},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


def power(a):
    """Would this design's interval even be honest? Run BEFORE spending a GPU.

    NOT a power check in the usual sense. The number that disqualified the
    2026-08-21 attempt was not low power -- low power only yields
    "inconclusive", which is harmless. It was the FALSE POSITIVE RATE: with all
    five NO events sitting in two recordings, a recording-clustered bootstrap
    repeatedly resamples the same cluster, the variance estimate degenerates,
    and a nominally 5% test rejected 13.5% of the time under a true D of zero
    -- biased toward the answer that would reprioritise real work.

    So this runs first, and the caller is expected to abandon the measurement
    rather than reinterpret it when the null rejection rate comes back wrong."""
    ev = json.load(open(a.events, encoding="utf-8"))
    rows = [(re.match(r"(recording_\d+)", k).group(1), v["support"])
            for k, v in ev.items()]
    recs = sorted({r for r, _ in rows})
    no_recs = {r for r, s in rows if s == "no"}
    print(f"{len(rows)} events over {len(recs)} recordings")
    print(f"  NO events {sum(1 for _, s in rows if s == 'no')} in "
          f"{len(no_recs)} recording(s)")
    if len(no_recs) < 5:
        print(f"  !! the whole NO side lives in {len(no_recs)} cluster(s); "
              f"a recording-clustered\n     interval cannot be trusted here "
              f"whatever the simulation says")

    rng = np.random.default_rng(a.seed)
    keys = list(ev)
    idx = {r: [i for i, (rr, _) in enumerate(rows) if rr == r] for r in recs}
    sup = [s for _, s in rows]
    print(f"\n  {'true D':>8}{'rejects':>10}{'median CI width':>18}")
    for g in (0.0, 0.5, 1.0, 1.5):
        win, wid = [], []
        for _ in range(a.n_sim):
            d = np.array([rng.normal(g if s == "yes" else 0.0, 1.0)
                          for s in sup])
            boot = []
            for _ in range(400):
                take = rng.choice(len(recs), len(recs), replace=True)
                sel = [i for j in take for i in idx[recs[j]]]
                y = [d[i] for i in sel if sup[i] == "yes"]
                n = [d[i] for i in sel if sup[i] == "no"]
                if y and n:
                    boot.append(np.mean(y) - np.mean(n))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            win.append(lo > 0)
            wid.append(hi - lo)
        tag = "   <- false positive rate, nominal 5%" if g == 0.0 else ""
        print(f"  {g:>8.1f}{np.mean(win):>9.1%}{np.median(wid):>18.2f}{tag}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--power", action="store_true",
                    help="simulate this design's false positive rate before "
                         "any GPU is spent")
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--events",
                    default="data/gold/window_effect_events_35.json")
    ap.add_argument("--media_root",
                    help="rewrite each event's video path to this directory, "
                         "for running where the media was copied to")
    ap.add_argument("--observations")
    ap.add_argument("--scores")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--n_frames", type=int, default=N_FRAMES)
    ap.add_argument("--total_pixels", type=int, default=3584 * 28 * 28)
    ap.add_argument("--frame_dir", default="/tmp/window_frames")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.power:
        return power(a)
    if a.emit:
        return emit(a)
    if a.score:
        return score(a)
    if a.evaluate:
        return evaluate(a)
    ap.print_help()


if __name__ == "__main__":
    main()
