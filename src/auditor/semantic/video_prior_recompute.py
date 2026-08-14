"""The video-only prior, on the SAME YES/NO events the naming features got.

WHY IT HAD TO BE RECOMPUTED. The 0.601 [0.495, 0.706] quoted next to the
naming features all session came from 186 events with the target `correct`
against everything else. These are the `yes` against `no` events of the
89-event gold. Different population, different label, different n -- so putting a naming
feature "above the video prior" was never a valid comparison, and the
diagnostic now says so rather than inviting it. Making it valid means running
the prior here, on this contrast, with the same folds and the same bar.

WHAT IT IS AND IS NOT. This head reads frozen visual features and never sees a
label, so it cannot verify one. What it bounds is how much of `claim_support`
is predictable from the scene alone -- which segments tend to carry bad labels
regardless of what the label says. If it lands where the naming features land,
neither arm has shown anything and the honest reading is that nothing in this
feature family separates the contrast. If it lands clearly higher, then the
scene predicts the status better than any label-reading signal does, which
would be an awkward and useful thing to know.

SAME EVERYTHING AS THE NAMING TABLE, on purpose: the same YES/NO events, the
same recording-grouped folds, the same recording-clustered bootstrap, and the
same random-scorer bar recomputed at this n. The point is a number that can be
put in the same table, and anything tuned here that was not tuned there would
break that.

OUT-OF-FOLD, because a head fitted and scored on the same events reports its
own memorisation. Folds group by recording, so no recording contributes to the
head that scores it.

Usage:
    python -m src.auditor.semantic.video_prior_recompute \
        --gold data/gold/semantic_ontology_gold_48.json \
        --gold data/gold/semantic_enrichment_gold_41.csv \
        --feat_cache ... --local_cache ...
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src.auditor.common.feature_loader import load_caches, build_events, stack
from src.auditor.boundary.model import build_input
from src.auditor.boundary.relation_experiment import RelationHead, pca_fit, proj
from src.auditor.semantic.claim_support_diagnostic import (
    auroc, grouped_boot, min_detectable)
from src.boundary.pairwise_verifier import stratified_grouped_folds

TIME = re.compile(r"_t(\d+(?:\.\d+)?)$")


def load_gold(paths):
    rows = []
    for p in paths:
        if p.lower().endswith(".csv"):
            with open(p, newline="", encoding="utf-8-sig") as f:
                rows += [r for r in csv.DictReader(f)
                         if (r.get("claim_support") or "").strip()]
        else:
            rows += json.load(open(p, encoding="utf-8-sig"))["events"]
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--feat_cache", action="append", required=True)
    ap.add_argument("--local_cache", action="append", required=True)
    ap.add_argument("--half_s", type=float, default=6.0)
    ap.add_argument("--n_frames", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = load_gold(a.gold)
    print(f"{len(rows)} audited events: "
          f"{dict(Counter(r['claim_support'] for r in rows).most_common())}")
    keep = [r for r in rows if r["claim_support"] in ("yes", "no")]
    src = []
    for r in keep:
        eid = r.get("event_id") or ""
        m = TIME.search(eid)
        rid = (r.get("recording_id")
               or (re.match(r"^(recording_\d+)", eid).group(1) if eid else ""))
        src.append({"event_id": eid or r.get("audit_key"),
                    "recording_id": rid,
                    "candidate_time": float(r["candidate_time"])
                    if r.get("candidate_time") else
                    (float(m.group(1)) if m else None),
                    "y": 1 if r["claim_support"] == "yes" else 0})
    src = [s for s in src if s["candidate_time"] is not None
           and s["recording_id"]]
    print(f"  {len(src)} YES/NO events with a time and a recording")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gcache, lcache = load_caches(a.feat_cache), load_caches(a.local_cache)
    ev = build_events(src, gcache, lcache, a.half_s, a.n_frames)
    ylab = {s["event_id"]: s["y"] for s in src}
    y = np.array([ylab[e["event_id"]] for e in ev])
    print(f"  {len(ev)} kept after feature loading: "
          f"{int(y.sum())} YES vs {int((1 - y).sum())} NO over "
          f"{len({e['recording_id'] for e in ev})} recordings")
    if int(y.sum()) < 2 or int((1 - y).sum()) < 2:
        raise SystemExit("not enough of one class after feature loading")

    # Mirrors relation_experiment.train_oof rather than reimplementing it:
    # stack takes a key, pca_fit returns a tuple that proj consumes,
    # build_input returns (X, M), and stratified_grouped_folds yields GROUPS
    # per fold, not index pairs. I got all four wrong writing this from
    # memory, and the first server run found them one at a time.
    G, L = stack(ev, "g"), stack(ev, "l")
    VG = torch.from_numpy(stack(ev, "valid_g"))
    VL = torch.from_numpy(stack(ev, "valid_l"))
    vg_np, vl_np = stack(ev, "valid_g"), stack(ev, "valid_l")
    groups = [e["recording_id"] for e in ev]

    bar = min_detectable(int(y.sum()), int((1 - y).sum()), a.n_boot, a.seed)
    print(f"\n  A RANDOM scorer reaches AUROC {bar:.3f} at the 97.5th "
          f"percentile with\n  {int(y.sum())} vs {int((1 - y).sum())} -- the "
          f"SAME bar the naming features were read against.")

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    per_seed, oof_all = [], np.zeros(len(ev))
    for sd in seeds:
        torch.manual_seed(sd)
        oof = np.full(len(ev), np.nan)
        for fi, f in enumerate(stratified_grouped_folds(groups, y, a.n_folds,
                                                        seed=a.fold_seed)):
            te = np.array([g in f for g in groups])
            tr = ~te
            if te.sum() < 2 or tr.sum() < 20 \
                    or len(set(y[tr].tolist())) < 2:
                print(f"    seed {sd} fold {fi}: too small or single-class, "
                      f"skipped")
                continue
            pg = pca_fit(G[tr][vg_np[tr]], a.pca_dim, sd)
            pl = pca_fit(L[tr][vl_np[tr]], a.pca_dim, sd)
            Pg = torch.from_numpy(proj(pg, G)).float()
            Pl = torch.from_numpy(proj(pl, L)).float()
            for P in (Pg, Pl):
                s_ = P[torch.from_numpy(tr)].reshape(-1,
                                                     P.shape[-1]).std(0)
                P /= s_.clamp(min=1e-6)
            X, M = build_input(Pg, Pl, VG, VL)
            model = RelationHead(X.shape[-1], a.hidden, a.dropout)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                    weight_decay=a.weight_decay)
            yt = torch.from_numpy(y).long()
            trt = torch.from_numpy(tr)
            cnt = np.bincount(y[tr].astype(int), minlength=2) + 1
            w = torch.tensor((cnt.sum() / cnt) / (cnt.sum() / cnt).mean(),
                             dtype=torch.float32)
            model.train()
            for _ in range(a.epochs):
                opt.zero_grad()
                loss = F.cross_entropy(model(X, M)[trt], yt[trt], weight=w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                oof[te] = F.softmax(model(X, M)[torch.from_numpy(te)],
                                    -1)[:, 1].numpy()
        ok = ~np.isnan(oof)
        au = auroc(oof[ok].tolist(), y[ok].tolist())
        per_seed.append(au)
        oof_all += np.nan_to_num(oof)
        print(f"  seed {sd}: OOF AUROC {au:.3f} on {int(ok.sum())} scored "
              f"events", flush=True)

    oof_all /= len(seeds)
    au = auroc(oof_all.tolist(), y.tolist())
    lo, hi = grouped_boot(oof_all.tolist(), y.tolist(), groups, a.n_boot,
                          a.seed)  # groups = recording per event
    m = sum(per_seed) / len(per_seed)
    sd_ = (sum((x - m) ** 2 for x in per_seed) / max(len(per_seed) - 1, 1)) ** .5
    print(f"\n{'=' * 74}\nVIDEO PRIOR on this contrast\n{'=' * 74}")
    print(f"  per-seed {['%.3f' % x for x in per_seed]}   "
          f"mean {m:.3f} +/- {sd_:.3f}")
    print(f"  seed-averaged OOF AUROC {au:.3f}   grouped 95% [{lo:.3f}, "
          f"{hi:.3f}]")
    print(f"  random-scorer bar       {bar:.3f}"
          + ("   <- CLEARS IT" if au > bar else "   <- inside the noise"))
    print(f"\n  naming features on the same gold, for the table "
          f"(46 YES / 17 NO, bar 0.666):")
    print(f"    verb_min 0.558   verb_mean 0.566   obj_min 0.566   "
          f"obj_mean 0.563   generic_any 0.500")
    print(f"  Quoted from the round-2 run. If the counts printed above differ "
          f"from 46/17, this\n  head is scoring a different subset than the "
          f"naming table did and the two rows are\n  not in the same table "
          f"after all.")
    print(f"\n  This head never reads a label, so it cannot verify one. If it "
          f"lands where the\n  naming features land, nothing in this feature "
          f"family separates the contrast. If\n  it lands clearly higher, the "
          f"SCENE predicts the status better than any\n  label-reading signal "
          f"does -- which would say the status is partly a property of\n  "
          f"which videos were audited rather than of the labels.")

    if a.out:
        json.dump({"n_yes": int(y.sum()), "n_no": int((1 - y).sum()),
                   "recordings": len(set(groups)), "per_seed": per_seed,
                   "auroc": au, "grouped_95": [lo, hi], "bar": bar},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
