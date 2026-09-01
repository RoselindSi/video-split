"""Label harvested hand crops fast, in a browser, off one self-contained file.

WHAT IT IS FOR. The classifier has 37 `other` hands over 13 recordings, and 28
of them are one person in one recording. That is the reason it flips: what it
has mostly learned is a uniform and a lighting. The sweep has 851 crops over
30 recordings waiting, none of them labelled, and until they are labelled they
are worth nothing. This is the tool that turns them into training data.

ONE HTML FILE, NO SERVER. The crops live on a machine reachable only over ssh
and the person labelling is not on it. Anything with a backend means port
forwarding and a process to babysit. The crops are small, so they are embedded
as data URIs and the whole sheet is one file to copy down, open, and label
offline; the labels come back as a CSV to copy up. Progress is written to the
browser's local storage on every keystroke, so closing the tab is not a loss.

RANKING IS FOR TRAINING SETS AND POISON FOR TEST SETS. At a 3.2% base rate,
labelling 400 crops in the order they were harvested buys about 13 positives.
Ordering by the current model's p(other) buys several times that for the same
work, which is the difference between a week and an afternoon. But a set built
that way is drawn from where the model already points, so measuring the model
on it reports the shape of its own bias. Both modes exist here and the mode is
stamped into the sheet, into the CSV, and into the merged rows, because these
two files must never be mixed up and a filename is not enough to keep them
apart.

RECORDING DIVERSITY IS AN EXPLICIT KNOB, NOT A HOPE. Ranking by score will
happily return 400 crops of the same person, which is the problem this set
exists to fix rather than a way to fix it. `--per_recording` caps how many any
one recording can contribute, so the budget is spread across the 30.
"""
from __future__ import annotations

import base64
import csv
import glob
import json
import os
import re

import numpy as np

STEM_RE = re.compile(r"^(.*?)f(\d{6})_h(\d+)$")
LABELS = ("owner", "other", "skip")


def load_rows(pkg, verbose=True):
    """-> [row] for every crop in the package, csv row or not.

    The sweep left 851 crops and 479 csv rows on this corpus. A crop with no
    row is still a labellable hand and its identity is entirely in its
    filename, so it is recovered rather than dropped -- silently labelling 56%
    of what was harvested would be the tool quietly deciding the sample."""
    rows, by_stem = [], {}
    q = os.path.join(pkg, "hands.csv")
    if os.path.exists(q):
        for r in csv.DictReader(open(q, encoding="utf-8-sig")):
            by_stem[r.get("stem", "")] = r
    orphans = 0
    for c in sorted(glob.glob(os.path.join(pkg, "crops", "*.jpg"))):
        stem = os.path.basename(c)[:-4]
        m = STEM_RE.match(stem)
        if not m:
            continue
        r = by_stem.get(stem)
        if r is None:
            orphans += 1
            r = {"stem": stem}
        rows.append({"stem": stem, "tag": m.group(1), "frame": int(m.group(2)),
                     "hand": int(m.group(3)), "_crop": c,
                     "label": (r.get("label") or "").strip(),
                     "rule_owner": r.get("rule_owner", ""),
                     "exit_y": r.get("exit_y", "")})
    if verbose:
        n_tag = len({r["tag"] for r in rows})
        print(f"  {len(rows)} crops over {n_tag} recordings"
              + (f", {orphans} of them had no csv row" if orphans else ""))
    return rows


def score_rows(rows, model_path, verbose=True):
    """Add `p_other` from a trained checkpoint. -> rows

    Missing checkpoint is not fatal: it means the sheet cannot be ranked, and
    an unranked sheet is exactly what a test set wants anyway."""
    from src.rig import own_cnn
    model, device = own_cnn.load_model(model_path)
    if model is None:
        if verbose:
            print(f"  no checkpoint at {model_path}; leaving the order alone")
        for r in rows:
            r["p_other"] = None
        return rows
    torch = own_cnn._torch()
    ds = own_cnn.Crops([dict(r, y=0) for r in rows], augment=False)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            xs = [ds[j][0] for j in range(i, min(i + 64, len(ds)))]
            p = torch.softmax(model(torch.stack(xs).to(device)), 1)
            out.append(p[:, 0].cpu().numpy())        # class 0 is `other`
    for r, p in zip(rows, np.concatenate(out)):
        r["p_other"] = float(p)
    if verbose:
        v = np.array([r["p_other"] for r in rows])
        print(f"  scored: p(other) median {np.median(v):.3f}, "
              f"{int((v > 0.5).sum())} of {len(v)} above 0.5")
    return rows


def order_rows(rows, how="score", per_recording=None, limit=None, seed=0):
    """-> the subset to label, in the order to label it.

    The cap is applied BEFORE the cut to `limit`, so spreading across
    recordings survives the budget instead of being taken off the end of it."""
    rows = [r for r in rows if not r["label"]]
    if how == "score" and any(r.get("p_other") is not None for r in rows):
        rows.sort(key=lambda r: -(r.get("p_other") or 0.0))
    else:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.permutation(len(rows))]
    if per_recording:
        seen, kept = {}, []
        for r in rows:
            n = seen.get(r["tag"], 0)
            if n < per_recording:
                seen[r["tag"]] = n + 1
                kept.append(r)
        rows = kept
    return rows[:limit] if limit else rows


def _thumb(path, size=160):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    s = size / float(max(h, w))
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def build_sheet(rows, out_path, mode, pkg, thumb=160, verbose=True):
    """Write the labelling sheet. -> number of crops in it."""
    items = []
    for r in rows:
        items.append({"stem": r["stem"], "tag": r["tag"],
                      "frame": r["frame"],
                      "p": (None if r.get("p_other") is None
                            else round(float(r["p_other"]), 3)),
                      "img": _thumb(r["_crop"], thumb)})
    payload = json.dumps({"mode": mode, "pkg": os.path.basename(pkg),
                          "items": items})
    html = _HTML.replace("__PAYLOAD__", payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        mb = os.path.getsize(out_path) / 1e6
        tags = len({r["tag"] for r in rows})
        print(f"  {len(items)} crops over {tags} recordings -> {out_path} "
              f"({mb:.1f} MB, mode={mode})")
    return len(items)


def merge_csv(pkg, label_csv, verbose=True):
    """Write labels from a downloaded sheet back into the package. -> counts

    The mode travels with the labels and is written into every row. A training
    sheet and a test sheet look identical once they are CSVs of stem,label,
    and a set that was ranked by the model cannot be used to measure it."""
    got = {}
    mode = ""
    for r in csv.DictReader(open(label_csv, encoding="utf-8-sig")):
        if r.get("label") in LABELS:
            got[r["stem"]] = r["label"]
            mode = r.get("mode", mode)
    q = os.path.join(pkg, "hands.csv")
    rows = list(csv.DictReader(open(q, encoding="utf-8-sig"))) \
        if os.path.exists(q) else []
    have = {r.get("stem"): r for r in rows}
    added = 0
    for stem, lab in got.items():
        if stem in have:
            have[stem]["label"] = lab
            have[stem]["label_mode"] = mode
        else:
            rows.append({"stem": stem, "label": lab, "label_mode": mode})
            added += 1
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(q, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n = {k: sum(1 for v in got.values() if v == k) for k in LABELS}
    if verbose:
        print(f"  merged {len(got)} labels into {q} "
              f"({added} rows were new)")
        print(f"  owner {n['owner']}   other {n['other']}   "
              f"skip {n['skip']}   mode={mode or 'unstamped'}")
    return n


_HTML = """<!doctype html><meta charset="utf-8">
<title>hand crops</title>
<style>
 body{font:13px system-ui;margin:0;background:#111;color:#eee}
 #bar{position:sticky;top:0;background:#000;padding:8px 12px;
      border-bottom:1px solid #333;display:flex;gap:16px;align-items:center}
 #grid{display:flex;flex-wrap:wrap;gap:6px;padding:10px}
 .c{width:160px;border:3px solid #333;border-radius:4px;position:relative;
    cursor:pointer;background:#1a1a1a}
 .c img{width:100%;display:block;border-radius:2px}
 .c .m{font:11px ui-monospace;padding:2px 4px;color:#aaa;
       display:flex;justify-content:space-between}
 .owner{border-color:#2ea043} .other{border-color:#d9534f}
 .skip{border-color:#888;opacity:.45}
 .cur{outline:3px solid #ffd33d;outline-offset:2px}
 button{font:13px system-ui;padding:5px 10px;cursor:pointer}
 b{color:#ffd33d}
</style>
<div id=bar>
 <span id=mode></span>
 <span id=prog></span>
 <span><b>1</b> owner &nbsp; <b>2</b> other &nbsp; <b>3</b> skip
   &nbsp; <b>&larr; &rarr;</b> move &nbsp; <b>u</b> undo</span>
 <button onclick="dl()">download CSV</button>
</div>
<div id=grid></div>
<script>
const D = __PAYLOAD__;
const KEY = "labels:" + D.pkg + ":" + D.mode;
let lab = {}, cur = 0, hist = [];
try { lab = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { lab={}; }
const g = document.getElementById("grid");
D.items.forEach((it, i) => {
  const d = document.createElement("div");
  d.className = "c"; d.id = "c" + i;
  d.innerHTML = '<img src="' + it.img + '"><div class=m><span>' + it.tag +
    it.frame + '</span><span>' + (it.p === null ? "" : it.p) + '</span></div>';
  d.onclick = () => { cur = i; draw(); };
  g.appendChild(d);
});
function draw(){
  D.items.forEach((it,i)=>{
    const e = document.getElementById("c"+i);
    e.className = "c " + (lab[it.stem] || "") + (i===cur ? " cur" : "");
  });
  const n = Object.keys(lab).length;
  const o = Object.values(lab).filter(v=>v==="other").length;
  document.getElementById("prog").textContent =
    n + "/" + D.items.length + " labelled, " + o + " other";
  document.getElementById("mode").innerHTML =
    "<b>" + D.pkg + "</b> &nbsp; mode=" + D.mode;
  localStorage.setItem(KEY, JSON.stringify(lab));
}
function set(v){
  const it = D.items[cur]; if(!it) return;
  hist.push([it.stem, lab[it.stem]]);
  lab[it.stem] = v; cur = Math.min(cur+1, D.items.length-1);
  draw();
  document.getElementById("c"+cur).scrollIntoView({block:"nearest"});
}
document.onkeydown = e => {
  if(e.key==="1") set("owner");
  else if(e.key==="2") set("other");
  else if(e.key==="3") set("skip");
  else if(e.key==="ArrowRight") { cur=Math.min(cur+1,D.items.length-1); draw(); }
  else if(e.key==="ArrowLeft") { cur=Math.max(cur-1,0); draw(); }
  else if(e.key==="u") { const h=hist.pop(); if(h){ if(h[1]===undefined)
      delete lab[h[0]]; else lab[h[0]]=h[1]; draw(); } }
  else return;
  e.preventDefault();
};
function dl(){
  let s = "stem,label,mode\\n";
  for(const k in lab) s += k + "," + lab[k] + "," + D.mode + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([s], {type:"text/csv"}));
  a.download = "labels_" + D.pkg + "_" + D.mode + ".csv";
  a.click();
}
draw();
</script>
"""


def _self_test():
    ok = []

    def chk(c, m):
        ok.append(bool(c))
        print(f"  {'ok ' if c else 'FAIL'} {m}")

    rows = [{"stem": f"R{t}_f{100+i:06d}_h0", "tag": f"R{t}_", "frame": i,
             "label": "", "p_other": (i % 10) / 10.0}
            for t in range(3) for i in range(10)]

    s = order_rows(list(rows), how="score")
    chk(s[0]["p_other"] >= s[-1]["p_other"],
        "ranking puts the likeliest `other` first")
    chk(len(s) == len(rows), "and keeps everything when nothing is capped")

    cap = order_rows(list(rows), how="score", per_recording=2)
    got = {}
    for r in cap:
        got[r["tag"]] = got.get(r["tag"], 0) + 1
    chk(set(got.values()) == {2} and len(got) == 3,
        "the per-recording cap spreads the budget over every recording")

    # The cap has to survive the budget, not be spent by it: taking the top 4
    # of a ranked list would be two recordings, not four.
    cap2 = order_rows(list(rows), how="score", per_recording=1, limit=3)
    chk(len({r["tag"] for r in cap2}) == 3,
        "...and the cap is applied before the limit, not after")

    r1 = order_rows(list(rows), how="random", seed=1)
    r2 = order_rows(list(rows), how="random", seed=2)
    chk([r["stem"] for r in r1] != [r["stem"] for r in r2],
        "a random sheet depends on its seed")
    chk(sorted(r["stem"] for r in r1) == sorted(r["stem"] for r in rows),
        "and drops nothing while reordering")

    done = order_rows([dict(r, label="owner") for r in rows], how="score")
    chk(done == [], "already-labelled crops are not offered again")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pkg = os.path.join(td, "pkg")
        os.makedirs(os.path.join(pkg, "crops"))
        with open(os.path.join(pkg, "hands.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "label"])
            w.writeheader()
            w.writerow({"stem": "A_f000100_h0", "label": ""})
        lc = os.path.join(td, "labels.csv")
        with open(lc, "w", newline="") as f:
            f.write("stem,label,mode\n"
                    "A_f000100_h0,other,rank\nB_f000200_h0,owner,rank\n")
        n = merge_csv(pkg, lc, verbose=False)
        back = list(csv.DictReader(open(os.path.join(pkg, "hands.csv"))))
        chk(n["other"] == 1 and n["owner"] == 1, "merge counts both classes")
        chk(len(back) == 2, "a label for a crop with no csv row adds one")
        chk(all(r.get("label_mode") == "rank" for r in back),
            "and every merged row carries the mode it was labelled under")

    print(f"\n  {sum(ok)}/{len(ok)} cases pass.")
    print("  The mode is stamped because a ranked sheet and a random one are "
          "the same two\n  columns once downloaded, and using the first to "
          "measure the model reports\n  the model's own bias back to it.")
    return all(ok)


def main():
    import argparse
    import sys
    if "--self_test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write a labelling sheet")
    b.add_argument("--pkg", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--mode", choices=("rank", "random"), required=True,
                   help="rank: order by p(other), for TRAINING data. "
                        "random: unbiased, the only kind a TEST set may use.")
    b.add_argument("--model", default="/workspace/own_cnn.pt")
    b.add_argument("--limit", type=int, default=400)
    b.add_argument("--per_recording", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--thumb", type=int, default=160)

    m = sub.add_parser("merge", help="write a downloaded sheet's labels back")
    m.add_argument("--pkg", required=True)
    m.add_argument("--csv", required=True)

    st = sub.add_parser("stats", help="what is labelled so far")
    st.add_argument("--pkg", action="append", required=True)

    a = ap.parse_args()
    if a.cmd == "merge":
        merge_csv(a.pkg, a.csv)
        return
    if a.cmd == "stats":
        import collections
        for p in a.pkg:
            rows = load_rows(p)
            c = collections.Counter(r["label"] or "-" for r in rows)
            tags = collections.Counter(r["tag"] for r in rows
                                       if r["label"] == "other")
            print(f"  {os.path.basename(p):<16} {dict(c)}")
            if tags:
                print(f"    `other` by recording: {dict(tags)}")
        return

    rows = load_rows(a.pkg)
    if a.mode == "rank":
        rows = score_rows(rows, a.model)
    else:
        for r in rows:
            r["p_other"] = None
    sel = order_rows(rows, how=a.mode, per_recording=a.per_recording,
                     limit=a.limit, seed=a.seed)
    if not sel:
        raise SystemExit("nothing left to label in this package")
    build_sheet(sel, a.out, a.mode, a.pkg, thumb=a.thumb)
    print("\n  Copy the sheet down, open it, label with 1/2/3, press "
          "`download CSV`,\n  copy that back up, and merge it with "
          f"`merge --pkg {a.pkg} --csv <file>`.")
    if a.mode == "rank":
        print("  This sheet is RANKED. It is training data. Measuring the "
              "model on crops\n  chosen because the model scored them high "
              "reports its own bias back.")
    else:
        print("  This sheet is RANDOM, so it can carry a test set. Keep it "
              "in a package\n  of its own -- a ranked and a random sheet are "
              "indistinguishable once merged.")


if __name__ == "__main__":
    main()
