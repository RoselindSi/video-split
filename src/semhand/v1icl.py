"""The V1 family: V1's frozen features, with and without the semantic layer.

Runs in `/workspace/venv_rig`.

THE IMAGE TOKEN IS V1 ITSELF. The 1088-d vector V1's head reads (512 hand +
512 context + 64 geometry), frozen. For the bank it comes from V1's own
training inputs (`own_ctx.Pairs`, overlay stripped, no augmentation); for the
test hands from the live render in `frames.py`. Both are checked against V1's
own output before use.

ONE MODEL, SWITCHED OFF IN PARTS. Every arm is the same two-layer transformer
(d 256, 4 heads) over per-item tokens, read at the query's label token:

    item   = [image token] [text token]? [label token]      + type and slot
    B0r    query only, no text                      (retraining alone)
    B1     query with its description token
    B2     K templates by semantic retrieval + query
    B2v    K templates by visual retrieval + query
    B2s    K random templates + query               (the null)

A template's label token carries its human label; the query's is MASK. While
training, each template label is also masked with probability 0.3 and
predicted (weight 0.5), the MAE-style objective EgoHandICL uses.

V1'S RECIPE WHERE IT APPLIES. V1's labelled rows, `split_by_recording(0.25)`,
AdamW 3e-4, unweighted cross-entropy, selection on dev `other` F1 -- over 40
epochs instead of 12, since a transformer from scratch on frozen features
needs more steps than a fine-tuned trunk; five seeds, and the test P is their
mean. While training, templates come only from the training recordings, for
dev queries too, so the dev score never sees a dev label.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random

import numpy as np

from src.semhand import K, SEEDS, TRAIN_PKGS, V1_ARMS

MODE = {"B2": "semantic", "B2v": "visual", "B2s": "shuffled"}


def build(arm, d=256, layers=2, heads=4):
    import torch
    import torch.nn as nn

    class SemICL(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_text = arm != "B0r"
            self.k = K if arm in MODE else 0
            self.img = nn.Sequential(nn.LayerNorm(1088), nn.Linear(1088, d))
            self.txt = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, d))
            self.lab = nn.Embedding(3, d)            # 0 other, 1 wearer, 2 MASK
            self.typ = nn.Embedding(3, d)
            self.slot = nn.Embedding(K + 1, d)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True,
                                           norm_first=True), layers)
            self.drop = nn.Dropout(0.1)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, 2)

        def item(self, img, txt, lab, slot):
            dev = img.device
            s = self.slot(torch.full((img.shape[0],), slot, dtype=torch.long, device=dev))
            ty = lambda n: self.typ(torch.full((img.shape[0],), n, dtype=torch.long, device=dev))
            toks = [self.img(img) + ty(0) + s]
            if self.use_text:
                toks.append(self.txt(txt) + ty(1) + s)
            toks.append(self.lab(lab) + ty(2) + s)
            return toks

        def forward(self, q_img, q_txt, t_img, t_txt, t_lab):
            seq = []
            for j in range(self.k):
                seq += self.item(t_img[:, j], t_txt[:, j], t_lab[:, j], j + 1)
            mask = torch.full((q_img.shape[0],), 2, dtype=torch.long, device=q_img.device)
            seq += self.item(q_img, q_txt, mask, 0)
            h = self.norm(self.enc(self.drop(torch.stack(seq, 1))))
            per = 3 if self.use_text else 2
            t_pos = [per * (j + 1) - 1 for j in range(self.k)]
            return self.out(h[:, -1]), (self.out(h[:, t_pos]) if t_pos else None)

    return SemICL()


# ------------------------------------------------------------------ data

def bank_rows(root):
    from src.rig import own_ctx
    ok = {}
    for p in glob.glob(os.path.join(root, "pkg", "index_*.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["status"] == "ok":
                ok[r["stem"]] = r
    rows = [r for r in own_ctx.load(list(TRAIN_PKGS), verbose=False) if r["stem"] in ok]
    for r in rows:
        r["databag"] = ok[r["stem"]]["databag"]
    return rows


def prep(a):
    """Bank features from V1's own training inputs."""
    import torch
    from src.rig import own_ctx
    from src.semhand.frames import v1_features
    rows = bank_rows(a.root)
    model, device, _ = own_ctx.load_model(a.clf_ctx)
    ds = own_ctx.Pairs(rows, augment=False)
    feats, ps = [], []
    with torch.no_grad():
        for s in range(0, len(ds), 64):
            h, c, g, _ = zip(*[ds[j] for j in range(s, min(len(ds), s + 64))])
            h, c, g = (torch.stack(x).to(device) for x in (h, c, g))
            f, p = v1_features(model, h, c, g)
            assert torch.allclose(p, torch.softmax(model(h, c, g), 1)[:, 1], atol=1e-5)
            feats.append(f.cpu().numpy())
            ps.append(p.cpu().numpy())
    p = np.concatenate(ps)
    y = np.array([r["y"] for r in rows])
    s = own_ctx.scores((p >= 0.5).astype(int), y)
    print(f"bank {len(rows)} 行（别人 {int((y == 0).sum())}），V1 在自己训练集上 other F1 "
          f"{s['f1']:.3f}（训练集内，只作核对）")
    os.makedirs(os.path.join(a.root, "v1"), exist_ok=True)
    np.savez(os.path.join(a.root, "v1", "bank_v1.npz"), keys=np.array([r["stem"] for r in rows]),
             feat=np.concatenate(feats), y=y, p=p)


def load_text(root):
    E = np.load(os.path.join(root, "embed.npz"))
    return {i: n for n, i in enumerate(E["ids"].tolist())}, E["text"].astype(np.float32)


def tensors(qids, F, T_at, T, retr, y_of, arm, pool, device):
    """-> dict of tensors for these queries; templates drawn from `pool`."""
    import torch
    qi = np.stack([F[q] for q in qids])
    qt = np.stack([T[T_at[q]] for q in qids])
    ti = np.zeros((len(qids), K, 1088), np.float32)
    tt = np.zeros((len(qids), K, 2048), np.float32)
    tl = np.zeros((len(qids), K), np.int64)
    if arm in MODE:
        for n, q in enumerate(qids):
            got = [t for t in retr[q][MODE[arm]] if t in pool][:K]
            if len(got) < K:                         # rare: pad from the pool, seeded
                rng = random.Random(q)
                got += rng.sample(sorted(pool - set(got)), K - len(got))
            for j, t in enumerate(got):
                ti[n, j], tt[n, j], tl[n, j] = F[t], T[T_at[t]], y_of[t]
    t = lambda x: torch.from_numpy(x).to(device)
    return {"q_img": t(qi), "q_txt": t(qt), "t_img": t(ti), "t_txt": t(tt), "t_lab": t(tl)}


def fit(arm, seed, rows, F, T_at, T, retr, device, epochs=40, batch=32):
    import torch
    from src.rig import own_ctx
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tr, dv = own_ctx.split_by_recording(rows, 0.25, seed)
    y_of = {r["stem"]: r["y"] for r in rows}
    pool = {r["stem"]: r["databag"] for r in tr}
    xtr = tensors([r["stem"] for r in tr], F, T_at, T, retr, y_of, arm, set(pool), device)
    xdv = tensors([r["stem"] for r in dv], F, T_at, T, retr, y_of, arm, set(pool), device)
    ytr = torch.tensor([r["y"] for r in tr], device=device)
    ydv = np.array([r["y"] for r in dv])
    model = build(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    best, best_state = None, None
    for e in range(epochs):
        model.train()
        idx = torch.randperm(len(tr), device=device)
        for s in range(0, len(tr), batch):
            b = idx[s:s + batch]
            x = {k: v[b] for k, v in xtr.items()}
            lab = x["t_lab"]
            hide = (torch.rand(lab.shape, device=device) < 0.3) if model.k else None
            if model.k:
                x["t_lab"] = torch.where(hide, torch.full_like(lab, 2), lab)
            lq, lt = model(**x)
            loss = ce(lq, ytr[b])
            if model.k and hide.any():
                loss = loss + 0.5 * ce(lt[hide], lab[hide])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pd = torch.softmax(model(**xdv)[0], 1)[:, 1].cpu().numpy()
        sc = own_ctx.scores((pd >= 0.5).astype(int), ydv)
        if best is None or (sc["f1"] == sc["f1"] and sc["f1"] > best["f1"]):
            best, best_state = dict(sc, epoch=e + 1), {k: v.detach().clone()
                                                        for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def train_predict(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    rows = bank_rows(a.root)
    B = np.load(os.path.join(a.root, "v1", "bank_v1.npz"))
    F = dict(zip(B["keys"].tolist(), B["feat"]))
    fresh_ids, fresh_bag = [], {}
    for p in sorted(glob.glob(os.path.join(a.root, "fresh", "*", "v1feat.npz"))):
        Z = np.load(p)
        for k, f in zip(Z["keys"].tolist(), Z["feat"]):
            F[k] = f
            fresh_ids.append(k)
    T_at, T = load_text(a.root)
    retr = json.load(open(os.path.join(a.root, "retrieval.json")))
    rows = [r for r in rows if r["stem"] in T_at and r["stem"] in retr]
    queries = [q for q in fresh_ids if q in T_at and q in retr]
    print(f"bank 可训练 {len(rows)} 行；fresh 可预测 {len(queries)}/{len(fresh_ids)}", flush=True)
    y_of = {r["stem"]: r["y"] for r in rows}
    bank_pool = set(y_of)
    out_dir = os.path.join(a.root, "v1")
    preds, report = {}, {}
    for arm in a.arms.split(","):
        ps = []
        for seed in range(SEEDS):
            model, best = fit(arm, seed, rows, F, T_at, T, retr, device)
            x = tensors(queries, F, T_at, T, retr, y_of, arm, bank_pool, device)
            with torch.no_grad():
                p = np.concatenate([torch.softmax(model(**{k: v[s:s + 512] for k, v in x.items()})[0],
                                                  1)[:, 1].cpu().numpy()
                                    for s in range(0, len(queries), 512)])
            ps.append(p)
            report.setdefault(arm, []).append(best)
            print(f"  {arm} seed {seed}: dev F1 {best['f1']:.3f} (epoch {best['epoch']})", flush=True)
        preds[arm] = np.mean(ps, 0)
    with open(os.path.join(out_dir, "pred.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + list(preds))
        for n, q in enumerate(queries):
            w.writerow([q] + [f"{preds[arm][n]:.6f}" for arm in preds])
    json.dump(report, open(os.path.join(out_dir, "dev.json"), "w"), indent=1, default=float)
    print(f"-> {out_dir}/pred.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("prep", "train"), required=True)
    ap.add_argument("--root", default="/workspace/semhand")
    ap.add_argument("--clf_ctx", default="/workspace/own_ctx_best.pt")
    ap.add_argument("--arms", default=",".join(V1_ARMS))
    a = ap.parse_args()
    (prep if a.mode == "prep" else train_predict)(a)


if __name__ == "__main__":
    main()
