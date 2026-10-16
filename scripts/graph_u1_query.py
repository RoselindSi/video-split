"""Nested-video-CV query diagnostic; oracle prefix memory, no online writes.

Default: previous observation is classified against the CURRENT prefix prototypes,
which include that observation with oracle identity. This is an explicitly
optimistic diagnostic interface shared by point and belief, not online state
maintenance. --causal instead uses the saved preceding decision scores.
Special-state boundaries yield uniform memory.
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=float)
    p = np.exp(x - x.max())
    return p / p.sum()


def history(q, names, edges, epsilon):
    """Preserve unsupported mass as uniform, smooth supported distributions."""
    uniform = np.ones(len(names)) / len(names)
    out = np.zeros(len(names))
    evidence = 0.0
    for i, mass in enumerate(q):
        successors = {t for (s, t), n in edges.items() if s == names[i] and n > 0}
        count = sum(n for (s, _), n in edges.items() if s == names[i])
        indices = [j for j, name in enumerate(names) if name in successors]
        dist = uniform.copy()
        if indices:
            dist = np.zeros(len(names))
            dist[indices] = 1 / len(indices)
            dist = (dist + epsilon) / (1 + epsilon * len(names))
        out += mass * dist
        evidence += mass * count
    assert np.isclose(out.sum(), 1)
    return out, evidence


def prepare(rows, raw, training_videos, causal=False):
    # Estimate centring on training recordings, including their prefix segments.
    train_ids = set()
    for r in rows:
        if r['video_id'] in training_videos:
            train_ids.add(r['seg_id'])
            train_ids.update(r['node_of'])
    center = np.mean([raw[s] for s in sorted(train_ids) if s in raw], axis=0)
    vec = {s: x - center for s, x in raw.items()}
    vec = {s: x / max(np.linalg.norm(x), 1e-9) for s, x in vec.items()}
    out = []
    for r in rows:
        names, protos = [], []
        for name in r['candidates']:
            xs = [vec[s] for s in r['prefix_ids'] if s in vec and r['node_of'].get(s) == name]
            if xs:
                mu = np.mean(xs, axis=0)
                names.append(name)
                protos.append(mu / max(np.linalg.norm(mu), 1e-9))
        if not names:
            raise ValueError('Missing prototypes: ' + r['seg_id'])
        sims = np.array([np.dot(vec[r['seg_id']], p) for p in protos])
        prev_sims = None
        # Only the immediately preceding segment; never bridge special states.
        number = int(r['seg_id'].rsplit('#', 1)[1])
        prev_id = r['seg_id'].rsplit('#', 1)[0] + '#%02d' % (number - 1)
        if r['prev_node'] is not None and prev_id in vec and prev_id in r['node_of']:
            prev_sims = np.array([np.dot(vec[prev_id], p) for p in protos])
        r = dict(r)
        cache, previous_node = {}, None
        for sid in r['prefix_ids']:
            node = r['node_of'].get(sid)
            if node is not None and previous_node is not None:
                cache[previous_node] = node
            previous_node = node  # Missing/special states reset adjacency.
        r['_cache_edges'] = {(s,t):1 for s,t in cache.items()}
        out.append((r, names + ['NEW'], sims, prev_sims))
    if causal:
        # Snapshot the preceding decision BEFORE its observation was added to
        # oracle-prefix prototypes. NEW mass stays NEW, never gold-aligned.
        snapshots = {r['seg_id']: (names, sims) for r,names,sims,_ in out}
        causal_out = []
        for r,names,sims,_ in out:
            r = dict(r)
            number = int(r['seg_id'].rsplit('#',1)[1])
            prev_id = r['seg_id'].rsplit('#',1)[0] + '#%02d' % (number-1)
            saved = snapshots.get(prev_id) if r['prev_node'] is not None else None
            previous = None
            r['_query_gold'] = None
            if saved:
                old_names, old_sims = saved
                old_scores = dict(zip(old_names[:-1], old_sims))
                previous = np.array([old_scores.get(name, -np.inf) for name in names[:-1]])
                r['_query_gold'] = r['prev_node'] if r['prev_node'] in old_names[:-1] else 'NEW'
            causal_out.append((r,names,sims,previous))
        return causal_out
    return out


def predict(prepared, videos, arm, pars, cfg, query_transform=None):
    temperature, tau, lam = pars
    result = []
    for r, names, sims, prev_sims in prepared:
        if r['video_id'] not in videos:
            continue
        logits = np.append(sims, tau) / temperature
        q = (softmax(np.append(prev_sims, tau) / cfg.get('query_temperature', temperature))
             if prev_sims is not None else np.ones(len(names)) / len(names))
        if query_transform is not None and prev_sims is not None:
            q = query_transform(r, q)
        point = np.zeros(len(names)); point[int(q.argmax())] = 1
        # States revealed since the snapshot have zero mass and must not
        # inflate the denominator of normalized entropy.
        query_k = (int(np.isfinite(prev_sims).sum()) + 1
                   if prev_sims is not None else len(q))
        identity = (1.0 if query_k == 1 else
                    1 + float(np.sum(q * np.log(np.maximum(q, 1e-15)))) / np.log(query_k))
        source = q if arm in ('B', 'BG') else point
        if arm == 'O':
            source = np.zeros(len(names))
            source[names.index(r['prev_node']) if r['prev_node'] in names else -1] = 1
        if prev_sims is None:
            prior, evidence = np.ones(len(names)) / len(names), 0
        else:
            edges = r['_cache_edges'] if cfg.get('memory') == 'cache' else r['edges']
            prior, evidence = history(source, names, edges, cfg['epsilon'])
        if arm == 'M':
            alpha = cfg['uniform_mix']
            prior = (1-alpha)*prior + alpha/len(names)
        weight = lam if arm != 'V' else 0
        if arm in ('PG', 'BG'):
            weight *= max(0, identity) * (1 - np.exp(-evidence / cfg['kappa']))
        scores = logits + weight * np.log(prior)
        if arm == 'K':
            scores[:-1] += cfg['known_bias']
        pick = names[int(scores.argmax())]
        gold = 'NEW' if r['gold'] == 'NEW' else r['gold_node']
        vis_pick = names[int(logits.argmax())]
        result.append(dict(video=r['video_id'], segment=r['seg_id'], gold=gold,
                           pick=pick, hit=pick == gold, visual_pick=vis_pick,
                           rescue=pick == gold and vis_pick != gold,
                           override=pick != gold and vis_pick == gold,
                           predecessor_top1=names[int(q.argmax())],
                           predecessor_gold=r.get('_query_gold',r['prev_node']), entropy_gate=identity,
                           gold_predecessor_mass=float(q[names.index(r.get('_query_gold',r['prev_node']))])
                           if r.get('_query_gold',r['prev_node']) in names else None,
                           memory_weight=float(weight)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--release', required=True)
    ap.add_argument('--nodemap', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--causal', action='store_true')
    ap.add_argument('--calibrate-query', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, a.source)
    from src.auditor.boundary.g1_probe import task_rows
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    z = np.load(a.emb, allow_pickle=True)
    raw = dict(zip(z['ids'].tolist(), z['vecs'].astype(float)))
    rows = [r for r in task_rows(a.release, a.nodemap) if r['seg_id'] in raw]
    vids = sorted({r['video_id'] for r in rows})
    random.Random(cfg['seed']).shuffle(vids)
    folds = [vids[i::cfg['outer_folds']] for i in range(cfg['outer_folds'])]
    grid = list(itertools.product(cfg['temperature_grid'], cfg['tau_grid'], cfg['lambda_grid']))
    all_predictions = collections.defaultdict(list)
    selections = []
    for fold, test in enumerate(folds):
        train = [v for v in vids if v not in test]
        inner = [train[i::cfg['inner_folds']] for i in range(cfg['inner_folds'])]
        prepared_inner = [(held, prepare(rows, raw, set(train) - set(held), a.causal)) for held in inner]
        prepared_outer = prepare(rows, raw, set(train), a.causal)
        calibrated_inner, calibrated_outer = {}, {}
        if a.calibrate_query:
            from graph_u1_calibration import calibration_rows, metrics, GRID
            for tau in cfg['tau_grid']:
                for index,(held,data) in enumerate(prepared_inner):
                    fitting = calibration_rows(data,set(train)-set(held),tau)
                    calibrated_inner[index,tau] = min(GRID,key=lambda qt: metrics(fitting,qt)['nll'])
                fitting = calibration_rows(prepared_outer,set(train),tau)
                calibrated_outer[tau] = min(GRID,key=lambda qt: metrics(fitting,qt)['nll'])
        for arm in cfg['arms']:
            candidates = grid if arm != 'V' else [p for p in grid if p[2] == 0]
            best_score, best = -1, None
            for pars in candidates:
                predictions = []
                for index,(held,data) in enumerate(prepared_inner):
                    local_cfg = dict(cfg)
                    if a.calibrate_query:
                        local_cfg['query_temperature'] = calibrated_inner[index,pars[1]]
                    predictions.extend(predict(data,set(held),arm,pars,local_cfg))
                score = sum(p['hit'] for p in predictions) / len(predictions)
                if score > best_score:
                    best_score, best = score, pars
            local_cfg=dict(cfg)
            if a.calibrate_query:
                local_cfg['query_temperature']=calibrated_outer[best[1]]
            selections.append(dict(fold=fold, arm=arm, parameters=best, inner_accuracy=best_score,
                                   query_temperature=local_cfg.get('query_temperature')))
            all_predictions[arm].extend(predict(prepared_outer, set(test), arm, best, local_cfg))
        print('fold', fold, 'complete', flush=True)
    summary = {arm: dict(n=len(ps), accuracy=sum(p['hit'] for p in ps)/len(ps),
                         rescue=sum(p['rescue'] for p in ps), override=sum(p['override'] for p in ps))
               for arm, ps in all_predictions.items()}
    deltas = {}
    for left, right in [('B','P'), ('BG','PG'), ('BG','P'), ('BG','V')]:
        by_video = {}
        for v in vids:
            x = [p['hit'] for p in all_predictions[left] if p['video'] == v]
            y = [p['hit'] for p in all_predictions[right] if p['video'] == v]
            by_video[v] = (sum(x)-sum(y),len(x))
        rng = random.Random(cfg['seed'])
        values = []
        for _ in range(cfg['bootstrap_repetitions']):
            sample = [by_video[rng.choice(vids)] for _ in vids]
            values.append(sum(x for x,n in sample)/sum(n for x,n in sample))
        deltas[left+'-'+right] = dict(delta=summary[left]['accuracy']-summary[right]['accuracy'],
                                      ci=np.quantile(values,[.025,.975]).tolist())
    output = dict(causal=a.causal,calibrate_query=a.calibrate_query,config=cfg, config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                  folds=folds, selection=selections, summary=summary, deltas=deltas,
                  predictions=all_predictions)
    Path(a.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=summary, deltas=deltas), indent=2))


if __name__ == '__main__':
    main()
