"""Matched-parameter contrasts and synthetic posterior corruption; no refitting."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from graph_u1_query import prepare, predict


def summarize(ps):
    usable = [p for p in ps if p['predecessor_gold'] is not None]
    return dict(n=len(ps), accuracy=float(np.mean([p['hit'] for p in ps])),
                rescue=sum(p['rescue'] for p in ps), override=sum(p['override'] for p in ps),
                mean_gate=float(np.mean([p['entropy_gate'] for p in ps])),
                mean_weight=float(np.mean([p['memory_weight'] for p in ps])),
                predecessor_accuracy=float(np.mean([p['predecessor_top1']==p['predecessor_gold'] for p in usable])),
                mean_gold_mass=float(np.mean([p['gold_predecessor_mass'] for p in usable if p['gold_predecessor_mass'] is not None])))


def contrast(a, b, seed=20261016):
    assert [(p['video'],p['segment']) for p in a] == [(p['video'],p['segment']) for p in b]
    by = collections.defaultdict(lambda: [0,0])
    for x,y in zip(a,b):
        by[x['video']][0] += int(x['hit'])-int(y['hit'])
        by[x['video']][1] += 1
    pairs = np.array(list(by.values()))
    rng = np.random.default_rng(seed)
    samples = pairs[rng.integers(len(pairs),size=(4000,len(pairs)))].sum(axis=1)
    return dict(delta=float((sum(p['hit'] for p in a)-sum(p['hit'] for p in b))/len(a)),
                ci=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist())


def corruption(rate, seed, kind):
    def transform(r, q):
        value = hashlib.sha256(('%s:%s' % (seed,r['seg_id'])).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(value[:8],'little'))
        if rng.random() >= rate:
            return q
        top = int(q.argmax())
        other = int(rng.choice([i for i in range(len(q)) if i != top]))
        altered = q.copy()
        altered[top], altered[other] = q[other], q[top]
        if kind == 'uncertain':
            altered = .5 * altered + .5 / len(q)
        return altered
    return transform


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--emb',required=True)
    ap.add_argument('--release',required=True)
    ap.add_argument('--nodemap',required=True)
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    sys.path.insert(0,args.source)
    from src.auditor.boundary.g1_probe import task_rows
    baseline=json.loads(Path(args.baseline).read_text(encoding='utf-8'))
    cfg=baseline['config']
    z=np.load(args.emb,allow_pickle=True)
    raw=dict(zip(z['ids'].tolist(),z['vecs'].astype(float)))
    rows=[r for r in task_rows(args.release,args.nodemap) if r['seg_id'] in raw]
    all_vids=set(v for fold in baseline['folds'] for v in fold)
    matched=collections.defaultdict(list)
    noisy=collections.defaultdict(list)
    # These settings are fixed before inspecting noise outcomes.
    settings=dict(rates=[0,.1,.2,.3,.4], seeds=list(range(10)),
                  kinds=['confident_swap','uncertain'], parameters_from='P inner-CV selection',
                  note='Synthetic posterior perturbation, not a calibrated sensor model. Rate is intervention probability, not actual predecessor error.')
    for fold,test in enumerate(baseline['folds']):
        data=prepare(rows,raw,all_vids-set(test))
        for reference in ('P','B'):
            pars=next(s['parameters'] for s in baseline['selection'] if s['fold']==fold and s['arm']==reference)
            for arm in cfg['arms']:
                matched[reference+':'+arm].extend(predict(data,set(test),arm,pars,cfg))
            if reference=='P':
                for kind in settings['kinds']:
                    for rate in settings['rates']:
                        for seed in settings['seeds']:
                            for arm in ('P','B','PG','BG'):
                                key='%s:%s:%s:%s'%(kind,rate,seed,arm)
                                noisy[key].extend(predict(data,set(test),arm,pars,cfg,corruption(rate,seed,kind)))
        print('diagnostic fold',fold,'complete',flush=True)
    contrasts={ref+':'+a+'-'+b:contrast(matched[ref+':'+a],matched[ref+':'+b])
               for ref in ('P','B') for a,b in [('B','P'),('BG','PG'),('BG','P')]}
    noise_summary=[]
    for kind in settings['kinds']:
        for rate in settings['rates']:
            for arm in ('P','B','PG','BG'):
                summaries=[summarize(noisy['%s:%s:%s:%s'%(kind,rate,seed,arm)]) for seed in settings['seeds']]
                noise_summary.append(dict(kind=kind,rate=rate,arm=arm,
                    means={k:float(np.mean([s[k] for s in summaries])) for k in summaries[0]},
                    accuracy_seed_range=[min(s['accuracy'] for s in summaries),max(s['accuracy'] for s in summaries)]))
    output=dict(settings=settings,matched_summary={k:summarize(ps) for k,ps in matched.items()},
                contrasts=contrasts,noise_summary=noise_summary,matched_predictions=matched,
                noise_predictions=noisy)
    Path(args.out).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(matched=output['matched_summary'],contrasts=contrasts),indent=2))


if __name__=='__main__':
    main()
