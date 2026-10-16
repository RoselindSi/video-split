"""Conditional successor-identity permutation for causal calibrated queries."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from graph_u1_query import prepare, predict
from graph_u1_diagnostics import contrast, summarize


def permute(data, seed):
    out=[]
    changed=total=0
    for row,names,sims,prev in data:
        row=dict(row)
        digest=hashlib.sha256(('%s:%s'%(seed,row['seg_id'])).encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
        targets=names[:-1]
        mapping=dict(zip(targets,rng.permutation(targets).tolist()))
        for field in ('edges','_cache_edges'):
            original=row[field]
            transformed={(s,mapping[t]):n for (s,t),n in original.items()}
            assert len(transformed)==len(original)
            before=collections.Counter(s for s,t in original)
            after=collections.Counter(s for s,t in transformed)
            assert before==after
            row[field]=transformed
        original_edges=data[len(out)][0]['edges']
        total+=len(original_edges)
        changed+=sum(mapping[t]!=t for s,t in original_edges)
        out.append((row,names,sims,prev))
    return out,dict(changed=changed,total=total)


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','emb','release','nodemap','baseline','config','out'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.source)
    from src.auditor.boundary.g1_probe import task_rows
    base=json.loads(Path(a.baseline).read_text())
    control=json.loads(Path(a.config).read_text())
    z=np.load(a.emb,allow_pickle=True)
    raw=dict(zip(z['ids'].tolist(),z['vecs'].astype(float)))
    rows=[r for r in task_rows(a.release,a.nodemap) if r['seg_id'] in raw]
    videos=set(v for fold in base['folds'] for v in fold)
    predictions=collections.defaultdict(list)
    stats=[]
    for fold,test in enumerate(base['folds']):
        data=prepare(rows,raw,videos-set(test),True)
        data=[d for d in data if d[0]['video_id'] in test]
        for memory in control['memories']:
            selected=next(s for s in base['selection'] if s['fold']==fold and s['memory']==memory and s['arm']=='B')
            cfg=dict(memory=memory,epsilon=.001,kappa=2,query_temperature=selected['query_temperature'])
            real=predict(data,set(test),'B',selected['parameters'],cfg)
            predictions[memory+':real'].extend(real)
            predictions[memory+':visual'].extend(predict(data,set(test),'V',selected['parameters'],cfg))
            for seed in control['seeds']:
                shuffled,stat=permute(data,seed)
                stats.append(dict(fold=fold,memory=memory,seed=seed,**stat))
                predictions[memory+':shuffle:'+str(seed)].extend(predict(shuffled,set(test),'B',selected['parameters'],cfg))
    summary={k:summarize(ps) for k,ps in predictions.items()}
    comparisons={}
    for memory in control['memories']:
        real=predictions[memory+':real']
        assert [p['pick'] for p in real]==[p['pick'] for p in base['predictions'][memory+':B']]
        per_seed=[]
        for seed in control['seeds']:
            shuffled=predictions[memory+':shuffle:'+str(seed)]
            per_seed.append(contrast(real,shuffled))
        vids=sorted(videos)
        differences=np.zeros((len(control['seeds']),len(vids)))
        counts=np.array([sum(p['video']==v for p in real) for v in vids])
        for i,seed in enumerate(control['seeds']):
            shuffled=predictions[memory+':shuffle:'+str(seed)]
            assert [p['segment'] for p in real]==[p['segment'] for p in shuffled]
            for j,v in enumerate(vids):
                differences[i,j]=sum(int(x['hit'])-int(y['hit']) for x,y in zip(real,shuffled) if x['video']==v)
        rng=np.random.default_rng(20261016)
        boot=[]
        for _ in range(4000):
            si=rng.integers(len(control['seeds']),size=len(control['seeds']))
            vi=rng.integers(len(vids),size=len(vids))
            boot.append(differences[np.ix_(si,vi)].mean(axis=0).sum()/counts[vi].sum())
        comparisons[memory]=dict(real_minus_shuffle_mean=float(differences.mean(axis=0).sum()/counts.sum()),
            crossed_seed_video_ci=np.quantile(boot,[.025,.975]).tolist(),per_seed=per_seed,
            shuffle_accuracy_mean=float(np.mean([summary[memory+':shuffle:'+str(s)]['accuracy'] for s in control['seeds']])),
            shuffle_accuracy_range=[min(summary[memory+':shuffle:'+str(s)]['accuracy'] for s in control['seeds']),max(summary[memory+':shuffle:'+str(s)]['accuracy'] for s in control['seeds'])])
    result=dict(config=control,summary=summary,comparisons=comparisons,permutation_stats=stats,predictions=predictions)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
