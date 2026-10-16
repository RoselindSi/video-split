"""Nested-CV content-free known-state bias against causal belief."""
import argparse
import collections
import itertools
import json
from pathlib import Path
import random
import sys
import numpy as np
from graph_u1_query import prepare,predict
from graph_u1_diagnostics import summarize,contrast


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','emb','release','nodemap','baseline','config','out'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.source)
    from src.auditor.boundary.g1_probe import task_rows
    base=json.loads(Path(a.baseline).read_text())
    control=json.loads(Path(a.config).read_text())
    cfg=base['config']
    z=np.load(a.emb,allow_pickle=True)
    raw=dict(zip(z['ids'].tolist(),z['vecs'].astype(float)))
    rows=[r for r in task_rows(a.release,a.nodemap) if r['seg_id'] in raw]
    videos=sorted(v for fold in base['folds'] for v in fold)
    random.Random(cfg['seed']).shuffle(videos)
    assert [videos[i::cfg['outer_folds']] for i in range(cfg['outer_folds'])]==base['folds']
    ps=[]; selection=[]
    grid=list(itertools.product(cfg['temperature_grid'],cfg['tau_grid'],control['bias_grid']))
    for fold,test in enumerate(base['folds']):
        train=[v for v in videos if v not in test]
        inner=[train[i::cfg['inner_folds']] for i in range(cfg['inner_folds'])]
        data_inner=[(held,prepare(rows,raw,set(train)-set(held),True)) for held in inner]
        best=None; accuracy=-1
        for t,tau,bias in grid:
            local=dict(cfg,known_bias=bias)
            predictions=[p for held,data in data_inner for p in predict(data,set(held),'K',(t,tau,0),local)]
            score=sum(p['hit'] for p in predictions)/len(predictions)
            if score>accuracy:
                accuracy=score; best=(t,tau,bias)
        t,tau,bias=best
        data=prepare(rows,raw,set(train),True)
        ps.extend(predict(data,set(test),'K',(t,tau,0),dict(cfg,known_bias=bias)))
        selection.append(dict(fold=fold,temperature=t,tau=tau,bias=bias,inner_accuracy=accuracy,
                              effective_cosine_threshold=tau-t*bias))
    result=dict(config=control,selection=selection,summary=summarize(ps),
        contrasts={'K-V':contrast(ps,base['predictions']['V']),
                   'B-K':contrast(base['predictions']['B'],ps)},predictions=ps)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))


if __name__=='__main__':
    main()
