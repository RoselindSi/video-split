"""Causal calibrated belief vs point-uniform softening, cache vs support."""
import argparse
import collections
import itertools
import json
import random
from pathlib import Path
import sys
import numpy as np
from graph_u1_query import prepare, predict
from graph_u1_calibration import calibration_rows, metrics, GRID
from graph_u1_diagnostics import contrast, summarize


def main():
    ap=argparse.ArgumentParser()
    for key in ('source','emb','release','nodemap','baseline','config','out'):
        ap.add_argument('--'+key,required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.source)
    from src.auditor.boundary.g1_probe import task_rows
    base=json.loads(Path(a.baseline).read_text())
    controls=json.loads(Path(a.config).read_text())
    cfg=base['config']
    z=np.load(a.emb,allow_pickle=True)
    raw=dict(zip(z['ids'].tolist(),z['vecs'].astype(float)))
    rows=[r for r in task_rows(a.release,a.nodemap) if r['seg_id'] in raw]
    videos=sorted(v for fold in base['folds'] for v in fold)
    random.Random(cfg['seed']).shuffle(videos)
    assert [videos[i::cfg['outer_folds']] for i in range(cfg['outer_folds'])] == base['folds']
    predictions=collections.defaultdict(list)
    selection=[]
    grid=list(itertools.product(cfg['temperature_grid'],cfg['tau_grid'],cfg['lambda_grid']))
    for fold,test in enumerate(base['folds']):
        train=[v for v in videos if v not in test]
        inner=[train[i::cfg['inner_folds']] for i in range(cfg['inner_folds'])]
        data_inner=[(held,prepare(rows,raw,set(train)-set(held),True)) for held in inner]
        data_outer=prepare(rows,raw,set(train),True)
        qt_inner={}
        qt_outer={}
        for tau in cfg['tau_grid']:
            for i,(held,data) in enumerate(data_inner):
                cal=calibration_rows(data,set(train)-set(held),tau)
                qt_inner[i,tau]=min(GRID,key=lambda t:metrics(cal,t)['nll'])
            cal=calibration_rows(data_outer,set(train),tau)
            qt_outer[tau]=min(GRID,key=lambda t:metrics(cal,t)['nll'])
        for memory in controls['memories']:
            for arm in controls['arms']:
                best=None; best_score=-1
                for pars in grid:
                    alphas=controls['uniform_mix_grid'] if arm=='M' else [0.]
                    for alpha in alphas:
                        ps=[]
                        for i,(held,data) in enumerate(data_inner):
                            local=dict(cfg,memory=memory,uniform_mix=alpha,query_temperature=qt_inner[i,pars[1]])
                            ps.extend(predict(data,set(held),arm,pars,local))
                        accuracy=sum(p['hit'] for p in ps)/len(ps)
                        if accuracy>best_score:
                            best_score=accuracy; best=(pars,alpha)
                pars,alpha=best
                local=dict(cfg,memory=memory,uniform_mix=alpha,query_temperature=qt_outer[pars[1]])
                key=memory+':'+arm
                predictions[key].extend(predict(data_outer,set(test),arm,pars,local))
                selection.append(dict(fold=fold,memory=memory,arm=arm,parameters=pars,
                    uniform_mix=alpha,query_temperature=local['query_temperature'],inner_accuracy=best_score))
        print('softening fold',fold,'complete',flush=True)
    summary={k:summarize(v) for k,v in predictions.items()}
    deltas={}
    for memory in controls['memories']:
        deltas[memory+':B-M']=contrast(predictions[memory+':B'],predictions[memory+':M'])
        deltas[memory+':B-P']=contrast(predictions[memory+':B'],predictions[memory+':P'])
    for arm in controls['arms']:
        deltas[arm+':support-cache']=contrast(predictions['support:'+arm],predictions['cache:'+arm])
    output=dict(controls=controls,folds=base['folds'],selection=selection,summary=summary,
                contrasts=deltas,predictions=predictions)
    Path(a.out).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,contrasts=deltas),indent=2))


if __name__=='__main__':
    main()
