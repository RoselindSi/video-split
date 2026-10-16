"""Train-only temperature calibration for the optimistic oracle-prefix query."""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np
from graph_u1_query import prepare, predict, softmax
from graph_u1_diagnostics import summarize, contrast


GRID = [.01, .02, .05, .1, .25, .5, 1., 2.]


def calibration_rows(data, videos, tau):
    result=[]
    for row,names,sims,prev in data:
        target = row.get('_query_gold',row['prev_node'])
        if row['video_id'] in videos and prev is not None and target in names:
            result.append((np.append(prev,tau),names.index(target)))
    return result


def metrics(rows, temperature):
    ps=[softmax(x/temperature) for x,y in rows]
    nll=float(np.mean([-np.log(max(p[y],1e-15)) for p,(_,y) in zip(ps,rows)]))
    confidences=np.array([p.max() for p in ps])
    hits=np.array([p.argmax()==y for p,(_,y) in zip(ps,rows)])
    ece=0.
    for i in range(10):
        mask=(confidences >= i/10)&(confidences < (i+1)/10 if i<9 else confidences<=1)
        if mask.any():
            ece+=float(mask.mean()*abs(hits[mask].mean()-confidences[mask].mean()))
    return dict(n=len(rows),nll=nll,ece=ece,accuracy=float(hits.mean()),
                confidence=float(confidences.mean()))


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','emb','release','nodemap','baseline','out'):
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    sys.path.insert(0,args.source)
    from src.auditor.boundary.g1_probe import task_rows
    base=json.loads(Path(args.baseline).read_text(encoding='utf-8'))
    z=np.load(args.emb,allow_pickle=True)
    raw=dict(zip(z['ids'].tolist(),z['vecs'].astype(float)))
    rows=[r for r in task_rows(args.release,args.nodemap) if r['seg_id'] in raw]
    vids=set(v for fold in base['folds'] for v in fold)
    predictions=collections.defaultdict(list)
    selections=[]
    for fold,test in enumerate(base['folds']):
        train=vids-set(test)
        data=prepare(rows,raw,train,base.get('causal',False))
        pars=next(s['parameters'] for s in base['selection'] if s['fold']==fold and s['arm']=='P')
        t,tau,lam=pars
        calibration=calibration_rows(data,train,tau)
        scores=[(metrics(calibration,qt)['nll'],qt) for qt in GRID]
        qt=min(scores)[1]
        held=calibration_rows(data,set(test),tau)
        selections.append(dict(fold=fold,visual_parameters=pars,query_temperature=qt,
            train_nll_grid=scores,held_before=metrics(held,t),held_after=metrics(held,qt)))
        for mode in ('before','after'):
            cfg=dict(base['config'])
            if mode=='after': cfg['query_temperature']=qt
            for arm in ('V','P','B','PG','BG'):
                predictions[mode+':'+arm].extend(predict(data,set(test),arm,pars,cfg))
    summaries={k:summarize(ps) for k,ps in predictions.items()}
    deltas={a+'-'+b:contrast(predictions[a],predictions[b]) for a,b in
            [('after:B','after:P'),('after:BG','after:PG'),('after:BG','after:P'),
             ('after:B','before:B'),('after:BG','before:BG')]}
    for arm in ('V','P'):
        assert [p['pick'] for p in predictions['before:'+arm]] == [p['pick'] for p in predictions['after:'+arm]]
    output=dict(query_temperature_grid=GRID,selection=selections,summary=summaries,
                contrasts=deltas,predictions=predictions,
                causal=base.get('causal',False),
                limitations=['Oracle-prefix identities and transition memory remain in use.',
                  'Causal mode preserves predecessor NEW mass without oracle alignment; first predecessor without a saved decision falls back to uniform.',
                  'Not recursive online tracking; posterior is visual-only and memory writes remain oracle.'])
    Path(args.out).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('predictions',)},indent=2))


if __name__=='__main__':
    main()
