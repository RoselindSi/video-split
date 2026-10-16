"""Audit causal belief rescues and predeclared nonzero lambda contrasts."""
import argparse
import json
from pathlib import Path
import sys
import collections
import numpy as np
from graph_u1_query import prepare, predict, softmax, history
from graph_u1_diagnostics import contrast, summarize


def main():
    ap=argparse.ArgumentParser()
    for key in ('source','emb','release','nodemap','baseline','out'):
        ap.add_argument('--'+key,required=True)
    args=ap.parse_args()
    sys.path.insert(0,args.source)
    from src.auditor.boundary.g1_probe import task_rows
    base=json.loads(Path(args.baseline).read_text())
    z=np.load(args.emb,allow_pickle=True)
    raw=dict(zip(z['ids'].tolist(),z['vecs'].astype(float)))
    rows=[r for r in task_rows(args.release,args.nodemap) if r['seg_id'] in raw]
    vids=set(v for fold in base['folds'] for v in fold)
    rescues={r['segment'] for r in base['predictions']['B'] if r['rescue']}
    audit=[]
    matched=collections.defaultdict(list)
    for fold,test in enumerate(base['folds']):
        prepared=prepare(rows,raw,vids-set(test),True)
        selection=next(s for s in base['selection'] if s['fold']==fold and s['arm']=='B')
        pars=selection['parameters']
        cfg=dict(base['config'],query_temperature=selection['query_temperature'])
        # Match B-selected parameters, and sweep ALL existing nonzero lambdas;
        # no post-hoc choice of the best held-out lambda.
        for key,lam in [('B_selected',pars[2])]+[(str(l),l) for l in cfg['lambda_grid'] if l>0]:
            for arm in ('V','P','B','PG','BG'):
                matched[key+':'+arm].extend(predict(prepared,set(test),arm,(pars[0],pars[1],lam),cfg))
        for r,names,sims,prev in prepared:
            if r['seg_id'] not in rescues or r['video_id'] not in test:
                continue
            assert prev is not None
            q=softmax(np.append(prev,pars[1])/cfg['query_temperature'])
            point=np.zeros(len(q)); point[q.argmax()]=1
            bp,_=history(q,names,r['edges'],cfg['epsilon'])
            pp,_=history(point,names,r['edges'],cfg['epsilon'])
            logits=np.append(sims,pars[1])/pars[0]
            node_scores=[]
            for i,name in enumerate(names):
                successors=sorted(t for (s,t),n in r['edges'].items() if s==name and n>0)
                node_scores.append(dict(node=name,predecessor_probability=float(q[i]),
                    visual_logit=float(logits[i]),point_prior=float(pp[i]),belief_prior=float(bp[i]),
                    point_score=float(logits[i]+pars[2]*np.log(pp[i])),
                    belief_score=float(logits[i]+pars[2]*np.log(bp[i])),successors=successors))
            audit.append(dict(segment=r['seg_id'],video=r['video_id'],fold=fold,
                gold=r['gold'],gold_node=r['gold_node'],true_predecessor=r['prev_node'],
                snapshot_predecessor_gold=r['_query_gold'],query_top1=names[int(q.argmax())],
                parameters=pars,query_temperature=cfg['query_temperature'],candidates=node_scores))
    assert len(audit)==len(rescues)
    summaries={k:summarize(ps) for k,ps in matched.items()}
    comparisons={key:contrast(matched[key+':B'],matched[key+':P']) for key in
                 ['B_selected']+[str(l) for l in base['config']['lambda_grid'] if l>0]}
    result=dict(audit=audit,summary=summaries,contrasts=comparisons,predictions=matched,
        warning='B-selected settings are one reference; sweep is sensitivity analysis, not held-out tuning.')
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summaries,contrasts=comparisons,audit=audit),indent=2))


if __name__=='__main__':
    main()
