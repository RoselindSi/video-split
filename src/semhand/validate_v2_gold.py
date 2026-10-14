import json, sys, collections
def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def check(path):
    g=json.load(open(path,encoding='utf-8')); out=[]
    samp=set(g['sampled_frames']); oth=g.get('others') or []
    comp=set(g.get('others_complete_frames') or [])
    out.append(('schema',g.get('schema_version')))
    out.append(('手 anchors',[len(h['anchors']) for h in g['hands']]))
    out.append(('手画',[sum(1 for a in h['anchors'] if a['drawn']) for h in g['hands']]))
    out.append(('others','%d 个 %s'%(len(oth),dict(collections.Counter(o['kind'] for o in oth)))))
    out.append(('判全的帧','%d / %d'%(len(comp),len(samp))))
    nc=sorted(samp-comp)
    if nc: out.append(('未判全的帧',nc))
    bad=sorted({o['frame'] for o in oth}-samp)
    if bad: out.append(('！others 帧不在网格里',bad))
    contra=sorted({o['frame'] for o in oth}-comp)
    if contra: out.append(('！有标注却不在 complete',contra))
    # 同一个框既被指认为本人的手、又被判成 others —— 这是矛盾，不是噪声
    anch=collections.defaultdict(list)
    for h in g['hands']:
        for a in h['anchors']: anch[a['frame']].append((h['hand_key'],tuple(a['box'])))
    dup=[]
    for o in oth:
        for k,b in anch.get(o['frame'],[]):
            if iou(tuple(o['box']),b)>=0.9:
                dup.append((o['frame'],k,o['kind'],round(iou(tuple(o['box']),b),2)))
    if dup: out.append(('！同一个框既是手又被判 others',dup))
    return out
for p in sys.argv[1:]:
    print('  === %s ==='%p.split('/')[-1])
    for k,v in check(p): print('    %-30s %s'%(k,v))
