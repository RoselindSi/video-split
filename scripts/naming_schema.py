"""What is in the naming jsonls, and can they be joined to the audited events?

The semantic verification arm needs a video-grounded name per audited event.
The naming pipeline produces one per SEGMENT, so the join is the question, not
a detail -- an arm built on a key that matches 12 of 188 events is a different
experiment on a different population.
"""
import json, os, sys, collections

for p in sys.argv[1:]:
    if not os.path.exists(p):
        print(f"{p}: missing"); continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"\n{os.path.basename(p)}: {len(rows)} rows")
    if not rows:
        continue
    keys = collections.Counter(k for r in rows for k in r)
    print(f"  fields: {dict(keys.most_common(24))}")
    r = rows[0]
    for k, v in list(r.items())[:14]:
        s = json.dumps(v, ensure_ascii=False)
        print(f"    {k:<26} {s[:88]}")
