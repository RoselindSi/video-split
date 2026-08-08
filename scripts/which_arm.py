import csv, json, collections, sys
DEC, POL = sys.argv[1], sys.argv[2]
ARMS = ["P1 (global) alone", "P1 + local, feature-level", "local alone"]
r = list(csv.DictReader(open(DEC, encoding="utf-8")))
print("== decisions table ==")
for c in ("reason", "policy_role", "source", "reliability"):
    if c in r[0]:
        print(f"  {c:<12}", collections.Counter(x[c] for x in r).most_common(6))
print("\n== where each arm name appears in the policy json ==")
blob = json.load(open(POL, encoding="utf-8"))
hits = collections.defaultdict(list)
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            for a in ARMS:
                if k == a or (isinstance(v, str) and v == a):
                    hits[a].append(f"{path}.{k}" + (f" = {v!r}" if isinstance(v, str) else " (as key)"))
            walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:40]):
            walk(v, f"{path}[{i}]")
    elif isinstance(o, str) and o in ARMS:
        hits[o].append(f"{path} = {o!r}")
walk(blob)
for a in ARMS:
    print(f"  {a!r}: {len(hits[a])} occurrences")
    for h in hits[a][:8]:
        print(f"      {h}")
print("\n== top-level keys ==", list(blob) if isinstance(blob, dict) else type(blob))
