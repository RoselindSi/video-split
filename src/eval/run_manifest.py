"""Shared helper: write a `<out>.manifest.json` next to every naming-eval
output so two reports can be compared HORIZONTALLY without guessing.

The concrete failure this fixes: N4 was re-run after a code change (force-
including a real inverse verb as a distractor), which silently changed the
candidate options for many items. The next N5 run consumed that new jsonl,
and its "bda" number (63.0%) got compared against the OLD N4 number (66.7%)
as if they were the same 81 questions -- they weren't. The manifest records
enough to catch this automatically: the exact git commit that generated the
file (so a code change between two runs is visible), the full argv, and a
content hash of every input file consumed (so "was this run built on the
SAME upstream jsonl as that run" is a one-line diff, not a guess).

Usage (inside any eval script's main(), right before/after writing --out):
    from src.eval.run_manifest import write_manifest
    write_manifest(a.out, input_paths=[a.target_data, a.prev_jsonl])
"""
import hashlib, json, os, subprocess, sys, time


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown (not a git checkout or git unavailable)"


def _git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode()
        return bool(out.strip())
    except Exception:
        return None


def _file_hash(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def write_manifest(out_path, input_paths=(), extra=None):
    manifest = {
        "git_commit": _git_commit(),
        "git_dirty_uncommitted_changes": _git_dirty(),
        "argv": sys.argv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "input_files": {p: {"sha256_16": _file_hash(p)} for p in input_paths if p},
        "extra": extra or {},
    }
    manifest_path = out_path + ".manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"wrote run manifest -> {manifest_path} (commit={manifest['git_commit'][:12]}"
          f"{'  [DIRTY: uncommitted local changes]' if manifest['git_dirty_uncommitted_changes'] else ''})")
    return manifest_path


def check_comparable(*manifest_paths):
    """Given 2+ manifest paths, print whether the runs are on the same code
    (git_commit) and, for any shared input file basename, whether the content
    hash matches -- i.e. whether two reports are actually safe to compare."""
    manifests = [json.load(open(p)) for p in manifest_paths]
    commits = {m["git_commit"] for m in manifests}
    if len(commits) > 1:
        print(f"!! DIFFERENT git commits across runs: {commits} -- "
              f"these reports may not be comparable (code changed between runs).")
    else:
        print(f"OK: all runs on the same commit {next(iter(commits))[:12]}")
    by_name = {}
    for p, m in zip(manifest_paths, manifests):
        for path, info in m["input_files"].items():
            by_name.setdefault(os.path.basename(path), {})[p] = info["sha256_16"]
    for name, hashes in by_name.items():
        if len(set(hashes.values())) > 1:
            print(f"!! input file '{name}' has DIFFERENT content across runs: {hashes} "
                  f"-- these runs used different underlying data/items.")
        else:
            print(f"OK: input file '{name}' identical across runs.")


def print_manifest_if_exists(data_path):
    """Call this right after loading any file that write_manifest() may have
    documented, so every downstream report visibly states which commit/config
    produced its input -- no more silently comparing numbers across runs that
    used different code or data without anyone noticing."""
    mp = data_path + ".manifest.json"
    if os.path.exists(mp):
        m = json.load(open(mp))
        print(f"[manifest] {data_path} was produced by commit "
              f"{m['git_commit'][:12]}{' [DIRTY]' if m.get('git_dirty_uncommitted_changes') else ''} "
              f"at {m['timestamp']}, extra={m.get('extra', {})}")
    else:
        print(f"[manifest] !! no manifest found for {data_path} -- can't verify "
              f"which commit/config produced it (was it saved before this "
              f"logging was added?).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check whether N runs are comparable")
    ap.add_argument("manifests", nargs="+")
    a = ap.parse_args()
    check_comparable(*a.manifests)
