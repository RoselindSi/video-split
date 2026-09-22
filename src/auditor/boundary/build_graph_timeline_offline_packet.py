"""Build a self-contained, label-blind whole-timeline annotation packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_packet(manifest_path: str | Path, out_dir: str | Path,
                 reviewer_id: str, batch_id: str, title: str,
                 expected_count: int = 30, force: bool = False) -> dict:
    source_manifest = Path(manifest_path).resolve()
    manifest_raw = source_manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    videos = manifest.get("videos") or []
    if len(videos) != expected_count:
        raise ValueError(
            f"expected {expected_count} videos but manifest contains {len(videos)}"
        )

    out = Path(out_dir).resolve()
    archive = out.with_suffix(".zip")
    if out.exists() or archive.exists():
        if not force:
            raise FileExistsError(f"output already exists: {out} or {archive}")
        if out.exists():
            shutil.rmtree(out)
        archive.unlink(missing_ok=True)
    video_dir = out / "videos"
    video_dir.mkdir(parents=True)

    public_cases = []
    checksum_rows = []
    for index, row in enumerate(videos, 1):
        source = Path(row["video"])
        if not source.is_file():
            raise FileNotFoundError(f"video is missing: {source}")
        expected = str(row.get("video_sha256") or "")
        actual = sha256(source)
        if expected and expected != actual:
            raise ValueError(f"video hash changed: {row['video_id']}")
        target = video_dir / f"{row['video_id']}.mp4"
        shutil.copy2(source, target)
        checksum_rows.append(f"{actual}  videos/{target.name}")
        public_cases.append({
            "video_id": str(row["video_id"]),
            "duration_s": float(row["duration_s"]),
            "video_sha256": actual,
            "blind_index": index,
        })

    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    template = Path(__file__).with_name(
        "interaction_graph_timeline_offline_page.html"
    ).read_text(encoding="utf-8")
    replacements = {
        "__CASES__": json.dumps(public_cases, ensure_ascii=False),
        "__MANIFEST_HASH__": manifest_hash,
        "__ANNOTATOR_ID__": reviewer_id,
        "__BATCH_ID__": batch_id,
        "__TITLE__": title,
    }
    page = template
    for marker, replacement in replacements.items():
        page = page.replace(marker, replacement)
    if any(marker in page for marker in replacements):
        raise AssertionError("offline page still contains an unreplaced placeholder")
    page_path = out / "打开标注页面.html"
    page_path.write_text(page, encoding="utf-8")

    readme = f"""{title}

打开方法
1. 点“解压全部文件”，把整个 ZIP 解压到普通文件夹。不要在 WPS 压缩包预览里直接打开。
2. 确认“打开标注页面.html”和 videos 文件夹在同一层。
3. 用 Chrome 或 Edge 打开“打开标注页面.html”。如果视频仍不显示，点页面里的“选择 videos 文件夹”。

统一规则
1. 同一次任务继续做，包括来回擦、反复拧：不要切段。
2. 同一种事已经结束，之后重新开始：在真正开始处切开，右边继续选择原事件。
3. 目标换了：在新目标真正开始处切开，右边新建或选择另一个事件。
4. 空白等待本身不单独建事件；切点放在后面动作真正开始的位置。
5. 每段都写事件，完整看完后再提交。

交回文件
30 条都提交后，点击“导出最终 JSON”，把导出的 JSON 发回。CSV 只用于人工查看。
页面会把草稿存在当前浏览器；换电脑前先点击“导出草稿备份”。
不要查看另一位标注者的答案，也不要讨论逐条判断。
"""
    (out / "README_先看这里.txt").write_text(readme, encoding="utf-8")
    packet_manifest = {
        "schema_version": "interaction_graph_timeline_offline_packet_v1",
        "batch_id": batch_id,
        "reviewer_id": reviewer_id,
        "n_videos": len(public_cases),
        "source_manifest_sha256": manifest_hash,
        "label_blind": True,
        "cases": public_cases,
    }
    packet_manifest_path = out / "packet_manifest.json"
    packet_manifest_path.write_text(
        json.dumps(packet_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_rows.extend([
        f"{sha256(page_path)}  {page_path.name}",
        f"{sha256(packet_manifest_path)}  {packet_manifest_path.name}",
    ])
    (out / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(out.name) / path.relative_to(out))
    return {
        "directory": str(out),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "n_videos": len(public_cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_packet(
        args.manifest, args.out, args.reviewer_id, args.batch_id, args.title,
        args.expected_count, args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
