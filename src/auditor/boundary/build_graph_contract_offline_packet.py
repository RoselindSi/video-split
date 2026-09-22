"""Build the blind Reviewer 2 packet that runs entirely from local files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


REMOTE_MANIFEST = (
    "/workspace/tr1/interaction_evidence_v2_20260914/results/auditor/"
    "interaction_graph_contract_v1_review30/review_manifest.json"
)
EXPECTED_MANIFEST_SHA256 = "85ae4ac60e571cae639530e4248ea68fd3edfa1f2bcaabf7109f49dada301d88"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    template = Path(__file__).with_name("interaction_graph_contract_offline_page.html")
    manifest_raw = subprocess.check_output(
        ["ssh", "sixiang-dev", f"cat {REMOTE_MANIFEST}"], text=False
    )
    if hashlib.sha256(manifest_raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("The public review manifest changed; do not mix packet versions")
    manifest = json.loads(manifest_raw)
    cases = manifest.get("cases") or []
    if manifest.get("n_cases") != 30 or len(cases) != 30:
        raise ValueError("Expected the blind 30-case public manifest")

    out = args.out.resolve()
    archive = out.with_suffix(".zip")
    if out.exists() or archive.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {out} or {archive}")
        if out.exists():
            shutil.rmtree(out)
        archive.unlink(missing_ok=True)
    videos = out / "videos"
    videos.mkdir(parents=True)

    public_cases = []
    checksums = []
    for case in cases:
        review_id = case["review_id"]
        source = args.clip_cache / f"{review_id}.mp4"
        expected = case["video_sha256"]
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Missing or changed cached clip: {review_id}")
        target = videos / source.name
        shutil.copy2(source, target)
        checksums.append(f"{expected}  videos/{source.name}")
        public_cases.append({
            "review_id": review_id,
            "duration_s": case["duration_s"],
            "candidate_offset_s": case["candidate_offset_s"],
            "video_sha256": expected,
        })

    page = template.read_text(encoding="utf-8")
    page = page.replace("__CASES__", json.dumps(public_cases, ensure_ascii=False))
    page = page.replace("__MANIFEST_HASH__", EXPECTED_MANIFEST_SHA256)
    if "__CASES__" in page or "__MANIFEST_HASH__" in page:
        raise AssertionError("Offline page placeholders were not replaced")
    page_path = out / "打开标注页面.html"
    page_path.write_text(page, encoding="utf-8")

    guide_target = out / "动作关系盲标填写指南.docx"
    shutil.copy2(args.guide, guide_target)
    readme = """第二位标注者离线包

1. 先把整个 ZIP 完整解压到一个文件夹，不要在压缩包预览里直接打开 HTML。
2. 确认“打开标注页面.html”旁边有 videos 文件夹，里面有 30 个 MP4。
3. 双击“打开标注页面.html”，用 Chrome 或 Edge 打开；不要用微信内置预览。
   如果播放器仍停在 0:00，按页面提示点击“选择 videos 文件夹”。
4. 每条先看完整视频，再填写。页面会自动在当前浏览器保存草稿。
5. 标记点在空白处，但后面清楚出现不同的新动作：选“换了另一种事”，边界填新动作真正开始的秒数。
6. 30 条全部提交后，点击“导出最终 CSV”。
7. 把 reviewer_2_graph_contract_review.csv 发给项目负责人即可。

若中途要换电脑、浏览器或新版标注包：先点旧页面的“导出备份 JSON”，
到新页面后点“恢复备份”。Word 指南中 127.0.0.1 的入口只适用于在线版，
离线标注请始终双击本文件夹里的 HTML。
不要查看第一位标注者的答案，也不要交换逐条判断。
"""
    (out / "README_使用说明.txt").write_text(readme, encoding="utf-8")
    packet_manifest = {
        "schema_version": manifest["schema_version"],
        "n_cases": 30,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "contract_sha256": manifest["contract_sha256"],
        "reviewer_id": "reviewer_2",
        "cases": public_cases,
    }
    (out / "packet_manifest.json").write_text(
        json.dumps(packet_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums.extend([
        f"{sha256(page_path)}  打开标注页面.html",
        f"{sha256(guide_target)}  动作关系盲标填写指南.docx",
    ])
    (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(out.name) / path.relative_to(out))
    print(out)
    print(archive)
    print(f"archive_sha256={sha256(archive)}")
    print(f"archive_bytes={archive.stat().st_size}")


if __name__ == "__main__":
    main()
