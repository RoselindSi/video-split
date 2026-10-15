"""Package the cut B2 clips into a label-blind folder with checksums.

LABEL-BLIND, AND THAT TAKES WORK HERE. The clips come from Ego-Exo4D, which
ships keystep annotations for every take -- including the `step_unique_id`
sequence the pre-screen reads. Those annotations are exactly the answer to the
coarse-labelling task, so nothing derived from them goes in the packet: not the
keystep count, not the scenario's pre-screen yield, not the window's predicted
G-branch. The manifest carries a video id and a duration, and the video id is
`<take_name>_w<start>` because the clipper already named the files that way.

WHY CHECKSUMS. The first delivery of this project's packets went out with
OpenCV's default fourcc and every video was a black frame in the browser, so
nobody could fill the sheet. The clipper now verifies each file with ffprobe,
and this records a hash so that "the file I labelled" and "the file I analysed"
can be shown to be the same file after the fact. `manifest_sha256` is the same
digest the labelling page embeds and echoes back in its export, which is what
ties an export to a batch.

GRAIN INSTRUCTION IS IN THE README ON PURPOSE. Batch 1 produced 0 G-branch at
a 3-states-per-3-minutes grain while the dataset's own annotation of the same
windows produced 9, so the grain is the binding constraint and leaving it
implicit would repeat the result. See
`docs/branching_pilot_b2_attempt2_3_preregistration.md` §4 -- it is a declared
protocol change, not a correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

README = """Ego-Exo4D 粗标：状态序列与分叉原因（第二批）

打开方法
1. 点“解压全部文件”，把整个 ZIP 解压到普通文件夹。不要在压缩包预览里直接打开。
2. 用标注页面（链接另发）打开后，点页面里的“选择 videos 文件夹”，选中解压出来的
   videos 目录。也可以直接用 Chrome 或 Edge 打开同目录下的“打开标注页面.html”。

状态要定在动作级，这一条这次是硬要求
第一批的状态定得太粗（3 分钟只有 3 个状态），结果一个分叉都测不到。所以这次：
  · 状态是动作级的可观察状态，不是活动级的目标。
    写“把拭子插进试管”，不要写“做检测”；写“翻看说明书”，不要写“准备”。
  · 自检标准：如果一个状态在这条录像里只可能出现一次，它就定得太粗了。

统一规则
1. 同一次动作继续做，包括来回擦、反复拧、重复搅动：不要切段。
2. 同一种动作已经结束，之后重新开始：在真正开始处切开，右边继续选原来的代号。
   回到做过的事，一定要选回原代号，不要新建——新建会让分叉消失。
3. 目标换了：在新目标真正开始处切开，右边新建或选择另一个代号。
4. 空白等待本身不单独建代号；切点放在后面动作真正开始的位置。
5. 同一个代号第二次出现、而且这次之后去了不一样的地方时，页面会让你写一句
   “为什么这次走了另一条路”。请写当时看得见的条件，不要写意图。
6. 每段都要有代号，完整看完后再提交。

交回文件
全部提交后点“导出 JSON”，把导出的内容发回。
页面会把草稿存在当前浏览器；换电脑前先导出备份。
不要查看另一位标注者的答案，也不要讨论逐条判断。
"""


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="egoexo_cut_clips 的 manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-id", dest="batch", default="egoexo_b2_antigen")
    ap.add_argument("--page", help="可选：离线标注页面 html，复制进包里")
    a = ap.parse_args()

    doc = json.load(open(a.manifest, encoding="utf-8"))
    vids = doc.get("videos") or []
    if not vids:
        raise SystemExit("manifest 里没有 videos；切片那一步没成功")

    vdir = os.path.join(a.out, "videos")
    os.makedirs(vdir, exist_ok=True)
    rows, sums = [], []
    for v in sorted(vids, key=lambda x: x["video_id"]):
        dst = os.path.join(vdir, v["video_id"] + ".mp4")
        if os.path.abspath(v["video"]) != os.path.abspath(dst):
            shutil.copy2(v["video"], dst)
        d = sha256(dst)
        sums.append("%s  videos/%s.mp4" % (d, v["video_id"]))
        # 只有 id 和时长。keystep 数、scenario 产出、预测分叉都不进包。
        rows.append({"video_id": v["video_id"],
                     "duration_s": round(float(v["duration_s"]), 3),
                     "sha256": d})

    body = json.dumps({"batch_id": a.batch, "n_videos": len(rows),
                       "videos": rows}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    mdigest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    json.dump({"batch_id": a.batch, "n_videos": len(rows),
               "manifest_sha256": mdigest, "videos": rows},
              open(os.path.join(a.out, "packet_manifest.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    open(os.path.join(a.out, "SHA256SUMS.txt"), "w",
         encoding="utf-8").write("\n".join(sums) + "\n")
    open(os.path.join(a.out, "README_先看这里.txt"), "w",
         encoding="utf-8").write(README)
    if a.page and os.path.exists(a.page):
        shutil.copy2(a.page, os.path.join(a.out, "打开标注页面.html"))

    total = sum(os.path.getsize(os.path.join(vdir, r["video_id"] + ".mp4"))
                for r in rows)
    print("包 %s" % a.out)
    print("  视频 %d 个，合计 %.1f MB" % (len(rows), total / 1e6))
    print("  时长 中位 %.0fs / 最短 %.0fs / 最长 %.0fs"
          % (sorted(r["duration_s"] for r in rows)[len(rows) // 2],
             min(r["duration_s"] for r in rows),
             max(r["duration_s"] for r in rows)))
    print("  manifest_sha256 %s" % mdigest[:12])
    print("  标注页面要用这个 batch_id 和 manifest：%s" % a.batch)


if __name__ == "__main__":
    main()
