# V1 — 交付清单

冻结版本：**手部 V1 + 人脸 F0**。端到端审计（29 段随机窗口）测的就是这一版；
此后测量过的每一项都**没有**进入这一版。

---

## 运行

```bash
python -m src.rig.demo_video \
  --databag <databag 目录> \
  --start 0 --n 1000 --stride 1 \
  --weights /shared/models/HaWoR/weights/external/detector.pt \
  --clf_ctx  <own_ctx_best.pt> \
  --geom     <geom_inv2.json> \
  --face_model <yolov8n-face-lindevs.onnx> \
  --panorama baseline \
  --out out.mp4
```

`--panorama baseline` 不可省。`depth` 那条渲染分支在仓库里还在，但它不属于 V1。

## 运行时文件（18 个，`demo_video.run` 的导入闭包）

```
demo_video.py        编排：读帧 → 渲染 → 检测 → 跟踪 → 归属 → 掩码 → 抑制
calibration.py       标定读取
geometry.py          虚拟宽相机、source_maps
render_wide.py       六路合成全景（baseline 选择式，非融合）
seam_fix.py          ClipReader / Prefetch
hand_detect.py       检测（IMGSZ 1024）、forearm_exit 规则、OwnHold 迟滞
hand_track.py        Tracker、FlipCount、Fragmentation
own_ctx.py           归属分类器 V1（手 + 上下文窗口 + 几何）
geom_prior.py        几何先验
suppress_other.py    掩码与打码
face_mask.py         人脸检测 F0、drop_on_hands、Hold、cover
own_cnn.py           旧版仅手部分类器（`--clf` 才用，V1 不用）
panorama.py          depth-aware 渲染（`--panorama depth` 才用，V1 不用）
depth.py wide_depth.py   同上，被 panorama.py 拉进来
own_label.py class2_census.py near_other_miner.py   标注/采样工具，被惰性导入
```

**真正在 V1 路径上跑的是前 11 个。** 后 7 个在闭包里是因为惰性导入，
V1 的参数组合下不会执行到。

`src/rig` 下其余 51 个 `.py` 全是测量与诊断工具，不参与运行。

## 冻结常数（从代码读出，不是凭记忆）

```
手部
  hand_detect.IMGSZ                1024      512→1024 验证过两次
  hand_detect.MIN_CONF             0.60      模块默认；demo_video 用 0.25 覆盖
  hand_detect.OWNER_EXIT_Y_FRAC    0.55      出口高度规则
  hand_track.GATE_FRAC             0.15
  hand_track.MAX_LOST              5
  hand_track.MAX_ASSOC_COST        1.60
  hand_track.UNMATCHED_COST        1.35
  demo_video.NEW_TRACK_CONF        0.60      新建轨迹
  demo_video.CONTINUE_TRACK_CONF   0.25      继续轨迹，同时是检测下限
  demo_video.MAX_PREDICTION_AGE    2

人脸
  face_mask.MIN_CONF               0.35      从 0.60 降下来（0.60 漏 66/140）
  face_mask.MAX_FACE_FRAC          None      尺寸上限已移除（曾误拒近处同事）
  face_mask.HAND_OVERLAP_VETO      0.45
  face_mask.HOLD_FRAMES            12
  face_mask.PAD                    0.35
  face_mask.BLOCK_FRAC             0.18
```

## 端到端表现（29 段随机 33 秒窗口，人工盲审）

```
29 段   5580 个采样对   15.5 分钟

漏了脸         20 事件 /  385 帧    77.4/小时
漏了别人的手    40 事件 /  710 帧   154.8/小时
误糊自己的手    76 事件 / 1290 帧   294.2/小时
误糊非脸非手   112 事件 / 2095 帧   433.5/小时

至少一次隐私关键漏检的窗口   13/29 = 0.448 [0.284, 0.625]
```

盲审看不到任何模型输出，所以它能看见「系统根本没提出」的漏检——
候选层的指标（人脸候选召回 1.000、手部轨迹召回 0.833）全都以「提出过」为条件，
两组数不可互换。

**审计本身有已知标注噪声**：偶然发现并修正过 25 处误标（假阳性、假阴性、
类别方向错各有）。这些率的真实区间比 Wilson 给的宽，未测量。

## 已知缺陷与归因（都已定位到代码）

```
漏脸 93.5%   人脸检测器根本没提出
             face_mask.py:210  SIDE=640 静态输入 + 1.78:1 硬拉（非 letterbox）
             ONNX 输入固定，960/1280 都在 reshape 处报错

漏他手       detector 43.4% / ownership 30.1% / admission 26.5% / mask 0%
             每段录像的失败机制不同，无单一修法

误覆盖       residue 44.6% / hand_oth 41.3% / face_fp 12.0% / hand_fp 2.2%
             residue 已归因：163 个事件回溯，零个凭空出现，
             脸占 88%（53 vs 7），全部落在 HOLD_FRAMES=12 内
             → 一次单帧人脸误检 = 0.4 秒马赛克
```

## 本次会话新增、默认关闭，不改变 V1 行为

```
face_mask.split_on_hands        drop_on_hands 改为调用它，语义不变
face_mask.detect_faces(tiles=)  切块，默认 None（关闭）
face_mask._YoloFace.detect_tiled 同上
hand_detect.OwnHold.last_pre_cap 仅记录 cap 之前的状态，类内无人读取
demo_video.run 返回值            增加第 4 项 flips.report()，唯一调用点已同步
```

自测：`hand_detect` 36 项 / `face_mask` 19 / `hand_track` 16 / `own_ctx` 9 /
`demo_video` 15，**全部通过，零 FAIL**。

## 测过但未采纳（不在这一版里）

```
人脸切块        候选层召回 +48.3% [0.359, 0.608]，40 随机帧零新增误检
                未接入：需先做端到端验证
有界插值 T2     误检糊帧 −67%，但新增漏脸 27.3 帧/分钟 vs 现有 24.8 —— 会翻倍
                否决；正确顺序是先补检测器召回再缩 hold
track 级归属聚合 修好 514 帧 / 弄坏 679 帧，净 −165，帧级一致率还降 0.008
                否决
handedness      整体纯净轨迹率 0.70，关键格 n=5 —— INCONCLUSIVE，非负结果
关系否决 / pose  signal exists, deployment rule not validated
人脸验证器 F1-F3 全部净负于 F0
```
