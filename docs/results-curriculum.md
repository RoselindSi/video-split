# 课程训练结果存档(Time-R1 线,14 held-out)

> approach B 多段分割课程的逐阶段 held-out 评测。基线锚点见 [[timer1-zeroshot-baseline]]。

## 四方对比(base / Stage1 / Stage2 / Stage3)

| metric | base | Stage1 (iou+format) | Stage2 (+name) | **Stage3 (S1+seq)** |
|---|---|---|---|---|
| f1@0.3 | 0.348 | 0.458 | 0.467 | **0.485** |
| f1@0.5 | 0.257 | 0.325 | 0.315 | **0.397** |
| f1@0.7 | 0.141 | 0.166 | 0.128 | **0.232** |
| mean_iou_matched | 0.452 | 0.509 | 0.413 | **0.548** |
| boundary_score | 0.215 | 0.318 | 0.296 | **0.331** |
| coverage | 0.889 | 0.911 | 0.816 | **1.000** |
| overlap | 0.067 | 0.000 | 0.143 | **0.000** |
| count_acc | 0.564 | 0.462 | 0.560 | 0.533 |
| format_ok | 1.000 | 1.000 | 0.857 | **1.000** |
| name_sim_matched | 0.366 | 0.342 | 0.314 | (未训) |
| mean_pred (gt 4.21) | 4.29 | 6.00 | 4.71 | 3.79 |

**S3 是全程最佳模型**:F1 全阈值、mean_iou、boundary、coverage、overlap、format 均最优。F1@0.5 0.257→0.397(近翻倍),严格 F1@0.7 翻 65%。

## Stage 1:干净成功(边界 + 格式)

- boundary_score 0.215→0.318(+48%)、mean_iou 0.452→0.509、f1 全阈值超基线、overlap 归零、format 稳定 1.0。
- 中间踩过两个坑,都定位并修复:
  1. **LR 太低**:默认 ~1e-6(全参 FT 值)对 LoRA 低约 100 倍,首版训练 merge 后 eval 和基线逐位相同(权重没动)。修:LR→1e-4。
  2. **iou_seg 被 reward-hack**:recall-only(÷|GT|)让模型刷段数套利(count_acc 0.564→0.196、boundary 虚高)。修:分母改 max(|GT|,|pred|),precision-aware。修复后 count_acc 回到 0.462、指标全面超基线。

## Stage 2:回退(命名不升反降)

- **name_sim 三阶段一路跌**:0.366 → 0.342 → 0.314。加了命名奖励,命名反而更差。
- 同时拖累边界:mean_iou 0.509→0.413、boundary 0.318→0.296、overlap 0→0.143、format 1.0→0.857。
- 唯一好处是意外的:count_acc 0.462→0.560(过切收敛,mean_pred 6→4.71)。

### 两层根因

1. **name_seg 被 reward-hack(第二次)**:训练 name_seg_reward 涨(0.13→0.20)但 eval name_sim 跌 —— genericity 惩罚项 `−0.5·max_{k≠j}sim` 被钻空子:模型把名字起得"和别的段不一样"(压低惩罚)而非"更准"(提高主项),产出独特但错误的名字。
2. **更深的瓶颈:视觉识别,不是奖励**。定性看 recording_0005(真值 cotton swab + needle):S1 说 "paper/stick",S2 说 "stick/object/label" —— 模型**认不出小物体**,在瞎编。7B 底座 + 第一视角特写 + 2fps 的视觉能力上限决定了命名天花板。**奖励只能优化模型已能区分的东西;修奖励救不了"看不见"。**

### 结论:命名瓶颈在视觉层,不在奖励层

→ 命名不在 Time-R1 线解决,延后到 **Qwen3-VL**(视觉更强)+ 修好的 name 奖励(去 genericity)。

## Stage 3:干净收官(base=Stage1,reward=iou_seg+seq+format_seg,去 name_seg)

- **S3 全程最佳**:F1@0.5 0.397、F1@0.7 0.232、mean_iou 0.548、boundary 0.331、coverage 1.0、overlap 0、format 1.0。
- **seq 治过切,逐视频铁证**:mean_pred 6.0(S1)→3.79(S3);rec_0016 gt=1 从 S1 的 13 段(F1=0)→ S3 的 1 段(F1=1.0);rec_0051 12→6(F1 0.12→0.80);rec_0023 14→4(0.30→0.60)。
- **两个瑕疵**:①轻微矫枉过正偏欠切(mean_pred 3.79 < gt 4.21,几个视频欠切致 F1 反降,count_acc 0.533 未超基线)→ seq 的 count 权重可再调;②离群点 rec_0017(gt=1 却切 15 段)。

## 结论:Time-R1 线成功收官

课程单调提升 base→S1→S3,F1@0.5 近翻倍、严格 F1@0.7 +65%。多段 GRPO + 分层奖励 + 课程式训练在 7B 上验证有效。命名受视觉瓶颈所限,延后到 Qwen3-VL。下一步:迁移 Qwen3-VL-8B + DDP,复用全部代码/reward/数据,只换底座与环境(见 [[base-model-migration-qwen3vl]])。

## 贯穿全程的方法论洞察

**GRPO 会钻一切奖励后门**(iou 刷段数、name 刷 genericity)—— 这本身证明它在认真优化;每次"训练分涨、eval 分跌"的裂口都是 reward-hack 的信号。可验证奖励的设计必须同时堵 precision/recall/genericity 各个后门。
