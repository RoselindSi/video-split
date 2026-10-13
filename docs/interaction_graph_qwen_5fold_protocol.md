# Qwen 五折盲测协议

## 目的

测量 Qwen 是否能从完整视频复现人工时间轴动作图，重点区分同一 episode
内部重复、同类新实例和真正的任务转移。Qwen 是机器标注者，不是第二位人工
标注者；结果不能作为人工一致性。

## 冻结输入

- 30 条人工最终标注：`results/auditor/interaction_graph_blind30_v2/human/reviewer_1_final.json`
- 原始盲选 manifest：`results/auditor/interaction_graph_blind30_v2/manifest.json`
- 折分只使用盲选时已有的 `low/medium/high` 马赛克关键帧层，不使用动作标签。
- 每折 6 条测试视频，每层 2 条；其余 24 条仅用于选择最多 8 个结构示例。
- 目标视频的人工事件、边界、描述和备注不得进入该视频的 prompt。

## 模型任务

Qwen观看完整目标视频并输出与人工网站相同的 `events` 和 `segments` JSON。
第一遍独立建立时间轴，第二遍重新观看并检查漏切、过切、节点复用和段内重复。
温度固定为 0。默认视频采样为 1 fps，边界精度由人工金标准评估。

## 主要指标

- 边界 micro-F1：0.25、0.5、1.0、2.0 秒容差。
- 1秒内匹配边界上的 `SAME_ACTION_NEW_INSTANCE` 与 `NEW_ACTION` 一致率。
- 0.5秒时间网格上的事件类型分区 ARI，事件字母通过分区比较自动消除置换。
- `repeats_within_segment` 的时间一致率与 Cohen kappa。
- 每折单独报告，避免一个容易折掩盖困难折。

自由文本事件描述保留供误差审查，但本轮不自动给语义相似度打分。

## 命令

```bash
ROOT=results/auditor/interaction_graph_blind30_v2
RUN=$ROOT/qwen_5fold_v1

python -m src.auditor.boundary.interaction_graph_qwen_cv prepare \
  --manifest $ROOT/manifest.json \
  --gold $ROOT/human/reviewer_1_final.json \
  --out-dir $RUN

# 无模型烟雾测试，不产生研究结果
python -m src.auditor.boundary.interaction_graph_qwen_cv run \
  --backend mock --manifest $ROOT/manifest.json \
  --gold $ROOT/human/reviewer_1_final.json --folds $RUN/folds.json \
  --out-dir $RUN/mock

# GPU 数据机上的真实运行
python -m src.auditor.boundary.interaction_graph_qwen_cv run \
  --backend qwen --model-id /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
  --manifest $ROOT/manifest.json \
  --gold $ROOT/human/reviewer_1_final.json --folds $RUN/folds.json \
  --out-dir $RUN/qwen --video-root /workspace/interaction_graph_blind30_v2/videos

python -m src.auditor.boundary.interaction_graph_qwen_cv score \
  --manifest $ROOT/manifest.json --gold $ROOT/human/reviewer_1_final.json \
  --predictions $RUN/qwen/predictions.json --folds $RUN/folds.json \
  --out-dir $RUN/qwen/score
```
