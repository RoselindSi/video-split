# E1 设计：Qwen 能力诊断

2026-09-26。对应实验计划 §9.2。E0 已冻结（`results/auditor/interaction_graph_schema_v2/E0_FROZEN.md`）。

E1 要把端到端 F1 0.138 这个结果拆开，分辨三件事：

```
看不出视觉语义？          → E1a
看得出语义但不会判 relation？ → E1b
relation 会但找不到边界？    → E1b 对 E1c 的差
```

---

## 候选集（已算出，来自冻结的 v1 release + v2.1 节点映射）

**307 个候选边界，分布在 49 条视频。** 两端都非特殊状态——v2.1 §4.1 禁止跨 special 间隔合成边，所以那些相邻对不构成候选。

| | | 多数类基线 |
|---|---|---:|
| **state_relation** | DIFFERENT_STATE 174 (56.7%) / SAME_STATE 133 (43.3%) | **56.7%** |
| **task_relation** | SAME_TASK 302 (98.4%) / DIFFERENT_TASK 3 / RETURN_TASK 2 | **98.4%** |

### 这个分布决定了两条臂必须分开处理

**state_relation 可测。** 43/57 的分布，多数类基线 56.7%，宏平均 F1 有意义，307 个样本够。

**task_relation 在这批数据上不可测。** 一个永远回答 SAME_TASK 的模型拿 98.4%。
跨任务只有 5 个正例（3 个 DIFFERENT_TASK + 2 个 RETURN_TASK），**任何准确率数字都不能作为 task 能力的证据**。

处置：

- task_relation **不报总体准确率**，只报 per-class recall，并把 n=3 / n=2 明确写在旁边。
- 它在 E1 里的作用降级为**可行性探针**：看模型会不会在该说 DIFFERENT 的地方说 DIFFERENT，而不是量化它多准。
- 真正的 task 能力测量要等 E3，而 E3 需要先富集跨任务样本——这是一个已知的、必须单独解决的采样问题（§9.4 已声明）。

**不要因为 task_relation 数字难看就去改判据**；它难看是因为这批开发视频里跨任务本来就稀少。

---

## E1a — 视觉语义识别（已执行，2026-09-26）

### 已冻结的东西

```
总体      378 个交互段 / 56 条视频（60 条里 4 条全是特殊状态）
          时长三分位切点 4.5s / 12.0s
样本      代表性 80（batch × 时长三分位 分层，种子 20260926）
          困难   20（四类各 5：运输位移 / 末段截断 / 长时重复 / v1 标 medium 的物体）
          在任何模型输出存在之前冻结 -> sample_frozen.json
模型      Qwen3-VL-8B-Instruct，1 fps，temperature 0
提示词    中文，与五折基线同语言；基线规则 5「不猜测看不到的意图」原样借用
输入      单段独立切片（重编码，不用 -c copy，避免关键帧对齐把窗口放宽一个 GOP）
          无相邻段、无节点标签、无簇、无 v1 描述、无图、无段后证据
```

### 模型侧结果（全量 378，未经人工核验）

```
解析成功        378 / 378      零格式失败（五折那轮 30 条里有 2 条 JSON 截断）
observed_result 非 null  322 (85.2%)
observed_result 为 null   56 (14.8%)
object 非 null           372 (98.4%)
uncertain = true           0 ( 0.0%)
```

null 率按时长**不单调**，呈 U 形：

```
<4.5s     32.3%
4.5-12s    2.4%   ← 最低
>12s       8.8%
```

若「看不见结果」主要由证据不足驱动，应随时长单调下降。中段 123 个里只有 3 个说没看见，
更像「片子够长就总能写出一个结果」。**但这只是分布形状，真假由审计的 `result_status` 决定。**

### 评分口径（在审计表还是空白时定死）

| 指标 | 定义 |
|---|---|
| 物体 / 动作 acceptable | `correct + partial` |
| 物体 / 动作 strict | `correct` only |
| joint usable | 物体与动作同时 acceptable |
| **结果 precision** | `correct / (correct + unsupported)`，**`not_observable` 不进分母** |
| 臆测率 | `unsupported_inference = yes` 的比例 |
| 整体可用 | `overall_usable = yes` 的比例 |

`not_observable` 是判对而不是弃权；把它算进分母会让模型靠多写 null 刷精度，
而 `missing`（该写没写）是反向约束，必须与 precision 并列报告。

**两组永远不合并**（`e1a_score.py` 里没有合并路径）。困难 20 段按构造富集，只描述自身，不给 CI。
置信区间按**视频**聚类 bootstrap，不按段——同一录像的段共享场景、操作者与相机。

### 本轮算不出来的两项

`Abstention rate` 与 `Accuracy when non-uncertain` 需要模型的 uncertain 字段，
而全量 378 段该字段为 **0**。**这个 0 本身是结果，不是缺测**，脚本会如实打印。

### E1a 是诊断不是门槛

不设「E1a ≥ X% 才允许继续」。即使自动语义标注不够好，E1b 仍然值得跑，只是解释方式不同——
见下面 E1b 的双条件设计。

## E1a 原始设计

固定人工时间窗口，模型只输出这一段里看到了什么，**不做任何 relation 判断**。

```
输入   该段的视频窗口（1 fps，与五折基线一致）
输出   {"object": ..., "observed_action": ..., "observed_result": ...,
        "uncertain": true|false}
```

- 提示词里不出现该视频的任何人工标注。
- 输出字段刻意与 v2.1 的交互节点字段同名，这样人工核验可以直接对照 E0 的判据。
- 主要结果：**人工核验的语义正确率**与 **uncertain 率**，按视频聚类。

**E1a 先跑、先看。** §9.2 的诊断分支写明：若 E1a 已差，后续 relation 错误不能全归因于 Graph。

## E1b — Oracle 边界上的 relation

**两个条件，不是一个。** 这样 E1a 的成绩无论好坏都能被解释，而不需要拿它当门槛：

| 条件 | 输入 | 隔离的问题 |
|---|---|---|
| **E1b-video** | 候选点前后的视频窗口 | 端到端的 relation 能力 |
| **E1b-oracle-text** | **人工**写的语义描述，不给视频 | 去掉感知之后，relation 推理本身行不行 |

三种组合各自指向一个不同的瓶颈：

```
E1a 差  +  oracle-text 好                  → 瓶颈在感知 / 语义抽取
E1a 好  +  video 差  +  oracle-text 也差    → 瓶颈在 relation 推理本身
E1a 好  +  E1b 好    +  旧端到端差          → 瓶颈在边界发现 / 全局生成
```

比「E1a 达到 80% 才准往下走」这种人为阈值科学得多。

给定已确认的候选点与其前后窗口：

```
输出   {"state_relation": "SAME_STATE" | "DIFFERENT_STATE" | "UNCERTAIN",
        "task_relation":  "SAME_TASK" | "RETURN_TASK" | "DIFFERENT_TASK" | "UNCERTAIN"}
```

- **关系判断与边界定位分开评分**：这里边界是给定的，模型不需要找。
- 按 video / source_group 聚类做配对 bootstrap，不把同一视频的多个边界当独立样本。
- state 报宏平均 F1 + 混淆矩阵；task 只报 per-class recall + 分母。

## E1c — 边界定位

冻结原端到端设置（1 fps，temperature 0，两遍检查），与 E1b 对照，把失败拆成：

```
没找到时间点  /  找到了但关系判错  /  自由生成时合并概括
```

容差 **1.0 s**（项目硬规则，不是 0.5）。

---

## 不能做的事

- **不拿 E1 的数字去支持 Graph 有用或没用。** E1 测的是模型的底层能力，Graph 的收益是 E2 的事。
- **不用 E1a 的输出当 gold。** §8.4：Qwen 的标签不能自动成为未经核验的 gold。
- **不混用 prefix 与 retrospective 证据。** E1 是离线的，可以用整段窗口；E4 不行（v2.1 §3.2）。

## 模型与配置

```
/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct     与五折基线同一个起点
1 fps, temperature 0                           原基线配置
```

必要时在开发集上做 1/2 fps 灵敏度检查，但**帧率变化的收益不能算进 Graph 的账**。
