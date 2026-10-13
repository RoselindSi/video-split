# Graph Schema v2（草案冻结候选）

对应实验计划 §8.2–8.5 / §9.1 / §10.1 P0。2026-09-24。

本文件只规定**表示**与**字段**。它不规定模型、不规定指标，也不对 v1 的任何统计量作等价换算——
v1 的 161 次 exact state repeat、171 次组内转移、9 次跨组转移属于 v1 合同口径，
改动 self-loop 与 Task Cluster 语义之后必须另行转换与审计（见 `v1_to_v2_mapping.md`）。

---

## 0. 三个必须保持可区分的概念

这是整个 schema 存在的理由，写在最前面。

```
A → B → C → A          同一任务内的流程重复      node recurrence
Task1 → Task2 → Task1   任务离开后回访            task recurrence
```

两者都是环，语义不同。**Cluster 的作用是帮助区分这些结构，而不是规定「出现环就属于同一 episode」。**

第三个概念是 Episode Instance：同一任务的一次具体执行。它与前两者都不等价：

- node revisit ≠ same episode
- node self-loop ≠ new episode
- multi-node cycle ≠ same task
- cluster membership ≠ 当前 episode instance

实例更新必须另有 completion / reset / restart 证据；证据不足标 `uncertain`，不得由拓扑推断。

---

## 1. 实体定义

### 1.1 Interaction Node

录像内部可复用的**局部交互状态**。不要求全局动作 taxonomy。
匹配由**动作、物体、可见结果**共同支持——三者不是任选其一。

```
node_id                  录像内局部匿名 ID（比较前做置换不变映射）
observable_description   只描述可见内容，不写意图、不写归属
object                   被作用的具体物体；无则不成立为 Node
interaction              施加于该物体的动作
visible_result           该动作在物体或场景上留下的可观察变化
evidence                 支持上述四项的时间区间与画面依据
confidence               high | medium | low
```

**准入判据**：视频中能观察到以具体物体为中心的主动取得、运输、交付或放置过程，
且能识别一个有意义的交互状态或结果。

判据落在**可观察证据**上，不落在标注者有没有把结果写出来。
否则同一个视觉事实会因为描述详略不同得到不同的本体身份，而这正是 v1 的一处缺陷
（`0022/D` 与 `0026/D` 在 v1 里描述形式相同，视频证据却不同）。

### 1.2 Special State

等待、跟随、巡视、一般位移、遮挡、不可见、idle。
**保留为带时间的状态，不删除、不拼接。**

```
special_id
kind          waiting | following | patrolling | locomotion | occluded | idle | not_started
[t_start, t_end]
```

范围限于本研究界定的交互状态；标为 special **不表示**这些行为在一般视频理解中没有语义。

### 1.3 Uncertain

可以观察到携带物体或位移，但**无法根据当前窗口**判断它是独立的目标导向交互状态，
还是其他任务中的附带动作。

`uncertain` 是一个一等取值，不是缺失值。它参与时间轴，不参与需要确定身份的统计，
并在所有汇总里单独披露分母。

### 1.4 Trajectory 与 Edge

**Trajectory** 是实际观测到的状态出现序列，**包含 special state**。
聚合邻接图不能取代轨迹。

```
occurrence_id; [t_start, t_end]; node_id | special_id; is_special
```

**Edge** 是相邻两次观测之间的转移出现：

```
edge_id; from_occurrence; to_occurrence; edge_time; transition_kind
transition_kind: self_transition | node_transition | intra_cluster | cross_cluster
```

> **Direct transition** requires an observed transition between two interaction
> states with no intervening special-state gap. **Task revisit** is defined over
> temporally ordered task occurrences, and an intervening special state does not
> break it.

两者可以同时成立。形式化：

```
timeline:  T1 → [SPECIAL] → T2 → T1

direct edge   T1→T2   不成立   （跨越 special 间隔，禁止合成）
direct edge   T2→T1   成立
task revisit  T1      成立     （T1 在时间序上再次出现）
```

**禁止**通过删除 special 节点再拼接剩余节点来计算转移。完整时间轴与 special 间隔必须保留，
转移由「相邻且两端都非 special」判定。

### 1.5 Task Cluster

共享一个**直接任务目标**的节点集合。允许单节点、多节点、一次性线性流程或循环。

```
cluster_id; member_nodes[]; goal_evidence; confidence
```

- 优先使用**可见的任务上下文**作为归属证据。
- **不把 cycle 等价成同任务**：`T1→T2→T1` 是任务回访，不表示 T1 与 T2 合并。
- `A→B→C` 只执行一次也成立为 Task Cluster，不要求回到 A。
- 等待与跟随不构成任务目标，因此不构成 Cluster。

### 1.6 Episode Instance（探索性）

同一任务的一次具体执行 / 恢复片段。当前**不强迫唯一粒度**。

```
instance_id（可空）; evidence_of_completion | evidence_of_reset; uncertain
```

---

## 2. 重复、循环、返回与新任务

| 情形 | 记法 | 禁止 |
|---|---|---|
| 来回擦桌子 | A 占据一段并带 `within_node_repeat=true`；若观察合同按步切分则记 A→A | A→A 不得自动记为新 episode |
| 清洗塞子 | A 拿出→B 冲洗→C 放回 ∈ T1；再次拿出则 A→B→C→A | 不得要求回到 A 才算任务完成 |
| 炒菜→扫地→炒菜 | T1→T2→T1，任务回访 | 不得因此把 T1 与 T2 归为同簇 |
| 新交互 | 始终允许 `NEW_NODE` / `NEW_CLUSTER` | 已有边只作可修正的历史先验，不禁止未观察过的转移 |
| 不可见 / idle | 保留为带时间的 special state | 不得跨其间隔凭空合成直接边 |

---

## 3. 无泄漏与快照

- 时间单位为秒。每条视频使用局部匿名 Node ID，比较前做置换不变映射。
- 对每个测试点 `t`，只允许输入截至 `t` 之前已观察到的 node、edge、cluster；
  `t` 处的新观测由视觉模块提供。
- 整条视频的最终人工图**仅用于离线评分**。
- 每个测试决策时刻另存 `prefix-only graph snapshot` 与模型可见字段。
- 跨视频模板属于另一项实验，必须用开发集学习并报告跨来源划分。

## 4. 必存字段

```
video_id / source_group / contract_version / annotator_id
人工边界、node occurrences、节点描述、逐时轨迹（含 special）、带时间的边
task clusters、可选 episode instance、uncertain 与 visibility
```

---

## 5. 本草案已知的未决项

1. **运输颗粒度。** `0002/C` 把空手位移、取得、搬回三件事打包成一个 38.7s 的状态。
   判据允许它成立，但没有规定这类复合状态是否应当拆分。E0 独立试标专门检验这一点。
2. **跨任务证据全部来自一个批次。** 见 `SELECTION.md`：重复动作批 0 条、马赛克补集 4 条。
   任何据此写下的跨任务规则都只来自一个批次。
3. **Episode Instance 粒度未冻结**，当前为探索性字段。
