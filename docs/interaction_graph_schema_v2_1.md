# Graph Schema v2.1

2026-09-26。取代 `interaction_graph_schema_v2.md` 作为生效合同；v2 与 11 条原始标注、E0 分析全部保留不覆盖。

本版只做一件事：**消除 E0 试标暴露出的、可避免的标注自由度**，同时不牺牲我们要研究的真实 recurrence 与 task structure。
验收目标**不是**把边界 F1 从 0.508 推到某个阈值——那可以靠「所有重复一律并成一段」买到，
而那样买来的一致性会把 Graph Recurrence 这个研究对象本身抹掉。

---

## 0 三个必须保持可区分的概念（不变）

```
A → B → C → A            同一任务内的流程重复      node recurrence
Task1 → Task2 → Task1     任务离开后回访            task recurrence
                          同一任务的一次具体执行     episode instance
```

- node revisit ≠ same episode
- node self-loop ≠ new episode
- multi-node cycle ≠ same task
- cluster membership ≠ 当前 episode instance

---

## 1 重复：occurrence 规则（v2.1 新增，取代 §8.3① 的二选一）

v2 同时允许「一段带反复标记」和「逐轮切成 A→A」，没规定选哪个。
E0 里 44 个丢失的 v1 边界有 **18 个（41%）**落在标注者标了反复的跨度内——这是合同的口子，不是人的分歧。

**规则：**

> 同一交互状态内部的连续重复，**不产生新的时间段**；
> 只有存在**可观察的完成、重置或重新开始证据**时，才允许产生同节点的新 occurrence。

两种 recurrence 因此各有唯一记法：

| 类型 | 记法 | 例 |
|---|---|---|
| **Continuous repetition** | 一个 occurrence + `within_node_repetition = true` | 来回擦桌子 40 秒 |
| **Discrete state recurrence** | `A → A`（或 `A → B → C → A`） | 洗完杯子 1 → 拿起杯子 2 → 洗杯子 2 |

判别只看一件事：**两次之间有没有可观察的完成 / 重置 / 重新开始。**

```
擦桌子来回换方向          不是完成，是同一次动作的内部往返  →  一段 + repetition
洗完一个杯子放下、拿起下一个  放下是完成，拿起是重新开始        →  A → A
```

**标注者不得自由选择。** 判不出有没有完成证据时标 `uncertain`，不要默认合并也不要默认切开。

> 注：E0 那 18 个界只是**候选归因**。落在反复跨度内也可能是标注者漏掉了一次真实的重新开始。
> 逐条回看视频属于 v2.1 重标后的复核项，不在本合同里预判。

---

## 2 交互节点的字段（v2.1 拆分）

v2 的单一 `visible_result` 让标注者把语义解释当成可观察结果写了进去
（`0010/A`：`visible_result = "选择到合适的摩托车"`——那是推测）。拆成四项：

```
observed_action           看到的动作。只写画面里发生了什么。
observed_result           看到的结果。看不到就写「无可观察结果」，不要留空、不要编。
inferred_goal             你推测的目的。可留空。辅助信息。
evidence_timestamp        支持上面两项的时间点（秒）。
```

**`inferred_goal` 不能作为 Interaction Node 成立的依据。**
一个状态成立与否只看 `object` + `observed_action` + `observed_result`。

示例（就是 `0010/A` 该有的样子）：

```
observed_action    手持手机在电动车之间走动
observed_result    无可观察的挑选或确定动作
inferred_goal      可能在挑选电动车
→ 无 observed_result  →  不成立为交互节点  →  special_state / patrolling
```

**工具只能校验字段完整性、时间戳和证据引用。**
关键词或 JSON schema 无法可靠识别推测性表述，语义真实性仍需人工审计。

---

## 3 运输：三条可检查的视觉证据（v2.1 明确）

v2 只写了「有目的 / 没目的」，结果 `0026/D` 和 `0022/D` 被标成了同一个东西。改成三条各自可查：

| 证据 | 要观察什么 |
|---|---|
| **Object identity** | 能不能确定被运输的物体 |
| **Transport event** | 能不能观察到主动取得、移动、交付或放置 |
| **Observable outcome** | 有没有看到送达、放置、移交，或进入后续操作 |

```
三条都有            →  interaction_node
前两条有、第三条无    →  uncertain
物体不明或纯位移      →  special_state
```

落到三个案例：

- **`0026/D` 携带杯子**：杯子被带到秤旁并进入称量操作 → 三条齐 → **interaction_node**
- **`0022/D` 携带茶叶**：只看到携带行走，视频在途中结束，无法判断是独立运输还是附带动作 → **uncertain**
- **`0002/C` 取凳子**：按实际可观察证据，**允许把空手位移与取得凳子后的运输拆开**

### 3.1 在线证据 vs 回顾性证据

离线人工标注能用整条视频里**后来**出现的证据；Online Graph Tracking 在时刻 `t` **看不到 `t` 之后的信息**。
`0026/D` 正是靠「后来用它称重」才成立的——如果不记下来，人工图会用未来结果确定一个运输节点，而模型在线预测时根本没有那个结果。

因此每个交互节点必须记：

```
evidence_timestamp           支持判断的证据出现在第几秒
retrospective_confirmation   true = 该判断依赖了本段结束之后才出现的证据
```

`retrospective_confirmation = true` 的节点在 E4 在线实验里**必须单独分层报告**，不能和其余节点混算。

### 3.2 两种证据范围可以给出不同的表示，不强行压成唯一真值

E0 逐帧核对三条运输案例之后发现：分歧的来源不是规则，是**证据窗口**。
`0026/D` 携带杯子那一段，段内只看得到「拿着杯子走到一张桌子前」；
它成为「运送称重容器」是靠段尾之后才出现的画面确立的。

因此 v2.1 不要求把这类案例压成一个答案：

```
prefix evidence        只用截至该段结束为止看得到的东西
retrospective evidence 允许用整条视频，包括该段之后

两者给出不同判定时，两个都记，不必二选一。
```

离线的 E1–E3 可以使用并额外分析 retrospective 证据；
**E4 的评分必须是 prefix-only**，它的 gold 只取 prefix 那一栏。
`0022/D` 属于两种范围都拿不到结果证据的情形（视频在段中结束），保留为 unresolved exemplar。

---

## 4 特殊状态：持续等待不得并入相邻节点（v2.1 明确）

`mosaic_graph_0020` 有 21 秒的等待同事，标注者把它并进了前一个交互节点 A，
于是 `waiting` 在 11 条里零次使用，而「直接边 vs 任务回访」这条区分从未被触发。

**区分两种停顿：**

| | 记法 |
|---|---|
| 交互内部的**短暂停顿**（换手、看一眼、调整姿势） | 留在该交互节点内，不切段 |
| **持续、可辨认的独立等待**（等他人完成、等设备、等叫号） | **必须单独切成 `special_state / waiting`，不允许并入相邻交互节点** |

「可辨认」的判据：这段时间里**佩戴者没有对任何物体施加动作**，且该状态有明确的起止。

其余 kind（`following` / `patrolling` / `locomotion` / `occluded` / `idle` / `not_started`）规则同 v2。

### 4.1 直接边与任务回访（不变，但复述一遍因为它一次都没被触发过）

```
时间轴   T1 → [SPECIAL] → T2 → T1

direct edge   T1→T2   不成立   跨越 special 间隔，禁止合成
direct edge   T2→T1   成立
task revisit  T1      成立     T1 在时间序上再次出现
```

> Task revisit is defined over temporally ordered task occurrences, whereas
> direct graph edges require an observed transition without an intervening
> special-state gap.

**完整时间轴与 special 间隔必须保留**，不得通过删除节点再拼接剩余节点来计算转移。

---

## 5 提交校验（v2.1 收紧）

v2 只检查被段引用到的事件和簇，孤儿能通过：E0 交回的 11 条里有 2 个全空事件、2 个无依据的空簇。

提交前必须全部通过：

```
1. 每段都有 seg_type
2. special_state 必须有 kind
3. uncertain 必须有说明（看到了什么、缺了什么）
4. 被引用的事件：object / observed_action / observed_result / evidence_timestamp 齐全
5. 被引用的事件必须归到某个任务，该任务必须有 goal_evidence
6. 【新】没有未被任何段引用的事件 —— 删掉或显式标记为弃用
7. 【新】没有零有效成员的任务 —— 删掉
8. 从头到尾看完
```

`inferred_goal` 可留空，不参与校验。

---

## 6 实体定义（相对 v2 的增量）

其余定义同 v2。变更的字段：

```
Interaction Node
  node_id; object; interaction
  observed_action; observed_result; inferred_goal
  evidence_timestamp; retrospective_confirmation
  within_node_repetition; confidence

Special State
  special_id; kind; [t_start, t_end]
  kind ∈ waiting | following | patrolling | locomotion | occluded | idle | not_started

Uncertain
  一等取值。参与时间轴，不参与需要确定身份的统计，所有汇总里单独披露分母。
  必须写说明。
```

---

## 7 本版已知的未决项

1. **运输复合状态的拆分粒度**：允许拆，但没规定拆到多细。
2. **跨任务证据仍全部来自马赛克补集批次**（重复动作批 0 条），两批转移速度相差三倍。
3. **Episode Instance 粒度未冻结**，仍为探索性字段。
4. **reviewer_2 对这 11 条已不盲**。下一轮用这 11 条只能验证规则是否消除了已知分歧，
   **不能当作独立盲标验证**；真正的 P1 双人一致性需要两位标注者都按同一冻结合同独立标注的材料。
