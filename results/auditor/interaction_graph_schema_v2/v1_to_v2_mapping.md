# v1 → v2 映射与未映射清单

2026-09-24。对应实验计划 §9.1 产出项与 §8「禁止直接沿用为新版验证结果」。

原始 v1 数据全部保留，未改写：`results/auditor/interaction_graph_release_v1/`。
本文件记录**哪些东西变了、变成什么、以及哪些没有对应物**。

---

## 1. 字段映射

| v1 | v2 | 说明 |
|---|---|---|
| `event_id` + `description` | `Interaction Node` | v2 额外要求 `object` / `interaction` / `visible_result` 三项可观察字段 |
| `segments[].repeats_within_segment` | `within_node_repeat`（一次出现的属性） | **不再是边**，见 §2 |
| `edge_occurrences[].kind = self_transition` | `transition_kind = self_transition` | 语义收窄为「状态复现」，不蕴含新实例 |
| `edge_occurrences[].kind = transition` | `node_transition` + `intra_cluster` / `cross_cluster` | 按两端 cluster 归属再分 |
| `procedure_group_id` | `cluster_id` | 要求 `goal_evidence`；等待/跟随不成立 |
| `special_nodes.json` | `Special State` + `kind` | 新增 `waiting` / `following` / `patrolling` / `locomotion` 四种 kind |
| — | `Uncertain` | v1 无对应物，新增一等取值 |
| — | `Episode Instance` | v1 无对应物，探索性字段 |
| — | `Trajectory`（含 special） | v1 的 `event_sequence` 已剔除 special，v2 要求完整时间轴 |

---

## 2. 统计量变化（必须另行转换，禁止沿用）

### 2.1 边

| | v1 发布口径 | v2 重算 | 差异来源 |
|---|---:|---:|---|
| exact state repeat | **161** | **133** | v1 把 28 次「段内往返」也计入；v2 里它是一次出现的属性，不是两次出现之间的边 |
| 组内转移 | 171 | **169** | 2 条一端接在新降级的 special 上，禁止跨间隔合成 |
| 跨组转移 | 9 | **5** | 4 条一端接在等待/跟随上 |
| **合计** | **341** | **307**（+ 28 个 within-node-repeat 属性） | |

**161 ≠ 133 是语义差异不是重算误差。** 341 与 307 不可互换引用。

### 2.2 节点与状态

| | v1 | v2 |
|---|---:|---:|
| 局部节点 | 134 | 130 交互节点 + 4 降为 special |
| 特殊状态段 | 29 / 412 | 34 / 412 |
| uncertain 节点 | — | 1 |

### 2.3 Cluster

| | v1 | v2 |
|---|---:|---:|
| procedure group / cluster | 45 | **43** |
| 跨视频复用 | 8 | **7** |
| 单节点组 | 16 | 16 |
| 只出现在一条视频 | 37 | 36 |

消失的两个组：`pg_person_following`（2 节点 2 视频，**是 8 个复用组之一**）、
`pg_collaboration_waiting`（1 节点 1 视频）。
缩小的一个组：`pg_electric_scooter_arrangement` 2 节点 → 1 节点。

### 2.4 六个结构

| 结构 | v1 | v2 | 变化 |
|---|---:|---:|---|
| 持续重复 | 21 条 / 28 次 | 21 / 28 | — |
| 同节点再次出现 | 18 / 134 | 18 / 134 | — |
| 线性多步骤 | 18 / 18 | 18 / 18 | — |
| 多步骤循环 | 25 / 25 | **24 / 24** | `mosaic_graph_0010` 的 A 降级后不再成环 |
| 跨任务切换 | 4 条 / 9 次 | 4 条 / **5 次** | 录像数不变 |
| 离开后返回 | 2 条 / 2 次 | 2 / 2 | **不变** |

**任务回访不变，是因为它定义在时间序的任务出现上，中间隔一个 special 不打断它。**
被砍掉的是跨越那个间隔的边，不是任务回来了这个事实。

---

## 3. 节点级裁决（依据是视频，不是描述）

| 节点 | v1 | v2 | 视频证据 |
|---|---|---|---|
| `0020/B` 等待他人继续组装 | pg_collaboration_waiting | **special** | 等待他人，不推进任何任务目标 |
| `0006/C` 跟随他人走到厨房另一端 | pg_person_following | **special** | 无被作用物体、无可见结果 |
| `0009/A` 跟随前方人员行走 | pg_person_following | **special** | 同上 |
| `0010/A` 拿着手机穿梭在电动车里 | pg_electric_scooter_arrangement | **special** | t=5/40/100：街边行走，手机屏上是通话人脸——手机在被使用不是被操作，无可见结果 |
| `0002/C` 走路去另一房间取凳子再走回来 | pg_support_item_retrieval | **interaction node** | t=60 空手走出 → t=76 到达另一房间 → t=94 双手抱圆形凳面走回。取得与运输都可观察 |
| `0026/D` 拿着杯子走到隔壁房间 | pg_dried_fruit_weighing | **interaction node** | t=77：走向的桌子上秤在同一帧可见；紧接事件是「调好称」「用杯子当容器在称上称果干」 |
| `0022/D` 拿着塑料包装里的茶叶走路 | pg_tea_portioning | **uncertain** | t=110/115/119：走廊中手持茶叶包，画面无目的地；该段 109.3–120.0s 而视频在 120.0s 结束 |

两条记录必须留下：

- **`0002/C` 的保留理由是视频证据，不是它影响任务回访指标。** 若后续复审发现它不符合统一视觉判据，应当修改。
- **`0026/D` 推翻了 2026-09-24 的初次建议（uncertain）。** 降级的理由只能是证据不足，而这一条证据是足的。
  它与 `0022/D` 在 v1 里描述形式相同，视频证据不同——这正是「判据不能落在描述详略上」的实例。

---

## 4. 未映射清单

| 项目 | 状态 | 原因 |
|---|---|---|
| `Episode Instance` 的粒度 | **未冻结** | v1 不携带 completion / reset 证据，无法从 v1 反推 |
| v1 的 9 个中置信节点 | **未复核** | 仍在 `review_queue.csv`，v2 未处理 |
| 运输类复合状态是否应拆分 | **未决** | `0002/C` 把空手位移、取得、搬回打包成 38.7s 一个状态；判据允许，但没有规定是否拆 |
| `0022/D` 的最终身份 | **uncertain** | 留给 E0 独立试标 |
| 跨视频 cluster 模板 | **不在本合同内** | §8.5 列为另一项实验，须开发集学习并报告跨来源划分 |
| `mosaic_graph_0010` 的标注粒度 | **记录但不修改** | 120s 里 111.4s 是 special，v2 下只剩一个 8.6s 的交互节点 |

---

## 5. 证据边界

- 全部数据来自**单一标注者**的 v1 冻结版，未经人际一致性验证。
- **跨任务切换与任务回访 100% 来自马赛克补集批次**，重复动作批 0 条。
  任何据此写下的跨任务规则只来自一个批次，而两批转移速度相差三倍（3.26 对 0.96 次/分钟）。
- 跨任务 5 次 / 4 条、任务回访 2 次 / 2 条，数量不足以支撑任何强主张（§9.4 已声明）。
