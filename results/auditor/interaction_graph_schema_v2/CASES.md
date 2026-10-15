# E0 八条样本：v1 / v2 对照案例

由 `src/auditor/boundary/interaction_graph_case_sheet.py` 生成，2026-09-24。

读法：`A` 交互节点，`A*` 该段内有往返重复，`[B]` 特殊状态。
直接边要求相邻且两端都非特殊状态；任务回访定义在时间序的任务出现上，中间隔一个特殊状态不打断它。

---

## mosaic_graph_0002（mosaic_complement）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 旋下风扇头 | pg_fan_disassembly | interaction node |
| B | 逐节拆下支撑杆 | pg_fan_disassembly | interaction node |
| C | 走路去另外一个房间取凳子再走回来 | pg_support_item_retrieval | interaction node（kept） |
| D | 拆风扇头 | pg_fan_disassembly | interaction node |

**v1 时间轴**

```
A* → B → B → C → D
```

v2 时间轴与 v1 相同。

**直接边** v1 4 条 → v2 4 条

v2 保留的跨任务边：
- `58.0s  B → C`
- `96.7s  C → D`

**结构** v1：持续重复（段内往返）、同节点再次出现（A→A）、线性多步骤（同簇≥3节点）、跨任务切换、离开后返回原任务

**结构** v2：持续重复（段内往返）、同节点再次出现（A→A）、线性多步骤（同簇≥3节点）、跨任务切换、离开后返回原任务

任务回访成立于 `pg_fan_disassembly`。

> **C 的判断依据（kept）**：看帧 t=60 空手走出、t=76 到达另一房间、t=94 双手抱着圆形凳面走回。取得与运输都可观察，符合 interaction_node 判据。注意：保留的理由是视频证据，不是它影响任务回访指标。

> 遗留歧义：38.7s 的跨度把空手位移、取得、搬回三件事打包成一个状态，颗粒度问题单独记录。

---

## mosaic_graph_0006（mosaic_complement）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 用抹布擦桌面 | pg_surface_cleaning | interaction node |
| B | 整理桌面和架子上的锅盖和盆 | pg_kitchenware_organization | interaction node |
| C | 跟随别人走到厨房另一端 | pg_person_following | **special**（decided） |
| D | 接过盆子，里面装着抹布 | pg_hot_water_collection | interaction node |
| E | 端盆子在热水机下接热水 | pg_hot_water_collection | interaction node |

**v1 时间轴**

```
A* → B* → C → D → E
```

**v2 时间轴**

```
A* → B* → [C] → D → E
```

**直接边** v1 4 条 → v2 2 条

被特殊状态间隔切断的：
- `19.0s  B → C`（cross_cluster）
- `30.5s  C → D`（cross_cluster）

v2 保留的跨任务边：
- `12.6s  A → B`

**结构** v1：持续重复（段内往返）、跨任务切换

**结构** v2：持续重复（段内往返）、跨任务切换

> **C 的判断依据（decided）**：跟随移动，无被作用的物体，无可见结果。

---

## mosaic_graph_0020（mosaic_complement）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 安装设备1 | pg_electrical_assembly | interaction node |
| B | 等待别人安装设备1 | pg_collaboration_waiting | **special**（decided） |
| C | 看手机并在盒子上做记录 | pg_work_documentation | interaction node |
| D | 逐个整理一捆电线的头的裸露部分 | pg_electrical_assembly | interaction node |

**v1 时间轴**

```
A → B → C → D*
```

**v2 时间轴**

```
A → [B] → C → D*
```

**直接边** v1 3 条 → v2 1 条

被特殊状态间隔切断的：
- `33.6s  A → B`（cross_cluster）
- `55.1s  B → C`（cross_cluster）

v2 保留的跨任务边：
- `73.2s  C → D`

**结构** v1：持续重复（段内往返）、跨任务切换、离开后返回原任务

**结构** v2：持续重复（段内往返）、跨任务切换、离开后返回原任务

任务回访成立于 `pg_electrical_assembly`。

> **B 的判断依据（decided）**：等待他人，本身不推进任何任务目标。

---

## mosaic_graph_0030（mosaic_complement）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 从推车里拿着一叠桌布走到一个隔间放在桌子上 | pg_tablecloth_delivery | interaction node |
| B | 走回推车 | pg_tablecloth_delivery | interaction node |
| C | 整理推车里面的物品 | pg_cart_organization | interaction node |

**v1 时间轴**

```
A → B → C
```

v2 时间轴与 v1 相同。

**直接边** v1 2 条 → v2 2 条

v2 保留的跨任务边：
- `39.0s  B → C`

**结构** v1：跨任务切换

**结构** v2：跨任务切换

---

## graph_review_0023（repeated_actions）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 充电宝放回盒子，放下盒子 | pg_power_bank_inspection | interaction node |
| B | 拿起盒子，拿出充电宝 | pg_power_bank_inspection | interaction node |
| C | inspect充电宝 | pg_power_bank_inspection | interaction node |

**v1 时间轴**

```
A → B → C → A → B → C → A → B → C → A → B
```

v2 时间轴与 v1 相同。

**直接边** v1 10 条 → v2 10 条

**结构** v1：线性多步骤（同簇≥3节点）、多步骤循环（A→B→…→A）

**结构** v2：线性多步骤（同簇≥3节点）、多步骤循环（A→B→…→A）

---

## graph_review_0002（repeated_actions）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 团纸巾扔到垃圾桶 | pg_surface_cleaning | interaction node |
| B | 抽纸巾 | pg_surface_cleaning | interaction node |
| C | 叠纸巾 | pg_surface_cleaning | interaction node |
| D | 用纸巾擦桌子 | pg_surface_cleaning | interaction node |

**v1 时间轴**

```
A → B → C → D* → A → B → C → D → A → B
```

v2 时间轴与 v1 相同。

**直接边** v1 9 条 → v2 9 条

**结构** v1：持续重复（段内往返）、线性多步骤（同簇≥3节点）、多步骤循环（A→B→…→A）

**结构** v2：持续重复（段内往返）、线性多步骤（同簇≥3节点）、多步骤循环（A→B→…→A）

---

## mosaic_graph_0008（mosaic_complement）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 收起上一束电线，拿出下一束电线 | pg_wire_harness_tie_processing | interaction node |
| B | 给一根电线 | pg_wire_harness_tie_processing | interaction node |

**v1 时间轴**

```
B → B → B → B → B → A → B → B → B → B → B → B → B
```

v2 时间轴与 v1 相同。

**直接边** v1 12 条 → v2 12 条

**结构** v1：同节点再次出现（A→A）、多步骤循环（A→B→…→A）

**结构** v2：同节点再次出现（A→A）、多步骤循环（A→B→…→A）

---

## graph_review_0016（repeated_actions）

| 节点 | 描述 | v1 cluster | v2 |
|---|---|---|---|
| A | 拿起水壶，平行移动水壶 | pg_kettle_repositioning | interaction node |

**v1 时间轴**

```
A → A → A → A → A → A → A → A → A → A → A → A → A → A → A → A → A → A → A → A
```

v2 时间轴与 v1 相同。

**直接边** v1 19 条 → v2 19 条

**结构** v1：同节点再次出现（A→A）

**结构** v2：同节点再次出现（A→A）

