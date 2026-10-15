# Interaction Graph Release v1

## Release summary

- Videos: 60.
- Action nodes: 134.
- Special nodes excluded: 18.
- Procedure groups: 45.
- Procedure groups reused across videos: 8.
- Single-node procedure groups: 16.
- Fine edge occurrences: 341.
- Aggregated fine edges: 141.
- Aggregated group edges: 62.
- Videos with an inter-group transition: 4.
- Videos that leave and later return to a procedure group: 2.
- Edges omitted because a special state was an endpoint: 39.
- No edge was synthesized across a special-state gap.

## Edge semantics

- `exact_state_repeat`: 161 occurrences.
- `inter_group_transition`: 9 occurrences.
- `intra_group_transition`: 171 occurrences.

## Batch comparison

| Batch | Videos | Minutes | Action nodes | Special nodes | Exact repeats | Intra-group | Inter-group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mosaic_complement | 30 | 53.38 | 69 | 9 | 101 | 42 | 9 |
| repeated_actions | 30 | 39.55 | 65 | 9 | 60 | 129 | 0 |

## Largest procedure groups

| Procedure group | Nodes | Videos | Duration (s) |
| --- | ---: | ---: | ---: |
| 清洁表面 | 15 | 5 | 448.7 |
| 收放充电线 | 12 | 4 | 292.2 |
| 清洗餐具 | 11 | 4 | 296.2 |
| 取放并检查充电宝 | 9 | 3 | 251.9 |
| 准备容器并称量果干 | 6 | 1 | 120.0 |
| 饮料上架与记录 | 5 | 1 | 98.2 |
| 操作碗上保鲜膜 | 4 | 1 | 100.0 |
| 收放拖鞋 | 4 | 2 | 49.7 |
| 整理厨房餐具 | 4 | 3 | 246.4 |
| 清洗并复位下水道塞子 | 4 | 4 | 307.6 |
| 茶叶分装与称量 | 4 | 1 | 120.0 |
| 拆卸风扇 | 3 | 1 | 81.3 |
| 清洗并收装玻璃瓶 | 3 | 1 | 120.0 |
| 组装风扇 | 3 | 1 | 51.3 |
| 缝制爱心图案 | 3 | 1 | 99.1 |

## Inter-group transitions

- `mosaic_complement/mosaic_graph_0002` at 58.0s: 逐节拆下风扇支撑杆 (拆卸风扇) -> 前往另一房间取凳子并返回 (取用辅助物品).
- `mosaic_complement/mosaic_graph_0002` at 96.7s: 前往另一房间取凳子并返回 (取用辅助物品) -> 继续拆解风扇头部组件 (拆卸风扇).
- `mosaic_complement/mosaic_graph_0006` at 12.6s: 用抹布擦拭桌面 (清洁表面) -> 整理桌面和架子上的锅盖及盆 (整理厨房餐具).
- `mosaic_complement/mosaic_graph_0006` at 19.0s: 整理桌面和架子上的锅盖及盆 (整理厨房餐具) -> 跟随他人走到厨房另一端 (跟随他人行走).
- `mosaic_complement/mosaic_graph_0006` at 30.5s: 跟随他人走到厨房另一端 (跟随他人行走) -> 接过装有抹布的盆 (接取热水).
- `mosaic_complement/mosaic_graph_0020` at 33.6s: 组装线缆测试组件 (组装线缆测试组件) -> 等待他人继续组装 (等待协作者).
- `mosaic_complement/mosaic_graph_0020` at 55.1s: 等待他人继续组装 (等待协作者) -> 查看手机并在盒子上记录 (查看并记录工作信息).
- `mosaic_complement/mosaic_graph_0020` at 73.2s: 查看手机并在盒子上记录 (查看并记录工作信息) -> 逐根整理线束端部裸线 (组装线缆测试组件).
- `mosaic_complement/mosaic_graph_0030` at 39.0s: 返回推车 (运送桌布) -> 整理推车内物品 (整理推车).

## Procedure-group returns

- `mosaic_complement/mosaic_graph_0002`.
- `mosaic_complement/mosaic_graph_0020`.

## Medium-confidence review queue

9 nodes remain medium-confidence. They are deliberately kept in narrow contextual groups rather than merged broadly.

- `mosaic_complement/mosaic_graph_0003/A`: 旋拧机器上的螺丝 -> 调整机器螺丝. 画面可确认旋拧，但无法稳定判断拧紧或拧松方向。
- `mosaic_complement/mosaic_graph_0005/A`: 在电机组件上安装零部件 -> 组装电机组件. 根据画面将对象收窄为机电组件，具体产品型号不可见。
- `mosaic_complement/mosaic_graph_0008/B`: 将线束上的白色扎带逐个送入机器处理 -> 处理线束扎带. 根据视频补全机器操作；设备具体是收紧、切断还是检测仍不可完全确认。
- `mosaic_complement/mosaic_graph_0010/A`: 拿着手机在电动车之间巡视 -> 巡视并摆放电动车. 可见巡视和手机操作，手机中的具体任务信息不可见。
- `mosaic_complement/mosaic_graph_0016/A`: 将黑色塑料件放入机器处理 -> 加工黑色塑料件. 物体外观可见，但机器执行的是检测还是加工不可完全确认。
- `mosaic_complement/mosaic_graph_0016/B`: 用气枪清理黑色塑料件 -> 加工黑色塑料件. 按画面将喷枪规范为气枪，处理目的推定为清理。
- `mosaic_complement/mosaic_graph_0020/A`: 组装线缆测试组件 -> 组装线缆测试组件. 画面可见小型线缆组件，无法确认设备正式名称。
- `mosaic_complement/mosaic_graph_0020/D`: 逐根整理线束端部裸线 -> 组装线缆测试组件. 按同一视频上下文归入线缆组件组装。
- `mosaic_complement/mosaic_graph_0023/B`: 拿取餐具，在水龙头下冲洗后放置 -> 清洗餐具. 原描述未明确具体餐具种类，保留上位对象名称。

## Completion of the nine-step plan

1. Complete: 60 latest finals frozen with source and snapshot hashes.
2. Complete: 134 action nodes extracted with provenance.
3. Complete: every action has an original and canonical description.
4. Complete: the procedure-group contract is versioned in `docs/`.
5. Complete as a single-agent consistency review; human inter-annotator agreement was not measured.
6. Complete: all 134 action nodes assigned to one of 45 procedure groups.
7. Complete: fine and group graphs generated with typed edges.
8. Complete: batch comparison, transition audit, and confidence review recorded here.
9. Complete: JSON, CSV, checksums, frozen annotations, and the standalone graph explorer are published together.

## Interpretation boundary

This v1 is a complete, reproducible single-agent release. It must not be reported as having human inter-annotator agreement. A later independent human audit can update the nine medium-confidence nodes without changing the frozen source timelines.
