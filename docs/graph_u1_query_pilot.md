# U1 query 首轮结果（2026-10-16）

配置在 configs/graph_u1_query.json，实现在 scripts/graph_u1_query.py，逐决策及外层划分在 results/graph_u1/u1_query_nested.json。
使用五折外层/四折内层录像分组；中心化仅由对应训练录像估计。378 embeddings、322 decisions、49 videos。

| 臂 | 外层准确率 |
|---|---:|
| V | 72.67% |
| O | 81.68% |
| P | 80.75% |
| B | 74.84% |
| PG | 74.53% |
| BG | 72.98% |

各臂独立内层选择 T/tau/lambda；B-P=-5.90pp，录像聚类配对 bootstrap CI [-13.15,-0.34]pp；BG-PG=-1.55pp，CI [-3.07,-0.36]pp。该区间未包含重新抽样后重新训练/选参的不确定性。

## 首轮判断与限制

- 本配置下没有通过进入全在线写入的关卡，暂不启动 FBG。不能将负结果归因为 prototype 一项。
- P 与旧 D 的 prev 不同：这里用当前 oracle 前缀原型对上一 observation 重新分类，原型含该 observation 的 gold 身份归组，因此 P 接近 O 是偏乐观接口的结果，不能解释为修复了在线状态维护。
- 本轮仅 support；cache、匹配参数机制对照、corruption 尚未完成。因此是首轮诊断，不是整个 U1 最终判决。
- rescue/override 当前对每臂自己的 T/tau 视觉对照计数；在参数不匹配时不能直接横向比较这些计数。
- 下一步先做匹配 T/tau/lambda 的 point-belief 对照，并检查 predecessor entropy/gold mass 与 memory 权重，判断稀释和门控退回视觉的程度；随后做相同观测噪声的 corruption，不扩大模型。

检查：NEW/无出边质量保持归一化、不被丢弃；空记忆为均匀分布。运行在本地 CPU，两个线程，无服务器任务。
