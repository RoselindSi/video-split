# U1：均匀软化与 cache 对照（2026-10-16）

固定因果快照与训练折温度校准，复用原五折外层和四折内层录像划分。配置 configs/graph_u1_softening.json，脚本 scripts/graph_u1_softening.py，结果 results/graph_u1/u1_causal_softening.json。

## 对照定义

- P：前驱 posterior 的 top1 查 memory。
- B：前驱 posterior 边缘化查询 memory。
- M：P 的 history distribution 与包含 NEW 的均匀分布混合，alpha 从 0/.25/.5/.75/1 内层选择。
- 三臂同时在内层选视觉 T、tau、lambda；查询温度仅用对应训练录像按前驱 NLL 选择。M 比 P/B 多搜 alpha，给简单替代解释更多预算。
- support 对已见 successors 均匀；cache 按 prefix occurrence 顺序取各 predecessor 最后一次观察后继，特殊状态断开邻接。两者都严格不读取当前转移。

## 结果

| 查询 | support | cache |
|---|---:|---:|
| P | 72.67% | 72.67% |
| B | 73.91% | 73.91% |
| M | 72.05% | 72.05% |

B-M=+1.86pp，录像聚类配对 bootstrap CI [+0.30,+3.85]pp；B-P=+1.24pp，CI [+0.27,+2.59]pp。
这是选定模型预测条件下的区间，不包含重复采样后重新校准/选参的不确定性。
M 相对其视觉对照救回 3 题、误伤 5 题，B 为救回 4 题、误伤 0 题。M 在 3 折关闭 prior，其他两折 alpha 分别为 .25/.5。

## 验证

- support P/B 与上一轮因果校准主表逐决策 pick 一致。
- 所有臂覆盖同一 322 decisions。
- 301 个不存在多后继的决策上，support 和 cache 产生的先验分布相同，通过断言检查。
- uniform mixture 的 alpha=0/1 端点分别为原 point prior/均匀分布，通过检查。

## 解释边界

当前结果不支持“仅将 point prior 均匀软化就能复现 belief 收益”这个具体解释；不是证明所有软化/收缩方法都不行。
cache/support 主结果完全打平，不支持完整 graph 或 successor set 的额外必要性。此前四题 audit 也显示它们用的是单后继信息。
仍是 oracle-prefix memory 的 query-only 实验；只比视觉多答对 4 题，且 benchmark 已反复查看，必须保留微小效应和多次分析的限制。

下一步先做 memory 内容打乱对照（保持每源支持集大小及后继类型处理规则，cache 与 support 各自定义），并在因果校准接口上重做噪声测试；只有验证收益依赖真实 successor identity 才讨论有限在线扩展。BG 的既定通过关卡仍未满足，不启动复杂在线写入。
