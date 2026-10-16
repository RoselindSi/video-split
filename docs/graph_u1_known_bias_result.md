# U1 内容无关偏置对照（2026-10-16）

配置 configs/graph_u1_known_bias.json；脚本 scripts/graph_u1_known_bias.py；结果 results/graph_u1/u1_causal_known_bias.json。
所有 known candidate 加相同 bias，NEW 不加；不使用 memory 内容或前驱置信度。T、tau、bias 在对应内层录像上联合选择，外层测试不参与。
偏置网格 -0.2/-0.1/-0.05/0/0.05/0.1/0.2，单位为除以视觉温度后的分数。

## 结果

| 方法 | 准确率 |
|---|---:|
| Visual | 72.67% |
| Known-state 固定偏置 K | 72.67% |
| 因果校准 Belief | 73.91% |

K 相对自身无偏置视觉救回 7 题、误伤 7 题，净作用为零。K-V 的录像聚类 bootstrap CI [-2.04,+1.89]pp；B-K=+1.24pp，CI [-1.00,+3.97]pp，不能宣布 superiority。

## 这个对照实际检验什么

known 分数为 cos/T+bias，NEW 分数为 tau/T，因此等价于纯视觉阈值 tau-T*bias。
五折有效阈值为 .925/.920/.895/.910/.895；这是比原先 .5/.7/.9 更细的一组视觉阈值搜索，而非独立 memory 方法。偏置臂获得额外搜索预算，但仍未显示净收益。
统一偏置不能区分不同旧状态；它只能改变 NEW/REVISIT 边界。结果不否定更复杂的候选/前驱相关 prior 会影响边界。

## 当前应停在哪

1. 简单固定 NEW 偏置没有复现 B 的点估计提升，但 B-K 区间跨零，不能据此排除该解释或证明 B 有独立价值。
2. 上一轮真实 memory 与 shuffled memory 不能区分仍是更关键的限制，不能因为这一个对照没有提升就宣布 belief 成功。
3. 现有 U1 证据：校准很重要；因果接口改变结论；cache 与 support 打平；belief 可减轻 point 的误伤，但 aggregate successor-content 依赖尚未成立。
4. 不启动全在线写入。建议先整理已有实验成一份完整结果表和可复现入口，审查 null 的力度及有限候选导致的随机恒等问题，再决定是否需要新独立数据验证；不在同一 49 条录像上不断追加调参寻找正结果。

未修改服务器资源，全部本地 CPU；未提交、未推送。
