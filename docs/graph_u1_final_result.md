# U1 最终结果与关闭记录

冻结日期：2026-10-16。状态：**实验关闭，停止新增 U1.x，回 manuscript。**
本文件是 U1 的最终引用入口；此前各轮文档和原始结果保留为审计记录，不作为相互独立的验证。

## 1. 问题与最终结论

问题：将预测前驱的单一身份改成概率分布、对 successor memory 边缘化，并按身份/记忆置信度门控，是否足以保住历史先验的收益？

**最直接的 uncertainty-aware fix 不足以解决 oracle-to-online gap。**
严格表述：所测试的因果查询、温度校准和门控组合，没有建立依赖真实转移内容的稳定收益，未通过继续全在线写入的预设关卡。因此没有证据表明它能弥合 gap。
这不是对所有 belief-state 方法的否定，也不是 FBG 的端到端失败测量：FBG 按关卡未运行。U1 的最终有效分支仍使用 oracle-prefix 身份归组和转移历史，不能称为 fully online。

## 2. 固定数据与基线复现

378 个 segment embeddings，322 个 decisions，49 条录像。复用原 release_v1、v2 node map、g1_emb.npz。
输入/旧源码 SHA256 在 results/graph_u1/input_hashes.json 与 release_hashes.json；最终 U1 文件校验和在 freeze_manifest.json。

| D 模式 | twostage | soft |
|---|---:|---:|
| Visual | 72.98% | 72.98% |
| Oracle | 78.57% | 78.88% |
| Pred-prev | 72.67% | 74.53% |
| Full | 61.18% | 61.49% |

两套旧基线均完整复现，文件 d_twostage_reproduction.json / d_soft_reproduction.json。不能把 twostage 的预测前驱与 soft 的 oracle 当成同一对比。
旧 g1_online 的分组评分选参后报告全量结果，不是独立外层留出；prev 答 NEW 后使用 gold ID 对齐。上述旧数字作为历史诊断复现，不包装成新的无 oracle 部署评估。

## 3. 最终查询协议

- 同一五折外层/四折内层录像划分；中心化均值仅由对应训练录像估计。
- 视觉为 prefix prototypes 的 cosine；当前 observation 不参与自己的 prototype。
- **因果前驱快照**：在上一段到来时，用更早 prefix 计算视觉 logits 并保存。当前决策不得用已加入上一段的新原型重新识别前驱。
- 快照仅为视觉 posterior，不递归吸收上一时刻 memory-fused posterior；这是 query-only 诊断。
- 262 个决策有快照：205 个前驱为当时已知状态、57 个为当时 NEW；其余 60 个初始/重置等决策无快照，统一均匀回退。
- 新增已知节点的前驱质量为零；NEW 质量保持 NEW，不用 gold 当前身份重新分配。NEW/无 outgoing memory 的质量回退到含 NEW 的均匀分布。
- support 在已见 successor 集内均匀；cache 仅留每源最后一次 prefix successor。特殊状态打断邻接。
- 分数为 visual/T + lambda*log(prior)，NEW 的 visual logit 为 tau/T。epsilon=.001、kappa=2。
- V 无 memory；O 为 oracle 前驱；P 为 posterior argmax；B 为 posterior 边缘化；PG/BG 再乘归一化 entropy confidence 与 memory-evidence gate。
- V 的视觉原型也来自 oracle prefix grouping；不能把 V 称为完全无 oracle 的部署模型。

### 校准与选参

视觉 T 网格 .1/.25/.5/1；tau .5/.7/.9；lambda 0/.05/.1/.2。
查询温度网格 .01/.02/.05/.1/.25/.5/1/2；每个内层训练集按前驱 NLL 选温度，再在内层留出选任务参数；外层对应训练集重新拟合查询温度。最终五折查询温度均为 .05。
校准改善概率质量但不改变前驱 top1；最终因果接口前驱 top1 约 74.0%。
早期非因果接口把上一段本身纳入 oracle prototype，前驱 top1 98.4%、温度选 .01，属乐观诊断，**不作为最终 U1 主表**。

## 4. 最终主表

文件：u1_causal_nested_calibrated.json。

| 条件 | 外层准确率 |
|---|---:|
| V | 72.67% |
| O | 81.68% |
| P | 72.67% |
| B | 73.91% |
| PG | 72.36% |
| BG | 72.67% |

B-P=+1.24pp，录像聚类配对 bootstrap CI [+0.28,+2.51]pp；BG-P=0，CI [-0.83,+0.96]pp。
P 五折均选 lambda=0，因此主表 B-P 也等于 B-V。多对 4 题不足以归因为正确 memory 内容；必须同时读第 6 节。
所有区间条件于已选模型/参数，不含重抽样后重新拟合、选参的不确定性，且来自已反复分析的既有 benchmark。

## 5. 四题差异与匹配参数

文件：u1_causal_rescue_audit.json。

| 决策 | 观察到的机制 |
|---|---|
| mosaic_graph_0015#06 | top1 前驱为 NEW(.738)，正确 C 仍有 .233；C→D 的残余质量纠正当前 NEW 误判。相同参数 P 不能救回。 |
| mosaic_graph_0022#05 | 前驱 top1 B 正确；P/B 都能救回当前 A。 |
| graph_review_0019#04 | 前驱 top1 A 正确；P/B 都能救回当前 B。 |
| graph_review_0008#10 | 前驱 top1 A 正确；P/B 都能救回 A，属于已知/NEW 边界纠正。 |

4 题在 4 条录像，只有第一题是直接的非 top1 前驱检索示例，不能写成 4 个 uncertainty 专属成功案例。所用 successor 均为单后继，不是 branching graph 的证据。
匹配 lambda=.10 时，P=71.12%、B=73.91%；相对共同视觉，P rescue/override=13/18，B=8/4。B 的优势主要为减害，而非增加救回数。

## 6. 最终控制组必须同时呈现

### 简单软化与表示复杂度

文件：u1_causal_softening.json。M 将 P prior 与 uniform 混合，alpha=0/.25/.5/.75/1 在内层联合选择。

| 查询 | support | cache |
|---|---:|---:|
| P | 72.67% | 72.67% |
| B | 73.91% | 73.91% |
| M | 72.05% | 72.05% |

B-M=+1.86pp，CI [+0.30,+3.85]pp。仅否定这个具体简单软化在当前协议下复现收益，不排除所有收缩方法。cache 与 support 打平，不支持完整图的必要性。

### True vs shuffled：决定解释的关键控制

文件：u1_causal_memory_null.json；20 个种子，不重调真实内层选出的参数。
每个决策对当前 known successor labels 做双射置换，保持源、每源后继数量与边计数，NEW 回退不变。

| memory | true | shuffled 均值 | true−shuffled | 种子×录像重抽 95% CI |
|---|---:|---:|---:|---:|
| support | 73.91% | 73.60% | +0.31pp | [-0.75,+1.54]pp |
| cache | 73.91% | 73.71% | +0.20pp | [-0.97,+1.47]pp |

真实内容优势未被建立。原四题在 shuffled support 下仍答对的次数分别为 6/20、11/20、10/20、20/20。
null 是逐决策 prefix intervention，未必对应一条连贯在线历史；小候选集导致只有约 47.7% 的边目标实际改变。保留这一限制，不为追求显著结果再追加更强 U1.x。

### 内容无关 NEW 偏置

文件：u1_causal_known_bias.json。known scores 统一加 bias，内层选择，等价于视觉阈值 tau-T*bias。
K=72.67%，B-K=+1.24pp、CI [-1.00,+3.97]pp。未复现净提升，也不能证明 B 优于 K；不能据此推翻 true-vs-shuffled 的限制。

### 噪声实验的地位

u1_query_diagnostics.json 的 0/.1/.2/.3/.4 posterior 干预、10 个种子，基于早期乐观非因果接口。只保留在探索性附录/审计记录，不能作为最终因果接口 robustness 主结果。
本次关闭后不再补做因果版 corruption。

## 7. 停止判据与决定

原执行计划要求 BG-P 的区间下界 >0，且不是 lambda=0 的视觉回退，才能进入全在线；BG-PG 用于区分 belief 与 gate。
最终 BG 未通过；true-vs-shuffled 又未建立内容依赖；cache/support 无差别。因此**停止 U1，不运行 FBG，不扩大模型，不继续细调 gate/温度/null 来寻找正结果。**
停止不是证明所有 memory 都无用，而是关闭“直接保留 uncertainty 就能修复该 gap”的最小补救路线。

## 8. 放回 manuscript

主线仍为：历史中的什么信息有用 → 何时有非平凡结构机会 → oracle 到 online 哪些假设失效。
U1 作为 Experiment D 后的一小节：直接补救的边界。正文保留最终主表、true/null、停止决定；校准迭代和 4 题细节入附录。
建议英文：
> In the tested query-only setting, calibrated belief marginalization yielded a small gain over vision, but the gain was not distinguishable from successor-identity permutations; confidence gating did not pass the criterion for online extension. Thus, the direct uncertainty-aware correction did not establish a remedy for the oracle-to-online gap.

不要写“FBG measured worse than vision”“all uncertainty-aware memory fails”“belief proves successor information online”。
未提交、未推送；所有实验本地 CPU，无服务器 GPU 任务。
