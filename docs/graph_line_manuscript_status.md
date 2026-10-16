# Graph 线梳理与 manuscript 收口

更新：2026-10-16。U1 已关闭，不追加 U1.x。本文件将 graph-experiment 的既有 manuscript 与 video-split 中 U1 最终结果接起来；没有修改另一个仓库。

## 1. 我们实际研究的问题

**第一视角视频中的状态条件化交互历史，什么时候能帮助状态身份判断；这种收益在身份与记忆必须自行维护时能否保留？**
状态不是动作名的简单字符串，也不是 hand-object 空间关系图。实验要求同一交互状态跨时间保持身份，历史仅来自严格前缀。
Graph 是被检验的一种 memory 表示，不是已经成功的方法贡献。cache 不等于 episodic retrieval，后者尚未测过。

## 2. 证据链

### A. 历史表示：E1b / E2

E1b 显示视频与文本在 SAME/DIFFERENT 上产生不同方向偏置。
E2 的文本 memory 表面收益需要排除长度、格式和 oracle 命名混淆。采用 10-12 重算：长度效应 -16.9pp，控制长度后的结构效应 -2.0pp（CI 跨零），标签置换 +0.7pp、占位符 +2.6pp。旧 +20.4→-6.6 与重算 +11.4→-3.9 不复现，使用后者并说明差异。
结论限定到所测文本序列化接口，不泛化成“语言不能表示历史”。

### B. 历史内容：G1b

纯先验排序 n=116：均匀 41.8%、频率 30.8%、度匹配打乱 44.0%、真实 successor support 80.8%。真−乱 +36.8pp。
41.8% 是均匀先验，**不是视觉基线**。这是特定 oracle 先验排序任务中的内容信号，不与 U1 的 322 题混成同一张效果表。

### C. 表示复杂度：ladder

322 decisions / 49 videos：visual 72.98%、single-successor cache 78.57%、full support 78.88%。support−cache 约 +0.31pp，CI 跨零。
19 题净增益全部来自 deterministic regime；11 个名义 branching 决策视觉已 11/11。9/11 的 support 包含全部候选，不能提供排除约束。
说明当前域没有建立 full successor set 比弱缓存更有价值，不说明 cache 普遍最优。

### D. 数据何时能检验结构：branch_b1 / B2 / joint regime

- 内部严格 G-branch 2/322；branch_b1 真标 0/96，62/62 的再次后继确定。
- Ego-Exo4D B2 初轮 20 个粗标窗口 G-branch=0，冻结门槛 15，NO-GO。
- 修车动作级 10 条出现 5 个 G-branch，但都来自“拿工具”；折叠纯取得工具状态后为 0。真实工作状态最大访问数没有达到所需重复水平。
- 后来 scenario census 修正了“联合区间为空”：CPR 等点有结构，联合区间稀疏占据；recurrence/branching 是样本中的中等负相关倾向（rho=-.535），不是自然定律。
- 分叉机会依赖 dataset×窗口长度×粒度/身份定义；不能只写 dataset 的固有属性。窗口敏感性和人工/keystep 不同样本比较都要保留限制。

“recurrence+有意义 branching”是检验完整结构相对单后继缓存**额外价值**的机会条件，不是任何历史收益的必要条件：确定性 cache 已经有用。

### E. 在线落差：Experiment D

旧 twostage 复现：V 72.98%、O 78.57%、Pred-prev 72.67%、Full 61.18%。soft 对应 72.98/78.88/74.53/61.49。
需区分三项 oracle 依赖：前驱身份、transition history、prototype 分组。只换前驱不足以测完整部署性；原 full 同时替换 edges/prototypes，不能单独归因为 prototype 或边的一项。
旧分组选择参数后在全量报告与 NEW gold 对齐的局限，写入 Methods/Limitations，不将数字当全新独立嵌套验证。

### F. 最直接补救：U1（已关闭）

最终接口先保存因果前驱 posterior，训练折 NLL 校准，五折外层/四折内层选择参数。
V/P=72.67%、B=73.91%、PG=72.36%、BG=72.67%。B 多对 4 题，仅 1 题是非 top1 前驱质量独有救回。
cache/support 打平；真实 support 73.91%、shuffled 平均 73.60%，区间跨零。未建立真实 successor identity 驱动的稳定收益，BG 未过关卡，不进入全在线写入。
最终入口：graph_u1_final_result.md。早期乐观结果不与最终因果结果混用。

## 3. 可以主张的贡献

1. **受控诊断框架**：分离历史内容、表示接口、结构机会及 oracle 依赖，明确前缀、候选和状态身份的评测口径。
2. **表示复杂度的实证边界**：已测域里的收益主要体现为确定性 successor identity；完整 support 未显示超出单条缓存的价值，名义高出度不等于决策机会。
3. **oracle usefulness 与 online usefulness 的区别**：历史信号存在不保证在线可用；最直接的 uncertainty-aware 查询修正也未建立跨越这一落差的证据。

这些是 diagnostic contributions。算法不应包装为新型 graph reasoning，U1 不是已验证成功的主方法。novelty 的强度仍需正常相关工作核查，本文不声称已完成新的文献检索。

## 4. Manuscript 排布

1. Introduction：提出 history utility 的条件问题。
2. Setup：状态身份、NEW、严格前缀、三项 oracle、分母与 G-branch 定义。
3. Content & complexity：G1b + ladder（首先给出正面信息证据和弱表示对照）。
4. Structural opportunity：内部 census、B2、粒度折叠、scenario/窗口敏感性。
5. Oracle-to-online：D；接一小节 U1 最终主表和 true-vs-shuffled，解释直接补救未过关。
6. Representation confounds：E1b/E2，或依篇幅放在前述相关段落/附录。
7. Discussion/Limitations：标注身份、窗口、旧协议选参、少量分叉、既有 benchmark 反复使用、在线未解决。

U1 正文最多一张紧凑表，包含 B 的小提升与 null。不得只放 B>V 而省略 shuffled。

## 5. 接下来是写作工作，不是实验队列

- 以本文件为章节导航，统一 E2 最新数值、D 两种融合族以及 U1 最终条件。
- 建 claim→数据→脚本→图表索引；区分旧 exploratory/复现与新外层留出结果。
- 修订旧大纲里“联合区间为空”“结构完全无效”“prototype 唯一瓶颈”等过强表述。
- 检查 U1 脚本/配置/输入与产物校验和及复现命令；大体积逐决策 JSON 单独归档，正文引用 summary。
- 当前不加 U1.x、不启动 FBG、不扩大 B2、不训练 GNN、不把 episodic retrieval 写成已测成果。

工作落点：旧 manuscript 在 graph-experiment/docs；本轮 U1 与最终梳理在 video-split/docs。尚未跨仓库迁移、提交或推送。
