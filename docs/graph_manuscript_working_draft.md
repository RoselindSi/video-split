# When Does Interaction History Help Egocentric State Tracking?

## Recurrence, Structural Opportunity, and the Oracle-to-Online Gap

中文整合工作稿 · 2026-10-16。本文整合既有 graph-experiment 结果与已关闭的 U1；不是新增实验计划。文献综述与引用尚待补齐，不作为已完成投稿稿。
数字与产物映射见 graph_manuscript_evidence_index.md；U1 最终协议见 graph_u1_final_result.md。

## 摘要

第一视角长视频中，交互状态可能在相隔多个操作后再次出现。过去观察到的转移能否帮助判断当前状态身份，以及这种帮助需要多复杂的历史表示？我们通过受控诊断区分历史内容、表示复杂度、结构机会和在线身份维护。在 oracle 身份下，真实 successor support 的纯先验排序明显优于度匹配的打乱控制；但在带视觉的状态识别中，完整 successor support 与单条后继缓存的差别无法区分，收益集中在确定性转移。跨粒度和窗口的分析进一步表明，名义分叉不自动构成额外结构的决策机会。将前驱身份和历史维护改为模型预测后，oracle 收益不能直接保留。一个最小的不确定性感知查询补救在因果、校准的评测中只产生微小提升，且无法与 successor 身份置换控制区分，置信度门控未通过进入全在线扩展的判据。这些结果为交互历史评估提供了明确的内容控制、复杂度对照与在线依赖审计，而不是提出一个已验证有效的新 graph 方法。

## 1. Introduction

同一个人在一段操作视频中可能反复回到相同工件、容器或工作状态。若模型已观察到这些状态之间的关系，历史应能为当前观察提供额外证据。问题在于：历史的哪部分有信息，完整图相对较弱缓存是否必要，以及模型是否能可靠地查询自己维护的历史。

我们研究 n_hat_t=f(x_t,M_<t)：当前视觉观察 x_t 与严格前缀记忆 M_<t 共同决定状态身份。本文不将 graph 作为贡献本身，而把它视为一种可被简化、置换和替换 oracle 依赖的历史表示。比较的核心是状态条件化的后继记忆，而非全局图规划或 message passing。

我们的诊断分为三个问题。第一，真实 successor identity 是否提供超出频率、出度等 nuisance statistics 的信息？第二，数据中是否存在需要多个合法后继才能帮助决策的机会，而非确定性 lookup？第三，当状态身份和记忆不再由标注给定时，oracle 收益能否保留？

本文贡献为：(1) 将内容控制、弱表示对照和 oracle 依赖拆分纳入同一评测框架；(2) 实证刻画现有数据上结构机会的稀疏性及其对状态粒度、窗口的敏感性；(3) 展示 oracle 信息收益与在线可用性的区别，并给出直接 uncertainty-aware 修正未建立补救证据的完整控制结果。

这些结论限定于已测数据与接口。我们不将 recurrence 与 branching 宣称为一切历史收益的必要条件：确定性缓存本身已能有用。二者的共同出现，针对的是检验完整 successor structure 超出弱缓存的额外价值。

## 2. Setup and evaluation

### 2.1 State identity and strict prefixes

输入由具有持续身份的交互状态 occurrence 组成。每个决策选择一个已见状态，或在第一次出现时选择 NEW。状态身份依赖具体标注合同，不能与动作字符串相同直接等同。
对决策 t，候选和转移仅由更早 occurrence 构造，当前转移在预测后才进入历史。特殊或未知状态打断邻接，不凭空连跨段边。

### 2.2 History representations and opportunity

单后继 cache 存储每个前驱最后一次观察到的 successor；support 存储所有已观察 successor。二者都不同于 episodic retrieval，后者不在已测方法内。

我们分别统计前驱历史可用率 P(S非空)、历史覆盖质量 P(gold∈S | S非空)、多后继比例及其中无法排除候选的比例。后两项不能混为“图复杂度”。严格 G-branch 使用现有分类器定义，需满足多个后继、候选收缩和最小历史等条件；精确定义以代码和合同为准。

### 2.3 Oracle dependencies and protocol families

识别三项 oracle 依赖：前驱身份、转移记录、视觉原型的身份归组。只替换前驱仍保留后两项，因此不等于完整在线系统。
旧 G1/ladder/D 的结果保留其原始协议，并注明选参和评分方式；U1 使用五折外层、四折内层录像划分，训练录像中心化与前驱 NLL 校准。两种融合族及旧/新评测绝不跨表计算方法增益。

## 3. What information does history provide?

### 3.1 Successor identity versus nuisance statistics

G1b 在 116 个决策、25 条录像上的纯先验排序结果如下。这一实验不含视觉 scorer，均匀先验的数值不是视觉基线。

| Prior | Top-1 |
|---|---:|
| Uniform | 41.81% |
| Node frequency | 30.82% |
| Degree-matched shuffled support | 43.99% |
| True successor support | 80.82% |

真实−度匹配打乱为 +36.82pp，95% CI [+21.02,+49.47]pp。打乱−均匀的区间跨零；频率的点估计低于均匀，但其差值区间也跨零。结果支持这个 oracle 排序任务中的内容信息来自具体 successor identity，而非仅仅增加一个 prior 的形状或节点频率。

### 3.2 Is full successor support necessary?

历史 ladder 在 322 个决策、49 条录像上得到 visual 72.98%、cache 78.57%、support 78.88%。support−cache 为约 +0.31pp，区间包含零：没有建立完整 support 的额外收益。

| Regime | Decisions | Visual | Cache | Support |
|---|---:|---:|---:|---:|
| No support | 117 | 60.68% | 60.68% | 60.68% |
| Deterministic | 185 | 78.38% | 88.11% | 88.65% |
| Nominal branching | 11 | 100% | 100% | 100% |
| Novel transition | 9 | 88.89% | 88.89% | 88.89% |

support 相对视觉多答对的 19 题全部来自 deterministic regime。11 个名义 branching 决策视觉已满分，其中 9 个 successor set 包含所有已见候选，条件化不能排除任何已知状态。这里的“无额外证据”不等于证明 graph 在其他数据上无用。

## 4. When is transition structure testable?

内部 release 严格 G-branch 仅 2/322；独立 branch_b1 标注中为 0/96，重复后继检查为 62/62 确定性。Ego-Exo4D 的首轮 B2 在 20 个粗标窗口得到 0 个 G-branch，未达冻结的 15 个门槛。

改变粒度产生的表面分叉也需要审查。修车动作级探针在 10 个窗口得到 5 个 G-branch，但前驱全部是取得工具状态。按事先陈述的取得类折叠规则归并后，这 5 个全部消失。这个局部结果说明粒度能改变名义结构，不能据此证明所有细化都只会产生伪分叉。

scenario 级 census 修正了“联合区间为空”的早期说法：区间稀疏占据，CPR 等 scenario 提供有效但低产出的结构机会。16 个 keystep scenario 上两轴呈中等负相关倾向（rho=-.535），不构成普遍规律。60/90/180 秒敏感性分析显示分叉率受窗口长度影响，而 CPR 的历史覆盖质量相对稳定。结构机会应描述为数据、窗口、粒度和身份合同的联合属性。

## 5. Oracle utility does not imply online utility

### 5.1 Experiment D

| Mode | Two-stage gate | Soft fusion |
|---|---:|---:|
| Visual | 72.98% | 72.98% |
| Oracle | 78.57% | 78.88% |
| Predicted predecessor | 72.67% | 74.53% |
| Self-maintained history and prototypes | 61.18% | 61.49% |

两种融合均显示：oracle 增益不能直接转化为全自维护收益。两阶段 prior 排除视觉原本正确答案的次数从 oracle 的 1 增到 predicted predecessor 的 18，说明错误查询可以转化为 harmful override。
Full 同时改变 transition history 与 prototype grouping，不能据此断言其中一项是唯一主因。旧 prev 对 NEW 使用 gold 对齐、旧模型选择并非独立外层留出等限制，必须与复现结果一并说明。

### 5.2 U1: a direct uncertainty-aware correction

U1 检验最小补救：保留前驱 posterior、边缘化 successor prior，再按身份 entropy 和记忆证据门控。最终实现保存上一段到来时的因果分数快照，不允许包含上一段本身的 prototype 重新识别它。NEW 质量保留未知，不按 gold 身份重分配。

查询温度在对应训练录像上按前驱 NLL 校准，再以内层留出选择融合参数；最终五折温度均为 .05。全部条件仍使用 oracle-prefix identity grouping 与转移历史，并非已完成全在线系统。

| Final U1 condition | Accuracy |
|---|---:|
| Visual / Point | 72.67% |
| Belief | 73.91% |
| Point-gated | 72.36% |
| Belief-gated | 72.67% |
| True-belief support | 73.91% |
| Shuffled-belief support, 20-seed mean | 73.60% |

Belief 比视觉多对 4 题，分属 4 条录像；但只有 1 题直接体现非 top1 前驱质量独有救回。真实−shuffled support 为 +0.31pp，种子×录像重抽区间 [-0.75,+1.54]pp，不能区分。cache 与 support 打平。简单 point-uniform 混合与固定 NEW bias 对照均未复现相同点估计提升，但也不能替代内容 null 或建立普遍方法优势。

因此我们不将微小 B-V 差值归因为有效的 successor-content retrieval。预设的 BG-P 正收益关卡未满足，full-online extension 未启动，U1 在此停止。最直接 uncertainty-aware fix 未建立弥合 oracle-to-online gap 的证据；这不是对所有 belief-state memory 的否定。

## 6. Representation-interface controls

E1b 中视频与文本对 SAME/DIFFERENT 的偏置方向不同，提示关系判断不能脱离表示接口评价。
E2 的 10-12 重算中，长打乱上下文相对无历史低 16.9pp；长度/排版对齐的 true−shuffled 结构差值为 -2.0pp，CI [-4.5,+0.7]pp。
原标签相对随机标签的差值为 +0.7pp、相对无意义占位符为 +2.6pp，二者均不显著。这些是语义内容对照差值，不是“随机标签/占位符带来提升”。
修正 canonical 名称泄漏后，M2raw 的对照收益从 +11.4pp 变为 -3.9pp，后者区间跨零。该重算与旧文档 +20.4→-6.6 不一致，原因未解决；正文采用可定位的重算产物并披露差异。结论限于所测序列化接口，不能推广为语言表达历史必然无效。

## 7. Discussion and limitations

结果区分了三件容易混淆的事：存在 oracle 内容信号、数据需要完整结构、模型能在线维护并查询该结构。它们不能相互替代。

主要限制包括：状态身份由标注合同决定；B2 各粒度部分不是相同窗口的配对实验；窗口长度改变可见历史；有效 branching 少；旧实验选参协议与 U1 不同；U1 的 49 条录像经过反复分析，不能作为新独立验证。U1 bootstrap 条件于已有模型和参数，未纳入重拟合的不确定性。内容 null 保留每源后继数量但只有约 47.7% 边目标实际改变，逐前缀置换不一定对应一条连贯在线历史。

因此贡献是可审计的诊断与评测边界，而非 graph 或 memory 的普遍胜负。Episodic retrieval、复杂 online write policy 与 GNN 不在本研究已支持结论中。

## 8. Conclusion

在已测第一视角交互数据上，状态条件化的后继身份具有 oracle 信息价值，但完整 successor support 未显示超出简单缓存的优势。结构机会依赖复现、候选约束以及粒度与窗口；oracle 收益也不保证在线可用。最直接的不确定性感知修正未建立跨越这一落差的内容依赖收益。评估交互历史时，应同时报告内容控制、弱表示对照、结构机会和身份维护假设。

---

写作状态：正文整合稿已形成；仍需相关工作与引用、图表编号/版式、旧结果协议说明和跨仓库同步。U1 实验已关闭，不再以这些写作待办为由追加实验。
