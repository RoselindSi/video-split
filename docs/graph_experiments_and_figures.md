# 按项目演化组织实验与图

整理日期：2026-10-16。依据用户附件“问题怎么演化—每阶段回答了什么—最终 paper 应该 claim 什么”。
**本轮仅整理已有实验和生成图，不追加 U1.x、不跑新模型。**

图册入口：[可浏览图册](../results/graph_paper_bundle/index.html)；[总览图](../results/graph_paper_bundle/contact_sheet.jpg)。每张图提供 PNG、SVG、PDF。
复现制图：`.venv/Scripts/python.exe -X utf8 scripts/build_graph_paper_bundle.py`。数据校验和与图源在 `results/graph_paper_bundle/figure_manifest.json`。

## 一、整条实验线：每一步究竟排查什么

| 顺序 / 实验 | 被检验的问题 | 关键条件和控制 | 主要结果 / 允许的解释 | 图与位置 |
|---|---|---|---|---|
| 0 原始 Graph hypothesis | 图是否能限制状态候选，让视觉在合法后继中判断？ | interaction-state nodes + prefix transitions；假设持续身份及有意义多后继 | 这是起点和待检验假设，不是贡献结论 | Fig.1，Introduction |
| 1 E1b | VLM 能否稳定判断同一状态关系？ | 同一关系问题的视频输入 vs 文本输入，307候选 | 文档记录偏置方向相反；本机只有候选，未定位成对评分产物，**只画定性示意** | Fig.2A，Representation；数字表不得凭记忆补 |
| 2 E2 | 文本历史的收益来自结构，还是接口混淆？ | true/shuffled history；语义标签置换/占位符；canonical泄漏更正 | 长上下文对照-16.9pp；结构差值-2.0pp CI跨零；标签对照差值0.7/2.6pp均不显著 | Fig.2B；完整臂表入附录 |
| 3 G1/G1b | 移除VLM后，历史内容本身有信息吗？ | 均匀、频率、真实support、度匹配rewiring，116/25；纯先验任务 | true80.82%，shuffle43.99%；差+36.82pp。内容信号在本oracle任务中成立 | Fig.3A/B，Content |
| 4 Ladder | 完整图是否优于最弱可用缓存？ | 同视觉族下self/recency/frequency/cache/support/null，322/49 | cache78.57%、support78.88%，额外差别不可区分；19题全来自确定性regime | Fig.3C/D，Complexity |
| 5 branch_b1 | 精选重标能找到真实分叉吗？ | 15条精选片段独立人工标注，96决策 | G-branch0；62/62重复后继确定。限定该池，不推广所有数据 | 正文Table 1，配Fig.4解释；不再单做一张零柱图 |
| 6 B2粗标/keystep/census | 换数据后是否出现可用joint regime？ | 同classifier，Ego-Exo粗/细粒度；scenario分解；报告空约束 | 首轮粗标20窗口G-branch0，NO-GO；scenario联合区间稀疏而非空 | Fig.4A；分母/每窗口产出入Table 1 |
| 7 B2修车folding | 细标新增branch是否为任务级结构？ | 同10个人工动作级窗口，取得工具状态fold前/后 | 5→0；全部来自拿工具；真实工作状态最多访问2次 | Fig.4B/C，Granularity |
| 8 Experiment D | oracle信息能否在预测身份/自维护历史中保留？ | V/O/Prev/Full；twostage与soft分别复现 | twostage72.98/78.57/72.67/61.18%；override1→18 | Fig.5，Online gap |
| 9 U1最终条件 | 最直接uncertainty query是否足以修复？ | 因果snapshot；训练折校准；P/B/PG/BG；5×4 CV | V/P72.67、B73.91、PG72.36、BG72.67%；4题微小收益 | Fig.6A；协议表必须标query-only |
| 10 U1最终controls | 小收益是否特定于真实历史？ | true/null20种子；cache/support；point-uniform；固定NEW bias | true73.91、shuffle73.60；内容优势未建立，停止，不运行FBG | Fig.6B；S1审计；不再追加U1.x |

E1b→E2→G1b→Ladder→B1/B2→D→U1 是研究演化；正式paper可把正面的内容证据G1b/复杂度对照提前，E1/E2放接口段落。数字任务不同，不能直接画一条连续“性能增长曲线”。

## 二、6张正文图

### Fig.1 — 研究逻辑图

文件 `fig01_study_logic`。
三问：历史有没有信息 → 需要完整结构吗 → 在线能保留吗。对应各阶段实验，终点是history diagnostic而非成功graph model。
**读者应记住**：信息价值、结构必要性、在线可用性是三个不同问题。

### Fig.2 — 表示接口与结构收益的区分

文件 `fig02_representation_controls`。
A：E1b 视频偏SAME、文本偏DIFFERENT的定性对照。B：E2四个测量差值及95%区间。
**图注要写**：E1b未绘评分数字；E2是臂间差值；长上下文对照不仅改变token数，不能称纯粹长度因果效应；不显著不等于严格零效应。
**放正文的目的**：排除“把history写进prompt就等于使用了结构”。

### Fig.3 — 真正的信息与最弱表示

文件 `fig03_content_and_complexity`。
A/B：G1b纯先验结果及true/null差值。C：visual/cache/support。D：19题净增益的regime归属。
**读者应记住**：oracle successor identity有信息，但当前收益不需要证明完整graph；branching题视觉已满分。
**不许省略**：G1b116/25与ladder322/49的分母区别，41.8%不是视觉baseline。

### Fig.4 — Joint regime、空约束与粒度

文件 `fig04_structural_opportunity`。
A：scenario点图，面积随决策数增长；内部nominal11个多后继中9个为空约束，另标严格2个。B：同10窗口folding5→0。C：工作状态与取得状态的最大访问次数分布。
**坐标口径**：横轴为P(gold∈S | S非空)，不是字面的状态复现频率；纵轴为名义多后继决策比例，必须和空约束标记一起读。
**读者应记住**：联合机会稀疏，不是不存在；细化粒度不保证新增有意义结构。
**不许写**：普遍证明recurrence与branching互斥，或“所有transition prior必须有branching”。

### Fig.5 — Oracle-to-online落差

文件 `fig05_oracle_online`。
左：D四模式，两种融合族分别画。右：oracle与pred-prev的harmful override。
**读者应记住**：查得到正确历史与维护可靠身份不同；强prior会放大错误查询。
**图内限制**：旧选参非独立外层CV；NEW gold对齐；Full同时改变历史和原型。不能归因为prototype单因素。

### Fig.6 — U1最终结果与关闭条件

文件 `fig06_u1_final_controls`。
A：最终因果+校准主表。B：B−V/P、true−shuffled support/cache、B−简单软化、B−固定NEW偏置的差值和区间。
**必须一起出现**：小幅B>V与true-vs-shuffled不能区分。不得只截左图宣称成功。
**结论**：直接uncertainty-aware补救未建立跨越gap的证据；BG未过关，FBG未运行，U1已关闭。
null区间为种子×录像重抽，其他为录像聚类；不包含重拟合的不确定性。

## 三、附录与表格

### 已生成的附录图

- **S1 `supp01_calibration_and_cases`**：因果query留出fold的NLL/ECE校准前后；4题正确前驱概率，唯一非top1专属案例标红。校准参数仅拟合训练录像。
- **S2 `supp02_window_sensitivity`**：60/90/180秒下CPR、salad、noodles、antigen的覆盖质量与branch机会。连线不是同一决策的配对实验。

### 表格分配

- **Table 1：数据与结构机会**——内部release、branch_b1、Ego-Exo粗标、keystep、bike action/folded；单位、决策数、S非空、命中质量、nominal/strict分叉、空约束分别列，不能统一混成一个branch count。
- **Table 2：Oracle依赖与协议**——D与U1各条件是否使用gold predecessor、oracle edges、oracle grouping、NEW对齐、因果snapshot、独立outer folds。
- **Appendix Table A1：E2全臂与数值更正**——canonical泄漏修正+11.4→−3.9，旧+20.4→−6.6不复现，采用measured产物。
- **Appendix Table A2：U1完整运行清单**——最终因果结果与早期乐观探索分开。早期噪声曲线不升格为最终causal robustness证据。

图册目录的 `tables.md` 已从JSON生成两张核心表的数值部分；branch_b1计数作为文档核对证据单列，不伪装成新重跑。

## 四、按附件三项contribution落图

| Contribution | 正文图 | 可以claim | 边界 |
|---|---|---|---|
| C1 Structural opportunity / identifiability | Fig.1、4 | 分离可复现身份、有意义多后继、有效候选收缩，测量粒度/窗口依赖 | 针对完整结构超出弱cache的价值，不是所有history收益的必要条件；不是普遍定理 |
| C2 Controlled isolation of history content | Fig.2、3 | 特定oracle任务中，真实successor身份比匹配null更有信息；完整support无已建立额外收益 | 不把G1b内容证据外推为U1在线内容证据；cache≈support不是严格等价证明 |
| C3 Oracle-to-online gap | Fig.5、6 | oracle信息不保证在线收益；所测直接uncertainty fix未建立补救证据 | 未证明所有belief方法无效；FBG未运行；不是prototype唯一瓶颈 |

## 五、与附件相比，保留的必要修正

1. “Temporal transition priors require recurrence+branching”过宽：确定性cache已有收益。改成评估更复杂结构额外价值的joint opportunity。
2. “Cache完全吸收全部收益”改成“完整support相对cache额外差别不可区分”。
3. E2的0.7/2.6为语义对照差值，不是随机标签或占位符本身的收益。
4. G1b的degree-matched rewiring与U1逐前缀target-label permutation是不同null，不能共用“入出度精确保持”的措辞。U1保留每源后继数及计数，未声称图全局同构控制。
5. 不把B2的局部失败写成“换任何数据、标任何粒度都不行”；CPR反例和窗口敏感性必须保留。
6. U1的4题不是4个uncertainty独有成功。真实/打乱无可区分优势必须在主图中报告。

## 六、完成状态

已完成：实验组织、6正文图+2附录图、三种导出格式、JSON来源与SHA256、静态图册、核心表格。
E1b成对评分未找到，因此只作明确标注的定性panel；未伪造数字。论文引用/版面尚未最终处理，图为可编辑工作版。
未提交、未推送，未改另一个仓库，未新增模型实验。
