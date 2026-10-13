# Manuscript skeleton（2026-10-11）

> **定位已被 `docs/paper_outline_v2.md` 取代（2026-10-12）。** 那份把研究问题
> 从「图能不能帮」提升到「模型能否利用自己维护的交互历史」，图降级为测量工具。
> 本文件保留，因为所有表格仍然有效，而且「为什么收窄到那里」是论文第 6 节的内容。
> 新大纲同时纠正了本文件里一处写法：转移结构的增益不是「主要」来自确定性 regime，
> 是**全部**（19/19），而 branching 那 11 题视觉已经满分。

定位已经收窄两次。第一次从「graph 能不能帮」收到「近确定性流程让 branching
不可识别」；第二次收到现在这句：

> **一个 transition prior 要发挥作用，需要两个性质同时成立：状态能被稳定复现，
> 并且同一状态存在多个可用的后继。现有第一视角交互数据通常只给其中一个。**

主结果不是 ladder，是下面那张两条轴的表。ladder 是它的脚注。

## 主表（`results/runs/b2_egoexo/joint_regime.json`，图 `docs/paper_joint_regime_figure.svg`）

同一个 `classify`、同一套严格前缀，两条轴分开报：

| 数据 | 单位 | 决策 | S 非空 | 命中\|S非空 | \|S\|≥2 | 其中空约束 | G-branch |
|---|---|---|---|---|---|---|---|
| 内部 release（人工合同、强制 id 复用） | 49 | 322 | 63.7% | **95.6%** | 3.42% | **9/11** | **0.62%** |
| 人工粗标 Ego-Exo4D 烹饪（活动级） | 20 | 54 | 7.4% | 0.0% | 0.00% | — | 0.00% |
| Ego-Exo4D keystep 烹饪 | 20 | 240 | 15.8% | **10.5%** | 3.75% | 0/9 | **3.75%** |
| Ego-Exo4D keystep 抗原（预测，未标注） | 30 | 356 | 19.1% | 10.3% | 4.78% | 0/17 | 4.78% |

G-novel : G-branch = **4.5 / — / 3.8 / 3.6 : 1**。

三件事要在正文里讲清楚，它们比「两个空象限」更锋利：

**① 两条轴反相关，不只是两个象限都空。** 内部数据复现 95.6% 而可用分叉 0.62%；
Ego-Exo4D 复现 10.5% 而可用分叉 3.75%（6 倍）。这不是两个独立的缺陷，是一个
权衡的两端。

**② 两种失败的机制不同，而且可分别测量。** 内部数据的分叉不是「少」，是**出现时
为空**：出度 ≥2 的 11 个决策里有 **9 个**的可达集已经包含了全部见过的候选，
条件化排除不掉任何东西。Ego-Exo4D 的分叉**不空**（0/9、0/17），但它长在不复现的
状态上。所以：
- 内部 = 有记忆、无约束；
- Ego-Exo4D = 有约束、无记忆。

这解释了 `support ≈ cache`（+0.003，CI 跨零）的**机制**，而不只是报告一个 null。

**③ `G-novel : G-branch ≈ 3.6–4.5 : 1` 跨四行稳定。** 跨数据集、跨域、跨标注者、
跨粒度。只要图开口，「这个转移没见过」压过「见过两个备选」约 4 倍。
这是全研究最不变的那个量，适合做最常被引用的一句。

## 第三轮补上的那一块：粒度细化不能制造 branching（2026-10-12）

主表里「两条轴反相关」最大的软肋是一句标准反驳：**「标细一点不就有了？」**
第三轮把它堵死了，详见 `docs/branching_pilot_b2_round3_result.md`。

修车族 10 个窗口，动作级粒度 + 强制 id 复用：

```
Ego-Exo4D keystep 粒度      G-branch 0.00 /窗口
人工动作级（如实计）         G-branch 0.50 /窗口   ← 刚够宽端阈值
去掉纯取得工具状态后         G-branch 0.00 /窗口
```

5 个 G-branch 的前驱**全部**是 `拿工具`（5/5 = 100%），而标注者写的 branch
condition 全是工具身份（「这次拿的是充气枪 / 刻刀 / 螺丝刀 / 钳子」）。
这是 EPIC 被本项目否掉的同一个 transport primitive，只不过这次由我们自己的
细化产生。

**更该进正文的是那个访问次数判据。** `|S| >= 2` 要求前驱已被离开两次，
所以必须有状态被访问 **3 次以上**：

```
真实状态最大访问次数   1 次 6 条 / 2 次 4 条 / >=3 次 0 条
取得类（拿工具）       >=3 次 4 条，最多 4 次
```

真实状态一条都没到 3 次。**粒度细化提高的是过渡动作的访问次数，
不是真实状态的复现次数** —— 而后者才是结构先验要利用的东西。
这也给出一个可以无限刷高的警告：把 `拿工具` 再拆成「伸手/握住/抬起」，
产出还能涨，所以产出率本身不能作为「域里有 branching」的证据。

一条可判定的粒度规则因此进入 Contribution 3 的评测协议：

> **状态 = 作用于什么 + 怎么作用。只为取得手段而存在的移动不建状态，
> 并入它服务的那个动作。** 自检：如果一个状态的全部内容就是「为了能做下一件事」，
> 它不是状态。

分界不在动词：`拿充气枪` 折叠，`拿充气枪给轮胎充气` 保留，
**取得零件也不折叠**（换胎而不是补胎是个决策）。

## 一个必须写成假设、不能写成发现的东西

反相关**可能是标注构造造成的**，不是数据的性质：强制 id 复用要求标注者判定两次
出现「是同一个状态」，而同一个判定会把备选也合并进同一个节点。换句话说，
制造复现的那个动作可能同时消灭分叉。

内部数据的 95.6% 恰恰来自合同里「回到做过的事要选回原代号」；Ego-Exo4D 的 10.5%
部分来自 `step_unique_id` 是「第几步」而不是「哪个状态」——回到同一个物理状态会
拿到新 id，构造上压低复现率。

**这是抗原检测那一批要回答的唯一问题**，它的页面同时带着动作级粒度规则和
id 复用规则。两种结果都能用，但**代价不对称**：
- 复现 ↑、分叉仍低 → 反相关是结构性的，论文结论更强，直接写。
- 复现 ↑、分叉也 ↑ → **主表里 Ego-Exo4D 那一行是器具产物**，必须用人工标签重做
  那一行才能投，不是「还有方法空间」那么轻。

## 章节骨架

**1. Introduction.** 问题改成：*When do temporal transition priors help maintain
interaction-state identity?* 答案：只在复现与分叉同时成立时，而现有数据不给。

**2. Setup.** 状态身份、严格前缀的决策构造、`classify` 的三类
（`src/auditor/boundary/branch_mine.py`）。强调 G-branch 的三个子句，尤其
`|S| < |candidates|`——出度高但允许全部候选的图不是约束。

**3. Does transition memory carry information?**（`g1_ladder.py`，
`results/runs/interaction_graph_ladder.json`）

| regime | n | 占比 | visual | cache | support |
|---|---|---|---|---|---|
| no-support | 117 | 36.3% | 0.607 | 0.607 | 0.607 |
| deterministic | 185 | 57.5% | 0.784 | 0.881 | 0.886 |
| branching | 11 | 3.4% | **1.000** | 1.000 | 1.000 |
| novel | 9 | 2.8% | 0.889 | 0.889 | 0.889 |
| 整体 | 322 | | 0.730 | 0.786 | **0.789** |

support 比 visual 多答对 19 题，**19 题全部来自 deterministic**，branching / novel /
no-support 各 +0。而 branching regime 只有 11 题、visual 单独已经 **11/11**。
→ 结构本该发挥作用的地方没有余量。这一段预先堵掉「多给 branching 数据就有用了」。

**写法警告：** `self / recency / frequency` 三臂的 0.7298 不是三个独立结果。CV 把
它们的 λ 选成 0，打分式退化成 `cos(x,μ)`，与 visual **逐位相同**。正确表述是
「没有任何权重能提升留出准确率，CV 因此归零」——关于权重选择，不是准确率排名。
`shuffled` 臂（λ=0.02，0.7298）是有效控制，可以报。

**4. Is it topology, or just memory?** `support = 0.7888` / `cache = 0.7857` /
`support − cache = +0.0031`，CI 跨零。机制是 ③ 里那个 9/11 空约束。
主动回答 "Why do you need a graph?"：在当前数据里不需要，它退化成 successor
memory，而这指明了未来工作应该去找什么样的环境。

**5. Does it survive online?**（`g1_online.py`，
`results/runs/interaction_graph_expD_online.json`）

| 模式 | identity top-1 | diff-state separated | prior_override |
|---|---|---|---|
| visual | 0.7298 | 0.824 | 0 |
| oracle 前驱 | **0.7857** | 0.939 | 1 |
| 预测前驱 | 0.7267 | **0.972** | 18 |
| 全自维护 | **0.6118** | 0.818 | — |

定位改成 **recurrence alone is insufficient**，不是「方法失败」。而且要用这个细节：
预测前驱那一列 `diff_state_separated` 反而**升到 0.972**（最高），整体准确率却掉回
基线。所以先验作为分离器是有效的，崩的是喂给它的身份估计。
这比「崩」准确，也让 negative result 更可信。

**6. External validation / dataset diagnosis.** B2：20 个 Ego-Exo4D 烹饪窗口，
冻结判据（`docs/branching_pilot_b2_spec.md`，冻结于 2026-09-28）打出 NO-GO。
粗标 0、keystep 9，两个粒度都是个位数。失败的子句是 `|S| ≥ 2` 而不是计数。
附预注册（`docs/branching_pilot_b2_attempt2_3_preregistration.md`），其中记录了
选族依据与代价，以及**任务形态直觉两次选中全数据集最线性的一族**（维修 0.03/窗口，
因为这里的维修 take 是被演示的规定流程而不是诊断）。

**7. What a benchmark should measure.** 不是只测 action segmentation，而是三件事：
状态复现率、可用分叉率（**带上空约束那一列**）、在线身份稳定性。
给出 `b2_joint_regime.py` 作为可直接复用的测量工具。

## Contributions

1. **Dataset diagnosis.** 现有第一视角交互数据很少同时提供评估 transition-based
   reasoning 所需的联合区间；而且两条轴在我们测到的数据上**反相关**。
2. **Transition prior analysis.** successor memory 在 oracle 下有效；graph support
   相对单条目缓存几无增益，机制是出度 ≥2 时约束多半为空；增益在在线状态估计下消失。
3. **Evaluation protocol.** 给出三项应测量的量与工具，并指出
   `G-novel : G-branch ≈ 4 : 1` 是跨数据集稳定的结构事实。

## 标题

同意 "Beyond Action Recognition" 太泛。推荐你的 Option 1：

> **When Do Temporal Transition Priors Help Egocentric Interaction-State Tracking?**
> *A Diagnostic Study of State Recurrence, Branching, and Online Reliability*

另给一个把发现直接写进标题的：

> **Recurrence Without Branching:** Why Transition Priors Collapse into
> Successor Memory in Egocentric Interaction Video

前者问题驱动、安全；后者一眼看到结论，但把论文锁死在 negative 一侧，
如果抗原那一批翻出联合区间就要改题。**先用 Option 1。**

## 路线

1. 冻结现有结果 —— 已完成，`graph-experiment` repo 两个 commit。
2. 骨架 —— 本文件。
3. 抗原检测那一批：**只做 recurrence sanity check**，不扩大规模。30 窗口已交付。
4. 按上面「代价不对称」的两个分支决定是否需要重做主表的 Ego-Exo4D 行。

**不再回头调 graph。**
