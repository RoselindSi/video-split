# 交互图这条线：阶段收口（2026-09-28 冻结）

当前数据线到此为止。不再从同分布池子里出新的完整标注批次。

## 收窄后的结论

> **Observed transition support is strongly predictive because local workflows are
> near-deterministic; the present dataset does not contain enough branching
> structure to test whether graphs help resolve genuinely ambiguous state choices.**

中文口径：**已观察到的转移支撑集之所以预测力很强，是因为局部流程近乎确定性；
这批数据里没有足够的分叉结构，无法检验图能否在真正有歧义的状态选择上起作用。**

这不是「Graph failed」。「branching graph 有没有价值」这个科学问题**仍然未被回答**，
因为我们从来没有足够的 branching cases。这两句话同时成立。

---

## Result 1 — VLM 的局部关系判断有两个方向相反的表示偏置

E1b，307 个 oracle-boundary 候选，两条臂：

- **看视频**：视觉连续性偏置 → 偏向 `SAME`
- **看文本**：措辞差异偏置 → 偏向 `DIFFERENT`

同一个关系问题，换一种表示就系统性地倒向另一边。任何只用单一表示的结论都带着
这个偏置，必须成对读。

## Result 2 — 把图序列化成文本，没有结构收益

E2 的表示消融阶梯（M0/M1raw/M1canon/M2raw/M2rawd/M2canon/M3/M3raw/…）：

- 长度与结构在 M0..M3raw 上完全共线，拆开之后**结构 ≈ −4 个点（不显著）、
  长度 ≈ −15 个点**
- 唯一的正面效应挂在「一行一个节点的紧凑字典」这种**格式**上，而且**内容无关**：
  把标签随机置换 +0.7%，换成无意义占位符 +2.6%
- `M2raw` 最初那个 +20.4% 是我自己的混淆——它的字典用了 canonical 名（oracle 标签）。
  加 `M2rawd` 之后效应消失（−6.6%，CI 跨 0）

结论：E2 测到的是**上下文长度和排版**，不是结构。

## Result 3 — 但图本身确实带信息（无语言模型、无提示词）

G1 只用冻结的视觉塔嵌入 + 算术先验；G1b 把 null 换成 configuration-model 双边交换
（出度入度精确不变，只打乱「谁接谁」）。n=116，只用先验排序：

| 先验 | top-1 |
|---|---|
| 均匀 | 41.8% |
| 节点频率 | 30.8% |
| **真可达集** | **80.8%** |
| 度匹配打乱 | 44.0% |

`Rt − Rs = +36.8%` [+21.0, +49.5]，而 `Rs − U = +2.2%`（CI 跨 0）。
**度匹配打乱之后一点不剩**，说明信号在成员身份上，不在度、频率或这一项的形状里。

前缀性已逐条核对：0/322 条决策的边超出 `G_<t`。

（两处更正已并入：名字序打破平局会白送「偏好最早节点」先验，频率先验 +25.2、
均匀 +8.2，改成随机平局期望后**频率先验反而低于均匀**；`真图 − 打乱图` 的 CI 跨 0，
转移**频次**的贡献不成立。）

## Result 4 — 这个优势几乎全部来自近乎确定的后继复现

branch_b1：15 段 ≤3 分钟、按 branching 结构挑到最狠、reviewer_2 按 v2.1 全新标注。
96 个决策：

| | |
|---|---|
| G-simple | 62 |
| **G-branch** | **0** |
| G-novel | 0 |

**15 段里没有任何节点的出度 ≥ 2。** 图的形状只有单节点自环（4 段）、
2-节点环（6 段）、4-节点环（5 段）。节点序列字面是环：`A B C D A B C D A B`。

**「上次从这里去了哪，这次还去哪」= 62/62 = 100%。**

跨录像也一样：`000435/000416/000438` 三条独立录像做同一件事，归到统一语义后是
同一个环 `盖→整→揭→扔→盖`，只是窗口切在不同相位；三张图并起来出度仍全为 1。

三次独立测量：最早 60 条 release → 2 个 G-branch；471 条旧池 mining → 约 5 个；
15 段精选重标 → **0 个**。这是分布事实，不是抽样运气。

### 由此得到的负向简化

当前数据支持的系统甚至不需要完整的图：

```
previous state
     ↓
cached previous successor  +  current visual evidence
```

一个「上一个状态 → 缓存的那个后继」的查表，加上当前视觉证据，就够了。
这个 negative simplification 本身是有价值的结果。

---

## 一并冻结的三件事

### Cluster / E3 暂停

**每个三分钟片段恰好只有一个 task cluster**，所有节点都在里面。
所以缺 branching 的不只是节点层，任务层同样没有 between-task 结构。
这不是 Cluster 这个概念被否定，而是**数据里没有 task-level decision points**。

### evidence_timestamp 标记为不可用，但不重标

36 个事件里 **26 个**的 `evidence_timestamp` 落在该节点第一段**结束之后**，
而 `retrospective_confirmation` 一次都没勾。这是工具 affordance 问题不是行为现象：
页面上那个按钮原本写作「用当前播放位置」，直接记录播放头、零校验，标注者编辑事件时
人在哪一帧就记成了哪一帧。

处理：
- 旧数据标记 `invalid_for_prefix_evidence_analysis`
  （`results/auditor/interaction_graph_branch_b1/evidence_timestamp_invalidation.json`），
  **不人工回填**——这 15 段的作用是 topology diagnosis，那个作用已经完成
- UI 已修：按钮改成「播放头停在证据那一帧，再点这里」，并加了 `checkTs()`——
  证据时间落在该节点第一段结束之后时当场提示并自动勾上 retrospective
- 只有修后的版本才用于新的 pilot

### 选片 proxy 的教训要留下

旧标签的「物件族切换」与真实节点切换 **r = −0.46**，是**反向**的。
预测 16 次切换的那条真实只有 1 个节点 20 个自环。
「动词会漂、物件不漂」在筛选上不成立——**物件不漂恰恰意味着物件分不出节点**。

所以以后任何选片判据，**必须在新标注上验证过才能用于扩量**，
不能再拿旧标签的 proxy 直接挑一批去做完整标注。

---

## 明确不做的事

- 不再从现有 471 池设计新 proxy → 挑 30 条 → 完整人工标注 → 指望这次有 branch
- 不为了造 branching 把节点粒度切得更细。把「拿出塞子」拆成
  「伸手→抓住→提起→移走」会制造更多节点，但那多半还是 `A→B→C→D`，
  而且牺牲了本来自然可解释的 interaction-state 粒度。那是为了方法制造 benchmark

---

## 如果还要给 Graph 最后一次公平检验

见 `docs/branching_pilot_b2_spec.md`：一个 10–20 条的 branching-native pilot，
只做粗标（state occurrence sequence），Go/No-Go 判据事前冻结。
那是**换 research regime**，不是继续修当前实验。

一句话提前说明：现有 471 条池子里，标签提到分拣/归类的 13 条、检查后决定的 11 条、
排障 0 条、中断式 0 条。**这个池子本身缺这类任务形态，pilot 需要新拍。**

---

产物索引：
`docs/interaction_graph_g1b_g1c_findings.md`（Result 3）、
`docs/branch_b1_result.md`（Result 4 全表）、
`docs/interaction_graph_branching_collection_spec.md`（选片判据的演化与失败）、
`results/auditor/interaction_graph_branch_b1/`（标注、失效标记）。
