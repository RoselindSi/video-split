# Manuscript claim—证据索引

2026-10-16。配 graph_manuscript_working_draft.md。路径默认相对 video-split；graph-experiment 文档为旧来源，不默认使用其全部旧措辞。

| Claim / 正文位置 | 证据产物 | 分母/协议 | 必须保留的限制 |
|---|---|---|---|
| §3.1 successor identity 有 oracle 信息 | results/auditor/interaction_graph_g1b_summary.json | 116 decisions / 25 videos，纯先验排序 | 41.8% 是均匀，不是视觉；频率低于均匀为点估计，差值区间跨零 |
| §3.2 support 与 cache 无可区分额外收益 | results/auditor/interaction_graph_ladder 2.json | 322 / 49，旧 ladder | 无差异不是普遍等价证明；self/recency/frequency lambda=0，不是独立正结果 |
| §3.2 19 题增益来自 deterministic | 同上 regimes 与 arms | 185 deterministic、11 nominal branching | branching 已视觉满分，不能评估额外收益 |
| §4 branch_b1 稀缺 | graph-experiment/docs/branch_b1_result.md | 96 decisions | 手工子集，不能推广为所有域 |
| §4 B2 NO-GO | graph-experiment/docs/branching_pilot_b2_result.md / spec.md | 20 粗标窗口，冻结门槛 | 是该采集/粒度协议的停止判决，不是 graph 普遍失败 |
| §4 取得工具造成表面分叉 | graph-experiment/docs/branching_pilot_b2_round3_result.md | 10 人工动作级窗口 | 与 30 keystep 窗口非完全配对；不写“所有细化都无用” |
| §4 联合区间稀疏/窗口依赖 | graph-experiment/docs/paper_figures_and_framing.md；paper_limitations.md | 16 keystep scenarios + 其他点 | CPR 反例、rho=-.535 是倾向；不能说区间为空 |
| §5.1 D oracle→online 落差 | results/graph_u1/d_twostage_reproduction.json；d_soft_reproduction.json | 322 / 49，原代码复现 | 旧调参非独立外层；prev NEW gold alignment；full 同时改 edges/prototypes |
| §5.2 U1 最终主表 | results/graph_u1/u1_causal_nested_calibrated.json | 同 322 / 49，5×4 nested video CV | oracle memory，query-only；P lambda=0；不可与旧 D 直接算方法提升 |
| §5.2 4 题差异 | results/graph_u1/u1_causal_rescue_audit.json | 4 cases / 4 videos | 只有 1 个非 top1 独有 rescue；不是 branching 收益 |
| §5.2 简单软化对照/cache | results/graph_u1/u1_causal_softening.json | 同 outer folds；混合系数内层选择 | M 额外搜索预算；只排查该具体收缩方式 |
| §5.2 true-vs-shuffled | results/graph_u1/u1_causal_memory_null.json | 20 fixed seeds × 49 videos | 边目标实际改变约47.7%；CI 跨零；不能省略 null 只报 B>V |
| §5.2 固定 NEW bias | results/graph_u1/u1_causal_known_bias.json | 同 outer/inner folds | 等价视觉阈值细化；B-K CI 跨零 |
| §6 E2 最新数值 | results/auditor/paper_fig4_e2_measured.json | 307 / 49，10-12 重算 | 旧 +20.4→-6.6 未复现；采用 +11.4→-3.9；标签对照是差值非绝对收益 |
| U1 停止判据、最终条件 | docs/graph_u1_final_result.md；configs/graph_u1_query.json | BG-P 关卡未满足 | FBG 未运行，不写为测过后失败 |

## 代码入口

- 旧 G1b/ladder/D/B2：graph-experiment/src/auditor/boundary 下对应脚本；本轮只复现 D，不声称重跑所有旧实验。
- U1 query：scripts/graph_u1_query.py（最终主表必须同时给 --causal --calibrate-query）。
- 校准与诊断：scripts/graph_u1_calibration.py、graph_u1_diagnostics.py。
- 最终控制：graph_u1_rescue_audit.py、graph_u1_softening.py、graph_u1_memory_null.py、graph_u1_known_bias.py。
- 冻结记录：results/graph_u1/freeze_manifest.json；最终数字汇总：u1_final_summary.json。

## 表格/图的正文分配

1. Content 表：G1b（独立标明纯先验任务）。
2. Complexity 图：ladder regime 分解与19题净增益归属。
3. Opportunity 图：scenario scatter +窗口/粒度敏感性，小样本反例保留。
4. Online 表：D 两种融合分别报告。
5. U1 小表：最终因果主表与 true/null；详细参数、4题、迭代审计入附录。
6. E2 图：只使用 measured 版本，标明效应是哪些臂的差值。

## 还没完成的写作项

- 相关工作与文献引用：目前未做新的检索，不声称 novelty 已被文献验证。
- 旧 G1b/ladder 的估计/选参协议在 Methods 中逐项确认，不能套用 U1 的 nested-CV 标签。
- 生成并视觉检查整合后的正式图表与投稿版式；当前只是 Markdown 工作稿。
- 将本轮内容同步到 graph-experiment 的正式 manuscript 位置需另行执行；本轮没有改那个仓库。
- 文件引用按实际文件名定位，注意旧产物名包含空格和“2”，不得假定去掉后缀的路径也存在。

这些是编辑与审计待办，不重新打开 U1 实验队列。
