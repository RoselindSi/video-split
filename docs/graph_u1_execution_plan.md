# Graph 后续实验：U1 执行顺序（2026-10-16）

> 状态更新：U1 已于 2026-10-16 关闭，不再追加实验。最终结论与条件见 graph_u1_final_result.md；本计划以下内容仅保留为事前顺序和停止判据的审计记录，不再是待执行队列。

来源：用户指定的“研究方向看法”对话；本文件将建议变成分阶段执行协议。
目标：检验保留前驱身份不确定性是否改善 successor-memory 查询，以及收益能否在在线写入后保留。
已有 graph NO-GO 仍然有效；U1 不据此宣称 branching graph 已有效。

## 0. 冻结输入、复现 D（首先执行）

- 复用 g1_emb.npz、release_v1、v2 node map；预期 378 段 embedding、322 decisions、49 videos。逐项核对实际值与文件 SHA256。
- 同时复现 g1_online 的 twostage 和 soft 两臂。引用对话中的 0.7298/0.7857/0.7267/0.6118 属于 twostage；不能拿它和 soft 的结果直接作方法差值。
- 保存命令、代码版本、输入校验和、CV 划分、参数与逐决策预测。若不能复现，先定位差异，不进入 U1。
- 本机已有 embedding，先 CPU 单进程运行；无需启动 vLLM、占用 GPU 或修改 cgroup。设置 OMP/MKL/OPENBLAS 线程数为 2。

## 1. 固定查询接口与公平对照

固定视觉 embedding、候选集、NEW 判定与 oracle-prefix prototypes/edges，只改变 predecessor query。
新增共同的 soft-log fusion 家族：visual/T + lambda*log(p_history+epsilon)。
必须包含 V、O、P、B、PG、BG：无记忆、oracle、point、belief、point+gate、belief+gate。
PG 是额外必要对照：否则 BG 的收益无法区分来自 belief 还是简单减弱 prior。
同时报告旧 D soft 作为历史基线，不将融合函数变化算作 belief 收益。

预先明确以下边界：
- q 只对当时已见状态及 NEW 建模，不读取当前或未来 gold。
- NEW、无 outgoing memory 的 predecessor 质量回退到候选上的均匀分布；不得丢弃这部分质量后重新归一化来虚增置信度。
- cache 与 support 分开做；cache 用最后一次已观察 successor，support 在已观察 successors 内均匀分配。
- T、lambda、epsilon、kappa、写入阈值的候选表在首次 U1 评测前落盘冻结，lambda 必须含 0。
- T 用训练录像校准；模型选择使用录像分组的嵌套 CV，所有臂相同预算。外层录像不可用于选参。
- K=1 时 identity gate 定义为 1；其他情况使用归一化 entropy。memory evidence 使用 belief 加权的前缀出边观测数。
- 既报各臂独立选参结果，也报匹配参数的机制对照；不把调参优势当查询优势。

## 2. B / BG：先量查询，不量在线写入

主比较：B-P、BG-PG；BG-P 为组合效应。按录像配对 bootstrap，4000 次，固定种子。
报告 top1、helpful rescue、harmful override、NEW/novel 错误、same-state linked、different-state separated。
按相同的 point predecessor 是否正确分层，并报告 gold predecessor 的 belief mass、entropy 与校准误差。
分层仅用于诊断，不能用 held-out gold 选择 gate。

进入下一阶段的关卡：BG-P 的区间下界 >0，且不是仅 lambda=0 回退到视觉；若 BG-PG 无收益，仅支持 gating，不能归因为 belief。
没有通过时停止扩大 online 方法；结果解释为“当前协议下未发现 uncertainty query 的收益”，不能据此证明 prototype 是唯一瓶颈。

## 3. Controlled corruption（机制实验，与第 2 步配套）

噪声率固定 0/0.1/0.2/0.3/0.4，多随机种子，所有臂共用抽样。
hard 与 belief 必须从同一含噪 observation 产生，belief 不得额外知道正确 predecessor。
区分可校准的不确定噪声与高置信错误；保存实际 top1 error、gold mass 和 entropy。
参数从干净训练录像冻结，不能在每个测试噪声点重新优化。
比较 accuracy/rescue/override 随噪声变化；不预设曲线一定改善。

## 4. 全在线：仅在第 2 步通过后执行

比较 FP、full-belief、full-point+write-gate、FBG；拆开 soft read 与 gated write 的贡献。
分别消融 prototype write gate 和 transition write gate；边写入要求前后两个端点都满足资格。
低置信预测仍推进时间；不得跨过未写入片段制造不存在的相邻转移。
NEW 节点创建/保留、拒写后的候选可见性必须显式定义，沿用 D 的评分映射并注明其宽容性。
主差值 FBG-FP，同时比较 FBG-V；前者改善而后者未改善仅算部分正结果。

## 5. Structural opportunity / 粒度诊断

复用现有 census、B2 coarse/keystep/folding 产物，统一 R、successor diversity、gold-in-support、视觉歧义口径。
将“有多个后继”与“能排除候选”“视觉需要帮助”分开统计；分母与窗口长度全部保留。
该分析可与 U1 独立完成，但不将 recurrence+branching 写成所有 history 收益的必要条件：确定性 cache 已经有收益。
仅对评估非平凡 successor-set 相对 cache 的额外价值讨论结构机会。

## 6. 收口与交付

每阶段输出 spec/config、输入 SHA、逐决策预测、汇总 JSON、可复现命令和结论；保留负结果。
不因结果好坏替换数据、视觉表示或评测分母。已反复看过的 49 条标为既有 benchmark；需要新数据才能支撑最终独立泛化验证。
本轮先落实第 0–1 步，再按关卡推进；不同时启动所有实验，不推送、不改其他会话任务。
