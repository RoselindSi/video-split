# U1 第 0 步：D 基线复现（2026-10-16）

在本机 CPU 上运行 graph-experiment/src/auditor/boundary/g1_online.py，使用 video-split 的 release 和 node map。两次均退出 0，未修改原代码。
数据规模：378 段 embedding、322 decisions、49 videos。

| 模式 | twostage | soft |
|---|---:|---:|
| Visual | 0.729814 | 0.729814 |
| Oracle | 0.785714 | 0.788820 |
| Pred-prev | 0.726708 | 0.745342 |
| Full | 0.611801 | 0.614907 |

twostage 选择 k=1、tau=0.90；soft 选择 lambda=0.10、tau=0.90。两种方法的数字不能混用。
完整 JSON/log 在 results/graph_u1/d_{twostage,soft}_reproduction.*；输入及源码 SHA256 在同目录 input_hashes.json、release_hashes.json。

## 复现揭示的协议限制

1. 源码将录像分组，对每组评分后取平均来选参数，但没有在每个外层 fold 排除该 fold 后重新选参；最后又报告全量表现。因此这是旧结果复现，不是独立外层留出评估。U1 必须采用嵌套录像 CV，并重新计算所有共同对照。
2. prev 模式答 NEW 后将内部前驱设置为当前 gold_node。这是 oracle 身份对齐接口，不能称为完全预测身份维护。U1 保留这个版本用于比较时必须明确标记；完全在线分支不可使用 gold ID。
3. 嵌入预处理对整个 embedding 文件做均值中心化。严格外层留出版本需在训练 fold 估计中心，并将改变后的所有基线重新报告，不把预处理变化记作 belief 收益。
4. 原 D 的逐决策结果是函数内部变量，输出文件只有汇总；U1 实现需要额外导出逐决策记录才能可靠计算 rescue、override 和分层比较。

首次运行在 Windows GBK 输出上因 Unicode 减号失败；重跑设置 PYTHONIOENCODING=utf-8 并使用 python -X utf8，主结果和置信区间均完成。

下一步：先冻结 U1 的候选、NEW 质量处理、参数网格及外层划分，建立共同 soft-log point 对照，再实现 belief 和 gate。不能将旧 D 与新融合函数的差值归因于不确定性建模。
