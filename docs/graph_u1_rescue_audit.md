# U1 四题审计与非零权重对照（2026-10-16）

脚本 scripts/graph_u1_rescue_audit.py；完整逐题数值与匹配预测在 results/graph_u1/u1_causal_rescue_audit.json。
固定因果快照、训练折校准温度和 B 在内层选出的视觉 T/tau；分别用同一组 lambda 比较 point 与 belief。不挑选外层表现最好的 lambda。

## 四个 rescue 不是四个 uncertainty retrieval 成功

| 决策 | 因果前驱 posterior | 当前答案 | 相同参数 point 能否救回 |
|---|---|---|---|
| mosaic_graph_0015#06 | top1=NEW 0.738，正确 C 仍有 0.233 | D | 否 |
| mosaic_graph_0022#05 | top1=B 0.759，正确 | A | 是 |
| graph_review_0019#04 | top1=A 0.528，正确 | B | 是 |
| graph_review_0008#10 | top1=A 0.865，正确 | A | 是 |

第一题：C→D 是已观察转移。hard point 查询 NEW 回退均匀，视觉把当前判成 NEW；belief 保留 C 的质量，将 D 的 prior 提高到约 0.380，高于 NEW 的 0.148，恰好纠正视觉边界决策。这是直接符合假设的一题。
其余三题前驱 point 正确，point 和 belief 都能提供有效转移证据。它们是 history rescue，不是 belief 相对 point 的独有 rescue。
这四题用到的已知后继都是单后继，不构成 branching graph 的收益证据。

## 匹配非零权重

下表都使用同折 B 选出的视觉参数和训练折查询校准温度。rescues/overrides 相对这一共同视觉对照计数。

| lambda | Point accuracy | Belief accuracy | Point rescue / override | Belief rescue / override |
|---|---:|---:|---:|---:|
| .05 | 71.43% | 74.22% | 10 / 14 | 6 / 1 |
| .10 | 71.12% | 73.91% | 13 / 18 | 8 / 4 |
| .20 | 70.81% | 72.98% | 17 / 23 | 10 / 9 |

B-P 在 .05/.10 下均 +2.80pp，录像聚类 bootstrap CI 分别 [+1.02,+5.02]pp 和 [+0.77,+5.14]pp；.20 下 +2.17pp、CI [-0.56,+5.02]pp。
内层 B-selected lambda（部分折为 0）下：P=71.74%、B=73.91%，差 +2.17pp、CI [+0.62,+4.15]pp。
这些是同一旧 benchmark 上的敏感性/机制对照，不能当多次独立确认，也不能挑 .05 宣布最优部署权重。

## 能说什么

1. 目前 belief 的相对优势主要表现为减少 harmful override，而不是比 point 救回更多题；point 提供的强 prior 同时能救更多题、也伤更多题。
2. 至少有一题确实利用了非 top1 前驱质量，说明该机制存在；一题不足以证明普遍性。
3. 五折独立选参的 P 关闭记忆，这里强制非零是检验机制，不是主评测应强迫模型使用 prior。
4. 不支持完整 graph 优于 successor cache；gating 没有通过关卡，继续不做全在线写入。

## 下一步

在同一因果/校准接口下比较单 successor cache、support 和保持相应 nuisance 统计的打乱 memory。需要判断 belief 的减害是否只是任何弱化 prior 都能实现：加入 point prior 与 uniform 的混合对照，混合系数仅在训练折选择。
旧乐观接口下的 noise 曲线不能直接用于这个因果接口。若后续仍有独立于软化 prior 的收益，再做对应 corruption 和全在线最小验证。
