# U1 匹配参数与噪声诊断（2026-10-16）

脚本 scripts/graph_u1_diagnostics.py；产物 results/graph_u1/u1_query_diagnostics.json（含逐决策结果，约 60 MB，不宜直接提交）。复用上一轮外层划分与训练数据中心，不重选参数。

## 匹配参数

| 内层选参来源 | Point | Belief | Point+gate | Belief+gate |
|---|---:|---:|---:|---:|
| Point | 80.75% | 72.98% | 74.53% | 72.98% |
| Belief | 80.12% | 74.84% | 74.53% | 73.29% |

同一行的 T/tau/lambda 完全相同。Point 参数下 B-P=-7.76pp，录像聚类 bootstrap CI [-15.39,-2.01]pp；Belief 参数下 -5.28pp，CI [-11.83,-0.63]pp。选参优势不是全部解释。

Point 参数下前驱 top1 正确率 98.37%（排除 special-reset），但 gold 的平均概率质量仅 0.484。平均 entropy confidence 为 0.0264。记忆权重 P=0.1494，PG/BG 约 0.00129。边缘化稀释与门控过弱都是当前实现中可见的现象。

## 合成 posterior 干预

固定噪声概率 0/.1/.2/.3/.4，10 个种子。每个 segment 使用同一确定性随机数，所有臂共享同一被扰动 posterior。
confident_swap 交换 posterior 最大项与另一个随机项，保留熵；uncertain 在交换后混入 50% 均匀质量。命名不代表 posterior 本来校准或高置信。干预概率不是实际前驱错误率；每点同时保存实际正确率。

| swap 概率 | Point | Belief | Point+gate | Belief+gate |
|---|---:|---:|---:|---:|
| 0 | 80.75% | 72.98% | 74.53% | 72.98% |
| .2 | 76.93% | 72.48% | 74.01% | 72.95% |
| .4 | 73.73% | 71.93% | 73.54% | 72.83% |

表中是 confident_swap，10 种子平均。Point harmful override 平均 8→13.8→17；BG 约 0，但主要因为没有使用多少 memory。较平的曲线不等于准确率更高，也不能单凭它宣布 robustness 方法成功。

## 当前结论和下一步

暂不启动 full-online。这里的前驱观察进入 oracle-prefix 原型，前驱 top1 本来就接近满分；不是 D 的递归预测前驱接口。
温度目前以内层任务 accuracy 联合选择，尚未按前驱 NLL/ECE 校准。因此不能把本轮负结果解释成不确定性本身无效，也不能将低 entropy gate 解释成模型正确表达了无知。

下一步先完成训练折内前驱概率校准（包含 NEW 的定义与可识别性审计），固定视觉打分尺度，单独改变查询温度；再复查 point/belief/gate。区分当前 oracle-aligned 诊断和真正递归身份维护，不把二者的结果混作一条方法提升曲线。
本轮噪声实验是 synthetic posterior intervention，不是已校准 observation-noise 模型。cache 尚未实现，本阶段不声称 support 相对 cache 有额外价值。
