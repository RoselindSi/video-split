# E0 冻结：schema 开发 / 合同诊断集

2026-09-26 冻结。**此后不再回头调整这个数据集，也不再为个例修改合同。**

## 这批是什么，不是什么

| | |
|---|---|
| **是** | schema-development / contract diagnostic set。用来暴露 schema 里真正会导致不同表示的自由度。 |
| **不是** | 人际一致性（IAA）测量。reviewer_1 按 v1 合同标、reviewer_2 按 v2/v2.1 标，语义已变；两轮之间 reviewer_2 对这 11 条也早已不盲。 |

正式的 IAA 要用**双方都没见过的新视频**、在冻结合同下重做一次，那件事不挡主线。

## 两轮做了什么

```
v2   11 条 / 64 段   全部 final
v2.1 11 条 / 64 段   10 条 final，1 条因工具 bug 卡住（已修，见下）
```

## 找到的合同缺口与处置

| 缺口 | 证据 | 处置 |
|---|---|---|
| §8.3① 重复记法二选一 | 44 个丢失的 v1 界有 18 个（41%）落在「段内反复」跨度内 | **已修**：v2.1 §1 occurrence 规则，连续重复不切段，只有可观察的完成/重置/重新开始才产生新 occurrence |
| 持续等待被并入相邻节点 | `0020` 的 21 秒等待并进了 A，`waiting` 零次使用 | **已修**：v2.1 §4，持续且可辨认的独立等待必须单独切出 |
| 跨 special 间隔合成边 | — | **已修**：v2.1 §4.1，完整时间轴保留，不删节点再拼接 |
| 可观察结果与推测混写 | `0010/A` 的 `visible_result` 填「选择到合适的摩托车」 | **已修**：拆成 `observed_action` / `observed_result` / `inferred_goal` |
| 孤儿事件与空簇能通过提交 | 2 个全空事件 + 2 个无成员簇进了 final | **已修**：v2.1 §5 校验加严 |
| 删任务的按钮藏在 `#nodeBox` 里 | 段类型为 `uncertain` 时该盒隐藏，校验却要求删孤儿簇 → 标注者卡死 | **已修**：按钮移到独立区块 |

## 最重要的发现：证据窗口，不是规则

三条运输案例逐帧核对之后（`transport_evidence/`）：

| 案例 | 段 | 结果证据落在哪 | 两轮判定 |
|---|---|---|---|
| `0002/C` 取凳子 | 57.1–97.4 | **全在段内**（t=58 空手走出，t=93 抱着凳面走回） | 位移 + 交互节点，两轮一致，合同一致 |
| `0026/D` 杯子 | 67.3–81.2 | **主要在段外**（t=77 只看到一张桌子，秤要 t=80 才勉强可辨，t=85 才无歧义——已出段） | 两轮都判位移 |
| `0022/D` 茶叶 | 113.3–120.0 | **不存在**（视频在段中结束） | 两轮都判位移 |

**分歧不在规则，在证据窗口。** 标注者只用段内看得到的东西；我当初判 `0026/D` 为交互节点，用的是段尾之后才出现的证据。
26 个事件里 `retrospective_confirmation` 只被勾 1 次——不是漏勾，是他根本没在用未来证据。

一处更正记录在案：我曾写「t=77 秤就在同一帧里可见」，**这是夸大**。那一帧只能看到远处一张桌子上摆着东西，分辨不出哪个是秤。

**他的口径反而更接近 E4 要评的设定**（online tracking 在时刻 t 看不到 t 之后）。

## 遗留，不再追

- `0022/D` 保留为 **unresolved / uncertain exemplar**。视频在段中结束，任何人都拿不到结果证据。不为它改合同。
- v2.1 的运输三步判断顺序目前只剩这一条真分歧，**样本量 1，不足以说规则失败**。要判断得换一批没见过的运输案例，不在本轮。
- `mosaic_graph_0010` 的第 11 条，修好后可补交；补不补都不影响上面的结论。

## 冻结的东西

```
原始标注
  ~/Downloads/reviewer_2_e0_trial_v2_final.json      第一轮 11/11
  ~/Downloads/reviewer_2_e0_trial_v21_backup.json    第二轮 10/11 + 草稿
  ~/Downloads/判断小测.pdf                            uncertain 可达性小测 9/9

合同与手册
  docs/interaction_graph_schema_v2.md
  docs/interaction_graph_schema_v2_1.md              生效版
  docs/interaction_graph_annotation_manual_v2_1.md

分析
  src/auditor/boundary/interaction_graph_structure_census.py
  src/auditor/boundary/interaction_graph_case_sheet.py
  src/auditor/boundary/interaction_graph_e0_agreement.py
  src/auditor/boundary/build_uncertain_probe.py
  src/auditor/boundary/build_v21_seed_draft.py
  results/auditor/interaction_graph_schema_v2/        普查、映射、案例、选择依据
  ~/Downloads/transport_evidence/                     三条运输案例的逐帧证据

工具
  src/auditor/boundary/interaction_graph_timeline_v21_offline_page.html
  src/auditor/boundary/serve_packet.py                支持 Range（python -m http.server 不支持）
```

## uncertain 小测的结论

```
选了 uncertain   1 / 9        用在唯一真正看不到东西的片段上，理由正确
考虑过 uncertain 4 / 9        三次「考虑了又否掉」的理由都是「反复看之后看清了」
```

**可达性成立**——不是界面问题，不是找不到。上一轮 64 段零使用不能归因于可达性。

对照组有缺陷需记录：我用 v1 标注者自己写的「看不到动作 / idle」来选「明确不可判」的对照，**三条里两条其实能判**（一条透过玻璃反光能看见，一条 v1 标成 idle 但实际有动作）。
用被审计的标签去选样本，这一组的结论强度比设计时低。真正的不可判对照只有 1 条。
