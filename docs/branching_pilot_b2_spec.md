# B2：branching-native pilot（判据事前冻结，未启动）

当前数据线已收口（`docs/interaction_graph_stage_closed.md`）。这份是**如果**还要给
Graph 一次公平检验时要走的路：换 research regime，不是继续修当前实验。

启不启动由你定。下面的门槛在看到任何数据之前冻结，写在这里就是为了事后不能改。

## 为什么不能从现有池子里继续挑

三次独立测量都指向同一个分布事实（2 / ~5 / 0 个 G-branch）。而且选片 proxy 被证明是
**反向**的（r = −0.46）。再从同分布里抽 30、50、100 条不是谨慎，是重复验证一个已经
稳定的结论。

而现有 471 条池子里，标签提到**分拣/归类的 13 条、检查后决定的 11 条、排障 0 条、
中断式 0 条**——这还只是「标签里出现了相关词」，不等于真有选择点。
**这个池子缺的是任务形态本身，所以 pilot 必须新拍。**

## 要拍什么：天然带 `|N⁺(A)| ≥ 2` 的任务

判据不是语义类别，是**同一个状态之后必须有多于一个合理去处，且由当时的观察决定**。

```
分拣          Inspect item ──┬── bin A
                             ├── bin B
                             └── reject bin

排障          Inspect device ┬── tighten component
                             ├── replace component
                             └── continue inspection

复杂料理      Check state ───┬── continue cooking
                             ├── add ingredient
                             └── remove from heat

中断式流程    Task A ────────┬── continue A
                             ├── service interruption B
                             └── retrieve tool C
```

拍摄时明确**不要**的：固定顺序的重复操作（拧盖子 ×20、装卸手机壳 ×20）。
那正是当前数据里占绝大多数的形态，产出恒为 0。

同时沿用已验证的两条硬约束：
- **手部可见**：≥100px 手框的出现帧率 ≥ 60%（`branch_hand_check.py --floors 60 100`）。
  裸检测率不能用，它在这批数据上把最差的一条排到了第一
- 单条标注单位 ≤ 3 分钟

## 第一阶段只做粗标

**不做完整 semantic graph annotation。** 只要 state occurrence sequence：

```
A B A C A B A D
```

够算 `outdegree(A) = 3` 就行。标注者只需要给每个状态一个稳定的代号并保持一致
（「回到之前做过的那件事，用同一个代号」）——这正是当前合同里最关键、也最容易漂的
那一条，粗标阶段先把它单独练出来。

规模：**10–20 条录像**。

## 要算的四个量

全部在严格前缀 `G_<t` 下算（工具已有：`branch_mine.classify`）：

1. 有多少录像存在 `outdegree ≥ 2`
2. 每条录像有多少 prefix-valid **G-branch** 决策
3. 有多少 **G-novel**（gold ∉ `N⁺(A)` 且 `N⁺(A)` 非空）
4. candidate count 分布

## Go / No-Go（在看到数据之前冻结）

**继续**，当且仅当同时满足：

- **≥ 15–20 个真正 prefix-valid 的 G-branch 决策**
- 这些决策**来自多个独立的 video / task**（不是一条录像贡献全部）
- 产出率 **≥ 0.5–1 个 G-branch / 录像**

满足则扩到：**50–60 个 G-branch + 约 30 个 G-novel + 简单 transition controls**。

**停止**，如果针对 branching task 专门挑选之后，仍然只有个位数 branch cases。
那时最多浪费十几条粗标，而不是又一批完整标注。

## 工具现状

已具备、可直接复用：

| 用途 | 位置 |
|---|---|
| 决策分型（G-simple/branch/novel，严格前缀） | `src/auditor/boundary/branch_mine.py` |
| 手部可见度（按框宽下限） | `src/auditor/boundary/branch_hand_check.py` |
| 抽帧联系表 + 画检测框 | `src/auditor/boundary/branch_contact_sheet.py` |
| ≤3 分钟窗口切分 | `src/auditor/boundary/branch_window.py` |
| 打包（label-blind + 校验和） | `src/auditor/boundary/build_graph_timeline_offline_packet.py` |
| 支持 Range 的本地服务器（视频拖动必需） | `src/auditor/boundary/serve_packet.py` |
| 产出统计 + 合同字段用量 | `src/auditor/boundary/branch_b1_yield.py` |

要新写的只有一个**粗标页面**：一条时间轴，一个「在此切开」，一个状态代号下拉
（含「新建代号」），没有 object/action/result/cluster/confidence 那一整套。
完整 v2.1 页面只在 pilot 通过 Go 之后才用上。

## 发包前的检查单（都是踩过的坑）

- [ ] 视频用 H.264（`_to_h264()`）。OpenCV 默认 `mp4v` 浏览器放不了，会让整批表回不来
- [ ] 并排双目要裁一半（`crop=640:480:0:0`）
- [ ] 规则页做泄漏机检：词表从**本批自己的源标签**自动抽，中英都要，不要手写列表
- [ ] `--readme-rules` 是**替换**统一规则块不是追加，通用规则必须自带
- [ ] 自己在浏览器里完整走一遍：切段、建状态、回到旧状态、存草稿、导出
- [ ] 代码传服务器用 `base64 | ssh 'base64 -d > f'`，**末尾不能接 `&`**（会写成 0 字节），
      传完 md5 对比
