# B2：branching-native topology-yield pilot（已决定启动，2026-09-28）

**这是一个严格受限的 topology-yield pilot，只回答一个问题：**

> 换成天然存在选择点的任务之后，能不能稳定地产生 prefix-valid branching decisions？

它**不是**当前 Graph 实验的下一批数据，**不是**对下一篇论文的承诺。

```
当前 hand-manipulation dataset  →  CLOSED（不再碰）
B2                              →  NEW DOMAIN / NEW REGIME
```

这个区分要保持到底。将来 B2 即使成功，也应该写成：原始 hand-manipulation domain
揭示了 deterministic-transition limitation；branching-native data 用于检验该限制
之外的结构推理。B2 失败则同样有清楚的停止理由。

下面的门槛在看到任何数据之前冻结，写在这里就是为了事后不能改。

## 为什么不能从现有池子里继续挑

三次独立测量都指向同一个分布事实（2 / ~5 / 0 个 G-branch）。而且选片 proxy 被证明是
**反向**的（r = −0.46）。再从同分布里抽 30、50、100 条不是谨慎，是重复验证一个已经
稳定的结论。

而现有 471 条池子里，标签提到**分拣/归类的 13 条、检查后决定的 11 条、排障 0 条、
中断式 0 条**——这还只是「标签里出现了相关词」，不等于真有选择点。
**这个池子缺的是任务形态本身，所以 pilot 必须新拍。**

## 要拍什么：天然带 `|N⁺(A)| ≥ 2` 的任务

### 判据一（最重要）：同一个前驱必须反复遇到不同的后继

要的是：

```
A → B
...
A → C
...
A → ?        ← 这里才有 branching
```

**不是**：

```
A → B      C → D      E → F
```

后者整段任务可以很复杂，但没有任何节点的出度 ≥2，产出仍然是 0。
所以选任务的标准要写死成一句话：

> **录像必须让同一种可观察状态 / 决策状态在同一条录像内重复出现，
> 并且根据当时的条件进入不同的后续状态。**

「任务看起来有分支」不算数。

### 判据二：分支要由可观察状态驱动，不能人为随机换路线

好的：

```
inspect item              inspect component
   ├─ recyclable → bin A     ├─ loose   → tighten
   ├─ non-recyc  → bin B     ├─ damaged → replace
   └─ uncertain  → 再看一次    └─ normal  → continue
```

不好的：

```
这次 A→B，下次故意 A→C
```

如果分支只是为了造图而随机制造，那么即使最后 Graph 有用，也说明不了它对应
真实的 sequential reasoning。

### 判据三：分支点要记一个最小的 branch condition 字段

粗标仍然很轻，但每个产生分叉的前驱多记一个可选短字段：

```
A = inspect object
A → B  because: object type 1
A → C  because: object type 2
```

不是完整语义标注，只是防止最后拿到一堆

```
A → B
A → C
```

却不知道**为什么**分叉。因为最终真正要问的不是「A 后面历史上出现过 B/C 吗」，
而是：

> 图给出 `{B, C}` 这个合法后继集合之后，当前的视觉证据能不能决定到底走 B 还是 C？

也就是那句一直想要的：

```
Graph          = possible trajectory prior
Visual evidence = route selector
```

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

**继续**，当且仅当**同时**满足：

- **≥ 15–20 个真正 prefix-valid 的 G-branch 决策**
- 这些决策**来自多个独立的 video / task**——不能全由一条异常高分支的录像贡献
- 产出率 **≥ 0.5–1 个 G-branch / 录像**
- **并且至少一部分分支是由可观察条件驱动的**（判据二、三），不是人为随机路径

**停止（硬性）**：如果刻意选了 branching-native 任务之后，仍然只有个位数 G-branch，
就**直接停掉 Graph 作为当前的主方法方向**。

这条 No-Go 是这份 spec 里最要紧的一行。B2 是 Graph 最后一次低成本的 feasibility
test，**不能再演变成「再拍 20 条看看」**。

### 过了 Go 之后也不要马上做完整 v2.1 标注

先在粗序列上跑最简单的结构检查：

```
previous state  →  N⁺(previous)  →  candidate reduction
```

确认新数据里确实出现 `|N⁺(A)| ≥ 2` **并且** `|N⁺(A)| < 现有候选总数`
（可达集必须真的排除掉东西，否则约束是空的）。然后再看这些 branch decision 里
有多少属于 visual easy / visual ambiguous / novel transition。

**只有拓扑本身合格，才值得做完整的视觉与语义标注。**

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
