# Experiment D：successor-support 能否在线存活

冻结于 2026-10-08。代码 `src/auditor/boundary/g1_online.py`，产物
`expD.json`。数据沿用 `interaction_graph_release_v1` + v2 node map，嵌入沿用
G1 的 `g1_emb.npz`（378 段，1152 维）。

---

## 1. Research question

> **Does the oracle benefit of successor-support memory survive when
> interaction-state identity and historical memory must be maintained online
> by the model itself?**

对应收窄后的 formulation：

```
n̂_t = f( x_t , S(n̂_{t-1}) )          S(n) = { v : (n,v) 在前缀中被观察过 }
```

Experiment D 测的是 **oracle state memory → predicted online state memory**
这一步转换，不测别的。

这里不再使用 `G` 这个符号：算法依赖的是 persistent interaction-state
identity 加 successor-set memory，而不是节点度、全局 topology、message
passing 或 graph planning。

---

## 2. 为什么「只把 n_prev 换成预测」是不够的设计

原来的打分路径有**三处**依赖 oracle，不是一处：

| | 符号 | 在代码里 |
|---|---|---|
| 前驱身份 | `n_{t-1}` | `prior()` 读 `r["prev_node"]` |
| successor-support 历史 | `E_{<t}` | `prior()` 读 `r["edges"]`，即前缀转移计数 |
| 节点视觉原型 | `μ_n` | `proto()` 对前缀中 **按 oracle 标签分组**的 occurrence 求均值 |

只替换第一处而保留后两处，测出来的部署性会被系统性高估。这本身是一条方法论
结论，值得进 Methods/Discussion：

> Evaluating an online state-memory method by replacing only the current
> predecessor while leaving oracle prototypes and transition history intact
> materially overestimates deployability.

---

## 3. Conditions

| Condition | Previous state | Edges / support | Node prototypes |
|---|---|---|---|
| **Visual** | — | — | 仅当前视觉打分器 |
| **Oracle** | gold | gold prefix | gold-prefix grouping |
| **Pred-Prev** | predicted | gold prefix | gold-prefix grouping |
| **Online-Full** | predicted | predicted history | predicted grouping |

- **All histories are prefix-only.** 判定在更新 edges 和 seen 之前进行。
- 保留 G1 的冻结视觉打分 `cos(x_t, μ_n)`、同一融合族（两阶段：prior 留
  top-k，视觉在其中挑，低于 τ 则判 NEW）、同一 grouped-CV 协议。
- **k 与 τ 只在 Oracle 下按录像分组 CV 定一次（k=1, τ=0.90），其余条件沿用
  不再重拟。** 每个条件各自调参会把退化吸收进拟合，比较就不再是关于历史
  来源的了。
- Online-Full 在答 NEW 时自行铸造新节点 id，因此候选库也由模型自己演化。
  其身份评分用「预测簇 → gold 节点」的多数映射，这个映射对把一个状态拆成
  两个却前后一致使用的模型是宽容的，所以 Online-Full 是 identity 质量的
  **上界**。

规模：322 个决策 / 49 条录像 / 378 段有嵌入。

---

## 4. Main result

| Mode | Accuracy | Δ vs visual | 95% CI |
|---|---:|---:|---:|
| Visual only | 0.730 | — | — |
| **Oracle** | **0.786** | **+0.056** | **[+0.008, +0.123]** |
| Pred-Prev | 0.727 | −0.003 | [−0.070, +0.077] |
| **Online-Full** | **0.612** | **−0.118** | **[−0.225, −0.033]** |

按录像聚类的配对 bootstrap，4000 次，seed 20260928，录像为重抽单位。

> Oracle successor support yields a small but detectable gain; none of that
> gain survives predecessor prediction, and fully self-maintained memory is
> substantially worse than the visual-only baseline.

逐级退化，**只报数值，不做进一步归因**：

```
Oracle → Pred-Prev     − 5.9 pp
Pred-Prev → Online-Full − 11.5 pp
```

> The larger degradation occurs after graph memory and node prototypes are
> also made self-maintained.

**不能据此说 prototype 是主因或 edges 是主因**——这两者在本实验中是一起
替换的，本实验不区分它们。

辅助口径（配对一致性拆正负，因为合并算会被大量「不同节点」负例抬高）：

| Mode | 同态连上 | 异态分开 | 最长错误串 | 恢复率 |
|---|---:|---:|---:|---:|
| Visual | 0.848 | 0.824 | 12 | 0.529 |
| Oracle | 0.866 | 0.939 | 4 | 0.594 |
| Pred-Prev | 0.866 | 0.972 | 9 | 0.398 |
| Online-Full | 0.741 | 0.818 | 8 | 0.504 |

---

## 4b. 基线阶梯：可达集比不比一条缓存强

`src/auditor/boundary/g1_ladder.py`，产物 `ladder.json`。

G1 证明了信号是真的（真 support 对度匹配 shuffle）。它**没有**证明信号需要一个
**集合**。本域「上次从这里去了哪这次还去哪」是 62/62，所以每个状态只留一条缓存
后继的方法可能就拿走全部收益——而更便宜的方法应当默认获胜。Experiment D 的 CV
选出 k=1 让这个疑问变得具体：prior 独自定下短名单，视觉只是确认。

统一成同一融合族（`score = cos(x, μ_n) + λ·prior`，每臂自选 λ/τ，录像分组 CV）：

| 臂 | λ | deterministic | overall |
|---|---|---|---|
| visual only | 0.00 | 0.784 | 0.730 |
| self（n == n_prev） | **0.00** | 0.784 | 0.730 |
| recency | **0.00** | 0.784 | 0.730 |
| frequency | **0.00** | 0.784 | 0.730 |
| **cached successor** | 0.05 | 0.881 | **0.786** |
| **full support set** | 0.10 | 0.886 | **0.789** |
| shuffled（度匹配 null） | 0.02 | 0.789 | 0.730 |

录像聚类配对 bootstrap，4000 次：

```
support − visual      +0.059  CI [+0.013, +0.125]  不跨零
support − cache       +0.003  CI [+0.000, +0.011]  跨零
```

> **No evidence that a full successor set is worth more than a single cached
> successor.** The entire measurable gain over vision is already taken by
> "last time I left this state I went to X".

**`self` / `recency` / `frequency` 的 λ 全被 CV 选成 0.00**，即 CV 把它们整个
关掉、逐位等于 visual。所以那三行与 `support − visual` 是同一个数，不是三次
独立比较。有用的只有 transition-specific memory，不是一般的时间先验。

**度匹配 null 落在 0.730 = visual**，所以 +0.059 不是「有个先验就行」，而是
特定于**哪些状态历史上真的跟在当前状态之后出现过**。

### 为什么集合赢不了缓存：假设空间本来就只有两个元素

**候选集中位 2 个，最大 5，99.7% ≤ 4。** `|S| = 2` 配 `|candidates| = 2` 等于
什么都没排除，这正是 G-branch 判据第二条失败、branching regime 只剩 11 条
决策的原因。

这比「这个域没有分叉」更能说明问题：**不是图没找到分叉，是假设空间一开始就
只有两个候选，没有留给图发挥的空间。** 图要有用，需要的是

```
S(A) = {B, C, D}      候选 = {B, C, D, E}      → E 被排除，视觉在 B/C/D 中选
```

而本域大量实际是

```
A → B,  A → B,  A → B      候选 = {B, C}       → 猜 B 就够
```

`branching` 列 n=11、`novel` 列 n=9，统计上是空的（visual 自己就 1.000），
标题数字只能读 `deterministic` 与 `overall`。

### 统一融合后的完整链

Experiment D 原本用两阶段 top-k 而阶梯用软融合，两者不可直接串联：候选集中位
只有 2，所以 D 的 k=1 是 prior 最强的形态，和阶梯的 λ=0.10 不是同一件事。
`g1_online --fusion soft` 重跑后同族可比：

| | 准确率 | Δ vs visual | CI |
|---|---|---|---|
| visual only | 0.730 | — | — |
| + cached successor | 0.786 | +0.056 | — |
| + full support（= Oracle） | 0.789 | +0.059 | [+0.013, +0.125] 不跨零 |
| + support，预测前驱 | 0.746 | +0.016 | [−0.046, +0.092] 跨零 |
| + support，全自维护 | 0.615 | −0.115 | [−0.226, −0.025] 不跨零 |

**交叉验证**：`g1_online` 的 oracle 臂与 `g1_ladder` 的 support 臂是两份独立
实现，同融合下均为 **0.789**，三位小数一致。

**两种融合给出一致结论但塌陷程度不同**，两者都要报：预测前驱时
两阶段 −0.003（跨零）、软融合 +0.016（跨零）——增益都消失，软融合没那么狠。
对应的 prior 排除正确答案次数：两阶段 1 / 18 / 0，软融合 0 / 12 / 0，即
**软融合比硬门控更抗错误前驱**，因为足够强的视觉分能压过错误的 prior。这是
soft-prior-vs-hard-constraint 那个问题的方向性证据。

---

## 5. Mechanism

### 5.1 Prior exclusion errors

视觉本来排第一且正确、而 prior 的短名单把它排除掉的决策数：

```
Oracle:        1
Pred-Prev:    18
Online-Full:   0
```

> A wrong predecessor queries the wrong successor set, causing the structural
> prior to actively suppress visually correct candidates.

即：

```
identity error  →  wrong support  →  correct state excluded
```

**identity drift 变成了 structural error propagation。**

Online-Full 的 0 不是优点：它的 `S` 由自己的 edges 在自己的节点 id 上建起，
与自身判断自洽，所以几乎不排除什么——prior 在那里是**失去信息**而不是
**排除错误**。那 −11.8 pp 来自自维护流程的别处，本实验不分离。

### 5.2 CV 选出 k = 1

当前确定性域的最优策略近似于：

```
previous state  →  one cached successor  →  visual confirmation
```

与 `branch_b1` 的 **62 / 62 = 100%**（「上次从这里去了哪这次还去哪」）完全
呼应：两者是同一件事的两种测法。

两条合起来是这个项目目前最核心的一句话：

> **The oracle system succeeds precisely in the regime where successor memory
> is almost deterministic; once state identity is uncertain, the same strong
> structural prior becomes brittle.**

---

## 6. Interpretation / consequence

**Finding D1**
> Oracle successor support contains usable predictive information, but the
> gain is modest (+5.6 pp).

**Finding D2**
> The gain does not survive predicted predecessor identity.

**Finding D3**
> Fully online state-memory maintenance is currently harmful because identity
> errors contaminate both successor support and node representations.

因此当前方法的状态**不是**「差一点就可部署」，而是：

> **oracle phenomenon 已成立，但 online mechanism 尚未成立。**

这应当作为 manuscript 的核心 limitation 出现在正文，不是 appendix。

> This does not rule out successor priors in branching regimes. It shows that
> the current deterministic domain provides an unusually brittle operating
> point: the prior is strong when correct, but catastrophic when state
> identity drifts.

### 对实验闭环的影响

```
Local state reasoning          DONE   (E1)
Representation diagnostics     DONE   (E2)
Oracle successor signal        DONE   (G1/G1b)
Null topology control          DONE   (degree-matched shuffle)
Online predicted tracking      DONE — negative   (D)
Branching regime               MISSING
Novel transitions              MISSING
Fresh confirmation             MISSING
```

写 **diagnostic / analysis manuscript**：实验闭环已经相当完整。
写 **method paper**：只剩一个不可替代的洞：

> 在真正 branching 的数据上，successor-support prior 能否在不发生
> deterministic-cache brittleness 的情况下帮助视觉 disambiguation？

因此主问题调整为：

> **When can historical successor support improve interaction-state tracking
> in egocentric video, and can that benefit survive online state estimation?**

因为 D 已经证明：**oracle usefulness ≠ deployable usefulness。**

### 不再做的

**不在旧数据上继续拆 `edges` vs `prototypes`。** 除非新 branching 数据回来后
online 仍然失败，否则继续拆只会再次钻进局部诊断。旧数据已经回答够多了。

---

## analysis_corrections

三处分析错误在本轮被发现并修正，留痕备查。

**C1 缺视觉-only 地板。** 第一版只报绝对准确率，而判据（「oracle +10 而
predicted +1 则方法不成熟」）问的是**相对增益**，没有地板无法换算。已补
Visual 条件。

**C2 配对一致性口径把松当成严。** 第一版把同/异 gold 的配对合并成一个
`pairwise_strict`，结果 Pred-Prev 得 0.899 高过 Oracle 的 0.888 —— 因为
「不同节点」的配对占绝大多数，负例把数字抬了上去，它测的是更容易的东西而
不是更严的。已拆成「同态连上 / 异态分开」两栏；identity 真正关心的是前者。

**C3 bootstrap 的点估计与 CI 符号相反。** `Online-Full` 点估计 −0.118 而
CI 曾算出 [+0.029, +0.190]，两者不可能同时成立。原因是为了填逐行的 `hit`，
对每一行单独调了 `score([r], mode)`；而 Online-Full 模式下用**单行**去建
「预测簇 → gold」的多数映射，映射对那一行必然正确，于是命中率被虚抬。已
改为整批 `score(res, mode)` 一次填完再按录像分桶。

**C5 度匹配 null 曾是空操作。** 阶梯第一版的 shuffled 臂与 support 臂**逐格
完全相同**（0.786，各 regime 也一致）。原因是置换的是 `edges` 的**计数**，而
support set 按**键是否存在**定义，所以置换对集合成员身份毫无影响。G1 那边有效
是因为它的 prior 用计数而非集合。改为保住出度、重新随机抽取**目标**后 null 才
生效（落到 0.730 = visual）。两列完全相同本该当场判为不可能。

**C6 两阶段门控在小候选集上是空的。** 候选集中位 2、99.7% ≤ 4，而 CV 给
recency/frequency 选了 k=4，于是 `top-k` 等于全部候选、偏好被整个旁路，
visual/self/recency/frequency 给出**逐位相同**的 0.730。改为软融合
`cos + λ·prior` 后才测得到，而那三臂的 λ 仍被选成 0.00 —— 结论不变，但第一版
是「测不到」而不是「测到了零」。

**C4 一处超出证据的归因（口头，未入文档）。** 曾表述为「大头在原型和
edges，不在前驱」。`edges` 与 `prototypes` 在本实验中一起替换，只能得出
联合退化大于前驱预测单独造成的退化，不能分别归因。本文档第 4 节已按此
口径表述。
