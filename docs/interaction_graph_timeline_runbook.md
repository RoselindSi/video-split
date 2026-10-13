# 完整视频动作图标注

这个工具把一条完整视频切成连续区间。每个区间选择一个可复用事件 A、B、C；
再次选择 A 会形成回到 A 的边，勾选“这段里同一动作反复发生”会形成 A → A。

## 1. 启动现有 30 条视频

```bash
python3 -m src.auditor.boundary.interaction_graph_timeline \
  --manifest results/auditor/interaction_graph_timeline_demo/manifest.json \
  --out-dir results/auditor/interaction_graph_timeline_demo/annotations \
  --annotator reviewer_1 \
  --port 18772
```

浏览器打开 `http://127.0.0.1:18772/`。每条视频提交后，首页可导出 CSV 和
包含 nodes、edges、edge occurrences 的 JSON。

## 2. 准备任意视频文件夹

```bash
python3 -m src.auditor.boundary.interaction_graph_timeline_prepare videos \
  --video-dir /path/to/videos \
  --source-pool my_video_pool \
  --out /path/to/manifest.json \
  --include-hashes
```

将上一步的 manifest 传给标注服务器即可。

## 3. 使用手部马赛克 databag 池

本仓库已经从手部马赛克标注中整理出 280 条来源：

`results/auditor/interaction_graph_timeline_databags/catalog.json`

这些视频在数据机的 `/shared`，本机没有挂载。在能访问原视频的数据机上运行：

```bash
python3 -m src.auditor.boundary.interaction_graph_timeline_prepare catalog-videos \
  --catalog results/auditor/interaction_graph_timeline_databags/catalog.json \
  --source-pool diverse_hand_mosaic_databags \
  --out results/auditor/interaction_graph_timeline_databags/manifest.json \
  --limit 30
```

如果挂载位置改变，追加例如
`--path-map /shared/datasets=/new/datasets/location`。先用 `--limit 30` 做一批，
确认颗粒度稳定后再扩大。

## 4. 数据含义

- `segments`：覆盖整条视频、没有空白和重叠的区间。
- `events`：当前视频内可复用的事件节点及其描述。
- `A → B`：相邻区间从事件 A 转到事件 B。
- `A → A / self_transition`：两个相邻区间都选择 A。
- `A → A / internal_repeat`：一个区间内部已经重复 A，由复选框明确记录。
- `event_sequence`：按时间排列的事件路径，例如 `A, B, C, A`。

事件字母只在当前视频内有效；跨视频汇总时应根据事件描述和后续统一的语义规则再做节点对齐，
不能因为两条视频都用了 A 就直接合并。
