# Time-R1 集成说明(approach B)

把 video-split 的模块接进服务器 `/workspace/tr1/time-r1`。改动尽量小,便于回滚。

## 1. 部署我们的文件

把这些拷进 Time-R1 repo(建议软链或直接 copy):

```
video-split/src/rewards/seg_rewards.py  -> time-r1/src/seg_rewards.py
video-split/src/prompt/seg_prompt.py    -> time-r1/src/seg_prompt.py
video-split/src/data/convert_multiseg.py-> time-r1/tools/convert_multiseg.py
video-split/src/eval/eval_multiseg.py   -> time-r1/eval_multiseg.py   (待写)
```

## 2. 生成训练数据(服务器上跑)

```bash
cd /workspace/tr1/time-r1
python tools/convert_multiseg.py \
  --dataset_dir /workspace/datasets/task_segmentation_annotation_dataset_v1 \
  --out dataset/handtask/train_multiseg.json
```

## 3. main.py — 注册 reward(~3 行)

```python
from src.seg_rewards import SEG_REWARD_FUNCS
reward_funcs_registry.update(SEG_REWARD_FUNCS)   # 放在 registry 定义之后
```

## 4. timer1_trainer.py — 加 prompt_type "seg"(~4 行)

```python
from src.seg_prompt import QUESTION_TEMPLATE_SEG   # 顶部

# make_conversation_video 内:
if self.prompt_type == "seg":
    prompt_text = QUESTION_TEMPLATE_SEG            # 不注入 [EVENT]
```

## 5. main.py `create_dataset_from_json` — 必改(已核实,原版硬编码单段)

原版读 `item["timestamp"]`+`item["sentence"]`,把 solution 写死成 `(s,e)` 二元组,
喂多段格式会直接崩。在循环里加 multi-seg 分支(检测 `solution` 是 list):

```python
for item in data:
    video_path = item.get("video")
    if isinstance(item.get("solution"), list):        # ← 多段分支
        if not os.path.isfile(video_path):
            continue
        examples.append({
            "task_type": "seg",
            "problem": "",                            # prompt_type=seg 忽略
            "choices": "",
            "solution": json.dumps(item["solution"]),  # JSON 串,躲 Arrow 混合类型坑
            "video_path": video_path,
            "durations": item.get("duration"),
            "video_start": None, "video_end": None,
            "preprocessed_path": "",
        })
        continue
    # ... 原有单段逻辑保持不变 ...
```

要点:
- `solution` 存成 **JSON 字符串**(`seg_rewards._as_segs` 会 `json.loads` 回来)。
- `video` 字段用**绝对路径**(convert_multiseg.py 已输出绝对路径)。
- reward 函数拿到的 `solution`/`durations` 就是这两列,自动经 reward_kwargs 传入。

## 6. 待核实(SFT 冷启动)

确认 `finetune.py` 能吃 `render_target()` 目标文本;否则用最小 TRL SFTTrainer 先训格式。

## 7. 训练顺序(课程式)

```
Stage0  SFT: 目标文本 = seg_prompt.render_target(solution, reasons)
Stage1  GRPO --reward_funcs "iou_seg format_seg"  --prompt_type seg
Stage2  GRPO --reward_funcs "iou_seg name_seg format_seg"
Stage3  GRPO --reward_funcs "iou_seg name_seg seq format_seg"
```
每 stage 后用 eval_multiseg.py 在 held-out 上看 Seg-F1。
