"""Pluggable video-vision backends for the auditor.

The auditor pipeline is deliberately model-agnostic: it builds messages and
parses JSON, and delegates the actual "watch this clip and answer" step to a
backend. Two are provided:

  MockBackend  -- no GPU, no model, no video. Returns deterministic,
                  schema-valid JSON seeded by the request text. Its ONLY
                  purpose is to smoke-test the plumbing and the scorer
                  end-to-end (it is NOT a result). Use it to prove the
                  pipeline runs before spending GPU time.

  QwenVLBackend -- transformers-based inference for the Qwen-VL family
                  (Qwen3-VL / Qwen2.5-VL, and Qwen3.5-VL when available).
                  The model id is a CLI argument; nothing here is hardcoded to
                  a specific checkpoint. Per the auditor design, run Pass A
                  (blind description) with an *Instruct* model and Pass B
                  (label comparison) with a *Think/reasoning* model -- pass two
                  ids on the command line and the driver routes each pass.

Adding a backend = subclass VisionBackend and implement generate(). The rest
of the pipeline does not change.
"""
from __future__ import annotations

import hashlib
import json
import random


class VisionBackend:
    """Interface: turn (system, user, optional video/images) into raw text."""

    name = "base"

    def generate(self, system: str, user: str, *, video: str | None = None,
                 images=(), fps: float = 2.0, temperature: float = 0.0,
                 max_new_tokens: int = 768, mock_keys=None) -> str:
        raise NotImplementedError


# --- mock backend (plumbing test only) --------------------------------------

_ENUM_HINTS = {
    # bias the mock toward plausible-but-not-perfect answers so the scorer
    # produces a non-degenerate confusion matrix during a smoke test.
    # NOTE: gt_boundary_relation / model_boundary_behavior / candidate_
    # boundary_validity / label_support / label_completeness /
    # label_granularity / semantic_relation / object_relation are no longer
    # asked of the VLM directly (see derive_fields.py) -- no hints needed.
    "temporal_truth": ["valid", "valid", "spurious", "ambiguous"],
    "semantic_action_changed": ["yes", "no", "unclear"],
    "motion_change_without_semantic_change": ["no", "yes", "unclear"],
    "visual_evidence": ["clear", "partial", "insufficient"],
    "conflicts_with_blind_description": ["no", "no", "yes"],
    "additional_action_visible": ["no", "no", "yes"],
    "agrees_with_pass_a_motion_judgment": ["yes", "yes", "no", "pass_a_uncertain"],
}


class MockBackend(VisionBackend):
    name = "mock"

    def generate(self, system, user, *, video=None, images=(), fps=2.0,
                 temperature=0.0, max_new_tokens=768, mock_keys=None) -> str:
        seed = int(hashlib.sha256((user + f"|{fps}|{temperature}").encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        out = {}
        for k in (mock_keys or []):
            if k in _ENUM_HINTS:
                out[k] = rng.choice(_ENUM_HINTS[k])
            elif k.endswith("_interval"):
                out[k] = None
            elif k.endswith("_time"):
                out[k] = round(rng.uniform(10, 500), 1) if rng.random() > 0.3 else None
            elif k.startswith("corrected_secondary") or k.startswith("observed_secondary"):
                out[k] = [] if rng.random() > 0.4 else ["rinse"]
            elif (k.startswith("corrected_primary") or k.startswith("observed_primary")
                  or k == "before_action" or k == "after_action"):
                out[k] = rng.choice(["scrub", "rinse", "fold", "wipe", "flip", "remove"])
            elif (k.startswith("corrected_object") or k.startswith("observed_object")
                  or k.startswith("object_")):
                out[k] = rng.choice(["mug", "tissue", "sink strainer", "remote control"])
            elif k in ("state_change", "rationale", "reasoning"):
                out[k] = "mock: describes a plausible state change"
            else:
                out[k] = None
        return json.dumps(out, ensure_ascii=False)


# --- Qwen-VL backend --------------------------------------------------------

class QwenVLBackend(VisionBackend):
    """transformers inference for a Qwen-VL-family checkpoint.

    Lazy-imports torch/transformers/qwen_vl_utils so the module is importable
    on a machine without them (e.g. for the mock smoke test). Instantiate once
    per model id; the driver may hold two (Instruct for Pass A, Think for Pass
    B/C).
    """

    def __init__(self, model_id: str, device: str = "cuda",
                 dtype: str = "bfloat16", max_pixels: int | None = 602112,
                 attn: str | None = None):
        self.name = f"qwen:{model_id}"
        self.model_id = model_id
        self.max_pixels = max_pixels
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except Exception:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM
        torch_dtype = getattr(torch, dtype)
        kw = {"device_map": device}
        if attn:
            kw["attn_implementation"] = attn
        self._torch = torch
        try:
            self.model = _AutoVLM.from_pretrained(model_id, dtype=torch_dtype, **kw).eval()
        except TypeError:  # transformers < ~4.56 used torch_dtype instead of dtype
            self.model = _AutoVLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _content(self, video, images, fps):
        content = []
        if video:
            item = {"type": "video", "video": video if "://" in video else f"file://{video}", "fps": fps}
            if self.max_pixels:
                item["max_pixels"] = self.max_pixels
            content.append(item)
        for img in images or ():
            content.append({"type": "image", "image": img if "://" in img else f"file://{img}"})
        return content

    def generate(self, system, user, *, video=None, images=(), fps=2.0,
                 temperature=0.0, max_new_tokens=768, mock_keys=None) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": self._content(video, images, fps) + [{"type": "text", "text": user}]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # Qwen3VL needs real per-frame timestamps to build its prompt, which
        # requires `video_metadata`, not just `fps`. With return_video_metadata=
        # True, qwen_vl_utils returns each video as a (tensor, metadata_dict)
        # tuple instead of a bare tensor -- the tensor and metadata must be
        # split apart and passed as separate `videos=`/`video_metadata=`
        # arguments (VideosKwargs.video_metadata expects list[dict], NOT a
        # tuple embedded in `videos`). Skipping this makes the processor fall
        # back to a hardcoded fps=24 sampling assumption regardless of what we
        # actually requested (confirmed via processing_utils.VideosKwargs and
        # Qwen3VLVideoProcessor.preprocess on the server -- see auditor
        # README/memory for the diagnostic trail).
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True)
        if video_inputs:
            video_tensors, video_metadata = zip(*video_inputs)
            video_inputs = list(video_tensors)
            video_kwargs["video_metadata"] = list(video_metadata)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **(video_kwargs or {})
        ).to(self.model.device)
        gen_kw = dict(max_new_tokens=max_new_tokens)
        if temperature and temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kw.update(do_sample=False)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kw)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        reply = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return reply


def build_backend(kind: str, model_id: str | None = None, **kw) -> VisionBackend:
    if kind == "mock":
        return MockBackend()
    if kind == "qwen":
        if not model_id:
            raise ValueError("qwen backend requires --model_id")
        return QwenVLBackend(model_id, **kw)
    raise ValueError(f"unknown backend {kind!r} (use 'mock' or 'qwen')")
