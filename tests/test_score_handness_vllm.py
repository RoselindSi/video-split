import base64
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_handness_vllm import hand_probability, request_payload, score


def test_payload_continues_exact_handness_prefix(tmp_path):
    full, crop = tmp_path / "full.jpg", tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    payload = request_payload(full, crop, "model")
    assert payload["messages"][-1] == {"role": "assistant", "content": '{"hand":'}
    assert payload["continue_final_message"] is True
    assert payload["add_generation_prompt"] is False
    assert payload["logit_bias"] == {"804": 100, "867": 100}


def test_probability_is_two_token_softmax():
    choice = {"logprobs": {"content": [{"top_logprobs": [
        {"token": " true", "logprob": -1.0},
        {"token": " false", "logprob": -2.0},
    ]}]}}
    probability, margin = hand_probability(choice)
    assert math.isclose(probability, 1 / (1 + math.exp(-1)))
    assert margin == 1


def test_probability_does_not_merge_distinct_whitespace_tokens():
    choice = {"logprobs": {"content": [{"top_logprobs": [
        {"token": " true", "logprob": -0.1},
        {"token": " false", "logprob": -2.1},
        {"token": "true", "logprob": -8.0},
    ]}]}}
    probability, margin = hand_probability(choice)
    assert probability > 0.8
    assert margin == 2


def test_score_accepts_manifest_image_paths(tmp_path, monkeypatch):
    full, crop = tmp_path / "full.jpg", tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "item_id", "rec", "frame", "canonical_tid", "full_path", "crop_path"])
        writer.writeheader()
        writer.writerow({"item_id": "item", "rec": "R1", "frame": 1,
                         "canonical_tid": 2, "full_path": full,
                         "crop_path": crop})

    def fake_post(url, payload):
        assert payload["messages"][0]["content"][0]["image_url"]["url"].endswith(
            base64.b64encode(b"full").decode())
        return {"choices": [{"logprobs": {"content": [{
            "token": " true", "top_logprobs": [
                {"token": " true", "logprob": -.1},
                {"token": " false", "logprob": -2.1},
            ]}]}}]}

    monkeypatch.setattr("scripts.score_handness_vllm.post_json", fake_post)
    output = tmp_path / "scores.jsonl"
    score(index, tmp_path / "unused", output, "http://local", "model")
    row = json.loads(output.read_text())
    assert row["stem"] == "item"
    assert row["item_id"] == "item"
