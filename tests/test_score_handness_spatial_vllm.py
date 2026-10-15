import csv
import json

import pytest

from scripts.score_handness_spatial_vllm import (
    parse_spatial_answer,
    request_payload,
    score,
)


def test_request_disables_thinking(tmp_path):
    full, crop = tmp_path / "full.jpg", tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    payload = request_payload(full, crop, "model")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0


def test_parse_spatial_answer_accepts_hand_box_and_no_hand():
    assert parse_spatial_answer(
        '{"hand":true,"bbox":[10,20,800,900]}') == (
            True, [10, 20, 800, 900])
    assert parse_spatial_answer(
        '```json\n{"hand":false,"bbox":null}\n```') == (False, None)


@pytest.mark.parametrize("answer", [
    '{"hand":true,"bbox":null}',
    '{"hand":false,"bbox":[1,2,3,4]}',
    '{"hand":true,"bbox":[500,0,400,1000]}',
])
def test_parse_spatial_answer_rejects_inconsistent_output(answer):
    with pytest.raises(ValueError):
        parse_spatial_answer(answer)


def test_score_supports_concurrent_requests(tmp_path, monkeypatch):
    full, crop = tmp_path / "full.jpg", tmp_path / "crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    index = tmp_path / "index.csv"
    fields = ["item_id", "rec", "frame", "canonical_tid", "human_label",
              "full_path", "crop_path"]
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_id in ("a", "b"):
            writer.writerow({"item_id": item_id, "rec": "R1", "frame": 1,
                             "canonical_tid": 1, "human_label": "hand",
                             "full_path": full, "crop_path": crop})

    def fake_post(_url, _payload):
        return {"choices": [{"message": {"content":
                '{"hand":true,"bbox":[1,2,3,4]}'}}]}

    monkeypatch.setattr("scripts.score_handness_spatial_vllm.post_json", fake_post)
    output = tmp_path / "scores.jsonl"
    score(index, output, "http://local", "model", workers=2)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["item_id"] for row in rows] == ["a", "b"]
    assert all(row["parse_ok"] for row in rows)
