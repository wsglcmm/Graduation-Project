"""B 的影片偵測格式 → 規格 §10.3 的轉換測試（樣本取自 B 在 2026-11-01 上傳的 road01 輸出格式）。"""
import json

import pytest
from pydantic import ValidationError

from coord_transform.messages import VideoDetectionFrame
from coord_transform.video_detections import MissingTrackIdError, convert, convert_frame, main

B_ROW = {"frame_index": 3, "image": "frames/frame_000003.jpg", "timestamp": "00:00:00.060", "timestamp_ms": 60,
         "image_width": 1920, "image_height": 1080,
         "detections": [{"class_id": 2, "class_name": "car", "confidence": 0.794, "xyxy": [27.8, 352.85, 120.11, 428.57]},
                        {"class_id": 0, "class_name": "person", "confidence": 0.8358, "xyxy": [1.08, 954.55, 71.53, 1079.59]}]}


def _with_ids(row, ids):
    return {**row, "detections": [{**d, "track_id": i} for d, i in zip(row["detections"], ids)]}


def test_maps_b_fields_to_spec():
    out = convert_frame(_with_ids(B_ROW, [7, 8]), "road01")
    assert out["frame_id"] == 3 and out["timestamp_ns"] == 60_000_000  # 影片時間 60 ms
    assert out["source"] == "video" and out["video_id"] == "road01" and out["schema_version"] == "0.2.0"
    car = out["detections"][0]
    assert car == {"track_id": 7, "class": "car", "confidence": 0.794, "bbox": {"x1": 27.8, "y1": 352.85, "x2": 120.11, "y2": 428.57}}
    VideoDetectionFrame.model_validate(out)


def test_accepts_ultralytics_id_key_and_skips_unknown_classes():
    row = _with_ids(B_ROW, [1, 2])
    row["detections"][1] = {**row["detections"][1], "class_name": "dog", "id": 5}
    row["detections"][0] = {k: v for k, v in row["detections"][0].items() if k != "track_id"} | {"id": 9}
    out = convert_frame(row, "road01")
    assert [d["track_id"] for d in out["detections"]] == [9]


def test_missing_track_id_is_an_error_by_default():
    with pytest.raises(MissingTrackIdError, match="model.track"):
        convert([B_ROW], "road01")


def test_placeholder_ids_are_unique_and_obvious():
    rows = [B_ROW, {**B_ROW, "frame_index": 4, "timestamp_ms": 80}]
    out = convert(rows, "road01", placeholder_track_ids=True)
    ids = [d["track_id"] for f in out for d in f["detections"]]
    assert len(ids) == len(set(ids)) == 4 and min(ids) >= 1_000_000  # 不重複 → 不會假裝成同一個目標


def test_rejects_duplicate_timestamps():
    with pytest.raises(ValueError, match="遞增"):
        convert([_with_ids(B_ROW, [1, 2]), _with_ids(B_ROW, [1, 2])], "road01")


def test_video_source_requires_video_id():
    out = convert_frame(_with_ids(B_ROW, [1, 2]), "road01")
    with pytest.raises(ValidationError, match="video_id"):
        VideoDetectionFrame.model_validate({**out, "video_id": None})


def test_cli(tmp_path):
    src, dst = tmp_path / "b.jsonl", tmp_path / "spec.jsonl"
    src.write_text(json.dumps(B_ROW) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="track_id"):
        main([str(src), str(dst), "--video-id", "road01"])
    main([str(src), str(dst), "--video-id", "road01", "--placeholder-track-ids"])
    assert json.loads(dst.read_text(encoding="utf-8").splitlines()[0])["video_id"] == "road01"
