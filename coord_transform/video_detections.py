"""把 B 的影片偵測輸出轉成規格 §3 + §10.3 的 VideoDetectionFrame。

B 的 main.py（2026-11-01）每幀輸出：
    {"frame_index": 0, "image": "frames/frame_000000.jpg", "timestamp": "00:00:00.000", "timestamp_ms": 0,
     "image_width": 1920, "image_height": 1080,
     "detections": [{"class_id": 2, "class_name": "car", "confidence": 0.79, "xyxy": [x1, y1, x2, y2]}]}

對應：frame_index → frame_id、timestamp_ms → timestamp_ns（影片時間）、class_name → class、xyxy → bbox，
補上 schema_version、source = "video"、video_id。

track_id：規格要求 ByteTrack 的追蹤 ID（同一目標連續幀一致）。B 目前用 model.predict，沒有 ID。
C 不在這裡自己做追蹤，避免和 B 的責任重疊；預設遇到缺 ID 就報錯。
--placeholder-track-ids 會給每個偵測一個不重複的 ID（不代表任何連續性），只用來先測地面標定與畫面，
不能拿來評估追蹤。B 改用 model.track(..., tracker="bytetrack.yaml", persist=True) 後就不需要。

執行：python -m coord_transform.video_detections <B 的 detections.jsonl> <輸出.jsonl> --video-id road01
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .messages import CLASSES, VideoDetectionFrame

VIDEO_SCHEMA_VERSION = "0.2.0"
TRACK_ID_KEYS = ("track_id", "id")  # Ultralytics 的 boxes.id；B 可能用任一個名稱


class MissingTrackIdError(ValueError):
    pass


def _track_id(det: dict) -> Optional[int]:
    for k in TRACK_ID_KEYS:
        v = det.get(k)
        if v is not None:
            return int(v)
    return None


def convert_frame(row: dict, video_id: str, placeholder_ids: Optional[Iterator[int]] = None) -> dict:
    """B 的一行 → §10.3 的 dict（已通過 VideoDetectionFrame 驗證）。不在規格 6 類的偵測直接略過。"""
    dets = []
    for d in row.get("detections", []):
        cls = d.get("class_name", d.get("class"))
        if cls not in CLASSES:
            continue
        tid = _track_id(d)
        if tid is None:
            if placeholder_ids is None:
                raise MissingTrackIdError(
                    f"第 {row.get('frame_index')} 幀的偵測沒有 track_id：請 B 改用 model.track(..., tracker='bytetrack.yaml', persist=True)，"
                    "或加 --placeholder-track-ids 只做畫面測試"
                )
            tid = next(placeholder_ids)
        x1, y1, x2, y2 = d["xyxy"] if "xyxy" in d else (d["bbox"][k] for k in ("x1", "y1", "x2", "y2"))
        dets.append({"track_id": tid, "class": cls, "confidence": float(d["confidence"]),
                     "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}})
    frame_id = row["frame_index"] if "frame_index" in row else row["frame_id"]
    t_ns = int(row["timestamp_ns"]) if "timestamp_ns" in row else int(round(float(row["timestamp_ms"]) * 1_000_000))
    out = {"schema_version": VIDEO_SCHEMA_VERSION, "frame_id": int(frame_id), "timestamp_ns": t_ns,
           "image_width": int(row["image_width"]), "image_height": int(row["image_height"]),
           "detections": dets, "source": "video", "video_id": video_id}
    VideoDetectionFrame.model_validate(out)  # 轉出來的一定要符合規格
    return out


def convert(rows: Iterable[dict], video_id: str, placeholder_track_ids: bool = False) -> List[dict]:
    ids = itertools.count(1_000_000) if placeholder_track_ids else None  # 從很大的數開始，一看就知道是佔位
    out = [convert_frame(r, video_id, ids) for r in rows]
    out.sort(key=lambda f: f["timestamp_ns"])
    for a, b in zip(out, out[1:]):
        if b["timestamp_ns"] <= a["timestamp_ns"]:
            raise ValueError(f"影片時間戳必須遞增：frame {a['frame_id']} 與 {b['frame_id']} 相同")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="B 的影片偵測 → 規格 §10.3")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--placeholder-track-ids", action="store_true", help="缺 track_id 時給不重複的佔位 ID（只供畫面測試）")
    args = ap.parse_args(argv)
    with open(args.src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    try:
        frames = convert(rows, args.video_id, args.placeholder_track_ids)
    except MissingTrackIdError as e:
        raise SystemExit(f"❌ {e}") from None
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")
    n = sum(len(fr["detections"]) for fr in frames)
    print(f"轉換 {len(frames)} 幀、{n} 個偵測 → {args.dst}" + ("（track_id 是佔位值，不能評估追蹤）" if args.placeholder_track_ids else ""))


if __name__ == "__main__":
    main()
