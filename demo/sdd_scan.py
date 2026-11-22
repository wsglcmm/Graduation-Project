"""掃描 Stanford Drone Dataset（SDD）的標註，找出最適合當 Demo「兔子」的影片與目標。

SDD 每段影片有一個 annotations.txt，每行一筆：
    track_id xmin ymin xmax ymax frame lost occluded generated "label"
（lost = 1 代表目標在畫面外，那些點不能用；標註為 30 fps。）

挑選標準（對應 Demo 需求）：
    - 類別是車輛或騎車的人（移動快、轉彎多，包圍才有看頭）
    - 「連續在畫面內」的時間夠長（預設 ≥ 30 秒），而且真的有移動（排除停著的車）
執行：python -m demo.sdd_scan <SDD 資料夾>   （資料夾內有 annotations/<scene>/<videoN>/annotations.txt）
"""
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

FPS = 30.0
# SDD 標籤 → 規格 §3 的類別（Biker 是騎自行車的人，Demo 以 bicycle 處理；Cart 為校園高爾夫球車）
LABEL_TO_CLASS = {"Car": "car", "Bus": "bus", "Cart": "car", "Biker": "bicycle", "Pedestrian": "person", "Skater": "person"}


@dataclass
class TrackStats:
    video: str
    track_id: int
    label: str
    longest_visible_s: float
    path_length_px: float
    mean_box_px: float


def parse_annotations(path) -> Dict[int, list]:
    """回傳 track_id → [(frame, cx, cy, box_size, lost, label), ...]（依 frame 排序）。"""
    tracks: Dict[int, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 10:
                continue
            tid, x1, y1, x2, y2, frame, lost = (int(v) for v in parts[:7])
            label = " ".join(parts[9:]).strip('"')
            tracks.setdefault(tid, []).append((frame, (x1 + x2) / 2, (y1 + y2) / 2, ((x2 - x1) + (y2 - y1)) / 2, lost, label))
    for rows in tracks.values():
        rows.sort()
    return tracks


def track_stats(video: str, tid: int, rows: list) -> TrackStats:
    """最長一段「連續在畫面內（lost = 0、frame 相鄰）」的長度與移動距離。"""
    best_len, best_path, cur_len, cur_path, prev = 0, 0.0, 0, 0.0, None
    for frame, cx, cy, _, lost, _ in rows:
        if lost or prev is None or frame != prev[0] + 1:
            cur_len, cur_path = (0 if lost else 1), 0.0
        else:
            cur_len += 1
            cur_path += math.hypot(cx - prev[1], cy - prev[2])
        if cur_len > best_len:
            best_len, best_path = cur_len, cur_path
        prev = (frame, cx, cy) if not lost else None
    sizes = [r[3] for r in rows if not r[4]]
    return TrackStats(video, tid, rows[0][5], best_len / FPS, best_path, sum(sizes) / len(sizes) if sizes else 0.0)


def scan(root, labels=("Car", "Bus", "Cart", "Biker"), min_seconds=30.0, min_path_px=300.0) -> List[TrackStats]:
    out = []
    for ann in sorted(Path(root).glob("annotations/*/*/annotations.txt")):
        video = f"{ann.parent.parent.name}/{ann.parent.name}"
        for tid, rows in parse_annotations(ann).items():
            if rows[0][5] not in labels:
                continue
            st = track_stats(video, tid, rows)
            if st.longest_visible_s >= min_seconds and st.path_length_px >= min_path_px:
                out.append(st)
    out.sort(key=lambda s: (s.label not in ("Car", "Bus", "Cart"), -s.longest_visible_s))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="找出 SDD 中適合當 Demo 兔子的目標")
    ap.add_argument("sdd_root")
    ap.add_argument("--min-seconds", type=float, default=30.0)
    ap.add_argument("--out", default="output/sdd_candidates.csv")
    args = ap.parse_args(argv)
    found = scan(args.sdd_root, min_seconds=args.min_seconds)
    print(f"找到 {len(found)} 個候選（車輛優先，再依連續可見時間排序）：")
    for s in found[:15]:
        print(f"  {s.video:20s} track {s.track_id:4d}  {s.label:6s} → {LABEL_TO_CLASS.get(s.label, '?'):8s}"
              f"  連續 {s.longest_visible_s:5.1f} s  移動 {s.path_length_px:6.0f} px  框約 {s.mean_box_px:4.0f} px")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "label", "class", "longest_visible_s", "path_length_px", "mean_box_px"])
        for s in found:
            w.writerow([s.video, s.track_id, s.label, LABEL_TO_CLASS.get(s.label, ""), round(s.longest_visible_s, 1), round(s.path_length_px), round(s.mean_box_px)])
    print(f"完整清單：{args.out}")
    return found


if __name__ == "__main__":
    main()
