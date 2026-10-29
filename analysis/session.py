"""讀寫 session 資料夾（格式見 analysis/log_format.md），以及真值的時間插值。"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from coord_transform import rotations
from coord_transform.frames import ENU_TO_NED, gazebo_pose_to_ned
from coord_transform.messages import CLASSES
from coord_transform.pixel_to_ned import Pose

FILES = ("meta.json", "poses.jsonl", "detections.jsonl", "truth.jsonl")


def read_jsonl(path) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{e}") from e
    return out


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


@dataclass
class TruthTrack:
    """一個 Gazebo model 的真值序列（ENU）。"""

    t: np.ndarray  # (N,) int64 ns，遞增
    pos_enu: np.ndarray  # (N, 3)
    quat_enu_flu: Optional[np.ndarray] = None  # (N, 4)，只有提供 rpy_rad 時才有

    def _bracket(self, t_ns):
        if len(self.t) == 0 or t_ns < self.t[0] or t_ns > self.t[-1]:
            return None
        i = int(np.searchsorted(self.t, t_ns))
        if self.t[i] == t_ns:
            return i, i, 0.0
        return i - 1, i, (t_ns - self.t[i - 1]) / (self.t[i] - self.t[i - 1])

    def position_enu(self, t_ns) -> Optional[np.ndarray]:
        b = self._bracket(t_ns)
        if b is None:
            return None
        i0, i1, a = b
        return (1 - a) * self.pos_enu[i0] + a * self.pos_enu[i1]

    def rpy_enu(self, t_ns) -> Optional[tuple]:
        b = self._bracket(t_ns)
        if b is None or self.quat_enu_flu is None:
            return None
        i0, i1, a = b
        return rotations.quat_to_euler(rotations.slerp(self.quat_enu_flu[i0], self.quat_enu_flu[i1], a))


@dataclass
class Session:
    path: Path
    meta: dict
    poses: List[dict]
    frames: List[dict]
    truth: Dict[str, TruthTrack] = field(default_factory=dict)

    @property
    def origin_enu(self) -> np.ndarray:
        o = self.meta["ned_origin_enu_m"]
        return np.array([o["x"], o["y"], o["z"]], dtype=np.float64)

    @property
    def ground_plane_d_m(self) -> float:
        """地面在 NED 的 d 值。原點通常是無人機停在地上時 base_link 的位置，會比地面高一點，
        所以地面的 d = 原點 z − 地面 z（> 0）。meta.json 沒給 ground_z_enu_m 時假設地面 z = 原點 z。"""
        ground_z = self.meta.get("ground_z_enu_m", self.origin_enu[2])
        return float(self.origin_enu[2] - ground_z)

    @property
    def drone_name(self) -> str:
        return self.meta["drone_name"]

    @property
    def targets(self) -> Dict[str, str]:
        return self.meta.get("targets", {})

    def target_ned(self, name: str, t_ns: int) -> Optional[np.ndarray]:
        """目標在 t_ns 的 NED 位置（已扣掉起飛點原點）。"""
        p = self.truth[name].position_enu(t_ns)
        return None if p is None else ENU_TO_NED @ (p - self.origin_enu)

    def drone_pose_ned(self, t_ns: int) -> Optional[Pose]:
        """無人機在 t_ns 的真值 Pose（NED、FRD，已扣掉原點）。"""
        track = self.truth[self.drone_name]
        p, rpy = track.position_enu(t_ns), track.rpy_enu(t_ns)
        if p is None or rpy is None:
            return None
        rel = p - self.origin_enu
        return gazebo_pose_to_ned(rel[0], rel[1], rel[2], *rpy)


def load_session(path) -> Session:
    path = Path(path)
    missing = [f for f in FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{path} 缺少 {missing}（格式見 analysis/log_format.md）")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    for key in ("ned_origin_enu_m", "drone_name"):
        if key not in meta:
            raise ValueError(f"meta.json 缺少 {key}")
    bad = {k: v for k, v in meta.get("targets", {}).items() if v not in CLASSES}
    if bad:
        raise ValueError(f"meta.json 的 targets 類別必須是 {CLASSES} 之一：{bad}")

    rows: Dict[str, list] = {}
    for r in read_jsonl(path / "truth.jsonl"):
        rows.setdefault(r["name"], []).append(r)
    truth = {}
    for name, rs in rows.items():
        rs.sort(key=lambda r: r["timestamp_ns"])
        t = np.array([r["timestamp_ns"] for r in rs], dtype=np.int64)
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"truth.jsonl 中 {name} 的時間戳有重複")
        pos = np.array([[r["position_enu_m"][k] for k in "xyz"] for r in rs], dtype=np.float64)
        quat = None
        if all("rpy_rad" in r for r in rs):
            quat = np.array([rotations.euler_to_quat(*(r["rpy_rad"][k] for k in ("roll", "pitch", "yaw"))) for r in rs])
        truth[name] = TruthTrack(t, pos, quat)
    if meta["drone_name"] not in truth:
        raise ValueError(f"truth.jsonl 沒有無人機 {meta['drone_name']!r} 的資料")
    if truth[meta["drone_name"]].quat_enu_flu is None:
        raise ValueError("truth.jsonl 的無人機資料每一行都必須有 rpy_rad")
    missing_targets = [n for n in meta.get("targets", {}) if n not in truth]
    if missing_targets:
        raise ValueError(f"meta.json 列出的目標在 truth.jsonl 找不到：{missing_targets}")

    poses = sorted(read_jsonl(path / "poses.jsonl"), key=lambda m: m["timestamp_ns"])
    frames = sorted(read_jsonl(path / "detections.jsonl"), key=lambda m: m["timestamp_ns"])
    return Session(path, meta, poses, frames, truth)


def angle_diff(a: float, b: float) -> float:
    """a − b，包到 [-π, π)。"""
    return (a - b + math.pi) % (2 * math.pi) - math.pi
