"""抓拍判定（裁判）：模擬「這架無人機的相機有沒有拍到一張合格的目標照片」。

判定條件（CaptureCriteria，之後寫進規格）：
    1. 目標落在影像中央區域：|u − cx| ≤ center_fraction × W / 2，|v − cy| ≤ center_fraction × H / 2
    2. 目標在影像中夠大：目標長度 × fx / 高度 ≥ min_target_px（nadir 相機近似）
    3. 1、2 連續成立 hold_s 秒（避免一閃而過也算）
每架無人機各自計時；一段連續成立的期間只記一次抓拍。任何角色拍到都記錄，並標示當時的角色，
報告可以分開看「抓拍機拍到」與「其他角色順便拍到」。

裁判用的是目標真值（模擬相機實際會拍到什麼），不是融合後的估計；Agent 與規劃器只拿得到估計值。
座標：目標 world_ned (n, e, d = 0)，單位 m；無人機為 fusion.DroneState。
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from coord_transform import CameraMount, ned_to_pixel
from coord_transform.config import CameraConfig


@dataclass(frozen=True)
class CaptureCriteria:
    center_fraction: float = 0.2  # 中央 20% × 20% 的範圍（30 m 高時約 ±5 m × ±7 m）
    min_target_px: float = 40.0
    target_length_m: float = 4.5  # 車長；行人約 0.5–0.6 m
    hold_s: float = 2.0


@dataclass(frozen=True)
class CaptureEvent:
    t_s: float  # 判定成立的時間（連續成立滿 hold_s 的那一刻）
    drone_id: str
    role: Optional[str]
    pixel_offset: tuple  # 判定當下目標相對影像中心的偏移 (du, dv)，px
    target_px: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class CaptureJudge:
    def __init__(self, cam: CameraConfig, criteria: CaptureCriteria = CaptureCriteria()):
        self.intr = cam.intrinsics
        self.crit = criteria
        self.mount = CameraMount(pitch_rad=-math.pi / 2, yaw_rad=0.0, offset_body_m=(0.0, 0.0, 0.0))  # 同 ViewModel：nadir
        self._since: Dict[str, Optional[float]] = {}
        self._done: Dict[str, bool] = {}
        self.events: List[CaptureEvent] = []

    def in_frame(self, drone, tn: float, te: float) -> Optional[tuple]:
        """條件 1、2 成立時回傳 (du, dv, target_px)，否則 None。"""
        uv = ned_to_pixel((tn, te, 0.0), self.intr, self.mount, drone.pose())
        if uv is None or drone.alt_m <= 0:
            return None
        du, dv = uv[0] - self.intr.cx, uv[1] - self.intr.cy
        c = self.crit
        if abs(du) > c.center_fraction * self.intr.width / 2 or abs(dv) > c.center_fraction * self.intr.height / 2:
            return None
        px = c.target_length_m * self.intr.fx / drone.alt_m
        return (du, dv, px) if px >= c.min_target_px else None

    def update(self, t_ns: int, drones: Sequence, roles: Mapping[str, str], tn: float, te: float) -> List[CaptureEvent]:
        """每個模擬步呼叫一次；回傳這一步新成立的抓拍。roles：drone_id → 目前角色。"""
        t = t_ns / 1e9
        new = []
        for d in drones:
            hit = self.in_frame(d, tn, te)
            if hit is None:
                self._since[d.drone_id], self._done[d.drone_id] = None, False
                continue
            if self._since.get(d.drone_id) is None:
                self._since[d.drone_id] = t
            if not self._done.get(d.drone_id) and t - self._since[d.drone_id] >= self.crit.hold_s - 1e-9:
                self._done[d.drone_id] = True
                ev = CaptureEvent(round(t, 3), d.drone_id, roles.get(d.drone_id), (round(hit[0], 1), round(hit[1], 1)), round(hit[2], 1))
                self.events.append(ev)
                new.append(ev)
        return new
