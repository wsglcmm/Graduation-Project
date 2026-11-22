"""各機視野模擬：每架無人機只「看得到」自己相機地面投影範圍內的目標（規格 §10）。

影片 Demo 中，目標的地面位置來自 B 對鳥瞰影片的偵測 + 地面標定；SITL 無人機沒有真的相機畫面，
所以由這裡模擬「這架飛機的相機看不看得到、看到的話誤差多大」。

幾何（nadir 相機、機體近似水平）：
    相機地面投影是一個矩形，以無人機正下方為中心，
    前後半長 = 高度 × tan(VFOV/2)（影像上下 = 機頭前後），左右半寬 = 高度 × tan(HFOV/2)，
    再依 yaw 旋轉到 world_ned。
觀測誤差：把目標投影回這架的影像像素，用 coord_transform.uncertainty 的同一套誤差模型算出 σ，
所以「離畫面中心越遠、飛得越高 → 誤差越大」會自然出現，和 Phase 2 的誤差研究一致。
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from coord_transform import CameraMount, Pose, horizontal_sigma_m, ned_to_pixel
from coord_transform.config import CameraConfig
from coord_transform.uncertainty import ErrorModel


@dataclass(frozen=True)
class DroneState:
    """一架無人機在 world_ned 的狀態。alt_m 為離地高度（> 0），yaw 0 = 朝北、順時針為正。"""

    drone_id: str
    n: float
    e: float
    alt_m: float
    yaw_rad: float = 0.0

    def pose(self) -> Pose:
        return Pose(n=self.n, e=self.e, d=-self.alt_m, roll=0.0, pitch=0.0, yaw=self.yaw_rad)


@dataclass(frozen=True)
class Observation:
    """一架無人機對一個目標的觀測（world_ned，m）。"""

    drone_id: str
    target_id: int
    n: float
    e: float
    uncertainty_m: float  # DRMS


class ViewModel:
    """由 camera.yaml 的內參決定視野；edge_margin < 1 代表目標太靠邊（bbox 會被切掉）時不算看到。"""

    def __init__(self, cam: CameraConfig, err: ErrorModel, edge_margin: float = 0.9):
        intr = cam.intrinsics
        self.cam, self.err, self.edge_margin = cam, err, edge_margin
        self.tan_half_h = (intr.width / 2.0) / intr.fx
        self.tan_half_v = (intr.height / 2.0) / intr.fy
        # 視野模擬假設相機朝正下方（Demo 的 SITL 無人機使用同一個 nadir 相機）
        self.mount = CameraMount(pitch_rad=-math.pi / 2, yaw_rad=0.0, offset_body_m=(0.0, 0.0, 0.0))

    def footprint_half_extents(self, alt_m: float) -> Tuple[float, float]:
        """（前後半長, 左右半寬），m。"""
        return alt_m * self.tan_half_v, alt_m * self.tan_half_h

    def footprint_corners(self, drone: DroneState) -> List[Tuple[float, float]]:
        """視野矩形四個角（world n, e），順序：右前、左前、左後、右後。畫圖用。"""
        f, r = self.footprint_half_extents(drone.alt_m)
        c, s = math.cos(drone.yaw_rad), math.sin(drone.yaw_rad)
        out = []
        for fwd, right in ((f, r), (f, -r), (-f, -r), (-f, r)):
            out.append((drone.n + fwd * c - right * s, drone.e + fwd * s + right * c))
        return out

    def sees(self, drone: DroneState, target_n: float, target_e: float) -> bool:
        dn, de = target_n - drone.n, target_e - drone.e
        c, s = math.cos(drone.yaw_rad), math.sin(drone.yaw_rad)
        fwd, right = dn * c + de * s, -dn * s + de * c
        f, r = self.footprint_half_extents(drone.alt_m)
        return abs(fwd) <= self.edge_margin * f and abs(right) <= self.edge_margin * r

    def sigma(self, drone: DroneState, target_n: float, target_e: float) -> Optional[float]:
        """這架看這個目標的 1σ 水平誤差（DRMS）；看不到時回傳 None。"""
        if not self.sees(drone, target_n, target_e):
            return None
        uv = ned_to_pixel((target_n, target_e, 0.0), self.cam.intrinsics, self.mount, drone.pose())
        if uv is None:
            return None
        return horizontal_sigma_m(uv[0], uv[1], self.cam.intrinsics, self.mount, drone.pose(), self.err)

    def observe(
        self, drones: Sequence[DroneState], target_id: int, target_n: float, target_e: float, rng: np.random.Generator
    ) -> List[Observation]:
        """模擬每架的觀測：看得到才有，量測 = 真值 + 高斯雜訊（每軸 σ = DRMS/√2）。"""
        obs = []
        for d in drones:
            u = self.sigma(d, target_n, target_e)
            if u is None:
                continue
            dn, de = rng.normal(0.0, u / math.sqrt(2.0), 2)
            obs.append(Observation(d.drone_id, target_id, target_n + dn, target_e + de, u))
        return obs
