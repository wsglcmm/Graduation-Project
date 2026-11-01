"""目標追蹤：2D 等速（constant velocity）卡爾曼濾波。

狀態 x = [n, e, vn, ve]（world_ned，m 與 m/s）。
    預測：x ← F x，P ← F P Fᵀ + Q（Q 為「白雜訊加速度」模型，σ_a 代表目標會多突然地轉彎或加減速）
    更新：用融合後的量測 z = [n, e]，R = diag(σ², σ²)，σ² = uncertainty_m² / 2
用途：
    1. 平滑：單幀量測會抖，濾波後的位置給 Agent 與隊形規劃比較穩。
    2. 速度：Agent 要判斷目標往哪裡跑、攔截要提前多少，只有濾波器估得出速度。
    3. 遮擋：所有無人機都看不到時，用速度外推（status = predicted），超過 lost_after_s 才判定 lost。
P 的位置部分可以直接換算成規格 §10.4 的 uncertainty_m，所以面板上的誤差圈會在看不到目標時逐漸變大。
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fuse import FusedMeasurement


@dataclass(frozen=True)
class TrackerConfig:
    accel_sigma_mps2: float = 1.5  # ASSUMED：行人/車輛的轉彎與加減速程度，可用重播資料調整
    lost_after_s: float = 3.0  # 連續多久沒有觀測就判定 lost
    init_velocity_sigma_mps: float = 5.0  # 第一筆量測時速度未知


class TargetTracker:
    """單一目標的卡爾曼濾波器。時間單位一律用 int 奈秒（sim time）。"""

    def __init__(self, target_id: int, cls: str = "car", cfg: TrackerConfig = TrackerConfig()):
        self.target_id, self.cls, self.cfg = target_id, cls, cfg
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.t_ns: Optional[int] = None
        self.last_seen_ns: Optional[int] = None
        self.observed_by: tuple = ()

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def _predict_to(self, t_ns: int) -> None:
        dt = (t_ns - self.t_ns) / 1e9
        if dt < 0:
            raise ValueError("時間不能倒退")
        if dt == 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.cfg.accel_sigma_mps2**2
        # 離散白雜訊加速度模型（每軸）：[[dt⁴/4, dt³/2], [dt³/2, dt²]]·σ_a²
        Q1 = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_ns = t_ns

    def step(self, t_ns: int, meas: Optional[FusedMeasurement]) -> Optional[dict]:
        """前進到 t_ns；有量測就更新。回傳規格 §10.4 的 FusedTarget dict（尚未初始化時回傳 None）。"""
        if not self.initialized:
            if meas is None:
                return None
            var = meas.uncertainty_m**2 / 2
            self.x = np.array([meas.n, meas.e, 0.0, 0.0])
            self.P = np.diag([var, var, self.cfg.init_velocity_sigma_mps**2, self.cfg.init_velocity_sigma_mps**2])
            self.t_ns = self.last_seen_ns = t_ns
            self.observed_by = meas.observed_by
            return self.state()

        self._predict_to(t_ns)
        self.observed_by = ()
        if meas is not None:
            Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
            R = np.eye(2) * (meas.uncertainty_m**2 / 2)
            y = np.array([meas.n, meas.e]) - Hm @ self.x
            S = Hm @ self.P @ Hm.T + R
            K = self.P @ Hm.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ Hm) @ self.P
            self.P = (self.P + self.P.T) / 2  # 保持對稱，避免數值誤差累積
            self.last_seen_ns = t_ns
            self.observed_by = meas.observed_by
        return self.state()

    def status(self) -> str:
        if self.last_seen_ns == self.t_ns and self.observed_by:
            return "tracking"
        if (self.t_ns - self.last_seen_ns) / 1e9 <= self.cfg.lost_after_s:
            return "predicted"
        return "lost"

    def state(self) -> dict:
        n, e, vn, ve = (float(v) for v in self.x)
        return {
            "target_id": self.target_id,
            "class": self.cls,
            "status": self.status(),
            "position_world_ned_m": {"n": n, "e": e},
            "velocity_world_ned_mps": {"n": vn, "e": ve},
            "uncertainty_m": float(math.sqrt(self.P[0, 0] + self.P[1, 1])),
            "observed_by": list(self.observed_by),
            "last_seen_ns": int(self.last_seen_ns),
        }
