"""一階誤差傳播：輸入誤差（像素、姿態、高度）→ 目標水平位置的 1σ 誤差（規格 §5 uncertainty_m）。

方法（線性化協方差傳播，EKF 也是用同一招）：
    P = f(x)，x = [u, v, roll, pitch, yaw, d]
    J = ∂P_ne / ∂x          （2×6，用中央差分數值計算）
    Σ_P = J · Σ_x · Jᵀ      （Σ_x 為對角矩陣：假設各誤差來源彼此獨立）
    再加上機體水平位置誤差（EKF / GPS）：它 1:1 平移到目標上，所以 Σ_P += σ_pos² · I
    uncertainty_m = sqrt(trace(Σ_P)) = sqrt(σ_n² + σ_e²)   ← 稱為 DRMS

適用範圍與限制：
    - 只涵蓋「隨機、零平均」的誤差。時間不同步、地形高低、bbox 參考點偏差屬於系統偏差，
      不在這裡，另見 analysis/error_study.py 的偏差分析。
    - 線性化在射線接近水平線時失準（誤差非常大、且非高斯），此時 Monte Carlo 才可靠。
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .pixel_to_ned import CameraIntrinsics, CameraMount, Pose, Status, pixel_to_ned

PARAMS = ("u", "v", "roll", "pitch", "yaw", "d")
_STEP = {"u": 0.5, "v": 0.5, "roll": 1e-6, "pitch": 1e-6, "yaw": 1e-6, "d": 1e-4}


@dataclass(frozen=True)
class ErrorModel:
    """各輸入誤差的 1σ。數值來源見 config/error_model.yaml（目前為假設值）。"""

    pixel_sigma_px: float = 2.0
    roll_pitch_sigma_rad: float = math.radians(0.5)
    yaw_sigma_rad: float = math.radians(2.0)
    altitude_sigma_m: float = 0.5
    horizontal_position_sigma_m: float = 0.0  # 無人機自身 n、e 位置誤差（各軸 1σ）；預設 0 = 不計入

    def variances(self) -> np.ndarray:
        """依 PARAMS 順序的變異數。"""
        rp, y = self.roll_pitch_sigma_rad, self.yaw_sigma_rad
        px, alt = self.pixel_sigma_px, self.altitude_sigma_m
        return np.array([px, px, rp, rp, y, alt]) ** 2


def _ground_ne(x, intr, mount, ground_d_m) -> Optional[np.ndarray]:
    u, v, roll, pitch, yaw, d = x
    pose = Pose(n=0.0, e=0.0, d=d, roll=roll, pitch=pitch, yaw=yaw)  # n, e 只是平移，不影響誤差
    hit = pixel_to_ned(u, v, intr, mount, pose, ground_d_m=ground_d_m, max_range_m=math.inf)
    return None if hit.status != Status.OK else np.array(hit.position_ned_m[:2])


def jacobian(
    u: float, v: float, intr: CameraIntrinsics, mount: CameraMount, pose: Pose, *, ground_d_m: float = 0.0
) -> Optional[np.ndarray]:
    """∂(n, e)/∂(u, v, roll, pitch, yaw, d)，2×6。任一擾動後射線打不到地面則回傳 None。"""
    x0 = np.array([u, v, pose.roll, pose.pitch, pose.yaw, pose.d], dtype=np.float64)
    J = np.empty((2, len(PARAMS)))
    for k, name in enumerate(PARAMS):
        h = np.zeros_like(x0)
        h[k] = _STEP[name]
        plus, minus = _ground_ne(x0 + h, intr, mount, ground_d_m), _ground_ne(x0 - h, intr, mount, ground_d_m)
        if plus is None or minus is None:
            return None
        J[:, k] = (plus - minus) / (2 * _STEP[name])
    return J


def horizontal_covariance(u, v, intr, mount, pose, err: ErrorModel, *, ground_d_m: float = 0.0):
    """目標 (n, e) 的 2×2 協方差（m²）。無法計算時回傳 None。"""
    J = jacobian(u, v, intr, mount, pose, ground_d_m=ground_d_m)
    if J is None:
        return None
    return J @ np.diag(err.variances()) @ J.T + np.eye(2) * err.horizontal_position_sigma_m**2


def horizontal_sigma_m(u, v, intr, mount, pose, err: ErrorModel, *, ground_d_m: float = 0.0) -> Optional[float]:
    """規格 §5 的 uncertainty_m：DRMS = sqrt(σ_n² + σ_e²)，單位 m。"""
    cov = horizontal_covariance(u, v, intr, mount, pose, err, ground_d_m=ground_d_m)
    return None if cov is None else float(math.sqrt(np.trace(cov)))


def error_budget(u, v, intr, mount, pose, err: ErrorModel, *, ground_d_m: float = 0.0) -> Optional[Dict[str, float]]:
    """各誤差來源單獨造成的 DRMS（m）。總和的平方 = 各項平方和（因為假設獨立）。"""
    J = jacobian(u, v, intr, mount, pose, ground_d_m=ground_d_m)
    if J is None:
        return None
    var = err.variances()
    per_param = {name: float(np.sum(J[:, k] ** 2) * var[k]) for k, name in enumerate(PARAMS)}
    groups = {"pixel": ("u", "v"), "roll_pitch": ("roll", "pitch"), "yaw": ("yaw",), "altitude": ("d",)}
    out = {g: math.sqrt(sum(per_param[p] for p in names)) for g, names in groups.items()}
    out["position"] = math.sqrt(2.0) * err.horizontal_position_sigma_m  # n、e 兩軸
    return out
