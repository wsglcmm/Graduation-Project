"""四元數工具（wxyz，Hamilton 慣例）。四元數 q 代表 R_nb：機體 FRD → NED。

為什麼不直接對 roll/pitch/yaw 做線性插值？
    1. yaw 跨越 ±π 時會繞遠路（179° → -179° 的中點會變成 0°，而不是 180°）。
    2. 三個歐拉角各自插值，得到的旋轉不在兩個姿態之間的「最短路徑」上。
slerp 在單位四元數球面上走大圓，兩個問題都不存在。規格 §2.3 也規定姿態用 slerp。
"""
import math

import numpy as np


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 歐拉角 → 四元數 wxyz。對應 R_nb = Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ]
    )


def quat_to_euler(q) -> tuple:
    """四元數 wxyz → (roll, pitch, yaw)，yaw 範圍 [-π, π]。pitch = ±90° 時 roll/yaw 不唯一（萬向鎖）。"""
    w, x, y, z = normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("四元數長度為 0")
    return q / n


def angle_between(q1, q2) -> float:
    """兩個姿態之間的旋轉角（rad，0 到 π）。q 與 -q 是同一個姿態，所以取 |dot|。"""
    d = abs(float(np.dot(normalize(q1), normalize(q2))))
    return 2.0 * math.acos(min(1.0, d))


def slerp(q0, q1, t: float) -> np.ndarray:
    """球面線性插值，t ∈ [0, 1]。走最短路徑（dot < 0 時把 q1 反號）。"""
    q0, q1 = normalize(q0), normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:  # 兩姿態幾乎相同：sin(θ) ≈ 0，改用線性插值再正規化，避免除以 0
        return normalize(q0 + t * (q1 - q0))
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) / s) * q0 + (math.sin(t * theta) / s) * q1
