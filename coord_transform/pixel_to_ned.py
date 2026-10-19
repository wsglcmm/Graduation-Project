"""像素 → NED 地面座標轉換（Phase 1 雛形）

流程（對應 docs/interface_spec.md §5）：
    1. 像素 (u, v) → 相機座標下的射線 d_c = K⁻¹·[u, v, 1]ᵀ（有畸變時先去畸變）
    2. 相機 → 機體：d_b = R_bc · d_c
    3. 機體 → NED：d_n = R_nb(roll, pitch, yaw) · d_b
    4. 與地面平面 d = ground_d_m 求交：λ = (ground_d − cam_d) / d_n[d]
    5. P_ned = cam_pos + λ · d_n

座標系（規格 §2.1，一次寫死）：
    像素      原點左上，u 向右、v 向下，單位 px
    相機      OpenCV 慣例：x 右、y 下、z 光軸向前
    機體 FRD  x 前、y 右、z 下
    世界 NED  x North、y East、z Down；原點為起飛點（d 向下為正，空中時 d 為負）

單位：長度 m、角度 rad。
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

_EPS = 1e-9


# ---------- 資料結構 ----------
class Status(str, Enum):
    """對應規格 §5 的 targets[].status。"""

    OK = "ok"
    NO_POSE = "no_pose"
    ABOVE_HORIZON = "above_horizon"  # 射線沒有打到地面（朝天、平行地面，或相機在地面平面之下）
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class CameraIntrinsics:
    """相機內參。fx, fy, cx, cy 單位 px；dist 順序 (k1, k2, p1, p2, k3)。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_rad: float) -> "CameraIntrinsics":
        """由水平視角與解析度算內參（模擬器、無畸變、正方形像素）。fx = (W/2)/tan(HFOV/2)。"""
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return any(abs(c) > 0.0 for c in self.dist)


@dataclass(frozen=True)
class CameraMount:
    """相機相對機體 FRD 的安裝（或雲台）角度。

    pitch_rad：光軸相對機體前方的俯仰角。0 = 平視前方，-π/2 = 正下方（nadir），負值向下。
    yaw_rad：光軸相對機體前方的水平轉角，向右為正。
    offset_body_m：相機位置相對機體原點，機體 FRD 座標，單位 m。
    """

    pitch_rad: float = -math.pi / 2.0
    yaw_rad: float = 0.0
    offset_body_m: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Pose:
    """無人機姿態（規格 §4）。位置為 NED（m），姿態為機體 FRD 相對 NED（rad，yaw 0 = 朝北）。"""

    n: float
    e: float
    d: float
    roll: float
    pitch: float
    yaw: float

    @property
    def position_ned_m(self) -> tuple:
        return (self.n, self.e, self.d)


@dataclass(frozen=True)
class GroundHit:
    """射線與地面平面的求交結果。status 非 OK 時 position_ned_m 為 None。"""

    status: Status
    position_ned_m: Optional[tuple] = None
    range_m: Optional[float] = None  # 相機到交點的距離


# ---------- 旋轉矩陣 ----------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# 平視前方的相機：相機 x（右）= 機體 y，相機 y（下）= 機體 z，相機 z（光軸）= 機體 x。
# 欄 = 相機各軸在機體座標的表示。
_R_CAM_LEVEL_TO_BODY = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R_nb：機體 FRD → NED。航太慣例 Rz(yaw)·Ry(pitch)·Rx(roll)。

    roll 右翼下沉為正；pitch 機頭上仰為正；yaw 由北順時針（俯視）為正。
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def camera_to_body(mount: CameraMount) -> np.ndarray:
    """R_bc：相機（OpenCV）→ 機體 FRD。先俯仰（pitch）再水平轉（yaw）。"""
    return rot_z(mount.yaw_rad) @ rot_y(mount.pitch_rad) @ _R_CAM_LEVEL_TO_BODY


# ---------- 畸變（純 numpy，不依賴 OpenCV 版本）----------
def distort_normalized(x: float, y: float, dist: tuple) -> tuple:
    """正規化座標（無畸變）→ 正規化座標（有畸變）。係數順序 (k1, k2, p1, p2, k3)，與 OpenCV 一致。"""
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort_normalized(xd: float, yd: float, dist: tuple, max_iter: int = 200, tol: float = 1e-14) -> tuple:
    """正規化座標（有畸變）→ 正規化座標（無畸變）。固定點迭代（與 OpenCV 同法）。

    畸變模型沒有封閉解的反函數，所以用迭代：x ← (x_d − 切向項) / 徑向係數。
    """
    k1, k2, p1, p2, k3 = dist
    x, y = xd, yd
    for _ in range(max_iter):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x_new, y_new = (xd - dx) / radial, (yd - dy) / radial
        if abs(x_new - x) < tol and abs(y_new - y) < tol:
            return x_new, y_new
        x, y = x_new, y_new
    return x, y


# ---------- 像素 ↔ 相機射線 ----------
def pixel_to_camera_ray(u: float, v: float, intr: CameraIntrinsics) -> np.ndarray:
    """像素 (u, v) → 相機座標射線 [x, y, 1]（z = 1，非單位向量）。有畸變時先去畸變。"""
    x = (u - intr.cx) / intr.fx
    y = (v - intr.cy) / intr.fy
    if intr.has_distortion:
        x, y = undistort_normalized(x, y, intr.dist)
    return np.array([x, y, 1.0])


def bbox_bottom_center(x1: float, y1: float, x2: float, y2: float) -> tuple:
    """bbox 的接地參考點：底邊中點（規格 §2.4）。y1 目前未使用，保留是為了與 bbox 欄位一致。"""
    return ((x1 + x2) / 2.0, y2)


# ---------- 核心：像素 → NED ----------
def pixel_to_ned(
    u: float,
    v: float,
    intr: CameraIntrinsics,
    mount: CameraMount,
    pose: Optional[Pose],
    *,
    ground_d_m: float = 0.0,
    max_range_m: float = 1000.0,
) -> GroundHit:
    """把像素投影到地面平面 d = ground_d_m，回傳 NED 座標。

    輸入：像素 (u, v)（px）；內參；相機安裝；無人機姿態（NED m、rad）。
    輸出：GroundHit。位置座標系為 NED，單位 m；平面假設下 d = ground_d_m。
    max_range_m 為暫定值（假設），超過視為不可信。
    """
    if pose is None:
        return GroundHit(Status.NO_POSE)

    r_nb = body_to_ned(pose.roll, pose.pitch, pose.yaw)
    d_n = r_nb @ camera_to_body(mount) @ pixel_to_camera_ray(u, v, intr)
    cam_pos = np.array(pose.position_ned_m) + r_nb @ np.array(mount.offset_body_m)

    height = ground_d_m - cam_pos[2]  # 相機在地面平面「上方」多少 m（正 = 在上方）
    if d_n[2] <= _EPS or height <= 0.0:
        return GroundHit(Status.ABOVE_HORIZON)

    lam = height / d_n[2]
    rng = lam * float(np.linalg.norm(d_n))
    if rng > max_range_m:
        return GroundHit(Status.OUT_OF_RANGE)

    p = cam_pos + lam * d_n
    return GroundHit(Status.OK, (float(p[0]), float(p[1]), float(ground_d_m)), rng)


# ---------- 反向：NED → 像素（供往返驗證與合成場景使用）----------
def ned_to_pixel(point_ned: tuple, intr: CameraIntrinsics, mount: CameraMount, pose: Pose) -> Optional[tuple]:
    """NED 點 → 像素 (u, v)。點在相機後方（z ≤ 0）時回傳 None。不檢查是否落在影像範圍內。"""
    r_nb = body_to_ned(pose.roll, pose.pitch, pose.yaw)
    cam_pos = np.array(pose.position_ned_m) + r_nb @ np.array(mount.offset_body_m)
    p_c = camera_to_body(mount).T @ (r_nb.T @ (np.asarray(point_ned, dtype=np.float64) - cam_pos))
    if p_c[2] <= _EPS:
        return None
    x, y = p_c[0] / p_c[2], p_c[1] / p_c[2]
    if intr.has_distortion:
        x, y = distort_normalized(x, y, intr.dist)
    return (float(intr.fx * x + intr.cx), float(intr.fy * y + intr.cy))
