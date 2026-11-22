"""Gazebo（ENU / FLU）↔ 本專案（NED / FRD）的座標轉換。

什麼時候會用到：
    1. 讀 Gazebo 的真值位姿（ground truth），拿來標定誤差模型（Phase 3）。
       Gazebo 世界座標是 ENU（x 東、y 北、z 上），yaw 0 = 朝東、逆時針為正；
       PX4 / MAVSDK 是 NED，yaw 0 = 朝北、順時針為正。兩者差 90° 且轉向相反（A 已實測確認）。
    2. 把 SDF 裡相機的 <pose> 換成 CameraMount。SDF 的 link 座標是 FLU（x 前、y 左、z 上），
       Gazebo 相機的光軸是自己的 +x、影像上方是 +z。

做法：全部用旋轉矩陣推導（不背公式），再由測試確認與手推的封閉解一致。
"""
import math

import numpy as np

from .pixel_to_ned import CameraMount, Pose, rot_x, rot_y, rot_z

# 世界座標：ENU 向量 → NED 向量（n = y_enu、e = x_enu、d = -z_enu）
ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# 機體座標：FRD ↔ FLU（y、z 反號；自己的反矩陣就是自己）
FRD_FLU = np.diag([1.0, -1.0, -1.0])
# 相機座標：OpenCV（x 右、y 下、z 光軸）→ Gazebo 相機（x 光軸、y 左、z 上）
CV_TO_GZCAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw)·Ry(pitch)·Rx(roll)。SDF 的 <pose> 與本專案的 body_to_ned 都是這個順序。"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple:
    """旋轉矩陣 → (roll, pitch, yaw)，與 rpy_matrix 互逆（pitch ≠ ±90° 時）。"""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def wrap_pi(a: float) -> float:
    """角度包到 [-π, π)。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def gazebo_pose_to_ned(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> Pose:
    """Gazebo 世界 ENU 的機體位姿（機體 FLU）→ 本專案的 Pose（NED、機體 FRD）。

    封閉解（測試會核對）：n = y、e = x、d = -z；roll 不變、pitch 反號、yaw_ned = π/2 − yaw_enu。
    注意：Gazebo 原點不一定等於起飛點（規格 §2.1 的 NED 原點），需要時再扣掉起飛點。
    """
    n, e, d = ENU_TO_NED @ np.array([x, y, z])
    R_ned_frd = ENU_TO_NED @ rpy_matrix(roll, pitch, yaw) @ FRD_FLU
    r, p, yw = matrix_to_rpy(R_ned_frd)
    return Pose(n=float(n), e=float(e), d=float(d), roll=r, pitch=p, yaw=wrap_pi(yw))


def ned_pose_to_gazebo(pose: Pose) -> tuple:
    """gazebo_pose_to_ned 的反函數：Pose（NED、FRD）→ (x, y, z, roll, pitch, yaw)（Gazebo ENU、FLU）。
    合成測試資料（analysis/synth_session.py）用它產生「Gazebo 真值」。"""
    x, y, z = ENU_TO_NED.T @ np.array([pose.n, pose.e, pose.d])  # ENU_TO_NED 是自己的反矩陣，寫成轉置較直觀
    R_enu_flu = ENU_TO_NED.T @ rpy_matrix(pose.roll, pose.pitch, pose.yaw) @ FRD_FLU
    r, p, yw = matrix_to_rpy(R_enu_flu)
    return float(x), float(y), float(z), r, p, wrap_pi(yw)


def sdf_camera_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF 相機 <pose> 的旋轉 → R_bc（OpenCV 相機 → 機體 FRD）。"""
    return FRD_FLU @ rpy_matrix(roll, pitch, yaw) @ CV_TO_GZCAM


def sdf_camera_pose_to_mount(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> CameraMount:
    """SDF 相機 <pose x y z roll pitch yaw>（相對機體 FLU）→ CameraMount（機體 FRD）。

    CameraMount 只有 pitch、yaw 兩個自由度（雲台常見），所以 SDF 的 roll 必須為 0，否則影像會旋轉，
    需要擴充 CameraMount。封閉解：pitch_frd = −pitch_sdf、yaw_frd = −yaw_sdf、offset = (x, −y, −z)。
    """
    if abs(wrap_pi(roll)) > 1e-6:
        raise ValueError("SDF 相機 roll 不為 0：CameraMount 目前不支援相機繞光軸旋轉")
    return CameraMount(pitch_rad=-pitch, yaw_rad=-yaw, offset_body_m=(x, -y, -z))


__all__ = [
    "ENU_TO_NED",
    "gazebo_pose_to_ned",
    "matrix_to_rpy",
    "ned_pose_to_gazebo",
    "rpy_matrix",
    "sdf_camera_pose_to_mount",
    "sdf_camera_rotation",
    "wrap_pi",
]
