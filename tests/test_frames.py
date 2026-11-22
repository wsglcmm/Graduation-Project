"""Gazebo ENU/FLU ↔ NED/FRD 轉換測試。預期值來自 A 的實測描述與手推幾何。"""
import math

import numpy as np
import pytest

from coord_transform import CameraMount, body_to_ned, camera_to_body
from coord_transform.frames import (
    ENU_TO_NED,
    FRD_FLU,
    gazebo_pose_to_ned,
    ned_pose_to_gazebo,
    rpy_matrix,
    sdf_camera_pose_to_mount,
    sdf_camera_rotation,
    wrap_pi,
)


# ---------- 世界位姿 ----------
def test_position_enu_to_ned():
    p = gazebo_pose_to_ned(10.0, 20.0, 30.0, 0, 0, 0)  # 東 10、北 20、上 30
    assert (p.n, p.e, p.d) == pytest.approx((20.0, 10.0, -30.0))


@pytest.mark.parametrize(
    "yaw_enu_deg, yaw_ned_deg",
    [(0, 90), (90, 0), (180, -90), (-90, 180)],  # Gazebo yaw 0 = 朝東 = NED 90°（A 實測）
)
def test_yaw_offset_and_direction(yaw_enu_deg, yaw_ned_deg):
    yaw = gazebo_pose_to_ned(0, 0, 0, 0, 0, math.radians(yaw_enu_deg)).yaw
    assert abs(wrap_pi(yaw - math.radians(yaw_ned_deg))) < 1e-9


def test_nose_down_has_negative_ned_pitch():
    """Gazebo FLU 繞左軸正轉 = 機頭朝下；NED/FRD 的機頭朝下是負 pitch。"""
    assert gazebo_pose_to_ned(0, 0, 0, 0, 0.1, 0).pitch == pytest.approx(-0.1)


def test_right_wing_down_keeps_roll_sign():
    """FLU 繞前軸正轉 = 左翼上抬（右翼下沉），在 FRD 也是正 roll。"""
    assert gazebo_pose_to_ned(0, 0, 0, 0.1, 0, 0).roll == pytest.approx(0.1)


@pytest.mark.parametrize("rpy", [(0.1, -0.2, 0.3), (-0.3, 0.25, 2.8), (0.05, 0.4, -1.9)])
def test_closed_form_matches_matrix_derivation(rpy):
    r, p, y = rpy
    pose = gazebo_pose_to_ned(0, 0, 0, r, p, y)
    expected = ENU_TO_NED @ rpy_matrix(r, p, y) @ FRD_FLU
    assert body_to_ned(pose.roll, pose.pitch, pose.yaw) == pytest.approx(expected, abs=1e-12)
    assert (pose.roll, pose.pitch) == pytest.approx((r, -p))
    assert abs(wrap_pi(pose.yaw - (math.pi / 2 - y))) < 1e-12


# ---------- SDF 相機安裝 ----------
def test_a_sdf_camera_pose_is_nadir():
    """A 回報的 <pose>0 0 .10 0 1.5707 0</pose>：光軸朝下、影像上方朝機頭、影像右方朝右翼。"""
    mount = sdf_camera_pose_to_mount(0, 0, 0.10, 0, 1.5707, 0)
    R = camera_to_body(mount)
    assert R @ [0, 0, 1] == pytest.approx([0, 0, 1], abs=1e-3)  # 光軸 → 機體下方
    assert R @ [0, -1, 0] == pytest.approx([1, 0, 0], abs=1e-3)  # 影像上方 → 機頭
    assert R @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-3)  # 影像右方 → 右翼
    assert mount.offset_body_m == pytest.approx((0.0, 0.0, -0.10))  # 上方 10 cm → FRD z = -0.10


def test_forward_camera():
    R = camera_to_body(sdf_camera_pose_to_mount(0, 0, 0, 0, 0, 0))
    assert R @ [0, 0, 1] == pytest.approx([1, 0, 0])
    assert R @ [1, 0, 0] == pytest.approx([0, 1, 0])


@pytest.mark.parametrize("pitch, yaw", [(1.5707, 0.0), (0.6, 0.3), (1.0, -0.8), (0.0, 1.2)])
def test_mount_closed_form_matches_matrix_derivation(pitch, yaw):
    mount = sdf_camera_pose_to_mount(0.2, -0.05, 0.1, 0.0, pitch, yaw)
    assert camera_to_body(mount) == pytest.approx(sdf_camera_rotation(0.0, pitch, yaw), abs=1e-12)
    assert mount.offset_body_m == pytest.approx((0.2, 0.05, -0.1))


def test_sdf_camera_roll_not_supported():
    with pytest.raises(ValueError, match="roll"):
        sdf_camera_pose_to_mount(0, 0, 0, 0.2, 1.57, 0)


def test_wrap_pi():
    assert wrap_pi(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_pi(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert np.isclose(wrap_pi(0.3), 0.3)


def test_mount_dataclass_from_sdf_is_camera_mount():
    assert isinstance(sdf_camera_pose_to_mount(0, 0, 0, 0, 1.0, 0), CameraMount)


@pytest.mark.parametrize("rpy", [(0.1, -0.2, 0.3), (-0.3, 0.25, 2.8), (0.0, 0.0, -3.0)])
def test_ned_to_gazebo_round_trip(rpy):
    x, y, z, r, p, yw = 3.0, -4.0, 25.0, *rpy
    back = ned_pose_to_gazebo(gazebo_pose_to_ned(x, y, z, r, p, yw))
    assert back[:3] == pytest.approx((x, y, z))
    assert back[3:5] == pytest.approx((r, p))
    assert abs(wrap_pi(back[5] - yw)) < 1e-12
