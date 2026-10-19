"""座標轉換的已知答案測試。

預期值皆由幾何直覺手算，不是從程式輸出反推：
    nadir 相機在高度 h：像素偏移 dx → 地面位移 h·dx/fx。
    機頭上仰 θ：機腹朝前，nadir 中心點落在正前方 h·tanθ。
    右翼下沉 φ：機腹朝左，nadir 中心點落在正左方 h·tanφ（即 East = -h·tanφ，yaw=0 時）。
"""
import math

import numpy as np
import pytest

from coord_transform import (
    CameraIntrinsics,
    CameraMount,
    Pose,
    Status,
    bbox_bottom_center,
    distort_normalized,
    ned_to_pixel,
    pixel_to_ned,
    undistort_normalized,
)

H = 30.0  # 高度 m
INTR_W, INTR_H = 1280, 720
INTR = CameraIntrinsics.from_hfov(INTR_W, INTR_H, math.radians(90))  # fx = fy = 640, cx = 640, cy = 360
NADIR = CameraMount(pitch_rad=-math.pi / 2)


def level_pose(h=H, yaw=0.0, n=0.0, e=0.0, roll=0.0, pitch=0.0):
    return Pose(n=n, e=e, d=-h, roll=roll, pitch=pitch, yaw=yaw)


# ---------- 內參 ----------
def test_from_hfov_90deg():
    assert INTR.fx == pytest.approx(640.0)
    assert INTR.fy == pytest.approx(640.0)
    assert (INTR.cx, INTR.cy) == (640.0, 360.0)


def test_hfov_matches_image_edge():
    """右邊緣像素的射線與光軸夾角應為 HFOV/2 = 45°。"""
    hit_ray_x = (1280 - INTR.cx) / INTR.fx
    assert math.degrees(math.atan(hit_ray_x)) == pytest.approx(45.0)


def test_bbox_bottom_center():
    assert bbox_bottom_center(100, 50, 200, 120) == (150.0, 120)


# ---------- nadir、水平姿態 ----------
def test_center_pixel_hits_point_below_drone():
    hit = pixel_to_ned(INTR.cx, INTR.cy, INTR, NADIR, level_pose(n=10.0, e=-5.0))
    assert hit.status == Status.OK
    assert hit.position_ned_m == pytest.approx((10.0, -5.0, 0.0), abs=1e-9)
    assert hit.range_m == pytest.approx(H)


def test_pixel_right_goes_east_and_up_goes_north_when_facing_north():
    hit_r = pixel_to_ned(INTR.cx + 64, INTR.cy, INTR, NADIR, level_pose())
    assert hit_r.position_ned_m[:2] == pytest.approx((0.0, H * 64 / 640), abs=1e-9)  # E = 3.0
    hit_u = pixel_to_ned(INTR.cx, INTR.cy - 36, INTR, NADIR, level_pose())
    assert hit_u.position_ned_m[:2] == pytest.approx((H * 36 / 640, 0.0), abs=1e-9)  # N = 1.6875


def test_yaw_east_image_up_is_east_and_image_right_is_south():
    pose = level_pose(yaw=math.pi / 2)  # 機頭朝東
    hit_u = pixel_to_ned(INTR.cx, INTR.cy - 64, INTR, NADIR, pose)
    assert hit_u.position_ned_m[:2] == pytest.approx((0.0, 3.0), abs=1e-9)
    hit_r = pixel_to_ned(INTR.cx + 64, INTR.cy, INTR, NADIR, pose)
    assert hit_r.position_ned_m[:2] == pytest.approx((-3.0, 0.0), abs=1e-9)  # 朝東時右手邊是南


# ---------- 姿態與安裝角 ----------
def test_pitch_up_shifts_nadir_center_forward():
    theta = 0.1
    hit = pixel_to_ned(INTR.cx, INTR.cy, INTR, NADIR, level_pose(pitch=theta))
    assert hit.position_ned_m[:2] == pytest.approx((H * math.tan(theta), 0.0), abs=1e-9)


def test_roll_right_wing_down_shifts_nadir_center_left():
    phi = 0.1
    hit = pixel_to_ned(INTR.cx, INTR.cy, INTR, NADIR, level_pose(roll=phi))
    assert hit.position_ned_m[:2] == pytest.approx((0.0, -H * math.tan(phi)), abs=1e-9)


def test_mount_pitched_45deg_center_hits_h_ahead():
    mount = CameraMount(pitch_rad=-math.pi / 4)
    hit = pixel_to_ned(INTR.cx, INTR.cy, INTR, mount, level_pose())
    assert hit.position_ned_m[:2] == pytest.approx((H, 0.0), abs=1e-9)
    assert hit.range_m == pytest.approx(H * math.sqrt(2))


def test_forward_camera_pixel_45deg_below_center_hits_h_ahead():
    mount = CameraMount(pitch_rad=0.0)
    hit = pixel_to_ned(INTR.cx, INTR.cy + INTR.fy, INTR, mount, level_pose())  # 向下 45°
    assert hit.position_ned_m[:2] == pytest.approx((H, 0.0), abs=1e-9)


def test_camera_offset_in_body_frame_follows_yaw():
    mount = CameraMount(pitch_rad=-math.pi / 2, offset_body_m=(1.0, 0.0, 0.0))  # 相機在機體前方 1 m
    north = pixel_to_ned(INTR.cx, INTR.cy, INTR, mount, level_pose(yaw=0.0))
    assert north.position_ned_m[:2] == pytest.approx((1.0, 0.0), abs=1e-9)
    east = pixel_to_ned(INTR.cx, INTR.cy, INTR, mount, level_pose(yaw=math.pi / 2))
    assert east.position_ned_m[:2] == pytest.approx((0.0, 1.0), abs=1e-9)


# ---------- 狀態 ----------
def test_no_pose():
    assert pixel_to_ned(0, 0, INTR, NADIR, None).status == Status.NO_POSE


def test_horizontal_camera_at_or_above_horizon_has_no_hit():
    mount = CameraMount(pitch_rad=0.0)
    assert pixel_to_ned(INTR.cx, INTR.cy, INTR, mount, level_pose()).status == Status.ABOVE_HORIZON  # 恰在地平線
    assert pixel_to_ned(INTR.cx, INTR.cy - 100, INTR, mount, level_pose()).status == Status.ABOVE_HORIZON


def test_camera_below_ground_plane_has_no_hit():
    pose = Pose(0, 0, 5.0, 0, 0, 0)  # d = +5：在地面平面之下
    assert pixel_to_ned(INTR.cx, INTR.cy, INTR, NADIR, pose).status == Status.ABOVE_HORIZON


def test_out_of_range_has_no_position():
    hit = pixel_to_ned(INTR.cx, INTR.cy, INTR, NADIR, level_pose(), max_range_m=10.0)
    assert hit.status == Status.OUT_OF_RANGE
    assert hit.position_ned_m is None


def test_ground_plane_offset():
    hit = pixel_to_ned(INTR.cx, INTR.cy, INTR, NADIR, level_pose(), ground_d_m=-10.0)  # 地面高 10 m
    assert hit.status == Status.OK
    assert hit.range_m == pytest.approx(20.0)
    assert hit.position_ned_m[2] == -10.0


# ---------- 往返驗證 ----------
def _random_scene(rng):
    pose = Pose(
        n=rng.uniform(-100, 100),
        e=rng.uniform(-100, 100),
        d=-rng.uniform(5, 100),
        roll=rng.uniform(-0.35, 0.35),
        pitch=rng.uniform(-0.35, 0.35),
        yaw=rng.uniform(-math.pi, math.pi),
    )
    mount = CameraMount(
        pitch_rad=rng.uniform(-math.pi / 2, -math.pi / 4),
        yaw_rad=rng.uniform(-0.3, 0.3),
        offset_body_m=tuple(rng.uniform(-0.3, 0.3, 3)),
    )
    return pose, mount


def test_roundtrip_pixel_to_ground_to_pixel():
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(400):
        pose, mount = _random_scene(rng)
        u, v = rng.uniform(0, INTR_W), rng.uniform(0, INTR_H)
        hit = pixel_to_ned(u, v, INTR, mount, pose)
        if hit.status != Status.OK:
            continue
        ok += 1
        assert hit.position_ned_m[2] == 0.0
        u2, v2 = ned_to_pixel(hit.position_ned_m, INTR, mount, pose)
        assert (u2, v2) == pytest.approx((u, v), abs=1e-6)
    assert ok > 200  # 確保測到足夠多的有效樣本


def test_roundtrip_ground_to_pixel_to_ground():
    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(600):
        pose, mount = _random_scene(rng)
        point = (rng.uniform(-80, 80), rng.uniform(-80, 80), 0.0)
        uv = ned_to_pixel(point, INTR, mount, pose)
        if uv is None or not (0 <= uv[0] <= INTR_W and 0 <= uv[1] <= INTR_H):
            continue
        ok += 1
        hit = pixel_to_ned(uv[0], uv[1], INTR, mount, pose, max_range_m=1e6)
        assert hit.status == Status.OK
        assert hit.position_ned_m == pytest.approx(point, abs=1e-6)
    assert ok > 50


# ---------- 有畸變 ----------
# 內參與畸變取自 calibration/ 對 OpenCV 範例照片的實跑結果（640x480）。
REAL = CameraIntrinsics(
    fx=536.07, fy=536.02, cx=342.37, cy=235.54, width=640, height=480,
    dist=(-0.26509, -0.04674, 0.00183, -0.00031, 0.25232),
)


def test_roundtrip_with_distortion():
    rng = np.random.default_rng(2)
    ok = 0
    for _ in range(200):
        pose, mount = _random_scene(rng)
        u, v = rng.uniform(0, 640), rng.uniform(0, 480)
        hit = pixel_to_ned(u, v, REAL, mount, pose)
        if hit.status != Status.OK:
            continue
        ok += 1
        u2, v2 = ned_to_pixel(hit.position_ned_m, REAL, mount, pose)
        assert (u2, v2) == pytest.approx((u, v), abs=1e-4)
    assert ok > 80


def test_distortion_formula_matches_opencv_project_points():
    """用 OpenCV 當獨立參考，驗證自己寫的畸變公式（係數順序與符號）。"""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-0.5, 0.5, 50), rng.uniform(-0.4, 0.4, 50), np.ones(50)])
    ref, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), REAL.K, np.array(REAL.dist))
    for p, r in zip(pts, ref.reshape(-1, 2)):
        xd, yd = distort_normalized(p[0], p[1], REAL.dist)
        assert (REAL.fx * xd + REAL.cx, REAL.fy * yd + REAL.cy) == pytest.approx(tuple(r), abs=1e-6)


def test_undistort_inverts_distort():
    rng = np.random.default_rng(4)
    for _ in range(100):
        x, y = rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)
        xd, yd = distort_normalized(x, y, REAL.dist)
        assert undistort_normalized(xd, yd, REAL.dist) == pytest.approx((x, y), abs=1e-10)


def test_distortion_changes_the_answer_near_image_corner():
    """去畸變確實有作用：同一像素在有、無畸變下的落點應不同；中心點則幾乎相同。"""
    plain = CameraIntrinsics(REAL.fx, REAL.fy, REAL.cx, REAL.cy, 640, 480)
    pose = level_pose()
    corner_d = pixel_to_ned(10, 10, REAL, NADIR, pose).position_ned_m
    corner_p = pixel_to_ned(10, 10, plain, NADIR, pose).position_ned_m
    assert np.hypot(corner_d[0] - corner_p[0], corner_d[1] - corner_p[1]) > 0.5
    c_d = pixel_to_ned(REAL.cx, REAL.cy, REAL, NADIR, pose).position_ned_m
    c_p = pixel_to_ned(REAL.cx, REAL.cy, plain, NADIR, pose).position_ned_m
    assert c_d == pytest.approx(c_p, abs=1e-6)
