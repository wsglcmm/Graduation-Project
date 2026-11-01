"""視野與多機融合的測試。預期值為手推幾何或統計性質。"""
import math

import numpy as np
import pytest

from coord_transform import CameraIntrinsics, CameraMount
from coord_transform.config import CameraConfig
from coord_transform.uncertainty import ErrorModel
from fusion import DroneState, Observation, ViewModel, fuse

INTR = CameraIntrinsics.from_hfov(1280, 960, 1.74)  # A 的 Gazebo 相機
CAM = CameraConfig(INTR, CameraMount(pitch_rad=-math.pi / 2), 0.0, 1000.0)
ERR = ErrorModel()
VIEW = ViewModel(CAM, ERR, edge_margin=1.0)
S = 1_000_000_000


# ---------- 視野 ----------
def test_footprint_size_matches_fov():
    """30 m 高：左右半寬 = 30·tan(HFOV/2) = 30·640/fx；前後半長 = 30·480/fx。"""
    f, r = VIEW.footprint_half_extents(30.0)
    assert r == pytest.approx(30 * 640 / INTR.fx)
    assert f == pytest.approx(30 * 480 / INTR.fx)


def test_sees_depends_on_heading():
    """機頭朝北：前後範圍較短（影像上下），正東 30 m 在左右範圍內、正北 30 m 不在。"""
    d = DroneState("uav1", 0.0, 0.0, 30.0, yaw_rad=0.0)
    f, r = VIEW.footprint_half_extents(30.0)
    assert f < 30 < r
    assert VIEW.sees(d, 0.0, 30.0) and not VIEW.sees(d, 30.0, 0.0)
    east = DroneState("uav1", 0.0, 0.0, 30.0, yaw_rad=math.pi / 2)  # 轉向朝東後相反
    assert VIEW.sees(east, 30.0, 0.0) and not VIEW.sees(east, 0.0, 30.0)


def test_footprint_corners_rotate_with_yaw():
    corners = VIEW.footprint_corners(DroneState("uav1", 10.0, 5.0, 30.0, math.pi / 2))
    f, r = VIEW.footprint_half_extents(30.0)
    assert corners[0] == pytest.approx((10.0 - r, 5.0 + f))  # 右前：朝東時「右」是南


def test_sigma_grows_off_center_and_none_outside():
    d = DroneState("uav1", 0.0, 0.0, 30.0)
    s0, s1 = VIEW.sigma(d, 0.0, 0.0), VIEW.sigma(d, 0.0, 25.0)
    assert 0 < s0 < s1
    assert VIEW.sigma(d, 0.0, 200.0) is None


def test_observe_only_drones_that_see_target():
    rng = np.random.default_rng(0)
    drones = [DroneState("uav1", 0, 0, 30), DroneState("uav2", 200, 200, 30)]
    obs = VIEW.observe(drones, 1, 5.0, 5.0, rng)
    assert [o.drone_id for o in obs] == ["uav1"]


# ---------- 融合 ----------
def test_two_equal_observations_shrink_by_sqrt2():
    obs = [Observation("uav1", 1, 0.0, 0.0, 1.0), Observation("uav2", 1, 2.0, 0.0, 1.0)]
    (m,) = fuse(obs)
    assert (m.n, m.e) == pytest.approx((1.0, 0.0))
    assert m.uncertainty_m == pytest.approx(1 / math.sqrt(2))
    assert m.observed_by == ("uav1", "uav2")


def test_fusion_leans_toward_precise_observation():
    """σ 比 1:3 → 權重 9:1 → 位置 = 0·0.9 + 10·0.1 = 1.0。"""
    (m,) = fuse([Observation("uav1", 1, 0.0, 0.0, 1.0), Observation("uav2", 1, 10.0, 0.0, 3.0)])
    assert m.n == pytest.approx(1.0)


def test_fusion_groups_by_target_and_skips_invalid():
    obs = [Observation("uav1", 2, 1, 1, 1.0), Observation("uav1", 1, 0, 0, 1.0), Observation("uav2", 1, 0, 0, 0.0)]
    out = fuse(obs)
    assert [m.target_id for m in out] == [1, 2]
    assert out[0].observed_by == ("uav1",)


def test_monte_carlo_fusion_matches_predicted_uncertainty():
    rng = np.random.default_rng(1)
    sig = [0.8, 1.5]
    errs = []
    for _ in range(4000):
        obs = [Observation(f"u{i}", 1, *rng.normal(0, s / math.sqrt(2), 2), s) for i, s in enumerate(sig)]
        (m,) = fuse(obs)
        errs.append(m.n**2 + m.e**2)
    assert math.sqrt(np.mean(errs)) == pytest.approx(m.uncertainty_m, rel=0.05)
