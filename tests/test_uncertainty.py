"""誤差傳播的已知答案測試。

手推（nadir 相機、水平姿態、yaw = 0、高度 H、目標在正東方離軸角 θ，地面距離 r = H·tanθ）：
    e = H·tan(θ − roll)       → ∂e/∂roll  = −H / cos²θ
    n = H·tan(pitch)（θ 方向無關）→ ∂n/∂pitch = H
    yaw 轉 δ，東方的點往南移 r·δ → ∂n/∂yaw   = −r
    e = (−d)·tanθ             → ∂e/∂d     = −tanθ
    e = H·(u − cx)/fx         → ∂e/∂u     = H / fx（平地 nadir 時與 θ 無關）
    影像向下 = 南              → ∂n/∂v     = −H / fy
"""
import math

import numpy as np
import pytest

from coord_transform import CameraIntrinsics, CameraMount, Pose, pixel_to_ned
from coord_transform.config import load_error_model
from coord_transform.uncertainty import PARAMS, ErrorModel, error_budget, horizontal_covariance, horizontal_sigma_m, jacobian

H = 30.0
INTR = CameraIntrinsics.from_hfov(1280, 720, math.radians(90))  # f = 640
F = INTR.fx
NADIR = CameraMount(pitch_rad=-math.pi / 2)
LEVEL = Pose(n=0.0, e=0.0, d=-H, roll=0.0, pitch=0.0, yaw=0.0)
ERR = ErrorModel(pixel_sigma_px=2.0, roll_pitch_sigma_rad=math.radians(0.5), yaw_sigma_rad=math.radians(2.0), altitude_sigma_m=0.5)


def expected_jacobian(theta):
    r = H * math.tan(theta)
    #                 u        v       roll                    pitch  yaw  d
    return np.array([[0.0, -H / F, 0.0, H, -r, 0.0], [H / F, 0.0, -H / math.cos(theta) ** 2, 0.0, 0.0, -math.tan(theta)]])


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 40.0])
def test_jacobian_matches_hand_derivation(theta_deg):
    theta = math.radians(theta_deg)
    u = INTR.cx + F * math.tan(theta)
    J = jacobian(u, INTR.cy, INTR, NADIR, LEVEL)
    assert J == pytest.approx(expected_jacobian(theta), abs=1e-5)


def test_sigma_at_nadir_center_closed_form():
    """正下方：σ_n² = (H·σ_px/f)² + (H·σ_pitch)²，σ_e 同理；yaw 與高度都不影響。"""
    expected = math.sqrt(2 * (H * ERR.pixel_sigma_px / F) ** 2 + 2 * (H * ERR.roll_pitch_sigma_rad) ** 2)
    assert horizontal_sigma_m(INTR.cx, INTR.cy, INTR, NADIR, LEVEL, ERR) == pytest.approx(expected, rel=1e-6)


def test_error_grows_with_altitude_and_off_nadir_angle():
    u_edge = INTR.cx + F * math.tan(math.radians(40))
    s = lambda h, u: horizontal_sigma_m(u, INTR.cy, INTR, NADIR, Pose(0, 0, -h, 0, 0, 0), ERR)  # noqa: E731
    assert s(10, INTR.cx) < s(30, INTR.cx) < s(80, INTR.cx)
    assert s(30, INTR.cx) < s(30, u_edge)


def test_drone_position_error_adds_in_quadrature():
    """機體水平位置誤差 σ_pos 直接平移目標：DRMS² 多出 2·σ_pos²（n、e 各一份）。"""
    base = horizontal_sigma_m(800, 500, INTR, NADIR, LEVEL, ERR)
    with_pos = horizontal_sigma_m(800, 500, INTR, NADIR, LEVEL, ErrorModel(**{**ERR.__dict__, "horizontal_position_sigma_m": 0.4}))
    assert with_pos**2 == pytest.approx(base**2 + 2 * 0.4**2)


def test_zero_noise_gives_zero_uncertainty():
    assert horizontal_sigma_m(800, 500, INTR, NADIR, LEVEL, ErrorModel(0, 0, 0, 0)) == pytest.approx(0.0, abs=1e-12)


def test_budget_squares_sum_to_total():
    u, v = 1000.0, 600.0
    budget = error_budget(u, v, INTR, NADIR, LEVEL, ERR)
    total = horizontal_sigma_m(u, v, INTR, NADIR, LEVEL, ERR)
    assert set(budget) == {"pixel", "roll_pitch", "yaw", "altitude", "position"}
    assert math.sqrt(sum(x * x for x in budget.values())) == pytest.approx(total, rel=1e-9)


def test_ray_near_horizon_returns_none():
    forward = CameraMount(pitch_rad=0.0)  # 平視前方：影像中心射線與地面平行
    assert jacobian(INTR.cx, INTR.cy, INTR, forward, LEVEL) is None
    assert horizontal_sigma_m(INTR.cx, INTR.cy, INTR, forward, LEVEL, ERR) is None


def test_monte_carlo_agrees_with_linear_propagation():
    """抽樣加雜訊、實際跑 pixel_to_ned，量到的協方差應與一階傳播一致（雜訊小時線性化成立）。"""
    rng = np.random.default_rng(0)
    u0, v0 = INTR.cx + 300.0, INTR.cy + 150.0
    pose0 = Pose(n=0.0, e=0.0, d=-H, roll=0.05, pitch=-0.03, yaw=0.8)
    sig = np.sqrt(ERR.variances())
    pts = []
    for _ in range(20000):
        du, dv, dr, dp, dy, dd = rng.normal(0.0, sig)
        pose = Pose(0.0, 0.0, pose0.d + dd, pose0.roll + dr, pose0.pitch + dp, pose0.yaw + dy)
        pts.append(pixel_to_ned(u0 + du, v0 + dv, INTR, NADIR, pose).position_ned_m[:2])
    emp = np.cov(np.array(pts).T)
    lin = horizontal_covariance(u0, v0, INTR, NADIR, pose0, ERR)
    assert math.sqrt(np.trace(emp)) == pytest.approx(math.sqrt(np.trace(lin)), rel=0.03)
    assert emp == pytest.approx(lin, rel=0.1, abs=0.01)


def test_params_order_is_fixed():
    assert PARAMS == ("u", "v", "roll", "pitch", "yaw", "d")


def test_load_error_model(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("pixel_sigma_px: 1.5\nroll_pitch_sigma_deg: 1.0\nyaw_sigma_deg: 3.0\naltitude_sigma_m: 0.2\n", encoding="utf-8")
    m = load_error_model(p)
    assert (m.pixel_sigma_px, m.altitude_sigma_m) == (1.5, 0.2)
    assert m.roll_pitch_sigma_rad == pytest.approx(math.radians(1.0))
    assert m.yaw_sigma_rad == pytest.approx(math.radians(3.0))


def test_load_error_model_rejects_missing_and_negative(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("pixel_sigma_px: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少"):
        load_error_model(p)
    p.write_text("pixel_sigma_px: -1\nroll_pitch_sigma_deg: 1\nyaw_sigma_deg: 1\naltitude_sigma_m: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="≥ 0"):
        load_error_model(p)


def test_repo_error_model_file_loads():
    from pathlib import Path

    m = load_error_model(Path(__file__).resolve().parents[1] / "config" / "error_model.yaml")
    assert m.pixel_sigma_px > 0
