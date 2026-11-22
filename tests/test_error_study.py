"""誤差研究與 bbox 參考點的測試。偏差的預期值皆為手推公式。"""
import math

import pytest

from analysis import error_study as es
from coord_transform import CameraIntrinsics, CameraMount
from coord_transform.config import CameraConfig
from coord_transform.reference_point import CLASS_HEIGHT_M, reference_point

INTR = CameraIntrinsics.from_hfov(1280, 720, math.radians(90))
NADIR = CameraMount(pitch_rad=-math.pi / 2)
CAM = CameraConfig(INTR, NADIR, 0.0, 1000.0)


# ---------- reference_point ----------
def test_reference_point_methods():
    bbox = (100.0, 50.0, 200.0, 150.0)
    assert reference_point(bbox, "car", "bottom_center") == (150.0, 150.0, 0.0)
    assert reference_point(bbox, "car", "center") == (150.0, 100.0, 0.0)
    assert reference_point(bbox, "car", "center_half_height") == (150.0, 100.0, -CLASS_HEIGHT_M["car"] / 2)
    assert reference_point(bbox, "car", "center_half_height", ground_d_m=1.0, height_m=2.0) == (150.0, 100.0, 0.0)


def test_reference_point_unknown_method():
    with pytest.raises(ValueError, match="未知"):
        reference_point((0, 0, 1, 1), "car", "top_left")


def test_every_detection_class_has_a_height_prior():
    from coord_transform.messages import CLASSES

    assert set(CLASSES) == set(CLASS_HEIGHT_M)


# ---------- 偏差的已知答案 ----------
@pytest.mark.parametrize("dt_ms, rate_dps", [(33, 30), (100, 10)])
def test_latency_rotation_error_is_h_omega_dt(dt_ms, rate_dps):
    """nadir 正下方：用錯 dt 秒後的姿態，roll 多了 ω·dt → 地面偏 H·tan(ω·dt)。"""
    h, dt, w = 30.0, dt_ms / 1000, math.radians(rate_dps)
    assert es.latency_error(INTR, NADIR, h, dt, roll_rate_rps=w) == pytest.approx(h * math.tan(w * dt))


def test_latency_translation_error_is_v_dt():
    assert es.latency_error(INTR, NADIR, 30.0, 0.05, speed_mps=5.0) == pytest.approx(0.25)


@pytest.mark.parametrize("theta_deg", [0, 20, 40])
def test_terrain_error_is_rise_times_tan(theta_deg):
    """目標高出平面 Δ：射線多走到 d = 0 才停，水平多出 Δ·tanθ。"""
    theta = math.radians(theta_deg)
    assert es.terrain_error(INTR, NADIR, 30.0, theta, 1.0) == pytest.approx(math.tan(theta), abs=1e-9)


def test_box_corners_footprint():
    pts = es.box_corners((10.0, 20.0), (4.0, 2.0, 1.5), heading_rad=0.0)
    ns, es_, ds = zip(*pts)
    assert (min(ns), max(ns), min(es_), max(es_)) == pytest.approx((8.0, 12.0, 19.0, 21.0))
    assert sorted(set(ds)) == pytest.approx([-1.5, 0.0])


def test_nadir_car_bottom_center_bias_is_half_length_when_heading_north():
    """nadir 正下方、車頭朝北：影像最下緣是「車頂的南端」（車頂離相機近，投影較大），
    其射線打到地面時，離車中心 (L/2)·H/(H − h) = 2.25 × 30 / 28.5 ≈ 2.368 m。"""
    rows = es.study_reference(CAM)
    r = next(x for x in rows if (x["camera"], x["object"], x["image_spot"], x["heading_deg"]) == ("nadir", "car", "center", 0))
    assert r["bottom_center_m"] == pytest.approx(2.25 * 30 / (30 - 1.5))
    assert r["center_half_height_m"] < 0.01  # 正下方時 bbox 中心就是物體中心的投影


def test_center_half_height_beats_bottom_center_for_cars():
    summary = es.summarize_reference(es.study_reference(CAM))
    for row in summary:
        if row["object"] == "car":
            assert row["center_half_height_max_m"] < row["bottom_center_mean_m"]


# ---------- 與正式流程整合：frame_to_target_world 的 reference 選項 ----------
def test_center_half_height_reference_recovers_car_on_ground():
    """用真實車子的 3D 外框產生 bbox；center_half_height 應接近車的接地中心，且輸出 d 仍是地面。"""
    from test_target_world import CAR, CFG, MS, T0, frame, stream, true_pose

    from coord_transform import ErrorModel, frame_to_target_world

    t = T0 + 500 * MS
    x1, y1, x2, y2 = es.project_bbox(es.box_corners(CAR[:2], (4.5, 1.8, 1.5), 0.3), INTR, CFG.mount, true_pose(t))
    det = {"track_id": 7, "class": "car", "confidence": 0.9, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
    good = frame_to_target_world(frame(t, [det]), stream(), CFG, ErrorModel(), reference="center_half_height")
    spec = frame_to_target_world(frame(t, [det]), stream(), CFG, reference="bottom_center")
    p_good, p_spec = good["targets"][0]["position_ned_m"], spec["targets"][0]["position_ned_m"]
    assert p_good["d"] == 0.0
    assert math.dist((p_good["n"], p_good["e"]), CAR[:2]) < 0.15
    assert math.dist((p_spec["n"], p_spec["e"]), CAR[:2]) > 1.0
    assert good["targets"][0]["uncertainty_m"] > 0


# ---------- 整份研究可以跑完 ----------
def test_main_runs_quickly(tmp_path):
    res = es.main(["--samples", "50", "--no-plots", "--out", str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} >= {"sweep.csv", "budget.csv", "latency.csv", "terrain.csv", "reference_summary.csv"}
    assert len(res["sweep"]) == len(es.ALTITUDES_M) * len(es.ANGLES_DEG)
