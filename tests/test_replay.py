"""session 讀取、重播評估、時間偏移估計、誤差模型標定的測試。

全部用 analysis/synth_session.py 產生「已知答案」的資料：注入多少誤差，工具就應該量回多少。
"""
import dataclasses
import json
import math
from pathlib import Path

import pytest

from analysis import calibrate_error_model as cal
from analysis import replay
from analysis.session import load_session
from analysis.synth_session import SynthConfig, T_START_NS, car_state, drone_truth, generate
from coord_transform.config import PipelineConfig, load_camera_config

ROOT = Path(__file__).resolve().parents[1]
CAM = load_camera_config(ROOT / "config" / "camera.yaml")
NOISELESS = SynthConfig(
    duration_s=6.0,
    pixel_edge_sigma_px=0.0,
    roll_pitch_sigma_deg=0.0,
    yaw_sigma_deg=0.0,
    altitude_sigma_m=0.0,
    horizontal_position_sigma_m=0.0,
)
LAG_NS = int(SynthConfig.camera_time_lag_ms * 1e6)


@pytest.fixture(scope="module")
def clean_session(tmp_path_factory):
    return load_session(generate(tmp_path_factory.mktemp("clean"), CAM, NOISELESS))


@pytest.fixture(scope="module")
def noisy_session(tmp_path_factory):
    return load_session(generate(tmp_path_factory.mktemp("noisy"), CAM, SynthConfig()))


# ---------- session 讀取 ----------
def test_truth_round_trips_through_gazebo_enu(clean_session):
    """synth 把真值寫成 Gazebo ENU（含原點偏移），讀回後應等於原本的 NED。"""
    t = 3.21
    t_ns = T_START_NS + int(t * 1e9)
    expected = drone_truth(t, NOISELESS)
    got = clean_session.drone_pose_ned(t_ns)
    assert (got.n, got.e, got.d) == pytest.approx((expected.n, expected.e, expected.d), abs=1e-3)  # 50 Hz 真值線性插值
    assert (got.roll, got.pitch) == pytest.approx((expected.roll, expected.pitch), abs=1e-4)  # 50 Hz 真值插值
    assert math.cos(got.yaw - expected.yaw) == pytest.approx(1.0, abs=1e-8)
    n, e, _ = car_state(t)
    assert clean_session.target_ned("car_1", t_ns)[:2] == pytest.approx((n, e), abs=0.01)


def test_ground_plane_from_meta(clean_session):
    assert clean_session.ground_plane_d_m == 0.0


@pytest.mark.parametrize("broken", ["missing_file", "bad_class", "no_drone_rpy"])
def test_load_session_rejects_broken_sessions(tmp_path, broken):
    d = generate(tmp_path / "s", CAM, dataclasses.replace(NOISELESS, duration_s=1.0))
    if broken == "missing_file":
        (d / "truth.jsonl").unlink()
        err = FileNotFoundError
    elif broken == "bad_class":
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta["targets"]["car_1"] = "Car"
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        err = ValueError
    else:
        rows = [json.loads(line) for line in (d / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r.pop("rpy_rad", None)
        (d / "truth.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        err = ValueError
    with pytest.raises(err):
        load_session(d)


# ---------- 重播 ----------
def test_noiseless_replay_with_correct_settings_is_near_exact(clean_session):
    cfg = PipelineConfig(reference_point="center_half_height", frame_time_offset_ns=-LAG_NS)
    _, matches, s = replay.evaluate(clean_session, CAM, cfg)
    assert s["matched"] > 150 and s["unmatched_ok"] == 0
    assert s["error_rmse_m"] < 0.15  # 只剩 center_half_height 的幾何近似誤差


def test_bottom_center_shows_large_bias_on_cars(clean_session):
    cfg = PipelineConfig(reference_point="bottom_center", frame_time_offset_ns=-LAG_NS)
    s = replay.evaluate(clean_session, CAM, cfg)[2]
    assert s["by_class"]["car"]["rmse_m"] > 1.5


def test_ignoring_camera_lag_costs_accuracy(clean_session):
    good = replay.evaluate(clean_session, CAM, PipelineConfig("center_half_height", -LAG_NS))[2]["error_rmse_m"]
    bad = replay.evaluate(clean_session, CAM, PipelineConfig("center_half_height", 0))[2]["error_rmse_m"]
    assert bad > good + 0.05


def test_time_offset_is_recovered_without_noise(clean_session):
    best, reliable, curve = replay.estimate_time_offset(clean_session, CAM, PipelineConfig("center_half_height"))
    assert reliable
    assert best == pytest.approx(-SynthConfig.camera_time_lag_ms, abs=2.0)
    assert len(curve) > 20


def test_uncertainty_coverage_and_status_counts(noisy_session):
    from coord_transform.uncertainty import ErrorModel

    err = ErrorModel(
        pixel_sigma_px=1.1,
        roll_pitch_sigma_rad=math.radians(0.5),
        yaw_sigma_rad=math.radians(2.0),
        altitude_sigma_m=0.3,
        horizontal_position_sigma_m=0.4,
    )
    _, matches, s = replay.evaluate(noisy_session, CAM, PipelineConfig("center_half_height", -LAG_NS), err)
    assert set(s["targets_by_status"]) <= {"ok", "no_pose"}
    assert 0.55 < s["within_1x_uncertainty"] < 0.9  # 已知雜訊下，uncertainty_m 應該大致準確
    assert s["within_2x_uncertainty"] > 0.93


def test_replay_cli_writes_outputs(tmp_path, clean_session):
    out = tmp_path / "out"
    res = replay.main([str(clean_session.path), "--reference", "center_half_height", "--time-offset-ms", "-24", "--no-plots", "--out", str(out)])
    assert {"targetworld.jsonl", "matches.jsonl", "metrics.json"} <= {p.name for p in out.iterdir()}
    assert res["error_rmse_m"] < 0.15


# ---------- 標定 ----------
def test_calibration_recovers_injected_noise(noisy_session):
    inj = SynthConfig()
    pose_err = cal.pose_errors(noisy_session)
    pix_err = cal.pixel_errors(noisy_session, CAM, "center_half_height", -LAG_NS)
    s, warnings = cal.suggest(pose_err, pix_err)
    assert s["roll_pitch_sigma_deg"] == pytest.approx(inj.roll_pitch_sigma_deg, rel=0.15)
    assert s["yaw_sigma_deg"] == pytest.approx(inj.yaw_sigma_deg, rel=0.15)
    assert s["altitude_sigma_m"] == pytest.approx(inj.altitude_sigma_m, rel=0.15)
    assert s["horizontal_position_sigma_m"] == pytest.approx(inj.horizontal_position_sigma_m, rel=0.15)
    # bbox 每個邊緣 σ = 1.5 px → 中心點 u、v 各為 σ/√2 ≈ 1.06 px
    assert s["pixel_sigma_px"] == pytest.approx(inj.pixel_edge_sigma_px / math.sqrt(2), rel=0.25)
    assert warnings == []


def test_calibration_warns_on_enu_yaw_mistake(tmp_path):
    """常見錯誤：A 把 Gazebo 的 ENU yaw 當成 NED → 偏 90°，工具要大聲警告。"""
    d = generate(tmp_path / "s", CAM, dataclasses.replace(NOISELESS, duration_s=2.0))
    rows = [json.loads(line) for line in (d / "poses.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["attitude_rad"]["yaw"] = (math.pi / 2 - r["attitude_rad"]["yaw"] + math.pi) % (2 * math.pi) - math.pi
        r.pop("quaternion_wxyz")
    (d / "poses.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    _, warnings = cal.suggest(cal.pose_errors(load_session(d)), {})
    assert any("yaw" in w and "慣例" in w for w in warnings)


def test_calibration_warns_on_wrong_origin(tmp_path):
    d = generate(tmp_path / "s", CAM, dataclasses.replace(NOISELESS, duration_s=2.0))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["ned_origin_enu_m"]["x"] += 3.0  # 原點東邊多算 3 m
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _, warnings = cal.suggest(cal.pose_errors(load_session(d)), {})
    assert any("位置 e" in w and "ned_origin_enu_m" in w for w in warnings)


def test_calibrate_cli_writes_yaml_that_loads(tmp_path, noisy_session):
    from coord_transform.config import load_error_model

    out = tmp_path / "cal"
    cal.main([str(noisy_session.path), "--reference", "center_half_height", "--time-offset-ms", "-24", "--out", str(out)])
    m = load_error_model(out / "error_model_suggested.yaml")
    assert m.horizontal_position_sigma_m == pytest.approx(0.4, rel=0.15)


def test_matching_requires_same_class(clean_session):
    """類別不同就不配對：一個標成 person 的輸出即使就在車上，也不能算成車的誤差。"""
    t_ns = T_START_NS + 2_000_000_000
    n, e, _ = car_state(2.0)
    fake = {
        "frame_id": 0,
        "timestamp_ns": t_ns,
        "targets": [{"track_id": 1, "class": "person", "status": "ok", "position_ned_m": {"n": n, "e": e, "d": 0.0}, "uncertainty_m": None}],
    }
    assert replay.match_targets(clean_session, [fake], 0) == []
    fake["targets"][0]["class"] = "car"
    assert [m["truth_name"] for m in replay.match_targets(clean_session, [fake], 0)] == ["car_1"]
