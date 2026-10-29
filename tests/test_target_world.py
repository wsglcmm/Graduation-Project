"""端到端測試：模擬 A 的姿態串流 + B 的偵測幀 → 檢查 TargetWorld 與真值一致。

合成方法：先決定目標真實 NED 位置，用「該影像時刻的真實姿態」ned_to_pixel 投影成 bbox 底邊中點，
再交給 frame_to_target_world。姿態串流只在 50 Hz 的取樣點上提供，影像時刻落在兩筆之間，
所以這個測試同時驗證了時間插值。
"""
import json
import math

import pytest
from pydantic import ValidationError

from coord_transform import CameraIntrinsics, CameraMount, ErrorModel, Pose, PoseBuffer, frame_to_target_world, ned_to_pixel
from coord_transform import rotations
from coord_transform.config import CameraConfig

MS = 1_000_000
T0 = 1_730_000_000_000_000_000
INTR = CameraIntrinsics.from_hfov(1280, 720, math.radians(90))
CFG = CameraConfig(intrinsics=INTR, mount=CameraMount(pitch_rad=-math.pi / 2), ground_d_m=0.0, max_range_m=1000.0)
CAR = (8.0, -5.0, 0.0)


def true_pose(t_ns):
    """等速往北 5 m/s、yaw 以 0.3 rad/s 轉、輕微 roll 擺動。"""
    s = (t_ns - T0) / 1e9
    return Pose(n=5.0 * s, e=0.0, d=-30.0, roll=0.02 * math.sin(2 * s), pitch=-0.03, yaw=0.3 * s)


def pose_msg(t_ns, gimbal=None):
    p = true_pose(t_ns)
    m = {
        "timestamp_ns": t_ns,
        "position_ned_m": {"n": p.n, "e": p.e, "d": p.d},
        "attitude_rad": {"roll": p.roll, "pitch": p.pitch, "yaw": p.yaw},
        "quaternion_wxyz": list(rotations.euler_to_quat(p.roll, p.pitch, p.yaw)),
        "altitude_agl_m": -p.d,
    }
    if gimbal is not None:
        m["gimbal_rad"] = {"pitch": gimbal[0], "yaw": gimbal[1]}
    return m


def stream(t_end_ms=1000, **kw):
    buf = PoseBuffer()
    for k in range(0, t_end_ms + 1, 20):  # 50 Hz
        buf.add(pose_msg(T0 + k * MS, **kw))
    return buf


def frame(t_ns, dets, w=1280, h=720):
    return {"schema_version": "0.1.0", "frame_id": 1042, "timestamp_ns": t_ns, "image_width": w, "image_height": h, "detections": dets}


def det_at(point, t_ns, mount=CFG.mount, track_id=7, cls="car"):
    u, v = ned_to_pixel(point, INTR, mount, true_pose(t_ns))
    return {"track_id": track_id, "class": cls, "confidence": 0.9, "bbox": {"x1": u - 20, "y1": v - 30, "x2": u + 20, "y2": v}}


def test_target_recovered_between_pose_samples():
    t = T0 + 510 * MS  # 位於 500 ms 與 520 ms 兩筆姿態之間
    out = frame_to_target_world(frame(t, [det_at(CAR, t)]), stream(), CFG)
    tgt = out["targets"][0]
    assert tgt["status"] == "ok"
    p = tgt["position_ned_m"]
    # 插值誤差只來自 roll 的正弦擺動（20 ms 內非線性很小），應遠小於 1 cm
    assert (p["n"], p["e"], p["d"]) == pytest.approx(CAR, abs=0.01)


def test_using_latest_pose_instead_would_be_wrong():
    """對照組：如果偷懶用「最新一筆」姿態（1000 ms），同一幀會差很多 → 證明時間對齊是必要的。"""
    from coord_transform import pixel_to_ned

    t = T0 + 510 * MS
    d = det_at(CAR, t)
    u, v = (d["bbox"]["x1"] + d["bbox"]["x2"]) / 2, d["bbox"]["y2"]
    wrong = pixel_to_ned(u, v, INTR, CFG.mount, true_pose(T0 + 1000 * MS)).position_ned_m
    assert math.dist(wrong[:2], CAR[:2]) > 2.0


def test_output_matches_spec_shape_and_is_json_serializable():
    t = T0 + 500 * MS
    out = frame_to_target_world(frame(t, [det_at(CAR, t)]), stream(), CFG)
    assert set(out) == {"schema_version", "frame_id", "timestamp_ns", "targets"}
    assert out["frame_id"] == 1042 and out["timestamp_ns"] == t
    assert set(out["targets"][0]) == {"track_id", "class", "status", "position_ned_m", "uncertainty_m"}
    assert out["targets"][0]["class"] == "car"
    assert out["targets"][0]["uncertainty_m"] is None  # 沒給誤差模型
    json.loads(json.dumps(out, allow_nan=False))


def test_uncertainty_filled_when_error_model_given():
    t = T0 + 500 * MS
    out = frame_to_target_world(frame(t, [det_at(CAR, t)]), stream(), CFG, ErrorModel())
    assert 0.0 < out["targets"][0]["uncertainty_m"] < 5.0


def test_no_pose_when_frame_time_not_covered():
    t = T0 + 2000 * MS  # 晚於最新姿態
    d = {"track_id": 1, "class": "person", "confidence": 0.5, "bbox": {"x1": 600, "y1": 300, "x2": 640, "y2": 380}}
    out = frame_to_target_world(frame(t, [d, {**d, "track_id": 2}]), stream(), CFG, ErrorModel())
    assert [x["status"] for x in out["targets"]] == ["no_pose", "no_pose"]
    assert all(x["position_ned_m"] is None and x["uncertainty_m"] is None for x in out["targets"])


def test_empty_detections():
    assert frame_to_target_world(frame(T0, []), stream(), CFG)["targets"] == []


def test_above_horizon_reported_per_target():
    fwd_cfg = CameraConfig(INTR, CameraMount(pitch_rad=0.0), 0.0, 1000.0)  # 平視前方
    sky = {"track_id": 3, "class": "car", "confidence": 0.8, "bbox": {"x1": 600, "y1": 10, "x2": 680, "y2": 40}}
    out = frame_to_target_world(frame(T0 + 500 * MS, [sky]), stream(), fwd_cfg)
    assert out["targets"][0]["status"] == "above_horizon"
    assert out["targets"][0]["position_ned_m"] is None


def test_gimbal_angles_override_fixed_mount():
    gimbal = (-1.047, 0.2)  # 往下 60°、右轉 0.2 rad
    t = T0 + 510 * MS
    mount = CameraMount(pitch_rad=gimbal[0], yaw_rad=gimbal[1])
    out = frame_to_target_world(frame(t, [det_at(CAR, t, mount=mount)]), stream(gimbal=gimbal), CFG)
    p = out["targets"][0]["position_ned_m"]
    assert (p["n"], p["e"]) == pytest.approx(CAR[:2], abs=0.01)


def test_unknown_reference_method_is_an_error():
    t = T0 + 500 * MS
    with pytest.raises(ValueError, match="未知"):
        frame_to_target_world(frame(t, [det_at(CAR, t)]), stream(), CFG, reference="magic")


def test_resolution_mismatch_is_an_error():
    with pytest.raises(ValueError, match="解析度"):
        frame_to_target_world(frame(T0, [], w=640, h=360), stream(), CFG)


@pytest.mark.parametrize(
    "bad_det",
    [
        {"track_id": 1, "class": "tank", "confidence": 0.5, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
        {"track_id": 1, "class": "car", "confidence": 1.5, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
        {"track_id": 1, "class": "car", "confidence": 0.5, "bbox": {"x1": 10, "y1": 0, "x2": 1, "y2": 1}},
        {"track_id": "1", "class": "car", "confidence": 0.5, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
    ],
)
def test_invalid_detection_rejected(bad_det):
    with pytest.raises(ValidationError):
        frame_to_target_world(frame(T0, [bad_det]), stream(), CFG)
