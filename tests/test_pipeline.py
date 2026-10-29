"""TargetWorldPipeline：模擬實際到達順序（影像 30 Hz、姿態 20 Hz、偵測有推論延遲）。"""
import math

import pytest

from coord_transform import CameraIntrinsics, CameraMount, TargetWorldPipeline, ned_to_pixel, rotations
from coord_transform.config import CameraConfig

from test_target_world import T0, MS, true_pose

INTR = CameraIntrinsics.from_hfov(1280, 960, 1.74)  # A 回覆的 Gazebo 相機
CFG = CameraConfig(INTR, CameraMount(pitch_rad=-math.pi / 2), 0.0, 1000.0)
CAR = (8.0, -5.0, 0.0)


def pose_msg(t_ns):
    p = true_pose(t_ns)
    return {
        "timestamp_ns": t_ns,
        "position_ned_m": {"n": p.n, "e": p.e, "d": p.d},
        "attitude_rad": {"roll": p.roll, "pitch": p.pitch, "yaw": p.yaw},
        "quaternion_wxyz": list(rotations.euler_to_quat(p.roll, p.pitch, p.yaw)),
    }


def frame_msg(t_ns, frame_id):
    u, v = ned_to_pixel(CAR, INTR, CFG.mount, true_pose(t_ns))
    det = {"track_id": 7, "class": "car", "confidence": 0.9, "bbox": {"x1": u - 20, "y1": v - 30, "x2": u + 20, "y2": v}}
    return {"schema_version": "0.1.0", "frame_id": frame_id, "timestamp_ns": t_ns, "image_width": 1280, "image_height": 960, "detections": [det]}


def test_frame_waits_for_next_pose_then_is_interpolated():
    pipe = TargetWorldPipeline(CFG)
    assert pipe.on_pose(pose_msg(T0)) == []
    assert pipe.on_frame(frame_msg(T0 + 30 * MS, 1)) == []  # 影像比最新姿態新：先等
    assert pipe.pending == 1
    out = pipe.on_pose(pose_msg(T0 + 50 * MS))  # 下一筆姿態到了 → 可以內插
    assert len(out) == 1 and pipe.pending == 0
    p = out[0]["targets"][0]["position_ned_m"]
    assert (p["n"], p["e"]) == pytest.approx(CAR[:2], abs=0.02)


def test_frame_older_than_latest_pose_is_processed_immediately():
    pipe = TargetWorldPipeline(CFG)
    pipe.on_pose(pose_msg(T0))
    pipe.on_pose(pose_msg(T0 + 50 * MS))
    out = pipe.on_frame(frame_msg(T0 + 20 * MS, 1))  # 推論延遲大，偵測到時姿態已經涵蓋
    assert out[0]["targets"][0]["status"] == "ok"


def test_realistic_stream_all_frames_ok_and_in_order():
    """1 秒：影像 30 Hz、姿態 20 Hz；每個偵測晚 40 ms 才送到（推論延遲）。"""
    events = [(k * 50 * MS, "pose", T0 + k * 50 * MS) for k in range(21)]
    for i in range(30):
        t_img = T0 + round(i * 1000 / 30) * MS
        events.append((t_img - T0 + 40 * MS, "frame", (t_img, i)))
    events.sort(key=lambda e: (e[0], e[1] == "frame"))

    pipe, outputs = TargetWorldPipeline(CFG), []
    for _, kind, payload in events:
        outputs += pipe.on_pose(pose_msg(payload)) if kind == "pose" else pipe.on_frame(frame_msg(*payload))
    outputs += pipe.flush()

    assert [o["frame_id"] for o in outputs] == list(range(30))
    assert all(o["targets"][0]["status"] == "ok" for o in outputs)
    worst = max(math.dist((o["targets"][0]["position_ned_m"]["n"], o["targets"][0]["position_ned_m"]["e"]), CAR[:2]) for o in outputs)
    assert worst < 0.05


def test_pose_stream_stall_releases_oldest_as_no_pose():
    pipe = TargetWorldPipeline(CFG, max_pending=2)
    pipe.on_pose(pose_msg(T0))
    assert pipe.on_frame(frame_msg(T0 + 10 * MS, 1)) == []
    assert pipe.on_frame(frame_msg(T0 + 43 * MS, 2)) == []
    out = pipe.on_frame(frame_msg(T0 + 77 * MS, 3))  # 超過 2 幀 → 最舊的放行
    assert [o["frame_id"] for o in out] == [1]
    assert out[0]["targets"][0]["status"] == "no_pose"
    assert pipe.pending == 2


def test_flush_outputs_everything_left():
    pipe = TargetWorldPipeline(CFG)
    pipe.on_frame(frame_msg(T0, 5))
    out = pipe.flush()
    assert [o["frame_id"] for o in out] == [5] and out[0]["targets"][0]["status"] == "no_pose"
    assert pipe.pending == 0


def test_invalid_max_pending():
    with pytest.raises(ValueError):
        TargetWorldPipeline(CFG, max_pending=-1)


def test_fixed_camera_time_offset_is_compensated():
    """影像時間戳固定比真實擷取晚 24 ms：不補償會偏，補償 -24 ms 後回到真值。"""
    lag = 24 * MS
    t_true = T0 + 500 * MS

    def run(offset):
        pipe = TargetWorldPipeline(CFG, frame_time_offset_ns=offset)
        for k in range(0, 1001, 50):
            pipe.on_pose(pose_msg(T0 + k * MS))
        f = frame_msg(t_true, 1)  # 用真實時刻的姿態產生 bbox
        f["timestamp_ns"] = t_true + lag  # 但時間戳晚了 24 ms
        out = pipe.on_frame(f)[0]
        assert out["timestamp_ns"] == t_true + lag  # 輸出的仍是原始影像時間戳
        p = out["targets"][0]["position_ned_m"]
        return math.dist((p["n"], p["e"]), CAR[:2])

    assert run(0) > 0.05
    assert run(-lag) < 0.02


def test_from_config_uses_pipeline_settings():
    from coord_transform.config import PipelineConfig

    pipe = TargetWorldPipeline.from_config(CFG, PipelineConfig(reference_point="center", frame_time_offset_ns=-5, max_pending_frames=3, pose_max_gap_ns=10, pose_horizon_ns=20))
    assert (pipe.reference, pipe.frame_time_offset_ns, pipe.max_pending) == ("center", -5, 3)
    assert (pipe.poses.max_gap_ns, pipe.poses.horizon_ns) == (10, 20)
