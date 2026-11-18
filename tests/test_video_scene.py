"""影片場景：軌跡建立（B 的 track_id / C 的暫代關聯）、選目標、VideoTarget 內插。"""
import pytest

from coord_transform.homography import GroundHomography
from demo.video_scene import (PLACEHOLDER_ID_MIN, VideoTarget, VideoTrack, associate, has_real_track_ids, pick_target,
                              tracks_from_ids)

H = GroundHomography.from_scale(0.1, (500, 500))  # 0.1 m/px，(500, 500) 為原點，北朝上


def det(tid, cls, u, v_bottom):
    return {"track_id": tid, "class": cls, "confidence": 0.9, "bbox": {"x1": u - 10, "y1": v_bottom - 20, "x2": u + 10, "y2": v_bottom}}


def frames_two_cars(ids=True, n=30):
    """兩台車同時往東開（2 m/s = 20 px/s）、相距 10 m；每 0.1 s 一幀。"""
    out, pid = [], PLACEHOLDER_ID_MIN
    for k in range(n):
        dets = []
        for car, v in ((1, 500), (2, 400)):  # v = 500 → n = 0；v = 400 → n = 10 m
            tid = car if ids else pid
            pid += 1
            dets.append(det(tid, "car", 500 + 2 * k, v))
        out.append({"frame_id": k, "timestamp_ns": int(k * 1e8), "detections": dets})
    return out


def test_ground_point_uses_bottom_center():
    t = tracks_from_ids(frames_two_cars(), H)
    assert t[2].samples[0][1:] == pytest.approx((10.0, 0.0))  # 底邊中點 (500, 400) → n = 10、e = 0


def test_real_ids_vs_placeholder():
    assert has_real_track_ids(frames_two_cars(ids=True))
    assert not has_real_track_ids(frames_two_cars(ids=False))


def test_association_recovers_two_tracks_from_placeholder_ids():
    tracks = associate(frames_two_cars(ids=False), H)
    long = [t for t in tracks.values() if len(t.samples) == 30]
    assert len(long) == 2 and len(tracks) == 2
    assert {round(t.samples[0][1]) for t in long} == {0, 10}


def test_association_respects_class_groups_and_gate():
    fr = [{"frame_id": 0, "timestamp_ns": 0, "detections": [det(PLACEHOLDER_ID_MIN, "car", 500, 500)]},
          {"frame_id": 1, "timestamp_ns": int(1e8), "detections": [det(PLACEHOLDER_ID_MIN + 1, "person", 500, 500)]},
          {"frame_id": 2, "timestamp_ns": int(2e8), "detections": [det(PLACEHOLDER_ID_MIN + 2, "car", 900, 500)]}]
    assert len(associate(fr, H)) == 3  # 換類別群組、跳太遠（40 m）都開新軌跡


def test_pick_target_and_video_target_interpolation():
    tracks = tracks_from_ids(frames_two_cars(), H)
    tracks[3] = VideoTrack(3, "car", [(0.0, 0.0, 0.0), (5.0, 0.0, 0.5)])  # 持續久但幾乎沒動
    tr = pick_target(tracks, min_path_m=5.0)
    assert tr.track_id in (1, 2) and tr.path_length_m() == pytest.approx(5.8, abs=0.01)
    vt = VideoTarget(tr, start_s=0.0)
    assert (vt.n, vt.e) == pytest.approx(tr.samples[0][1:])
    vt.step(1.45)
    assert vt.e == pytest.approx(2.9, abs=1e-6) and vt.video_time_s == pytest.approx(1.45)
    vt.step(10.0)
    assert vt.finished and vt.e == pytest.approx(5.8)  # 軌跡結束後停在最後位置
    assert vt.cfg.arena_radius_m > 10.0
    with pytest.raises(ValueError):
        pick_target(tracks, classes=("bus",))


def test_video_scene_end_to_end(tmp_path):
    """run.load_video_scene（暫定比例尺 + 佔位 ID → C 的暫代關聯）→ DemoEngine 用影片目標跑捕捉策略。"""
    import argparse
    import json

    from coord_transform.config import load_camera_config, load_error_model
    from demo.engine import DemoEngine
    from demo.run import ROOT, load_video_scene, make_commander

    vdir = tmp_path / "road99"
    vdir.mkdir()
    rows = []
    for k in range(300):  # 一台車 30 s 往東開 4 m/s（0.1 m/px → 40 px/s）
        f = {"frame_id": k, "timestamp_ns": int(k * 1e8), "source": "video", "image_width": 1000, "image_height": 1000,
             "detections": [det(PLACEHOLDER_ID_MIN + k, "car", 300 + 4 * k, 500)]}
        rows.append(json.dumps(f))
    (vdir / "detections.jsonl").write_text("\n".join(rows), encoding="utf-8")
    args = argparse.Namespace(video=str(vdir), ground=str(tmp_path / "none.json"), ground_scale=0.1, video_track=None,
                              calibrate=False, video_file="x.mp4", web=False)
    target, info, _ = load_video_scene(args)
    assert info["provisional_ids"] and info["provisional_ground"] and info["width"] == 1000
    assert target.track.samples[0][1:] == pytest.approx((0.0, -20.0))  # 畫面中心 (500, 500) 為原點
    from planner.capture import CaptureConfig

    eng = DemoEngine(load_camera_config(ROOT / "config" / "camera.yaml"), load_error_model(ROOT / "config" / "error_model.yaml"),
                     make_commander("rule", 2.0, "capture", "multi"), seed=0, drone_max_speed_mps=5.0,
                     target_source=target, capture_cfg=CaptureConfig(zone_direction="velocity"))
    for _ in range(250):
        eng.step()
    assert eng.rabbit is target and target.video_time_s == pytest.approx(25.0, abs=0.05)
    assert eng.commander.capture_cfg.zone_direction == "velocity"
    assert all(a.capture_cfg is eng.commander.capture_cfg for a in eng.commander.agents.values())
    eng.commander.close()


def test_pick_target_prefers_moving_time_over_duration():
    parked = VideoTrack(1, "car", [(0.0, 0.0, 0.0), (5.0, 0.0, 20.0), (60.0, 0.0, 20.0)])  # 動 5 s、停 55 s
    flash = VideoTrack(2, "car", [(0.0, 0.0, 0.0), (3.0, 0.0, 100.0)])  # 3 s 穿過畫面
    steady = VideoTrack(3, "car", [(0.0, 0.0, 0.0), (20.0, 0.0, 40.0)])  # 20 s 等速
    assert parked.moving_time_s() == pytest.approx(5.0)
    assert pick_target({1: parked, 2: flash, 3: steady}).track_id == 3


def test_moving_time_ignores_frame_jitter():
    jitter = VideoTrack(1, "car", [(k * 0.02, 0.0, 0.05 * (k % 2)) for k in range(500)])  # 50 fps、原地抖 5 cm
    assert jitter.moving_time_s() == 0.0
