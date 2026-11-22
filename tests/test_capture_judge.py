"""抓拍判定（裁判）與策略場景整合。"""
import pytest

from coord_transform.config import load_camera_config
from demo.capture_judge import CaptureCriteria, CaptureJudge
from demo.run import ROOT
from fusion import DroneState
from llm_agent.commander import Commander

CAM = load_camera_config(ROOT / "config" / "camera.yaml")
S = 1_000_000_000


def feed(judge, drone, tn, te, seconds, t0=0.0, role="capturer"):
    out = []
    for k in range(1, int(round(seconds / 0.1)) + 1):
        out += judge.update(int((t0 + k * 0.1) * S), [drone], {drone.drone_id: role}, tn, te)
    return out


def test_judge_captures_after_hold_once_per_streak():
    j = CaptureJudge(CAM)
    above = DroneState("uav3", 0.0, 0.0, 30.0)
    evs = feed(j, above, 0.0, 0.0, 4.0)
    assert len(evs) == 1 and evs[0].role == "capturer"
    assert evs[0].t_s == pytest.approx(2.1)  # 0.1 s 開始成立，連續 2 s
    assert evs[0].target_px == pytest.approx(4.5 * CAM.intrinsics.fx / 30.0, abs=0.1)
    feed(j, above, 50.0, 0.0, 0.5, t0=4.0)  # 離開畫面
    assert len(feed(j, above, 0.0, 0.0, 2.5, t0=4.5)) == 1  # 再次成立 → 再記一次


def test_judge_rejects_off_center_short_and_too_small():
    assert feed(CaptureJudge(CAM), DroneState("uav1", 0.0, 0.0, 30.0), 10.0, 0.0, 3.0) == []  # 偏離中心 10 m
    assert feed(CaptureJudge(CAM), DroneState("uav1", 0.0, 0.0, 30.0), 0.0, 0.0, 1.5) == []  # 不到 2 s
    assert feed(CaptureJudge(CAM), DroneState("uav1", 0.0, 0.0, 100.0), 0.0, 0.0, 3.0) == []  # 太高，目標太小
    loose = CaptureJudge(CAM, CaptureCriteria(center_fraction=0.6))
    assert len(feed(loose, DroneState("uav1", 0.0, 0.0, 30.0), 10.0, 0.0, 3.0)) == 1


def test_engine_capture_strategy_against_evader():
    """策略場景整合：會逃的目標 + 捕捉策略（3 m/s）在 90 秒內由抓拍機完成抓拍、目標進入捕捉區。"""
    from coord_transform.config import load_error_model
    from demo.engine import DemoEngine
    from sim2d import EvaderConfig

    err = load_error_model(ROOT / "config" / "error_model.yaml")
    c = Commander(["uav1", "uav2", "uav3"], llm=None, strategy="capture")
    eng = DemoEngine(CAM, err, c, seed=0, evader=EvaderConfig(seed=0), drone_max_speed_mps=3.0)
    for _ in range(900):
        eng.step()
    m = eng.summary()
    assert m["time_to_capture_by_capturer_s"] is not None and m["zone_entered_at_s"] is not None
    assert m["min_drone_distance_m"] >= 5.0
