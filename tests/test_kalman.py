"""卡爾曼濾波目標追蹤的測試：初始化、估速度、平滑、遮擋外推、lost 判定。"""
import math

import numpy as np
import pytest

from fusion import TargetTracker, TrackerConfig
from fusion.fuse import FusedMeasurement

S = 1_000_000_000


# ---------- 卡爾曼 ----------
def meas(n, e, u=1.0, by=("uav1",)):
    return FusedMeasurement(1, n, e, u, by)


def test_tracker_initializes_on_first_measurement():
    tr = TargetTracker(1)
    assert tr.step(0, None) is None
    st = tr.step(0, meas(3.0, 4.0))
    assert st["status"] == "tracking" and st["position_world_ned_m"] == {"n": 3.0, "e": 4.0}


def test_tracker_estimates_velocity_and_smooths_noise():
    rng = np.random.default_rng(2)
    tr = TargetTracker(1, cfg=TrackerConfig(accel_sigma_mps2=0.3))
    raw_err, kf_err = [], []
    for k in range(300):  # 30 秒、10 Hz，等速 (2, 1) m/s
        t = k * S // 10
        truth = np.array([2.0, 1.0]) * (t / S)
        z = truth + rng.normal(0, 1.0 / math.sqrt(2), 2)
        st = tr.step(t, meas(*z))
        if k > 50:
            raw_err.append(np.sum((z - truth) ** 2))
            p = st["position_world_ned_m"]
            kf_err.append((p["n"] - truth[0]) ** 2 + (p["e"] - truth[1]) ** 2)
    v = st["velocity_world_ned_mps"]
    assert (v["n"], v["e"]) == pytest.approx((2.0, 1.0), abs=0.15)
    assert math.sqrt(np.mean(kf_err)) < 0.6 * math.sqrt(np.mean(raw_err))


def test_tracker_predicts_through_occlusion_then_lost():
    tr = TargetTracker(1, cfg=TrackerConfig(lost_after_s=3.0))
    for k in range(50):
        tr.step(k * S // 10, meas(1.0 * k / 10, 0.0, u=0.2))  # 1 m/s 往北
    u_before = tr.state()["uncertainty_m"]
    st = tr.step(6 * S, None)  # 1 秒沒觀測
    assert st["status"] == "predicted"
    assert st["position_world_ned_m"]["n"] == pytest.approx(6.0, abs=0.3)
    assert st["uncertainty_m"] > u_before  # 看不到時誤差圈變大
    assert tr.step(9 * S, None)["status"] == "lost"


def test_time_cannot_go_backwards():
    tr = TargetTracker(1)
    tr.step(S, meas(0, 0))
    with pytest.raises(ValueError):
        tr.step(0, None)
