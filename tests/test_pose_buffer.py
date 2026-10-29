"""姿態緩衝與四元數的測試。預期值皆為手算（中點、跨 ±π、已知旋轉角）。"""
import math

import numpy as np
import pytest
from pydantic import ValidationError

from coord_transform import rotations
from coord_transform.messages import PoseMessage
from coord_transform.pose_buffer import PoseBuffer, PoseSample

MS = 1_000_000  # 1 ms = 1e6 ns
T0 = 1_730_000_000_000_000_000


def msg(t_ns, n=0.0, e=0.0, d=-30.0, roll=0.0, pitch=0.0, yaw=0.0, **extra):
    m = {
        "timestamp_ns": t_ns,
        "position_ned_m": {"n": n, "e": e, "d": d},
        "attitude_rad": {"roll": roll, "pitch": pitch, "yaw": yaw},
    }
    m.update(extra)
    return m


# ---------- 四元數 ----------
@pytest.mark.parametrize("rpy", [(0, 0, 0), (0.1, -0.2, 0.3), (-0.5, 0.4, 3.0), (0.2, 0.1, -2.9)])
def test_euler_quat_round_trip(rpy):
    assert rotations.quat_to_euler(rotations.euler_to_quat(*rpy)) == pytest.approx(rpy, abs=1e-12)


def test_quat_matches_rotation_matrix_convention():
    """四元數必須和 pixel_to_ned 的 R_nb = Rz·Ry·Rx 是同一個旋轉。"""
    from coord_transform import body_to_ned

    r, p, y = 0.1, -0.2, 0.7
    w, x, yy, z = rotations.euler_to_quat(r, p, y)
    R = np.array(
        [
            [1 - 2 * (yy * yy + z * z), 2 * (x * yy - w * z), 2 * (x * z + w * yy)],
            [2 * (x * yy + w * z), 1 - 2 * (x * x + z * z), 2 * (yy * z - w * x)],
            [2 * (x * z - w * yy), 2 * (yy * z + w * x), 1 - 2 * (x * x + yy * yy)],
        ]
    )
    assert R == pytest.approx(body_to_ned(r, p, y), abs=1e-12)


def test_angle_between_known_rotation():
    q0 = rotations.euler_to_quat(0, 0, 0)
    q1 = rotations.euler_to_quat(0, 0, math.radians(30))
    assert math.degrees(rotations.angle_between(q0, q1)) == pytest.approx(30.0)
    assert rotations.angle_between(q1, -q1) == pytest.approx(0.0, abs=1e-6)  # q 與 -q 同一姿態


def test_slerp_midpoint_of_yaw():
    q = rotations.slerp(rotations.euler_to_quat(0, 0, 0), rotations.euler_to_quat(0, 0, 1.0), 0.5)
    assert rotations.quat_to_euler(q)[2] == pytest.approx(0.5)


# ---------- 緩衝插值 ----------
def make_buffer(*msgs, **kw):
    buf = PoseBuffer(**kw)
    for m in msgs:
        buf.add(m)
    return buf


def test_exact_timestamp_returns_that_sample():
    buf = make_buffer(msg(T0, n=1.0), msg(T0 + 20 * MS, n=2.0))
    assert buf.at(T0 + 20 * MS).pose.n == pytest.approx(2.0)


def test_position_is_linearly_interpolated():
    buf = make_buffer(msg(T0, n=0.0, e=10.0), msg(T0 + 20 * MS, n=4.0, e=10.0))
    p = buf.at(T0 + 5 * MS).pose  # 1/4 處
    assert (p.n, p.e, p.d) == pytest.approx((1.0, 10.0, -30.0))


def test_yaw_across_pi_takes_short_path():
    """179° → -179° 中點是 180°（±π），不是線性插值會得到的 0°。"""
    buf = make_buffer(msg(T0, yaw=math.radians(179)), msg(T0 + 20 * MS, yaw=math.radians(-179)))
    yaw = buf.at(T0 + 10 * MS).pose.yaw
    assert abs(math.degrees(yaw)) == pytest.approx(180.0, abs=1e-9)


def test_roll_midpoint():
    buf = make_buffer(msg(T0, roll=0.0), msg(T0 + 20 * MS, roll=0.2))
    p = buf.at(T0 + 10 * MS).pose
    assert (p.roll, p.pitch, p.yaw) == pytest.approx((0.1, 0.0, 0.0), abs=1e-12)


def test_no_extrapolation_outside_buffer():
    buf = make_buffer(msg(T0), msg(T0 + 20 * MS))
    assert buf.at(T0 - 1) is None
    assert buf.at(T0 + 20 * MS + 1) is None  # 偵測時間晚於最新姿態：等下一筆，不外插
    assert PoseBuffer().at(T0) is None


def test_large_gap_is_not_interpolated():
    buf = make_buffer(msg(T0), msg(T0 + 150 * MS), max_gap_ns=100 * MS)
    assert buf.at(T0 + 75 * MS) is None


def test_old_samples_are_trimmed_but_horizon_kept():
    buf = PoseBuffer(horizon_ns=2000 * MS)
    for k in range(200):  # 50 Hz × 4 秒
        buf.add(msg(T0 + k * 20 * MS))
    latest = T0 + 199 * 20 * MS
    assert buf.at(latest - 2000 * MS) is not None  # 2 秒前還查得到
    assert buf.at(latest - 2100 * MS) is None
    assert len(buf) <= 102


def test_timestamps_must_increase():
    buf = make_buffer(msg(T0 + 20 * MS))
    with pytest.raises(ValueError, match="嚴格遞增"):
        buf.add(msg(T0 + 20 * MS))
    with pytest.raises(ValueError):
        buf.add(msg(T0))


def test_gimbal_is_interpolated_and_must_be_consistent():
    g = lambda p: {"pitch": p, "yaw": 0.0}  # noqa: E731
    buf = make_buffer(msg(T0, gimbal_rad=g(-1.0)), msg(T0 + 20 * MS, gimbal_rad=g(-1.2)))
    assert buf.at(T0 + 10 * MS).gimbal_rad == pytest.approx((-1.1, 0.0))

    mixed = make_buffer(msg(T0, gimbal_rad=g(-1.0)), msg(T0 + 20 * MS))
    assert mixed.at(T0 + 10 * MS) is None


# ---------- Pose 訊息驗證 ----------
def test_quaternion_is_preferred_and_checked_against_euler():
    yaw = 1.571
    q = list(rotations.euler_to_quat(0.0, 0.0, yaw))
    s = PoseSample.from_message(msg(T0, yaw=yaw, quaternion_wxyz=q))
    assert s.to_pose().yaw == pytest.approx(yaw)

    q_enu = list(rotations.euler_to_quat(0.0, 0.0, math.pi / 2 - yaw))  # 常見錯誤：ENU 的 yaw 沒轉成 NED
    with pytest.raises(ValidationError, match="相差"):
        PoseMessage.model_validate(msg(T0, yaw=yaw, quaternion_wxyz=q_enu))


@pytest.mark.parametrize(
    "bad",
    [
        msg(1.73e18),  # 時間戳是 float
        msg("1730000000000000000"),  # 時間戳是字串
        msg(T0, quaternion_wxyz=[1.0, 0.0, 0.0]),
        msg(T0, quaternion_wxyz=[2.0, 0.0, 0.0, 0.0]),
        {**msg(T0), "postion_ned_m": {"n": 0, "e": 0, "d": 0}},  # 欄位名打錯
        {k: v for k, v in msg(T0).items() if k != "attitude_rad"},
    ],
)
def test_invalid_pose_messages_rejected(bad):
    with pytest.raises(ValidationError):
        PoseMessage.model_validate(bad)


def test_nan_pose_rejected():
    with pytest.raises(ValidationError):
        PoseMessage.model_validate(msg(T0, n=float("nan")))
