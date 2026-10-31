"""姿態時間緩衝：把 A 的姿態序列對齊到 B 的影像時間戳（規格 §2.3）。

為什麼需要：影像約 30 Hz、姿態約 50 Hz，兩者時間戳不會剛好對上；而且 B 的偵測結果
要等推論完成才送到，此時「最新一筆姿態」已經是之後的姿態了。用錯時刻的姿態，
誤差約為 高度 × 角速度 × 時間差（30 m、30°/s、50 ms → 約 0.8 m）。
A 的 flight_api 改用 MAVSDK odometry 後約 100 Hz，位置與姿態共用 time_usec（sim time），見規格 §2.3。

規則：
    - 位置：線性插值；姿態：四元數 slerp；雲台角（若有）：線性插值。
    - 不外插：查詢時間不在 [最早, 最新] 之間 → None（呼叫端回報 status = "no_pose"）。
    - 兩筆姿態相隔超過 max_gap_ns（掉資料）→ None，不在大缺口中硬插。
"""
import bisect
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import rotations
from .messages import PoseMessage
from .pixel_to_ned import Pose

DEFAULT_MAX_GAP_NS = 150_000_000  # 規格 §2.3：相鄰兩筆超過 150 ms 就不插值（原本為 20 Hz 姿態設計，100 Hz 下仍適用）
DEFAULT_HORIZON_NS = 2_000_000_000  # 規格 §2.3：至少保留最近 2 秒


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    position_ned_m: Tuple[float, float, float]
    quaternion_wxyz: Tuple[float, float, float, float]
    gimbal_rad: Optional[Tuple[float, float]] = None  # (pitch, yaw) 相對機體；None = 固定相機

    @classmethod
    def from_message(cls, msg) -> "PoseSample":
        m = msg if isinstance(msg, PoseMessage) else PoseMessage.model_validate(msg)
        p = m.position_ned_m
        g = (m.gimbal_rad.pitch, m.gimbal_rad.yaw) if m.gimbal_rad is not None else None
        return cls(m.timestamp_ns, (p.n, p.e, p.d), tuple(float(c) for c in m.quaternion()), g)

    def to_pose(self) -> Pose:
        roll, pitch, yaw = rotations.quat_to_euler(self.quaternion_wxyz)
        n, e, d = self.position_ned_m
        return Pose(n=n, e=e, d=d, roll=roll, pitch=pitch, yaw=yaw)


@dataclass(frozen=True)
class InterpolatedPose:
    pose: Pose
    gimbal_rad: Optional[Tuple[float, float]]


class PoseBuffer:
    def __init__(self, max_gap_ns: int = DEFAULT_MAX_GAP_NS, horizon_ns: int = DEFAULT_HORIZON_NS):
        if max_gap_ns <= 0 or horizon_ns <= 0:
            raise ValueError("max_gap_ns 與 horizon_ns 必須為正")
        self.max_gap_ns = max_gap_ns
        self.horizon_ns = horizon_ns
        self._t: list = []
        self._s: list = []

    def __len__(self) -> int:
        return len(self._s)

    @property
    def latest_ns(self):
        """最新一筆姿態的時間戳；緩衝為空時為 None。"""
        return self._t[-1] if self._t else None

    def add(self, sample) -> None:
        """加入一筆姿態（PoseSample、PoseMessage 或規格 §4 的 dict）。時間戳必須嚴格遞增。"""
        if not isinstance(sample, PoseSample):
            sample = PoseSample.from_message(sample)
        if self._t and sample.timestamp_ns <= self._t[-1]:
            raise ValueError(
                f"姿態時間戳必須嚴格遞增：收到 {sample.timestamp_ns}，最新一筆是 {self._t[-1]}"
                "（確認 A 使用的是 sim time，且沒有重送舊資料）"
            )
        self._t.append(sample.timestamp_ns)
        self._s.append(sample)
        cut = bisect.bisect_left(self._t, sample.timestamp_ns - self.horizon_ns)
        if cut > 0:
            del self._t[:cut], self._s[:cut]

    def at(self, t_ns: int) -> Optional[InterpolatedPose]:
        """查詢 t_ns 時刻的姿態；無法可靠插值時回傳 None。"""
        if not self._t or t_ns < self._t[0] or t_ns > self._t[-1]:
            return None
        i = bisect.bisect_left(self._t, t_ns)
        if self._t[i] == t_ns:
            s = self._s[i]
            return InterpolatedPose(s.to_pose(), s.gimbal_rad)

        t0, t1 = self._t[i - 1], self._t[i]
        if t1 - t0 > self.max_gap_ns:
            return None
        s0, s1 = self._s[i - 1], self._s[i]
        a = (t_ns - t0) / (t1 - t0)

        pos = (1 - a) * np.array(s0.position_ned_m) + a * np.array(s1.position_ned_m)
        q = rotations.slerp(s0.quaternion_wxyz, s1.quaternion_wxyz, a)
        roll, pitch, yaw = rotations.quat_to_euler(q)

        gimbal = None
        if s0.gimbal_rad is not None and s1.gimbal_rad is not None:
            g0, g1 = np.array(s0.gimbal_rad), np.array(s1.gimbal_rad)
            gimbal = tuple(float(x) for x in (1 - a) * g0 + a * g1)
        elif (s0.gimbal_rad is None) != (s1.gimbal_rad is None):
            return None  # 雲台資料時有時無，不可信

        pose = Pose(n=float(pos[0]), e=float(pos[1]), d=float(pos[2]), roll=roll, pitch=pitch, yaw=yaw)
        return InterpolatedPose(pose, gimbal)
