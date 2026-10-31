"""C 的即時處理流程：同時接 A 的姿態串流與 B 的偵測幀，輸出 TargetWorld。

為什麼需要「等待」：姿態是離散取樣（目前 odometry 約 100 Hz）。偵測幀送到時，若影像時間晚於最新一筆姿態，
就只能外插，而 PoseBuffer 刻意不外插。所以先把這種幀放進佇列，等下一筆姿態到了、
可以內插了再處理（ROS 的 message_filters 也是同一個想法）。

規則：
    - 佇列先進先出，輸出順序與偵測幀到達順序一致。
    - 佇列超過 max_pending 幀（姿態串流中斷）時，最舊的幀直接輸出，此時它的 status 會是 no_pose，
      避免無限等待、記憶體一直長大。
"""
from typing import List, Optional

from .config import CameraConfig, PipelineConfig
from .messages import DetectionFrame
from .pose_buffer import PoseBuffer
from .target_world import frame_to_target_world
from .uncertainty import ErrorModel

DEFAULT_MAX_PENDING = 10  # 30 Hz 影像下約 330 ms


class TargetWorldPipeline:
    def __init__(
        self,
        cfg: CameraConfig,
        err: Optional[ErrorModel] = None,
        *,
        reference: str = "bottom_center",
        poses: Optional[PoseBuffer] = None,
        max_pending: int = DEFAULT_MAX_PENDING,
        frame_time_offset_ns: int = 0,
    ):
        """frame_time_offset_ns：影像時間戳要加多少才等於姿態時鐘的同一時刻。
        例：影像時間戳固定比真實擷取晚 24 ms → 填 -24 ms。可用 analysis/replay.py 從模擬紀錄估計。"""
        if max_pending < 0:
            raise ValueError("max_pending 必須 ≥ 0")
        self.cfg, self.err, self.reference = cfg, err, reference
        self.poses = poses if poses is not None else PoseBuffer()
        self.max_pending = max_pending
        self.frame_time_offset_ns = int(frame_time_offset_ns)
        self._pending: List[DetectionFrame] = []

    @classmethod
    def from_config(cls, cam: CameraConfig, pipe: PipelineConfig, err: Optional[ErrorModel] = None) -> "TargetWorldPipeline":
        """由 config/camera.yaml + config/pipeline.yaml (+ error_model.yaml) 建立。"""
        return cls(
            cam,
            err,
            reference=pipe.reference_point,
            poses=PoseBuffer(max_gap_ns=pipe.pose_max_gap_ns, horizon_ns=pipe.pose_horizon_ns),
            max_pending=pipe.max_pending_frames,
            frame_time_offset_ns=pipe.frame_time_offset_ns,
        )

    @property
    def pending(self) -> int:
        return len(self._pending)

    def on_pose(self, msg) -> List[dict]:
        """收到一筆姿態（規格 §4）。回傳因此可以處理的 TargetWorld 列表（可能為空）。"""
        self.poses.add(msg)
        return self._drain()

    def on_frame(self, frame) -> List[dict]:
        """收到一幀偵測（規格 §3）。回傳可以處理的 TargetWorld 列表（可能為空，代表還在等姿態）。"""
        f = frame if isinstance(frame, DetectionFrame) else DetectionFrame.model_validate(frame)
        self._pending.append(f)
        return self._drain()

    def flush(self) -> List[dict]:
        """結束時把佇列全部輸出（等不到姿態的會是 no_pose）。"""
        out = [self._convert(f) for f in self._pending]
        self._pending.clear()
        return out

    def _ready(self, f: DetectionFrame) -> bool:
        latest = self.poses.latest_ns
        return latest is not None and latest >= f.timestamp_ns + self.frame_time_offset_ns

    def _drain(self) -> List[dict]:
        out = []
        while self._pending and (self._ready(self._pending[0]) or len(self._pending) > self.max_pending):
            out.append(self._convert(self._pending.pop(0)))
        return out

    def _convert(self, f: DetectionFrame) -> dict:
        return frame_to_target_world(
            f, self.poses, self.cfg, self.err, reference=self.reference, pose_time_ns=f.timestamp_ns + self.frame_time_offset_ns
        )
