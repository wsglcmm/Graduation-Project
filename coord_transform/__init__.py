"""座標轉換模組（C）：像素 → 相機射線 → 機體 → NED 地面座標。

對外主要入口（即時流程）：
    pipe = TargetWorldPipeline(cfg, err)
    pipe.on_pose(pose_msg)         ← A 的姿態（規格 §4）
    pipe.on_frame(detection_frame) ← B 的偵測（§3）
    兩者都回傳「現在可以輸出」的 TargetWorld 列表（§5）
"""
from .pixel_to_ned import (
    CameraIntrinsics,
    CameraMount,
    GroundHit,
    Pose,
    Status,
    bbox_bottom_center,
    body_to_ned,
    camera_to_body,
    distort_normalized,
    ned_to_pixel,
    pixel_to_camera_ray,
    pixel_to_ned,
    undistort_normalized,
)
from .frames import gazebo_pose_to_ned, sdf_camera_pose_to_mount
from .pipeline import TargetWorldPipeline
from .pose_buffer import PoseBuffer, PoseSample
from .target_world import frame_to_target_world
from .uncertainty import ErrorModel, error_budget, horizontal_sigma_m

__all__ = [
    "distort_normalized",
    "undistort_normalized",
    "CameraIntrinsics",
    "CameraMount",
    "ErrorModel",
    "GroundHit",
    "Pose",
    "PoseBuffer",
    "PoseSample",
    "Status",
    "TargetWorldPipeline",
    "bbox_bottom_center",
    "body_to_ned",
    "camera_to_body",
    "error_budget",
    "frame_to_target_world",
    "gazebo_pose_to_ned",
    "sdf_camera_pose_to_mount",
    "horizontal_sigma_m",
    "ned_to_pixel",
    "pixel_to_camera_ray",
    "pixel_to_ned",
]
