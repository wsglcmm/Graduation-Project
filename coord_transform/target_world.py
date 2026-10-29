"""Detection（B，規格 §3）+ Pose 緩衝（A，§4）→ TargetWorld（C 的輸出，§5）。

這是 C 對外的主要入口：B 每送來一幀偵測，就呼叫一次 frame_to_target_world()。
"""
from typing import Optional

from .config import CameraConfig
from .messages import SCHEMA_VERSION, DetectionFrame
from .pixel_to_ned import CameraMount, Status, pixel_to_ned
from .pose_buffer import PoseBuffer
from .reference_point import reference_point
from .uncertainty import ErrorModel, horizontal_sigma_m


def frame_to_target_world(
    frame,
    poses: PoseBuffer,
    cfg: CameraConfig,
    err: Optional[ErrorModel] = None,
    *,
    reference: str = "bottom_center",
    pose_time_ns: Optional[int] = None,
) -> dict:
    """把一幀偵測轉成規格 §5 的 TargetWorld dict（可直接 json.dumps）。

    frame：DetectionFrame 或規格 §3 的 dict。
    err：誤差模型；None 時 uncertainty_m 一律為 null。
    reference：bbox 參考點方法，見 reference_point.py。預設照規格 §2.4 用底邊中點。
    pose_time_ns：查姿態用的時間；None = 影像時間戳。影像與姿態時鐘有固定偏移時由 pipeline 傳入校正後的時間，
        輸出的 timestamp_ns 仍是原本的影像時間戳（規格 §5）。
    影像解析度與 camera.yaml 不符時直接報錯：內參是依解析度算的，對不上代表設定錯了，硬算只會得到錯的座標。
    """
    f = frame if isinstance(frame, DetectionFrame) else DetectionFrame.model_validate(frame)
    intr = cfg.intrinsics
    if (f.image_width, f.image_height) != (intr.width, intr.height):
        raise ValueError(
            f"影像解析度 {f.image_width}x{f.image_height} 與 camera.yaml 的 {intr.width}x{intr.height} 不符"
            "（確認 B 的 bbox 是原始影像座標、未縮放；或更新 camera.yaml）"
        )

    ip = poses.at(f.timestamp_ns if pose_time_ns is None else pose_time_ns)
    mount = cfg.mount
    if ip is not None and ip.gimbal_rad is not None:
        mount = CameraMount(pitch_rad=ip.gimbal_rad[0], yaw_rad=ip.gimbal_rad[1], offset_body_m=cfg.mount.offset_body_m)

    targets = []
    for det in f.detections:
        b = det.bbox
        u, v, plane = reference_point((b.x1, b.y1, b.x2, b.y2), det.cls, reference, ground_d_m=cfg.ground_d_m)
        hit = pixel_to_ned(u, v, intr, mount, ip.pose if ip else None, ground_d_m=plane, max_range_m=cfg.max_range_m)
        ok = hit.status == Status.OK
        sigma = None
        if ok and err is not None:
            sigma = horizontal_sigma_m(u, v, intr, mount, ip.pose, err, ground_d_m=plane)
        position = None
        if ok:  # 半高平面求交後，垂直投影回地面：輸出的 d 永遠是接地平面
            position = {"n": hit.position_ned_m[0], "e": hit.position_ned_m[1], "d": cfg.ground_d_m}
        targets.append(
            {
                "track_id": det.track_id,
                "class": det.cls,
                "status": hit.status.value,
                "position_ned_m": position,
                "uncertainty_m": None if sigma is None else round(sigma, 3),
            }
        )

    return {"schema_version": SCHEMA_VERSION, "frame_id": f.frame_id, "timestamp_ns": f.timestamp_ns, "targets": targets}
