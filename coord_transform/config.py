"""讀取 config/camera.yaml → CameraConfig。欄位意義見 camera.yaml 內註解。"""
import math
from dataclasses import dataclass
from pathlib import Path

from .pixel_to_ned import CameraIntrinsics, CameraMount
from .uncertainty import ErrorModel


@dataclass(frozen=True)
class CameraConfig:
    intrinsics: CameraIntrinsics
    mount: CameraMount
    ground_d_m: float
    max_range_m: float


def _require(d: dict, key: str, where: str):
    if key not in d or d[key] is None:
        raise ValueError(f"camera.yaml 缺少必要欄位：{where}.{key}")
    return d[key]


def _hfov_rad(intr: dict) -> float:
    """hfov_rad（與 Gazebo SDF 的 <horizontal_fov> 同單位，可直接照抄）或 hfov_deg，二擇一。"""
    has_rad, has_deg = intr.get("hfov_rad") is not None, intr.get("hfov_deg") is not None
    if has_rad == has_deg:
        raise ValueError("intrinsics 需要 hfov_rad 或 hfov_deg 其中一個（不能都給，也不能都沒給）")
    hfov = float(intr["hfov_rad"]) if has_rad else math.radians(float(intr["hfov_deg"]))
    if not 0.0 < hfov < math.pi:
        raise ValueError(f"水平視角必須介於 0 與 180° 之間，收到 {math.degrees(hfov):.1f}°")
    return hfov


def load_camera_config(path) -> CameraConfig:
    import yaml  # 只有讀設定檔才需要 PyYAML

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("camera.yaml 內容必須是 YAML 物件")

    image = _require(raw, "image", "(root)")
    width, height = int(_require(image, "width", "image")), int(_require(image, "height", "image"))

    intr = _require(raw, "intrinsics", "(root)")
    source = _require(intr, "source", "intrinsics")
    dist = tuple(float(c) for c in intr.get("dist", [0.0] * 5))
    if len(dist) != 5:
        raise ValueError("intrinsics.dist 必須有 5 個係數 (k1, k2, p1, p2, k3)")

    if source == "hfov":
        base = CameraIntrinsics.from_hfov(width, height, _hfov_rad(intr))
        intrinsics = CameraIntrinsics(base.fx, base.fy, base.cx, base.cy, width, height, dist)
    elif source == "calibrated":
        intrinsics = CameraIntrinsics(
            fx=float(_require(intr, "fx", "intrinsics")),
            fy=float(_require(intr, "fy", "intrinsics")),
            cx=float(_require(intr, "cx", "intrinsics")),
            cy=float(_require(intr, "cy", "intrinsics")),
            width=width,
            height=height,
            dist=dist,
        )
    else:
        raise ValueError(f"intrinsics.source 必須是 'hfov' 或 'calibrated'，收到 {source!r}")

    m = _require(raw, "mount", "(root)")
    offset = tuple(float(x) for x in m.get("offset_body_m", [0.0, 0.0, 0.0]))
    if len(offset) != 3:
        raise ValueError("mount.offset_body_m 必須有 3 個數值 (x前, y右, z下)")
    mount = CameraMount(
        pitch_rad=math.radians(float(_require(m, "pitch_deg", "mount"))),
        yaw_rad=math.radians(float(m.get("yaw_deg", 0.0))),
        offset_body_m=offset,
    )

    g = _require(raw, "ground", "(root)")
    return CameraConfig(
        intrinsics=intrinsics,
        mount=mount,
        ground_d_m=float(g.get("plane_d_m", 0.0)),
        max_range_m=float(g.get("max_range_m", 1000.0)),
    )


def load_error_model(path) -> ErrorModel:
    """讀取 config/error_model.yaml → ErrorModel。角度在 YAML 用度，內部轉成弧度。"""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("error_model.yaml 內容必須是 YAML 物件")
    values = {
        "pixel_sigma_px": float(_require(raw, "pixel_sigma_px", "(root)")),
        "roll_pitch_sigma_rad": math.radians(float(_require(raw, "roll_pitch_sigma_deg", "(root)"))),
        "yaw_sigma_rad": math.radians(float(_require(raw, "yaw_sigma_deg", "(root)"))),
        "altitude_sigma_m": float(_require(raw, "altitude_sigma_m", "(root)")),
        "horizontal_position_sigma_m": float(raw.get("horizontal_position_sigma_m", 0.0)),
    }
    negative = [k for k, v in values.items() if not v >= 0.0]
    if negative:
        raise ValueError(f"error_model.yaml 的誤差必須 ≥ 0：{negative}")
    return ErrorModel(**values)


@dataclass(frozen=True)
class PipelineConfig:
    """config/pipeline.yaml：即時流程的行為設定（不是相機參數）。"""

    reference_point: str = "bottom_center"
    frame_time_offset_ns: int = 0
    max_pending_frames: int = 10
    pose_max_gap_ns: int = 150_000_000
    pose_horizon_ns: int = 2_000_000_000


def load_pipeline_config(path) -> PipelineConfig:
    import yaml

    from .reference_point import REFERENCE_METHODS

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pipeline.yaml 內容必須是 YAML 物件")
    ref = str(_require(raw, "reference_point", "(root)"))
    if ref not in REFERENCE_METHODS:
        raise ValueError(f"reference_point 必須是 {REFERENCE_METHODS} 之一，收到 {ref!r}")
    ms = lambda key, default: int(round(float(raw.get(key, default)) * 1_000_000))  # noqa: E731
    cfg = PipelineConfig(
        reference_point=ref,
        frame_time_offset_ns=ms("frame_time_offset_ms", 0.0),
        max_pending_frames=int(raw.get("max_pending_frames", 10)),
        pose_max_gap_ns=ms("pose_max_gap_ms", 150.0),
        pose_horizon_ns=ms("pose_horizon_ms", 2000.0),
    )
    if cfg.max_pending_frames < 0 or cfg.pose_max_gap_ns <= 0 or cfg.pose_horizon_ns <= 0:
        raise ValueError("max_pending_frames 須 ≥ 0，pose_max_gap_ms 與 pose_horizon_ms 須 > 0")
    return cfg
