"""讀取 config/camera.yaml → CameraConfig。欄位意義見 camera.yaml 內註解。"""
import math
from dataclasses import dataclass
from pathlib import Path

from .pixel_to_ned import CameraIntrinsics, CameraMount


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
        base = CameraIntrinsics.from_hfov(width, height, math.radians(float(_require(intr, "hfov_deg", "intrinsics"))))
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
