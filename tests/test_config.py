"""camera.yaml 載入器測試。"""
import math
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from coord_transform.config import load_camera_config  # noqa: E402

REPO_YAML = Path(__file__).resolve().parents[1] / "config" / "camera.yaml"


def test_repo_camera_yaml_matches_gazebo_sdf():
    """A 回覆的 SDF：horizontal_fov 1.74 rad、1280x960、nadir。fx = 640 / tan(0.87)。"""
    cfg = load_camera_config(REPO_YAML)
    assert (cfg.intrinsics.width, cfg.intrinsics.height) == (1280, 960)
    assert cfg.intrinsics.fx == pytest.approx(640.0 / math.tan(0.87))  # ≈ 539.9
    assert cfg.intrinsics.fy == cfg.intrinsics.fx
    assert (cfg.intrinsics.cx, cfg.intrinsics.cy) == (640.0, 480.0)
    assert not cfg.intrinsics.has_distortion
    assert cfg.mount.pitch_rad == pytest.approx(-math.pi / 2)
    assert cfg.mount.offset_body_m == pytest.approx((0.0, 0.0, -0.10))
    assert cfg.ground_d_m == 0.0


def test_repo_mount_equals_sdf_conversion():
    """camera.yaml 手填的安裝角，必須和 SDF <pose>0 0 .10 0 1.5707 0</pose> 經 frames 換算的結果一致。"""
    from coord_transform import camera_to_body
    from coord_transform.frames import sdf_camera_pose_to_mount

    cfg = load_camera_config(REPO_YAML)
    from_sdf = sdf_camera_pose_to_mount(0, 0, 0.10, 0, 1.5707, 0)
    assert camera_to_body(cfg.mount) == pytest.approx(camera_to_body(from_sdf), abs=1e-3)
    assert cfg.mount.offset_body_m == pytest.approx(from_sdf.offset_body_m)


def test_hfov_rad_and_deg_are_equivalent(tmp_path):
    def load(hfov_line):
        f = tmp_path / "c.yaml"
        f.write_text(
            "image: {width: 1280, height: 960}\n"
            f"intrinsics: {{source: hfov, {hfov_line}}}\n"
            "mount: {pitch_deg: -90}\nground: {}\n",
            encoding="utf-8",
        )
        return load_camera_config(f).intrinsics.fx

    assert load("hfov_rad: 1.5707963267948966") == pytest.approx(load("hfov_deg: 90"))


def test_calibrated_source(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        "image: {width: 640, height: 480}\n"
        "intrinsics: {source: calibrated, fx: 536.07, fy: 536.02, cx: 342.37, cy: 235.54,\n"
        "             dist: [-0.26509, -0.04674, 0.00183, -0.00031, 0.25232]}\n"
        "mount: {pitch_deg: -45}\n"
        "ground: {plane_d_m: -2.0, max_range_m: 500}\n",
        encoding="utf-8",
    )
    cfg = load_camera_config(f)
    assert cfg.intrinsics.fx == 536.07 and cfg.intrinsics.has_distortion
    assert cfg.mount.pitch_rad == pytest.approx(-math.pi / 4)
    assert (cfg.ground_d_m, cfg.max_range_m) == (-2.0, 500.0)


@pytest.mark.parametrize(
    "text, msg",
    [
        ("image: {width: 640, height: 480}\nintrinsics: {source: hfov}\nmount: {pitch_deg: -90}\nground: {}\n", "hfov_deg"),
        ("image: {width: 640, height: 480}\nintrinsics: {source: hfov, hfov_deg: 90, hfov_rad: 1.57}\nmount: {pitch_deg: -90}\nground: {}\n", "不能都給"),
        ("image: {width: 640, height: 480}\nintrinsics: {source: hfov, hfov_rad: 4.0}\nmount: {pitch_deg: -90}\nground: {}\n", "180"),
        ("image: {width: 640, height: 480}\nintrinsics: {source: magic}\nmount: {pitch_deg: -90}\nground: {}\n", "source"),
        ("image: {width: 640, height: 480}\nintrinsics: {source: hfov, hfov_deg: 90, dist: [0, 0]}\nmount: {pitch_deg: -90}\nground: {}\n", "5 個係數"),
        ("image: {width: 640, height: 480}\nintrinsics: {source: hfov, hfov_deg: 90}\nground: {}\n", "mount"),
    ],
)
def test_bad_config_raises(tmp_path, text, msg):
    f = tmp_path / "bad.yaml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_camera_config(f)


# ---------- pipeline.yaml ----------
def test_repo_pipeline_yaml_loads_with_spec_defaults():
    from coord_transform.config import load_pipeline_config

    cfg = load_pipeline_config(REPO_YAML.parent / "pipeline.yaml")
    assert cfg.reference_point == "bottom_center"  # 規格 §2.4 未改之前維持
    assert cfg.frame_time_offset_ns == 0
    assert cfg.pose_max_gap_ns == 150_000_000 and cfg.pose_horizon_ns == 2_000_000_000


def test_pipeline_yaml_converts_ms_to_ns(tmp_path):
    from coord_transform.config import load_pipeline_config

    f = tmp_path / "p.yaml"
    f.write_text("reference_point: center_half_height\nframe_time_offset_ms: -24.5\n", encoding="utf-8")
    cfg = load_pipeline_config(f)
    assert cfg.reference_point == "center_half_height"
    assert cfg.frame_time_offset_ns == -24_500_000


@pytest.mark.parametrize("text, msg", [("frame_time_offset_ms: 1\n", "reference_point"), ("reference_point: magic\n", "之一")])
def test_bad_pipeline_yaml(tmp_path, text, msg):
    from coord_transform.config import load_pipeline_config

    f = tmp_path / "p.yaml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_pipeline_config(f)


def test_error_model_without_position_sigma_defaults_to_zero(tmp_path):
    from coord_transform.config import load_error_model

    f = tmp_path / "e.yaml"
    f.write_text("pixel_sigma_px: 1\nroll_pitch_sigma_deg: 1\nyaw_sigma_deg: 1\naltitude_sigma_m: 1\n", encoding="utf-8")
    assert load_error_model(f).horizontal_position_sigma_m == 0.0
