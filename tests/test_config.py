"""camera.yaml 載入器測試。"""
import math
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from coord_transform.config import load_camera_config  # noqa: E402

REPO_YAML = Path(__file__).resolve().parents[1] / "config" / "camera.yaml"


def test_repo_camera_yaml_loads():
    cfg = load_camera_config(REPO_YAML)
    assert cfg.intrinsics.width == 1280 and cfg.intrinsics.height == 720
    assert cfg.intrinsics.fx == pytest.approx(640.0)  # hfov 90° 的假設值
    assert cfg.mount.pitch_rad == pytest.approx(-math.pi / 2)
    assert cfg.ground_d_m == 0.0


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
