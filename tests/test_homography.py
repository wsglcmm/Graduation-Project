"""地面標定的已知答案測試：先自己定一個 H，產生對應點，再看 fit 能不能解回來。"""
import json
import math

import numpy as np
import pytest

from coord_transform.homography import GroundHomography, fit_homography, load_video_ground

# 一個「真的」透視 homography：旋轉 + 縮放 + 平移 + 輕微透視
H_TRUE = np.array([[0.05, -0.01, -20.0], [0.012, 0.048, -15.0], [1e-5, 2e-5, 1.0]])
PIXELS = [(100, 80), (1180, 60), (1200, 900), (90, 920), (640, 480), (300, 700)]


def project(H, pts):
    out = []
    for u, v in pts:
        p = H @ np.array([u, v, 1.0])
        out.append((p[0] / p[2], p[1] / p[2]))
    return out


def test_fit_recovers_known_homography():
    g = GroundHomography.fit(PIXELS, project(H_TRUE, PIXELS))
    assert g.H == pytest.approx(H_TRUE / H_TRUE[2, 2], rel=1e-7, abs=1e-10)
    assert g.rms_error_m == pytest.approx(0.0, abs=1e-9)
    # 沒用來標定的點也要對
    assert g.pixel_to_ground(777, 333) == pytest.approx(project(H_TRUE, [(777, 333)])[0], abs=1e-8)


def test_exactly_four_points_is_enough():
    g = GroundHomography.fit(PIXELS[:4], project(H_TRUE, PIXELS[:4]))
    assert g.pixel_to_ground(640, 480) == pytest.approx(project(H_TRUE, [(640, 480)])[0], abs=1e-7)


def test_round_trip_pixel_ground():
    g = GroundHomography(H_TRUE / H_TRUE[2, 2])
    n, e = g.pixel_to_ground(512.3, 222.7)
    assert g.ground_to_pixel(n, e) == pytest.approx((512.3, 222.7), abs=1e-8)


def test_noisy_points_report_reprojection_error():
    rng = np.random.default_rng(0)
    ground = [(n + rng.normal(0, 0.1), e + rng.normal(0, 0.1)) for n, e in project(H_TRUE, PIXELS)]
    g = GroundHomography.fit(PIXELS, ground)
    assert 0.01 < g.rms_error_m < 0.2


@pytest.mark.parametrize(
    "src",
    [
        [(0, 0), (1, 1), (2, 2), (3, 3)],  # 共線
        [(0, 0), (1, 0), (2, 0), (3, 1)][:3],  # 少於 4 點
    ],
)
def test_degenerate_points_rejected(src):
    with pytest.raises(ValueError):
        fit_homography(src, src)


def test_from_scale_north_up():
    """比例 0.05 m/px、原點在 (640, 480)：往上 100 px = 北 5 m，往右 100 px = 東 5 m。"""
    g = GroundHomography.from_scale(0.05, (640, 480))
    assert g.pixel_to_ground(640, 380) == pytest.approx((5.0, 0.0))
    assert g.pixel_to_ground(740, 480) == pytest.approx((0.0, 5.0))


def test_uncertainty_for_pure_scale_is_scale_times_sigma_times_sqrt2():
    g = GroundHomography.from_scale(0.05, (0, 0))
    assert g.uncertainty_m(300, 300, pixel_sigma_px=2.0) == pytest.approx(0.05 * 2.0 * math.sqrt(2))


def test_jacobian_matches_finite_difference():
    g = GroundHomography(H_TRUE / H_TRUE[2, 2])
    u, v, h = 400.0, 300.0, 1e-4
    fd = np.column_stack(
        [
            (np.array(g.pixel_to_ground(u + h, v)) - g.pixel_to_ground(u - h, v)) / (2 * h),
            (np.array(g.pixel_to_ground(u, v + h)) - g.pixel_to_ground(u, v - h)) / (2 * h),
        ]
    )
    assert g.jacobian(u, v) == pytest.approx(fd, rel=1e-6)


def test_load_video_ground_points_and_scale(tmp_path):
    f = tmp_path / "g.json"
    pts = [{"pixel": list(p), "ground_ne_m": list(q)} for p, q in zip(PIXELS, project(H_TRUE, PIXELS))]
    f.write_text(json.dumps({"video_id": "nexus_0", "points": pts}), encoding="utf-8")
    g = load_video_ground(f)
    assert g.video_id == "nexus_0" and g.rms_error_m == pytest.approx(0.0, abs=1e-9)

    f.write_text(json.dumps({"meters_per_pixel": 0.1, "origin_pixel": [0, 0]}), encoding="utf-8")
    assert load_video_ground(f).pixel_to_ground(10, 0) == pytest.approx((0.0, 1.0))

    f.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_video_ground(f)
