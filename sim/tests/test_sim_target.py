"""sim_target.Route 的測試（不需要 Gazebo）：python -m unittest discover -s sim/tests -v"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from sim_target import Route  # noqa: E402


class TestRoute(unittest.TestCase):
    def assertPose(self, got, x, y, heading_deg, places=9):
        self.assertAlmostEqual(got[0], x, places=places)
        self.assertAlmostEqual(got[1], y, places=places)
        self.assertAlmostEqual(math.remainder(got[2] - math.radians(heading_deg), 2 * math.pi), 0.0, places=places)

    def test_line(self):
        r = Route(start=(-30, 3), heading_deg=0).line(60)
        self.assertEqual(r.length, 60)
        self.assertPose(r.at(30), 0, 3, 0)
        self.assertPose(r.at(1000), 30, 3, 0)          # 超過終點就停在終點
        self.assertPose(r.at(-5), -30, 3, 0)

    def test_left_and_right_arcs(self):
        r = Route(start=(0, -10), heading_deg=0).arc(10, 90)       # 繞 (0, 0) 逆時針四分之一圈
        self.assertAlmostEqual(r.length, 5 * math.pi)
        self.assertPose(r.at(r.length), 10, 0, 90)
        self.assertAlmostEqual(r.at(1.0)[3], 0.1)                   # 曲率 = 1 / 半徑，左轉為正
        r = Route(start=(0, 0), heading_deg=90).arc(8, -90)         # 朝北右轉 → 朝東
        self.assertPose(r.at(r.length), 8, 8, 0)
        self.assertAlmostEqual(r.at(1.0)[3], -0.125)

    def test_full_circle_returns_to_start(self):
        r = Route(start=(0, -10), heading_deg=0).arc(10, 360)
        self.assertPose(r.at(r.length), 0, -10, 0)
        for s in (1.0, 7.3, 20.0, 50.0):                             # 每一點都離圓心 10 m
            x, y, _, _ = r.at(s)
            self.assertAlmostEqual(math.hypot(x, y), 10.0)

    def test_chained_route_is_continuous(self):
        r = Route(start=(5, 0)).line(15).arc(8, 90).line(10).arc(8, -90)
        eps = 1e-6
        s = 0.0
        for seg in r.segments:
            s += seg[3]
            a, b = r.at(s - eps), r.at(s + eps)
            self.assertLess(math.hypot(a[0] - b[0], a[1] - b[1]), 1e-5)
        self.assertPose(r.at(r.length), 5 + 15 + 8 + 8, 8 + 10 + 8, 0, places=6)

    def test_describe(self):
        d = Route(start=(1, 2), heading_deg=90).line(3).describe()
        self.assertEqual(d["start_enu_m"], (1.0, 2.0))
        self.assertEqual(d["segments"], [{"length_m": 3.0, "curvature_1_m": 0.0}])


if __name__ == "__main__":
    unittest.main()
