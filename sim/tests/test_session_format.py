"""session_format 的測試：python -m unittest discover -s sim/tests -v"""
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import session_format as sf  # noqa: E402

POSE_KEYS = {"timestamp_ns", "position_ned_m", "attitude_rad", "quaternion_wxyz",
             "altitude_agl_m", "velocity_ned_mps", "gimbal_rad"}   # C 的 PoseMessage（extra="forbid"）


class TestSessionFormat(unittest.TestCase):
    def test_quat_round_trip(self):
        rng = random.Random(2)
        for _ in range(500):
            rpy = (rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-math.pi, math.pi))
            for a, b in zip(sf.quat_to_rpy(*sf.euler_to_quat(*rpy)), rpy):
                self.assertLess(abs(sf.wrap_pi(a - b)), 1e-9)

    def test_pose_msg(self):
        m = sf.pose_msg(324_516_000_000, 1.0, 2.0, -30.0, 0.0, 0.0, 270.0, vel_ned=[0.1, 0.2, 0.3])
        self.assertLessEqual(set(m), POSE_KEYS)
        self.assertIsInstance(m["timestamp_ns"], int)
        self.assertAlmostEqual(m["attitude_rad"]["yaw"], -math.pi / 2)    # 包到 [-π, π)
        self.assertEqual(m["velocity_ned_mps"], {"n": 0.1, "e": 0.2, "d": 0.3})
        self.assertNotIn("velocity_ned_mps", sf.pose_msg(0, 0, 0, 0, 0, 0, 0, vel_ned=[None, 0, 0]))

    def test_truth_msg(self):
        self.assertNotIn("rpy_rad", sf.truth_msg(1, "car_red", 1, 2, 0))
        m = sf.truth_msg(1, "x500_0", 1, 2, 3, (0.1, 0.2, 0.3))
        self.assertEqual(m["rpy_rad"], {"roll": 0.1, "pitch": 0.2, "yaw": 0.3})

    def test_meta_dict(self):
        m = sf.meta_dict((0.0, 0.0, -0.013), 0.0, "x500_0", ["car_blue", "person_1"], note="x")
        self.assertEqual(m["format_version"], 1)
        self.assertEqual(m["targets"], {"car_blue": "car", "person_1": "person"})
        self.assertEqual(m["note"], "x")
        with self.assertRaises(ValueError):
            sf.meta_dict((0, 0, 0), 0.0, "x500_0", ["tree_1"])

    def test_jsonl_writer(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jsonl")
            w = sf.JsonlWriter(path)
            w.write({"a": 1})
            w.write({"b": "中文"})
            with open(path, encoding="utf-8") as f:      # 每筆都 flush：還沒 close 就讀得到
                self.assertEqual([json.loads(line) for line in f], [{"a": 1}, {"b": "中文"}])
            w.close()
            self.assertEqual(w.count, 2)


if __name__ == "__main__":
    unittest.main()
