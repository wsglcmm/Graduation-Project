"""convert_session 的測試（只用標準函式庫）。

    python -m unittest discover -s sim/tests -v
    python -m pytest sim/tests              # 有 pytest 也可以

設定 SIM_SAMPLE_SESSION=<A 錄的 session 資料夾> 時，會多跑一個用真實資料的往返測試。
"""
import contextlib
import csv
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import convert_session as cs  # noqa: E402
import session_format as sf  # noqa: E402

POSE_KEYS = {"timestamp_ns", "position_ned_m", "attitude_rad", "quaternion_wxyz",
             "altitude_agl_m", "velocity_ned_mps", "gimbal_rad"}   # C 的 PoseMessage（extra="forbid"）


def enu_flu_to_ned_frd(roll, pitch, yaw):
    """C 的 gazebo_pose_to_ned 的封閉解，用來核對反方向。"""
    return roll, -pitch, sf.wrap_pi(math.pi / 2 - yaw)


def angle_close(a, b, tol=1e-9):
    return abs(sf.wrap_pi(a - b)) < tol


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def make_session(folder):
    """兩幀的迷你 session，欄位與 record_dataset.py 相同。"""
    s_cols = ["frame", "stamp", "segment", "roll_deg", "pitch_deg", "yaw_deg", "north_m", "east_m", "down_m",
              "vn_m_s", "ve_m_s", "vd_m_s"]
    write_csv(os.path.join(folder, "states.csv"), s_cols, [
        {"frame": "000001.jpg", "stamp": "132.2680", "segment": 0, "roll_deg": 1.0, "pitch_deg": -2.0,
         "yaw_deg": 95.0, "north_m": 0.06, "east_m": 0.07, "down_m": -7.97, "vn_m_s": 0.1, "ve_m_s": 0.2, "vd_m_s": 0.0},
        {"frame": "000002.jpg", "stamp": "132.3640", "segment": 0, "roll_deg": 0.5, "pitch_deg": 0.3,
         "yaw_deg": 179.9, "north_m": 0.07, "east_m": 0.08, "down_m": -7.98, "vn_m_s": "", "ve_m_s": "", "vd_m_s": ""},
    ])
    t_cols = ["frame", "stamp", "name", "n", "e", "d", "roll_deg", "pitch_deg", "yaw_deg"]
    rows = []
    for frame, stamp in (("000001.jpg", "132.2680"), ("000002.jpg", "132.3640")):
        rows += [
            {"frame": frame, "stamp": stamp, "name": cs.DRONE, "n": 0.05, "e": 0.06, "d": -7.89,
             "roll_deg": 0.2, "pitch_deg": 0.3, "yaw_deg": 90.2},
            {"frame": frame, "stamp": stamp, "name": "car_blue", "n": 0.0, "e": 5.0, "d": 0.0,
             "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 90.0},
            {"frame": frame, "stamp": stamp, "name": "person_1", "n": 4.0, "e": 1.0, "d": -0.00028,
             "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 90.0},
        ]
    write_csv(os.path.join(folder, "truth.csv"), t_cols, rows)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"camera": {"width": 1280, "height": 960}, "route": []}, f)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestMath(unittest.TestCase):
    def test_attitude_matches_closed_form_and_round_trips(self):
        rng = random.Random(0)
        for _ in range(500):
            r, p, y = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-math.pi, math.pi)
            er, ep, ey = cs.ned_frd_to_enu_flu(r, p, y)
            self.assertTrue(angle_close(er, r) and angle_close(ep, -p) and angle_close(ey, math.pi / 2 - y))
            back = enu_flu_to_ned_frd(er, ep, ey)
            for a, b in zip(back, (r, p, y)):
                self.assertTrue(angle_close(a, b))

    def test_heading_examples(self):
        # NED yaw 90°（朝東）= ENU yaw 0；NED yaw 0（朝北）= ENU yaw 90°
        self.assertTrue(angle_close(cs.ned_frd_to_enu_flu(0, 0, math.pi / 2)[2], 0.0))
        self.assertTrue(angle_close(cs.ned_frd_to_enu_flu(0, 0, 0)[2], math.pi / 2))

    def test_quaternion_matches_rotation_matrix(self):
        rng = random.Random(1)
        for _ in range(200):
            r, p, y = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-math.pi, math.pi)
            w, x, qy, z = sf.euler_to_quat(r, p, y)
            self.assertAlmostEqual(w * w + x * x + qy * qy + z * z, 1.0, places=12)
            R = cs.rpy_matrix(r, p, y)
            self.assertAlmostEqual(R[0][1], 2 * (x * qy - w * z), places=12)
            self.assertAlmostEqual(R[2][0], 2 * (x * z - w * qy), places=12)

    def test_stamp_to_ns_is_exact(self):
        self.assertEqual(sf.seconds_str_to_ns("132.2680"), 132_268_000_000)
        self.assertEqual(sf.seconds_str_to_ns("0.0040"), 4_000_000)
        self.assertEqual(sf.seconds_str_to_ns("324.516"), 324_516_000_000)

    def test_target_class(self):
        self.assertEqual(sf.target_class("car_red"), "car")
        self.assertEqual(sf.target_class("person_2"), "person")
        with self.assertRaises(ValueError):
            sf.target_class("tree_1")


class TestConvertSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "20261028_000000")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(self.src)
        make_session(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_outputs(self):
        stats = cs.convert_session(self.src, self.out)
        self.assertEqual((stats["poses"], stats["truth"]), (2, 6))
        self.assertFalse(os.path.exists(os.path.join(self.out, "detections.jsonl")))

        poses = read_jsonl(os.path.join(self.out, "poses.jsonl"))
        self.assertEqual(poses[0]["timestamp_ns"], 132_268_000_000)
        self.assertLessEqual(set(poses[0]), POSE_KEYS)
        self.assertAlmostEqual(poses[0]["attitude_rad"]["yaw"], math.radians(95.0))
        self.assertEqual(poses[0]["velocity_ned_mps"], {"n": 0.1, "e": 0.2, "d": 0.0})
        self.assertNotIn("velocity_ned_mps", poses[1])          # 空欄位就不寫

        truth = read_jsonl(os.path.join(self.out, "truth.jsonl"))
        car = next(r for r in truth if r["name"] == "car_blue")
        self.assertEqual(car["position_enu_m"], {"x": 5.0, "y": 0.0, "z": -0.0})
        self.assertTrue(angle_close(car["rpy_rad"]["yaw"], 0.0))  # NED 90° = ENU 0
        self.assertEqual({r["timestamp_ns"] for r in truth}, {p["timestamp_ns"] for p in poses})

        meta = read_json(os.path.join(self.out, "meta.json"))
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(meta["drone_name"], cs.DRONE)
        self.assertEqual(meta["targets"], {"car_blue": "car", "person_1": "person"})
        self.assertEqual(meta["ned_origin_enu_m"], dict(zip("xyz", cs.DEFAULT_ORIGIN_ENU)))
        self.assertEqual(meta["source_meta"]["camera"]["width"], 1280)

        frames = read_jsonl(os.path.join(self.out, "frames.jsonl"))
        self.assertEqual(frames[1], {"frame_id": 2, "file": "000002.jpg", "timestamp_ns": 132_364_000_000})

    def test_empty_detections(self):
        cs.convert_session(self.src, self.out, empty_detections=True)
        dets = read_jsonl(os.path.join(self.out, "detections.jsonl"))
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0]["detections"], [])
        self.assertEqual((dets[0]["image_width"], dets[0]["image_height"]), (1280, 960))
        meta = read_json(os.path.join(self.out, "meta.json"))
        self.assertTrue(meta["converted_from"]["detections_placeholder"])

    def test_cli_custom_origin(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cs.main([self.src, "--out", self.out, "--origin", "1", "2", "3"])
        meta = read_json(os.path.join(self.out, "meta.json"))
        self.assertEqual(meta["ned_origin_enu_m"], {"x": 1.0, "y": 2.0, "z": 3.0})


@unittest.skipUnless(os.environ.get("SIM_SAMPLE_SESSION"), "沒有設定 SIM_SAMPLE_SESSION")
class TestRealSession(unittest.TestCase):
    def test_round_trip_against_original_ned(self):
        src = os.environ["SIM_SAMPLE_SESSION"]
        with tempfile.TemporaryDirectory() as out:
            cs.convert_session(src, out)
            truth = read_jsonl(os.path.join(out, "truth.jsonl"))
        with open(os.path.join(src, "truth.csv"), newline="", encoding="utf-8") as f:
            orig = list(csv.DictReader(f))
        self.assertEqual(len(truth), len(orig))
        for t, o in zip(truth, orig):
            p, a = t["position_enu_m"], t["rpy_rad"]
            self.assertEqual((p["y"], p["x"], -p["z"]), (float(o["n"]), float(o["e"]), float(o["d"])))
            back = enu_flu_to_ned_frd(a["roll"], a["pitch"], a["yaw"])
            for got, key in zip(back, ("roll_deg", "pitch_deg", "yaw_deg")):
                self.assertTrue(angle_close(got, math.radians(float(o[key])), 1e-9), (o["frame"], o["name"], key))


if __name__ == "__main__":
    unittest.main()
