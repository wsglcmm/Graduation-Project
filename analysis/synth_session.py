"""產生合成的 session（格式與 A、B 要錄的完全相同，見 analysis/log_format.md）。

用途：
    1. 在 A、B 的真實紀錄到手前，先把重播（replay）與標定（calibrate）工具做完並驗證。
    2. 所有誤差都是「已知注入」的，所以工具算出來的數字可以和答案對照（測試就是這樣寫的）。
    3. 給 A、B 當格式範本。

情境：車沿著 S 形路線以約 5 m/s 前進，無人機在 30 m 高度落後約 1 秒跟車，機身有 roll/pitch 擺動
（模擬轉彎與加減速），另有一個靜止的行人。

執行：python -m analysis.synth_session analysis/output/synth_demo
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from analysis.error_study import box_corners, project_bbox
from analysis.session import write_jsonl
from coord_transform import rotations
from coord_transform.config import CameraConfig, load_camera_config
from coord_transform.frames import ned_pose_to_gazebo
from coord_transform.pixel_to_ned import Pose

ROOT = Path(__file__).resolve().parents[1]
T_START_NS = 10_000_000_000  # 模擬時間從 10 s 開始（不是 0，避免把 0 當成「沒有時間戳」也能過）


@dataclass(frozen=True)
class SynthConfig:
    duration_s: float = 20.0
    image_hz: float = 30.0
    pose_hz: float = 20.0  # A 實測
    truth_hz: float = 50.0
    altitude_m: float = 30.0
    follow_lag_s: float = 1.0
    # ---- 注入的誤差（1σ，白雜訊）----
    pixel_edge_sigma_px: float = 1.5  # 每個 bbox 邊緣各自的抖動
    roll_pitch_sigma_deg: float = 0.5
    yaw_sigma_deg: float = 2.0
    altitude_sigma_m: float = 0.3
    horizontal_position_sigma_m: float = 0.4
    # ---- 注入的系統偏差 ----
    camera_time_lag_ms: float = 24.0  # 影像時間戳比真實擷取晚多少
    pose_timestamp_quantum_ms: float = 1.0  # att_timestamp_us 精度 1 ms
    seed: int = 0


ORIGIN_ENU = (2.0, -3.0, 0.0)  # PX4 NED 原點（起飛點）在 Gazebo 世界的位置
CAR = ("car_1", "car", (4.5, 1.8, 1.5))
PERSON = ("person_1", "person", (0.5, 0.5, 1.7), (55.0, 12.0))  # 靜止於 NED (55, 12)
TRACK_IDS = {"car_1": 7, "person_1": 12}


def car_state(t: float):
    """車的 NED 位置與車頭方向。S 形路線：n = 5t、e = 12·sin(0.25t)。"""
    n, e = 5.0 * t, 12.0 * math.sin(0.25 * t)
    heading = math.atan2(3.0 * math.cos(0.25 * t), 5.0)
    return n, e, heading


def drone_truth(t: float, cfg: SynthConfig) -> Pose:
    tl = t - cfg.follow_lag_s
    n, e, heading = car_state(tl)
    return Pose(
        n=n,
        e=e,
        d=-(cfg.altitude_m + 0.5 * math.sin(0.4 * t)),
        roll=0.12 * math.sin(2 * math.pi * t / 3.0),
        pitch=-0.05 + 0.08 * math.sin(2 * math.pi * t / 4.3),
        yaw=heading,
    )


def _enu(name, t_ns, n, e, d, rpy=None):
    x, y, z = e + ORIGIN_ENU[0], n + ORIGIN_ENU[1], -d + ORIGIN_ENU[2]
    row = {"timestamp_ns": t_ns, "name": name, "position_enu_m": {"x": x, "y": y, "z": z}}
    if rpy is not None:
        row["rpy_rad"] = dict(zip(("roll", "pitch", "yaw"), rpy))
    return row


def generate(out_dir, cam: CameraConfig, cfg: SynthConfig = SynthConfig()) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    to_ns = lambda s: T_START_NS + int(round(s * 1e9))  # noqa: E731

    # ---- truth.jsonl（Gazebo ENU）----
    truth = []
    for k in range(int(cfg.duration_s * cfg.truth_hz) + 1):
        t = k / cfg.truth_hz
        p = drone_truth(t, cfg)
        _, _, _, r, pi, yw = ned_pose_to_gazebo(p)  # 姿態轉成 Gazebo 的 ENU/FLU 慣例
        truth.append(_enu("x500_0", to_ns(t), p.n, p.e, p.d, (r, pi, yw)))
        cn, ce, ch = car_state(t)
        truth.append(_enu(CAR[0], to_ns(t), cn, ce, 0.0, (0.0, 0.0, math.pi / 2 - ch)))
        truth.append(_enu(PERSON[0], to_ns(t), *PERSON[3], 0.0))

    # ---- poses.jsonl（A：估計值 = 真值 + 雜訊）----
    poses = []
    q = cfg.pose_timestamp_quantum_ms * 1e-3
    for k in range(int(cfg.duration_s * cfg.pose_hz) + 1):
        t = round((k / cfg.pose_hz) / q) * q
        p = drone_truth(t, cfg)
        rp, yw = math.radians(cfg.roll_pitch_sigma_deg), math.radians(cfg.yaw_sigma_deg)
        dn, de = rng.normal(0, cfg.horizontal_position_sigma_m, 2)
        roll, pitch = p.roll + rng.normal(0, rp), p.pitch + rng.normal(0, rp)
        yaw = (p.yaw + rng.normal(0, yw) + math.pi) % (2 * math.pi) - math.pi
        poses.append(
            {
                "timestamp_ns": to_ns(t),
                "position_ned_m": {"n": p.n + dn, "e": p.e + de, "d": p.d + rng.normal(0, cfg.altitude_sigma_m)},
                "attitude_rad": {"roll": roll, "pitch": pitch, "yaw": yaw},
                "quaternion_wxyz": list(rotations.euler_to_quat(roll, pitch, yaw)),
            }
        )

    # ---- detections.jsonl（B：用真實姿態投影 3D 外框 + bbox 雜訊）----
    intr = cam.intrinsics
    frames = []
    for i in range(int(cfg.duration_s * cfg.image_hz) + 1):
        t = i / cfg.image_hz
        pose = drone_truth(t, cfg)
        cn, ce, ch = car_state(t)
        objects = [(CAR[0], CAR[1], CAR[2], (cn, ce), ch), (PERSON[0], PERSON[1], PERSON[2], PERSON[3], 0.0)]
        dets = []
        for name, cls, size, center, heading in objects:
            bbox = project_bbox(box_corners(center, size, heading), intr, cam.mount, pose)
            if bbox is None or bbox[0] < 0 or bbox[1] < 0 or bbox[2] > intr.width or bbox[3] > intr.height:
                continue  # 不在畫面內（或被切到）就不輸出
            x1, y1, x2, y2 = (np.array(bbox) + rng.normal(0, cfg.pixel_edge_sigma_px, 4)).tolist()
            dets.append({"track_id": TRACK_IDS[name], "class": cls, "confidence": 0.9, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
        frames.append(
            {
                "schema_version": "0.1.0",
                "frame_id": i,
                "timestamp_ns": to_ns(t) + int(round(cfg.camera_time_lag_ms * 1e6)),
                "image_width": intr.width,
                "image_height": intr.height,
                "detections": dets,
            }
        )

    meta = {
        "format_version": 1,
        "ned_origin_enu_m": dict(zip("xyz", ORIGIN_ENU)),
        "ground_z_enu_m": ORIGIN_ENU[2],
        "drone_name": "x500_0",
        "targets": {CAR[0]: CAR[1], PERSON[0]: PERSON[1]},
        "synthetic": asdict(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out / "truth.jsonl", truth)
    write_jsonl(out / "poses.jsonl", poses)
    write_jsonl(out / "detections.jsonl", frames)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="產生合成 session（已知誤差），供重播與標定工具驗證")
    ap.add_argument("out", help="輸出資料夾")
    ap.add_argument("--camera", default=ROOT / "config" / "camera.yaml")
    ap.add_argument("--duration", type=float, default=SynthConfig.duration_s)
    ap.add_argument("--camera-lag-ms", type=float, default=SynthConfig.camera_time_lag_ms)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = SynthConfig(duration_s=args.duration, camera_time_lag_ms=args.camera_lag_ms, seed=args.seed)
    out = generate(args.out, load_camera_config(args.camera), cfg)
    print(f"已產生：{out}（注入的誤差見 meta.json 的 synthetic 欄位）")


if __name__ == "__main__":
    main()
