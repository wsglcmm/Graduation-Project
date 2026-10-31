"""把 record_dataset.py 錄的 CSV session 轉成 analysis/log_format.md 規格（poses.jsonl / truth.jsonl / meta.json）。

    python sim/tools/convert_session.py /mnt/2disk/vicelead/data/20261028_033102
    python sim/tools/convert_session.py <session> --out <輸出資料夾> --origin 0.0157 0.0345 -0.0130
    python sim/tools/convert_session.py <session> --empty-detections   # 讓 C 的 load_session 先跑得動

輸出（預設 <session>_spec/）：
    poses.jsonl        規格 §4 Pose：states.csv 每列一行；角度度 → 弧度，另附 quaternion_wxyz
    truth.jsonl        truth.csv 轉回 Gazebo ENU（n,e,d → x=e, y=n, z=-d），姿態 NED/FRD → ENU/FLU（弧度）
    meta.json          規格必填欄位；A 原本的 meta 放在 source_meta
    frames.jsonl       畫面檔名 → timestamp_ns，給 B 對 frames/ 跑偵測時對時用
    detections.jsonl   只有加 --empty-detections 才寫：每幀 detections = []（佔位，不是 B 的結果）

限制（會寫進 meta.converted_from）：姿態與真值都已內插到畫面時間（約 10 Hz），不是原始串流。
新錄的 session 由 record_dataset.py 直接輸出規格格式，不需要再轉換。
只用標準函式庫，本機可以直接拿 a_data/ 的範例測試。
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from session_format import (SCHEMA_VERSION, JsonlWriter, meta_dict, pose_msg,  # noqa: E402
                            seconds_str_to_ns, truth_msg, wrap_pi)

DRONE = "x500_mono_cam_down_0"
# 2026-10-31 在新啟動的模擬中、起飛前讀到的模型 pose（Gazebo ENU）；不是 10/28 錄製那一次量的
DEFAULT_ORIGIN_ENU = (0.0157, 0.0345, -0.0130)
DEFAULT_ORIGIN_SOURCE = "2026-10-31 新啟動模擬、起飛前的 Gazebo 模型 pose（不是錄製當次量的）"

# ENU 向量 → NED 向量（自己的反矩陣就是自己）；機體 FRD ↔ FLU
ENU_NED = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
FRD_FLU = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


# ---------- 姿態轉換 ----------
def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def rpy_matrix(roll, pitch, yaw):
    """Rz(yaw)·Ry(pitch)·Rx(roll)，與 SDF 及 C 的 coord_transform 同一個順序。"""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def matrix_to_rpy(R):
    pitch = -math.asin(max(-1.0, min(1.0, R[2][0])))
    return math.atan2(R[2][1], R[2][2]), pitch, math.atan2(R[1][0], R[0][0])


def ned_frd_to_enu_flu(roll, pitch, yaw):
    """NED/FRD 姿態 → Gazebo ENU/FLU 姿態（弧度）。封閉解：roll 不變、pitch 反號、yaw_enu = π/2 − yaw_ned。"""
    r, p, y = matrix_to_rpy(_matmul(_matmul(ENU_NED, rpy_matrix(roll, pitch, yaw)), FRD_FLU))
    return r, p, wrap_pi(y)


# ---------- 轉換 ----------
def _num(row, key):
    v = row.get(key)
    return None if v in (None, "") else float(v)


def _frame_id(name):
    return int(os.path.splitext(name)[0])


def convert_states(rows):
    """states.csv 的列 → (poses, frames)；缺欄位的列略過。"""
    poses, frames = [], []
    for r in rows:
        vals = [_num(r, k) for k in ("north_m", "east_m", "down_m", "roll_deg", "pitch_deg", "yaw_deg")]
        if None in vals:
            continue
        t = seconds_str_to_ns(r["stamp"])
        poses.append(pose_msg(t, *vals, vel_ned=[_num(r, k) for k in ("vn_m_s", "ve_m_s", "vd_m_s")]))
        frames.append({"frame_id": _frame_id(r["frame"]), "file": r["frame"], "timestamp_ns": t})
    return poses, frames


def convert_truth(rows):
    """truth.csv 的列（NED、度）→ truth.jsonl 的列（Gazebo ENU、弧度）。"""
    out = []
    for r in rows:
        vals = [_num(r, k) for k in ("n", "e", "d", "roll_deg", "pitch_deg", "yaw_deg")]
        if None in vals:
            continue
        n, e, d, *rpy_deg = vals
        rpy = ned_frd_to_enu_flu(*(math.radians(v) for v in rpy_deg))
        out.append(truth_msg(seconds_str_to_ns(r["stamp"]), r["name"], e, n, 0.0 - d, rpy))   # 0.0 - d：避免寫出 -0.0
    return out


def _check_increasing(rows, key=lambda r: None):
    last = {}
    for r in rows:
        k = key(r)
        if k in last and r["timestamp_ns"] <= last[k]:
            raise ValueError(f"{k or 'poses'} 的時間戳沒有嚴格遞增（{r['timestamp_ns']}）")
        last[k] = r["timestamp_ns"]


def build_meta(src_meta, truth, origin, origin_source, ground_z, source_name, empty_detections):
    names = sorted({r["name"] for r in truth})
    if DRONE not in names:
        raise ValueError(f"truth.csv 裡沒有無人機 {DRONE}")
    return meta_dict(origin, ground_z, DRONE, [n for n in names if n != DRONE],
                     converted_from={
                         "tool": "sim/tools/convert_session.py",
                         "source_session": source_name,
                         "source_format": "record_dataset.py CSV（states.csv / truth.csv）",
                         "sampling": "姿態與真值都已在畫面時間內插（約 10 Hz），不是原始 odometry / pose 串流",
                         "ned_origin_source": origin_source,
                         "detections_placeholder": empty_detections,
                     },
                     source_meta=src_meta)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_jsonl(path, rows):
    w = JsonlWriter(path)
    for r in rows:
        w.write(r)
    w.close()


def convert_session(src, out, origin=DEFAULT_ORIGIN_ENU, origin_source=DEFAULT_ORIGIN_SOURCE,
                    ground_z=0.0, empty_detections=False):
    """轉換一個 session，回傳統計 dict。"""
    states = _read_csv(os.path.join(src, "states.csv"))
    truth_rows = _read_csv(os.path.join(src, "truth.csv"))
    meta_path = os.path.join(src, "meta.json")
    src_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            src_meta = json.load(f)

    poses, frames = convert_states(states)
    truth = convert_truth(truth_rows)
    _check_increasing(poses)
    _check_increasing(truth, key=lambda r: r["name"])
    meta = build_meta(src_meta, truth, origin, origin_source, ground_z,
                      os.path.basename(os.path.normpath(src)), empty_detections)

    os.makedirs(out, exist_ok=True)
    _write_jsonl(os.path.join(out, "poses.jsonl"), poses)
    _write_jsonl(os.path.join(out, "truth.jsonl"), truth)
    _write_jsonl(os.path.join(out, "frames.jsonl"), frames)
    if empty_detections:
        w, h = src_meta.get("camera", {}).get("width", 1280), src_meta.get("camera", {}).get("height", 960)
        _write_jsonl(os.path.join(out, "detections.jsonl"),
                     [{"schema_version": SCHEMA_VERSION, "frame_id": fr["frame_id"], "timestamp_ns": fr["timestamp_ns"],
                       "image_width": w, "image_height": h, "detections": []} for fr in frames])
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"poses": len(poses), "skipped_states": len(states) - len(poses),
            "truth": len(truth), "skipped_truth": len(truth_rows) - len(truth), "targets": meta["targets"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="把 record_dataset.py 的 CSV session 轉成規格格式")
    ap.add_argument("session", help="含 states.csv、truth.csv、meta.json 的資料夾")
    ap.add_argument("--out", help="輸出資料夾（預設 <session>_spec）")
    ap.add_argument("--origin", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help=f"PX4 NED 原點在 Gazebo ENU 的位置（預設 {DEFAULT_ORIGIN_ENU}）")
    ap.add_argument("--ground-z", type=float, default=0.0, help="地面在 Gazebo 的 z（預設 0）")
    ap.add_argument("--empty-detections", action="store_true",
                    help="寫出每幀 detections = [] 的佔位 detections.jsonl（不是 B 的結果）")
    args = ap.parse_args(argv)

    out = args.out or os.path.normpath(args.session) + "_spec"
    origin, source = DEFAULT_ORIGIN_ENU, DEFAULT_ORIGIN_SOURCE
    if args.origin:
        origin, source = tuple(args.origin), "命令列 --origin 指定"
    stats = convert_session(args.session, out, origin, source, args.ground_z, args.empty_detections)
    print(f"poses {stats['poses']} 筆（略過 {stats['skipped_states']}）、truth {stats['truth']} 筆"
          f"（略過 {stats['skipped_truth']}）、目標 {stats['targets']}")
    if args.empty_detections:
        print("注意：detections.jsonl 是空的佔位檔，只能用來跑 calibrate_error_model 的姿態誤差")
    print(f"完成 → {out}")


if __name__ == "__main__":
    main()
