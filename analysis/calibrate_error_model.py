"""從 session 標定誤差模型（config/error_model.yaml 的各個 σ）。

兩部分：
    1. 姿態 / 高度 / 水平位置：A 的估計值（poses.jsonl）− Gazebo 真值（truth.jsonl），逐筆比對。
    2. 像素：B 的 bbox 參考點 − 「真值目標用真值姿態投影」的像素。用真值姿態，是為了只量到 B 自己的誤差，
       不把 A 的姿態誤差重複算進來。

每一項都分成「平均（偏差）」與「標準差」：
    - error_model.yaml 要填的是標準差（零平均的隨機誤差）。
    - 偏差應該從源頭修正（原點設錯、座標慣例、參考點選擇），不該塞進 σ。偏差過大時本工具會警告。

執行：python -m analysis.calibrate_error_model <session> [--reference center_half_height]
輸出：<session>/output/error_model_suggested.yaml、calibration.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analysis.session import angle_diff, load_session
from coord_transform import ned_to_pixel, rotations
from coord_transform.config import load_camera_config, load_pipeline_config
from coord_transform.messages import DetectionFrame, PoseMessage
from coord_transform.reference_point import CLASS_HEIGHT_M, reference_point

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_PX = 80.0


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return {"n": int(x.size), "mean": float("nan"), "std": float("nan")}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1))}


def pose_errors(session):
    """A 的姿態估計 − 真值。角度單位 rad，長度 m。"""
    err = {k: [] for k in ("roll", "pitch", "yaw", "n", "e", "d")}
    for raw in session.poses:
        m = PoseMessage.model_validate(raw)
        truth = session.drone_pose_ned(m.timestamp_ns)
        if truth is None:
            continue
        roll, pitch, yaw = rotations.quat_to_euler(m.quaternion())
        err["roll"].append(angle_diff(roll, truth.roll))
        err["pitch"].append(angle_diff(pitch, truth.pitch))
        err["yaw"].append(angle_diff(yaw, truth.yaw))
        err["n"].append(m.position_ned_m.n - truth.n)
        err["e"].append(m.position_ned_m.e - truth.e)
        err["d"].append(m.position_ned_m.d - truth.d)
    return {k: _stats(v) for k, v in err.items()}


def pixel_errors(session, cam, reference, frame_time_offset_ns=0, gate_px=DEFAULT_GATE_PX):
    """B 的參考點像素 − 真值投影像素，依類別分組。"""
    res = {}
    for raw in session.frames:
        f = DetectionFrame.model_validate(raw)
        t = f.timestamp_ns + frame_time_offset_ns
        pose = session.drone_pose_ned(t)
        if pose is None:
            continue
        expected = {}
        for name, cls in session.targets.items():
            p = session.target_ned(name, t)
            if p is None:
                continue
            if reference == "center_half_height":
                p = p + np.array([0.0, 0.0, -CLASS_HEIGHT_M[cls] / 2])
            uv = ned_to_pixel(tuple(p), cam.intrinsics, cam.mount, pose)
            if uv is not None:
                expected[name] = (cls, np.array(uv))
        used = set()
        for det in f.detections:
            b = det.bbox
            u, v, _ = reference_point((b.x1, b.y1, b.x2, b.y2), det.cls, reference)
            cands = [(float(np.linalg.norm(np.array([u, v]) - uv)), name, uv) for name, (cls, uv) in expected.items() if cls == det.cls and name not in used]
            if not cands:
                continue
            dist, name, uv = min(cands)
            if dist > gate_px:
                continue
            used.add(name)
            res.setdefault(det.cls, []).append((u - uv[0], v - uv[1]))
    out = {}
    for cls, r in res.items():
        r = np.array(r)
        out[cls] = {"u": _stats(r[:, 0]), "v": _stats(r[:, 1])}
    return out


def suggest(pose_err, pix_err):
    """組出建議的 error_model 值與警告。"""
    warnings = []
    rp = math.sqrt((pose_err["roll"]["std"] ** 2 + pose_err["pitch"]["std"] ** 2) / 2)
    suggestion = {
        "roll_pitch_sigma_deg": math.degrees(rp),
        "yaw_sigma_deg": math.degrees(pose_err["yaw"]["std"]),
        "altitude_sigma_m": pose_err["d"]["std"],
        "horizontal_position_sigma_m": math.sqrt((pose_err["n"]["std"] ** 2 + pose_err["e"]["std"] ** 2) / 2),
    }
    if pix_err:
        all_var = [s[axis]["std"] ** 2 for s in pix_err.values() for axis in ("u", "v") if s[axis]["n"] >= 2]
        suggestion["pixel_sigma_px"] = math.sqrt(sum(all_var) / len(all_var))
    else:
        warnings.append("沒有任何偵測成功配對到真值，像素誤差無法標定（檢查 meta.json 的 targets 與時間戳）")

    for k, label, limit, unit in (
        ("roll", "roll", math.radians(2), "°"),
        ("pitch", "pitch", math.radians(2), "°"),
        ("yaw", "yaw", math.radians(5), "°"),
    ):
        b = pose_err[k]["mean"]
        if abs(b) > limit:
            warnings.append(f"{label} 平均偏差 {math.degrees(b):+.1f}{unit}：可能是座標慣例錯誤（例如 ENU/NED、度/弧度混用）")
    for k in ("n", "e", "d"):
        b, s = pose_err[k]["mean"], pose_err[k]["std"]
        if abs(b) > max(0.5, 2 * s):
            warnings.append(f"位置 {k} 平均偏差 {b:+.2f} m：檢查 meta.json 的 ned_origin_enu_m 是否為 PX4 的起飛點")
    for cls, s in pix_err.items():
        for axis in ("u", "v"):
            b, sd = s[axis]["mean"], s[axis]["std"]
            if abs(b) > max(3.0, sd):
                warnings.append(f"{cls} 的像素 {axis} 平均偏差 {b:+.1f} px：參考點方法可能不適合（見 analysis/README.md 實驗 5）")
    return suggestion, warnings


def to_yaml(s, session_path, reference):
    lines = [
        "# 由 analysis/calibrate_error_model.py 自動產生，確認後再複製到 config/error_model.yaml",
        f"# 來源 session：{session_path}；像素參考點：{reference}",
        "",
    ]
    order = ("pixel_sigma_px", "roll_pitch_sigma_deg", "yaw_sigma_deg", "altitude_sigma_m", "horizontal_position_sigma_m")
    lines += [f"{k}: {s[k]:.3f}" for k in order if k in s]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="用 Gazebo 真值標定誤差模型")
    ap.add_argument("session")
    ap.add_argument("--camera", default=ROOT / "config" / "camera.yaml")
    ap.add_argument("--pipeline", default=ROOT / "config" / "pipeline.yaml")
    ap.add_argument("--reference", help="像素誤差用哪種參考點（預設讀 pipeline.yaml）")
    ap.add_argument("--time-offset-ms", type=float, help="影像時間偏移（預設讀 pipeline.yaml）")
    ap.add_argument("--out", help="輸出資料夾（預設 <session>/output）")
    args = ap.parse_args(argv)

    session = load_session(args.session)
    cam = load_camera_config(args.camera)
    pipe = load_pipeline_config(args.pipeline)
    reference = args.reference or pipe.reference_point
    offset_ns = int(round(args.time_offset_ms * 1e6)) if args.time_offset_ms is not None else pipe.frame_time_offset_ns

    pose_err = pose_errors(session)
    pix_err = pixel_errors(session, cam, reference, offset_ns)
    suggestion, warnings = suggest(pose_err, pix_err)

    print(f"姿態比對 {pose_err['roll']['n']} 筆：")
    for k in ("roll", "pitch", "yaw"):
        print(f"  {k:5s} 偏差 {math.degrees(pose_err[k]['mean']):+6.2f}°  標準差 {math.degrees(pose_err[k]['std']):5.2f}°")
    for k in ("n", "e", "d"):
        print(f"  {k:5s} 偏差 {pose_err[k]['mean']:+6.2f} m  標準差 {pose_err[k]['std']:5.2f} m")
    for cls, s in pix_err.items():
        print(
            f"像素（{cls}，{s['u']['n']} 筆，{reference}）：u 偏差 {s['u']['mean']:+.2f}、標準差 {s['u']['std']:.2f} px；"
            f"v 偏差 {s['v']['mean']:+.2f}、標準差 {s['v']['std']:.2f} px"
        )
    for w in warnings:
        print(f"⚠️ {w}")

    out_dir = Path(args.out) if args.out else Path(args.session) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = to_yaml(suggestion, session.path, reference)
    (out_dir / "error_model_suggested.yaml").write_text(text, encoding="utf-8")
    (out_dir / "calibration.json").write_text(
        json.dumps({"pose": pose_err, "pixel": pix_err, "suggestion": suggestion, "warnings": warnings, "reference": reference}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n建議值（已寫入 error_model_suggested.yaml）：\n" + text)
    return suggestion, warnings


if __name__ == "__main__":
    main()
