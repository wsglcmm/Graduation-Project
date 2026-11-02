"""從 session 標定誤差模型（config/error_model.yaml 的各個 σ）。

兩部分：
    1. 姿態 / 高度 / 水平位置：A 的估計值（poses.jsonl）− Gazebo 真值（truth.jsonl），逐筆比對。
    2. 像素：B 的 bbox 參考點 − 「真值目標用真值姿態投影」的像素。用真值姿態，是為了只量到 B 自己的誤差，
       不把 A 的姿態誤差重複算進來。

每一項都分成「平均（偏差）」與「標準差」：
    - error_model.yaml 要填的是標準差（零平均的隨機誤差）。
    - 偏差應該從源頭修正（原點設錯、座標慣例、參考點選擇），不該塞進 σ。偏差過大時本工具會警告。

另外兩件事：
    - 系統性 yaw 偏差（例如 PX4 EKF 航向還沒對準）：σ 只看標準差會低估，另外提供 RMS（偏差與抖動合計）。
    - 位置的時間偏移：--estimate-position-lag 用移動中的水平位置誤差網格搜尋。
      2026-11-01 A 的資料：roll/pitch 與 Gazebo 同步（最佳 0 ms），位置卻約 −90 ms，
      代表同一筆 Pose 的位置和姿態不是同一時刻（或位置有和速度成正比的偏差），所以只回報、不自動套用；
      --position-lag-ms 只修正位置比對，不動姿態。
      原因已由 A 查明：EKF2_GPS_DELAY 預設 110 ms，但 Gazebo GPS 沒有延遲。不需要真值的補償式是
      「位置 − 速度 × τ」（--position-lead-s，11/01 以前的資料最佳 τ ≈ 0.095 s；A 已把參數設 0）。

沒有 detections.jsonl（B 還沒跑偵測）時只做姿態標定（pose-only），像素 σ 沿用原設定。

執行：python -m analysis.calibrate_error_model <session> [<session> ...] [--estimate-position-lag] [--reference center_half_height]
輸出：<第一個 session>/output/（或 --out）error_model_suggested.yaml、calibration.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analysis.session import angle_diff, compensate_position_lead, load_session
from coord_transform import ned_to_pixel, rotations
from coord_transform.config import load_camera_config, load_pipeline_config
from coord_transform.messages import DetectionFrame, PoseMessage
from coord_transform.reference_point import CLASS_HEIGHT_M, reference_point

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_PX = 80.0


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return {"n": int(x.size), "mean": float("nan"), "std": float("nan"), "rms": float("nan")}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)), "rms": float(np.sqrt(np.mean(x ** 2)))}


POSE_KEYS = ("roll", "pitch", "yaw", "n", "e", "d")


def pose_error_samples(session, position_lag_ns: int = 0) -> dict:
    """A 的姿態估計 − 真值的逐筆樣本。角度單位 rad，長度 m。

    position_lag_ns：只用於位置。Pose 在 t 的位置和真值在 t − position_lag_ns 比對
    （< 0 代表位置領先時間戳）；姿態一律和 t 時刻的真值比對。
    """
    err = {k: [] for k in POSE_KEYS}
    for raw in session.poses:
        m = PoseMessage.model_validate(raw)
        truth = session.drone_pose_ned(m.timestamp_ns)
        truth_pos = truth if position_lag_ns == 0 else session.drone_pose_ned(m.timestamp_ns - position_lag_ns)
        if truth is None or truth_pos is None:
            continue
        roll, pitch, yaw = rotations.quat_to_euler(m.quaternion())
        err["roll"].append(angle_diff(roll, truth.roll))
        err["pitch"].append(angle_diff(pitch, truth.pitch))
        err["yaw"].append(angle_diff(yaw, truth.yaw))
        err["n"].append(m.position_ned_m.n - truth_pos.n)
        err["e"].append(m.position_ned_m.e - truth_pos.e)
        err["d"].append(m.position_ned_m.d - truth_pos.d)
    return err


def pose_errors(session, position_lag_ns: int = 0):
    """A 的姿態估計 − 真值的統計（平均 = 偏差、標準差、RMS）。"""
    return {k: _stats(v) for k, v in pose_error_samples(session, position_lag_ns).items()}


def _moving_position_rmse(session, lag_ns: int, min_speed: float, stride: int) -> tuple:
    """移動中（水平速度 > min_speed）的水平位置誤差 RMS。速度取自 poses 的 velocity_ned_mps。"""
    sq = []
    for raw in session.poses[::stride]:
        v = raw.get("velocity_ned_mps")
        if not v or math.hypot(v["n"], v["e"]) <= min_speed:
            continue
        truth = session.drone_pose_ned(raw["timestamp_ns"] - lag_ns)
        if truth is None:
            continue
        p = raw["position_ned_m"]
        sq.append((p["n"] - truth.n) ** 2 + (p["e"] - truth.e) ** 2)
    return (math.sqrt(sum(sq) / len(sq)) if sq else float("nan")), len(sq)


def estimate_position_lead(session, max_s: float = 0.2, step_s: float = 0.005, min_speed: float = 0.5, stride: int = 2,
                           min_samples: int = 200) -> dict:
    """估計「位置 − 速度 × τ」補償的最佳 τ（秒），用移動中的水平位置誤差 RMS 網格搜尋。

    和 estimate_position_lag（把真值時間平移）不同，這個補償式不需要真值，可以直接用在重播與即時流程。
    """
    samples = []
    for raw in session.poses[::stride]:
        v = raw.get("velocity_ned_mps")
        if not v or math.hypot(v["n"], v["e"]) <= min_speed:
            continue
        truth = session.drone_pose_ned(raw["timestamp_ns"])
        if truth is not None:
            p = raw["position_ned_m"]
            samples.append((p["n"] - truth.n, p["e"] - truth.e, v["n"], v["e"]))
    if len(samples) < min_samples:
        return {"lead_s": 0.0, "reliable": False, "n": len(samples), "note": "移動中的樣本太少（需要跟車或直線飛行的紀錄）"}
    a = np.array(samples)
    def rmse(tau):
        return float(np.sqrt(np.mean((a[:, 0] - a[:, 2] * tau) ** 2 + (a[:, 1] - a[:, 3] * tau) ** 2)))
    taus = np.arange(0.0, max_s + 1e-9, step_s)
    errs = [rmse(t) for t in taus]
    k = int(np.argmin(errs))
    return {"lead_s": float(taus[k]), "reliable": bool(errs[k] < 0.8 * errs[0]), "n": len(samples),
            "rmse_zero_m": errs[0], "rmse_best_m": errs[k]}


def estimate_position_lag(session, max_ms: float = 300.0, min_speed: float = 0.5, stride: int = 2, min_samples: int = 200) -> dict:
    """網格搜尋位置相對時間戳的固定偏移（先 10 ms 粗搜，再 2 ms 細搜）。懸停資料沒有移動，會回報不可靠。"""
    def rmse(ms):
        return _moving_position_rmse(session, int(round(ms * 1e6)), min_speed, stride)

    zero, n = rmse(0.0)
    if n < min_samples:
        return {"lag_ms": 0.0, "reliable": False, "n": n, "rmse_zero_m": zero, "rmse_best_m": zero,
                "note": "移動中的樣本太少（需要跟車或直線飛行的紀錄）"}
    coarse = min((rmse(ms)[0], ms) for ms in np.arange(-max_ms, max_ms + 1e-9, 10.0))
    best, ms = min((rmse(m)[0], m) for m in np.arange(coarse[1] - 10, coarse[1] + 10 + 1e-9, 2.0))
    reliable = best < 0.8 * zero
    return {"lag_ms": float(ms), "reliable": bool(reliable), "n": n, "rmse_zero_m": zero, "rmse_best_m": best}


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
    yb, ys = pose_err["yaw"]["mean"], pose_err["yaw"]["std"]
    if abs(yb) > max(math.radians(1.0), 2 * ys):
        suggestion["yaw_sigma_deg_rms"] = math.degrees(pose_err["yaw"]["rms"])
        warnings.append(
            f"yaw 有系統性偏差 {math.degrees(yb):+.1f}°（標準差只有 {math.degrees(ys):.2f}°）：多半是 PX4 的磁偏角和模擬磁場不一致"
            "（EKF2_DECL_TYPE / EKF2_MAG_DECL），要水平加速後 EKF 才會修正。"
            f"偏差沒修正前，yaw_sigma_deg 建議用 RMS {math.degrees(pose_err['yaw']['rms']):.2f}°，否則 uncertainty_m 會低估"
        )
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
    if "yaw_sigma_deg_rms" in s:
        lines.append(f"# yaw 有系統性偏差：偏差修正前建議 yaw_sigma_deg 改用 RMS = {s['yaw_sigma_deg_rms']:.3f}")
    return "\n".join(lines) + "\n"


def _fmt_pose(pose_err) -> list:
    lines = []
    for k in ("roll", "pitch", "yaw"):
        e = pose_err[k]
        lines.append(f"  {k:5s} 偏差 {math.degrees(e['mean']):+6.2f}°  標準差 {math.degrees(e['std']):5.2f}°  RMS {math.degrees(e['rms']):5.2f}°")
    for k in ("n", "e", "d"):
        e = pose_err[k]
        lines.append(f"  {k:5s} 偏差 {e['mean']:+6.3f} m  標準差 {e['std']:5.3f} m")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="用 Gazebo 真值標定誤差模型（可一次給多個 session，結果合併計算）")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--camera", default=ROOT / "config" / "camera.yaml")
    ap.add_argument("--pipeline", default=ROOT / "config" / "pipeline.yaml")
    ap.add_argument("--reference", help="像素誤差用哪種參考點（預設讀 pipeline.yaml）")
    ap.add_argument("--time-offset-ms", type=float, help="影像時間偏移（預設讀 pipeline.yaml）")
    ap.add_argument("--estimate-position-lag", action="store_true", help="估計位置相對時間戳的偏移（只回報，不套用）")
    ap.add_argument("--position-lag-ms", type=float, default=0.0, help="比對位置時套用的偏移（只影響位置，不影響姿態）")
    ap.add_argument("--position-lead-s", type=float, default=0.0,
                    help="先做「位置 − 速度 × τ」補償（PX4 EKF2_GPS_DELAY 造成的位置超前；11/01 以前的資料約 0.095）")
    ap.add_argument("--out", help="輸出資料夾（預設 <第一個 session>/output）")
    args = ap.parse_args(argv)

    cam = load_camera_config(args.camera)
    pipe = load_pipeline_config(args.pipeline)
    reference = args.reference or pipe.reference_point
    offset_ns = int(round(args.time_offset_ms * 1e6)) if args.time_offset_ms is not None else pipe.frame_time_offset_ns
    sessions = [load_session(p) for p in args.sessions]
    if args.position_lead_s:
        for ses in sessions:
            compensate_position_lead(ses, args.position_lead_s)
        print(f"已套用位置超前補償：位置 − 速度 × {args.position_lead_s:.3f} s")

    # 1) 位置的時間偏移：只有移動中的 session 估得出來；只回報，要套用請明確給 --position-lag-ms
    lags = {}
    if args.estimate_position_lag:
        for ses in sessions:
            lags[ses.path.name] = est = estimate_position_lag(ses)
            tag = "可靠" if est["reliable"] else "不可靠"
            print(f"位置時間偏移（{ses.path.name}）：{est['lag_ms']:+.0f} ms（{tag}，移動樣本 {est['n']}，"
                  f"移動中水平 RMS {est['rmse_zero_m']:.3f} → {est['rmse_best_m']:.3f} m）")
            lead = est["velocity_lead"] = estimate_position_lead(ses)
            if lead["reliable"]:
                print(f"  └ 等效的「位置 − 速度 × τ」補償：τ = {lead['lead_s']:.3f} s（水平 RMS {lead['rmse_zero_m']:.3f} → {lead['rmse_best_m']:.3f} m）")
    lag_ms = args.position_lag_ms
    lag_ns = int(round(lag_ms * 1e6))

    # 2) 姿態誤差：逐 session 列出，合併所有樣本算建議值
    merged = {k: [] for k in POSE_KEYS}
    per_session = {}
    for ses in sessions:
        samples = pose_error_samples(ses, lag_ns)
        per_session[ses.path.name] = {k: _stats(v) for k, v in samples.items()}
        for k in POSE_KEYS:
            merged[k] += samples[k]
        print(f"\n{ses.path.name}（{len(samples['yaw'])} 筆{'' if ses.has_detections else '，無 detections.jsonl'}）：")
        print("\n".join(_fmt_pose(per_session[ses.path.name])))
    pose_err = {k: _stats(v) for k, v in merged.items()}

    # 3) 像素誤差：只有附 detections.jsonl 的 session
    pix_samples = [ses for ses in sessions if ses.has_detections]
    pix_err = {}
    for ses in pix_samples:
        for cls, st in pixel_errors(ses, cam, reference, offset_ns).items():
            pix_err.setdefault(cls, st)  # 多 session 的像素合併留待 B 的資料到齊再做
    suggestion, warnings = suggest(pose_err, pix_err)
    if not pix_samples:  # pose-only：不是配對失敗，是還沒有偵測資料
        warnings = [w for w in warnings if "像素誤差無法標定" not in w]
        warnings.append("這些 session 都沒有 detections.jsonl（pose-only）：像素 σ 無法標定，沿用 config/error_model.yaml 原值")

    print(f"\n合併 {len(sessions)} 個 session，共 {pose_err['yaw']['n']} 筆：")
    print("\n".join(_fmt_pose(pose_err)))
    for cls, s in pix_err.items():
        print(
            f"像素（{cls}，{s['u']['n']} 筆，{reference}）：u 偏差 {s['u']['mean']:+.2f}、標準差 {s['u']['std']:.2f} px；"
            f"v 偏差 {s['v']['mean']:+.2f}、標準差 {s['v']['std']:.2f} px"
        )
    for w in warnings:
        print(f"⚠️ {w}")

    out_dir = Path(args.out) if args.out else sessions[0].path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ", ".join(ses.path.name for ses in sessions)
    text = to_yaml(suggestion, src, reference)
    (out_dir / "error_model_suggested.yaml").write_text(text, encoding="utf-8")
    report = {"sessions": [str(ses.path) for ses in sessions], "position_lead_s_applied": args.position_lead_s,
              "position_lag_ms_applied": lag_ms, "position_lag_estimates": lags,
              "pose": pose_err, "pose_per_session": per_session, "pixel": pix_err, "suggestion": suggestion,
              "warnings": warnings, "reference": reference}
    (out_dir / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n建議值（已寫入 error_model_suggested.yaml）：\n" + text)
    return suggestion, warnings


if __name__ == "__main__":
    main()
