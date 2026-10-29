"""重播 session：把 A 的姿態、B 的偵測照時間順序餵給 TargetWorldPipeline，再和 Gazebo 真值比對。

這就是「用真實模擬資料量化誤差」：Phase 2 的誤差研究是純合成，這裡用的是 A、B 實際錄下來的資料。

執行：
    python -m analysis.replay <session>                         # 用 config/*.yaml 的設定
    python -m analysis.replay <session> --reference center_half_height
    python -m analysis.replay <session> --estimate-time-offset  # 順便估計影像與姿態的時間偏移
輸出：<session>/output/targetworld.jsonl、metrics.json、replay_error.png
"""
import argparse
import dataclasses
import heapq
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analysis.session import Session, load_session, write_jsonl
from coord_transform import TargetWorldPipeline
from coord_transform.config import (
    CameraConfig,
    PipelineConfig,
    load_camera_config,
    load_error_model,
    load_pipeline_config,
)
from coord_transform.uncertainty import ErrorModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARRIVAL_LATENCY_NS = 50_000_000  # 假設偵測在影像時間後 50 ms 才送到（只影響排隊，不影響結果）
DEFAULT_GATE_M = 10.0
MIN_IMPROVEMENT_M, MIN_IMPROVEMENT_FRAC = 0.02, 0.03  # 時間偏移估計的可靠門檻


def run_pipeline(
    session: Session,
    cam: CameraConfig,
    pipe_cfg: PipelineConfig,
    err: Optional[ErrorModel] = None,
    arrival_latency_ns: int = DEFAULT_ARRIVAL_LATENCY_NS,
) -> List[dict]:
    """依「到達時間」合併兩個串流後逐筆餵入，模擬即時運作（含排隊等待姿態）。"""
    pipe = TargetWorldPipeline.from_config(cam, pipe_cfg, err)
    events = [(m["timestamp_ns"], 0, k, m) for k, m in enumerate(session.poses)]
    events += [(f["timestamp_ns"] + arrival_latency_ns, 1, k, f) for k, f in enumerate(session.frames)]
    heapq.heapify(events)
    out: List[dict] = []
    while events:
        _, kind, _, msg = heapq.heappop(events)
        out += pipe.on_pose(msg) if kind == 0 else pipe.on_frame(msg)
    return out + pipe.flush()


def match_targets(session: Session, outputs: List[dict], time_offset_ns: int, gate_m: float = DEFAULT_GATE_M) -> List[dict]:
    """每個 status=ok 的輸出，配對到同類別、最近的真值目標（每幀每個真值只用一次）。"""
    rows = []
    for out in outputs:
        t = out["timestamp_ns"] + time_offset_ns
        truth = {}
        for name, cls in session.targets.items():
            p = session.target_ned(name, t)
            if p is not None:
                truth[name] = (cls, p)
        pairs = []
        for k, tgt in enumerate(out["targets"]):
            if tgt["status"] != "ok":
                continue
            est = np.array([tgt["position_ned_m"]["n"], tgt["position_ned_m"]["e"]])
            for name, (cls, p) in truth.items():
                if cls == tgt["class"]:
                    pairs.append((float(np.linalg.norm(est - p[:2])), k, name, est, p))
        used_k, used_name = set(), set()
        for dist, k, name, est, p in sorted(pairs, key=lambda x: x[0]):
            if dist > gate_m or k in used_k or name in used_name:
                continue
            used_k.add(k)
            used_name.add(name)
            tgt = out["targets"][k]
            rows.append(
                {
                    "frame_id": out["frame_id"],
                    "timestamp_ns": out["timestamp_ns"],
                    "truth_name": name,
                    "track_id": tgt["track_id"],
                    "class": tgt["class"],
                    "err_n_m": float(est[0] - p[0]),
                    "err_e_m": float(est[1] - p[1]),
                    "error_m": dist,
                    "uncertainty_m": tgt["uncertainty_m"],
                }
            )
    return rows


def summarize(outputs: List[dict], matches: List[dict]) -> dict:
    statuses: Dict[str, int] = {}
    for o in outputs:
        for t in o["targets"]:
            statuses[t["status"]] = statuses.get(t["status"], 0) + 1
    n_ok = statuses.get("ok", 0)
    m = {"frames": len(outputs), "targets_by_status": statuses, "matched": len(matches), "unmatched_ok": n_ok - len(matches)}
    if not matches:
        return m
    e = np.array([r["error_m"] for r in matches])
    m.update(
        {
            "error_mean_m": float(e.mean()),
            "error_rmse_m": float(np.sqrt(np.mean(e**2))),
            "error_median_m": float(np.median(e)),
            "error_p95_m": float(np.percentile(e, 95)),
            "error_max_m": float(e.max()),
            "bias_n_m": float(np.mean([r["err_n_m"] for r in matches])),
            "bias_e_m": float(np.mean([r["err_e_m"] for r in matches])),
        }
    )
    with_u = [(r["error_m"], r["uncertainty_m"]) for r in matches if r["uncertainty_m"]]
    if with_u:
        err_, u = np.array(with_u).T
        # 若 uncertainty_m 準確且誤差近似 2D 高斯：誤差 ≤ 1×DRMS 約 63–68%，≤ 2×DRMS 約 95–98%
        m.update(
            {
                "uncertainty_mean_m": float(u.mean()),
                "within_1x_uncertainty": float(np.mean(err_ <= u)),
                "within_2x_uncertainty": float(np.mean(err_ <= 2 * u)),
            }
        )
    by_class = {}
    for cls in sorted({r["class"] for r in matches}):
        ec = np.array([r["error_m"] for r in matches if r["class"] == cls])
        by_class[cls] = {"n": int(ec.size), "rmse_m": float(np.sqrt(np.mean(ec**2))), "median_m": float(np.median(ec))}
    m["by_class"] = by_class
    return m


def evaluate(session, cam, pipe_cfg, err=None, gate_m=DEFAULT_GATE_M):
    outputs = run_pipeline(session, cam, pipe_cfg, err)
    matches = match_targets(session, outputs, pipe_cfg.frame_time_offset_ns, gate_m)
    return outputs, matches, summarize(outputs, matches)


def estimate_time_offset(session, cam, pipe_cfg, search_ms=(-100.0, 100.0), coarse_step_ms=4.0, fine_step_ms=0.5):
    """網格搜尋 frame_time_offset，使與真值的 RMSE 最小（先粗後細）。

    原理：用錯時刻的姿態，誤差 ≈ 高度 × 角速度 × 時間差；無人機姿態有在變化時，RMSE 對偏移很敏感。
    懸停不動時估不出來（RMSE 對偏移是平的），所以請用「跟車轉彎」的紀錄。
    回傳 (最佳偏移 ms, 可靠與否, [(offset_ms, rmse_m), ...])。
    可靠的判斷：最佳偏移的 RMSE 要比「不補償（0 ms）」至少低 MIN_IMPROVEMENT；
    曲線太平代表時間偏移造成的誤差被雜訊淹沒，估出來的數字不能信。
    """

    def rmse(offset_ms):
        cfg = dataclasses.replace(pipe_cfg, frame_time_offset_ns=int(round(offset_ms * 1e6)))
        _, matches, s = evaluate(session, cam, cfg, err=None)
        return s.get("error_rmse_m", math.inf) if matches else math.inf

    curve = [(o, rmse(o)) for o in np.arange(search_ms[0], search_ms[1] + 1e-9, coarse_step_ms)]
    best = min(curve, key=lambda x: x[1])[0]
    fine = [(o, rmse(o)) for o in np.arange(best - coarse_step_ms, best + coarse_step_ms + 1e-9, fine_step_ms)]
    curve = sorted(set(curve + fine))
    best_ms, best_rmse = min(fine, key=lambda x: x[1])
    zero_rmse = rmse(0.0)
    reliable = (zero_rmse - best_rmse) >= max(MIN_IMPROVEMENT_M, MIN_IMPROVEMENT_FRAC * zero_rmse)
    return float(best_ms), bool(reliable), curve


def plot_errors(matches, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, text, text2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(True, color=grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=text2, labelsize=9)
    t0 = matches[0]["timestamp_ns"]
    for cls, color in zip(sorted({r["class"] for r in matches}), ("#2a78d6", "#eb6834")):
        rs = [r for r in matches if r["class"] == cls]
        x = [(r["timestamp_ns"] - t0) / 1e9 for r in rs]
        e = np.array([r["error_m"] for r in rs])
        w = min(15, len(e))  # 約 0.5 秒（30 Hz）的滑動 RMS，讓趨勢看得出來
        rms = np.sqrt(np.convolve(e**2, np.ones(w) / w, mode="same"))
        ax.plot(x, e, color=color, linewidth=0.8, alpha=0.25)
        ax.plot(x, rms, color=color, linewidth=2, solid_capstyle="round", label=f"{cls}: error (0.5 s RMS)")
        if all(r["uncertainty_m"] for r in rs):
            ax.plot(x, [r["uncertainty_m"] for r in rs], color=color, linewidth=1.5, linestyle=(0, (5, 3)), label=f"{cls}: uncertainty_m")
    ax.set_xlabel("Time (s)", color=text2)
    ax.set_ylabel("Horizontal error (m)", color=text2)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", color=text, fontsize=12)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    for t in leg.get_texts():
        t.set_color(text2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def _print_summary(s):
    print(f"\n幀數 {s['frames']}，目標狀態 {s['targets_by_status']}，配對成功 {s['matched']}（未配對的 ok：{s['unmatched_ok']}）")
    if "error_rmse_m" not in s:
        print("沒有任何配對成功的目標，請檢查 meta.json 的原點與 targets")
        return
    print(
        f"水平誤差：平均 {s['error_mean_m']:.2f}、RMSE {s['error_rmse_m']:.2f}、中位數 {s['error_median_m']:.2f}、"
        f"95% {s['error_p95_m']:.2f}、最大 {s['error_max_m']:.2f} m"
    )
    print(f"平均偏差：n {s['bias_n_m']:+.2f}、e {s['bias_e_m']:+.2f} m")
    if "within_1x_uncertainty" in s:
        print(
            f"uncertainty_m 平均 {s['uncertainty_mean_m']:.2f} m；誤差 ≤ 1 倍：{s['within_1x_uncertainty']:.0%}（理想約 63–68%），"
            f"≤ 2 倍：{s['within_2x_uncertainty']:.0%}（理想約 95–98%）"
        )
    for cls, c in s.get("by_class", {}).items():
        print(f"  {cls}：{c['n']} 筆，RMSE {c['rmse_m']:.2f}、中位數 {c['median_m']:.2f} m")


def main(argv=None):
    ap = argparse.ArgumentParser(description="重播模擬紀錄並與 Gazebo 真值比對")
    ap.add_argument("session")
    ap.add_argument("--camera", default=ROOT / "config" / "camera.yaml")
    ap.add_argument("--pipeline", default=ROOT / "config" / "pipeline.yaml")
    ap.add_argument("--error-model", default=ROOT / "config" / "error_model.yaml")
    ap.add_argument("--reference", help="覆寫 pipeline.yaml 的 reference_point")
    ap.add_argument("--time-offset-ms", type=float, help="覆寫 pipeline.yaml 的 frame_time_offset_ms")
    ap.add_argument("--estimate-time-offset", action="store_true", help="網格搜尋最佳的影像時間偏移")
    ap.add_argument("--gate", type=float, default=DEFAULT_GATE_M, help="配對真值的最大距離 m")
    ap.add_argument("--out", help="輸出資料夾（預設 <session>/output）")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    session = load_session(args.session)
    cam = load_camera_config(args.camera)
    pipe_cfg = load_pipeline_config(args.pipeline)
    if args.reference:
        pipe_cfg = dataclasses.replace(pipe_cfg, reference_point=args.reference)
    if args.time_offset_ms is not None:
        pipe_cfg = dataclasses.replace(pipe_cfg, frame_time_offset_ns=int(round(args.time_offset_ms * 1e6)))
    err = load_error_model(args.error_model)
    out_dir = Path(args.out) if args.out else Path(args.session) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if abs(session.ground_plane_d_m - cam.ground_d_m) > 0.05:
        print(
            f"⚠️ 地面高度不一致：session 推得地面 d = {session.ground_plane_d_m:.2f} m，"
            f"camera.yaml 的 plane_d_m = {cam.ground_d_m:.2f} m（會造成約 差值×tan(離軸角) 的誤差）"
        )

    result = {"session": str(session.path)}
    if args.estimate_time_offset:
        best, reliable, curve = estimate_time_offset(session, cam, pipe_cfg)
        result.update({"time_offset_estimate_ms": best, "time_offset_reliable": reliable})
        result["time_offset_curve"] = [{"offset_ms": o, "rmse_m": r} for o, r in curve]
        if reliable:
            print(f"估計的影像時間偏移：{best:+.1f} ms → 可在 config/pipeline.yaml 設 frame_time_offset_ms: {best:.1f}")
            pipe_cfg = dataclasses.replace(pipe_cfg, frame_time_offset_ns=int(round(best * 1e6)))
        else:
            print(
                f"⚠️ 時間偏移估計不可靠（最佳 {best:+.1f} ms，但 RMSE 幾乎沒有比 0 ms 好）：誤差被其他雜訊淹沒。"
                "請改用姿態變化較大的紀錄（跟車轉彎），或先標定並降低其他誤差。以下沿用原設定。"
            )

    outputs, matches, summary = evaluate(session, cam, pipe_cfg, err, args.gate)
    result.update({"reference_point": pipe_cfg.reference_point, "frame_time_offset_ms": pipe_cfg.frame_time_offset_ns / 1e6, **summary})
    print(f"參考點 {pipe_cfg.reference_point}、影像時間偏移 {pipe_cfg.frame_time_offset_ns / 1e6:+.1f} ms")
    _print_summary(summary)

    write_jsonl(out_dir / "targetworld.jsonl", outputs)
    write_jsonl(out_dir / "matches.jsonl", matches)
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if matches and not args.no_plots:
        plot_errors(matches, out_dir / "replay_error.png", f"Replay: {pipe_cfg.reference_point}")
    print(f"\n輸出位置：{out_dir}")
    return result


if __name__ == "__main__":
    main()
