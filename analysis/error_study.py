"""Phase 2 誤差研究：用合成場景量化「像素 → NED」的誤差。

不需要 A、B 的程式：目標的真實位置由我們決定，用 ned_to_pixel 產生「完美的」像素，
再加上各種誤差，量測 pixel_to_ned 的輸出偏離真值多少。

五個實驗（結果與解讀見 analysis/README.md）：
    1. sweep      隨機誤差 vs 高度、離軸角：一階傳播（解析）與 Monte Carlo 對照
    2. budget     誤差預算：哪個誤差來源最大
    3. latency    影像與姿態時間不同步造成的偏差
    4. terrain    目標不在假設的地面平面上（地形起伏）造成的偏差
    5. reference  bbox 參考點（底邊中點 / 中心）對有高度的物體（車、人）造成的偏差

執行（專案根目錄）：
    python -m analysis.error_study
    python -m analysis.error_study --samples 500 --no-plots    # 快速版
輸出：analysis/output/*.csv、*.png（output/ 已在 .gitignore，結果可重現，不進版控）
"""
import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

from coord_transform import (
    CameraMount,
    ErrorModel,
    Pose,
    Status,
    error_budget,
    horizontal_sigma_m,
    ned_to_pixel,
    pixel_to_ned,
)
from coord_transform.config import load_camera_config, load_error_model
from coord_transform.reference_point import CLASS_HEIGHT_M, REFERENCE_METHODS, reference_point

ROOT = Path(__file__).resolve().parents[1]
ALTITUDES_M = (10, 20, 30, 50, 80)
ANGLES_DEG = (0, 10, 20, 30, 40, 45)
OBJECTS = {  # 長 × 寬 × 高（m）。ASSUMED：一般轎車與成人的典型尺寸；高度與 CLASS_HEIGHT_M 一致
    "car": (4.5, 1.8, CLASS_HEIGHT_M["car"]),
    "person": (0.5, 0.5, CLASS_HEIGHT_M["person"]),
}


def level_pose(h, **kw):
    return Pose(n=kw.get("n", 0.0), e=kw.get("e", 0.0), d=-h, roll=kw.get("roll", 0.0), pitch=kw.get("pitch", 0.0), yaw=kw.get("yaw", 0.0))


def pixel_at_angle(intr, theta_rad):
    """影像水平中線上、離光軸 θ 的像素（nadir、水平姿態時 = 正東方離軸角 θ 的地面點）。"""
    return intr.cx + intr.fx * math.tan(theta_rad), intr.cy


# ---------- 1. 隨機誤差掃描 ----------
def monte_carlo(u, v, intr, mount, pose, err: ErrorModel, n, rng):
    """回傳 (DRMS, 95% 水平誤差)。每次抽樣同時對像素、姿態、高度加高斯雜訊。"""
    truth = np.array(pixel_to_ned(u, v, intr, mount, pose, max_range_m=math.inf).position_ned_m[:2])
    sig = np.sqrt(err.variances())
    errs = []
    for du, dv, dr, dp, dy, dd in rng.normal(0.0, sig, size=(n, 6)):
        p = Pose(pose.n, pose.e, pose.d + dd, pose.roll + dr, pose.pitch + dp, pose.yaw + dy)
        hit = pixel_to_ned(u + du, v + dv, intr, mount, p, max_range_m=math.inf)
        if hit.status == Status.OK:
            errs.append(np.array(hit.position_ned_m[:2]) - truth)
    e = np.array(errs)
    radial = np.linalg.norm(e, axis=1)
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1)))), float(np.percentile(radial, 95))


def study_sweep(cam, err, n, rng):
    rows = []
    for h, a in itertools.product(ALTITUDES_M, ANGLES_DEG):
        u, v = pixel_at_angle(cam.intrinsics, math.radians(a))
        pose = level_pose(h)
        lin = horizontal_sigma_m(u, v, cam.intrinsics, cam.mount, pose, err)
        mc, p95 = monte_carlo(u, v, cam.intrinsics, cam.mount, pose, err, n, rng)
        rows.append({"altitude_m": h, "off_nadir_deg": a, "drms_linear_m": lin, "drms_mc_m": mc, "p95_mc_m": p95, "mc_over_linear": mc / lin})
    return rows


# ---------- 2. 誤差預算 ----------
BUDGET_SOURCES = ("pixel", "roll_pitch", "yaw", "altitude")


def study_budget(cam, err, h=30, angles=(0, 40)):
    rows = []
    for a in angles:
        u, v = pixel_at_angle(cam.intrinsics, math.radians(a))
        b = error_budget(u, v, cam.intrinsics, cam.mount, level_pose(h), err)
        rows.append({"altitude_m": h, "off_nadir_deg": a, **{f"src_{k}_m": b[k] for k in BUDGET_SOURCES}})
    return rows


# ---------- 3. 時間不同步 ----------
def latency_error(intr, mount, h, dt_s, speed_mps=0.0, roll_rate_rps=0.0):
    """影像在 t=0 拍攝，但用了 t=dt 的姿態。目標 = t=0 時影像中心正下方的地面點。"""
    true_pose = level_pose(h)
    target = pixel_to_ned(intr.cx, intr.cy, intr, mount, true_pose).position_ned_m
    u, v = ned_to_pixel(target, intr, mount, true_pose)
    wrong = level_pose(h, n=speed_mps * dt_s, roll=roll_rate_rps * dt_s)
    est = pixel_to_ned(u, v, intr, mount, wrong).position_ned_m
    return math.dist(est[:2], target[:2])


def study_latency(cam, h=30):
    rows = []
    for dt_ms in (10, 20, 33, 50, 100):
        dt = dt_ms / 1000
        rows.append(
            {
                "altitude_m": h,
                "dt_ms": dt_ms,
                "translate_5mps_m": latency_error(cam.intrinsics, cam.mount, h, dt, speed_mps=5.0),
                "roll_10dps_m": latency_error(cam.intrinsics, cam.mount, h, dt, roll_rate_rps=math.radians(10)),
                "roll_30dps_m": latency_error(cam.intrinsics, cam.mount, h, dt, roll_rate_rps=math.radians(30)),
                "roll_60dps_m": latency_error(cam.intrinsics, cam.mount, h, dt, roll_rate_rps=math.radians(60)),
            }
        )
    return rows


# ---------- 4. 地形 ----------
def terrain_error(intr, mount, h, theta_rad, rise_m):
    """目標實際站在比假設平面高 rise_m 的地方（例如人行道、斜坡）。"""
    r = (h - rise_m) * math.tan(theta_rad)
    target = (0.0, r, -rise_m)
    pose = level_pose(h)
    u, v = ned_to_pixel(target, intr, mount, pose)
    est = pixel_to_ned(u, v, intr, mount, pose).position_ned_m
    return math.dist(est[:2], target[:2])


def study_terrain(cam, h=30):
    rows = []
    for rise, a in itertools.product((0.2, 0.5, 1.0, 2.0), (0, 20, 40)):
        rows.append({"altitude_m": h, "rise_m": rise, "off_nadir_deg": a, "error_m": terrain_error(cam.intrinsics, cam.mount, h, math.radians(a), rise)})
    return rows


# ---------- 5. bbox 參考點 ----------
def box_corners(center_ne, size, heading_rad):
    """地面上的長方體 8 個角（NED）。center_ne 為接地面中心；heading 為長邊方向（0 = 朝北）。"""
    L, W, Hh = size
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    pts = []
    for dl, dw, up in itertools.product((-L / 2, L / 2), (-W / 2, W / 2), (0.0, Hh)):
        pts.append((center_ne[0] + dl * c - dw * s, center_ne[1] + dl * s + dw * c, -up))
    return pts


def project_bbox(points, intr, mount, pose):
    uv = [ned_to_pixel(p, intr, mount, pose) for p in points]
    if any(x is None for x in uv):
        return None
    us, vs = zip(*uv)
    return min(us), min(vs), max(us), max(vs)


# 第 4 欄是敏感度：center_half_height 但高度先驗高估 30%（例如把 1.5 m 的車當成 1.95 m）
REFERENCE_COLUMNS = REFERENCE_METHODS + ("center_half_height_prior_plus30",)


def estimate(column, bbox, obj, obj_height, intr, mount, pose):
    """用 coord_transform.reference_point（與正式流程同一份程式）估計接地點的 (n, e)。"""
    if column == "center_half_height_prior_plus30":
        u, v, plane = reference_point(bbox, obj, "center_half_height", height_m=1.3 * obj_height)
    else:
        u, v, plane = reference_point(bbox, obj, column, height_m=obj_height)
    hit = pixel_to_ned(u, v, intr, mount, pose, ground_d_m=plane, max_range_m=math.inf)
    return None if hit.status != Status.OK else hit.position_ned_m[:2]


def study_reference(cam, h=30):
    intr = cam.intrinsics
    cameras = {
        "nadir": CameraMount(pitch_rad=-math.pi / 2, offset_body_m=cam.mount.offset_body_m),
        "oblique45": CameraMount(pitch_rad=-math.pi / 4, offset_body_m=cam.mount.offset_body_m),
    }
    image_spots = {  # 物體放在哪個像素位置的地面上
        "center": (intr.cx, intr.cy),
        "lower": (intr.cx, intr.cy + 0.35 * intr.height),
        "right": (intr.cx + 0.4 * intr.width, intr.cy),
    }
    pose = level_pose(h)
    rows = []
    for cam_name, mount in cameras.items():
        for (obj, size), (spot, uv), heading in itertools.product(OBJECTS.items(), image_spots.items(), (0.0, math.pi / 2)):
            ground = pixel_to_ned(*uv, intr, mount, pose, max_range_m=math.inf).position_ned_m
            bbox = project_bbox(box_corners(ground[:2], size, heading), intr, mount, pose)
            row = {"camera": cam_name, "object": obj, "image_spot": spot, "heading_deg": round(math.degrees(heading))}
            for m in REFERENCE_COLUMNS:
                est = estimate(m, bbox, obj, size[2], intr, mount, pose) if bbox else None
                row[f"{m}_m"] = math.dist(est, ground[:2]) if est else float("nan")
            rows.append(row)
    return rows


def summarize_reference(rows):
    """每種相機 × 物體，各方法的平均與最大誤差。"""
    out = []
    for cam_name, obj in itertools.product(("nadir", "oblique45"), OBJECTS):
        sub = [r for r in rows if r["camera"] == cam_name and r["object"] == obj]
        row = {"camera": cam_name, "object": obj}
        for m in REFERENCE_COLUMNS:
            vals = [r[f"{m}_m"] for r in sub]
            row[f"{m}_mean_m"] = float(np.mean(vals))
            row[f"{m}_max_m"] = float(np.max(vals))
        out.append(row)
    return out


# ---------- 輸出 ----------
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def print_table(title, rows, cols=None):
    cols = cols or list(rows[0])
    print(f"\n### {title}\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")


# 圖表配色：高度是有順序的量 → 單一色相由淺到深（ordinal ramp，已通過對比驗證）
_ALT_COLORS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
_SERIES = ("#2a78d6", "#eb6834")
_SURFACE, _TEXT, _TEXT2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_TEXT2, labelsize=9)
    ax.xaxis.label.set_color(_TEXT2)
    ax.yaxis.label.set_color(_TEXT2)


def plot_sweep(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150, facecolor=_SURFACE)
    _style(ax)
    for h, color in zip(ALTITUDES_M, _ALT_COLORS):
        sub = [r for r in rows if r["altitude_m"] == h]
        x = [r["off_nadir_deg"] for r in sub]
        ax.plot(x, [r["drms_linear_m"] for r in sub], color=color, linewidth=2, solid_capstyle="round", label=f"{h} m")
        ax.scatter(x, [r["drms_mc_m"] for r in sub], s=36, color=color, edgecolors=_SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(f"{h} m", (x[-1], sub[-1]["drms_linear_m"]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=_TEXT2)
    ax.set_xlabel("Off-nadir angle of target (deg)")
    ax.set_ylabel("Horizontal error, DRMS (m)")
    ax.set_xlim(-2, 52)
    ax.set_ylim(bottom=0)
    ax.set_title("Random error vs altitude and off-nadir angle", loc="left", color=_TEXT, fontsize=12, pad=22)
    ax.text(0, 1.02, "Line: linear propagation   Dot: Monte Carlo", transform=ax.transAxes, fontsize=9, color=_TEXT2, va="bottom")
    leg = ax.legend(title="Altitude", frameon=False, fontsize=9, title_fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(_TEXT2)
    leg.get_title().set_color(_TEXT2)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def plot_budget(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = BUDGET_SOURCES
    labels = ("Pixel jitter", "Roll / pitch", "Yaw", "Altitude")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150, facecolor=_SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(sources))
    bar_h = 0.34
    for k, (r, color) in enumerate(zip(rows, _SERIES)):
        vals = [r[f"src_{s}_m"] for s in sources]
        pos = y - bar_h / 2 + k * bar_h
        ax.barh(pos, vals, height=bar_h * 0.92, color=color, label=f"{r['off_nadir_deg']}° off-nadir")
        for p, val in zip(pos, vals):
            ax.text(val, p, f" {val:.2f}", va="center", fontsize=8, color=_TEXT2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to horizontal error, 1-sigma (m)")
    ax.set_title(f"Error budget at {rows[0]['altitude_m']} m altitude", loc="left", color=_TEXT, fontsize=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(_TEXT2)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2 座標轉換誤差研究（合成場景）")
    ap.add_argument("--camera", default=ROOT / "config" / "camera.yaml")
    ap.add_argument("--error-model", default=ROOT / "config" / "error_model.yaml")
    ap.add_argument("--out", default=ROOT / "analysis" / "output")
    ap.add_argument("--samples", type=int, default=4000, help="Monte Carlo 每個情境的抽樣數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    cam = load_camera_config(args.camera)
    err = load_error_model(args.error_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"相機：{cam.intrinsics.width}x{cam.intrinsics.height}, fx={cam.intrinsics.fx:.1f}, mount pitch={math.degrees(cam.mount.pitch_rad):.0f} deg")
    print(f"誤差模型：{err}")

    results = {
        "sweep": study_sweep(cam, err, args.samples, rng),
        "budget": study_budget(cam, err),
        "latency": study_latency(cam),
        "terrain": study_terrain(cam),
        "reference": study_reference(cam),
    }
    results["reference_summary"] = summarize_reference(results["reference"])
    for name, rows in results.items():
        write_csv(out / f"{name}.csv", rows)

    print_table("1. 隨機誤差掃描", results["sweep"])
    print_table("2. 誤差預算", results["budget"])
    print_table("3. 時間不同步", results["latency"])
    print_table("4. 地形", results["terrain"])
    print_table("5. bbox 參考點（摘要）", results["reference_summary"])

    if not args.no_plots:
        plot_sweep(results["sweep"], out / "sweep.png")
        plot_budget(results["budget"], out / "budget.png")
    print(f"\n輸出位置：{out}")
    return results


if __name__ == "__main__":
    main()
