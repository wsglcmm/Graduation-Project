"""Phase 2 Step 3：同步資料記錄器（2026-10-31 起同時輸出規格格式）
自動起飛、依情境飛行，以約 10 fps 存下畫面；同時記錄原始 ODOMETRY 串流與 Gazebo 真值。

    python record_dataset.py                                 # basic：A 原本的路線（8–14 m，含 30 deg/s 旋轉）
    python record_dataset.py --scenario hover --alt 30       # 懸停 30 m，car_blue 直線慢速通過畫面
    python record_dataset.py --scenario hover --alt 20       # 20 m、50 m 各錄一段
    python record_dataset.py --scenario follow --alt 20      # 跟車：car_blue 走 S 形，無人機跟隨並對齊車頭

輸出：/mnt/2disk/vicelead/data/<日期_時間>_<情境>/
    規格格式（analysis/log_format.md，C 的 calibrate_error_model / replay 直接讀）：
    meta.json          ned_origin_enu_m（起飛前實測）、ground_z_enu_m、drone_name、targets，加上鏡頭參數、路線
    poses.jsonl        原始 ODOMETRY 串流（約 100 Hz），規格 §4 Pose
    truth.jsonl        Gazebo 原始 ENU 真值（無人機與每個目標，約 50 Hz）
    frames.jsonl       每張存下的畫面：frame_id、檔名、timestamp_ns（B 跑偵測時對時用）
    原本的格式（保留）：
    frames/、previews/、states.csv（畫面時間內插的 EKF 狀態）、truth.csv（真值轉 NED、真實像素）
執行前模擬需已啟動（start_sim.sh），無人機停在地上。資料夾在起飛成功後才建立。
"""
import argparse
import csv
import json
import math
import os
import time
import traceback

import cv2
import sim_truth as stm
from flight_api import HIST_KEYS, Drone
from session_format import JsonlWriter, meta_dict, pose_msg, quat_to_rpy, truth_msg
from sim_camera import SimCamera
from sim_target import MovingTarget, Route
from sim_truth import DRONE, TARGETS, SimTruth

WS = "/mnt/2disk/vicelead"
SAVE_EVERY = 3            # 30 fps 取 1/3 → 約 10 fps
PREVIEW_EVERY = 50
OBJ_HEIGHT = {"car_blue": 1.45, "car_red": 1.45, "person_1": 1.75, "person_2": 1.75}  # 約略高度（m）
GROUND_Z = 0.0            # default_targets.sdf 的 ground_plane 在 z = 0
TRUTH_HZ = 50.0
CAR = "car_blue"          # hover / follow 情境中會移動的目標
FOLLOW_SPEED = 2.5        # m/s
K_POS, K_YAW = 0.8, 1.5   # 跟車：位置誤差 (1/s)、航向誤差 (deg/s per deg)
HOVER = (0, 0, 0, 0)
BASIC_ROUTE = [  # (說明, 秒數, (前, 右, 下, 偏航角速度))；A 原本的路線
    ("懸停 8 m", 3, (0, 0, 0, 0)),
    ("爬升到約 14 m", 6, (0, 0, -1, 0)),
    ("懸停 14 m", 3, (0, 0, 0, 0)),
    ("往前 1.5 m/s", 3, (1.5, 0, 0, 0)),
    ("往右 1.5 m/s", 3, (0, 1.5, 0, 0)),
    ("往後 1.5 m/s", 6, (-1.5, 0, 0, 0)),
    ("往左 1.5 m/s", 6, (0, -1.5, 0, 0)),
    ("往前 1.5 m/s", 3, (1.5, 0, 0, 0)),
    ("往右 1.5 m/s", 3, (0, 1.5, 0, 0)),
    ("原地旋轉 30 deg/s", 12, (0, 0, 0, 30)),
    ("下降到約 8 m", 6, (0, 0, 1, 0)),
    ("斜飛加旋轉", 6, (1.0, 0.5, 0, 15)),
    ("懸停 8 m", 3, (0, 0, 0, 0)),
]


# ---------- 情境 ----------
def make_plan(args):
    """回傳 (高度, 路線, (目標路徑, 速度) 或 None)。路線中的 "follow" 代表跟車控制。"""
    if args.scenario == "basic":
        return 8.0, BASIC_ROUTE, None
    if args.scenario == "hover":
        alt = args.alt or 30.0
        length = 2.2 * alt   # 畫面沿機頭方向約涵蓋 ±0.89 × 高度，車從畫面外開到畫面外
        # y = -8：避開 car_red (-4, 3)、person_2 (-2, -4)；無人機機頭朝東，車沿畫面上下方向通過
        route = Route(start=(-length / 2, -8.0), heading_deg=0.0).line(length)
        return alt, [(f"懸停 {alt:g} m，{CAR} 直線通過", args.seconds, HOVER)], (route, length / args.seconds)
    alt = args.alt or 20.0
    route = (Route(start=(5.0, 0.0), heading_deg=0.0)       # car_blue 在世界檔的位置，朝東
             .line(15).arc(8, 90).line(10).arc(8, -90).line(10).arc(8, -90).line(15))
    return alt, [(f"跟車 {FOLLOW_SPEED:g} m/s（S 形）", route.length / FOLLOW_SPEED + 3, "follow"),
                 ("懸停", 3, HOVER)], (route, FOLLOW_SPEED)


def follow_cmd(d, tr, car, alt):
    """用 Gazebo 真值跟車（只為了產生有轉彎的資料）：車速前饋 + 位置 P 控制，航向對齊車頭。"""
    stamp = tr.latest_stamp()
    rb, dr = tr.relative_body(stamp, CAR), tr.truth_at(stamp)[DRONE]
    if rb is None or dr is None:
        return HOVER
    st = car.state()
    y = math.radians(dr["yaw_deg"])
    vn, ve = st.vy, st.vx                                    # 車速 ENU → NED
    fwd = vn * math.cos(y) + ve * math.sin(y) + K_POS * rb[0]
    right = -vn * math.sin(y) + ve * math.cos(y) + K_POS * rb[1]
    yaw_err = (math.degrees(math.pi / 2 - st.heading) - dr["yaw_deg"] + 180.0) % 360.0 - 180.0
    yaw_rate = -math.degrees(st.yaw_rate) + K_YAW * yaw_err  # ENU 逆時針為正 → NED 順時針為正
    down = 0.5 * (d.get_state().get("rel_alt_m", alt) - alt)
    return fwd, right, down, yaw_rate


# ---------- 起飛前後 ----------
def measure_origin(d, tr, secs=1.0, max_offset_m=0.2):
    """起飛前讀無人機模型的 Gazebo pose（取最後 secs 秒平均）= PX4 local NED 原點（規格 ned_origin_enu_m）。

    PX4 的原點是開機時的位置、之後固定不變；上一次飛行降落在別處時，現在的位置就不是原點，
    所以 EKF 顯示離原點超過 max_offset_m 就停下來，請使用者重開模擬。回傳 (原點, 當下的 EKF n/e/d)。
    """
    s = d.get_state()
    if s.get("in_air"):
        raise RuntimeError("無人機已經在空中：請先降落或重開模擬，ned_origin_enu_m 要在地上量")
    ekf = tuple(round(s.get(k) or 0.0, 4) for k in ("north_m", "east_m", "down_m"))
    if math.hypot(ekf[0], ekf[1]) > max_offset_m:
        raise RuntimeError(f"EKF 顯示無人機離 local NED 原點 {math.hypot(ekf[0], ekf[1]):.2f} m（上一次飛行降落在別處）。"
                           "PX4 的原點是開機位置：請重開模擬（start_sim.sh）再錄製")
    time.sleep(secs)
    raw = tr.raw_since(DRONE, 0)
    if not raw:
        raise RuntimeError(f"收不到 {DRONE} 的 Gazebo 真值")
    pts = [p for t, p, _ in raw if t >= raw[-1][0] - secs * 1e9]
    return tuple(round(sum(p[i] for p in pts) / len(pts), 4) for i in range(3)), ekf


def goto_alt(d, alt, tol=0.3, timeout=90.0):
    """以速度指令升降到相對高度 alt（m），不依賴 PX4 的起飛高度參數上限。"""
    end = time.time() + timeout
    while time.time() < end:
        h = d.get_state().get("rel_alt_m", 0.0)
        if abs(h - alt) <= tol:
            break
        d.set_velocity(0, 0, max(-2.0, min(2.0, 0.8 * (h - alt))), 0)   # 下為正
        time.sleep(0.1)
    d.hover()
    time.sleep(2)


class StreamRecorder:
    """把 flight_api / sim_truth 累積的原始串流寫成 poses.jsonl、truth.jsonl。每張畫面呼叫一次 drain()。"""

    def __init__(self, out, d, tr, models):
        self.d, self.tr, self.models = d, tr, models
        self.poses = JsonlWriter(os.path.join(out, "poses.jsonl"))
        self.truth = JsonlWriter(os.path.join(out, "truth.jsonl"))
        self.last_usec = d.get_state().get("odom_time_usec") or 0
        self.last_seen = {m: self.last_usec * 1000 for m in models}
        self.last_written = dict.fromkeys(models)
        self.min_gap_ns = 0.95e9 / TRUTH_HZ   # 留 5% 容差：Gazebo 時間是 4 ms 格點，50 Hz 剛好每 20 ms 一筆

    def drain(self):
        for s in self.d.history_since(self.last_usec):
            self.last_usec = s["time_usec"]
            vals = [s.get(k) for k in ("north_m", "east_m", "down_m", "roll_deg", "pitch_deg", "yaw_deg")]
            if None not in vals:
                self.poses.write(pose_msg(s["time_usec"] * 1000, *vals,
                                          vel_ned=[s.get(k) for k in ("vn_m_s", "ve_m_s", "vd_m_s")]))
        for m in self.models:
            for t_ns, p, q in self.tr.raw_since(m, self.last_seen[m]):
                self.last_seen[m] = t_ns
                if self.last_written[m] is not None and t_ns - self.last_written[m] < self.min_gap_ns:
                    continue
                self.truth.write(truth_msg(t_ns, m, *p, rpy=quat_to_rpy(*q)))
                self.last_written[m] = t_ns

    def close(self):
        self.poses.close()
        self.truth.close()


# ---------- 主程式 ----------
def record(args, d, cam, tr):
    alt, plan, moving = make_plan(args)
    origin, ekf_at_origin = measure_origin(d, tr)
    print(f"起飛點（Gazebo ENU）：{origin}；當下 EKF n/e/d：{ekf_at_origin}", flush=True)
    car = None
    if moving:
        car = MovingTarget(CAR, moving[0], moving[1], tr.latest_stamp)
        if not car.place_at_start():
            raise RuntimeError("set_pose 失敗：用 gz service -l | grep set_pose 確認世界有 UserCommands")

    try:
        print(f"起飛到 {min(alt, 10.0):g} m …", flush=True)
        d.takeoff(min(alt, 10.0))
        if alt > 10.0:
            print(f"爬升到 {alt:g} m …", flush=True)
            goto_alt(d, alt)
        time.sleep(2)

        session = time.strftime("%Y%m%d_%H%M%S") + f"_{args.scenario}" + ("" if args.scenario == "basic" else f"{alt:g}m")
        out = os.path.join(WS, "data", session)
        os.makedirs(os.path.join(out, "frames"))
        os.makedirs(os.path.join(out, "previews"))

        state_cols = ["frame", "stamp", "segment"] + list(HIST_KEYS)
        truth_cols = ["frame", "stamp", "name", "n", "e", "d", "roll_deg", "pitch_deg", "yaw_deg",
                      "rel_n", "rel_e", "rel_d", "rel_fwd", "rel_right", "rel_down",
                      "u_ground", "v_ground", "u_top", "v_top", "ground_in_view"]
        fs = open(os.path.join(out, "states.csv"), "w", newline="", encoding="utf-8")
        ft = open(os.path.join(out, "truth.csv"), "w", newline="", encoding="utf-8")
        ws_, wt = csv.DictWriter(fs, state_cols), csv.DictWriter(ft, truth_cols)
        ws_.writeheader()
        wt.writeheader()
        frames_w = JsonlWriter(os.path.join(out, "frames.jsonl"))
        rec = StreamRecorder(out, d, tr, (DRONE,) + TARGETS)
        if car:
            car.start()

        n_saved, k, t_start = 0, 0, time.time()
        for seg, (label, secs, cmd) in enumerate(plan):
            print(f"[{seg:2d}] {label}", flush=True)
            end = time.time() + secs
            while time.time() < end:
                if cmd == "follow":
                    if car.state().done:
                        break
                    d.set_velocity(*follow_cmd(d, tr, car, alt))
                else:
                    d.set_velocity(*cmd)
                ok, frame = cam.read(timeout=0.2)
                rec.drain()
                if not ok:
                    continue
                k += 1
                if k % SAVE_EVERY:
                    continue
                stamp = cam.last_stamp
                st = d.get_state_at(stamp)
                truth = tr.truth_at(stamp)
                if st is None or truth.get(DRONE) is None:
                    continue
                n_saved += 1
                name = f"{n_saved:06d}.jpg"
                cv2.imwrite(os.path.join(out, "frames", name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                frames_w.write({"frame_id": n_saved, "file": name, "timestamp_ns": cam.last_stamp_ns})
                ws_.writerow({"frame": name, "stamp": f"{stamp:.4f}", "segment": seg,
                              **{key: (None if st.get(key) is None else round(st[key], 5)) for key in HIST_KEYS}})
                preview = frame.copy() if n_saved % PREVIEW_EVERY == 1 else None
                for m, tv in truth.items():
                    if tv is None:
                        continue
                    row = {"frame": name, "stamp": f"{stamp:.4f}", "name": m,
                           **{key: round(tv[key], 5) for key in ("n", "e", "d", "roll_deg", "pitch_deg", "yaw_deg")}}
                    if m != DRONE:
                        rel, rb = tr.relative_ned(stamp, m), tr.relative_body(stamp, m)
                        pg, pt = tr.project(stamp, m), tr.project(stamp, m, OBJ_HEIGHT.get(m, 0.0))
                        row.update(rel_n=round(rel[0], 4), rel_e=round(rel[1], 4), rel_d=round(rel[2], 4),
                                   rel_fwd=round(rb[0], 4), rel_right=round(rb[1], 4), rel_down=round(rb[2], 4))
                        if pg:
                            inside = 0 <= pg[0] < stm.WIDTH and 0 <= pg[1] < stm.HEIGHT
                            row.update(u_ground=round(pg[0], 2), v_ground=round(pg[1], 2), ground_in_view=int(inside))
                            if preview is not None and inside:
                                cv2.circle(preview, (int(pg[0]), int(pg[1])), 10, (0, 0, 255), 2)
                                cv2.putText(preview, m, (int(pg[0]) + 12, int(pg[1]) - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        if pt:
                            row.update(u_top=round(pt[0], 2), v_top=round(pt[1], 2))
                            if preview is not None and 0 <= pt[0] < stm.WIDTH and 0 <= pt[1] < stm.HEIGHT:
                                cv2.circle(preview, (int(pt[0]), int(pt[1])), 6, (0, 220, 255), 2)
                    wt.writerow(row)
                if preview is not None:
                    cv2.putText(preview, f"{name}  h={st['height_m']:.1f}m  yaw={st['yaw_deg']:.0f}  seg{seg}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(out, "previews", name), preview)

        if car:
            car.stop()
        d.hover()
        time.sleep(1)
        rec.drain()
        rec.close()
        frames_w.close()
        fs.close()
        ft.close()
        dur = time.time() - t_start

        meta = meta_dict(
            origin, GROUND_Z, DRONE, TARGETS,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            scenario=args.scenario, altitude_m=alt,
            ned_origin_source="起飛前 1 秒的 Gazebo 模型 pose 平均（本次錄製實測）",
            ekf_ned_at_origin_measure_m=ekf_at_origin,
            frames=n_saved, duration_s=round(dur, 1), save_every=SAVE_EVERY,
            counts={"poses": rec.poses.count, "truth": rec.truth.count},
            world="default_targets.sdf",
            camera={"model": "x500_mono_cam_down", "width": stm.WIDTH, "height": stm.HEIGHT,
                    "fx": stm.FX, "fy": stm.FY, "cx": stm.CX, "cy": stm.CY, "distortion": None,
                    "offset_frd_m": stm.CAM_OFFSET_FRD, "offset_reference": "Gazebo 模型原點（規格 §6）",
                    "axes": "畫面右 = 機體右、畫面下 = 機體後、光軸 = 機體下（正下方）"},
            object_height_m=OBJ_HEIGHT,
            route=[{"segment": i, "label": label, "seconds": round(secs, 1),
                    "cmd_fwd_right_down_yawrate": cmd if cmd != "follow" else "follow（真值跟車）"}
                   for i, (label, secs, cmd) in enumerate(plan)],
            moving_target=None if not moving else {
                "name": CAR, "speed_m_s": round(moving[1], 3), "path": moving[0].describe(),
                "method": "Gazebo set_pose 依模擬時間移動（運動學）", "set_pose_failures": car.fail_count},
            control_note=("跟車控制使用 Gazebo 真值，只為了產生有轉彎的資料，不是待評估的演算法"
                          if args.scenario == "follow" else None),
            columns={
                "poses.jsonl": "規格 §4 Pose：原始 ODOMETRY（約 100 Hz），timestamp_ns = time_usec × 1000",
                "truth.jsonl": f"Gazebo 原始 ENU 位置與 ENU/FLU rpy（弧度），每個模型約 {TRUTH_HZ:g} Hz",
                "frames.jsonl": "存下的畫面：frame_id、檔名、timestamp_ns（Gazebo 影像 header）",
                "states.csv": "EKF 狀態（ODOMETRY，於畫面時間內插）；角度為度、距離為公尺、NED，height_m = -down_m",
                "truth.csv": "Gazebo 真值轉 NED；rel_* = 目標 − 無人機（模型原點）；rel_fwd/right/down 只依無人機 yaw 旋轉；"
                             "u/v_ground = 目標地面點的真實像素，u/v_top = 目標頂部（object_height_m）的真實像素，"
                             "皆用真實姿態（含傾斜）與針孔模型投影",
                "stamp": "Gazebo 模擬時間（秒），畫面、EKF、真值共用",
            },
        )
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n完成：存了 {n_saved} 張、poses {rec.poses.count} 筆、truth {rec.truth.count} 筆，"
              f"歷時 {dur:.0f} 秒 → {out}", flush=True)
        if car and car.fail_count:
            print(f"注意：set_pose 失敗 {car.fail_count} 次，目標軌跡可能不連續", flush=True)
    except BaseException:
        if car:
            car.stop()
        if d.get_state().get("in_air"):
            print("\n發生錯誤或被中斷，先降落 …", flush=True)
            try:
                d.land(timeout=180)
            except Exception as e:  # noqa: BLE001  降落失敗也要讓原本的錯誤印出來
                print(f"降落失敗：{e!r}（請在 PX4 終端手動處理）", flush=True)
        raise

    if alt > 15.0:
        print("下降到 8 m …", flush=True)
        goto_alt(d, 8.0)
    print("降落中 …", flush=True)
    d.land(timeout=180)
    print("已落地", flush=True)


def main():
    ap = argparse.ArgumentParser(description="錄製模擬 session（規格格式 + 原本的 CSV）")
    ap.add_argument("--scenario", choices=("basic", "hover", "follow"), default="basic")
    ap.add_argument("--alt", type=float, help="飛行高度 m（hover 預設 30、follow 預設 20；basic 固定 8–14）")
    ap.add_argument("--seconds", type=float, default=45.0, help="hover 情境的懸停秒數（預設 45）")
    args = ap.parse_args()

    alt = args.alt or 30.0
    d = Drone(verbose=False, max_xy=5.0, max_z=2.0, max_alt=max(30.0, alt + 10.0))
    cam = SimCamera()
    tr = SimTruth()
    code = 0
    try:
        d.connect()
        record(args, d, cam, tr)
    except BaseException:  # noqa: BLE001  包含 Ctrl+C；os._exit 前先把錯誤印出來
        traceback.print_exc()
        code = 1
    finally:
        d.close()
        cam.release()
        tr.release()
        print("結束", flush=True)
        os._exit(code)   # 直接結束，略過 C++ 函式庫的關閉程序（避免 terminate called…）


if __name__ == "__main__":
    main()
