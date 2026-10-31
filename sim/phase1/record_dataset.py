"""Phase 2 Step 3：同步資料記錄器
自動起飛、飛一段固定路線，以約 10 fps 存下畫面，並記錄每張畫面拍攝當下的
EKF 狀態（flight_api.get_state_at）與模擬真值（sim_truth）。

輸出：/mnt/2disk/vicelead/data/<日期_時間>/
    frames/000001.jpg …   原始畫面 1280×960
    previews/*.jpg         每 50 張一張疊圖（紅 = 目標地面點、黃 = 目標頂部）
    states.csv             每張畫面一列：EKF 狀態
    truth.csv              每張畫面 × 每個模型一列：真值、相對位移、真實像素
    meta.json              鏡頭參數、路線、欄位說明
執行前模擬需已啟動（start_sim.sh），無人機停在地上。
"""
import csv
import json
import os
import time

import cv2
import sim_truth as stm
from flight_api import HIST_KEYS, Drone
from sim_camera import SimCamera
from sim_truth import DRONE, TARGETS, SimTruth

WS = "/mnt/2disk/vicelead"
SAVE_EVERY = 3            # 30 fps 取 1/3 → 約 10 fps
PREVIEW_EVERY = 50
OBJ_HEIGHT = {"car_blue": 1.45, "car_red": 1.45, "person_1": 1.75, "person_2": 1.75}  # 約略高度（m）
ROUTE = [  # (說明, 秒數, 前, 右, 下, 偏航角速度)
    ("懸停 8 m", 3, 0, 0, 0, 0),
    ("爬升到約 14 m", 6, 0, 0, -1, 0),
    ("懸停 14 m", 3, 0, 0, 0, 0),
    ("往前 1.5 m/s", 3, 1.5, 0, 0, 0),
    ("往右 1.5 m/s", 3, 0, 1.5, 0, 0),
    ("往後 1.5 m/s", 6, -1.5, 0, 0, 0),
    ("往左 1.5 m/s", 6, 0, -1.5, 0, 0),
    ("往前 1.5 m/s", 3, 1.5, 0, 0, 0),
    ("往右 1.5 m/s", 3, 0, 1.5, 0, 0),
    ("原地旋轉 30 deg/s", 12, 0, 0, 0, 30),
    ("下降到約 8 m", 6, 0, 0, 1, 0),
    ("斜飛加旋轉", 6, 1.0, 0.5, 0, 15),
    ("懸停 8 m", 3, 0, 0, 0, 0),
]

out = os.path.join(WS, "data", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(os.path.join(out, "frames"))
os.makedirs(os.path.join(out, "previews"))

d = Drone(verbose=False)
cam = SimCamera()
tr = SimTruth()
d.connect()
print("起飛到 8 m …", flush=True)
d.takeoff(8.0)
time.sleep(2)

state_cols = ["frame", "stamp", "segment"] + list(HIST_KEYS)
truth_cols = ["frame", "stamp", "name", "n", "e", "d", "roll_deg", "pitch_deg", "yaw_deg",
              "rel_n", "rel_e", "rel_d", "rel_fwd", "rel_right", "rel_down",
              "u_ground", "v_ground", "u_top", "v_top", "ground_in_view"]
fs = open(os.path.join(out, "states.csv"), "w", newline="", encoding="utf-8")
ft = open(os.path.join(out, "truth.csv"), "w", newline="", encoding="utf-8")
ws_, wt = csv.DictWriter(fs, state_cols), csv.DictWriter(ft, truth_cols)
ws_.writeheader()
wt.writeheader()

n_saved, k, t_start = 0, 0, time.time()
for seg, (label, secs, fwd, right, down, yaw) in enumerate(ROUTE):
    print(f"[{seg:2d}] {label}", flush=True)
    end = time.time() + secs
    while time.time() < end:
        d.set_velocity(fwd, right, down, yaw)
        ok, frame = cam.read(timeout=0.2)
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

d.hover()
time.sleep(1)
fs.close()
ft.close()
dur = time.time() - t_start

meta = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "frames": n_saved, "duration_s": round(dur, 1), "save_every": SAVE_EVERY,
    "world": "default_targets.sdf",
    "camera": {"model": "x500_mono_cam_down", "width": stm.WIDTH, "height": stm.HEIGHT,
               "fx": stm.FX, "fy": stm.FY, "cx": stm.CX, "cy": stm.CY, "distortion": None,
               "offset_frd_m": stm.CAM_OFFSET_FRD,
               "axes": "畫面右 = 機體右、畫面下 = 機體後、光軸 = 機體下（正下方）"},
    "object_height_m": OBJ_HEIGHT,
    "route": [{"segment": i, "label": r[0], "seconds": r[1], "cmd_fwd_right_down_yawrate": r[2:]}
              for i, r in enumerate(ROUTE)],
    "columns": {
        "states.csv": "EKF 狀態（ODOMETRY，於畫面時間內插）；角度為度、距離為公尺、NED，height_m = -down_m",
        "truth.csv": "Gazebo 真值轉 NED；rel_* = 目標 − 無人機（模型原點）；rel_fwd/right/down 只依無人機 yaw 旋轉；"
                     "u/v_ground = 目標地面點的真實像素，u/v_top = 目標頂部（object_height_m）的真實像素，"
                     "皆用真實姿態（含傾斜）與針孔模型投影",
        "stamp": "Gazebo 模擬時間（秒），畫面、EKF、真值共用",
    },
}
with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n完成：存了 {n_saved} 張，歷時 {dur:.0f} 秒 → {out}", flush=True)
print("降落中 …", flush=True)
d.land()
print("已落地", flush=True)
d.close()
cam.release()
tr.release()
print("結束", flush=True)
os._exit(0)   # 直接結束，略過 C++ 函式庫的關閉程序（避免 terminate called…）
