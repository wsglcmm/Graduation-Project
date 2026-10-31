"""Phase 2 Step 2：驗證 sim_truth
1. pose/info 頻率、時間戳、各模型真值（對照世界檔）
2. 無人機真值 vs EKF（get_state_at）：地面、懸停、前飛、旋轉
3. 回到原點上空 10 m，用真值 + 針孔模型把目標投影到畫面上，存成疊圖
執行前模擬需已啟動（start_sim.sh），無人機停在地上。
"""
import math
import statistics as st
import time

import cv2
from flight_api import Drone
from sim_camera import SimCamera
from sim_truth import DRONE, TARGETS, SimTruth

OUT = "/mnt/2disk/vicelead/tmp/truth_overlay_10m.jpg"
FX, CX, CY = 540.0, 640.0, 480.0
CAM_ABOVE_MODEL = 0.10          # camera_link 在模型原點上方 0.10 m（SDF）
EXPECT = {"car_blue": (0, 5), "car_red": (3, -4), "person_1": (4, 1), "person_2": (-4, -2)}

d = Drone(verbose=False)
cam = SimCamera()
tr = SimTruth()
d.connect()
time.sleep(2)

print("===== 1. pose/info 頻率與各模型真值 =====")
with tr._lock:
    h = list(tr._hist[DRONE])
gaps = [b[0] - a[0] for a, b in zip(h, h[1:])]
print(f"無人機真值 {len(h)} 筆，間隔中位數 {st.median(gaps) * 1000:.1f} ms，最大 {max(gaps) * 1000:.1f} ms，"
      f"時間遞增：{all(g > 0 for g in gaps)}")
s = tr.latest_stamp()
print(f"真值最新時間 {s:.3f} s；EKF sim_time {d.get_state()['sim_time']:.3f} s")
for m, v in tr.truth_at(s).items():
    exp = f"（世界檔 n,e = {EXPECT[m]}）" if m in EXPECT else ""
    txt = "None" if v is None else f"n {v['n']:7.2f}  e {v['e']:7.2f}  d {v['d']:6.2f}  yaw {v['yaw_deg']:7.1f}"
    print(f"  {m:22s} {txt} {exp}")


def compare(label, secs, cmd=None):
    dn, de, dd, dy = [], [], [], []
    end = time.time() + secs
    while time.time() < end:
        if cmd:
            d.set_velocity(*cmd)
        time.sleep(0.1)
        t = tr.latest_stamp() - 0.05
        a, b = tr.truth_at(t)[DRONE], d.get_state_at(t)
        if a is None or b is None:
            continue
        dn.append(b["north_m"] - a["n"])
        de.append(b["east_m"] - a["e"])
        dd.append(b["down_m"] - a["d"])
        dy.append((b["yaw_deg"] - a["yaw_deg"] + 180) % 360 - 180)
    f = lambda xs: f"平均 {st.mean(xs):+.3f}，範圍 {min(xs):+.3f} ~ {max(xs):+.3f}"  # noqa: E731
    print(f"[{label}] EKF − 真值（{len(dn)} 筆）")
    print("   北 m  ", f(dn))
    print("   東 m  ", f(de))
    print("   下 m  ", f(dd))
    print("   yaw 度", f(dy))


print("\n===== 2. 無人機：EKF vs 真值 =====")
compare("地面靜止", 3)
d.takeoff(10.0)
time.sleep(2)
compare("10 m 懸停", 3)
compare("前飛 2 m/s", 4, (2, 0, 0, 0))
compare("旋轉 30 deg/s", 4, (0, 0, 0, 30))
d.hover()
time.sleep(2)

print("\n===== 3. 原點上空 10 m：真值投影到畫面 =====")
end = time.time() + 12
while time.time() < end:              # 用 EKF 位置飛回原點上空
    s = d.get_state()
    y = math.radians(s["yaw_deg"])
    en, ee = -s["north_m"], -s["east_m"]
    d.set_velocity(0.6 * (en * math.cos(y) + ee * math.sin(y)),
                   0.6 * (-en * math.sin(y) + ee * math.cos(y)), 0, 0)
    time.sleep(0.1)
d.hover()
time.sleep(3)
ok, frame = cam.read()
stp = cam.last_stamp
at = d.get_state_at(stp)
dr = tr.truth_at(stp)[DRONE]
print(f"畫面時間 {stp:.3f}；EKF 高度 {at['height_m']:.2f} m、yaw {at['yaw_deg']:.1f}；"
      f"真值 n {dr['n']:.2f} e {dr['e']:.2f} d {dr['d']:.2f} yaw {dr['yaw_deg']:.1f}")
for name in TARGETS:
    rb = tr.relative_body(stp, name)
    if rb is None:
        print(f"  {name:9s} 查不到真值，略過")
        continue
    fwd, right, down = rb
    h = down - CAM_ABOVE_MODEL
    u, v = CX + right * FX / h, CY - fwd * FX / h
    inside = 0 <= u < 1280 and 0 <= v < 960
    print(f"  {name:9s} 前 {fwd:6.2f}  右 {right:6.2f}  下 {down:6.2f} m → 預測像素 ({u:6.1f}, {v:6.1f})"
          f"{'' if inside else '（畫面外）'}")
    if inside:
        cv2.circle(frame, (int(u), int(v)), 14, (0, 0, 255), 2)
        cv2.putText(frame, name, (int(u) + 16, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite(OUT, frame)
print("疊圖已存：", OUT)

d.land()
d.close()
cam.release()
tr.release()
print("\n完成")
