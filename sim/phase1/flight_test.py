"""Phase 1：flight_api 功能測試（正方形、限幅、看門狗、偏航）＋ 10 m 高空拍照"""
import math
import time

import cv2
from flight_api import Drone
from sim_camera import SimCamera

OUT = "/mnt/2disk/vicelead/tmp"


def fly(d, fwd, right, down, yaw, secs, hz=10):
    """以 hz 頻率持續送速度指令 secs 秒（C 的追蹤迴圈也會是這種寫法）。"""
    end = time.time() + secs
    while time.time() < end:
        d.set_velocity(fwd, right, down, yaw)
        time.sleep(1.0 / hz)


def show(d, tag):
    s = d.get_state()
    v = math.hypot(s.get("vn_m_s", 0), s.get("ve_m_s", 0))
    print(f"[{tag}] 高度 {s.get('rel_alt_m', 0):5.2f} m | N {s.get('north_m', 0):6.2f} "
          f"E {s.get('east_m', 0):6.2f} | 水平速度 {v:4.2f} m/s | yaw {s.get('yaw_deg', 0):6.1f} "
          f"| {s.get('flight_mode')}", flush=True)


d = Drone()
cam = SimCamera()
d.connect()
d.takeoff(10.0)
show(d, "起飛完成")
ok, f = cam.read()
cv2.imwrite(f"{OUT}/cam_10m.jpg", f)
print("10 m 高空畫面已存：", f"{OUT}/cam_10m.jpg")

print("== 測試 1：正方形，每邊 2 m/s × 4 s ==")
for name, v in [("前", (2, 0, 0, 0)), ("右", (0, 2, 0, 0)), ("後", (-2, 0, 0, 0)), ("左", (0, -2, 0, 0))]:
    fly(d, *v, 4)
    show(d, f"往{name}後")
d.hover()
time.sleep(3)
show(d, "懸停 3 秒")

print("== 測試 2：限幅 ==")
print("  要求 (10, 0, -5, 90) -> 實際採用", d.set_velocity(10, 0, -5, 90))
d.hover()
time.sleep(3)

print("== 測試 3：看門狗（送 2 m/s 後突然不送） ==")
fly(d, 2, 0, 0, 0, 3)
t0 = time.time()
for w in (0.25, 0.5, 1.0, 2.0, 3.0):
    time.sleep(max(0.0, w - (time.time() - t0)))
    s = d.get_state()
    print(f"  停送 {w:4.2f}s 後 水平速度 {math.hypot(s['vn_m_s'], s['ve_m_s']):4.2f} m/s", flush=True)

print("== 測試 4：偏航 30 deg/s × 3 s ==")
y0 = d.get_state()["yaw_deg"]
fly(d, 0, 0, 0, 30, 3)
d.hover()
time.sleep(1)
print(f"  yaw {y0:.1f} -> {d.get_state()['yaw_deg']:.1f}（預期約 +90）")

d.land()
show(d, "降落後")
d.close()
cam.release()
print("全部測試完成")
