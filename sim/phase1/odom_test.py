"""驗證 flight_api v3（資料來源改為 ODOMETRY 100 Hz）
1. ODOMETRY 實際頻率與時間戳  2. 旋轉中拍照的配對/內插比例  3. 舊功能回歸（正方形回原點）
執行前模擬需已啟動，無人機停在地上。
"""
import json
import math
import statistics as stats
import time

from flight_api import Drone
from sim_camera import SimCamera


def ms(xs):
    return f"平均 {stats.mean(xs) * 1000:6.1f} ms，最小 {min(xs) * 1000:6.1f}，最大 {max(xs) * 1000:6.1f}"


d = Drone(verbose=False)
cam = SimCamera()
d.connect()
time.sleep(3)

print("===== 1. ODOMETRY 頻率與時間戳 =====")
with d._lock:
    hs = [h["stamp"] for h in d._hist]
gaps = [b - a for a, b in zip(hs, hs[1:])]
print(f"最近 {len(hs)} 筆：間隔中位數 {stats.median(gaps) * 1000:.1f} ms（約 {1 / stats.median(gaps):.0f} Hz），"
      f"最大間隔 {max(gaps) * 1000:.1f} ms，時間遞增：{all(g > 0 for g in gaps)}")
s = d.get_state()
print("get_state() 欄位：", ", ".join(s.keys()))
print("範例：", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
                         ensure_ascii=False))

print("\n===== 2. 旋轉中拍照：配對與內插 =====")
d.takeoff(5.0)
time.sleep(2)
lag, interp, miss, yr = [], 0, 0, []
end = time.time() + 4
while time.time() < end:
    d.set_velocity(0, 0, 0, 30)
    ok, _ = cam.read(timeout=0.2)
    if not ok:
        continue
    st = cam.last_stamp
    at = d.get_state_at(st)
    if at is None:
        miss += 1
        continue
    interp += at["interpolated"]
    lag.append(d.get_state()["sim_time"] - st)
    yr.append(at["yawspeed_deg_s"])
d.hover()
time.sleep(2)
n = len(lag) + miss
print(f"畫面 {n} 張，對不到 {miss} 張，內插比例 {interp / max(1, len(lag)) * 100:.0f}%")
print(f"最新 odometry 時間 − 畫面時間：{ms(lag)}")
print(f"拍照當下 yawspeed：平均 {stats.mean(yr[len(yr)//2:]):.1f} deg/s（指令 30）")

print("\n===== 3. 回歸：正方形 2 m/s × 3 s 回原點 =====")
s0 = d.get_state()
for v in ((2, 0, 0, 0), (0, 2, 0, 0), (-2, 0, 0, 0), (0, -2, 0, 0)):
    t_end = time.time() + 3
    while time.time() < t_end:
        d.set_velocity(*v)
        time.sleep(0.1)
d.hover()
time.sleep(3)
s1 = d.get_state()
print(f"起點 N {s0['north_m']:.2f} E {s0['east_m']:.2f} → 終點 N {s1['north_m']:.2f} E {s1['east_m']:.2f}，"
      f"誤差 {math.hypot(s1['north_m'] - s0['north_m'], s1['east_m'] - s0['east_m']):.2f} m；"
      f"高度 height_m {s1['height_m']:.2f}，rel_alt_m {s1['rel_alt_m']:.2f}")

d.land()
d.close()
cam.release()
print("\n完成")
