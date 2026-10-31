"""驗證 get_state_at：一邊旋轉一邊拍照，比對「拍照當下的 yaw」與「讀取當下最新的 yaw」"""
import time

from flight_api import Drone
from sim_camera import SimCamera

d = Drone(verbose=False)
cam = SimCamera()
d.connect()
d.takeoff(5.0)

rows, miss, t0 = [], 0, time.time()
while time.time() - t0 < 4:
    d.set_velocity(0, 0, 0, 30)          # 每秒轉 30 度
    ok, _ = cam.read(timeout=0.2)
    if not ok:
        continue
    st = cam.last_stamp
    at = d.get_state_at(st)
    now = d.get_state()
    if at is None:
        miss += 1
        continue
    rows.append((st, at["yaw_deg"], at["interpolated"], now["sim_time"], now["yaw_deg"]))
d.hover()
time.sleep(1)

print(f"畫面 {len(rows) + miss} 張，對得到姿態 {len(rows)} 張，對不到 {miss} 張")
lag = [r[3] - r[0] for r in rows]
print(f"最新姿態時間 - 畫面時間：平均 {sum(lag) / len(lag) * 1000:.0f} ms，"
      f"最小 {min(lag) * 1000:.0f} ms，最大 {max(lag) * 1000:.0f} ms")
print(f"內插的比例：{sum(r[2] for r in rows) / len(rows) * 100:.0f}%")
print("  畫面時間   拍照當下yaw  最新姿態時間   最新yaw")
for r in rows[::15]:
    print(f"{r[0]:10.3f}  {r[1]:10.2f}  {r[3]:11.3f}  {r[4]:9.2f}")

d.land()
d.close()
cam.release()
print("完成")
