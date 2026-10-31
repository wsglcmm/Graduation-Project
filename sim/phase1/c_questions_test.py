"""回答 C 的四個問題：
Q1 PX4 有沒有送 ODOMETRY（MAVSDK telemetry.odometry()），time_usec 是什麼
Q2 畫面與姿態的時間差是固定偏移，還是取樣錯開（20 Hz vs 100 Hz 比較）
Q3 get_state() 的實際輸出
Q4 camera_link 相對 base_link 的精確位移（直接讀 Gazebo 的 pose）
執行前模擬需已啟動（start_sim.sh），無人機停在地上。
"""
import asyncio
import json
import math
import statistics as stats
import threading
import time

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

from flight_api import Drone
from sim_camera import SimCamera


# ---------- 四元數小工具（w, x, y, z） ----------
def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def q_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def q_rot(q, v):
    return q_mul(q_mul(q, (0.0, *v)), q_conj(q))[1:]


def q_rpy_deg(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(round(math.degrees(a), 3) for a in (roll, pitch, yaw))


def ms(xs):
    return f"平均 {stats.mean(xs) * 1000:6.1f} ms，最小 {min(xs) * 1000:6.1f}，最大 {max(xs) * 1000:6.1f}，筆數 {len(xs)}"


d = Drone(verbose=False)
cam = SimCamera()
d.connect()
time.sleep(1)

# ---------- Q3 ----------
print("\n===== Q3 get_state() 實際輸出 =====")
print(json.dumps(d.get_state(), indent=2, ensure_ascii=False, default=str))

# ---------- Q4 ----------
print("\n===== Q4 camera_link 相對 base_link（Gazebo 實際 pose） =====")
got, ev = {}, threading.Event()


def on_pose(msg):
    for p in msg.pose:
        if p.name in ("base_link", "camera_link", "x500_mono_cam_down_0"):
            got[p.name] = ((p.position.x, p.position.y, p.position.z),
                           (p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z))
    if "base_link" in got and "camera_link" in got:
        ev.set()


node = Node()
node.subscribe(Pose_V, "/world/default/pose/info", on_pose)
if ev.wait(5):
    (pb, qb), (pc, qc) = got["base_link"], got["camera_link"]
    print("base_link   （相對模型）位置", tuple(round(v, 4) for v in pb), " rpy(度)", q_rpy_deg(qb))
    print("camera_link （相對模型）位置", tuple(round(v, 4) for v in pc), " rpy(度)", q_rpy_deg(qc))
    rel_p = q_rot(q_conj(qb), tuple(c - b for c, b in zip(pc, pb)))
    rel_q = q_mul(q_conj(qb), qc)
    print(">> camera_link 在 base_link 座標下：位置 (x前, y左, z上) =",
          tuple(round(v, 4) for v in rel_p), "m； rpy(度) =", q_rpy_deg(rel_q))
else:
    print("5 秒內沒收到 /world/default/pose/info 裡的 base_link / camera_link")
node.unsubscribe("/world/default/pose/info")

# ---------- Q1 ----------
print("\n===== Q1 ODOMETRY =====")
odo = []


async def collect_odo():
    try:
        async for o in d.drone.telemetry.odometry():
            odo.append(o)
    except asyncio.CancelledError:
        pass


fut = asyncio.run_coroutine_threadsafe(collect_odo(), d._loop)
time.sleep(3)
if not odo:
    print("3 秒內沒收到任何 odometry → PX4 沒送（或 MAVSDK 沒轉）")
else:
    t = [o.time_usec for o in odo]
    print(f"3 秒收到 {len(odo)} 筆，約 {len(odo) / 3:.1f} Hz；time_usec 是否遞增：{all(b > a for a, b in zip(t, t[1:]))}")
    o = odo[-1]
    s = d.get_state()
    print("frame_id =", o.frame_id, "，child_frame_id =", o.child_frame_id)
    print("time_usec =", o.time_usec, "（秒：", o.time_usec / 1e6, "）； 同時 get_state 的 sim_time =", s["sim_time"])
    print("position_body (x,y,z) =", (o.position_body.x_m, o.position_body.y_m, o.position_body.z_m),
          "； get_state NED =", (s["north_m"], s["east_m"], s["down_m"]))
    print("velocity_body (x,y,z) =", (o.velocity_body.x_m_s, o.velocity_body.y_m_s, o.velocity_body.z_m_s),
          "； get_state NED 速度 =", (s["vn_m_s"], s["ve_m_s"], s["vd_m_s"]))
    print("q → rpy(度) =", q_rpy_deg((o.q.w, o.q.x, o.q.y, o.q.z)),
          "； get_state rpy(度) =", (s["roll_deg"], s["pitch_deg"], s["yaw_deg"]))
    print("q.timestamp_us =", o.q.timestamp_us)

# ---------- Q2 ----------
print("\n===== Q2 時間差：固定偏移 or 取樣錯開 =====")
d.takeoff(5.0)
time.sleep(2)


def sample(label, secs=4.0):
    la, lo = [], []
    end = time.time() + secs
    while time.time() < end:
        ok, _ = cam.read(timeout=0.2)
        if not ok:
            continue
        st = cam.last_stamp
        la.append(d.get_state()["sim_time"] - st)
        if odo:
            lo.append(odo[-1].time_usec / 1e6 - st)
    with d._lock:
        hs = [h["stamp"] for h in d._hist]
    gaps = [b - a for a, b in zip(hs[-60:], hs[-59:])]
    print(f"[{label}] 姿態實際間隔中位數 {stats.median(gaps) * 1000:.1f} ms")
    print(f"[{label}] 最新姿態時間 − 畫面時間：{ms(la)}")
    if lo:
        print(f"[{label}] 最新 odometry 時間 − 畫面時間：{ms(lo)}")


sample("姿態 20 Hz")
d._run(d.drone.telemetry.set_rate_attitude_euler(100.0), 10)
time.sleep(1)
sample("姿態 100 Hz")

fut.cancel()
d.land()
d.close()
cam.release()
print("\n完成")
