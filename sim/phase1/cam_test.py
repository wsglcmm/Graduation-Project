import time
import cv2
from sim_camera import SimCamera

cam = SimCamera()
print("topic:", cam.topic)
n, t0, s0, frame = 0, time.time(), None, None
while time.time() - t0 < 5:
    ok, f = cam.read()
    if ok:
        n += 1
        frame = f
        s0 = cam.last_stamp if s0 is None else s0
wall = time.time() - t0
print(f"5 秒收到 {n} 張，約 {n / wall:.1f} fps，shape={frame.shape}，dtype={frame.dtype}")
print(f"模擬時間 {s0:.2f} -> {cam.last_stamp:.2f} s（模擬/真實 速度比約 {(cam.last_stamp - s0) / wall:.2f}）")
out = "/mnt/2disk/vicelead/tmp/cam_snapshot.jpg"
cv2.imwrite(out, frame)
print("已存：", out)
cam.release()
