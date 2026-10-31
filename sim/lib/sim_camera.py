"""sim_camera：讀取 Gazebo 模擬鏡頭畫面，用法比照 cv2.VideoCapture。

    from sim_camera import SimCamera
    cam = SimCamera()
    ok, frame = cam.read()      # frame: BGR numpy (H, W, 3)
    t = cam.last_stamp          # 這張影像的模擬時間（秒）
    t_ns = cam.last_stamp_ns    # 同一個時間的整數奈秒（規格的 timestamp_ns）
    cam.release()

影像方向（下視鏡頭）：畫面上方 = 機頭前方，畫面右方 = 機體右方。
"""
import atexit
import subprocess
import threading
import time

import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node


def find_image_topic(keyword="imager/image", timeout=10.0):
    """從 `gz topic -l` 找出鏡頭影像 topic。"""
    end = time.time() + timeout
    while time.time() < end:
        out = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True).stdout
        for t in out.split():
            if t.endswith("/image") and keyword in t:
                return t
        time.sleep(0.5)
    raise TimeoutError("找不到 Gazebo 影像 topic，模擬是否已啟動？")


class SimCamera:
    def __init__(self, topic=None, timeout=10.0):
        self.topic = topic or find_image_topic(timeout=timeout)
        self._cv = threading.Condition()
        self._frame = None
        self._stamp = self._stamp_ns = None
        self._seq = 0          # 已收到的張數
        self._read_seq = 0     # 上次 read() 回傳到第幾張
        self.last_stamp = self.last_stamp_ns = None
        self.width = self.height = None
        self._node = Node()
        if not self._node.subscribe(Image, self.topic, self._on_image):
            raise RuntimeError(f"訂閱失敗：{self.topic}")
        atexit.register(self.release)  # 程式結束前自動釋放

    def _on_image(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        ch = step // w
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = buf[: h * step].reshape(h, step)[:, : w * ch].reshape(h, w, ch)
        if ch == 3:
            img = img[:, :, ::-1]            # RGB -> BGR
        elif ch == 4:
            img = img[:, :, 2::-1]           # RGBA -> BGR
        elif ch == 1:
            img = np.repeat(img, 3, axis=2)  # 灰階 -> 3 通道
        img = np.ascontiguousarray(img)
        stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
        with self._cv:
            self._frame, self._stamp, self._stamp_ns = img, stamp, stamp_ns
            self.width, self.height = w, h
            self._seq += 1
            self._cv.notify_all()

    def read(self, timeout=1.0):
        """等下一張「新」畫面；回傳 (ok, frame)。逾時回傳 (False, None)。"""
        with self._cv:
            if not self._cv.wait_for(lambda: self._seq > self._read_seq, timeout):
                return False, None
            self._read_seq = self._seq
            self.last_stamp, self.last_stamp_ns = self._stamp, self._stamp_ns
            return True, self._frame

    def isOpened(self):
        return not getattr(self, "_released", False) and self._seq > 0

    def release(self):
        """取消訂閱；可重複呼叫。"""
        if not getattr(self, "_released", False):
            self._released = True
            self._node.unsubscribe(self.topic)   # Node 物件保留到程式結束，不在這裡銷毀
