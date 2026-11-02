"""鳥瞰影片的地面標定：像素 (u, v) ↔ 地面座標 world_ned 的 (n, e)，單位 m（規格 §10.1、§10.3）。

為什麼用 homography：網路上的鳥瞰影片沒有無人機姿態與相機參數，無法走 pixel_to_ned 的完整相機模型。
但「平面 → 平面」的透視投影一定可以寫成一個 3×3 矩陣 H（8 個自由度），
所以只要在畫面上標出 ≥ 4 個已知地面位置的點，就能求出 H。

    [n, e, 1]ᵀ ∝ H · [u, v, 1]ᵀ

求解用正規化 DLT（Hartley 正規化 + SVD）：先把兩組點各自平移到原點、縮放到平均距離 √2，
避免像素（數百）與公尺（數十）數量級差太多造成數值不穩，這是 OpenCV findHomography 內部也做的事。
限制：只對「地面平面上的點」成立，無法和半高平面求交，所以影片偵測用 bbox 底邊中點當接地點（規格 §2.4 v0.2.2）。
斜角拍攝時底邊大致就是接地點；正上方拍攝的影片，車的底邊會對應車頂邊緣（見 analysis/README.md 實驗 5）。
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _normalize(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """回傳 (T, 正規化後的齊次點)：平移到重心、縮放到平均距離 √2。"""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    if d < 1e-12:
        raise ValueError("標定點全部重合")
    s = math.sqrt(2.0) / d
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    h = np.column_stack([pts, np.ones(len(pts))]) @ T.T
    return T, h


def fit_homography(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """由對應點求 H，使 dst ∝ H·src。至少 4 點，且不能有 3 點以上共線到退化。"""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src 與 dst 必須是相同數量的 (x, y) 點")
    if len(src) < 4:
        raise ValueError("至少需要 4 組對應點")
    Ts, s = _normalize(src)
    Td, d = _normalize(dst)
    rows = []
    for (x, y, _), (X, Y, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    _, sv, vt = np.linalg.svd(np.array(rows))
    if sv[-2] < 1e-9 * sv[0]:  # 第二小的奇異值也接近 0 → 解不唯一（點共線）
        raise ValueError("標定點退化（共線或太集中），請在畫面四周分散選點")
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _apply(H: np.ndarray, x: float, y: float) -> Point:
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-12:
        raise ValueError("點位在地平線上，無法對應到地面")
    return float(p[0] / p[2]), float(p[1] / p[2])


@dataclass(frozen=True)
class GroundHomography:
    """像素 → 地面 (n, e) 的 homography。H 為 3×3，H[2, 2] = 1。"""

    H: np.ndarray
    video_id: str = ""
    rms_error_m: Optional[float] = None  # 標定點的重投影誤差（地面公尺）

    @classmethod
    def fit(cls, pixel_pts: Sequence[Point], ground_ne_pts: Sequence[Point], video_id: str = "") -> "GroundHomography":
        H = fit_homography(pixel_pts, ground_ne_pts)
        g = cls(H, video_id)
        err = [math.dist(g.pixel_to_ground(*p), q) for p, q in zip(pixel_pts, ground_ne_pts)]
        rms = math.sqrt(sum(e * e for e in err) / len(err))
        return cls(H, video_id, rms)

    @classmethod
    def from_scale(cls, meters_per_pixel: float, origin_pixel: Point, north_up: bool = True, video_id: str = "") -> "GroundHomography":
        """正射俯拍、比例已知時的簡化版：畫面上方 = 北，origin_pixel 為 world 原點。"""
        if meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel 必須 > 0")
        if not north_up:
            raise NotImplementedError("非北朝上的影片請用 fit() 以標定點求解")
        s, (u0, v0) = meters_per_pixel, origin_pixel
        # n = -(v - v0)·s（影像往下 = 往南）、e = (u - u0)·s
        H = np.array([[0.0, -s, s * v0], [s, 0.0, -s * u0], [0.0, 0.0, 1.0]])
        return cls(H, video_id, 0.0)

    def pixel_to_ground(self, u: float, v: float) -> Point:
        """像素 (u, v)（px）→ 地面 (n, e)（m，world_ned）。"""
        return _apply(self.H, u, v)

    def ground_to_pixel(self, n: float, e: float) -> Point:
        """地面 (n, e)（m）→ 像素 (u, v)（px）。"""
        return _apply(np.linalg.inv(self.H), n, e)

    def jacobian(self, u: float, v: float) -> np.ndarray:
        """∂(n, e)/∂(u, v)，2×2，解析式（商法則）。"""
        H = self.H
        p = H @ np.array([u, v, 1.0])
        w = p[2]
        J = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                J[i, j] = (H[i, j] * w - p[i] * H[2, j]) / (w * w)
        return J

    def uncertainty_m(self, u: float, v: float, pixel_sigma_px: float) -> float:
        """像素抖動 σ（u、v 各自）造成的地面 1σ 誤差，DRMS（與規格 §5 定義一致）。"""
        J = self.jacobian(u, v)
        return float(math.sqrt(np.trace(J @ J.T)) * pixel_sigma_px)

    # ---------- 設定檔 ----------
    def to_dict(self) -> dict:
        return {"video_id": self.video_id, "H": self.H.tolist(), "rms_error_m": self.rms_error_m}


def load_video_ground(path) -> GroundHomography:
    """讀 config/video_ground*.json。兩種寫法擇一：

    1. 標定點：{"video_id": ..., "points": [{"pixel": [u, v], "ground_ne_m": [n, e]}, ...]}（≥ 4 點）
    2. 比例：{"video_id": ..., "meters_per_pixel": 0.05, "origin_pixel": [u, v]}（正射、北朝上）
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    vid = str(raw.get("video_id", ""))
    if "points" in raw:
        pts = raw["points"]
        return GroundHomography.fit([p["pixel"] for p in pts], [p["ground_ne_m"] for p in pts], vid)
    if "meters_per_pixel" in raw:
        return GroundHomography.from_scale(float(raw["meters_per_pixel"]), tuple(raw["origin_pixel"]), video_id=vid)
    raise ValueError("video_ground 設定需要 points 或 meters_per_pixel 其中一種")
