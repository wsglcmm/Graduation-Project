"""影片場景（場景二）：真實鳥瞰影片的目標當「兔子」，3 架無人機疊在影片上追（目標不會對無人機反應）。

流程：
    B 的偵測（§10.3，或 B 的原始格式）─ bbox 底邊中點 ─→ homography（config/video_ground*.json）─→ 地面 (n, e)
        ─→ 軌跡（同一個 track_id 的一串地面座標）─→ 選一條當目標 ─→ VideoTarget（介面同 RabbitPath / Evader）

track_id 的來源：
    - B 用 ByteTrack 輸出的 track_id（正式做法）。
    - B 還沒提供時（只有佔位 ID，每個偵測一個不重複的號碼），用 C 這邊的暫代關聯：在地面座標做最近鄰 +
      等速預測（gating 3 m、同類別群組），只為了先把流程接通。報告與評估一律以 B 的 ByteTrack ID 為準。

限制（報告要寫）：影片是錄好的，目標不會對無人機做出反應，所以這個場景只能展示預測攔截 → 抓拍，不展示驅趕。
座標：world_ned 的 (n, e)，m；時間：影片時間（s，從第一幀起算）。
"""
import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from coord_transform.homography import GroundHomography

PLACEHOLDER_ID_MIN = 1_000_000  # coord_transform.video_detections 的佔位 ID 從這裡開始
CLASS_GROUP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "person": "person",
               "motorcycle": "two_wheel", "bicycle": "two_wheel"}


@dataclass
class VideoTrack:
    track_id: int
    cls: str
    samples: List[Tuple[float, float, float]] = field(default_factory=list)  # (t_s, n, e)

    @property
    def duration_s(self) -> float:
        return self.samples[-1][0] - self.samples[0][0] if self.samples else 0.0

    def path_length_m(self) -> float:
        return sum(math.dist(a[1:], b[1:]) for a, b in zip(self.samples, self.samples[1:]))

    def moving_time_s(self, min_speed_mps: float = 0.5, window_s: float = 1.0) -> float:
        """真的在移動的時間：每 window_s 取一次位置，位移 ≥ min_speed_mps × window_s 的區間加總；停等紅燈、路邊停車不算。

        不能用相鄰兩幀算速度：50 fps 時 bbox 抖動 5 cm 就等於 2.5 m/s，停著的車也會被當成在動（road01 實測踩到）。"""
        if len(self.samples) < 2:
            return 0.0
        t0, t_end = self.samples[0][0], self.samples[-1][0]
        tot, t = 0.0, t0
        while t + window_s <= t_end + 1e-9:
            if math.dist(self.position(t), self.position(t + window_s)) >= min_speed_mps * window_s:
                tot += window_s
            t += window_s
        return tot

    def position(self, t: float) -> Tuple[float, float]:
        """t（影片時間 s）的位置，線性內插；超出範圍時停在端點。"""
        ts = [s[0] for s in self.samples]
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return self.samples[0][1:]
        if i >= len(ts):
            return self.samples[-1][1:]
        (t0, n0, e0), (t1, n1, e1) = self.samples[i - 1], self.samples[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return n0 + a * (n1 - n0), e0 + a * (e1 - e0)


def load_detection_frames(path) -> List[dict]:
    """讀偵測檔：§10.3 格式（有 track_id 或佔位 ID），或 B 的原始格式（自動轉換，沒有 track_id 時給佔位 ID）。"""
    from coord_transform.video_detections import convert_frame

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows and "frame_id" in rows[0] and "source" in rows[0]:
        return rows
    import itertools

    ids = itertools.count(PLACEHOLDER_ID_MIN)
    return [convert_frame(r, "video", ids) for r in rows]


def ground_of(det: dict, H: GroundHomography) -> Tuple[float, float]:
    """bbox 底邊中點 → 地面 (n, e)（規格 §2.4：影片 homography 用底邊中點）。"""
    b = det["bbox"]
    return H.pixel_to_ground((b["x1"] + b["x2"]) / 2, b["y2"])


def has_real_track_ids(frames: Sequence[dict]) -> bool:
    ids = [d["track_id"] for f in frames for d in f["detections"]]
    return bool(ids) and max(ids) < PLACEHOLDER_ID_MIN


def tracks_from_ids(frames: Sequence[dict], H: GroundHomography) -> Dict[int, VideoTrack]:
    """B 已經提供 ByteTrack track_id：直接依 ID 分組。"""
    t0 = frames[0]["timestamp_ns"] if frames else 0
    out: Dict[int, VideoTrack] = {}
    for f in frames:
        t = (f["timestamp_ns"] - t0) / 1e9
        for d in f["detections"]:
            n, e = ground_of(d, H)
            out.setdefault(d["track_id"], VideoTrack(d["track_id"], d["class"])).samples.append((t, n, e))
    return out


def associate(frames: Sequence[dict], H: GroundHomography, gate_m: float = 3.0, max_gap_s: float = 1.0) -> Dict[int, VideoTrack]:
    """C 的暫代關聯（B 還沒提供 track_id 時用）：地面座標最近鄰 + 等速預測，同類別群組內配對。

    每一幀依距離由小到大貪婪配對；配不到的偵測開新軌跡；超過 max_gap_s 沒更新的軌跡關閉。
    這只是讓流程先跑起來的簡化版，不處理遮擋後換 ID 等問題（那是 ByteTrack 的工作）。
    """
    t0 = frames[0]["timestamp_ns"] if frames else 0
    tracks: Dict[int, VideoTrack] = {}
    active: List[VideoTrack] = []
    next_id = 1
    for f in frames:
        t = (f["timestamp_ns"] - t0) / 1e9
        active = [tr for tr in active if t - tr.samples[-1][0] <= max_gap_s]
        dets = [(d, ground_of(d, H)) for d in f["detections"]]
        pairs = []
        for i, (d, (n, e)) in enumerate(dets):
            for j, tr in enumerate(active):
                if CLASS_GROUP.get(tr.cls) != CLASS_GROUP.get(d["class"]):
                    continue
                pn, pe = _predict(tr, t)
                dist = math.hypot(n - pn, e - pe)
                if dist <= gate_m:
                    pairs.append((dist, i, j))
        used_d, used_t = set(), set()
        for dist, i, j in sorted(pairs):
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            active[j].samples.append((t, *dets[i][1]))
        for i, (d, (n, e)) in enumerate(dets):
            if i not in used_d:
                tr = VideoTrack(next_id, d["class"], [(t, n, e)])
                next_id += 1
                tracks[tr.track_id] = tr
                active.append(tr)
    return tracks


def _predict(tr: VideoTrack, t: float) -> Tuple[float, float]:
    s = tr.samples
    if len(s) < 2 or s[-1][0] <= s[-2][0]:
        return s[-1][1:]
    dt = s[-1][0] - s[-2][0]
    vn, ve = (s[-1][1] - s[-2][1]) / dt, (s[-1][2] - s[-2][2]) / dt
    return s[-1][1] + vn * (t - s[-1][0]), s[-1][2] + ve * (t - s[-1][0])


def pick_target(tracks: Dict[int, VideoTrack], classes: Sequence[str] = ("car",), min_path_m: float = 10.0) -> VideoTrack:
    """自動選目標：指定類別、路徑 ≥ min_path_m 的軌跡中，「真的在移動的時間」最長的那一條。

    不用「持續最久」：road01 實測，持續最久的車有 3/4 的時間停著等（追一台停住的車看不出攔截）；
    也不用「路徑最長」：那通常是幾秒內穿過畫面的車，無人機來不及就位。"""
    cand = [t for t in tracks.values() if t.cls in classes and t.path_length_m() >= min_path_m]
    if not cand:
        raise ValueError(f"找不到符合條件的軌跡（類別 {classes}、移動 ≥ {min_path_m} m）")
    return max(cand, key=lambda t: (t.moving_time_s(), t.duration_s))


@dataclass(frozen=True)
class VideoTargetConfig:
    arena_radius_m: float  # 給畫面與捕捉區用：以 world 原點為中心、涵蓋整條軌跡
    start_s: float = 0.0  # 影片從第幾秒開始（通常是目標軌跡開始前一點）


class VideoTarget:
    """介面同 RabbitPath / Evader：屬性 n, e, cfg.arena_radius_m；step(dt, drones) 回傳 (n, e)。目標不理無人機。"""

    def __init__(self, track: VideoTrack, start_s: Optional[float] = None, video_id: str = ""):
        self.track = track
        self.video_id = video_id
        r = max(math.hypot(n, e) for _, n, e in track.samples) + 10.0
        st = track.samples[0][0] - 2.0 if start_s is None else start_s
        self.cfg = VideoTargetConfig(arena_radius_m=r, start_s=max(0.0, st))
        self.t = self.cfg.start_s  # 影片時間（s）
        self.n, self.e = track.position(self.t)

    @property
    def video_time_s(self) -> float:
        return self.t

    @property
    def finished(self) -> bool:
        return self.t > self.track.samples[-1][0]

    def step(self, dt: float, drones=None) -> Tuple[float, float]:
        self.t += dt
        self.n, self.e = self.track.position(self.t)
        return self.n, self.e


def load_video_target(detections_path, ground: GroundHomography, track_id: Optional[int] = None,
                      classes: Sequence[str] = ("car",)) -> Tuple[VideoTarget, Dict[int, VideoTrack], bool]:
    """回傳（目標、全部軌跡、是否用了 C 的暫代關聯）。"""
    frames = load_detection_frames(detections_path)
    real = has_real_track_ids(frames)
    tracks = tracks_from_ids(frames, ground) if real else associate(frames, ground)
    tr = tracks[track_id] if track_id is not None else pick_target(tracks, classes)
    return VideoTarget(tr, video_id=ground.video_id), tracks, not real
