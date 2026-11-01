"""多機觀測融合：反變異數加權（inverse-variance weighting）。

每個觀測的 uncertainty_m 是 DRMS，假設誤差在 n、e 兩軸獨立且相等，每軸變異數 = DRMS² / 2。
融合後每軸變異數 = 1 / Σ(1/σᵢ²)，所以：
    - 兩架同樣準的觀測，誤差縮成 1/√2
    - 一架準、一架不準，結果會靠近準的那一架
這是卡爾曼濾波「更新」步驟在靜態情況下的特例，也是 uncertainty_m 存在的意義：沒有它就無從加權。
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .visibility import Observation


@dataclass(frozen=True)
class FusedMeasurement:
    target_id: int
    n: float
    e: float
    uncertainty_m: float  # DRMS
    observed_by: tuple


def fuse(observations: Sequence[Observation]) -> List[FusedMeasurement]:
    """依 target_id 分組融合。uncertainty_m ≤ 0 的觀測視為無效（避免除以 0）。"""
    groups: Dict[int, List[Observation]] = {}
    for o in observations:
        if o.uncertainty_m > 0 and math.isfinite(o.uncertainty_m):
            groups.setdefault(o.target_id, []).append(o)
    out = []
    for tid, obs in sorted(groups.items()):
        w = [2.0 / (o.uncertainty_m**2) for o in obs]  # 1 / 每軸變異數
        wsum = sum(w)
        n = sum(wi * o.n for wi, o in zip(w, obs)) / wsum
        e = sum(wi * o.e for wi, o in zip(w, obs)) / wsum
        out.append(FusedMeasurement(tid, n, e, math.sqrt(2.0 / wsum), tuple(o.drone_id for o in obs)))
    return out
