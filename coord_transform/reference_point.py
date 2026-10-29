"""bbox → 用來反投影的參考像素與求交平面。

為什麼重要：bbox 框住的是「有高度的物體」，不是地上的一個點。選錯參考點造成的是系統偏差，
不會因為多平均幾幀而消失。analysis/error_study.py 的實驗 5 量化了三種方法：

    bottom_center        底邊中點，與地面 d = 0 求交（規格 §2.4 目前的做法）
                         適合斜視相機看站立的人（腳 ≈ 底邊）；nadir 看車會偏 1–3 m
    center               bbox 中心，與地面求交；物體頂部離相機較近，中心射線會打到物體後方
    center_half_height   bbox 中心，與「物體半高」平面求交，再垂直投影回地面
                         bbox 中心 ≈ 物體 3D 中心的投影，所以在半高平面求交幾乎無偏差；
                         代價是需要「各類別的典型高度」這個先驗
"""
from typing import Optional, Tuple

REFERENCE_METHODS = ("bottom_center", "center", "center_half_height")

# ASSUMED：各類別典型高度（m），只含物體本身（COCO 的 bicycle/motorcycle 不含騎士，騎士另被偵測為 person）。
# 誤差敏感度：高度先驗差 Δh，水平誤差約 (Δh/2)·tan(離軸角)，見 analysis/README.md。
CLASS_HEIGHT_M = {
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.2,
    "motorcycle": 1.1,
    "bicycle": 1.0,
    "person": 1.7,
}


def reference_point(
    bbox: Tuple[float, float, float, float],
    cls: str,
    method: str = "bottom_center",
    *,
    ground_d_m: float = 0.0,
    height_m: Optional[float] = None,
) -> Tuple[float, float, float]:
    """回傳 (u, v, plane_d_m)：反投影用的像素，以及要求交的平面（NED down，m）。

    height_m：覆寫類別預設高度（做敏感度分析用）。
    呼叫端拿到交點後，應把 d 設回 ground_d_m（目標的接地點）。
    """
    x1, y1, x2, y2 = bbox
    if method == "bottom_center":
        return (x1 + x2) / 2.0, y2, ground_d_m
    if method == "center":
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, ground_d_m
    if method == "center_half_height":
        h = CLASS_HEIGHT_M[cls] if height_m is None else height_m
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, ground_d_m - h / 2.0  # NED：往上是負的
    raise ValueError(f"未知的參考點方法 {method!r}，可用：{REFERENCE_METHODS}")
