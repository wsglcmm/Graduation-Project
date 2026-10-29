# 模擬紀錄格式（session）：給 A、B 的錄製規格

目的：先不做即時串接，A、B 各自把模擬跑出來的資料存檔，C 再用離線重播的方式，算出**真實誤差**並標定誤差模型。

- 重播：`python -m analysis.replay <session 資料夾>`
- 標定：`python -m analysis.calibrate_error_model <session 資料夾>`
- 範例：`python -m analysis.synth_session <輸出資料夾>` 可以產生一份合成的 session，格式完全相同，A、B 可以拿來對照

## 資料夾結構

建議放在專案的 `data/sessions/` 底下（`data/` 已在 `.gitignore`，紀錄檔不會被 commit）。

```
session_2026-10-21_run1/
├── meta.json          ← A 寫：座標原點與目標清單
├── poses.jsonl        ← A 寫：無人機姿態（PX4 / MAVSDK 估計值）
├── truth.jsonl        ← A 寫：Gazebo 真值（無人機與目標）
└── detections.jsonl   ← B 寫：偵測結果
```

`.jsonl` 是「一行一筆 JSON」：每收到一筆資料就 `json.dumps` 寫一行，不需要讀回整個檔案，程式中斷也不會壞掉。

```python
import json
with open("poses.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(msg) + "\n")
```

## 所有時間戳

`timestamp_ns`（int，奈秒），**一律是 Gazebo 模擬時間**。三個檔案用的必須是同一個時鐘，不得使用電腦本機時鐘（例如 `get_state()['time']`、`time.time()`）。

## poses.jsonl（A）

每行就是規格 §4 的 Pose 訊息，與即時串接時要送的格式一模一樣：

```json
{"timestamp_ns": 12345000000, "position_ned_m": {"n": 1.2, "e": -0.4, "d": -30.0}, "attitude_rad": {"roll": 0.01, "pitch": -0.03, "yaw": 1.57}}
```

- `timestamp_ns` = `att_timestamp_us × 1000`
- 角度單位是**弧度**（MAVSDK 的 `EulerAngle` 是度，要換算）
- 有 `quaternion_wxyz` 的話也一起寫

## detections.jsonl（B）

每行就是規格 §3 的 Detection 訊息，**沒有偵測到東西的影像也要寫一行**（`"detections": []`），重播才知道那一幀存在：

```json
{"schema_version": "0.1.0", "frame_id": 42, "timestamp_ns": 12345033000, "image_width": 1280, "image_height": 960, "detections": [{"track_id": 7, "class": "car", "confidence": 0.91, "bbox": {"x1": 600.5, "y1": 400.2, "x2": 700.1, "y2": 460.8}}]}
```

- `timestamp_ns` = Gazebo 影像訊息的擷取時間
- 影像大小必須是 Gazebo 相機的 1280×960，bbox 用 `result.boxes.xyxy`（原始影像像素）

## truth.jsonl（A）

Gazebo 的真值位姿，**直接用 Gazebo 世界座標（ENU）**，不要自己轉成 NED，轉換由 C 處理。無人機和每個目標各自一行，建議 ≥ 20 Hz：

```json
{"timestamp_ns": 12345000000, "name": "x500_0", "position_enu_m": {"x": 2.0, "y": -3.0, "z": 30.1}, "rpy_rad": {"roll": 0.01, "pitch": 0.03, "yaw": 0.0}}
{"timestamp_ns": 12345000000, "name": "car_1", "position_enu_m": {"x": 8.5, "y": 20.0, "z": 0.0}}
```

- `name`：Gazebo 的 model 名稱
- `rpy_rad`：無人機一定要有，因為要拿來標定姿態誤差；目標可以省略

## meta.json（A）

```json
{
  "format_version": 1,
  "ned_origin_enu_m": {"x": 2.0, "y": -3.0, "z": 0.24},
  "ground_z_enu_m": 0.0,
  "drone_name": "x500_0",
  "targets": {"car_1": "car", "person_1": "person"}
}
```

- `ned_origin_enu_m`：**PX4 local NED 原點（通常是起飛點）在 Gazebo 世界座標的位置**。最簡單的取法：起飛前、無人機還停在地上時，讀它的 Gazebo 位置。這個值錯了，所有誤差都會多出一個固定偏移，標定工具會用「偏差」欄位提醒你。
- `ground_z_enu_m`（選填）：地面在 Gazebo 的高度（平地通常是 0）。無人機停在地上時 base_link 比地面高一點，所以 NED 的地面其實在 d = 原點 z − 地面 z。重播工具會拿它和 `config/camera.yaml` 的 `plane_d_m` 比對，不一致時會提醒。
- `targets`：Gazebo model 名稱 → 類別，類別必須是規格 §3 的 6 種之一

## 建議錄製的情境（每段 30–60 秒）

1. 懸停 30 m，車在畫面中直線慢速通過：看基本誤差
2. 跟車飛行，有轉彎：看姿態變化與時間同步的影響
3. 不同高度（20、50 m）各一段：驗證誤差與高度成正比
