# 模組介面規格（Interface Specification）

- 版本：`0.2.5`（§10.5 新增捕捉策略 `capture` 模式與 `herder` / `capturer` / `blocker` 角色；新增 §10.9 各機 Agent 動作、§10.10 抓拍判定、§10.11 動作日誌；§10.3 影片場景的 track_id 與地面標定。以上都是 C 內部格式，A、B 的訊息沒有變動。v0.2.4：§10.8 新增 A → C 執行狀態：flight_command、ActionResult、Yield；§10.2 / §10.6 時間戳的時鐘；§10.6 速度上限改為啟動參數、地面站失聯行為；§10.6 機間安全距離改寫：3D ≥ 5 m、目標點推開 6 m；§2.4 參考點改為中心 + 半高平面；§10 多機與影片 Demo 介面：A 已於 2026-11-01 合併 PR #3，同意 §10.1 local ↔ world 由 A 轉換、§10.6 目標點逾時與安全距離；§10.3 影片 Detection 待 B 確認）
- §3–§7 的單機訊息 `schema_version` 仍為 `"0.1.0"`；§10 的新訊息為 `"0.2.0"`
- 維護者：C
- 適用範圍：B（影像辨識）→ C（座標轉換 / LLM Agent）→ A（模擬 / 飛控）

> 本文件是三人整合的唯一依據。任何欄位、單位、座標系的更動，都必須開 PR 並經 A、B 同意後才能合併。

---

## 1. 系統資料流

```
[Gazebo 相機] → B: YOLOv8 + ByteTrack ──(介面 1: Detection)──┐
                                                              ▼
[PX4 SITL] → A: 姿態/位置 API ───────(介面 2: Pose)──────→ C: 座標轉換
                                                              │
                          (介面 3: TargetWorld) ◄─────────────┘
                                    │
                                    ▼
                    C: LLM Agent ──(介面 4: Action)──→ A: MAVSDK 控制
```

最終 Demo（影片 + 模擬混合，介面見 §10）：

```
[鳥瞰影片] → B: 偵測 / 追蹤 ──(§10.3 影片 Detection)──→ C: 地面標定 + 各機視野 + 融合 + 卡爾曼
                                                              │ (§10.4 FusedTarget)
[3 架 PX4 SITL] → A ──(§10.2 FleetPose, world_ned)──→ C ──────┤
                                                              ▼
                       C: Agent（§10.5 Plan，0.5–2 Hz）→ C: 隊形規劃（10 Hz）
                                                              │ (§10.6 GoalPoint)
                                                              ▼
                                                A: 各機目標點追蹤 + 安全層
```

---

## 2. 全域慣例（一次寫死，不得各自解釋）

### 2.1 座標系

| 名稱 | 定義 | 用途 |
|---|---|---|
| **世界座標 NED** | 原點 = 起飛點；x = North、y = East、z = Down（向下為正） | 所有位置輸出（與 PX4 local NED 一致） |
| **機體座標 FRD** | x = Forward、y = Right、z = Down | 姿態、相機安裝外參 |
| **相機座標（OpenCV 慣例）** | x = 影像右、y = 影像下、z = 光軸向前 | 內參 K、像素反投影 |
| **像素座標** | 原點 = 影像左上角；u 向右、v 向下；單位 px | bbox |

> ⚠️ **常見陷阱**：Gazebo / SDF 的相機座標是 x 前、y 左、z 上，與 OpenCV 不同。轉換由 C 負責處理，A、B 只需確認輸出的影像是「正常方向的圖片」。
>
> ⚠️ **Gazebo 世界座標是 ENU**（x 東、y 北、z 上；yaw 0 = 朝東、逆時針為正），與 NED 差 90° 且轉向相反（A 已實測確認）。
> 所有給 C 的位姿一律用 PX4 / MAVSDK 的 NED。若需要 Gazebo 真值（例如標定誤差），由 C 用 `coord_transform/frames.py` 的 `gazebo_pose_to_ned` 轉換。

### 2.2 單位

| 量 | 單位 |
|---|---|
| 長度 / 位置 | 公尺（m） |
| 角度 | 弧度（rad），範圍 [-π, π] |
| 速度 | m/s |
| 像素 | px（浮點數） |
| 時間戳 | `timestamp_ns`：int64，奈秒；**統一使用模擬時鐘（sim time）**，不得混用系統時間。A 的 `att_timestamp_us`（微秒）× 1000 即可；`get_state()['time']` 是本機時鐘，**不得**用於對時 |

### 2.3 時間同步約定

- 影像與姿態的取樣頻率不同（實測影像 30 Hz、姿態約 100 Hz，見下方），**由 C 負責對齊**：對影像時間戳在姿態序列上做插值（位置線性插值、姿態用 slerp）。
- 偵測到達時可能還沒有「之後」的姿態：C 會先讓該幀排隊，等下一筆姿態到了再內插，不做外插。
- 相鄰兩筆姿態相隔超過 150 ms（連續掉 2 筆以上）時不插值，回傳 `no_pose`。
- **位置與姿態必須是同一時刻**：A 的 `flight_api` 已改用 MAVSDK `telemetry.odometry()`（2026-10-31 實測 99.3 Hz，間隔 8–12 ms），位置、四元數、速度、角速度共用同一個 `time_usec`，而且 `time_usec` 就是 sim time。注意：MAVSDK 回報的 `frame_id` / `child_frame_id` 是 `UNDEF`，但 `position_body`、`velocity_body` 實際是 local NED（已用位置差分驗證）；`q.timestamp_us` 恆為 0，不要使用。
- **先前回報的「相機與姿態同一刻差 24 ms」不是固定偏移**：那是舊的 20 Hz 姿態串流取樣錯開造成的（半個間隔 ≈ 25 ms）。改用 odometry 後，「讀到畫面當下最新 odometry − 畫面時間」平均 14 ms（4–36 ms），這是畫面送到程式所需的時間，依畫面 `timestamp_ns` 在姿態序列上內插即可消除。`frame_time_offset_ms` 建議設 0；10/28 的 SITL 紀錄在 30°/s 旋轉段估得的殘差約 16 ms，在 yaw 雜訊範圍內。
- **A 必須提供帶時間戳的姿態序列**，並保留至少最近 2 秒的緩衝，不可只提供「最新一筆」。
- **B 的輸出時間戳 = 該張影像的擷取時間**，不是推論完成時間。
- 若姿態緩衝找不到涵蓋該影像時間戳的區間，C 回傳 `status = "no_pose"`，不輸出座標。

### 2.4 地面平面假設（目前版本）

- 假設目標位於地面平面 `d = 0`（NED 的 down = 0，即與起飛點同高）。
- 需要 A 提供「相對地面高度」（`altitude_agl_m`）。
- bbox 的參考點（相機模型流程，§5）：**中心 + 半高平面**（`center_half_height`）。取 bbox 中心 `(u, v) = ((x1+x2)/2, (y1+y2)/2)` 反投影，射線與「物體半高」的平面 `d = −h/2` 求交，再垂直投影回地面 `d = 0`。`h` 是各類別的典型高度（C 內部的先驗值，見 `coord_transform/reference_point.py`：car 1.5 m、person 1.7 m、truck 3.0 m、bus 3.2 m、motorcycle 1.1 m、bicycle 1.0 m）。
  - 理由（Phase 2 誤差研究實驗 5）：bbox 框住的是有高度的物體。nadir 相機看車時，bbox 底邊對應的是車頂邊緣而不是著地點，舊的「底邊中點」最多偏 **3.4 m**（平均 2.1 m）；改用中心 + 半高平面降到 **0.1 m 以下**，即使高度先驗錯 30% 仍小於 0.45 m。
  - 舊方法「底邊中點」`((x1+x2)/2, y2)` 保留為選項（`config/pipeline.yaml` 的 `reference_point: bottom_center`），適合斜視相機看站立的人。
  - **影片 Demo（§10.3）不適用**：homography 只對應地面平面，無法和半高平面求交，影片偵測維持用 bbox 底邊中點當接地點。
  - B 的輸出格式不受影響：參考點是 C 內部的計算方式。
- 已知誤差來源：地形起伏、姿態/時間戳誤差、bbox 抖動、類別高度先驗與實際高度的差異（差 Δh 時水平誤差約 (Δh/2)·tanθ）。
- SITL 實測（A，2026-10-28 紀錄）：**EKF yaw 比 Gazebo 真值大 +1° 到 +5.4°**，飛行中會漂移；roll、pitch 誤差 < 0.1°。停在地上時 EKF yaw 已經是約 96°，而那段紀錄開頭的真值約 90°，所以偏差很可能從開機就存在（待同一次執行中直接比對確認）。目前看起來是最大的姿態誤差來源，原因調查中。
- **同一次執行比對確認（A 的 yawcheck 紀錄，2026-11-01）**：地上未起飛時 EKF yaw 就比真值大 +4.37°；起飛、爬升、懸停、原地轉一圈（10°/s）都維持 +3.9° 到 +4.6°；直線前飛 2 m/s 約 15 s 後收斂到 +0.1°。偏差從開機就存在，只有水平加速才會修正。
- **原因（A 查明，2026-11-01）**：地面 yaw 誤差 = PX4 用的磁偏角（地磁模型 +3.41°）− 模擬磁力計實際量到的磁偏角；後者是 Gazebo 磁場本身的 −2.56°（固定差 6.0°），再加上 PX4 磁力計自動校正累積的偏移（約 +1.1°，每次飛完都可能變，所以不同次量到 4.4°～7.2°）。每段錄影內偏差固定。
- **處理方式（待 A 確認後實施）**：建議 SITL 設 `EKF2_DECL_TYPE = 0`、`EKF2_MAG_DECL` = Gazebo 的磁偏角，並關閉磁力計自動校正，使偏差接近 0；A 改參數前會先通知 C，改完錄一段短 yawcheck 驗證。修正前 C 的 `error_model.yaml` 以 RMS 4.1° 計入 yaw 誤差，並建議起飛後先直線飛約 15 s 再開始追蹤。

---

## 3. 介面 1：B → C　Detection（偵測輸出）

- 頻率：與影像同步（Gazebo 相機 30 Hz）
- 每個影像一則訊息，內含零到多個偵測
- B 目前以**靜態照片**測試（640×640，推論約 7.4 ms）。改接 Gazebo 相機後須符合：
  - `image_width` / `image_height` = 實際影像大小（Gazebo 為 **1280×960**），取自 Ultralytics 的 `result.orig_shape`；C 會檢查它與 `config/camera.yaml` 一致，不一致就拒收
  - bbox 用 `result.boxes.xyxy`（原始影像像素），**不是**模型輸入（640 letterbox）的座標
  - `timestamp_ns` = Gazebo 影像訊息的擷取時間（sim time）；照片沒有 sim time，無法與姿態配對
  - 延遲請回報端到端時間（前處理 + 推論 + 後處理 + ByteTrack），不只推論

### 欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `schema_version` | string | 本文件版本 |
| `frame_id` | int | 影像序號，單調遞增 |
| `timestamp_ns` | int64 | 影像擷取時間 |
| `image_width` / `image_height` | int | 影像解析度（px） |
| `detections[]` | array | 偵測結果 |
| `detections[].track_id` | int | ByteTrack 的追蹤 ID；同一目標在連續幀須保持一致 |
| `detections[].class` | string | `car` / `person` / `truck` / `bus` / `motorcycle` / `bicycle` |
| `detections[].confidence` | float | 0–1 |
| `detections[].bbox` | object | `x1, y1, x2, y2`（px，左上與右下，原始影像座標，未縮放） |

### 範例

```json
{
  "schema_version": "0.1.0",
  "frame_id": 1042,
  "timestamp_ns": 1730000000123456789,
  "image_width": 1280,
  "image_height": 960,
  "detections": [
    {
      "track_id": 7,
      "class": "car",
      "confidence": 0.91,
      "bbox": { "x1": 512.4, "y1": 300.1, "x2": 640.8, "y2": 388.9 }
    }
  ]
}
```

### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "DetectionFrame",
  "type": "object",
  "required": ["schema_version", "frame_id", "timestamp_ns", "image_width", "image_height", "detections"],
  "properties": {
    "schema_version": { "type": "string" },
    "frame_id": { "type": "integer", "minimum": 0 },
    "timestamp_ns": { "type": "integer" },
    "image_width": { "type": "integer", "minimum": 1 },
    "image_height": { "type": "integer", "minimum": 1 },
    "detections": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["track_id", "class", "confidence", "bbox"],
        "properties": {
          "track_id": { "type": "integer", "minimum": 0 },
          "class": { "enum": ["car", "person", "truck", "bus", "motorcycle", "bicycle"] },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "bbox": {
            "type": "object",
            "required": ["x1", "y1", "x2", "y2"],
            "properties": {
              "x1": { "type": "number" }, "y1": { "type": "number" },
              "x2": { "type": "number" }, "y2": { "type": "number" }
            }
          }
        }
      }
    }
  }
}
```

---

## 4. 介面 2：A → C　Pose（無人機姿態）

- 頻率：實測約 100 Hz（MAVSDK `odometry()`，見 §2.3）
- 以時間序列提供（見 2.3）
- 來源對照（A 的 `flight_api`）：`timestamp_ns` = `odom_time_usec × 1000`；`position_ned_m` = `north_m / east_m / down_m`；`attitude_rad` = `roll_deg / pitch_deg / yaw_deg` × π/180；`velocity_ned_mps` = `vn_m_s / ve_m_s / vd_m_s`
- **SITL 參數 `EKF2_GPS_DELAY` 必須設為 0**（A 查明，2026-11-01）：預設 110 ms 是給實體 GPS 的延遲補償，Gazebo 的 GPS 沒有延遲，保留預設值會讓 EKF 位置比實際狀態超前（實測約 0.1 s，速度 2.5 m/s 時約 0.23 m），姿態則不受影響。A 已改為 0（超前降到 0.005 s）。之前錄的資料由 C 以「位置 − 速度 × 0.095 s」補償（`analysis` 的 `--position-lead-s`）。

### 欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `timestamp_ns` | int64 | 取樣時間（sim time）= `att_timestamp_us × 1000`；位置須為同一時刻（見 §2.3） |
| `position_ned_m` | object | `n, e, d`；d 向下為正（飛在空中時 d 為負） |
| `attitude_rad` | object | `roll, pitch, yaw`（機體 FRD 相對 NED，yaw 0 = 朝北、90° = 朝東，範圍 ±π）。**單位是弧度**：MAVSDK 的 `EulerAngle` 是度，需換算 |
| `quaternion_wxyz` | array[4] | 選填，建議提供，方便 slerp 插值。若同時提供，C 會檢查它與 `attitude_rad` 相差不超過 1°，不一致時拒收（多半是 ENU/NED 混用） |
| `altitude_agl_m` | float | 相對地面高度（m）；平坦地面下 = `-d` |
| `velocity_ned_mps` | object | 選填，`n, e, d` |
| `gimbal_rad` | object | 若有雲台：`pitch, yaw`（相對機體）；固定相機則為安裝角，見 §6 |

### 範例

```json
{
  "timestamp_ns": 1730000000120000000,
  "position_ned_m": { "n": 12.31, "e": -4.07, "d": -30.0 },
  "attitude_rad": { "roll": 0.012, "pitch": -0.035, "yaw": 1.571 },
  "quaternion_wxyz": [0.7071, 0.0044, -0.0123, 0.7070],
  "altitude_agl_m": 30.0,
  "velocity_ned_mps": { "n": 2.1, "e": 0.0, "d": 0.0 },
  "gimbal_rad": { "pitch": -1.047, "yaw": 0.0 }
}
```

### A 的 `get_state()` 實測輸出（2026-10-31，無人機停在地上）

角度是**度**、角速度是 deg/s、位置與速度是 NED（m、m/s）、`sim_time` 是秒。`time` 是本機時鐘，**不得**用於對時。

```json
{
  "roll_deg": -0.0037, "pitch_deg": 0.0116, "yaw_deg": 95.9722,
  "rollspeed_deg_s": -0.0066, "pitchspeed_deg_s": -0.0146, "yawspeed_deg_s": -0.0275,
  "north_m": 0.0476, "east_m": -0.0539, "down_m": 0.0201, "height_m": -0.0201,
  "vn_m_s": 0.0087, "ve_m_s": -0.0055, "vd_m_s": 0.0059,
  "odom_time_usec": 324516000, "att_timestamp_us": 324516000, "sim_time": 324.516,
  "lat_deg": 47.3979712, "lon_deg": 8.5461632, "abs_alt_m": 0.208, "rel_alt_m": -0.027,
  "armed": false, "flight_mode": "HOLD", "in_air": false, "offboard": false,
  "time": 1793458809.2691777
}
```

---

## 5. 介面 3：C → A / Agent　TargetWorld（目標世界座標）

- 頻率：與 Detection 同步
- 一則訊息對應一個影像幀，內含各 track 的世界座標

### 欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `frame_id` | int | 對應的影像序號 |
| `timestamp_ns` | int64 | 對應影像時間（非輸出時間） |
| `targets[].track_id` | int | 同 Detection |
| `targets[].class` | string | 同 Detection |
| `targets[].status` | string | `ok` / `no_pose` / `above_horizon` / `out_of_range` |
| `targets[].position_ned_m` | object | `n, e, d`；平面假設下 `d = 0`；status 非 ok 時為 `null` |
| `targets[].uncertainty_m` | float / null | 1σ 水平誤差估計，定義為 DRMS = √(σ_n² + σ_e²)，單位 m；由像素、姿態、高度、機體水平位置的隨機誤差以一階傳播計算（σ 見 `config/error_model.yaml`），**不含**系統偏差（時間不同步、地形、參考點）。status 非 ok 時為 `null` |

### 範例

```json
{
  "schema_version": "0.1.0",
  "frame_id": 1042,
  "timestamp_ns": 1730000000123456789,
  "targets": [
    {
      "track_id": 7,
      "class": "car",
      "status": "ok",
      "position_ned_m": { "n": 41.8, "e": 6.2, "d": 0.0 },
      "uncertainty_m": null
    }
  ]
}
```

### 像素 → 世界座標（C 內部實作摘要，供 A、B 理解）

1. 取 bbox 中心 `(u, v)`（見 §2.4），反投影成相機座標射線：`d_c = K⁻¹ · [u, v, 1]ᵀ`
2. 相機 → 機體：`d_b = R_bc · d_c`（R_bc 為安裝外參，見 §6）
3. 機體 → NED：`d_n = R_nb(roll, pitch, yaw) · d_b`
4. 與物體半高平面 `d = −h/2` 求交：`λ = (−h/2 − p_d) / d_n[d]`（需 `d_n[d] > 0`，否則射線朝天，status = `above_horizon`）
5. `P_ned = p_ned + λ · d_n`，再把 `d` 設回 0（垂直投影到地面，即目標接地點）

---

## 6. 相機參數（C 維護，A / B 確認）

| 項目 | 內容 |
|---|---|
| 解析度 | **1280×960**（A 確認） |
| 影像頻率 | 30 Hz |
| 水平視角 HFOV | **1.74 rad（約 99.7°）**（SDF `horizontal_fov`） |
| 內參 K | 由 HFOV 算：`fx = (W/2) / tan(HFOV/2) ≈ 539.9`，`fy = fx`，`cx = 640`，`cy = 480` |
| 畸變 | 0（SDF 未設 distortion） |
| 安裝外參 | SDF `<pose>0 0 .10 0 1.5707 0</pose>` → 光軸朝正下方（nadir），影像上方 = 機頭方向；換算見 `coord_transform/frames.py` |
| 位置偏移 | **機體 FRD (0, 0, −0.10) m，相對 Gazebo 模型原點**（A 於 2026-10-31 由 Gazebo pose 實測：camera_link 在模型原點上方 0.10 m，水平 0；base_link 在模型原點上方 0.24 m，所以 camera_link 在 base_link **下方** 0.14 m）。mono_cam 內部的 sensor 可能另有沿光軸 0.014 m 的偏移，待確認 |

> 設定檔：`config/camera.yaml`。
>
> ⚠️ **相機偏移和地面平面必須使用同一個參考點**，混用會造成 0.24 m 的高度誤差（離軸 40° 時，地面誤差約 0.2 m）。PX4 EKF 追蹤的是 base_link（IMU），local NED 原點是開機時 base_link 的位置；但只要機體接近水平，EKF 的位移就等於模型原點的位移。兩套一致的約定：
>
> | 參考點 | 相機偏移（FRD） | 地面平面 d | 適用 |
> |---|---|---|---|
> | **模型原點（建議）** | (0, 0, −0.10) | `ned_origin_enu_m.z − ground_z_enu_m` ≈ −0.013（≈ 0） | 和模擬紀錄的 `ned_origin_enu_m`（讀模型 pose）一致；`camera.yaml` 現有的 `offset_body_m` 與 `plane_d_m: 0` 就屬於這套 |
> | base_link | (0, 0, +0.14) | ≈ +0.227 | 只有直接使用 base_link 真值時才用 |
>
> 兩套算出的相機離地高度都是「EKF 高度 + 0.087 m」。

---

## 7. 介面 4：C（Agent）→ A　Action（高階動作指令）

LLM 不直接輸出底層控制量，只能從下列**白名單動作**中選擇，並輸出 JSON。C 會先做 schema 驗證，不合法的輸出一律丟棄、不轉給 A。

### 動作白名單（初版）

| action | params | 說明 |
|---|---|---|
| `follow` | `track_id`: int、`distance_m`: float、`altitude_m`: float | 追蹤指定目標，保持距離與高度 |
| `hover` | 無 | 原地懸停 |
| `search` | `pattern`: `"orbit"` / `"lawnmower"`、`center_ned_m`、`radius_m` | 目標丟失時的搜尋模式 |
| `set_altitude` | `altitude_m`: float | 調整飛行高度 |
| `return_home` | 無 | 返航 |

### 範例

```json
{
  "schema_version": "0.1.0",
  "request_id": "a1b2c3",
  "action": "follow",
  "params": { "track_id": 7, "distance_m": 15.0, "altitude_m": 30.0 }
}
```

### A 的回應（ack）

```json
{
  "request_id": "a1b2c3",
  "accepted": true,
  "reason": null
}
```

### 安全責任劃分

- **安全限制由 A 擁有且最終生效**：最大速度、最大/最小高度、地理圍欄、電量保護、失聯 fallback。
- C 的 Agent 不得覆寫這些限制；超出限制的動作由 A 回覆 `accepted: false` 並附 `reason`。
- Agent 的角色定位（只做任務層決策，還是涉入即時控制）尚待老師討論；**預設為只做任務層決策**，即時追蹤控制由 A 的控制迴路負責。

---

## 8. 傳輸方式（待討論）

| 方案 | 優點 | 缺點 |
|---|---|---|
| Python 模組函式呼叫 + dataclass | 最簡單，開發最快 | A、B、C 需在同一個程式中 |
| ZeroMQ / socket 傳 JSON | 模組獨立、語言無關 | 需自行處理序列化與重連 |
| ROS 2 topic | 業界主流，PX4 生態相容 | 學習成本較高 |

**C 的建議**：先以 Python dataclass / pydantic 定義在 `shared/` 作為單一事實來源，Phase 3 前再決定是否升級為 ROS 2 topic。

---

## 9. 待確認事項

- [x] （A）Gazebo 相機的解析度、HFOV、安裝角度（2026-10-19）
- [x] （A）camera_link 到 base_link 的精確位移：在 base_link 下方 0.14 m、模型原點上方 0.10 m，水平 0（2026-10-31，見 §6）
- [x] （A）姿態資料的實際更新頻率：改用 odometry 後約 100 Hz（2026-10-31，見 §2.3）
- [x] （A）位置的時間戳：已改用 MAVSDK `odometry()`，位置與姿態共用 `time_usec`（2026-10-31，見 §2.3）
- [x] （A）「相機與姿態同一刻差 24 ms」：取樣錯開造成，不是固定偏移（2026-10-31，見 §2.3）
- [x] （A）提供一筆 `get_state()` 的實際輸出範例（2026-10-31，見 §4）
- [x] （A）模擬的時鐘來源：相機與姿態都使用 Gazebo 模擬時間
- [x] （B）推論延遲：約 7.4 ms（靜態照片，僅推論）
- [ ] （B）改接 Gazebo 相機影像（1280×960、sim time 時間戳）
- [ ] （B）確認 bbox 為原始影像像素（`boxes.xyxy`），並回報端到端延遲
- [ ] （B）class 列表是否與本文件一致
- [ ] （A）§10.1：由 A 做 local ↔ world 轉換，並提供各機 `origin_world_ned_m`
- [ ] （A）§10.6：各機目標點追蹤、1 秒逾時懸停、最小機間距離與高度分層的數值
- [ ] （B）§10.3：影片 Detection 的 `source`、`video_id` 欄位與影片時間戳
- [ ] （全員）§10.5：Agent 角色清單是否足夠
- [ ] （全員）傳輸方式
- [ ] （全員）Agent 的角色定位（老師確認）


## 10. 多機協同與影片 Demo（v0.2.0 新增，待 A、B 確認）

設計原則：**LLM 只決定「誰扮演什麼角色、用什麼隊形」**；「每架飛到哪裡」由 C 的確定性隊形規劃器算出各機目標點；「怎麼安全地飛過去」由 A 的飛控與安全層負責。LLM 失效時，系統沿用上一個合法計畫或懸停。

### 10.1 共同世界座標 `world_ned`

- 多機時每架 PX4 的 local NED 原點都是自己的起飛點，三架互不相同。**跨模組的多機訊息一律使用共同的 `world_ned`**：軸向與 §2.1 的 NED 相同（n 北、e 東、d 下），原點為 Demo 場景的地面原點。
- **由 A 負責 local ↔ world 轉換**：A 知道每架的起飛位置 `origin_world_ned_m`，送出位姿前加上、收到目標點後減去。轉換只在一個地方做，C 與 Agent 只看到單一座標系。
- 影片 Demo：`world_ned` 就是影片地面標定（homography）的地面座標系，原點與方向由 `config/video_ground.json` 的標定點決定；SITL 在空曠的 Gazebo 世界飛行，位置直接疊到影片上。
- Gazebo 技術驗證：`world_ned` = Gazebo 世界原點轉成 NED（ENU → NED 見 §2.1）。

### 10.2 `drone_id` 與 FleetPose（A → C，每架約 10–100 Hz）

`drone_id` 為字串：`"uav1"`、`"uav2"`、`"uav3"`。

| 欄位 | 型別 | 說明 |
|---|---|---|
| `schema_version` | string | `"0.2.0"` |
| `drone_id` | string | 哪一架 |
| `timestamp_ns` | int64 | PX4 的時間（SITL 跟 Gazebo 模擬時鐘走，是「模擬開始後經過的時間」，不是 Unix 時間），見下方說明 |
| `position_world_ned_m` | object | `n, e, d`（world_ned） |
| `attitude_rad` | object | `roll, pitch, yaw`（同 §4） |
| `velocity_world_ned_mps` | object | 選填，`n, e, d` |
| `status` | string | `"landed"` / `"flying"` / `"failsafe"` |

**時間戳（A 2026-11-18 實測）**：SITL 中三架的 `timestamp_ns` 是同一個時鐘（Gazebo 模擬時鐘），比對 752 次同一輪推播，三架相差中位數 8 ms、最多 16 ms，可以直接跨機比較與內插。例：`80544000000` = 模擬第 80.544 s。實機上每架 PX4 從自己開機開始計時，到時候必須另外換算成共同時間。

### 10.3 影片來源 Detection（B → C）

格式同 §3，另加：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `source` | string | `"video"`（影片 Demo）或 `"gazebo"`（技術驗證，§3 的原始用法） |
| `video_id` | string | 影片檔名或資料集場景代號（`source = "video"` 時必填） |
| `drone_id` | string | 選填；Gazebo 多機時表示哪一架的相機 |

- `source = "video"` 時，`timestamp_ns` = **影片時間**（從影片開頭起算的奈秒），`image_width / image_height` = 影片解析度。影片時間與模擬時間的對應由 C 的 Demo 播放器負責。
- 影片偵測由 C 用地面標定（homography）換成 world 座標，不使用 §5 的相機模型。
- **影片場景需要 `track_id`**（2026-11-18 補充）：同一台車在整支影片中要維持同一個 `track_id`（B 的 ByteTrack 輸出）。C 用它把每一幀的地面座標串成一條軌跡，再選一條當 Demo 目標。B 還沒提供時，C 會先用地面座標最近鄰做暫代關聯（只為了接通流程，畫面會標示；報告與評估以 B 的 `track_id` 為準）。
- 影片場景的地面參考點：bbox **底邊中點**（§2.4）。
- 地面標定檔（C 維護，`config/video_ground_<video_id>.json`）：`{"video_id": "...", "points": [{"pixel": [u, v], "ground_ne_m": [n, e]}, ...]}`，至少 4 點（建議 ≥ 6 點、分散在畫面四角，才能用重投影 RMS 檢查標錯），由 C 的標定工具產生。

### 10.4 FusedTarget（C 內部 → Agent / 面板）

C 依各機視野判斷誰看得到目標，以 `uncertainty_m` 加權融合，再經卡爾曼濾波：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `target_id` | int | 全域目標編號（跨無人機） |
| `class` | string | 同 §3 |
| `status` | string | `"tracking"`（本幀有觀測）/ `"predicted"`（暫時沒看到，用模型外推）/ `"lost"` |
| `position_world_ned_m` | object | `n, e` |
| `velocity_world_ned_mps` | object | `n, e`（卡爾曼估計） |
| `uncertainty_m` | float | 1σ 水平誤差（DRMS，定義同 §5） |
| `observed_by` | array[string] | 本幀看得到目標的 `drone_id` |
| `last_seen_ns` | int64 | 最後一次有觀測的時間 |

### 10.5 Agent 多機計畫 Plan（LLM 輸出，C 驗證後使用）

LLM 只輸出下列 JSON，C 以 pydantic strict schema 驗證，不合法就丟棄並沿用上一個計畫：

```json
{
  "mode": "encircle",
  "target_id": 1,
  "assignments": [
    {"drone_id": "uav1", "role": "chaser"},
    {"drone_id": "uav2", "role": "blocker_left"},
    {"drone_id": "uav3", "role": "blocker_right"}
  ],
  "formation": {"radius_m": 15.0},
  "reason": "目標往東北移動，兩翼從前方攔截、uav1 從後方追"
}
```

| 欄位 | 規則 |
|---|---|
| `mode` | `search` / `track` / `encircle` / `capture` / `hold` / `return_home` |
| `target_id` | `track`、`encircle`、`capture` 必填，且必須是目前存在的目標 |
| `assignments` | 每架無人機恰好出現一次；`role` 為 `chaser` / `blocker_left` / `blocker_right` / `overwatch` / `searcher` / `idle` / `herder` / `capturer` / `blocker` |
| `capture` 的角色 | 只能用 `herder` / `capturer` / `blocker` / `overwatch`；**恰好一架 `capturer`、至少一架 `herder`**。`herder` / `capturer` / `blocker` 只能在 `capture` 使用 |
| `formation.radius_m` | 5–60 m（包圍或跟隨半徑；預設 15） |
| `reason` | 必填，≤ 80 字，顯示在 Agent 指令面板 |

`plan_id`、時間戳與延遲由 C 加上，不由 LLM 產生。單機的 §7 白名單維持不變，供單機測試使用。

**捕捉策略 `capture`**（2026-11-18 新增，老師 11/16 建議 3）：

| 角色 | 中文 | 做什麼 |
|---|---|---|
| `herder` | 驅趕機 | 從目標後方 / 側方施壓，把目標趕向捕捉區 |
| `capturer` | 抓拍機 | 先到捕捉區伏擊，目標靠近時迎上去，讓目標進入自己相機畫面中央 |
| `blocker` | 封鎖機 | 守住目標偏離方向的側翼（和驅趕機形成 V 字） |

- **捕捉區**（capture zone）由 C 的確定性規劃器決定，不由 LLM 輸出：中心在目標前方 35 m（預設方向為「機群中心 → 目標」的延長線），半徑 10 m，離場地邊界至少 12 m；同一次 capture 期間固定不動。影片場景的目標不會反應，捕捉區放在目標前進的方向，目標轉彎（夾角 > 60°）時每 3 s 重新預測。
- 同一次 capture 期間沿用同一組角色（不會每次重算就換人）；換目標、操作員下新指令或角色交接時才重新分派。
- 「驅趕成功」：目標進入捕捉區（`zone_entered_at_s`）。「抓拍成功」見 §10.10。

### 10.6 GoalPoint（C → A，各機目標點，5–10 Hz）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `schema_version` | string | `"0.2.0"` |
| `plan_id` | string | 由哪一個計畫產生 |
| `drone_id` | string | 給哪一架 |
| `timestamp_ns` | int64 | 產生時間，**C 的時鐘**（Demo 引擎啟動後經過的時間）；A 在 ack 原樣放回。和 FleetPose 的 PX4 時間不同源，**兩者不能直接相減**，延遲請用各自的時鐘量 |
| `position_world_ned_m` | object | `n, e, d`（world_ned） |
| `yaw_rad` | float | 選填，期望航向（例如機頭朝向目標） |
| `max_speed_mps` | float | 選填，不得超過 A 的速度上限 |

A 的責任（安全層最終生效）：
- 追蹤目標點；**超過 1.0 秒沒收到新的目標點就懸停**。
- **機間安全距離（安全要求）**：任何時刻，任兩架的 3D 距離 ≥ 5 m。
  - 高度分層 uav1 / uav2 / uav3 = 25 / 30 / 35 m（GoalPoint 的 `d` 一律改成該架的高度層）。不同層之間相差 5 m，所以飛行途中水平交錯時（水平距離可能只剩約 1 m），3D 距離仍滿足要求。
  - **目標點推開距離 6 m**（A 的安全層，2026-11-10 由 5 m 調整）：兩架目標點的水平距離 < 6 m 時，A 會把目標點推開到 6 m，比安全要求多留 1 m 餘裕（推開到 5 m 時實測飛到的距離只有 4.94 m）。
  - 修改目標點時，GoalPointAck 的 `adjusted = true`，`reason` 寫明改了什麼。
- 其他安全處理（A 實作，2026-11-02）：飛行範圍 world_ned 北、東各 ±80 m（超出會夾到邊界）。
- **速度上限是 A 的啟動參數**（2026-11-18）：`agent_server.py --max-speed 5 --default-speed 5`；預設仍是上限 3 m/s、省略 `max_speed_mps` 時 2.5 m/s。上限超過約 6 m/s 時（煞車距離大於 6 m 推開距離）啟動會印警告；8 m/s 以上要先加大推開距離與機間距離再使用。A 實測 5 m/s 同層對向交換約 16 s、邊界測試 36 項通過。C 的 `demo.run --fleet sitl --max-speed` 要設成和 A 相同。
- **地面站（agent_server）失聯**（2026-11-18 改）：PX4 偵測到 Offboard 指令中斷後**原地懸停，30 s 沒恢復才原地降落**。原本是 PX4 自動直線返航，11/10 同層測試時兩架返航路線交叉、最近 0.59 m（見 §10.8 備註）。
- **GoalPointAck**（A 對每一則 GoalPoint 的回覆）：`drone_id`、`plan_id`、`timestamp_ns`（原樣放回）、`accepted`、`adjusted`、`reason`、`applied_position_world_ned_m`，以及選填的 `flight_command`（實際呼叫的 MAVSDK 函式，格式見 §10.8）。
- 地理圍欄（對應影片場景範圍）。

### 10.7 AgentEvent（C 內部，Agent 指令面板的每一筆紀錄）

| 欄位 | 說明 |
|---|---|
| `timestamp_ns` | 產生時間 |
| `plan_id` | 計畫編號 |
| `trigger` | `periodic` / `target_lost` / `target_found` / `operator` |
| `planner` | `llm` / `rule`（規則式備援） |
| `latency_ms` | 從觸發到得到合法計畫的時間 |
| `valid` | 是否通過 schema 驗證 |
| `plan` | 驗證後的 Plan（不合法時為 `null`） |
| `error` | 不合法或逾時的原因 |
| `operator_command` | 選填，觸發這筆計畫的操作員指令原文（`trigger = operator` 時） |
| `model` | 選填，實際回應的 LLM 模型名稱（有備援模型時用來統計）；規則式為 `null` |

AgentEvent 只在 C 內部使用（面板與報告指標），A、B 不需要實作。

### 10.8 執行狀態（A → C，2026-11-18 A 提供）

用途：C 的動作日誌與面板要能證明「Agent 的動作確實對應到無人機的操作指令，而且有執行到」（老師 2026-11-16 的建議）。三種訊息都是 A 主動推播，所有連線都收得到；C 寬鬆解析（欄位可能增加）。

**(a) `flight_command`：實際呼叫的 MAVSDK 函式**（放在 GoalPointAck 裡，選填）

```json
{"mavsdk": "offboard.set_position_ned", "arg": "PositionNedYaw", "frame": "local_ned",
 "frame_origin_world_ned_m": [-8.0, 0.0, 0.0], "north_m": 7.321, "east_m": 0.038, "down_m": -30.0,
 "yaw_deg": 91.8, "rate_hz": 20.0, "max_speed_mps": 2.5}
```

- A 每 0.05 s 呼叫一次 `offboard.set_position_ned`；座標是**那架自己的 local NED**（原點 = 出生點 `frame_origin_world_ned_m`），送出的是引導點，會以 `max_speed_mps` 往目標移動；ack 裡寫的是最後的目標。
- 第一次切進 Offboard 時多一個 `"also_called": ["offboard.start"]`。
- 讓路升降中，目標點先記下：`{"mavsdk": null, "deferred": "yield"}`。被拒絕時 `flight_command` 為 `null`。
- ActionAck 也有 `flight_commands`（陣列，依呼叫順序）：
  - `takeoff`：`action.set_takeoff_altitude` → `action.arm` → `action.takeoff` → `offboard.set_position_ned`
  - `land`：`offboard.stop` → `action.land`
  - `return_home`：`offboard.set_position_ned`（飛到出生點上方）→ `offboard.stop` → `action.land`

**(b) `ActionResult`：動作結束時的結果**

```json
{"type": "ActionResult", "action_id": "mv_ok", "drone_id": "uav1", "action": "move_to", "agent_id": "agent_uav1",
 "role": "tracker", "result": "succeeded", "why": "到達目標 1.5 m 內",
 "metrics": {"duration_s": 4.99, "time_to_reach_s": 4.99, "final_err_h_m": 0.4, "final_err_v_m": 0.01}}
```

- `result`：`succeeded` / `failed` / `overridden` / `cancelled` / `rejected`。
- `move_to`：水平 1.5 m 內、高度差 1 m 內算到達；超過時限還沒到判 `failed`（`why` 寫原因，例如「逾時：沒有到達目標」）。
- `follow`、`encircle` 沒有「到達某一點」，在 ActionEnd 時結算追蹤誤差 `err_mean_m`、`err_p95_m`、`err_max_m`。
- **C 對 GoalPoint 串流的驗證採用相同的到達標準**（水平 1.5 m、高度 1 m），兩邊的「動作驗證通過率」才可以比較。

**(c) `Yield`：讓路事件**（日誌裡叫 `yield_start`、`yield_end`）

```json
{"type": "Yield", "schema_version": "0.2.0", "event": "start", "drone_id": "uav3", "yield_to": "uav1",
 "timestamp_ns": 183880000000, "sim_time": 183.88, "action_id": "sw7_uav3",
 "from_alt_m": 30.0, "to_alt_m": 25.0, "direction": "down", "other_waits": true}
{"type": "Yield", "schema_version": "0.2.0", "event": "end", "drone_id": "uav3", "yield_to": "uav1",
 "timestamp_ns": 193576000000, "sim_time": 193.576, "action_id": "sw7_uav3",
 "from_alt_m": 25.0, "to_alt_m": 30.0, "reason": "back_to_layer"}
```

- 每個 `start` 之後一定有一個 `end`。`end` 的 `reason` 也可能是 `landed`（讓路中直接落地），這時 `to_alt_m` 為 `null`。
- `timestamp_ns` 是 PX4 時間（同 FleetPose）。
- 防撞分工：A 在正常飛行路線交叉時用**升降讓路**；C 的規劃器在送出前做**水平去衝突**（目標點 ≥ 6 m、飛行途中預測交錯時低優先權的角色橫移）。兩者可同時作用。

**備註：11/10 的「互撞」（A 2026-11-18 說明）**：三架同在 30 m 的同層測試，故意關掉地面站；PX4 斷線 2 s 後自己以約 4.9 m/s 直線飛回出生點，uav1 與 uav3 的路線在 (2.5, 0) 附近交叉，第 6.6 s 最近 0.59 m（沒有碰撞，PX4 與 Gazebo 都沒記到碰撞，三架正常降落）。原因是「失聯 fallback 的返航路線」不經過任何防撞層，已改成上述原地懸停 → 30 s 後原地降落；同樣測試重做多次，最近約 14 m。

**待確認**：
- A 實測 5 m/s 上下錯開時 3D 最近 4.94 m，低於 §10.6「任何時刻 3D ≥ 5 m」6 cm，需要 A 決定是放寬容許誤差還是加大讓路距離。
- §7 的 Action 請求格式還是初版（`follow`、`hover`、`search`…）；A 的實作已有 `move_to`、`follow`、`encircle`、`agent_id`、`role`（`tracker`、`guard`…），請 A 提供目前的 Action 請求格式後更新 §7，並和 C 的捕捉策略角色（`herder`、`capturer`、`blocker`）對齊。

### 10.9 各機 Agent 動作 DroneAction（C 內部，2026-11-18 新增）

老師 11/16 建議 2、3：每架無人機有自己的 Agent。協調 Agent（§10.5）選策略、分派角色；**各機 Agent 只從自己角色的白名單選一個動作**，不輸出座標或速度。座標由 C 的確定性規劃器算出，再以 GoalPoint（§10.6）送給 A。

```json
{"action": "engage", "reason": "目標離捕捉區 15 m，迎上去抓拍"}
```

| 欄位 | 規則 |
|---|---|
| `action` | 必須在下表「該架目前角色」的白名單內 |
| `reason` | 必填，1–60 字，顯示在即時指令翻譯面板 |

`drone_id`、角色、時間戳由 C 的程式填入，Agent 無法冒充別架或改自己的角色。schema 版本 `"0.3.0"`。

| 角色 | 動作白名單 | 對應的 GoalPoint（C 的規劃器） |
|---|---|---|
| `herder` | `push` / `circle_behind` / `hold` | `push`：目標後方 12 m（捕捉區的反方向）；`circle_behind`：目標側方 18 m（繞到後方途中，避免把目標往反方向趕） |
| `capturer` | `ambush` / `engage` / `hold` | `ambush`：捕捉區中心；`engage`：目標位置 + 速度 × 1 s |
| `blocker` | `flank` / `hold` | 目標側後方 14 m，邊依目標偏離方向決定 |
| `chaser` / `blocker_left` / `blocker_right` | `follow_slot` / `hold` | 包圍隊形位置（§10.5 `formation.radius_m`） |
| `overwatch` | `overwatch` / `hold` | 目標正上方，該架高度層再加 10 m |
| `searcher` | `orbit` / `hold` | 繞圈搜尋 |
| `idle` | `hold` | 原地懸停 |

- 每個角色都可以選 `hold`（原地懸停，在選定當下的位置），這是 Agent 能做的最保守選擇。
- 不合法（格式錯、用了別的角色的動作、逾時）→ **不送出**、記入動作日誌（§10.11，`valid = false`），該架沿用上一個合法動作；沒有合法動作時用規則式。LLM 延遲或失效時無人機不會失控。
- 動作送出前，C 的規劃器先做去衝突（目標點水平 ≥ 6 m；預測飛行途中最近距離 < 8 m 時，低優先權角色橫移讓路，`capturer` 優先權最高），A 的安全層最後把關（§10.6、§10.8）。防撞不交給 LLM。
- 各機 Agent 的決策頻率約每 6 s 一次（或角色、情況改變時），協調 Agent 約每 2 s；GoalPoint 由規劃器以 10 Hz 更新。

### 10.10 抓拍判定（C 內部，2026-11-18 新增）

C 用每架的模擬相機（§6 的內參、該架的位姿與相機安裝角）把目標的**真值**位置投影到影像上，以下條件**同時連續成立 2 s** 即判定該架抓拍成功：

| 條件 | 預設 |
|---|---|
| 目標中心在影像中央區域 | 中央 20% × 20%（30 m 高時地面約 ±5 m × ±7 m） |
| 目標在影像上的尺寸 | ≥ 40 px（以車長 4.5 m 投影；行人約 0.5–0.6 m） |
| 持續時間 | 2 s |

- 用真值而不是估計值：判定的是「相機畫面裡是否真的拍到」，和 C 的估計誤差無關。
- 任何一架符合都會記錄（`capture_events`：時間、哪一架、角色、像素偏移、像素尺寸）；報告的「抓拍成功時間」以 `capturer` 完成的第一次為準（`time_to_capture_by_capturer_s`）。
- 影片場景的「真值」是影片中目標經地面標定後的位置。

### 10.11 動作日誌 ActionRecord（C 內部，2026-11-18 新增）

老師 11/16 建議 4：每個動作都要記錄、可驗證，且能對應到實際的飛控指令。每一筆記錄一個 Agent 動作的完整鏈：

```
Agent 動作 → 白名單驗證 → 對應的 GoalPoint（含 C 的去衝突修改）→ A 的 ack 與實際 MAVSDK 呼叫 → 實際位姿是否在期限內達成
```

| 欄位 | 說明 |
|---|---|
| `action_id` | `uav1-0007`（多 Agent）/ `plan-0003`（單一指揮官） |
| `drone_id` / `agent_id` / `role` / `mode` / `target_id` | 哪一架、哪個 Agent（`agent_uav1`…；單一指揮官為 `coordinator`）、角色、模式、目標 |
| `action` / `reason` / `planner` / `latency_ms` / `model` | 動作、理由、`llm` / `rule`、延遲、LLM 模型 |
| `valid` / `validation_error` | 白名單驗證結果；不合法的動作不送出 |
| `goals_sent` / `last_goal_ned` / `c_adjustments` | 送出的 GoalPoint 數、最後一個目標點（world_ned）、C 的去衝突修改次數 |
| `acks` / `accepted` / `adjusted` / `rejected` / `last_ack_reason` | A 的 GoalPointAck 統計與原因（照實記錄，不重送繞過 A） |
| `flight_command` | A 實際呼叫的 MAVSDK 函式（§10.8(a)） |
| `t_start_s` / `deadline_s` / `t_end_s` / `time_to_reach_s` | 開始、期限、結束、到達所需時間（s，C 的時鐘） |
| `track_err_mean_m` / `track_err_p95_m` | 跟隨目標點的水平誤差 |
| `command_consistency` | 指令一致率：無人機速度方向與「指向目標點的方向」夾角 < 45°（或已到達）的時間比例 |
| `result` / `why` | 結果（用語同 A 的 ActionResult，§10.8）與原因 |

**`result` 的判定**：

| `result` | 條件 |
|---|---|
| `succeeded` | 目標點不動時：水平 1.5 m、垂直 1.0 m 內（同 A 的 `move_to`）；目標點在動時：水平 1.5 m + 目標點速度 × 1 s 內，連續 2 s |
| `failed` | 期限內沒有達成。期限 = max(8 s, 距離 / 速度上限 × 1.5 + 5 s) |
| `rejected` | A 拒絕，且之後沒有任何一則被接受 |
| `overridden` | 還沒判定前就被新的動作或計畫取代（不列入通過率） |
| `cancelled` | Demo 結束時仍在進行（不列入通過率） |

- 面板顯示：通過（`succeeded`）/ 被 A 修改（`succeeded` 但 `adjusted > 0`）/ 被拒絕 / 未達成 / 已改派 / 已取消 / 進行中。
- **動作驗證通過率** = `succeeded` /（`succeeded` + `failed` + `rejected`）。
- 輸出：`actions.jsonl`（完整）、`actions.csv`（UTF-8 BOM，Excel 可直接開）；各機 Agent 每次決策另記 `drone_events.jsonl`。
- 動作日誌只記錄、不改寫。A、B 不需要實作。

---

## 11. 變更紀錄

| 版本 | 日期 | 變更 | 作者 |
|---|---|---|---|
| 0.1.0 | 2026-10 | 初稿 | C |
| 0.1.1 | 2026-10-19 | 依 A 回覆填入相機參數（§6）、姿態頻率與時間戳（§2.2、§2.3、§4）、ENU 警告（§2.1）；依 B 回覆補充 Detection 的影像來源要求（§3）；定義 `uncertainty_m`（§5） | C |
| 0.1.2 | 2026-10-31 | 依 A 的實測回覆 §9 四項：odometry 時間戳與頻率、24 ms 的成因（§2.3、§4），`get_state()` 範例（§4），相機位移與參考點約定（§6）；補充 EKF yaw 偏差（§2.4）。訊息欄位沒有變動 | A（C 代為整理） |
| 0.2.0 | 2026-11-01 | 草案：新增 §10 多機與影片 Demo 介面（world_ned、FleetPose、影片 Detection、FusedTarget、Agent Plan、GoalPoint、AgentEvent）；§1 補 Demo 資料流 | C |
| 0.2.1 | 2026-11-01 | 標示確認狀態：A 合併 PR #3 同意 §10.1、§10.6；§10.3 待 B 確認。§10.7 補上 C 內部欄位 `operator_command`、`model`。§2.4 補上 EKF yaw 偏差的同次執行確認與原因（磁偏角）；§4 註明 SITL 的 `EKF2_GPS_DELAY` 須為 0。A、B 的訊息格式沒有變動 | C |
| 0.2.2 | 2026-11-02 | §2.4、§5：相機模型流程的 bbox 參考點由「底邊中點」改為「中心 + 半高平面」（依 Phase 2 實驗 5：nadir 看車最大誤差 3.4 m → 0.1 m 以下）；影片 homography 流程維持底邊中點。A、B 的訊息格式沒有變動 | C |
| 0.2.3 | 2026-11-11 | §10.6：機間安全距離改寫為「任何時刻 3D ≥ 5 m」，並記錄 A 的實作：目標點推開距離 6 m（11/10 由 5 m 調整）、高度分層、飛行範圍 ±80 m、速度上限 3 m/s。訊息格式沒有變動 | C |
| 0.2.4 | 2026-11-18 | 依 A 11/18 回覆：§10.2 FleetPose 時間戳為 PX4 / 模擬時鐘（三架同源）、§10.6 GoalPoint 時間戳為 C 的時鐘（兩者不能相減）；§10.6 速度上限改為 A 的啟動參數、地面站失聯改為原地懸停 → 30 s 後原地降落、GoalPointAck 欄位；新增 §10.8 執行狀態（`flight_command`、`ActionResult`、`Yield`）與 11/10 近距離事件說明。C 依此新增解析，A 的既有訊息沒有破壞性變動 | A（C 代為整理） |
| 0.2.5 | 2026-11-18 | 老師 11/16 建議（多 Agent、捕捉策略、動作可驗證）：§10.5 新增 `capture` 模式與 `herder` / `capturer` / `blocker` 角色、捕捉區；新增 §10.9 各機 Agent 動作白名單、§10.10 抓拍判定、§10.11 動作日誌；§10.3 補充影片場景需要 B 的 `track_id` 與地面標定檔格式。都是 C 內部格式，A、B 的訊息沒有變動 | C |
