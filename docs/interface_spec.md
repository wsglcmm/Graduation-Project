# 模組介面規格（Interface Specification）

- 版本：`0.1.1`（A、B 已初步回覆，仍有待確認項目，見 §9）
- 訊息的 `schema_version` 仍為 `"0.1.0"`（本版沒有新增或更改欄位）
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

- 影像與姿態的取樣頻率不同（實測影像 30 Hz、姿態約 20 Hz），**由 C 負責對齊**：對影像時間戳在姿態序列上做插值（位置線性插值、姿態用 slerp）。
- 姿態只有 20 Hz，偵測到達時可能還沒有「之後」的姿態：C 會先讓該幀排隊，等下一筆姿態到了再內插，不做外插。
- 相鄰兩筆姿態相隔超過 150 ms（連續掉 2 筆以上）時不插值，回傳 `no_pose`。
- **位置與姿態必須是同一時刻**：目前位置沒有時間戳。建議 A 改用 MAVSDK `telemetry.odometry()`，它的位置、四元數、速度共用同一個 `time_usec`。
- **A 必須提供帶時間戳的姿態序列**，並保留至少最近 2 秒的緩衝，不可只提供「最新一筆」。
- **B 的輸出時間戳 = 該張影像的擷取時間**，不是推論完成時間。
- 若姿態緩衝找不到涵蓋該影像時間戳的區間，C 回傳 `status = "no_pose"`，不輸出座標。

### 2.4 地面平面假設（目前版本）

- 假設目標位於地面平面 `d = 0`（NED 的 down = 0，即與起飛點同高）。
- 需要 A 提供「相對地面高度」（`altitude_agl_m`）。
- bbox 的參考點：**底邊中點** `(u, v) = ((x1+x2)/2, y2)`，代表目標接地點。
- 已知誤差來源：地形起伏、姿態/時間戳誤差、bbox 抖動、車輛有高度（底邊中點不等於幾何中心）。誤差量化屬 Phase 2 的工作。

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

- 頻率：實測約 20 Hz（可接受，C 會內插）；若能提高到 ≥ 30 Hz 更好
- 以時間序列提供（見 2.3）

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

1. 取 bbox 底邊中點 `(u, v)`，反投影成相機座標射線：`d_c = K⁻¹ · [u, v, 1]ᵀ`
2. 相機 → 機體：`d_b = R_bc · d_c`（R_bc 為安裝外參，見 §6）
3. 機體 → NED：`d_n = R_nb(roll, pitch, yaw) · d_b`
4. 與地面平面 `d = 0` 求交：`λ = (0 − p_d) / d_n[d]`（需 `d_n[d] > 0`，否則射線朝天，status = `above_horizon`）
5. `P_ned = p_ned + λ · d_n`

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
| 位置偏移 | 暫定機體 FRD (0, 0, −0.10) m；camera_link 到 base_link 的精確位移待 A 計算（估計差 10–20 cm） |

> 設定檔：`config/camera.yaml`。

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
- [ ] （A）camera_link 到 base_link 的精確位移
- [x] （A）姿態資料的實際更新頻率：約 20 Hz，有 `att_timestamp_us`
- [ ] （A）位置的時間戳：改用 MAVSDK `odometry()`，或說明位置與姿態的時間關係
- [ ] （A）「相機與姿態同一刻差 24 ms」是固定偏移，還是取樣間隔造成的最近距離
- [ ] （A）提供一筆 `get_state()` 的實際輸出範例（確認欄位名稱、角度單位）
- [x] （A）模擬的時鐘來源：相機與姿態都使用 Gazebo 模擬時間
- [x] （B）推論延遲：約 7.4 ms（靜態照片，僅推論）
- [ ] （B）改接 Gazebo 相機影像（1280×960、sim time 時間戳）
- [ ] （B）確認 bbox 為原始影像像素（`boxes.xyxy`），並回報端到端延遲
- [ ] （B）class 列表是否與本文件一致
- [ ] （全員）傳輸方式
- [ ] （全員）Agent 的角色定位（老師確認）

## 10. 變更紀錄

| 版本 | 日期 | 變更 | 作者 |
|---|---|---|---|
| 0.1.0 | 2026-10 | 初稿 | C |
| 0.1.1 | 2026-10-19 | 依 A 回覆填入相機參數（§6）、姿態頻率與時間戳（§2.2、§2.3、§4）、ENU 警告（§2.1）；依 B 回覆補充 Detection 的影像來源要求（§3）；定義 `uncertainty_m`（§5） | C |
