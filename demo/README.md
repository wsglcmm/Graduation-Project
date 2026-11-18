# 多機捕捉 Demo（多 Agent、即時指令翻譯、動作驗證）

3 架無人機各有一個 AI Agent，由協調 Agent 分派角色（驅趕 / 抓拍 / 封鎖…），合力搜尋、追上、捕捉一個移動目標。

## 最快的開始方式：網頁版畫面

```powershell
python -m demo.run --web --target evader --strategy capture --drone-speed 5
```

會自動開瀏覽器到 `http://127.0.0.1:8000/`（Ctrl+C 結束；只用 Python 標準函式庫與單一 HTML 檔，不需要網路）。畫面：

| 區塊 | 內容 |
|---|---|
| 即時指令翻譯（右上，主角） | 操作員的自然語言 → 協調 Agent 選的策略與分工 → 每架 Agent 的動作 → ① Agent 決策 ② 飛控指令（GoalPoint；接 A 時加上 A 實際呼叫的 MAVSDK 函式）③ A 的回覆 ④ 驗證，最新在上 |
| 鳥瞰圖（左上） | 各機位置、航向、相機視野、角色與動作、目標點連線；捕捉區；目標估計與誤差圈、真值；視角自動跟著目前的動態 |
| 抓拍畫面（鳥瞰圖右上角） | 抓拍成功時，依抓拍判定的像素偏移重建的模擬相機畫面（中央虛線框 = 判定用的中央 20%） |
| 各機角色卡片（左下） | 角色、目前動作、最近一筆動作的驗證結果與指令一致率、A 的狀態（飛行 / 讓路給誰 / failsafe） |
| 動作日誌（右下） | 每筆動作的飛控指令、A 的回覆（接受 / 修改 / 拒絕）、指令一致率、驗證結果，顏色：綠 = 通過、橘 = 被 A 修改、紅 = 被拒絕 / 未達成、灰 = 已改派 |

輸入框可以直接打中文，或按下方的常用指令按鈕。終端機同時印出文字紀錄，結束時輸出 `actions.csv`、`drone_events.jsonl` 等。

常用組合：

```powershell
python -m demo.run --web --target evader --strategy capture --drone-speed 5 --agents single
python -m demo.run --web --target evader --planner llm --period 5
python -m demo.run --web --fleet sitl --max-speed 5 --target evader --strategy capture
```

第一行是單一指揮官（對照組）；第二行是協調 Agent 用 LLM，在輸入框打「把那台車趕進捕捉區抓拍」；第三行接 A 的 SITL（或假伺服器）。

## 多 Agent（協調 Agent + 各機 Agent）

| 層 | 誰 | 決定什麼 | 輸出（白名單） |
|---|---|---|---|
| 協調 Agent | `llm_agent/commander.py` | 選策略（搜尋 / 追蹤 / 包圍 / 驅趕抓拍 / 懸停 / 返航）、分派角色 | Plan（規格 §10.5） |
| 各機 Agent × 3 | `llm_agent/drone_agent.py` | 自己這個角色**現在要怎麼做** | 只能從角色專屬的動作選（`llm_agent/drone_actions.py`） |
| 確定性規劃器 | `planner/` | 動作 → 目標點、防撞去衝突 | GoalPoint（規格 §10.6） |

角色專屬動作：驅趕機 `push / circle_behind / hold`、抓拍機 `ambush / engage / hold`、封鎖機 `flank / hold`、包圍隊形 `follow_slot / hold`、搜尋 `orbit / hold`。用了別的角色的動作 → 驗證失敗、記入動作日誌、不送出，該架沿用上一個合法動作。

各機 Agent 預設規則式（重現原本規劃器內建的決策，所以多 Agent 版的規則式表現和單一指揮官一樣：驅趕抓拍 10/10、38.9 s）；`--drone-planner llm` 時每架各自呼叫 LLM（角色改變時立刻問，之後每 `--drone-period` 秒），LLM 還沒回來或失敗時先用規則式。3 架 + 協調 Agent 一起用 LLM 時呼叫量是 4 倍，免費額度下 `--drone-period` 建議 ≥ 6。

比較多 Agent 與單一指揮官：

```powershell
python -m demo.benchmark --scenario evader_capture command_capture --agents single multi --seeds 0 1 2
python -m demo.benchmark --planner llm --period 5 --agents single multi --drone-planner llm --scenario command_capture --seeds 0
```

## 執行（matplotlib 視窗版與無畫面模式）

在專案根目錄、`.venv` 啟用的狀態下：

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m demo.run
```

這是**規則式指揮官**，不需要 API 金鑰，先用它確認畫面正常。

### 接上 LLM（預設 Gemini）

1. 到 Google AI Studio 建立 Gemini API 金鑰（免費層即可；免費層的內容會被 Google 用來改善產品，不要放敏感資料）。
2. 把 `env.example` 複製成 `.env`，填入 `GEMINI_API_KEY`。`.env` 已在 `.gitignore`，**絕對不要 commit**。
3. 執行：

```powershell
python -m demo.run --planner llm
```

想改用 Claude：在 `.env` 設 `AGENT_PROVIDER=anthropic` 並填 `ANTHROPIC_API_KEY`（Claude API 需要預付額度）。

**輸入指令**：視窗出現後，回到執行程式的終端機（PowerShell 或 VS Code 終端機）直接打字按 Enter，支援中文輸入法，例如「三架一起把那台車包圍起來」「先懸停」「回家」。視窗下方的輸入框也能用，但 matplotlib 不支援輸入法，只能打英文。LLM 回覆通常要一到數秒；這段時間無人機會照上一個合法計畫飛，不會停下來等。

**操作員指令優先**：操作員指令產生的計畫會一直有效，直到下新指令。`hold`（懸停）與 `return_home`（返航）會讓 Agent 完全暫停；`search` 期間 Agent 不能自己切回包圍；`track` / `encircle` 期間只能在追蹤、包圍、搜尋（目標遺失時）之間切換。操作員指令的請求失敗會自動重送 2 次，仍失敗就改用規則式解讀。

**預排指令**（報告要可重現時用）：`--cmd "秒數=指令"`，可重複，例如：

```powershell
python -m demo.run --planner llm --cmd "15=三架一起把那台車包圍起來" --cmd "60=先懸停"
```

**模型與備援**：預設主模型 `gemini-3.1-flash-lite`（上一代，尖峰時較少 503），忙線（503）、額度用完（429）或逾時時自動改用 `gemini-3.5-flash-lite`；面板標題會顯示「備援中」。

### 不開視窗，產生報告用的指標

```powershell
python -m demo.run --headless --duration 90
```

```powershell
python -m demo.run --planner llm --headless --duration 90
```

輸出位置是 `output/demo/<時間>/`，裡面有 `metrics.json`（包圍完成時間、視野內比例、估計誤差、LLM 合法率與延遲、最小機間距離）和 `events.jsonl`（每一筆 Agent 事件）。`output/` 已在 `.gitignore`。

估計誤差依目標狀態分開統計：`estimate_rmse_m` 只算「追蹤中 + 外推中」；`estimate_rmse_by_status_m` 列出三種狀態各自的 RMSE，`target_status_fraction` 是各狀態的時間比例。懸停、返航時目標跑遠被判定遺失，遺失時段的誤差不會再拉高主指標。

### 固定劇本比較（報告用）

同一個劇本、同一組 seed，規則式與 LLM 才能公平比較。劇本定義在 `demo/scenarios.py`：

| 劇本 | 內容 |
|---|---|
| `autonomous` | 沒有操作員指令，Agent 自己搜尋、追蹤、包圍（90 s） |
| `command_encircle` | 第 10 秒下令包圍（90 s） |
| `hold_resume` | 15 s 包圍 → 40 s 懸停 → 60 s 恢復包圍（100 s） |
| `return_home` | 15 s 包圍 → 50 s 返航（75 s） |
| `evader_encircle` | 策略場景：會逃的目標，預設策略包圍（對照組，120 s） |
| `evader_capture` | 策略場景：會逃的目標，預設策略驅趕 → 抓拍（120 s） |
| `command_capture` | 策略場景：第 10 秒下令「把那台車趕進捕捉區抓拍」（120 s） |

策略場景的無人機速度上限用 A 的 3 m/s；只跑策略比較：

```powershell
python -m demo.benchmark --scenario evader_encircle evader_capture --seeds 0 1 2 3 4 5 6 7 8 9
```

```powershell
python -m demo.benchmark
python -m demo.benchmark --planner llm --period 5
```

規則式是快轉執行，全部劇本 × 3 個 seed 約 10 秒；LLM 照真實時間跑，約 18 分鐘（會用到約 200 次 API 請求，注意免費層額度）。輸出在 `output/benchmark/<時間>_<規劃者>/`：`summary.md`（平均 ± 標準差，可直接貼進報告）、`runs.csv`（每次執行一列，Excel 可直接開）與每次執行的 `metrics.json`、`events.jsonl`。

除了上面的指標，還會計算：計畫合法率與延遲（不分規劃者）、操作員指令回應時間、懸停後恢復到重新包圍的時間，以及「懸停／返航期間 Agent 自己產生的計畫數」（應為 0）。

## 捕捉策略（策略場景）

```powershell
python -m demo.run --target evader --strategy capture
python -m demo.run --target evader --planner llm --period 5
```

第二行在終端機輸入「把那台車趕進捕捉區抓拍」，看 LLM 把自然語言轉成捕捉策略。

| 元件 | 檔案 | 說明 |
|---|---|---|
| 會逃的目標 | `sim2d/evader.py` | 簡化逃跑模型：離無人機越近、無人機越是逼近，逃得越快（最高 2.2 m/s < 無人機 3 m/s）；懸停的無人機威脅較小；圓形場地 60 m |
| 捕捉策略幾何 | `planner/capture.py` | 捕捉區（機群 → 目標延長線、目標前方 35 m）；驅趕機站在目標遠離捕捉區那側；抓拍機在捕捉區伏擊、目標進入 18 m 後迎上去；封鎖機在側後方和驅趕機成 V 字 |
| 角色分派 | `llm_agent/rule_planner.py` | 依距離分派；同一次捕捉中角色固定（實測：每 2 秒重算就會換人，策略失效） |
| 抓拍判定 | `demo/capture_judge.py` | 目標在抓拍機影像中央 20% × 20%、像素長度 ≥ 40 px、連續 2 s；裁判用目標真值 |

**不同速度上限的結果**（規則式、20 seeds、120 s、含防撞第二層；逃跑模型不變，最高 2.2 m/s）：

| 無人機速度上限 | 抓拍機完成抓拍 | 目標被趕進捕捉區 | 最小水平距離（平均 / 最差） |
|---|---|---|---|
| 3 m/s（A 預設） | 19/20，51.5 ± 5.9 s | 13/20 | 6.4 / 3.8 m |
| 5 m/s（A 已支援） | 20/20，38.0 ± 3.6 s | 18/20 | 7.8 / 6.3 m |
| 8 m/s | 20/20，36.2 ± 6.0 s | 20/20 | 7.2 / 3.2 m |

抓拍機「迎上去」有遲滯（目標進入 18 m 才迎上去、超過 30 m 才回伏擊點）。只用一個門檻時，抓拍機一迎上去就把目標嚇回門檻外、退回伏擊點、目標又靠近，來回擺盪：5 m/s 時 52.8 ± 19.1 s（最慢 108 s）、8 m/s 只有 4/10 成功。加上遲滯後速度越快越好，不需要依速度另外調參數。

畫面上的半透明橘色圓是捕捉區；抓拍成功時面板出現「★ 抓拍成功」。指標：`time_to_capture_by_capturer_s`（抓拍機完成抓拍）、`zone_entered_at_s`（目標被趕進捕捉區）、`capture_events`。

**限制（報告要寫）**：逃跑模型是我們設計的簡化模型，不代表真實車輛或行人；影片場景的目標不會反應，只能展示預測攔截 → 抓拍。

## 影片場景（場景二：真實鳥瞰影片）

目標換成鳥瞰影片中的一台車，3 架無人機疊在影片上追；同一套多 Agent 與動作日誌。PowerShell：

```
python -m demo.run --video data/videos/road01 --video-file "data/videos/road 01_1080p50.mp4" --web
```

- `--video`：B 的偵測資料夾（內含 `detections.jsonl`，§10.3 格式或 B 的原始格式都可以）。
- `--video-file`：背景影片，網頁版用它疊圖（無人機、視野、捕捉區、目標估計都用 homography 反投影到影片像素上）。
- `--video-track`：指定當目標的 track_id；預設自動挑「真的在移動的時間」最長的車（每 1 s 取樣，避開停等紅燈的車與幾秒就穿過畫面的車）。
- 預設策略是 `capture`，捕捉區放在目標前進的方向，目標轉彎時每 3 s 重新預測（`CaptureConfig.replan_zone_s`）。
  影片目標不會理無人機、趕不動，所以畫面上的角色改叫「攔截抓拍 / 尾隨 / 側翼」、模式叫「預測攔截抓拍」（只改顯示文字，日誌裡的 role 代碼不變）。

### 地面標定（每支影片做一次，約 10–20 分鐘）

```
python -m demo.run --video data/videos/road01 --video-file "data/videos/road 01_1080p50.mp4" --calibrate
```

瀏覽器會開標定工具：在畫面上點地面上的特徵點（斑馬線端點、車道線頭尾、停止線的角），每點填實際地面座標（公尺，原點與方向自己選），按「計算誤差」看重投影 RMS，滿意後「儲存」→ `config/video_ground_road01.json`。之後的 `--video` 會自動讀這個檔。

- 點 **6 個以上**、分散在畫面四角：剛好 4 點時誤差一定是 0，檢查不出標錯。
- 距離來源：Google 地圖「測量距離」，或已知尺寸（台灣車道寬約 3–3.5 m）。報告要寫標定方法與 RMS。
- 還沒標定時可以加 `--ground-scale 0.05`（每像素 0.05 m、畫面中心為原點）先看流程；畫面左下角會標「地面標定為暫定值，距離不準」。

### track_id

B 的 `detections.jsonl` 還沒有 ByteTrack 的 `track_id` 時（只有佔位 ID），C 這邊用地面座標最近鄰 + 等速預測做暫代關聯（gating 3 m、同類別群組），只為了先把流程接通，畫面會標「track_id 為 C 的暫代關聯」。road01 實測：884 條軌跡（真實車輛遠少於此，遮擋或漏偵測就斷成新軌跡）。報告與評估一律以 B 的 ByteTrack ID 為準；B 交付後直接換檔即可，不用改程式。

暫定比例尺（0.05 m/px）下 road01 自動選到軌跡 #413（19 s），抓拍機約 10 s 完成抓拍；正式數字等地面標定與 B 的 track_id 都到位後再跑。

**限制（報告要寫）**：影片是錄好的，目標不會對無人機做出反應，所以只展示預測攔截 → 抓拍，不展示驅趕；無人機的位置與相機畫面是模擬的（疊在真實影片上），抓拍判定用影片中目標的地面位置。

## 動作日誌與驗證（`ledger/`）

對應老師建議 4：動作要記錄下來、可以被驗證，且可對應到無人機操作的指令。每次 Demo 結束會在輸出資料夾寫出 `actions.jsonl`（完整）與 `actions.csv`（Excel 直接開），面板與終端機會即時印出每筆動作的結果。

一筆「動作」= 一架無人機的一段任務（模式、角色、目標不變；Agent 每 2 秒重新規劃但角色沒變時不開新動作）。每筆的驗證鏈：

| 層 | 記錄什麼 | 來源 |
|---|---|---|
| 1. 白名單驗證 | Agent 輸出是否合法；不合法 → `rejected`，不送出 | AgentEvent |
| 2. 對應飛控指令 | 送出幾個 GoalPoint、最後的目標點、C 去衝突改了幾次 | 規劃器 |
| 3. A 的回覆 | 接受 / 修改 / 拒絕次數與原因、A 實際呼叫的 MAVSDK 函式（`flight_command`） | GoalPointAck（規格 §10.8，只有 `--fleet sitl` 才有） |
| 4. 實際執行 | **指令一致率**：無人機是不是朝目標點飛（夾角 < 45° 或已到達）<br>**到達**：目標點不動時水平 1.5 m、高度 1 m（同 A 的 move_to）；目標點在動時 1.5 m + 目標點速度 × 1 s，並連續 2 s | FleetPose / 替身 |

結果用和 A 的 ActionResult 相同的詞彙：`succeeded`（通過；被 A 修改過的顯示「被 A 修改」）/ `failed`（未達成）/ `rejected`（被拒絕）/ `overridden`（已改派）/ `cancelled`（已取消）。

**兩層分開看**（規則式、5 seeds）：

| 劇本 | 指令一致率 | 到達率 |
|---|---|---|
| autonomous（兔子、8 m/s） | 97% | 66% |
| hold_resume | 97% | 80% |
| evader_encircle（3 m/s） | 95% | 39% |
| evader_capture（3 m/s） | 90% | 20% |

指令一致率都在 90% 以上：無人機確實照指令飛。到達率低的原因是目標點本身追不上，屬於規劃問題，動作日誌把它找了出來：
- **搜尋圈太快**：搜尋時目標點以固定 0.25 rad/s 繞圈，半徑 60 m 時切線速度 15 m/s，是無人機的 3–5 倍。改成「切線速度 ≤ 速度上限的 80%」後，5 m/s 時找到目標更快（15.2 vs 18.8 s），但捕捉策略的開局要重調（找到目標時三架平均散在圈上，抓拍時間 38 → 43 s），所以先保留原本的搜尋，列為下一步。
- **封鎖機的目標點**跟著目標側向移動，速度接近無人機上限，3 m/s 時常常追不上。

**後續實驗：修正搜尋圈 + 換捕捉區方向**（規則式、20 seeds、120 s；`search_speed_fraction` 是搜尋圈切線速度 / 速度上限，0 = 不限制）：

| 搜尋圈 | 捕捉區方向 | 3 m/s 抓拍 | 5 m/s 抓拍 | 5 m/s 動作到達率 |
|---|---|---|---|---|
| 不限制（預設） | 機群 → 目標（預設） | 20/20，53.8 s | 20/20，**38.1 s** | 43% |
| 不限制 | 目標速度方向 | 16/20，62.0 s | 20/20，38.1 s | 43% |
| 不限制 | 朝場地中心 | 4/20，79.9 s | 20/20，54.6 s | 44% |
| 0.8 | 機群 → 目標 | 12/20，62.0 s | 20/20，43.3 s | **60%** |
| 1.5 | 機群 → 目標 | 20/20，67.0 s | 20/20，37.6 s | 45% |

修正搜尋圈讓「動作到達率」變好（目標點追得上），但抓拍變慢：找到目標時三架平均散在圈上，開局要多繞路。原本的搜尋雖然目標點追不上，三架反而擠在一起、找到後馬上能分工。抓拍是主要指標，所以維持原本的預設，兩個參數都保留給報告做比較。這是個好例子：局部看起來「對」的修正，不一定讓整體指標變好，要用固定劇本量過才知道。

## 架構（對應規格 §10）

```
兔子路線 ─→ 各機視野觀測 ─→ 融合 ─→ 卡爾曼 ─→ 指揮官（Agent，背景執行，約每 2 秒或有事件時）
 sim2d        fusion          fusion   fusion    llm_agent.commander
                                                        │ Plan（角色、模式、隊形半徑、理由）
                                                        ▼
                     無人機替身 ←── 各機目標點（10 Hz）←── 隊形規劃器
                     sim2d.fleet_sim     GoalPoint         planner.formation
```

| 元件 | 現在 | 之後換成 |
|---|---|---|
| 兔子（目標） | `sim2d/scenario.py` 合成路線 | B 的影片偵測 + `coord_transform/homography.py` 地面標定 |
| 老鷹（無人機） | `sim2d/fleet_sim.py` 運動學替身 | A 的 3 架 PX4 SITL（輸入 GoalPoint、輸出 FleetPose，格式相同） |
| 指揮官 | 規則式，或 LLM（預設 Gemini 3.1 Flash-Lite，備援 3.5 Flash-Lite，可切換 Claude Haiku 5.5） | 不變 |

## 安全設計（口試重點）

- LLM **只決定角色與模式**，不輸出座標或速度；座標由確定性的隊形規劃器計算。
- LLM 輸出一律經 strict schema 與語意檢查（每架恰好一次、目標存在），不合法就丟棄並**沿用上一個合法計畫**；還沒有計畫時改用規則式。
- 無人機超過 1 秒沒收到新目標點就懸停；速度、加速度上限與機間互斥在替身中模擬，正式版由 A 的飛控安全層最終把關。
- **防撞三層**（LLM 不負責防撞）：① 高度層 25 / 30 / 35 m；② C 的規劃器送出前去衝突（`planner/deconflict.py`）：目標點水平 ≥ 6 m，並預測飛行途中的最近距離（CPA），< 8 m 時優先權低的角色橫移讓路（抓拍機優先權最高，不會被推離目標）；③ A 的安全層。
  只靠高度層時，水平疊在一起的 3D 距離剛好 5 m，沒有餘裕。實測（規則式、10 seeds）：

  | 劇本 | 最小水平距離：關閉 → 開啟 | 最小 3D 距離：關閉 → 開啟 | 任務影響 |
  |---|---|---|---|
  | 捕捉策略（會逃的目標，3 m/s） | 0.9 → 6.3 m | 5.3 → 8.1 m | 抓拍 10/10 → 9/10，時間 48.9 → 49.9 s |
  | 包圍（兔子，8 m/s） | 1.5 → 11.0 m | 5.2 → 14.1 m | 包圍完成 26.1 → 29.9 s |

  讓路觸發距離越大越安全、但抓拍越難（10 m 時抓拍只剩 1/10），取捨寫在 `planner/deconflict.py`。指標 `min_drone_horizontal_m`、`goal_deconflict_adjustments` 會列在 metrics 與 benchmark。

## 可以調的參數

- `--period`：多久問一次 Agent（預設 2 秒）。越短反應越快，但 API 花費和排隊都會增加。
- `.env` 的 `AGENT_PROVIDER`（`gemini` / `anthropic`）、`AGENT_MODEL`（預設 `gemini-3.1-flash-lite`）、`AGENT_FALLBACK_MODELS`（預設 `gemini-3.5-flash-lite`，逗號分隔，空字串代表不備援）、`AGENT_THINKING`（Gemini 思考程度，例如 `minimal` 可降低延遲）、`AGENT_TIMEOUT_S`（預設 10）。
- `--seed`：換一條兔子路線。

## 接 A 的 3 架 SITL（Phase 3）

Demo 引擎的機隊後端可以切換：預設用內建替身 `sim2d`，加 `--fleet sitl` 就改連 A 的 `agent_server`（規格 §10.2 FleetPose、§10.6 GoalPoint）。融合、Agent、隊形規劃都不用改。

**連 A 在學校伺服器上的模擬**（A 的伺服器只綁 127.0.0.1:8765，要先用 SSH 通道轉到自己電腦；帳號向 A 索取）：

```powershell
ssh -N -L 8765:127.0.0.1:8765 帳號@伺服器位址
```

保持這個視窗開著，另開一個終端機：

```powershell
python -m demo.run --fleet sitl
python -m demo.run --fleet sitl --planner llm --period 5
python -m demo.run --fleet sitl --max-speed 5 --target evader --strategy capture
```

`--max-speed` 要和 A 啟動 `agent_server.py --max-speed` 的值相同（預設 3 m/s）。

還在地上的話會先送 takeoff，等三架都到高度層（約 25 秒）再開始。結束時會印出連線統計：收到幾則 FleetPose、送出幾個 GoalPoint、被 A 修改或拒絕幾次與最近的原因、ack 來回延遲，以及 A 的讓路次數（`yields`）與動作結果（`action_results`，規格 §10.8）。

**沒有學校伺服器時**：`fleet_link/fake_server.py` 模仿 A 的協定與安全規則（高度層 25 / 30 / 35 m、目標點推開 6 m、±80 m、速度上限 3 m/s、1 s 沒收到目標點就懸停、3 s 沒任何訊息全部懸停），可以先在自己電腦上測：

```powershell
python -m fleet_link.fake_server
python -m fleet_link.fake_server --max-speed 5 --default-speed 5
```

另開一個終端機跑 `python -m demo.run --fleet sitl`。

注意：
- 連 SITL 時一律照真實時間跑，不能快轉。
- 隊形規劃的目標點速度會自動限制在 `--max-speed`。實測（規則式、10 seeds）追 2～5 m/s 的兔子：3 m/s 時包圍時間比例 14%、5 m/s 47%、8 m/s 74%。
- 規格 §10.8 的執行狀態都會收下來，只記錄、不影響控制：ack 的 `flight_command`（A 實際呼叫的 MAVSDK 函式，`FleetLink.last_flight_command`）、讓路事件（`FleetLink.yielding()` 給面板顯示「誰在讓誰」）、`ActionResult`（`FleetLink.action_log`）。假伺服器會回 `flight_command`，不會產生讓路事件。

