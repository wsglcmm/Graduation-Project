# 多機包圍 Demo（鳥瞰圖 + Agent 指令面板）

3 架無人機在 AI Agent 指揮下，合力搜尋、追上、包圍一個移動目標。左邊是鳥瞰圖，右邊是 Agent 指令面板：每一筆計畫都會顯示時間、規劃者（LLM 或規則式）、延遲、模式、各機角色與理由。

## 執行

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

## 可以調的參數

- `--period`：多久問一次 Agent（預設 2 秒）。越短反應越快，但 API 花費和排隊都會增加。
- `.env` 的 `AGENT_PROVIDER`（`gemini` / `anthropic`）、`AGENT_MODEL`（預設 `gemini-3.1-flash-lite`）、`AGENT_FALLBACK_MODELS`（預設 `gemini-3.5-flash-lite`，逗號分隔，空字串代表不備援）、`AGENT_THINKING`（Gemini 思考程度，例如 `minimal` 可降低延遲）、`AGENT_TIMEOUT_S`（預設 10）。
- `--seed`：換一條兔子路線。
