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

### 接上 Claude（LLM 模式）

1. 把 `env.example` 複製成 `.env`，填入你的 Anthropic API 金鑰。`.env` 已在 `.gitignore`，**絕對不要 commit**。
2. 執行：

```powershell
python -m demo.run --planner llm
```

視窗下方可以輸入自然語言指令，按 Enter 送出，例如「三架一起把那台車包圍起來」「先懸停」「回家」。LLM 回覆通常要一到數秒；這段時間無人機會照上一個合法計畫飛，不會停下來等。

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
| 指揮官 | 規則式或 Claude Haiku 5.5 | 不變 |

## 安全設計（口試重點）

- LLM **只決定角色與模式**，不輸出座標或速度；座標由確定性的隊形規劃器計算。
- LLM 輸出一律經 strict schema 與語意檢查（每架恰好一次、目標存在），不合法就丟棄並**沿用上一個合法計畫**；還沒有計畫時改用規則式。
- 無人機超過 1 秒沒收到新目標點就懸停；速度、加速度上限與機間互斥在替身中模擬，正式版由 A 的飛控安全層最終把關。

## 可以調的參數

- `--period`：多久問一次 Agent（預設 2 秒）。越短反應越快，但 API 花費和排隊都會增加。
- `.env` 的 `AGENT_MODEL`（預設 `claude-haiku-5-5`，官方標示延遲最低）、`AGENT_EFFORT`（預設 `low`，降低延遲）、`AGENT_TIMEOUT_S`（預設 10）。
- `--seed`：換一條兔子路線。
