# CLAUDE.md

## 專案概述

無人機智慧視覺追蹤系統（資工系期末專題，3 人團隊，13 週）。無人機透過 AI 影像辨識自動追蹤地面車輛或行人，並以 LLM Agent 做高階任務決策。全程在模擬環境（PX4 SITL + Gazebo）進行。

資料流：`影像 → B: YOLOv8 + ByteTrack → C: 座標轉換 → C: LLM Agent → A: MAVSDK 控制`

## 我的角色

我是團隊中的 **C**，負責：
1. **座標轉換模組**：像素座標 → 相機座標 → 世界座標（NED）
2. **LLM Agent 設計**：自然語言指令 → 任務規劃 → 白名單動作 JSON

A（模擬/飛控）與 B（影像辨識）是我模組的上下游：A 提供姿態資料與飛控 API，B 提供 bbox 偵測輸出。除非我明確提到要看 A、B 的程式或跨模組介接，否則預設只處理 C 的程式碼。

## 介面規格（必讀）

座標系、單位、時間同步與各介面的 JSON 格式，以此文件為唯一依據：

@docs/interface_spec.md

寫任何程式前先對照該文件。若需要的欄位或約定不在文件內，**先告訴我，不要自行假設**。

## 全域慣例（摘要，細節見規格文件）

- 世界座標 NED（x=North, y=East, z=Down，原點為起飛點）；機體 FRD；相機座標採 OpenCV 慣例（x 右、y 下、z 前）
- 角度用弧度、長度用公尺、時間戳用 int64 奈秒（sim time）
- 目前假設目標在地面平面 `d = 0`，bbox 參考點為底邊中點
- Gazebo 相機座標（x 前、y 左、z 上）與 OpenCV 不同，轉換要明確寫在程式註解

## 目錄結構（建議，依實際 repo 調整）

```
docs/            文件（interface_spec.md 等）
shared/          跨模組共用的資料格式定義（A、B、C 都依賴，修改需走 PR）
calibration/     相機校正 demo（C）
coord_transform/ 座標轉換模組（C）
llm_agent/       LLM Agent（C）
tests/           單元測試
```

## 程式碼規範

- 語言：Python 3；風格遵循 PEP 8，函式加 type hints
- 資料格式用 dataclass 或 pydantic 定義，不要在各處手寫 dict
- 每個座標轉換函式的 docstring 必須寫明：輸入/輸出的座標系、單位
- 座標轉換、時間插值等數學邏輯要有單元測試（可用已知答案驗證）
- 說明用繁體中文，專有名詞保留英文（YOLO、MAVLink、NED 等）
- 提供程式碼前先用 2–3 句說明邏輯與設計理由，不要冗長鋪陳

## Git 規範（由 Claude 直接操作）

git 流程由你直接完成（開分支、commit、push、開 PR），不需要我手動操作。完成後簡短回報：分支名稱、commit 清單（hash + 訊息）、測試結果，以及需要我知道或決定的事。

規則：
- **分支**：我的工作在 `feature/coordinate-transform`、`feature/llm-agent`、`feature/multi-drone-fusion`、`feature/demo-dashboard` 等分支，不直接改 `main`。文件類變更用 `docs/...` 分支
- **Commit 訊息**：Conventional Commits，例如 `feat(coord): 新增相機內參校正模組`、`fix(agent): 修正任務規劃 prompt 的邊界條件`、`docs: 更新模組介面規格文件`；每個 commit 只做一件事
- **push 前**：測試必須全部通過（`python -m pytest -q`），並自行檢查 (a) 沒有大型檔案或暫存檔（模擬影片、模型權重 `.pt`、校正圖片、模擬紀錄放 `.gitignore`），(b) commit 訊息清楚，(c) 是否需要開 PR 讓 A、B 看過
- **`shared/` 與 `docs/interface_spec.md`**：修改會影響 A、B，一律開 Pull Request，不直接 push 到 `main`
- **先問我才能做**：合併任何 PR 到 `main`、force push、改寫已推上去的歷史、刪除遠端分支、修改 A 或 B 負責的程式
- **絕對不 commit**：`.env`、API 金鑰、模型權重、資料集與模擬紀錄檔
- **同步**：在我電腦上工作時（例如 VS Code 裡的 Claude Code），開始前先 `git pull`，避免和雲端推上去的 commit 衝突
- **權限**：push 被拒絕時直接告訴我，不要改用其他方式繞過

## 安全

- 這是飛行器控制專案：LLM 輸出不得直接成為底層控制量，只能是規格文件定義的白名單動作 JSON，且須先通過 schema 驗證
- 最大速度、高度限制、地理圍欄等安全限制由 A 的飛控層負責，Agent 程式不得繞過
- 若涉及真機飛行，提醒我確認當地法規與場地許可

## 回答方式

- 技術選型請同時說明「業界主流做法」與「適合期末專題複雜度的做法」，並指出哪些值得深入鑽研、哪些可先用現成工具跳過
- 涉及會隨時間變動的資訊（套件版本、API 變更），請先查證再回答，不要憑記憶推測
- 我是資工系學生，請以教授指導的角度回覆：指出重點、容易出錯的地方，以及口試可能被追問的問題
