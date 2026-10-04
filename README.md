# 無人機智慧視覺追蹤系統

資訊工程系期末專題。無人機透過 AI 影像辨識自動追蹤地面車輛或行人，並結合 LLM / AI Agent 進行高階任務決策。全程於模擬環境（PX4 SITL + Gazebo）進行。

## 系統架構

```
影像 → YOLOv8 + ByteTrack → 座標轉換 → LLM Agent → MAVSDK 飛控
        (B)                    (C)         (C)         (A)
```

## 團隊分工

| 成員 | 負責範圍 |
|---|---|
| A | 模擬環境與飛控（PX4 SITL、Gazebo、MAVSDK） |
| B | 影像辨識（YOLOv8、ByteTrack） |
| C | 座標轉換模組、LLM Agent 設計 |

## 目錄結構

```
docs/            文件（模組介面規格等）
shared/          跨模組共用的資料格式定義
calibration/     相機校正 demo
coord_transform/ 座標轉換模組
llm_agent/       LLM Agent
tests/           單元測試
```

## 協作規範

- 分支：不直接修改 `main`，各自在 `feature/...` 或 `docs/...` 分支開發
- Commit 訊息：Conventional Commits（例如 `feat(coord): ...`、`fix(agent): ...`、`docs: ...`）
- `shared/` 與 `docs/interface_spec.md` 的變更需開 Pull Request，經相關成員確認後才合併
- 模型權重、模擬影片、API key 不得 commit（已列入 `.gitignore`）
