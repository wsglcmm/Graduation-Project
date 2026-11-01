# Demo 鳥瞰影片候選（給 B 一起確認）

> 結論先講：**首選 Stanford Drone Dataset（SDD）的 `nexus` 場景**，次選 `deathCircle`。最後要選哪一段影片、哪一個目標，用 `python -m demo.sdd_scan` 對實際標註掃描後再決定。

## 挑選條件

| 條件 | 理由 |
|---|---|
| 空拍機懸停、俯拍 | 地面標定（homography）只要做一次；鏡頭會動的影片要逐幀重新標定 |
| 有移動的車或騎車的人，連續可見 ≥ 30 秒 | 要有時間讓三架無人機追上並包圍 |
| 授權清楚 | 期末報告與 Demo 要能公開展示並註明出處 |
| 有標註 | B 的偵測結果可以和標註比對準確度；B 還沒做完時，也能先用標註當偵測結果 |

## SDD 各場景（依分析論文的統計）

| 場景 | 適合度 | 說明 |
|---|---|---|
| **nexus** | ★★★ | 車輛比例最高（約 29.5%），最像「兔子是一台車」的情境 |
| **deathCircle** | ★★ | 圓環，騎車的人約 56%、車約 4.7%；轉彎多，包圍畫面好看 |
| little、gates | ★ | 騎車的人過半，但車很少 |
| quad、coupa | ✗ | 幾乎都是行人（quad 沒有車），移動慢、目標太小 |

SDD 共 8 個場景、60 段影片，標註 30 fps，授權為 **CC BY-NC-SA 3.0**（非商業、需註明出處、衍生作品同授權），課堂專題可以用，報告要註明出處。

## 已知的坑（分析論文指出）

- **很多車是停著的**：要排除不動的目標，`sdd_scan` 會用「移動距離」篩掉。
- **lost 標註**：目標出畫面時標註仍存在（lost = 1），常停在畫面邊緣不動，**不能當成真實位置**。
- **同一個人可能被拆成多個 track ID**，也有幾段影片拍的是同一地點，挑選時要人工看一下。
- **SDD 沒有提供像素對應公尺的比例**。地面標定的建議做法：在影片畫面上選 4 個以上容易辨認的地面點（路口、轉角、地面標線），再到 Google 地圖的史丹佛校園量出它們之間的實際距離，填進 `config/video_ground.json`；也可以拿車長約 4.5 m 做粗估與檢查。

## 怎麼用（拿到資料後）

1. 下載 SDD（含 `videos/` 與 `annotations/`），放在 `data/sdd/`（`data/` 已在 `.gitignore`）。
2. 掃描候選：

```powershell
python -m demo.sdd_scan data/sdd
```

會列出「車輛優先、連續可見時間最長」的候選，完整清單寫到 `output/sdd_candidates.csv`。

3. 和 B 一起看影片，選定一段與一個目標，接著做地面標定。

## 參考

- Stanford Drone Dataset：https://cvgl.stanford.edu/projects/uav_data/
- 分析論文：*The Stanford Drone Dataset is More Complex than We Think*（arXiv 2203.11743）
- 標註格式：https://github.com/flclain/StanfordDroneDataset
