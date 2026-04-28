# PRD — Eta EMS 離線版 Stage 1–5

**對照來源**：Notion「Eta EMS — AI 建築能源最佳化控制系統完整建置計畫」及 Stage 1–5 子頁（CSV-first；Stage 6 前不接即時 API）。  
**專案**：missionnnn（本 repo 根目錄）。  
**版本**：0.1  
**狀態**：草案 — 與程式庫對照後整理。

---

## 1. 現況 vs Notion：缺口摘要

| Stage | Notion 要交付什麼 | 專案現況 | 缺口（相對 Notion 驗收） |
|-------|-------------------|----------|---------------------------|
| **1** CSV 管線 | `engine/core/*` + `HVACAddon`、九步 pipeline、`q_delivered_kw` / `is_steady`、notebook + pytest、Core 不 import addons、可選第二個最小 add-on | 已有 pipeline、HVAC B+ 特徵、MockAddon 測試、`notebooks/01_data_pipeline.ipynb` | **import-linter / CI** 未設；**正式第二領域 add-on**（空壓）未做；Notion 列的 Polars / Pandera / GE 等未強制對齊 |
| **2** 預測 | `core/ml`：trainer、Optuna、MLflow、SHAP；P_model；Q_demand（quantile）；`02_prediction.ipynb` | 已有 trainer、benchmark、q_demand、HVACQDemandPart、`mlruns/`、`ml_service` | **Notebook 02** 缺；`engine/core/ml/optimizer.py` **直接 import `engine.addons.hvac.features`**，違反「Core ⊥ addons」，應重構為 Protocol / 注入 |
| **3** 最佳化 | DecisionPart、Optuna 逐時段、savings 報告、`03_optimization.ipynb`、L1/L2/L3 範圍 | 已有 Optuna + Q 約束、tests；hvac_optimizer 另有 scipy L-BFGS-B + 物理 | **HVACDecisionPart** 未獨立；**兩套最佳化**需產品級統一；**optimization_results.parquet + savings_report.json** 未標準化；**Notebook 03** 缺 |
| **4** MPC | rolling_horizon、`simulate_dynamics`、rate limit、SafetyChecker、MPC 回放、MPC vs Stage3 節能比 ≥0.85、`04_mpc_simulation.ipynb` | 幾乎未實作 | **整個 Stage 4** 為最大缺口 |
| **5** Dashboard | `ui_framework`、DashboardPart、五區塊、WeasyPrint + Jinja PDF | hvac_optimizer Streamlit + `ui/` AI Studio | **Core + DashboardPart** 未建；**MPC 頁**、**PDF 月報**、與 Notion 五頁結構對齊待補 |

**架構債務**：`engine/core/ml/optimizer.py` 對 `hvac.features` 的 import 需移除；hvac_optimizer 與 engine 雙線需定義單一主線。

---

## 2. 產品概述

### 2.1 背景與問題

案場 CSV / BAS 匯出需可重複清洗、可驗收預測、可解釋節能建議；在 **不連 BAS** 下做到可 demo、可出報告。

### 2.2 目標（Stage 1–5 結束時）

- 路徑打通：**CSV → 清洗 parquet → P_model + Q_demand → 最佳化 →（Stage 4）MPC 離線回放 → Dashboard + PDF**。
- 架構符合 **Core + Add-on**：Core 設備無關；HVAC 知識在 `engine/addons/hvac/`。

### 2.3 範圍

| 在範圍內 | 不在範圍內 |
|----------|------------|
| Stage 1–5 離線能力 | Stage 6 即時 API、Stage 7 BACnet 閉環 |
| 單租戶、單案場 CSV | 多租戶 SaaS、K8s、Edge 量產 |

### 2.4 使用者故事（精簡）

1. 上傳 HVAC CSV → 自動 mapping、清洗 → parquet + 品質報告。  
2. 同一 parquet 訓練功率模型與 Q_demand P90 → MAPE / coverage / SHAP。  
3. 在 L1/L2/L3 內搜尋 setpoints → 最小化預測功率且 **Q_capability ≥ Q_demand × margin**。  
4. Stage 4：滾動視野 + rate limit + 安全 clip 回放 → 控制日誌；節能不劣於全知 Stage 3 約定比例。  
5. 一頁式 KPI + 圖 + **PDF** 對外簡報。

---

## 3. 功能需求（FR）

### Stage 1 — 資料管線

| ID | 需求 | 驗收 |
|----|------|------|
| FR-1.1 | `run_pipeline(csv, addon, timestep)` → `CleanResult` + 品質報告 | pytest 通過 |
| FR-1.2 | HVAC 衍生 **`q_delivered_kw`、`is_steady`** | 欄位契約文件化 |
| FR-1.3 | **Core 不得 import `engine.addons.*`** | CI：import-linter 或同等 |
| FR-1.4 | 第二個最小 add-on（空壓）+ 同 pipeline | 測試 + 範例 CSV |

### Stage 2 — 預測

| ID | 需求 | 驗收 |
|----|------|------|
| FR-2.1 | P_model：XGBoost + Optuna + 時序 CV + holdout | MAPE、CV(RMSE) 依 Notion 門檻 |
| FR-2.2 | Q_demand：LightGBM quantile α=0.9，coverage 0.85–0.93 | 自動驗收報告 |
| FR-2.3 | `notebooks/02_prediction.ipynb` | 可重現訓練 |

### Stage 3 — 最佳化（Advisory）

| ID | 需求 | 驗收 |
|----|------|------|
| FR-3.1 | HVACDecisionPart（或 Protocol）：控制變數、約束、rate limit | 對齊 Notion DecisionPart |
| FR-3.2 | 最佳化器僅依賴 Protocol，不依賴具體 hvac 模組 | 靜態檢查 + 單測 |
| FR-3.3 | `optimization_results.parquet` + `savings_report.json`（或 API 同等 schema） | 與 Dashboard 約定 |
| FR-3.4 | `notebooks/03_optimization.ipynb` | 新增 |
| FR-3.5 | hvac_optimizer 與 engine **單一官方優化路徑** | README / deprecation 說明 |

### Stage 4 — MPC 模擬

| ID | 需求 | 驗收 |
|----|------|------|
| FR-4.1 | `simulate_dynamics`：狀態、動作、預測 → 下一狀態 + 功率 | 單測 + 介面對齊 Notion |
| FR-4.2 | `get_rate_limits` + SafetyChecker（hard clip + rate limit） | raw vs safe 可稽核 |
| FR-4.3 | `run_mpc` / MPCSimulator：滾動視野，每步只執行第一步 | 一個月資料可跑；CHWS 每步 ≤0.5°C 等 |
| FR-4.4 | MPC 節能 / Stage3 全知節能 ≥ **0.85** | 自動報表 |
| FR-4.5 | `notebooks/04_mpc_simulation.ipynb` + Streamlit MPC 頁 | 與 Stage 5 導覽一致 |

### Stage 5 — Dashboard 與報告

| ID | 需求 | 驗收 |
|----|------|------|
| FR-5.1 | DashboardPart + WidgetSpec（可先最小集合） | HVAC widgets 清單 |
| FR-5.2 | 五區塊：數據品質 / 模型 / 優化 / MPC / 報告 | 對齊 Notion KPI 與圖表 |
| FR-5.3 | PDF：Jinja + WeasyPrint（或既定替代） | 模板版本 + 範例輸出 |
| FR-5.4 | Demo 劇本：上傳→清洗→訓練→優化→（MPC）→報告 ≤30 分鐘 | README / 內訓文件 |

---

## 4. 非功能需求（NFR）

| ID | 類別 | 需求 |
|----|------|------|
| NFR-1 | 可維護性 | Core/addon 邊界可機械檢查；公開 API 具型別與 docstring |
| NFR-2 | 可重現性 | Notebook + seed + 資料版本或 hash |
| NFR-3 | 效能 | Stage 3 單步、Stage 4 全日回放時間上限（依 Notion 訂 KPI） |
| NFR-4 | 安全 | 離線版無帳號；Stage 6 再接 auth |

---

## 5. 里程碑

| 里程碑 | 內容 | 完成定義 |
|--------|------|----------|
| M1 | 修憲法 + Stage 1 CI | import-linter 綠燈；optimizer 無 hvac import |
| M2 | Stage 2/3 文件與產物 | notebooks 02–03；標準 parquet/json schema |
| M3 | Stage 4 MVP | simulate_dynamics + MPC 回放 + 驗收指標 |
| M4 | Stage 5 封版 | DashboardPart + PDF + 五區塊 demo |

---

## 6. 風險與依賴

- 兩套最佳化實作長期分裂會拖慢 Stage 4/5。  
- WeasyPrint 在 Windows 環境較重，可預留 HTML 匯出或 Docker。  
- Stage 4 動態模型過簡可能難達 MPC / Stage3 比值 0.85。

---

## 7. 待決議題

1. 主入口：以 `hvac_optimizer` 為唯一 demo，或合併 `ui/` AI Studio？  
2. Stage 3：Optuna（Notion）與 scipy L-BFGS-B（現有 physics）並存或淘汰一條？  
3. PDF 是否必須 WeasyPrint，或先接受 HTML/PNG？

---

## 8. 修訂紀錄

| 日期 | 版本 | 說明 |
|------|------|------|
| 2026-04-28 | 0.1 | 初稿：對照 Notion Stage 1–5 與 missionnnn 程式庫掃描結果 |
