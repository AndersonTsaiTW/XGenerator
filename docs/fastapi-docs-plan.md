# FastAPI 靜態文件網站規劃（XGenerator）

## 1. 目標與範圍

### 目標
- 在 FastAPI 服務停止時，仍可提供可閱讀的 API 文件網站。
- 以 OpenAPI 規格為單一真實來源（Single Source of Truth）。
- 文件自動化更新，避免程式與文件不同步。

### 範圍
- 匯出 OpenAPI JSON。
- 用 ReDoc 產生靜態文件頁。
- 以 GitHub Actions 自動建置並部署到 GitHub Pages（或可替換成 Netlify / Vercel）。

### 非目標（第一階段先不做）
- 在文件站上直接呼叫私有 API（Try it out）並處理複雜授權流程。
- 多版本文件切換 UI（v1, v2, ...）。

## 2. 推薦實作策略（務實最小可行）

### 為什麼選 ReDoc + 靜態站
- 對 OpenAPI 支援完整，輸出乾淨。
- 只要靜態檔案，可低成本託管。
- 可快速從目前 FastAPI 專案導入，不需大改程式。

### 核心流程
1. 從 FastAPI app 取得 OpenAPI schema。
2. 產出 `openapi.json` 到 docs 產物目錄。
3. 產生 `index.html`（ReDoc 載入 `openapi.json`）。
4. CI 在 push 到主分支時自動重建與部署。

## 3. 建議目錄結構

```text
docs/
  fastapi-docs-plan.md         # 本規劃
  guides/                      # 人寫的教學與操作文件
  api/                         # API 補充文件（命名規則、範例請求等）
  plans/                       # 規劃與 ADR 類文件
  site/
    index.html                 # 靜態文件入口（ReDoc，建置產物）
    openapi.json               # 匯出的 API 規格（建置產物）
```

備註：
- 你可以把很多文件放在 `docs/`，這是常見且推薦的做法。
- 只要把「手寫文件」與「自動建置產物」分層管理即可。
- `docs/site/` 建議視為部署產物目錄，避免手動編修。

### 文件分層規則（建議）
- 人工維護文件：放在 `docs/guides/`、`docs/api/`、`docs/plans/`。
- 自動生成文件：只放在 `docs/site/`。
- CI 僅覆蓋 `docs/site/`，不改動其他文件。
- 若未來導入 MkDocs，可把來源維持在 `docs/guides/` 與 `docs/api/`，再輸出到 `site/`（或 `public/`）。

## 4. 技術落地步驟

### Step A：OpenAPI 匯出腳本
- 新增一支腳本（例如 `scripts/export_openapi.py`）。
- 在腳本中 import FastAPI app（目前看起來可從 `app/main.py` 取得 app 實例）。
- 呼叫 `app.openapi()` 並寫入 `docs/site/openapi.json`。

### Step B：建立 ReDoc 靜態頁
- 新增 `docs/site/index.html`。
- 用 ReDoc CDN 載入 `openapi.json`。
- 設定標題、favicon（可後續補）。

### Step C：本機驗證
- 執行 OpenAPI 匯出。
- 以任一靜態伺服器開啟 `docs/site/`（例如 python http.server）。
- 確認 endpoint、schema、範例都可瀏覽。

### Step D：CI/CD 自動化
- 新增 GitHub Actions workflow：
  - 安裝 Python 依賴。
  - 執行匯出腳本。
  - 發佈 `docs/site/` 到 GitHub Pages。
- 設定觸發條件：
  - push 到主分支。
  - 可加上手動觸發（workflow_dispatch）。

## 5. 與現有專案整合注意事項

### 1) 匯入 app 時的副作用
- 若 `app/main.py` 在 import 時會連外部資源（例如 Redis、Celery）可能導致匯出失敗。
- 建議把 OpenAPI 匯出流程設計成「最少依賴」模式，必要時以環境變數關閉非必要初始化。

### 2) 安全資訊
- 不要把敏感範例、內部主機名稱、私密欄位描述輸出到公開文件。
- 若是公開站，需先檢查 schema 中是否有內部細節。

### 3) 文件可用性與 Try it out
- 靜態站可完整閱讀文件。
- 若 API server 不在線，Try it out 無法實際呼叫，屬正常行為。

## 6. 第一階段交付定義（Definition of Done）
- 有固定可訪問網址可閱讀 API 文件。
- 每次主分支更新後，文件會自動重新部署。
- `openapi.json` 與程式碼同步更新。
- README 有一段「如何更新文件」說明。

## 7. 第二階段可擴充項目
- 加入版本化文件（v1/v2）。
- 導入 MkDocs，整合 API 文件與操作手冊。
- 在 CI 新增 schema 變更檢查（例如 breaking changes 檢測）。
- 客製化 ReDoc 樣式（品牌色、標誌、導覽）。

## 8. 建議的執行順序（你現在可以這樣做）
1. 先做 Step A + Step B（把靜態頁跑起來）。
2. 在本機確認文件內容正確。
3. 再補 Step D（自動部署）。
4. 最後更新 README 的文件維運說明。

---

如果你願意，我下一步可以直接幫你把以下檔案一次建好：
- `scripts/export_openapi.py`
- `docs/site/index.html`
- `.github/workflows/publish-docs.yml`
- `README.md` 的「API 文件」章節
