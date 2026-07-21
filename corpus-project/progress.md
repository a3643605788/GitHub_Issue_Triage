# progress.md

## 第1週：GitHub語料庫 + 環境建置 + LLM問答腳本

### 進度
- 選定repo: pallets/click (closed issues 1698筆，Sphinx docs 37筆.md檔)
- 完成 fetch_issues.py / load_docs.py / parquet_a.py / LLM_QA.py
- 產出物: corpus.parquet (1735筆), LLM問答腳本可跑(Gemini)

### 決策日誌（錯誤→原則）
| 決策/錯誤 | 錯在哪 | 正確原則 |
|---|---|---|
| 濾掉PR用`"pull_request" not in i` | 一開始不確定為何issues API會混入PR | GitHub把PR視為issue子類型，共用`/issues`端點，PR資料一定帶`pull_request`欄位 |
| GitHub API 401 | 以為是REPO變數打錯，實際是`.env`檔案根本不存在 | 401代表認證機制完全沒生效，不是"哪裡打錯"這種404等級錯誤，要先查認證來源有沒有真的存在 |
| Gemini 404 model not found | 寫死`gemini-2.5-flash`，模型已對新用戶下架 | 這類快速迭代的API，模型名稱容易過期，優先用`-latest`別名，或遇到404先查官方release notes，不要assume自己設定錯 |
| `pathlib.py`檔名撞標準庫 | import Path時觸發circular import | 自訂檔名不能跟標準庫模組同名 |
| 誤以為要手動拼接url字串成`.../pallets/click/issues` | 少了`/repos/`路徑段（現場驗證確實回傳404 Not Found）；且誤解了程式設計邏輯——以為要編輯url本身 | GitHub API的issue端點固定格式是`/repos/{owner}/{repo}/issues`；程式碼設計用f-string自動組url，只需要改`REPO`變數，不要手動改url字串 |
| `fetch_issues(page)`vs`fetch_closed_issues(page)`函示名稱不一致 | 函示宣告的是`fetch_closed_issues()`所以才找不到`fetch_issues()` | 把`fetch_issues()`換成`fetch_closed_issues()` |
| 找不到「Python: Select Interpreter」 | 直接點一般搜尋欄位輸入，而不是用`Ctrl+Shift+P`打開command palette | VS Code的指令（如切換直譯器）只能透過command palette (`Ctrl+Shift+P`)存取，跟檔案內容搜尋欄是不同功能 |