# progress.md

## 第1週：GitHub語料庫 + 環境建置 + LLM問答腳本

### 進度

- 選定repo: pallets/click (closed issues 1698筆，Sphinx docs 37筆.md檔)
- 完成 fetch_issues.py / load_docs.py / parquet_a.py / LLM_QA.py
- 產出物: corpus.parquet (1735筆), LLM問答腳本可跑(Gemini)

### 決策日誌（錯誤→原則）

|決策/錯誤|錯在哪|正確原則|
|---|---|---|
|濾掉PR用`"pull_request" not in i`|一開始不確定為何issues API會混入PR|GitHub把PR視為issue子類型，共用`/issues`端點，PR資料一定帶`pull_request`欄位|
|GitHub API 401|以為是REPO變數打錯，實際是`.env`檔案根本不存在|401代表認證機制完全沒生效，不是"哪裡打錯"這種404等級錯誤，要先查認證來源有沒有真的存在|
|Gemini 404 model not found|寫死`gemini-2.5-flash`，模型已對新用戶下架|這類快速迭代的API，模型名稱容易過期，優先用`-latest`別名，或遇到404先查官方release notes，不要assume自己設定錯|
|`pathlib.py`檔名撞標準庫|import Path時觸發circular import|自訂檔名不能跟標準庫模組同名|
|誤以為要手動拼接url字串成`.../pallets/click/issues`|少了`/repos/`路徑段（現場驗證確實回傳404 Not Found）；且誤解了程式設計邏輯——以為要編輯url本身|GitHub API的issue端點固定格式是`/repos/{owner}/{repo}/issues`；程式碼設計用f-string自動組url，只需要改`REPO`變數，不要手動改url字串|
|`fetch_issues(page)`vs`fetch_closed_issues(page)`函示名稱不一致|函示宣告的是`fetch_closed_issues()`所以才找不到`fetch_issues()`|把`fetch_issues()`換成`fetch_closed_issues()`|
|找不到「Python: Select Interpreter」|直接點一般搜尋欄位輸入，而不是用`Ctrl+Shift+P`打開command palette|VS Code的指令（如切換直譯器）只能透過command palette (`Ctrl+Shift+P`)存取，跟檔案內容搜尋欄是不同功能|

---

## 第2週：RAG基礎版 — embedding/向量DB/hybrid search/reranking

### 進度

- 完成 embedding 模型選型（BGE-base-en-v1.5）
- 完成 pgvector 環境建置（Docker，非本機安裝）
- 完成 dense + sparse + RRF fusion 架構
- 完成 reranker 選型（MiniLM-L6）
- 完成 10題人工標註（`fusion_review.txt`），並用於計算 Precision@5
- 產出物：`setup_pgvector.py`（建表+ingest）、`fusion_compare.py`、`export_fusion_candidates.py`、`reranker_eval.py`、`compute_precision.py`

### 技術選型（含理由）

**Embedding：`BAAI/bge-base-en-v1.5`**

- 從1735筆抽樣10題，每題Top5命中率：
    - MiniLM-L6：28/50 (56%)
    - **BGE-base-en-v1.5：38/50 (76%)** ← 選定
    - multilingual-e5-base：26/50 (52%)

**向量DB：`pgvector`（非Chroma）**

- 原先傾向Chroma（安裝快），但這是對不熟悉工具的迴避，非量化決策
- 最終依據：pgvector資料寫在容器外部，支援Cloud SQL持久化+Cloud Run無狀態部署；Chroma預設寫進容器內本地檔案，對Cloud Run而言不構成持久化

**Chunking：`Recursive chunking`（策略選定，**未實作**，見下方技術債）

**Hybrid Search Fusion：`RRF`**

- 候選：RRF / 加權線性組合(Weighted Sum) / CombSUM-MNZ / Learned Fusion(reranker取代融合)
- 排除CombSUM/MNZ：對分數尺度敏感，現代業界少用
- RRF vs Weighted Sum：Top5比對9/10題完全一致
- 架構性理由：`rrf_fusion`與`weighted_fusion`皆對同一組`dense_results ∪ sparse_results`計分，兩者候選**集合**必然100%相同，差異僅在候選內部排序；而排序最終會被reranker重新決定，故兩方法在此架構下實質等價
- 選RRF：等價前提下，選運算更簡單、零調參的方案（Occam's razor）

**Reranker：`cross-encoder/ms-marco-MiniLM-L-6-v2`**

- 候選：MiniLM-L6 / MiniLM-L12 / bge-reranker-base（-large、v2-m3、v2-gemma因CPU inference延遲29s~93s/query，判定不可用於即時場景，排除測試）
- 平均延遲：MiniLM-L6 1.75s／MiniLM-L12 3.0s／bge-reranker-base 10.3s
- 平均Precision@5：MiniLM-L6 0.44／MiniLM-L12 0.44／bge-reranker-base 0.46
- 決策：Precision@5差距僅0.02（且10題樣本量小，此差距在雜訊範圍內），延遲差距達5.7倍；選MiniLM-L6

### 已知技術債

**Chunking未實作**

- 計畫選定策略：Recursive chunking（保留完整段落/句子/詞邊界，語意層面交給embedding處理）
- 實際狀況：`setup_pgvector.py`的ingest流程直接對整篇`content`(完整issue內文/完整.md文件)做embedding，未實作任何chunking，1735筆embedding皆為整篇文件向量
- 風險：
    - 長文件(尤其部分.md文件)可能超出BGE-base-en-v1.5上下文長度限制，導致截斷或語意稀釋
    - 目前所有已完成的評估數字(embedding選型28/38/26對50、hybrid search、reranker Precision@5)皆建立在「未chunking」的前提下，若後續補做chunking，這些數字可能需要重新驗證
- 決定：因應第4週硬性時間盒，先不回頭實作，記錄此技術債，帶著限制進入Week 3
- 面試/README揭露：需誠實說明「chunking策略已選定但受時間限制未實作驗證」，不可包裝成已完成

**Precision@5的天花板效應**

- 部分查詢golden label僅1篇相關文件，Precision@5理論上限為0.20，與其他模型比較時會被此上限掩蓋真實差距
- 面試talking point：展示對評估指標局限性的理解

### 決策日誌（錯誤→原則）

|決策/錯誤|錯在哪|正確原則/原因|
|---|---|---|
|pgvector vs Chroma|原先選用`Chroma`，原因是可直接使用速度快，但其實`pgvector`的安裝不會太冗長太久，這個原因只是用時間效益對不熟悉工具的迴避|做選擇要有具體量化的指標。最後選`pgvector`原因是可做Cloud SQL持久化，支援Cloud Run無狀態部署；`Chroma`預設寫進指定的本地檔案，對Cloud Run來說，這是一個暫時的檔案，不構成持久化。簡言之，`pgvector`資料寫在容器外部，`Chroma`資料寫在容器內部。|
|`psycopg2` vs `psycopg2-binary`|直接裝`psycopg2`，在Windows上需要編譯C擴充套件，容易安裝失敗|Windows開發環境一律用`psycopg2-binary`（預編譯版），正式production環境才考慮source版|
|Docker daemon連線失敗|誤以為是指令語法問題|錯誤訊息是連不到Docker daemon，代表Docker Desktop應用程式本身沒啟動，跟指令語法無關|
|中文路徑導致檔案找不到|專案路徑含中文字元「D槽」，Python在某些情境下（尤其搭配輸出重導向`>`）會將路徑解析為亂碼，導致`FileNotFoundError`|Windows開發應一律使用純ASCII路徑；此問題在本機執行階段已反覆出現，Docker化/Cloud Run部署前必須將整個專案搬遷至無中文字元的路徑|
|輸出重導向`>`寫入檔案出現中文亂碼|誤以為`chcp 65001`能解決寫檔編碼問題|`chcp 65001`只改變終端機**顯示**編碼，不影響Python**寫入檔案**時的編碼；根本解法是程式碼內部明確指定`encoding="utf-8"`寫檔，不依賴終端機/系統編碼設定|
|SQL查詢欄位名稱`type`寫成`type`但建表用`doc_type`|兩支腳本（`setup_pgvector.py`與`export_fusion_candidates.py`）欄位命名沒有保持一致|多支腳本共用同一張表時，欄位名稱需要統一命名，避免各自腳本使用不同名稱造成`UndefinedColumn`錯誤|
|Precision@5空陣列導致`ValueError`|`weighted_fusion`裡`_min_max_normalize`未處理sparse檢索完全無結果(空陣列)的情境|純關鍵字比對(tsvector)可能因查詢用詞與文件字面無重疊而完全撈不到結果，這本身也是hybrid search存在價值的例證（dense能補足sparse的字面侷限）；程式邏輯需針對空結果做防呆|
|誤以為reranker需要對全部1735筆語料做效能評估|混淆了「reranker的評估範圍」與「檢索系統整體的語料規模」|Reranker只對dense+sparse+fusion後的初步候選池(Top20量級)做精選，不會也不需要看過整個語料庫；兩階段檢索架構的意義正是靠粗篩(dense/sparse)縮小範圍，reranker才在小範圍內做精確排序|
|誤以為RRF vs Weighted Sum需要比較「候選集合」是否重疊|兩個fusion函式皆對同一組`dense_results ∪ sparse_results`計分，候選集合本身architecturally保證100%相同，此驗證邏輯上必然得到平凡結果|Fusion方法在此架構下只影響候選池內部排序，不影響候選池組成；當排序最終會被reranker覆蓋時，需重新評估該用什麼證據來比較fusion方法的實質差異|