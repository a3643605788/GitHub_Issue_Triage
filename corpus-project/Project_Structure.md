# 專案結構說明

text

```text
corpus-project/
├── .venv/                          # Python虛擬環境(uv管理)
├── .env                            # 環境變數(GitHub token等，不進版控)
├── .python-version                 # Python版本鎖定
├── corpus.parquet                  # 主要語料庫(1735筆: issue+doc)
├── raw_issues.json                 # GitHub API原始issue資料(fetch_issues.py產出)
├── pyproject.toml                  # 專案依賴設定(uv)
├── uv.lock                         # 依賴版本鎖定檔
├── progress.md                     # 專案進度與決策日誌
├── retrieval_review.txt            # 【保留-證據】embedding選型人工標註(28/38/26對50命中率來源)
│
├── fetch_issues.py                 # 【Week1】從GitHub API抓取closed issues
├── load_docs.py                    # 【Week1】讀取click官方.md文件
├── parquet_a.py                    # 【Week1】整合issue+doc輸出成corpus.parquet
├── LLM_QA.py                       # 【Week1】基本LLM問答腳本(Gemini)
│
└── experiments/
    └── second_week/                # Week2: RAG基礎版技術選型與評估
        ├── README.md                        # 本週檔案取捨判斷紀錄
        ├── test_retrieval.py                # 【證據鏈①embedding選型】3模型命中率比較，產出retrieval_review.txt同等格式資料
        ├── test_setup_pgvector.py           # 【Pipeline核心】建表+讀取corpus.parquet+產生embedding+寫入pgvector+建tsvector
        ├── test_fusion_compare.py           # 【證據鏈②fusion選型】RRF vs Weighted Sum比較(Top5一致率、候選集合重疊)
        ├── test_export_fusion_candidates.py # 【證據鏈③人工標註前置】輸出dense+sparse+RRF候選名單，產出test_fusion_review.txt(未標記版)
        ├── test_fusion_review.txt           # 【未標記版】剛跑完腳本產出的空白候選名單(相關嗎? [ ]是 [ ]否)
        ├── test_fusion_review_result.txt    # 【已標記版-證據鏈③結果】10題×10筆人工標註完成，reranker Precision@5的golden label來源
        └── test_compute_precision.py        # 【證據鏈④reranker最終選型】用golden label算3個reranker的Precision@5+延遲
```

---

## 證據鏈總覽(可對照面試/README引用)

Week2完整決策鏈： test_retrieval.py (embedding選型: 28/38/26對50) → test_setup_pgvector.py (依選定embedding建置pgvector) → test_fusion_compare.py (fusion方法選型: RRF vs Weighted 9/10題一致 + 架構性等價論證) → test_export_fusion_candidates.py + test_fusion_review_result.txt (10題人工標註golden label) → test_compute_precision.py (reranker選型: Precision@5 0.44/0.44/0.46, 延遲1.8s/3.0s/10.3s)

## 已知技術債(詳見progress.md)

- Chunking策略已選定(Recursive chunking)但未實作，目前embedding皆為整篇文件向量
- Precision@5存在天花板效應(部分query僅1篇golden label，上限0.20)

## 待確認事項

- 目前無