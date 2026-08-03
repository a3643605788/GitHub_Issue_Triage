| 檔案                                 | 判斷     | 理由                                                          |
| ---------------------------------- | ------ | ----------------------------------------------------------- |
| `test_retrieval.py`                | **保留** | 產出`retrieval_review.txt`，是你embedding選型（28/38/26對50）的原始證據來源  |
| `test_setup_pgvector.py`           | **保留** | Production pipeline的建表/ingestion邏輯，非評估腳本                    |
| `test_fusion_compare.py`           | **保留** | 你fusion方法選型（RRF vs Weighted 9/10題一致）的實測證據                   |
| `test_export_fusion_candidates.py` | **保留** | 產出`fusion_review.txt`，是你reranker Precision@5的golden label來源 |
| `test_fusion_review.txt`           | **保留** | 人工標註結果本身，是最終Precision@5計算的直接輸入                              |
| `test_compute_precision.py`        | **保留** | 你reranker最終選型（Precision@5+延遲）的決定性證據                         |

`test_retrieval.py`（embedding選型）→ `test_setup_pgvector.py`（建置pipeline）→ `test_fusion_compare.py`（fusion選型）→ `test_export_fusion_candidates.py` + `test_fusion_review.txt`（人工標註）→ `test_compute_precision.py`（reranker最終選型）。