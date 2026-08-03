# """
# Fusion方法比較：RRF vs 加權線性組合(Weighted Sum)
 
# 用法：
# 1. 填入你自己的DB連線資訊、embedding模型、測試查詢清單
# 2. 跑這支腳本，會印出兩種fusion方法在同一批查詢下的Top-5結果
# 3. 你自己人工看結果差異，決定哪個更符合你的需求
 
# 需要的套件：
#     pip install sentence-transformers psycopg2-binary numpy --break-system-packages
# """
 
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
 
# ============ 1. 設定區（改成你自己的） ============
 
DB_CONFIG = {
    "host": "localhost",   # Docker Desktop on Windows會自動把容器port映射到localhost，這裡不用改
    "port": 5432,
    "dbname": "your_db",   # 換成你上面-e POSTGRES_DB設的名稱
    "user": "postgres",    # Docker官方image預設superuser是postgres
    "password": "yourpassword",  # 換成你上面-e POSTGRES_PASSWORD設的密碼
}
 
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 20  # 初步召回筆數（dense/sparse各自撈幾筆候選）
RRF_K = 60  # RRF公式裡的k常數，業界慣例值
 
# 測試查詢清單 —— 建議用你 retrieval_review.txt 裡已經人工看過的那10-20題
TEST_QUERIES = [
    "When I use del as the function name, I will report a syntax error. The other names are correct.",
    "Feature request: undeprecate __version__",
    "Order of commands in help is unreliable in Python <3.6",
    "auto_envvar_prefix does not work with command groups",
    "Callback behavior has changed in 8.0.0 and raises exception now",
    "Different portion of text with different color in echo_via_pager",
    "Release 8.3.3",
    "Prompting options if certain flag is given",
    "Multi-value flags broken by recent change",
    "Completion behaviour for options/arguments with empty incomplete",
]
 
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
 
# ============ 2. Dense檢索（pgvector向量距離） ============
 
def get_dense_results(query: str, conn, top_k: int = TOP_K):
    """回傳 [(doc_id, rank, distance), ...]，rank從1開始，distance越小越相關"""
    query_vec = model.encode(query).tolist()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, top_k),
        )
        rows = cur.fetchall()
    return [(doc_id, rank + 1, distance) for rank, (doc_id, distance) in enumerate(rows)]
 
 
# ============ 3. Sparse檢索（tsvector全文檢索） ============
 
def get_sparse_results(query: str, conn, top_k: int = TOP_K):
    """回傳 [(doc_id, rank, ts_rank_score), ...]，rank從1開始，score越大越相關"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
            FROM documents
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        rows = cur.fetchall()
    return [(doc_id, rank + 1, score) for rank, (doc_id, score) in enumerate(rows)]
 
 
# ============ 4a. RRF融合 ============
 
def rrf_fusion(dense_results, sparse_results, k: int = RRF_K):
    """只用排名，不用原始分數"""
    scores = {}
    for doc_id, rank, _ in dense_results:
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for doc_id, rank, _ in sparse_results:
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
 
 
# ============ 4b. 加權線性組合 ============
 
def _min_max_normalize(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return values
    if values.max() == values.min():
        return np.ones_like(values)
    return (values - values.min()) / (values.max() - values.min())
 
 
def weighted_fusion(dense_results, sparse_results, alpha: float = 0.5):
    """
    alpha: dense的權重，(1-alpha)是sparse的權重
    注意：dense是距離(越小越好)，sparse是分數(越大越好)，方向要對齊
    """
    dense_ids = [d[0] for d in dense_results]
    dense_raw = [d[2] for d in dense_results]
    # 距離轉成「越大越好」：用負號後再normalize，或用1-normalized_distance
    dense_norm = 1 - _min_max_normalize(dense_raw)
 
    sparse_ids = [s[0] for s in sparse_results]
    sparse_raw = [s[2] for s in sparse_results]
    sparse_norm = _min_max_normalize(sparse_raw)
 
    scores = {}
    for doc_id, norm_score in zip(dense_ids, dense_norm):
        scores[doc_id] = scores.get(doc_id, 0) + alpha * norm_score
    for doc_id, norm_score in zip(sparse_ids, sparse_norm):
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * norm_score
 
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
 
 
# ============ 5. 主流程：跑一批查詢，並排印出結果 ============
 
def compare_fusion_methods():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        for query in TEST_QUERIES:
            dense_res = get_dense_results(query, conn)
            sparse_res = get_sparse_results(query, conn)
 
            rrf_top5 = rrf_fusion(dense_res, sparse_res)[:5]
            weighted_top5 = weighted_fusion(dense_res, sparse_res, alpha=0.5)[:5]
 
            print(f"\n===== Query: {query} =====")
            print("RRF Top-5:      ", [doc_id for doc_id, _ in rrf_top5])
            print("Weighted Top-5: ", [doc_id for doc_id, _ in weighted_top5])
            overlap = len(set(d for d, _ in rrf_top5) & set(d for d, _ in weighted_top5))
            print(f"重疊筆數: {overlap}/5")
    finally:
        conn.close()
 
 
if __name__ == "__main__":
    compare_fusion_methods()