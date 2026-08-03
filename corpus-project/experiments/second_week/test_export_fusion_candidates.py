# """
# 輸出 dense+sparse+RRF 融合後的候選名單，格式仿照 retrieval_review.txt，
# 供人工標註「相關/不相關」，作為之後算reranker Precision@5的golden label。
 
# 用法：
#     python export_fusion_candidates.py > fusion_review.txt
# """
 
import psycopg2
from sentence_transformers import SentenceTransformer
 
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "your_db",
    "user": "postgres",
    "password": "yourpassword",
}
 
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RECALL_TOP_K = 20
RRF_K = 60
SHOW_TOP_N = 10  # 印出前幾筆給人工標註（20筆全標太累，先標前10筆通常夠用）
 
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
 
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
 
def get_dense_results(query, conn, top_k=RECALL_TOP_K):
    query_vec = embed_model.encode(query).tolist()
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
    return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(rows)]
 
 
def get_sparse_results(query, conn, top_k=RECALL_TOP_K):
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
    return [(doc_id, rank + 1) for rank, (doc_id, _) in enumerate(rows)]
 
 
def rrf_fusion(dense_results, sparse_results, k=RRF_K):
    scores = {}
    for doc_id, rank in dense_results:
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for doc_id, rank in sparse_results:
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
 
 
def get_titles(doc_ids, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, doc_type FROM documents WHERE id = ANY(%s)", (doc_ids,))
        rows = cur.fetchall()
    return {doc_id: (title, doc_type) for doc_id, title, doc_type in rows}
 
 
def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        for query in TEST_QUERIES:
            dense_res = get_dense_results(query, conn)
            sparse_res = get_sparse_results(query, conn)
            fused = rrf_fusion(dense_res, sparse_res)[:SHOW_TOP_N]
 
            doc_ids = [doc_id for doc_id, _ in fused]
            title_map = get_titles(doc_ids, conn)
 
            print(f"\n--- Query: {query}")
            for rank, (doc_id, score) in enumerate(fused, start=1):
                title, doc_type = title_map.get(doc_id, ("(unknown)", "?"))
                print(f"  Top{rank} [id={doc_id}] ({doc_type}) {title}    <- 相關嗎? [ ]是 [ ]否")
    finally:
        conn.close()
 
 
if __name__ == "__main__":
    main()