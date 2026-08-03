"""
用人工標註的golden label，計算每個reranker模型的Precision@5
 
用法：
    python compute_precision.py
"""
 
import time
 
import psycopg2
from sentence_transformers import CrossEncoder, SentenceTransformer
 
# ============ 1. 設定區 ============
 
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
 
RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    # large / v2-m3 延遲過高(29s~93s/query)，先排除在精準度比較外，
    # 除非你的使用情境能接受這種延遲，否則不需要浪費時間測
    # "BAAI/bge-reranker-large",
    # "BAAI/bge-reranker-v2-m3",
]
 
# 從你標註的 fusion_review.txt 整理出的 golden label：
# key是query文字，value是人工標「是」的doc_id清單
GOLDEN_LABELS = {
    "When I use del as the function name, I will report a syntax error. The other names are correct.": ["1276"],
    "Feature request: undeprecate __version__": ["310", "2900"],
    "Order of commands in help is unreliable in Python <3.6": ["1505"],
    "auto_envvar_prefix does not work with command groups": ["2424", "1253"],
    "Callback behavior has changed in 8.0.0 and raises exception now": ["1888", "2745"],
    "Different portion of text with different color in echo_via_pager": ["130", "183", "901", "3416", "3417", "2542"],
    "Release 8.3.3": ["..\\click\\docs\\upgrade-guides.md", "2891", "3185", "2896", "2789", "2793", "1386"],
    "Prompting options if certain flag is given": ["956", "2575", "1992", "420", "1380", "491"],
    "Multi-value flags broken by recent change": ["2292", "1891", "3050", "2246", "2001", "2952", "1960"],
    "Completion behaviour for options/arguments with empty incomplete": ["1907", "780", "2040", "1929", "2995", "534", "2283"],
}
 
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
 
# ============ 2. Dense + Sparse + RRF（跟之前腳本一致） ============
 
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
 
 
def get_candidate_texts(doc_ids, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, content FROM documents WHERE id = ANY(%s)", (doc_ids,))
        rows = cur.fetchall()
    text_map = {doc_id: f"{title or ''} {content or ''}".strip() for doc_id, title, content in rows}
    return [(doc_id, text_map.get(doc_id, "")) for doc_id in doc_ids]
 
 
# ============ 3. Precision@5 計算 ============
 
def precision_at_5(ranked_doc_ids, relevant_doc_ids):
    top5 = ranked_doc_ids[:5]
    hits = sum(1 for doc_id in top5 if doc_id in relevant_doc_ids)
    return hits / 5
 
 
def main():
    conn = psycopg2.connect(**DB_CONFIG)
 
    # 先撈出每題的fusion候選(含文字)，所有模型共用同一份候選池
    query_candidates = {}
    for query in GOLDEN_LABELS:
        dense_res = get_dense_results(query, conn)
        sparse_res = get_sparse_results(query, conn)
        fused = rrf_fusion(dense_res, sparse_res)
        doc_ids = [doc_id for doc_id, _ in fused]
        candidates = get_candidate_texts(doc_ids, conn)
        query_candidates[query] = candidates
    conn.close()
 
    summary = []
    for model_name in RERANKER_MODELS:
        print(f"\n===== {model_name} =====")
        model = CrossEncoder(model_name)
 
        precisions = []
        latencies = []
        for query, candidates in query_candidates.items():
            pairs = [(query, text) for _, text in candidates]
 
            start = time.perf_counter()
            scores = model.predict(pairs)
            latency = time.perf_counter() - start
            latencies.append(latency)
 
            ranked = sorted(
                zip([doc_id for doc_id, _ in candidates], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            ranked_ids = [doc_id for doc_id, _ in ranked]
 
            p5 = precision_at_5(ranked_ids, GOLDEN_LABELS[query])
            precisions.append(p5)
            print(f"  Precision@5={p5:.2f} | 耗時={latency*1000:.0f}ms | {query[:50]}")
 
        avg_p5 = sum(precisions) / len(precisions)
        avg_latency = sum(latencies) / len(latencies)
        summary.append({"model": model_name, "avg_p5": avg_p5, "avg_latency_ms": avg_latency * 1000})
 
    print("\n\n===== 總結 =====")
    print(f"{'模型':<40} {'平均Precision@5':>18} {'平均延遲(ms)':>15}")
    for r in sorted(summary, key=lambda x: -x["avg_p5"]):
        print(f"{r['model']:<40} {r['avg_p5']:>18.2f} {r['avg_latency_ms']:>15.0f}")
 
 
if __name__ == "__main__":
    main()