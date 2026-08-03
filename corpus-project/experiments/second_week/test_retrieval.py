import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json

# ===== 設定 =====
MODELS = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE-base": "BAAI/bge-base-en-v1.5",
    "e5-base": "intfloat/multilingual-e5-base",
}
TOP_N = 5
N_QUERIES = 10
RANDOM_SEED = 42

# ===== 讀取資料 =====
df = pd.read_parquet("corpus.parquet")
df = df.reset_index(drop=True)
df["text"] = df["title"].fillna("") + "\n" + df["content"].fillna("")
print(f"總筆數: {len(df)}")

# e5系列模型要求query/passage分別加instruction prefix
def prepare_texts(texts, model_name, is_query=False):
    if "e5" in model_name.lower():
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]
    return texts

# ===== 固定挑10題query（用同一組query比較3個模型才公平）=====
np.random.seed(RANDOM_SEED)
query_indices = np.random.choice(len(df), size=N_QUERIES, replace=False).tolist()
print(f"Query索引: {query_indices}")

# ===== 對每個模型跑embedding + 檢索 =====
all_results = {}

for name, model_id in MODELS.items():
    print(f"\n{'='*50}\n處理模型: {name}\n{'='*50}")

    start = time.time()
    model = SentenceTransformer(model_id)
    print(f"載入耗時: {time.time()-start:.1f}s")

    # 全語料庫embedding（passage模式）
    corpus_texts = prepare_texts(df["text"].tolist(), name, is_query=False)
    start = time.time()
    corpus_embeddings = model.encode(
        corpus_texts, show_progress_bar=True, batch_size=32
    )
    print(f"全量編碼({len(df)}筆)耗時: {time.time()-start:.1f}s")

    # normalize，之後用內積當cosine similarity
    corpus_norm = corpus_embeddings / np.linalg.norm(
        corpus_embeddings, axis=1, keepdims=True
    )

    # 針對10個query分別檢索
    model_results = []
    for q_idx in query_indices:
        query_text_raw = df.iloc[q_idx]["text"]
        query_text = prepare_texts([query_text_raw], name, is_query=True)[0]
        query_emb = model.encode([query_text])[0]
        query_emb_norm = query_emb / np.linalg.norm(query_emb)

        sims = corpus_norm @ query_emb_norm
        # 排除自己，取topN
        top_idx = np.argsort(sims)[::-1]
        top_idx = [i for i in top_idx if i != q_idx][:TOP_N]

        model_results.append({
            "query_idx": int(q_idx),
            "query_title": df.iloc[q_idx]["title"],
            "query_type": df.iloc[q_idx]["type"],
            "top_results": [
                {
                    "idx": int(i),
                    "title": df.iloc[i]["title"],
                    "type": df.iloc[i]["type"],
                    "score": round(float(sims[i]), 4),
                }
                for i in top_idx
            ],
        })

    all_results[name] = model_results

# ===== 輸出成人工檢查用的檔案 =====
with open("retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# ===== 同時輸出成易讀的文字檔，方便你人工打分 =====
with open("retrieval_review.txt", "w", encoding="utf-8") as f:
    for model_name, results in all_results.items():
        f.write(f"\n{'#'*60}\n# 模型: {model_name}\n{'#'*60}\n")
        for r in results:
            f.write(f"\n--- Query [{r['query_idx']}] ({r['query_type']}): {r['query_title']}\n")
            for rank, item in enumerate(r["top_results"], 1):
                f.write(
                    f"  Top{rank} [score={item['score']}] "
                    f"({item['type']}) {item['title']}  "
                    f"  <- 相關嗎? [ ]是 [ ]否\n"
                )

print("\n完成！請打開 retrieval_review.txt 人工標註每筆是否相關")