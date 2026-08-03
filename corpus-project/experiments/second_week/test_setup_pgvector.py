"""
建表 + 讀取corpus.parquet + 產生embedding + 寫入pgvector + 建立tsvector欄位
 
用法：
    python setup_pgvector.py
 
需要的套件：
    uv add psycopg2-binary sentence-transformers pandas pyarrow
"""
 
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
 
# ============ 1. 設定區（跟 fusion_compare.py 保持一致） ============
 
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "your_db",
    "user": "postgres",
    "password": "yourpassword",
}
 
CORPUS_PATH = "corpus.parquet"  # 換成你實際的檔案路徑
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768  # bge-base-en-v1.5 的輸出維度
BATCH_SIZE = 32  # 批次算embedding，避免記憶體一次爆掉
 
 
# ============ 2. 建表 ============
 
def create_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                doc_type TEXT,
                embedding VECTOR({EMBEDDING_DIM}),
                content_tsv TSVECTOR
            );
        """)
        # 向量索引（近似最近鄰，加速dense檢索；資料量小的話沒有也能跑，但養成習慣先建）
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        # 全文檢索索引（加速tsvector檢索）
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_tsv_idx
            ON documents USING GIN (content_tsv);
        """)
    conn.commit()
    print("Schema與索引建立完成")
 
 
# ============ 3. 讀取corpus.parquet ============
 
def load_corpus(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"讀取corpus: {len(df)}筆")
    print(f"欄位: {list(df.columns)}")
    return df
 
 
# ============ 4. 產生embedding + 寫入DB ============
 
def ingest(conn, df: pd.DataFrame, text_column: str, id_column: str,
           title_column: str = None, type_column: str = None):
    """
    text_column: 用來產生embedding、寫入content_tsv的主要文字欄位
    id_column: 唯一識別欄位
    title_column/type_column: 選填，沒有的話會存NULL
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
    rows_to_insert = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        texts = batch[text_column].fillna("").tolist()
        embeddings = model.encode(texts, show_progress_bar=False)
 
        for i, (_, row) in enumerate(batch.iterrows()):
            rows_to_insert.append((
                str(row[id_column]),
                str(row[title_column]) if title_column else None,
                str(row[text_column]),
                str(row[type_column]) if type_column else None,
                embeddings[i].tolist(),
            ))
 
        print(f"已處理 {min(start + BATCH_SIZE, len(df))}/{len(df)}")
 
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO documents (id, title, content, doc_type, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                doc_type = EXCLUDED.doc_type,
                embedding = EXCLUDED.embedding
            """,
            rows_to_insert,
        )
        # 寫完embedding後，另外更新content_tsv欄位（英文語料用english設定）
        cur.execute("""
            UPDATE documents
            SET content_tsv = to_tsvector('english', content)
            WHERE content_tsv IS NULL;
        """)
    conn.commit()
    print(f"寫入完成，共 {len(rows_to_insert)} 筆")
 
 
# ============ 5. 主流程 ============
 
def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        create_schema(conn)
 
        df = load_corpus(CORPUS_PATH)
 
        # 印出欄位讓你確認要對應哪些欄位名稱，避免直接猜錯欄位名稱
        print("\n請確認你的corpus.parquet實際欄位名稱，並修改下方 ingest() 呼叫參數")
        print(df.head(2))
 
        # TODO: 把下面的欄位名稱換成你 corpus.parquet 實際的欄位名稱
        ingest(
            conn,
            df,
            text_column="content",      # 換成實際的文字內容欄位名稱
            id_column="id",             # 換成實際的唯一ID欄位名稱
            title_column="title",       # 沒有的話設 None
            type_column="type",     # 沒有的話設 None
        )
    finally:
        conn.close()
 
 
if __name__ == "__main__":
    main()