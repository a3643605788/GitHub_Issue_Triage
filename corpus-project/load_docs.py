from pathlib import Path

docs_dir = Path("../click/docs")
md_files = list(docs_dir.glob("**/*.md"))
print(f"找到 {len(md_files)} 個文件檔案")

# 先讀一個檔案確認內容正常
if md_files:
    print(md_files[0].read_text(encoding="utf-8")[:200])