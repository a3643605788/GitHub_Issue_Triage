import json
import pandas as pd
from pathlib import Path

# 讀issues
with open("raw_issues.json", "r", encoding="utf-8") as f:
    issues = json.load(f)

issue_records = [
    {"id": str(i["number"]), "type": "issue", "title": i["title"],
     "content": i["body"] or "", "labels": [l["name"] for l in i["labels"]],
     "source": "github_issue"}
    for i in issues
]

# 讀docs
docs_dir = Path("../click/docs")
doc_records = [
    {"id": str(f), "type": "doc", "title": f.stem,
     "content": f.read_text(encoding="utf-8"), "labels": [],
     "source": "official_docs"}
    for f in docs_dir.glob("**/*.md")
]

df = pd.DataFrame(issue_records + doc_records)
df.to_parquet("corpus.parquet")
print(f"共{len(df)}筆，issue {len(issue_records)}筆，doc {len(doc_records)}筆")