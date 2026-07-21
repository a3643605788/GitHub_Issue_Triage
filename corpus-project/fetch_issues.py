import requests, os, json, time
from dotenv import load_dotenv
load_dotenv()

HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
REPO = "pallets/click"  # 換成你選的repo  ({owner}/{repo})

def fetch_closed_issues(page=1):
    url = f"https://api.github.com/repos/{REPO}/issues"
    params = {"state": "closed", "per_page": 100, "page": page}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

all_issues = []
page = 1
while True:
    batch = fetch_closed_issues(page)
    if not batch:
        break
    all_issues.extend([i for i in batch if "pull_request" not in i])  # 濾掉PR
    page += 1
    time.sleep(0.5)

with open("raw_issues.json", "w") as f:
    json.dump(all_issues, f)

print(f"抓到 {len(all_issues)} 筆issue")