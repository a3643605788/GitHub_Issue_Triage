from google import genai
from dotenv import load_dotenv

load_dotenv()  # 會自動讀 .env 裡的 GEMINI_API_KEY

client = genai.Client()  # 自動抓 GEMINI_API_KEY 環境變數
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="這是一個測試問題：什麼是Click套件？"
)
print(response.text)