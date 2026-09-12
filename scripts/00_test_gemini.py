"""
Sanity check: confirm the Gemini API key in .env works before we
build anything on top of it.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly one word: pong",
)
print("model replied:", response.text)
