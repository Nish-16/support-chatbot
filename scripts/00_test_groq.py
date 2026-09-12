"""
Sanity check: confirm the Groq API key in .env works before we
build anything on top of it.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
)
print("model replied:", response.choices[0].message.content)
