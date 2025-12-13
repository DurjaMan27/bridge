from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

print("KEY PRESENT:", key is not None)

client = OpenAI(api_key=key)

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "ping"}]
)

print(resp.choices[0].message.content)