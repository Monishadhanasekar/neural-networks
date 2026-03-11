import os
import json
import re
import time
import requests
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# setup openrouter

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

FREE_MODEL = "openrouter/free"
TOOL_MODEL = "openrouter/free"

#First Api call to free model

response = client.chat.completions.create(
    model=FREE_MODEL,
    messages=[{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
    temperature=0.0,
    max_tokens=50,
)

print("Response:", response.choices[0].message.content)
print(f"Tokens — in: {response.usage.prompt_tokens}, out: {response.usage.completion_tokens}")
print(f"Model used: {response.model}")

#Temperature - a quick demo

prompt = "Write a one-sentence story about a robot."

for temp in [0.0, 1.0, 2.0]:
    results = []

    for _ in range(3):
        r = client.chat.completions.create(
            model=FREE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=60,
        )

        content = r.choices[0].message.content

        if content:
            results.append(content.strip())
        else:
            results.append("[No content returned]")

    print(f"\nTemperature = {temp}:")
    for i, text in enumerate(results):
        print(f"  [{i+1}] {text[:120]}")