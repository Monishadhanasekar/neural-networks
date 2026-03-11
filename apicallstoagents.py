import os
import json
import re
import time
import requests
import subprocess
from openai import OpenAI


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
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