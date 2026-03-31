#Recursive Language Model

import requests
import json
import re
import io
import sys
import traceback
import os
from contextlib import redirect_stdout, redirect_stderr

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Models — use cheap/free ones for the class
ROOT_MODEL = "google/gemini-3-flash-preview"  # root agent (stronger)
SUB_MODEL  = "google/gemini-3-flash-preview"  # sub-agent (can be cheaper)

def llm_call(prompt, system="", model=ROOT_MODEL, max_tokens=4000):
    """Call an LLM via OpenRouter. Returns the text response."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": model, "messages": msgs, "max_tokens": max_tokens}
    )
    data = r.json()
    if "choices" not in data:
        raise Exception(f"API error: {data}")
    return data["choices"][0]["message"]["content"]

# Quick test
print(llm_call("Say hello in one word."))