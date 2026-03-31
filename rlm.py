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

#Part 1: The Problem — Long Context Fails
#Let's create a task that's easy for humans but hard for LLMs at scale: Count specific items buried in a long, noisy dataset.

import random
random.seed(42)

# ──────────────────────────────────────────────
# Generate a synthetic dataset: people with cities and professions
# ──────────────────────────────────────────────

first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
               "Grace", "Hank", "Ivy", "Jack", "Karen", "Leo",
               "Mona", "Nate", "Olivia", "Paul", "Quinn", "Rita",
               "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander", "Yara", "Zane"]

cities = ["New York", "London", "Tokyo", "Paris", "Berlin",
          "Mumbai", "Sydney", "Toronto", "Dubai", "Singapore",
          "Seoul", "Bangkok", "Cairo", "Lagos", "Lima"]

professions = ["engineer", "doctor", "teacher", "artist", "chef",
               "pilot", "lawyer", "nurse", "writer", "musician"]

def generate_dataset(n_entries=500):
    """Generate a dataset of people with names, cities, and professions."""
    entries = []
    for i in range(n_entries):
        name = f"{random.choice(first_names)} {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}."
        city = random.choice(cities)
        prof = random.choice(professions)
        age = random.randint(22, 65)
        entries.append(f"Entry {i+1}: {name}, age {age}, {prof}, based in {city}")
    return entries

entries = generate_dataset(5000)
dataset_text = "\n".join(entries)

# Ground truth: count engineers in Tokyo
ground_truth = sum(1 for e in entries if "engineer" in e and "Tokyo" in e)

print(f"Dataset: {len(entries)} entries, {len(dataset_text)} characters")
print(f"\nFirst 5 entries:")
for e in entries[:5]:
    print(f"  {e}")
print(f"\n🎯 Ground truth: {ground_truth} engineers in Tokyo")