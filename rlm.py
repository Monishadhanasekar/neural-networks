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
ROOT_MODEL = "openrouter/free"  # root agent (stronger)
SUB_MODEL  = "openrouter/free"  # sub-agent (can be cheaper)

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

#Try it with a vanilla LLM — just stuff the context in

# ──────────────────────────────────────────────
# VANILLA APPROACH: stuff everything into the prompt
# ──────────────────────────────────────────────

vanilla_prompt = f"""Here is a dataset of people. Count EXACTLY how many are engineers in Tokyo.
Return ONLY the number, nothing else.

{dataset_text}"""

print(f"Prompt length: {len(vanilla_prompt)} characters")
print("Asking vanilla LLM...")

vanilla_answer = llm_call(vanilla_prompt)
print(f"\nVanilla LLM answer: {vanilla_answer}")
print(f"Ground truth:        {ground_truth}")
print(f"Correct?             {'✅ Yes' if str(ground_truth) in vanilla_answer else '❌ No'}")

#Even if the model gets this one right with 500 entries, the approach doesn't scale. At 5,000 or 50,000 entries, context rot destroys accuracy.

#The RLM approach: instead of feeding all 500 entries to the model, let the model write code to count them itself.

#Part 2: Build the REPL Environment
#The REPL is where the model's code runs. We need:

#The context stored as a variable (not in the prompt)
#print() output captured and returned to the model
#A FINAL() function to signal the answer
#A llm_query() function for recursive sub-LLM calls

# ──────────────────────────────────────────────
# THE REPL: a Python execution environment
# ──────────────────────────────────────────────

class RLMRepl:
    """A REPL environment for RLM.

    The context is stored as a Python variable.
    The model writes code that runs here.
    print() output is captured and returned to the model.
    """

    def __init__(self, context: str, max_output_chars: int = 5000):
        self.final_answer = None
        self.max_output_chars = max_output_chars
        self.sub_call_count = 0

        # The namespace where the model's code runs.
        # 'context' is the key variable — the long input stored here,
        # NOT in the LLM's prompt.
        self.namespace = {
            "context": context,            # <-- the long input lives here
            "FINAL": self._final,           # call FINAL("answer") to finish
            "llm_query": self._llm_query,   # call a sub-LLM (recursion!)
            # Standard library modules the model might want
            "re": re,
            "json": json,
            "len": len,
            "print": print,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "sum": sum,
            "sorted": sorted,
            "min": min,
            "max": max,
            "abs": abs,
            "set": set,
            "tuple": tuple,
            "zip": zip,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
            "True": True,
            "False": False,
            "None": None,
        }

    def _final(self, answer):
        """Called by the model to submit its final answer."""
        self.final_answer = str(answer)
        print(f"[FINAL ANSWER SUBMITTED: {answer}]")

    def _llm_query(self, query: str, sub_context: str = "") -> str:
        """Recursive sub-LLM call.

        The model can call this from within its code to get
        a sub-LLM to process a chunk of context.
        The result comes back as a string variable — NOT
        loaded into the parent's context window.
        """
        self.sub_call_count += 1
        print(f"  [Sub-LLM call #{self.sub_call_count}: '{query[:80]}...'")

        prompt = query
        if sub_context:
            prompt = f"{query}\n\nContext:\n{sub_context}"

        result = llm_call(prompt, model=SUB_MODEL, max_tokens=2000)
        print(f"   Sub-LLM returned: '{result[:100]}...']")
        return result

    def execute(self, code: str) -> str:
        """Execute Python code in the REPL. Returns captured stdout."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self.namespace)
            output = stdout_capture.getvalue()
        except Exception as e:
            output = f"ERROR: {type(e).__name__}: {e}"

        # Truncate if too long — the model shouldn't be overwhelmed
        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars] + f"\n... [truncated to {self.max_output_chars} chars]"

        return output

# Quick test of the REPL
repl = RLMRepl(context="Hello world! This is a test context with some data.")
print("Test 1 — peek at context:")
print(repl.execute('print(context[:30])'))

print("\nTest 2 — search with regex:")
print(repl.execute('import re\nmatches = re.findall(r"\\w+", context)\nprint(f"Words: {len(matches)}")'))

print("\nTest 3 — submit final answer:")
print(repl.execute('FINAL(42)'))
print(f"Final answer stored: {repl.final_answer}")

print("\n✅ REPL works!")

#Part 3: Build the RLM Agent Loop
#The core loop:

#Give the LLM the query + system prompt (but NOT the context)
#LLM writes Python code
#Execute the code in the REPL (where context lives)
#Send the output back to the LLM
#Repeat until FINAL() is called or we hit max iterations

# ──────────────────────────────────────────────
# THE RLM SYSTEM PROMPT
# This tells the model HOW to use the REPL
# ──────────────────────────────────────────────

RLM_SYSTEM_PROMPT = """You are an RLM (Recursive Language Model) agent.

You have access to a Python REPL environment. The user's data is stored
in a variable called `context` — it may be very long (millions of characters).
You CANNOT see the context directly. You must write Python code to explore it.

Available tools:
- `context` — the full input text (Python string variable)
- `print()` — use this to see output from your code
- `llm_query(query, sub_context)` — call a sub-LLM to analyze a chunk.
  The sub-LLM's response is returned as a string. It does NOT enter your context.
- `FINAL(answer)` — call this when you have the final answer.
- Standard Python: `re`, `json`, `len`, `sum`, etc.

Strategy:
1. First, check the size: `print(len(context))`
2. Peek at the structure: `print(context[:500])`
3. Use code to search, filter, count, or slice the data
4. For complex subtasks, use `llm_query()` to delegate to a sub-LLM
5. When done, call `FINAL(your_answer)`

Rules:
- Write ONLY Python code. No markdown, no explanation.
- Your code block must be wrapped in ```python ... ```
- Use print() to see results — you only see what you print.
- Variables persist between steps (like Jupyter cells).
- Be systematic. Explore first, then solve.
"""

print("System prompt defined ✅")