import os
import json
import re
import time
import requests
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

#Part 1: Setup and first API calls to free model

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

def fetch_webpage(url):
    """Fetch webpage content from a URL."""

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            return json.dumps({"error": f"Failed to fetch page: {r.status_code}"})

        # Limit page size to avoid huge token usage
        content = r.text[:5000]

        return json.dumps({
            "url": url,
            "content": content
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


NATIVE_TOOLS = [
    {
    "type": "function",
    "function": {
        "name": "fetch_webpage",
        "description": "Fetch the content of a webpage from a given URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full webpage URL to fetch"
                }
            },
            "required": ["url"]
        }
    }
}
]

NATIVE_FNS = {
    "fetch_webpage": fetch_webpage,
}


def run_native_agent(user_query, max_iterations=10, verbose=True):
    """
    Agent using native function calling with web browsing capability.
    """
    messages = [
        {
            "role": "system",
            "content":
            "You are a web research assistant.\n"
            "You can fetch webpages using the fetch_webpage tool.\n"
            "If the user provides a URL or asks to read a webpage, "
            "use the fetch_webpage tool to retrieve the content.\n"
            "After reading the webpage content, summarize or answer the user's question."
        },
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"🧑 USER: {user_query}")
        print(f"{'='*60}")

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {i+1} ---")

        response = client.chat.completions.create(
            model=TOOL_MODEL, messages=messages,
            tools=NATIVE_TOOLS, temperature=0, max_tokens=800,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        # Text response = done
        if finish == "stop" and msg.content:
            if verbose:
                print(f"✅ FINAL ANSWER:\n  {msg.content[:500]}")
            return {"answer": msg.content, "iterations": i + 1}

        # Tool calls
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                if verbose:
                    print(f"🔧 TOOL: {fn_name}({json.dumps(fn_args)[:80]})")

                if fn_name in NATIVE_FNS:
                    result = NATIVE_FNS[fn_name](**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                if verbose:
                    print(f"👁️  RESULT: {result[:150]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                break

    return {"answer": "Max iterations reached", "iterations": max_iterations}

result = run_native_agent(
    "Fetch and summarize this webpage https://en.wikipedia.org/wiki/Artificial_intelligence"
)

