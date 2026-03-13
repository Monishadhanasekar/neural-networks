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

def search_wikipedia(query):
    """Search Wikipedia and return a summary."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return json.dumps({
                "title": data.get("title", ""),
                "summary": data.get("extract", "No summary found.")[:800]
            })
        return json.dumps({"error": f"Page not found for '{query}'. Try a different term."})
    except Exception as e:
        return json.dumps({"error": str(e)})

def calculate_math(expression):
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() eE")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": f"Invalid characters in: {expression}"})
        result = eval(expression)
        return json.dumps({"expression": expression, "result": round(result, 6)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_current_date():
    """Get the current date and time."""
    from datetime import datetime
    return json.dumps({"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    return f"File written to {path}"


def run_python(code):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)



NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression like '(5+3)*2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

NATIVE_FNS = {
    "search_wikipedia": search_wikipedia,
    "calculate": calculate_math,
    "get_current_date": get_current_date,
}

CODER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
]

CODER_FNS = {
    "write_file": write_file,
    "run_python": run_python,
}


def run_researcher_agent(task, verbose=True):

    messages = [
        {"role": "system",
         "content": "You are a researcher. Use Wikipedia tools to find information."},
        {"role": "user", "content": task},
    ]

    response = client.chat.completions.create(
        model=TOOL_MODEL,
        messages=messages,
        tools=[NATIVE_TOOLS[0]],  # search_wikipedia only
        temperature=0,
        max_tokens=500,
    )

    msg = response.choices[0].message

    if msg.tool_calls:

        tc = msg.tool_calls[0]
        fn_args = json.loads(tc.function.arguments)

        result = search_wikipedia(**fn_args)

        if verbose:
            print("🔎 Researcher result:", result[:200])

        return result

    return msg.content

def run_coder_agent(task, verbose=True):

    messages = [
        {"role": "system",
         "content": "You are a coding assistant. Use file and python tools."},
        {"role": "user", "content": task},
    ]

    response = client.chat.completions.create(
        model=TOOL_MODEL,
        messages=messages,
        tools=CODER_TOOLS,
        temperature=0,
        max_tokens=500,
    )

    msg = response.choices[0].message

    if msg.tool_calls:

        tc = msg.tool_calls[0]

        fn_name = tc.function.name
        fn_args = json.loads(tc.function.arguments)

        result = CODER_FNS[fn_name](**fn_args)

        if verbose:
            print("💻 Coder result:", result)

        return result

    return msg.content

def orchestrator(user_query, verbose=True):

    routing_prompt = [
        {
            "role": "system",
            "content":
            "You are a router deciding which agent should handle a task.\n\n"
            "Agents:\n"
            "researcher → information lookup\n"
            "coder → programming or files\n\n"
            "Return only one word: researcher or coder."
        },
        {"role": "user", "content": user_query}
    ]

    response = client.chat.completions.create(
        model=FREE_MODEL,
        messages=routing_prompt,
        temperature=0,
        max_tokens=10
    )

    agent_text = response.choices[0].message.content.lower()

    if "research" in agent_text:
        agent = "researcher"
    elif "code" in agent_text or "program" in agent_text:
        agent = "coder"
    else:
        agent = "researcher"

    if verbose:
        print("🧭 Orchestrator selected:", agent)

    if agent == "researcher":
        return run_researcher_agent(user_query)

    if agent == "coder":
        return run_coder_agent(user_query)


result = orchestrator(
    "Search Wikipedia for ai"
)

print("\nFINAL RESULT:")
print(result)

result = orchestrator(
    "Write a python file that prints Hello World"
)

print("\nFINAL RESULT:")
print(result)
