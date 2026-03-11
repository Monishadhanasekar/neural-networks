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

# Now let's have a quick look at how the conversation history affects token usage.

conversation = [
    {"role": "system", "content": "You are a math tutor. Be concise."},
    {"role": "user", "content": "What is a derivative?"},
]

r1 = client.chat.completions.create(
    model=FREE_MODEL,
    messages=conversation,
    max_tokens=150
)

turn1 = r1.choices[0].message.content or "[No content returned]"
print("Turn 1:", turn1[:200])
print(f"  Prompt tokens: {r1.usage.prompt_tokens}")

conversation.append({"role": "assistant", "content": turn1})
conversation.append({"role": "user", "content": "Give me an example with x²."})

r2 = client.chat.completions.create(
    model=FREE_MODEL,
    messages=conversation,
    max_tokens=150
)

turn2 = r2.choices[0].message.content or "[No content returned]"
print(f"\nTurn 2: {turn2[:200]}")
print(f"  Prompt tokens: {r2.usage.prompt_tokens}")

print(f"\n→ Prompt tokens grew from {r1.usage.prompt_tokens} to {r2.usage.prompt_tokens}")

#Part2 : Function calling with tool model

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '(5 + 3) * 2'"}
                },
                "required": ["expression"]
            }
        }
    }
]

r = client.chat.completions.create(
    model=TOOL_MODEL,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    temperature=0,
)

msg = r.choices[0].message
print(f"Finish reason: {r.choices[0].finish_reason}")
print(f"Content: {msg.content}")
print(f"Tool calls: {msg.tool_calls}")

if msg.tool_calls:
    tc = msg.tool_calls[0]
    print(f"\n→ Model wants to call: {tc.function.name}({tc.function.arguments})")
    print("  It GENERATED JSON requesting a function. Our code must execute it.")

#Complete the tool call loop

#User → Model requests tool → OUR code runs tool → Result back to model → Final answer

# Fake tool implementations
def get_weather(city):
    fake = {
        "Tokyo": {"temp": "22°C", "condition": "partly cloudy"},
        "London": {"temp": "14°C", "condition": "rainy"},
        "Delhi": {"temp": "38°C", "condition": "sunny"},
    }
    return json.dumps(fake.get(city, {"temp": "unknown", "condition": "unknown"}))

def calculate(expression):
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters"})
        return json.dumps({"result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})

TOOL_FNS = {"get_weather": get_weather, "calculate": calculate} #Tool Registry - maps tool names to our Python functions that implement them

# Execute the tool call
if msg.tool_calls:
    tc = msg.tool_calls[0]
    fn = TOOL_FNS[tc.function.name]
    result = fn(**json.loads(tc.function.arguments))
    print(f"Tool result: {result}")

    # Feed result back to model
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        msg,
        {"role": "tool", "tool_call_id": tc.id, "content": result}
    ]

    r2 = client.chat.completions.create(
        model=TOOL_MODEL, messages=messages, tools=tools
    )
    print(f"\nFinal answer: {r2.choices[0].message.content}")
    print("\n→ Full cycle: user → model requests tool → we run it → model answers")

#Part 3: Build a ReAct Agent From Scratch

#The ReAct pattern: THINK → ACT → OBSERVE → repeat until done

REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step using tools.

You have access to these tools:
{tool_descriptions}

## How to respond

When you need to use a tool, respond in EXACTLY this format:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
ACTION_INPUT: <arguments as valid JSON>

When you have enough information for the final answer:

THOUGHT: <your final reasoning>
FINAL_ANSWER: <your complete answer to the user>

## Rules
- Always start with THOUGHT
- Use only ONE action per turn
- Wait for the OBSERVATION before continuing
- If a tool returns an error, reason about it and try a different approach
- Be concise in your thoughts
"""

# Real tools
# Wikipedia search hits a real API. The calculator does real math. These are the "hands" of our agent.

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

# Tool registry
TOOLS = {
    "search_wikipedia": {
        "fn": search_wikipedia,
        "desc": "search_wikipedia(query: str) — Search Wikipedia. Use simple topic names like 'France' or 'Albert Einstein'."
    },
    "calculate": {
        "fn": calculate_math,
        "desc": "calculate(expression: str) — Evaluate a math expression. Example: '(5 + 3) * 2.5'"
    },
    "get_current_date": {
        "fn": get_current_date,
        "desc": "get_current_date() — Get today's date and time. Pass empty JSON: {}"
    },
}

print(f"Tools registered: {list(TOOLS.keys())}")

# THE AGENT LOOP

def run_agent(user_query, max_iterations=10, verbose=True):
    """
    A ReAct agent from scratch.
    No frameworks. Just an LLM + tools + a loop.
    """
    # Build system prompt with tool descriptions
    tool_desc = "\n".join(f"- {t['desc']}" for t in TOOLS.values())
    system = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"🧑 USER: {user_query}")
        print(f"{'='*60}")

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")

        # STEP 1: Ask the LLM what to do
        response = client.chat.completions.create(
            model=FREE_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=800,
        )

        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # STEP 2: Parse the response
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        if verbose and thought:
            print(f"💭 THOUGHT: {thought[:200]}")

        # Check for FINAL_ANSWER
        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"✅ AGENT FINISHED in {i+1} iteration(s)")
                print(f"{'='*60}")
                print(f"📝 ANSWER: {answer[:500]}")
            return {"answer": answer, "iterations": i + 1}

        # Check for ACTION
        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:
            messages.append({
                "role": "user",
                "content": "Please respond with either ACTION + ACTION_INPUT or FINAL_ANSWER."
            })
            if verbose:
                print("⚠️  Format issue — nudging...")
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        # STEP 3: Execute the tool
        if tool_name not in TOOLS:
            observation = json.dumps({
                "error": f"Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
            })
        else:
            try:
                if raw_input.startswith("{"):
                    args = json.loads(raw_input)
                else:
                    args = {"query": raw_input.strip("\"'")}
                observation = TOOLS[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": f"Failed to call {tool_name}: {e}"})

        if verbose:
            print(f"🔧 ACTION: {tool_name}({raw_input[:80]})")
            obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"👁️  OBSERVATION: {obs_preview}")

        # STEP 4: Feed observation back
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    if verbose:
        print(f"\n⚠️  Max iterations ({max_iterations}) reached.")
    return {"answer": "Max iterations reached.", "iterations": max_iterations}

result = run_agent("What is 123*123?")
result = run_agent(
    "Search Wikipedia for 'Wakanda' and tell me about its history."
)
result = run_agent(
    "Who was born first: the person who wrote 'Romeo and Juliet' "
    "or the person who painted the Mona Lisa? By how many years?"
)