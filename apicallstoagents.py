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