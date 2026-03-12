import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

FREE_MODEL = "openrouter/free"
TOOL_MODEL = "openrouter/free"


# ----------- TOOLS -----------

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
        return json.dumps({"error": f"Page not found for '{query}'"})
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


# ----------- TOOL SCHEMA -----------

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
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
                    "expression": {"type": "string"}
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


# ----------- TOOL FUNCTION MAP -----------

NATIVE_FNS = {
    "search_wikipedia": search_wikipedia,
    "calculate": calculate_math,
    "get_current_date": get_current_date,
}


# ----------- SUMMARIZATION FUNCTION -----------

def summarize_conversation(messages):
    """
    Compress conversation history into a short summary
    """
    response = client.chat.completions.create(
        model=FREE_MODEL,
        messages=[
            {"role": "system", "content": "Summarize the following conversation briefly while preserving key facts and tool results."},
            {"role": "user", "content": json.dumps(messages)}
        ],
        temperature=0,
        max_tokens=200
    )

    return response.choices[0].message.content


# ----------- AGENT -----------

def run_native_agent(user_query, max_iterations=10, verbose=True):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful research assistant. Use tools when needed and produce a final answer."
        },
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("🧑 USER:", user_query)
        print("=" * 60)

    for i in range(max_iterations):

        if verbose:
            print(f"\n--- Iteration {i+1} ---")

        # -------- CONTEXT SUMMARIZATION --------

        if i > 0 and i % 4 == 0:
            if verbose:
                print("\n🧠 Summarizing conversation to reduce tokens...")

            summary = summarize_conversation(messages)

            # Keep system + user + summary + last 4 messages
            messages = [
                messages[0],
                messages[1],
                {"role": "system", "content": f"Conversation summary: {summary}"}
            ] + messages[-4:]

            if verbose:
                print("✅ Context compressed.")

        # -------- MODEL CALL --------

        response = client.chat.completions.create(
            model=TOOL_MODEL,
            messages=messages,
            tools=NATIVE_TOOLS,
            temperature=0,
            max_tokens=800,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        if verbose:
            print(
                f"📊 Tokens — in: {response.usage.prompt_tokens}, "
                f"out: {response.usage.completion_tokens}"
            )

        # -------- FINAL ANSWER --------

        if finish == "stop" and msg.content:
            if verbose:
                print("\n✅ FINAL ANSWER:\n", msg.content[:500])
            return {"answer": msg.content, "iterations": i + 1}

        # -------- TOOL CALLS --------

        if msg.tool_calls:

            messages.append(msg)

            for tc in msg.tool_calls:

                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                if verbose:
                    print(f"🔧 TOOL: {fn_name}({json.dumps(fn_args)})")

                if fn_name in NATIVE_FNS:
                    result = NATIVE_FNS[fn_name](**fn_args)
                else:
                    result = json.dumps({"error": "Unknown tool"})

                if verbose:
                    print("👁️ RESULT:", result[:150])

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


# ----------- RUN AGENT -----------

result = run_native_agent(
    "Calculate (345 * 89) + 200. Tell me today's date. Add 10 days to it and tell me the resulting date. Also calculate (45 * 8) + 120. Calculate 25% of 1500. Summarize everything."
)

print("\nResult:", result)
