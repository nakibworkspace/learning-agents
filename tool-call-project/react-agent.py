"""
react-agent.py

A ReAct (Reason + Act) agent built on top of the same tools as my-agent.py.

Key differences from my-agent.py:
- NO native tool calling. We pass plain text to the model and parse its output.
- The model is forced to output a strict format:
      Thought: ...
      Action: <tool_name>
      Action Input: <json>
      (Observation: <filled in by us>)
      ... repeat ...
      Thought: ...
      Final Answer: ...
- Reasoning is visible text, not hidden inside the model.

Run:
    python3 react-agent.py
"""

import json
import re

import requests
from duckduckgo_search import DDGS
from ollama import chat

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOOL SCHEMAS (kept identical to my-agent.py so behavior matches)
# ─────────────────────────────────────────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g. Athens",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation on a pure arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A pure arithmetic expression, e.g. '(25.9 * 2) + 10'. Only numeric literals and + - * / operators.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web and return a list of results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOL IMPLEMENTATIONS (identical to my-agent.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_current_weather(city: str, unit: str = "celsius"):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    ).json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "temperature_unit": unit,
        },
    ).json()
    return {
        "city": city,
        "temperature": weather["current"]["temperature_2m"],
        "unit": unit,
    }


def calculate(expression: str):
    """Safe arithmetic evaluator — refuses function calls (see my-agent.py)."""
    import ast
    import operator

    _binops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _binops:
            return _binops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _unops:
            return _unops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return {
                "expression": expression,
                "error": (
                    "calculate only supports pure arithmetic. Use numeric "
                    "literals from prior tool results; do not call other tools "
                    "inside the expression."
                ),
            }
    return {"expression": expression, "result": _eval(tree.body)}


def get_time(city: str):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    ).json()
    timezone = geo["results"][0]["timezone"]
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m",
            "timezone": timezone,
        },
    ).json()
    return {
        "city": city,
        "timezone": timezone,
        "datetime": resp["current"]["time"],
    }


def search_web(query: str, max_results: int = 5):
    max_results = max(1, min(int(max_results), 10))
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            })
    return {
        "query": query,
        "result_count": len(results),
        "results": results,
    }


# Map tool name -> Python function. Re-used by the dispatcher below.
available_functions = {
    "get_current_weather": get_current_weather,
    "calculate": calculate,
    "get_time": get_time,
    "search_web": search_web,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. REACT PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# A few-shot example is critical: llama3.2 is small and needs to see exactly
# what shape its output should take.
FEW_SHOT_EXAMPLE = """
Example:

Question: What is the weather in Tokyo, then compute (temperature * 2) + 10?

Thought: I need the current weather in Tokyo first.
Action: get_current_weather
Action Input: {"city": "Tokyo", "unit": "celsius"}
Observation: {"city": "Tokyo", "temperature": 25.9, "unit": "celsius"}
Thought: Now I have 25.9. I can compute (25.9 * 2) + 10.
Action: calculate
Action Input: {"expression": "(25.9 * 2) + 10"}
Observation: {"expression": "(25.9 * 2) + 10", "result": 61.8}
Thought: I have the final answer.
Final Answer: 61.8
"""


def render_tool_descriptions():
    """Render a human-readable description of each tool for the prompt."""
    lines = []
    for t in tools:
        f = t["function"]
        lines.append(f"- {f['name']}: {f['description']}")
        params = f["parameters"]["properties"]
        required = set(f["parameters"].get("required", []))
        for pname, pinfo in params.items():
            mark = " (required)" if pname in required else ""
            lines.append(f"    - {pname}{mark}: {pinfo.get('description', '')}")
    return "\n".join(lines)


def build_react_prompt():
    tool_descriptions = render_tool_descriptions()
    tool_names = ", ".join(t["function"]["name"] for t in tools)
    return f"""You are an agent that solves problems by thinking step-by-step and using tools.

Available tools:
{tool_descriptions}

You MUST follow this exact format for every turn. Do not deviate.

Thought: <your reasoning about what to do next>
Action: <one of [{tool_names}]>
Action Input: <a JSON object matching the chosen tool's parameters>
Observation: <filled in by the system — do NOT write this yourself>

When you have enough information to answer, output:

Thought: <your final reasoning>
Final Answer: <the answer to the user's original question>

Rules:
- Always start a turn with "Thought:".
- Output exactly ONE Action per turn, then STOP and wait for the Observation.
- Never invent tools. Never call tools inside Action Input.
- If a tool returns an error, reason about it in the next Thought and try a different approach.
- When computing math, substitute numeric values from prior Observations directly. Do not call tools inside calculations.

{FEW_SHOT_EXAMPLE}

Now answer the real question below.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. PARSING THE MODEL'S TEXT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

# Each regex captures one piece of a single ReAct step.
THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action:|Final Answer:)|$)", re.DOTALL)
ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)")
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)
FINAL_RE = re.compile(r"Final Answer:\s*(.+?)\s*$", re.DOTALL)


def parse_react_step(text: str):
    """Parse a model's output. Returns one of:
        ("final", answer_str)
        ("action", tool_name, args_dict)
    Raises ValueError if neither pattern matches.
    """
    # Final Answer has priority — if we see it, we're done.
    final = FINAL_RE.search(text)
    if final:
        return ("final", final.group(1).strip())

    action = ACTION_RE.search(text)
    if not action:
        raise ValueError(f"No Action or Final Answer found in:\n{text}")

    action_input = ACTION_INPUT_RE.search(text)
    if not action_input:
        raise ValueError(f"Action '{action.group(1)}' has no Action Input:\n{text}")

    tool_name = action.group(1).strip()
    try:
        args = json.loads(action_input.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad JSON in Action Input ({e}): {action_input.group(1)}") from e

    return ("action", tool_name, args)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOL DISPATCHER (mirrors my-agent.py / but signature is name + args)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_tool(name: str, args: dict):
    """Run a tool by name. Returns a dict (either the tool result or an error)."""
    fn = available_functions.get(name)
    if fn is None:
        return {"error": f"Tool '{name}' is not registered."}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. THE REACT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_react_agent(user_prompt: str, max_steps: int = 8, model: str = "llama3.2:latest"):
    """Run a ReAct agent until it emits Final Answer or hits max_steps."""
    system_prompt = build_react_prompt()

    # Transcript grows turn-by-turn. We send it as a single user message
    # because Ollama chat() doesn't natively support a multi-turn ReAct
    # format — we synthesize the whole conversation ourselves.
    transcript = f"Question: {user_prompt}\n"

    for step in range(1, max_steps + 1):
        print(f"\n========== REACT STEP {step} ==========")

        # Ask the model to produce the next Thought/Action/Final Answer.
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )
        model_text = (response.message.content or "").strip()
        transcript += model_text + "\n"

        # Show what the model said.
        print("--- MODEL OUTPUT ---")
        print(model_text)
        print("--------------------")

        # Try to parse it.
        try:
            parsed = parse_react_step(model_text)
        except ValueError as e:
            # Parse failure: feed the error back so the model can recover.
            print(f"[parse error] {e}")
            transcript += (
                f"Observation: Your last reply did not match the required "
                f"Thought/Action/Action Input format. System error: {e}\n"
            )
            continue

        if parsed[0] == "final":
            print(f"\nFINAL ANSWER: {parsed[1]}")
            return parsed[1]

        # Otherwise it's an action.
        _, tool_name, args = parsed
        print(f"\nACTION: {tool_name}({args})")

        # Run the tool, capture result, append as Observation.
        result = dispatch_tool(tool_name, args)
        observation_json = json.dumps(result)
        print(f"OBSERVATION: {observation_json}")

        transcript += f"Observation: {observation_json}\n"

    # Fell out of the loop without a Final Answer.
    print(f"\n[max_steps={max_steps} reached without Final Answer]")
    return "Agent stopped: maximum steps reached without a final answer."


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    user_prompt = input("\nYou: ").strip()
    if not user_prompt:
        raise SystemExit("Empty prompt — nothing to ask the agent.")
    answer = run_react_agent(user_prompt)
    print("\n=== ANSWER ===")
    print(answer)
