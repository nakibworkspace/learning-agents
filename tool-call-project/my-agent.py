from ollama import chat
import requests

# tool schema
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
                        "description": "The name of the city, e.g. Athens"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression such as 25 * 4"
                    }
                },
                "required": ["expression"]
            }
        }
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
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information and return a list of results with title, URL, and snippet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'latest NVIDIA earnings'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5, max 10)",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# define python function for the tool
def get_current_weather(city: str, unit: str = "celsius"):

    # Geocode city -> latitude/longitude
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1
        }
    ).json()

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    # Get weather
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "temperature_unit": unit
        }
    ).json()

    temp = weather["current"]["temperature_2m"]

    return {
        "city": city,
        "temperature": temp,
        "unit": unit
    }


# Safe calculator: parses a math expression and evaluates it without using eval()
# (Never give a model direct access to eval — it can execute arbitrary Python.)
import ast
import operator


def calculate(expression: str):
    # Whitelist of allowed binary operators
    _binops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def calculate(expression):
      tree = ast.parse(expression, mode="eval")
      for node in ast.walk(tree):
          if isinstance(node, ast.Call):
              return {
                  "error": "calculate only supports pure arithmetic. "
                           "Use numeric literals from prior tool results; "
                           "do not call other tools inside the expression."
              }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _binops:
            return _binops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _unops:
            return _unops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _eval(tree.body)}


def get_time(city: str):
    # Look up the city's timezone via the Open-Meteo geocoding API
    # (same free, no-key endpoint we already use for weather).
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    ).json()

    timezone = geo["results"][0]["timezone"]

    # Ask Open-Meteo for the current time in that timezone.
    # Passing latitude/longitude is required by the API but ignored
    # when we only request `current=...`.
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m",
            "timezone": timezone,
        }
    ).json()

    return {
        "city": city,
        "timezone": timezone,
        "datetime": resp["current"]["time"],
    }


# DuckDuckGo web search (no API key required).
# pip install duckduckgo-search
from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5):
    # Clamp to a sensible range — DDGS will complain or return junk if you
    # ask for too many results.
    max_results = max(1, min(int(max_results), 10))

    results = []
    # DDGS() is a context manager. The .text() method returns an iterator
    # of dicts with keys: title, href, body.
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

# map tool name to function
available_functions = {
    "get_current_weather": get_current_weather,
    "calculate": calculate,
    "get_time": get_time,
    "search_web": search_web,
}

# initialize conversation
user_prompt = input("You: ").strip()
if not user_prompt:
    raise SystemExit("Empty prompt — nothing to ask the agent.")

messages = [
    {
        "role": "user",
        "content": user_prompt
    }
]

while True:
    # ask the model
    response = chat(
        model="llama3.2:latest",
        messages=messages,
        tools=tools
    )

    # model didn't call the tool
    if not response.message.tool_calls:
        print("\nMODEL ANSWER:")
        print(response.message.content)
        break

    # check whether the model called the tool
    if response.message.tool_calls:

        # Add the assistant's tool-call message to history
        ## model requested for one or more tools
        messages.append(response.message)

        # There can be multiple tool calls
        for tool_call in response.message.tool_calls:

            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print("\nMODEL WANTS TO CALL:")
            print(function_name)

            print("ARGUMENTS:")
            print(arguments)

            # ----------------------------------------------------
            # Find the actual Python function
            # ----------------------------------------------------

            function_to_call = available_functions.get(function_name)

            if function_to_call:
                # Execute Python function
                try:
                    result = function_to_call(**arguments)
                except Exception as e:
                    result = {
                        "error": f"{type(e).__name__}: {str(e)}"
                    }

                print("\nTOOL RESULT:")
                print(result)

            else:

                result = f"Tool {function_name} not found"

            # ----------------------------------------------------
            # Give tool result back to the model
            # ----------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_name": function_name,
                # tool_call_id is required by the chat protocol so the model
                # can associate each tool result with the specific call it
                # answers. Without it, multi-tool rounds can confuse the model.
                "tool_call_id": getattr(tool_call, "id", None) or tool_call.function.name,
                "content": str(result)
            })

        # # ask model again with tool result
        # final_response = chat(
        #     model="llama3.2:latest",
        #     messages=messages
        # )

        # print("\nFINAL ANSWER:")
        # print(final_response.message.content)

        # # Stop looping — we got our answer
        # break
