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
    }
]

# define python function for the tool
# def get_current_weather(city: str, unit: str = "celsius"):

#     # Geocode city -> latitude/longitude
#     geo = requests.get(
#         "https://geocoding-api.open-meteo.com/v1/search",
#         params={
#             "name": city,
#             "count": 1
#         }
#     ).json()

#     lat = geo["results"][0]["latitude"]
#     lon = geo["results"][0]["longitude"]

#     # Get weather
#     weather = requests.get(
#         "https://api.open-meteo.com/v1/forecast",
#         params={
#             "latitude": lat,
#             "longitude": lon,
#             "current": "temperature_2m,weather_code",
#             "temperature_unit": unit
#         }
#     ).json()

#     temp = weather["current"]["temperature_2m"]

#     return {
#         "city": city,
#         "temperature": temp,
#         "unit": unit
#     }

## for learning purpose, we are using dummy data
def get_current_weather(city, unit="celsius"):
    return {
        "city": city,
        "temperature": 30,
        "unit": unit
    }


def calculate(expression):
    # Just for our learning experiment
    return eval(expression)


def get_time(city):
    return {
        "city": city,
        "time": "03:30 PM"
    }

# map tool name to function
available_functions = {
    "get_current_weather": get_current_weather,
    "calculate": calculate,
    "get_time": get_time
}

# initialize conversation
messages = [
    {
        "role": "user",
        "content": "What time is it in Tokyo? What's the weather there? What is 45*32?"
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
                result = function_to_call(**arguments)

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
                "content": str(result)
            })

        # ask model again with tool result
        final_response = chat(
            model="llama3.2:latest",
            messages=messages
        )

        print("\nFINAL ANSWER:")
        print(final_response.message.content)

        # Stop looping — we got our answer
        break
