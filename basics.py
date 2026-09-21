from langchain.tools import tool
from langchain_ollama import ChatOllama

model = ChatOllama(model= "llama3.2:latest", temperature=0.0)

# define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """add a and b"""
    return a + b

@tool
def divide(a: int, b: int) -> int:
    """divide a and b"""
    return a / b

tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# state
from langchain_core.messages import AnyMessage
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

from langchain_core.messages import SystemMessage

## llm calls node
def llm_calls(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

## tool node
from langchain_core.messages import ToolMessage

def tool_node(state: State):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# logic to determine whether to end (Agent loop)
from typing import Literal
from langgraph.graph import StateGraph, START, END

# conditional edge
def should_continue(state: State) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    # if llm makes tool_call
    if last_message.tool_calls:
        return "tool_node"
    
    # if no tool_call
    return END

# Build agent

# workflow
agent_builder = StateGraph(State)

# nodes
agent_builder.add_node("llm_calls", llm_calls)
agent_builder.add_node("tool_node", tool_node)

# edges
agent_builder.add_edge(START, "llm_calls")
agent_builder.add_edge("tool_node", "llm_calls")
agent_builder.add_conditional_edges("llm_calls", should_continue)

# compile
agent = agent_builder.compile()

try:
    from IPython.display import Image, display
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
except ImportError:
    pass  # not running in a notebook; skip the visualization

# Invoke
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Multiply 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()