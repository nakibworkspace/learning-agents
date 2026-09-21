from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

@tool
def research(query: str) -> str:
    """Look up information about a topic. Use this when you need facts."""
    # Stub — real version would call a search API or fetch a URL
    return f"[Stubbed research results about: {query}]"

@tool
def summary(text: str) -> str:
    """Rewrite text clearly for the user. Use this to format a final answer."""
    # Stub — could be a separate LLM call
    return f"[Stubbed summary of: {text}]"

tools = [research, summary]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

AGENT_SYSTEM = """You are an agent. Decide what to do next.
   Reply with EXACTLY one of:
     RESEARCH: <what to look up>
     FINAL: <your final answer>"""

def agent_node(state: State):
    msgs = state["messages"] + [SystemMessage(content=AGENT_SYSTEM)]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

def tool_node(state: State):
    """Runs whichever tool is called"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    results = []
    for call in tool_calls:
        tool = tool_map[call["name"]]
        result = tool.invoke(call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": results}

def should_continue(state: State) -> Literal["tools", END]:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

graph = StateGraph(State)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent = graph.compile()

result = agent.invoke(
    {"messages": [("user", "What is LangGraph?")]}
)

print(result)
