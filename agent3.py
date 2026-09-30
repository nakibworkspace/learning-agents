from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.2:latest", temperature=0.0)


def research_subagent(query: str) -> str:
    """Look up information about a topic. Use this when you need facts."""
    # Stub — real version would call a search API or fetch a URL
    return f"[Stubbed research results about: {query}]"

def summary_subagent(text: str) -> str:
    """Rewrite text clearly for the user. Use this to format a final answer."""
    # Stub — could be a separate LLM call
    return f"[Stubbed summary of: {text}]"

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


PARENT_PROMPT = """You are an orchestrator. You do NOT answer questions directly.
You have three specialist tools you MUST delegate to:
  - research: for finding facts
  - summary: for summarizing text

If the user's request needs multiple steps (e.g., "research X then summarize"),
delegate to the right tool for each step. NEVER answer from your own knowledge.
After you have all the information, give a final answer to the user."""

def parent_node(state: State):
    response = llm_with_tools.invoke([
        SystemMessage(content=PARENT_PROMPT),
        *state["messages"],
    ])
    return {"messages": [response]}

def tool_node(state: State):
    """Standard LangGraph tool node — runs whichever tool the parent called.
    Handles malformed tool calls gracefully (some small models pass empty args)."""
    last_msg = state["messages"][-1]
    results = []
    tool_map = {t.name: t for t in tools}
    for call in last_msg.tool_calls:
        tool_fn = tool_map.get(call["name"])
        if tool_fn is None:
            results.append(ToolMessage(
                content=f"Error: unknown tool {call['name']}",
                tool_call_id=call["id"],
            ))
            continue
        try:
            # If the model passed empty args, fall back to the user's last message
            args = call["args"] if call["args"] else {}
            if not args:
                user_msgs = [m for m in state["messages"]
                             if type(m).__name__ == "HumanMessage"]
                if user_msgs:
                    # Pick a sensible default param based on the tool name
                    default_param = "query" if "research" in call["name"] else \
                                    "text" if "summarize" in call["name"] else \
                                    "problem"
                    args = {default_param: user_msgs[-1].content}
            result = tool_fn.invoke(args)
            results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        except Exception as e:
            results.append(ToolMessage(
                content=f"Error running {call['name']}: {e}",
                tool_call_id=call["id"],
            ))
    return {"messages": results}

def should_continue(state: State) -> str:
    """If the parent called a tool, run it. Otherwise, end."""
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = StateGraph(State)

graph.add_node("parent", parent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "parent")
graph.add_conditional_edges("parent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "parent")  # ← KEY: back to parent, NOT END

checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "subagent-demo"}}

# Turn 1 — establish identity (should be answered by writer or researcher)
result1 = agent.invoke(
    {"messages": [("user", "My name is Raj. Remember it.")]},
    config=config,
)
print("=== TURN 1 ===")
print(result1["messages"][-1].content)
print()

# Turn 2 — ask the agent to recall (no re-telling, same thread_id)
result2 = agent.invoke(
    {"messages": [("user", "What's my name?")]},
    config=config,
)
print("=== TURN 2")
print(result2["messages"][-1].content)
print()

# Show the full conversation history the agent now sees
print("=== FULL HISTORY ON THIS THREAD ===")
for m in result2["messages"]:
    print(f"[{type(m).__name__}] {m.content}")

# === STREAMING DEMO ===
print("\n=== STREAMING (per-node) ===")
for chunk in agent.stream(
    {"messages": [("user", "What is Python?")]},
    config={"configurable": {"thread_id": "stream_demo"}},
):
    # chunk = {node_name: {state_field: new_value}}
    node_name = list(chunk.keys())[0]
    state_update = chunk[node_name]
    msgs = state_update.get("messages", [])
    for m in msgs:
        kind = type(m).__name__
        if kind == "AIMessage":
            print(f"  [{node_name}] AIMessage: {m.content!r}")
            if m.tool_calls:
                print(f"           tool_calls: {[(tc['name'], tc['args']) for tc in m.tool_calls]}")
        elif kind == "ToolMessage":
            print(f"  [{node_name}] ToolMessage: {m.content!r}")
        else:
            print(f"  [{node_name}] {kind}: {m.content!r}")