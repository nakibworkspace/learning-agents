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

# Handoff tools — these are SIGNALS, not real tools. Their bodies never run.
@tool
def transfer_to_researcher():
    """Hand off the conversation to the researcher specialist. Use when the user
    needs facts looked up, web searches, or factual information gathered."""
    return "HANDOFF_TO_RESEARCHER"

@tool
def transfer_to_writer():
    """Hand off the conversation to the writing specialist. Use when the user
    wants text rewritten, formatted, or summarized clearly."""
    return "HANDOFF_TO_WRITER"

HANDOFF_TOOLS = [transfer_to_researcher, transfer_to_writer]
llm_with_handoffs = llm.bind_tools(HANDOFF_TOOLS)

TRIAGE_PROMPT = """You are a triage agent.
Look at the user's request and decide which specialist should handle it:
  - Researcher (looking up facts, web search, finding information) → call transfer_to_researcher
  - Writer (rewriting, formatting, summarizing existing text)       → call transfer_to_writer
Do not answer the question yourself. Just call the right transfer tool."""

def triage_node(state: State):
    response = llm_with_handoffs.invoke([
        SystemMessage(content=TRIAGE_PROMPT),
        *state["messages"],
    ])
    return {"messages": [response]}

def researcher_node(state: State):
    """Runs the research tool with whatever query is in the user's last message."""
    # Find the original user request
    user_msgs = [m for m in state["messages"] if type(m).__name__ == "HumanMessage"]
    query = user_msgs[-1].content if user_msgs else ""
    # Stub: just call the research tool directly
    result = research.invoke({"query": query})
    return {"messages": [AIMessage(content=f"[RESEARCHER] {result}")]}

def writer_node(state: State):
    """Takes the last research output (or user text) and produces a clean summary."""
    user_msgs = [m for m in state["messages"] if type(m).__name__ == "HumanMessage"]
    text = user_msgs[-1].content if user_msgs else ""
    result = summary.invoke({"text": text})
    return {"messages": [AIMessage(content=f"[WRITER] {result}")]}

def handoff_seam(state: State) -> Command[Literal["researcher_node", "writer_node", "__end__"]]:
    """Reads the last message (which should be triage's handoff tool call),
    injects a synthetic ToolMessage so history stays valid, then routes
    to the right specialist."""
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    tool_msg = ToolMessage(
        content=f"Handoff to {tool_call['name']} complete.",
        tool_call_id=tool_call["id"],
    )
    if tool_call["name"] == "transfer_to_researcher":
        return Command(goto="researcher_node", update={"messages": [tool_msg]})
    if tool_call["name"] == "transfer_to_writer":
        return Command(goto="writer_node", update={"messages": [tool_msg]})
    return Command(goto=END)

graph = StateGraph(State)

graph.add_node("triage_agent", triage_node)
graph.add_node("handoff_seam", handoff_seam)
graph.add_node("researcher_node", researcher_node)
graph.add_node("writer_node", writer_node)

graph.add_edge(START, "triage_agent")
graph.add_edge("researcher_node", END)
graph.add_edge("writer_node", END)

# After triage: if a handoff tool was called → go to seam, else END
def triage_router(state: State) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "handoff_seam"
    return END

graph.add_conditional_edges("triage_agent", triage_router, {
    "handoff_seam": "handoff_seam",
    END: END,
})

checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "handoff-demo"}}

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