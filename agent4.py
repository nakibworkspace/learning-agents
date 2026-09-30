from pathlib import Path
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

# Skills directory lives next to this file
SKILLS_DIR = Path(__file__).parent / "skills"

def load_skill(name: str) -> str:
    """Read a skill markdown file from disk."""
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        available = [p.stem for p in SKILLS_DIR.glob("*.md")]
        return f"[ERROR] Skill '{name}' not found. Available: {available}"
    return path.read_text()

@tool
def use_skill(skill_name: str) -> str:
    """Load a skill's instructions into your context.
    Available skills: 'research' (looking up facts), 'writing' (formatting/summarizing).
    Call this BEFORE using the matching domain tool."""
    return load_skill(skill_name)

@tool
def research(query: str) -> str:
    """Look up information about a topic. Use this when you need facts.
    Requires the 'research' skill to be loaded first via use_skill."""
    return f"[Stubbed research results about: {query}]"

@tool
def summary(text: str) -> str:
    """Format and polish text into a final answer for the user.
    This is the WRITING domain tool — pair it with the 'writing' skill.
    Call use_skill('writing') FIRST to load the writing workflow instructions."""
    return f"[Stubbed summary of: {text}]"

tools = [use_skill, research, summary]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


PARENT_PROMPT = """You are an orchestrator that uses skills to handle tasks.

You have three tools:
  - use_skill(name): loads a skill's instructions into your context.
      Available skill names: 'research', 'writing'.
  - research(query): looks up facts. Pair with the 'research' skill.
  - summary(text): polishes text into a final answer. Pair with the 'writing' skill.

CRITICAL RULES:
1. NEVER end your turn on a tool call. Your LAST message must be plain text
   answering the user directly — no tool_calls on that final message.
2. Before calling a domain tool, ALWAYS call use_skill() first to load its skill.
3. After you've gathered enough information (or if no skill matches), reply with
   a clear, concise natural-language answer to the user. That reply is the end.

Example flow for "tell me about X":
  - Call use_skill('research')   ← load instructions
  - Call research('X')           ← follow the skill's instructions
  - Reply in plain text with what you found. STOP HERE.

Do NOT chain multiple skills unless the user clearly needs both research AND
a polished written summary. For simple conversational queries, just answer
directly without using any tools.

If the request needs both research and writing, load the research skill,
do research, then load the writing skill, then summarize."""

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
graph.add_edge("tools", "parent")

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