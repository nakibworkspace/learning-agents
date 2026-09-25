"""
Scaffold for Stage S1 — Migrate your hand-rolled LangGraph agent to LangChain v1 `create_agent`.

This is a FRESH file. Your existing agent.py in /agents-dive stays untouched.
Copy this file to agents-dive/agent_v2.py and fill the TODOs.

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v2.py

Order of operations:
    TODO 1 — Migrate the basic ReAct agent (drop-in replacement of agent.py)
    TODO 2 — Add a third tool (real-world value)
    TODO 3 — Streaming demo (compare modes)
"""

# ============================================================
# IMPORTS (don't touch unless you know why)
# ============================================================
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# TODO 1: migrate tools from agent.py
# ------------------------------------------------------------

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


# ============================================================
# TODO 2: a third tool of your own design
# ------------------------------------------------------------
@tool
def current_date() -> str:
    """Return today's date in YYYY-MM-DD format. Use this when the user asks 'what's the date', 'what day is it', or 'today's date'."""
    from datetime import date
    return date.today().isoformat()


# ============================================================
# TODO 3: build the agent with create_agent

agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    tools=[research, summary, current_date],
    system_prompt="You are a good and concise research agent",
    checkpointer = InMemorySaver()
)


# ============================================================
# TODO 4: streaming demo + multi-turn

config = {"configurable": {"thread_id": "v2-demo"}}


def stream_turn(agent, user_msg: str, thread_id: str, mode: str = "messages"):
    """
    One streamed turn against the agent.

    mode="messages"  -> (chunk, meta) pairs, tokens as they arrive
    mode="updates"   -> dict of {node_name: state_delta} per step

    Why two modes? "messages" is for UI typewriter feel.
    "updates" is for debugging/observability (see WHICH node did WHAT).
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    print(f"\nUSER: {user_msg}")

    if mode == "messages":
        for chunk, _meta in agent.stream(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=cfg,
            stream_mode="messages",
        ):
            # chunk may be a message OR a list of message chunks (langchain versions vary)
            content = getattr(chunk, "content", None)
            if not content:
                # some versions yield tuples or message chunks
                continue
            if isinstance(content, str):
                print(content, end="", flush=True)
            elif isinstance(content, list):
                # content can be a list of blocks (text, tool_use, etc.)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(block.get("text", ""), end="", flush=True)
                    elif hasattr(block, "text"):
                        print(block.text, end="", flush=True)
        print()  # newline after stream finishes
    else:
        # "updates" mode: see exactly which node fired and what changed
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=cfg,
            stream_mode="updates",
        ):
            for node_name, state_delta in event.items():
                msgs = state_delta.get("messages", [])
                for m in msgs:
                    kind = type(m).__name__
                    extra = ""
                    if kind == "AIMessage" and getattr(m, "tool_calls", None):
                        extra = f" tool_calls={[(tc['name'], tc['args']) for tc in m.tool_calls]}"
                    content = getattr(m, "content", "")
                    print(f"  [{node_name}] {kind}: {content!r}{extra}")
    print()


def run():
    """
    Demo: 3 turns on the same thread_id.
    - Turn 1: streaming tokens (mode="messages")
    - Turn 2: streaming per-node updates (mode="updates") — shows the tool routing
    - Turn 3: plain invoke — proves checkpointer remembered turns 1+2
    """
    if agent is None:
        raise RuntimeError("agent is None. Finish TODO 3 first.")

    # Turn 1 — typewriter feel
    stream_turn(agent, "What's today's date?", config["configurable"]["thread_id"], mode="messages")

    # Turn 2 — see the routing (model -> tool -> model)
    stream_turn(
        agent,
        "Research LangChain v1 middleware and summarize it in 1 sentence.",
        config["configurable"]["thread_id"],
        mode="updates",
    )

    # Turn 3 — proves memory: ask the model to recall the previous topic
    print("--- Turn 3 (verify memory, plain invoke) ---")
    r = agent.invoke(
        {"messages": [{"role": "user", "content": "What topic did I just ask you to research?"}]},
        config=config,
    )
    print("ASSISTANT:", r["messages"][-1].content)


if __name__ == "__main__":
    run()
