"""
Stage S2 — Long-term memory via the store.

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v3.py

What this adds over agent_v2.py:
    - InMemoryStore wired into create_agent
    - Two tools: remember_about_user (writes) and recall_about_user (reads)
    - The agent can now save and recall facts across threads

Order of operations:
    TODO 1 — Define remember_about_user + recall_about_user tools (real store I/O)
    TODO 2 — Wire store= into create_agent
    TODO 3 — Decide how user_id flows (config vs hardcoded vs parsed from message)
    TODO 4 — run() that proves cross-thread recall works
"""

# ============================================================
# IMPORTS (don't touch unless you know why)
# ============================================================
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# Store + checkpointer instances (don't touch unless you know why)
# These are the SAME memory primitives from agent_v2 — store is the only NEW one.
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()

NS = ("user_profile", "u-001")


# ============================================================
# Tools from agent_v2.py — copy them in here
# ============================================================
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


@tool
def current_date() -> str:
    """Return today's date in YYYY-MM-DD format. Use this when the user asks 'what's the date', 'what day is it', or 'today's date'."""
    from datetime import date
    return date.today().isoformat()


# ============================================================
# TODO 1: store tools (this is the new part)
# ============================================================
# Single source of truth for who the current user is.
# Tools read this directly — no prompt-reliant string parsing,
# no chance the model gets user_id wrong.
USER_ID = "u-001"


@tool
def remember_about_user(key: str, value: str) -> str:
    """Save a personal fact about the current user that should be remembered across all conversations.
    Use when the user says 'remember...', 'my X is Y', 'I like Z', 'save this', etc.
    Always call this when the user shares a personal preference or fact."""
    store.put(("user_profile", USER_ID), key, {"value": value})
    return f"Saved {key}"

@tool
def recall_about_user(key: str) -> str:
    """Recall a previously saved fact about the current user across conversations.
    Use when the user asks 'what's my X?', 'do you remember my Y?', 'what did I tell you about Z?'.
    Returns the saved value, or 'unknown' if not found."""
    item = store.get(("user_profile", USER_ID), key)
    if item is None:
        return f"unknown: no saved value for {key}"
    return f"{key} = {item.value['value']}"


# ============================================================
# TODO 2: build the agent with create_agent
# ============================================================
agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    # Trimmed to ONLY the store tools — fewer tools = llama3.2 actually uses them.
    tools=[remember_about_user, recall_about_user],
    checkpointer=checkpointer,
    store=store,
    system_prompt=(
        "You have two tools: remember_about_user(key, value) and recall_about_user(key). "
        "When the user tells you to remember something or shares a personal fact, you MUST "
        "call remember_about_user with a short snake_case key and the value. "
        "When the user asks you to recall a fact, you MUST call recall_about_user with the key. "
        "Do not answer from your own memory — always use the tools."
    ),
)


# ============================================================
# TODO 3: decide how user_id flows
# ============================================================
# Resolved above — USER_ID is a module constant read by the tools directly.
# This is the deterministic path (Option B): no prompt parsing required.


# ============================================================
# TODO 4: prove cross-thread recall
# ============================================================
def run():
    if agent is None:
        raise RuntimeError("agent is None. Finish TODO 2 first.")
    
    thread_a = "thread-A"
    thread_b = "thread-B"
    user_id = "u-001"

    # ---- Turn 1 on thread A: save a fact ----
    print("=== Turn 1 (thread A): save a fact ===")
    r1 = agent.invoke(
        {"messages": [{"role": "user",
                        "content": "Please remember that my favorite color is teal. Use the remember tool with key 'favorite_color' and value 'teal'."}]},
        config={"configurable": {"thread_id": thread_a}},
    )
    print("AGENT:", r1["messages"][-1].content)

    # ---- Turn 2 on thread B: recall it ----
    # Different thread_id = fresh short-term history.
    # If the store works, the agent still remembers teal.
    print("\n=== Turn 2 (thread B, different thread_id!): recall it ===")
    r2 = agent.invoke(
        {"messages": [{"role": "user",
                        "content": "What is my favorite color? Use the recall tool with key 'favorite_color'."}]},
        config={"configurable": {"thread_id": thread_b}},
    )
    print("AGENT:", r2["messages"][-1].content)

    # ---- Prove the store really has it (bypass the LLM) ----
    print("\n=== Direct store inspection ===")
    item = store.get(("user_profile", user_id), "favorite_color")
    print("store has:", item.value if item else "<missing>")


if __name__ == "__main__":
    run()
