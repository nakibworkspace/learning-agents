"""
Stage S3 — SummarizationMiddleware added on top of v3.

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v4.py

What this adds over agent_v3.py:
    - SummarizationMiddleware wired into the agent
    - Long conversations get squished into a summary before the model sees them
    - Store + checkpointer still work (they're orthogonal to summarization)

Order of operations:
    TODO 1 — Pick the trigger + keep values (the actual learning moment)
    TODO 2 — Pick the tools to include + write the system prompt
    TODO 3 — Write run() Part A: a conversation that triggers summarization
    TODO 4 — Write run() Part B: prove the store is unaffected by summarization
"""

# ============================================================
# IMPORTS — don't touch
# ============================================================
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# Store + checkpointer — IDENTICAL to v3 (don't touch)
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()
USER_ID = "u-001"


# ============================================================
# Tools — copy these from agent_v3.py (remember + recall)
# Plus one new tool of your choice from agent_v2.py.
# The new tool exists so summarization has something meaty to chew on.
# ============================================================
@tool
def remember_about_user(key: str, value: str) -> str:
    """Save a personal fact about the current user across conversations."""
    store.put(("user_profile", USER_ID), key, {"value": value})
    return f"Saved {key}"

@tool
def recall_about_user(key: str) -> str:
    """Recall a previously saved fact about the current user."""
    item = store.get(("user_profile", USER_ID), key)
    if item is None:
        return f"unknown: no saved value for {key}"
    return f"{key} = {item.value['value']}"

# TODO 2a: add ONE more tool here. Pick from agent_v2.py:
@tool
def current_date() -> str:
    """Return today's date in YYYY-MM-DD format. Use this when the user asks 'what's the date', 'what day is it', or 'today's date'."""
    from datetime import date
    return date.today().isoformat()


# ============================================================
# TODO 1: pick your trigger + keep values
# ============================================================

SUMMARY_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)

# TODO 1: replace the two placeholders below with your chosen trigger and keep tuples.
TRIGGER = ("tokens", 100)
KEEP    = ("messages", 4)


# ============================================================
# TODO 2: build the agent
# ============================================================
agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    # TODO 2b: which tools go here? remember, recall, + your new one?
    tools=[remember_about_user, recall_about_user, current_date],
    checkpointer=checkpointer,
    store=store,
    middleware=[
        SummarizationMiddleware(
            model=SUMMARY_MODEL,
            trigger=TRIGGER,
            keep=KEEP,
        ),
    ],
    # TODO 2c: write a system_prompt that tells the agent WHEN to use each tool.
    system_prompt=(
        "You are a concise assistant. "
        "Use remember_about_user ONLY when the user shares a personal fact. "
        "Use recall_about_user ONLY when the user asks about a saved fact. "
        "Use current_date when the user asks about today's date."
    ),
)


# ============================================================
# TODO 3: write run() Part A — long conversation triggers summarization
# ============================================================
def run():
    """
    Two-part demo:
      Part A — long conversation triggers summarization (peek at buffer)
      Part B — a remembered fact still survives the summary (peek at store)
    """
    if not agent:
        raise RuntimeError("agent is None. Finish TODO 2 first.")

    cfg = {"configurable": {"thread_id": "v4-demo"}}

    # ---- Part A: drive a long conversation ----
    print("=== Part A: drive a long conversation ===")
    for i in range(6):
        msg = (
            f"Turn {i}: tell me a fun fact about the number {i} "
            f"and explain why context engineering matters in agent systems."
        )
        print(f"\nUSER: {msg}")
        r = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config=cfg)
        print(f"AGENT: {r['messages'][-1].content}")

    # TODO 3b: peek at the buffer and print THREE diagnostics:
    #   1) how many messages are in the buffer (should be << 2 * turns)
    #   2) what type is msgs[0]?
    #   3) the first 150 chars of msgs[0].content (should start with "Here is a summary")
    state = agent.get_state(cfg)
    msgs= state.values["messages"]
    print(f"\n--- Buffer diagnostic ---")
    print(f"Messages in buffer: {len(msgs)}")
    print(f"First message type: {type(msgs[0]).__name__}")
    print(f"First message preview: {str(msgs[0].content)[:150]}")

    # ---- Part B: store survives the summary ----
    print("\n=== Part B: store is unaffected by summarization ===")
    cfg_b = {"configurable": {"thread_id": "v4-demo-fresh-thread"}}

    # TODO 4: write two turns on cfg_b:
    #   turn 1 — ask the agent to remember your favorite animal using the remember tool
    #   turn 2 — ask the agent to recall your favorite animal using the recall tool
    # Then DIRECTLY peek at the store: store.get(("user_profile", USER_ID), "favorite_animal")
    # and print the value. It should be a dict with your chosen animal.
    # turn 1 — save
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": (
            "Remember that my favorite animal is a capybara. "
            "Use the remember tool with key 'favorite_animal' and value 'capybara'."
        )}]},
        config=cfg_b,
    )
    print(f"AGENT: {r1['messages'][-1].content}")

    # turn 2 — recall
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": (
            "What is my favorite animal? Use the recall tool with key 'favorite_animal'."
        )}]},
        config=cfg_b,
    )
    print(f"AGENT: {r2['messages'][-1].content}")

    # direct store peek
    item = store.get(("user_profile", USER_ID), "favorite_animal")
    print(f"\nstore has: {item.value if item else '<missing>'}")


if __name__ == "__main__":
    run()
