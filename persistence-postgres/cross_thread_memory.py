"""
Cross-thread memory — facts about the user persist across conversations.

This is the production pattern:
  - checkpointer: remembers THIS thread's conversation
  - store: remembers FACTS about the user across all threads

Concretely:
  thread_id = "user-raj-conv-1"  →  conversation 1 of user raj
  thread_id = "user-raj-conv-2"  →  conversation 2 of user raj
  facts about raj                →  stored ONCE in the store, available in BOTH threads

Run:
    python3 cross_thread_memory.py conv1   # conversation 1 — save some facts
    python3 cross_thread_memory.py conv2   # conversation 2 — recall those facts
"""

import sys
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"
USER_ID = "raj"
NS = ("user_profile", USER_ID)


def build_agent():
    # checkpointer: per-thread conversation memory
    cm_saver = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = cm_saver.__enter__()
    checkpointer.setup()
    checkpointer._cm = cm_saver

    # store: cross-thread user memory
    cm_store = PostgresStore.from_conn_string(DB_URI)
    store = cm_store.__enter__()
    store.setup()
    store._cm = cm_store

    # Tools read/write the store. The store is captured in closure — same
    # instance shared across threads because it's the same Postgres DB.
    @tool
    def remember(key: str, value: str) -> str:
        """Save a fact about the user. Available across all conversations."""
        store.put(NS, key, {"value": value})
        return f"saved {key}"

    @tool
    def recall(key: str) -> str:
        """Recall a fact about the user. Works across all conversations."""
        it = store.get(NS, key)
        return it.value["value"] if it else f"unknown: {key}"

    @tool
    def who_am_i() -> str:
        """List all known facts about the user."""
        items = list(store.search(NS))
        if not items:
            return "I know nothing about you yet."
        return "\n".join(f"- {it.key} = {it.value.get('value', it.value)}" for it in items)

    agent = create_agent(
        model=ChatOllama(model="llama3.2:latest", temperature=0.0),
        tools=[remember, recall, who_am_i],
        checkpointer=checkpointer,
        store=store,
        system_prompt=(
            f"You are talking to user '{USER_ID}'. You have tools to remember "
            "and recall facts about them. Use them proactively. When asked "
            "'who am I', call who_am_i. When told a fact, call remember. "
            "When asked a question about the user, call recall."
        ),
    )
    return agent


def run(thread_id: str, opening_prompt: str):
    print(f"\nTHREAD: {thread_id}")
    print(f"USER:   {opening_prompt}\n")

    agent = build_agent()
    cfg = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": opening_prompt}]},
        config=cfg,
    )
    print(f"AGENT:  {result['messages'][-1].content}\n")

    # Turn 2: ask it to recall (proves the store works within the same thread)
    followup = "What do you know about me?"
    print(f"USER:   {followup}")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": followup}]},
        config=cfg,
    )
    print(f"AGENT:  {result['messages'][-1].content}")


def conv1_mode():
    """Conversation 1 of user raj."""
    run(
        "user-raj-conv-1",
        "Hi! My name is Raj. I'm a Python developer who works on AI agents. "
        "Please remember my name and profession.",
    )


def conv2_mode():
    """Conversation 2 of the same user, different thread.

    This proves: even though the checkpointer is fresh (different thread),
    the store still has facts from conversation 1.
    """
    run(
        "user-raj-conv-2",
        "Hi! Do you know who I am? What do you know about me?",
    )


def interactive_mode(thread_id):
    """REPL on a specific thread — useful for ad-hoc testing."""
    print(f"INTERACTIVE on thread '{thread_id}' (type 'exit' to quit)")

    agent = build_agent()
    cfg = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=cfg,
        )
        print(f"AGENT: {result['messages'][-1].content}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "conv1"
    if mode == "conv1":
        conv1_mode()
    elif mode == "conv2":
        conv2_mode()
    elif mode == "chat" and len(sys.argv) >= 3:
        interactive_mode(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} [conv1|conv2|chat <thread_id>]")
        sys.exit(1)