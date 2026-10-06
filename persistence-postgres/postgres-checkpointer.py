"""
Postgres checkpointer.

Replaces InMemorySaver with PostgresSaver so thread state survives
process restarts.

Run modes:
    python3 postgres_checkpointer.py write      writes 15 checkpoints on thread 'pgr-1'
    python3 postgres_checkpointer.py read       reads them back in a fresh process
    python3 postgres_checkpointer.py chat       interactive REPL on thread 'chat-1'

Setup:
    docker run -d --name pg-langgraph -p 5432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres postgres:16

    pip install "langgraph-checkpoint-postgres" psycopg2-binary langchain langchain-ollama
"""

import sys
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver

# Postgres connection. user=postgres, password=postgres, db=postgres, port=5432.
# This is the default Docker setup; change if you point at a real Postgres.
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"


@tool
def echo(msg: str) -> str:
    """Echo back the message. Used to prove the agent was actually invoked."""
    return f"echoed: {msg}"


def build_agent():
    # from_conn_string returns a context manager (in current langgraph-checkpoint-postgres).
    # You must enter it manually to get the actual saver. This is the API quirk.
    cm = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = cm.__enter__()

    # .setup() creates the required tables (checkpoints, checkpoint_blobs, ...)
    # Idempotent — safe to call every startup. Required once per fresh DB.
    checkpointer.setup()

    agent = create_agent(
        model=ChatOllama(model="llama3.2:latest", temperature=0.0),
        tools=[echo],
        checkpointer=checkpointer,
        system_prompt=(
            "You are a friendly, conversational assistant. "
            "Use the echo tool only if the user explicitly asks you to repeat "
            "something. Otherwise just reply naturally and concisely."
        ),
    )

    # Stash the context manager so phase code can close the connection cleanly.
    # Without this, we'd lose the handle to .__exit__() and the connection
    # would leak until process exit.
    checkpointer._cm = cm
    return agent, checkpointer


def write_phase():
    print("PHASE 1: First process — write checkpoints to Postgres")

    agent, checkpointer = build_agent()

    # thread_id is the key for persistence. Same thread_id across processes
    # reads the same history. Different thread_id = isolated conversation.
    cfg = {"configurable": {"thread_id": "pgr-1"}}

    # Turn 1: establish identity.
    # Each invoke writes a checkpoint to Postgres keyed by (thread_id, step).
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Raj. Use the echo tool on my message."}]},
        config=cfg,
    )
    print(f"\n[turn 1] AGENT: {r1['messages'][-1].content}")

    # Turn 2: recall. Same thread_id → agent sees prior turns automatically.
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config=cfg,
    )
    print(f"\n[turn 2] AGENT: {r2['messages'][-1].content}")

    # Inspect what's on disk. The checkpointer's .list() returns all checkpoints
    # for the given config. Multiple per turn because each agent step writes one.
    history = list(checkpointer.list(cfg))
    print(f"\n[disk] Checkpoints stored for thread 'pgr-1': {len(history)}")
    print("\n>>> Now Ctrl-C / kill this process. Postgres keeps the state.")
    print(">>> Then run:  python postgres_checkpointer.py read")

    # Close the Postgres connection cleanly.
    try:
        checkpointer._cm.__exit__(None, None, None)
    except Exception:
        pass


def read_phase():
    print("PHASE 2: Second process — read checkpoints BACK from Postgres")

    agent, checkpointer = build_agent()
    cfg = {"configurable": {"thread_id": "pgr-1"}}

    # This is a fresh process — but Postgres is the same. Should find 15+ checkpoints.
    history = list(checkpointer.list(cfg))
    print(f"\n[disk] Found {len(history)} checkpoints for thread 'pgr-1' in Postgres")

    # The killer move: ask a NEW question on the SAME thread_id.
    # If the agent remembers Raj, persistence is proven.
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name? (process restarted — do you remember?)"}]},
        config=cfg,
    )
    print(f"\n[turn 3, after restart] AGENT: {r3['messages'][-1].content}")

    # r3['messages'] contains the FULL restored history (3 turns) — that's the
    # checkpointer's job: rebuild the message list from Postgres before invoke.
    print("\n[restored history]")
    for m in r3["messages"]:
        kind = type(m).__name__
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"  [{kind}] {content[:100]}")

    if "raj" in r3["messages"][-1].content.lower():
        print("\nPERSISTENCE PROVEN — agent remembers across processes.")
    elif len(history) == 0:
        print("\nNo checkpoints found for this thread.")
        print("   Did you run `write` first? That writes 15+ checkpoints.")
        print("   Or did you change the DB_URI / connect to a different Postgres?")
    else:
        print("\nPERSISTENCE FAILED — checkpoints exist but agent didn't recall.")
        print(f"   {len(history)} checkpoints on disk; AI history broken.")
        print("   Possible: model didn't read the restored history, or thread_id drift.")

    try:
        checkpointer._cm.__exit__(None, None, None)
    except Exception:
        pass


def chat_phase():
    """Interactive REPL on a persistent Postgres thread."""
    # Thread_id can be passed on the CLI: python postgres_checkpointer.py chat my-thread-1
    # Default thread is 'chat-1'. Use a custom name to resume that exact thread later.
    thread_id = sys.argv[2] if len(sys.argv) > 2 else "chat-1"
    print(f"PHASE 3: Interactive chat on thread '{thread_id}' (Postgres-backed)")
    print("Type 'exit' to quit. The thread persists in Postgres — resume anytime.")

    agent, checkpointer = build_agent()
    cfg = {"configurable": {"thread_id": thread_id}}

    # On startup, check if this thread has prior history. If yes, we're resuming.
    # If no, we're starting fresh.
    history_before = list(checkpointer.list(cfg))
    if history_before:
        print(f"[disk] Resuming thread '{thread_id}' — {len(history_before)} prior checkpoints")
    else:
        print(f"[disk] New thread '{thread_id}' — starting fresh")

    # Same agent.invoke pattern as write/read, just inside while True.
    # Each iteration appends one user turn to the same thread → Postgres stores it.
    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] Saved thread state to Postgres. Bye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("[exit] Saved thread state to Postgres. Bye.")
            break
        if not user_input:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=cfg,
        )
        print(f"AGENT: {result['messages'][-1].content}")

    try:
        checkpointer._cm.__exit__(None, None, None)
    except Exception:
        pass


if __name__ == "__main__":
    # CLI dispatcher: write | read | chat [thread_id]
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"
    if mode == "write":
        write_phase()
    elif mode == "read":
        read_phase()
    elif mode == "chat":
        chat_phase()
    else:
        print(f"Unknown mode: {mode}. Use 'write', 'read', or 'chat [thread_id]'.")
        sys.exit(1)