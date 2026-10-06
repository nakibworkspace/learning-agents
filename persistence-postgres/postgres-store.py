"""
Postgres store.

Long-term memory that survives process restarts AND crosses thread boundaries.
A checkpointer remembers the conversation on one thread.
A store remembers FACTS about the user across all threads (and all sessions).

The mental model:
  checkpointer  = "what was said in THIS conversation"
  store         = "what we KNOW about the user, period"

This file proves:
  1. Save a fact in thread A.
  2. Read it back from thread B in a fresh process.
  3. Namespace the same key across different users (no collision).

Run:
    python3 postgres_store.py write   # saves facts on user 'raj'
    python3 postgres_store.py read    # reads them in a fresh process on thread 'whatever'
    python3 postgres_store.py list    # lists everything for user 'raj'
"""

import sys
from langchain_core.tools import tool
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"

# Namespace = (category, user_id). Tuple-based namespaces give you
# structured lookup: "give me all entries for raj", or "give me all entries in user_profile".
# Different user_ids never collide because they're in different namespaces.
USER_ID = "raj"
NAMESPACE = ("user_profile", USER_ID)


def build_store():
    # Same context-manager pattern as PostgresSaver.
    cm = PostgresStore.from_conn_string(DB_URI)
    store = cm.__enter__()
    # .setup() creates the 'store' table (separate from 'checkpoints').
    # Idempotent — safe to call every startup.
    store.setup()
    store._cm = cm
    return store


def remember(store, key: str, value: dict) -> str:
    # store.put writes to Postgres. The 3rd arg is a dict (any JSON-serializable value).
    store.put(NAMESPACE, key, value)
    return f"saved {key}={value}"


def recall(store, key: str):
    # store.get returns an Item or None. Item.value is your dict.
    item = store.get(NAMESPACE, key)
    return item.value if item else None


def search(store, prefix: str = ""):
    # store.search walks keys starting with prefix (lexical match).
    # Use prefix="" to list everything in the namespace.
    return list(store.search(NAMESPACE, prefix=prefix))


# ---------------------------------------------------------
# MODE: write — save a few facts
# ---------------------------------------------------------
def write_mode():
    print("WRITE mode — saving facts about user 'raj'")

    store = build_store()

    # Multiple writes to the same namespace. Each (namespace, key) is unique.
    remember(store, "name", {"value": "Raj"})
    remember(store, "profession", {"value": "AI engineer"})
    remember(store, "favorite_python", {"value": "asyncio"})

    items = search(store)
    print(f"\n[disk] {len(items)} entries in namespace {NAMESPACE}:")
    for it in items:
        print(f"  - {it.key} = {it.value}")

    try:
        store._cm.__exit__(None, None, None)
    except Exception:
        pass


# ---------------------------------------------------------
# MODE: read — recall in a fresh process, on a DIFFERENT thread
# ---------------------------------------------------------
def read_mode():
    print("READ mode — fresh process, different thread_id")

    store = build_store()

    # Simulate: agent is in a brand new conversation (no checkpointer context).
    # But the store is the SAME — should still find his facts.
    name = recall(store, "name")
    profession = recall(store, "profession")
    missing = recall(store, "favorite_python")

    print(f"\n[store] name           = {name}")
    print(f"[store] profession     = {profession}")
    print(f"[store] favorite_python= {missing}")

    # Cross-thread proof: even if your checkpointer thread is fresh,
    # the store has the facts.
    if name and name["value"] == "Raj":
        print("\nCROSS-THREAD MEMORY PROVEN — store survived process death.")
    else:
        print("\nNo facts found. Did you run write_mode first?")

    try:
        store._cm.__exit__(None, None, None)
    except Exception:
        pass


# ---------------------------------------------------------
# MODE: list — show everything in the namespace
# ---------------------------------------------------------
def list_mode():
    print("LIST mode — all entries in the namespace")

    store = build_store()
    items = search(store)

    print(f"\n[disk] {len(items)} entries in namespace {NAMESPACE}:")
    for it in items:
        print(f"  - {it.key} = {it.value}")

    try:
        store._cm.__exit__(None, None, None)
    except Exception:
        pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"
    if mode == "write":
        write_mode()
    elif mode == "read":
        read_mode()
    elif mode == "list":
        list_mode()
    else:
        print(f"Unknown mode: {mode}. Use 'write', 'read', or 'list'.")
        sys.exit(1)