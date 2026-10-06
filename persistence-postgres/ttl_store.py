"""
TTL on the store — automatic expiry of facts.

Some facts expire. Examples:
  - "user is on a 7-day free trial"  → expires after 7 days
  - "current_promo_code"              → expires after 24 hours
  - "session_token"                   → expires after 1 hour

PostgresStore supports TTL via store.put(namespace, key, value, ttl=...).

This file proves:
  1. Set a fact with ttl=2 (seconds, for demo).
  2. Read it back right away → found.
  3. Sleep past the TTL.
  4. Read again → expired (returns None).

Run:
    python3 ttl_store.py demo
"""

import sys
import time
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"
NS = ("user_profile", "ttl_demo_user")


def build_store():
    cm = PostgresStore.from_conn_string(DB_URI)
    store = cm.__enter__()
    store.setup()
    store._cm = cm
    return store


def demo_mode():
    store = build_store()

    # Clean up any prior demo data
    existing = store.get(NS, "session_token")
    if existing:
        store.delete(NS, "session_token")

    # Write a fact with a 2-second TTL
    # ttl_minutes is the unit; we use 0.033 minutes = 2 seconds.
    # (For very short demos, you may need to pass ttl as minutes.)
    store.put(
        NS,
        "session_token",
        {"value": "abc123"},
        ttl=0.033,  # minutes; = 2 seconds
    )

    print(f"\n[1] Wrote session_token with ttl=0.033 min (2 sec)")
    item = store.get(NS, "session_token")
    print(f"[2] Read back immediately: {item.value if item else '(expired)'}")

    print("\n[3] Sleeping 3 seconds...")
    time.sleep(3)

    item = store.get(NS, "session_token")
    if item is None:
        print("[4] Read after TTL: (expired) — TTL works")
    else:
        print(f"[4] Read after TTL: {item.value} — still there, TTL not enforced in this version")

    # Inspect the underlying Postgres row to see expires_at
    try:
        store._cm.__exit__(None, None, None)
    except Exception:
        pass


def crud_mode():
    """Interactive TTL demo — put with custom ttl."""
    store = build_store()
    print("TTL CRUD mode — put/get with TTL")
    print("Commands: put <key> <value> [ttl_minutes] | get <key> | list | exit")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue

        parts = line.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == "exit":
            break
        elif cmd == "list":
            for it in store.search(NS):
                print(f"  {it.key} = {it.value}")
        elif cmd == "get" and len(parts) >= 2:
            it = store.get(NS, parts[1])
            print(it.value if it else "(none)")
        elif cmd == "put":
            # Format: put <key> <value> [ttl_minutes]
            rest = line.split(maxsplit=2)[2] if len(line.split(maxsplit=2)) >= 3 else ""
            pieces = rest.rsplit(maxsplit=1)
            if len(pieces) == 2:
                kv, ttl_str = pieces
                kv_parts = kv.split(maxsplit=1)
                key, value = kv_parts[0], kv_parts[1]
                ttl = float(ttl_str)
                store.put(NS, key, {"value": value}, ttl=ttl)
                print(f"saved {key}={value} ttl={ttl}min")
            else:
                key, value = rest.split(maxsplit=1)
                store.put(NS, key, {"value": value})
                print(f"saved {key}={value} (no TTL)")
        else:
            print("Usage: put <key> <value> [ttl_minutes] | get <key> | list | exit")

    try:
        store._cm.__exit__(None, None, None)
    except Exception:
        pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        demo_mode()
    elif mode == "crud":
        crud_mode()
    else:
        print(f"Unknown mode: {mode}. Use 'demo' or 'crud'.")
        sys.exit(1)