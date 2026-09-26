"""
Stage S7 — Real tool integration (cumulative on v7).

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v8.py

What this adds over agent_v7.py:
    - read_file(path)  → real pathlib.Path.read_text()
    - write_file(path, content) → real pathlib.Path.write_text()
    - web_search(query) → real DuckDuckGo search (can fail on network)
    - ToolFailureMiddleware (custom) → catches exceptions and returns a graceful error
      string instead of crashing the agent loop

What this REPLACES from v7:
    - research_doc (stub)  →  real read_file + web_search
    - write_doc (stub)     →  real write_file

Single-tool phases only — llama3.2 won't chain two tools in one invoke.
"""

# ============================================================
# IMPORTS
# ============================================================
import time
from pathlib import Path
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# Store + checkpointer — IDENTICAL to v7 (don't touch)
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()
USER_ID = "u-001"


# ============================================================
# Memory tools — same as v7 (don't touch)
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

@tool
def current_date() -> str:
    """Return today's date in YYYY-MM-DD format. Use this when the user asks 'what's the date', 'what day is it', or 'today's date'."""
    from datetime import date
    return date.today().isoformat()


# ============================================================
# TODO 1: REAL file tools (replace write_doc + research_doc stubs)
# ============================================================
# These are real tools — they actually read/write your disk and can FAIL.
#
# read_file failure modes:
#   - FileNotFoundError: file doesn't exist
#   - PermissionError:   file exists but you can't read it
#   - IsADirectoryError: path is a directory, not a file
#
# write_file failure modes:
#   - PermissionError:  no write access to the path
#   - IsADirectoryError: trying to write to a directory
#   - OSError:           disk full, path too long, etc.

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path.
    Use this to read existing notes, code, or any text file."""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. DESTRUCTIVE — overwrites existing files.
    Requires human approval before execution."""
    Path(path).write_text(content)
    return f"wrote {len(content)} chars to {path}"


# ============================================================
# TODO 2: REAL web_search (DuckDuckGo — can fail on network)
# ============================================================
# pip install duckduckgo-search first.
# If this import fails, run: pip install duckduckgo-search

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return the top results.
    Use this when you need up-to-date information about a topic."""
    from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return f"No results found for: {query}"
    formatted = []
    for r in results:
        formatted.append(f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}")
    return "\n\n---\n\n".join(formatted)


# ============================================================
# TODO 3: ToolFailureMiddleware (custom, wrap_tool_call hook)
# ============================================================
# This is the v8 lesson: real tools fail. The agent loop must not crash.
#
# Pattern:
#   class ToolFailureMiddleware(AgentMiddleware):
#       def wrap_tool_call(self, request, handler):
#           try:
#               result = handler(request)
#               return result
#           except Exception as e:
#               # Return a ToolMessage that the LLM can read
#               return ToolMessage(
#                   content=f"Tool {request.tool_call['name']} failed: {type(e).__name__}: {e}. "
#                           f"Try a different approach or tool.",
#                   tool_call_id=request.tool_call["id"],
#               )
#
# Why this matters:
#   - Without this middleware, an exception in a tool call CRASHES the agent loop.
#   - With it, the LLM sees a clean error message and can retry or pivot.
#
# Question: what would happen if you DIDN'T have this middleware and
# `read_file("/nonexistent")` raised FileNotFoundError?

class ToolFailureMiddleware(AgentMiddleware):
    """Catch exceptions in tool calls, return a graceful error message."""

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(
                content=(
                    f"Tool {request.tool_call['name']} failed with "
                    f"{type(e).__name__}: {e}. "
                    f"Try a different approach or tool."
                ),
                tool_call_id=request.tool_call["id"],
            )


# ============================================================
# TODO 4: ToolTimingMiddleware (custom, wrap_tool_call hook)
# ============================================================
class ToolTimingMiddleware(AgentMiddleware):
    """Log how long each tool call takes."""

    def wrap_tool_call(self, request, handler):
        start = time.perf_counter()
        try:
            result = handler(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"\n[timing] tool={request.tool_call['name']} FAILED after {elapsed_ms:.1f}ms")
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n[timing] tool={request.tool_call['name']} -> {elapsed_ms:.1f}ms")
        return result


# ============================================================
# Middleware config — same as v7 (don't touch)
# ============================================================
SUMMARY_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)
TRIGGER = ("tokens", 300)
KEEP = ("messages", 5)
INTERRUPT_ON = {"write_file": True, "web_search": False, "read_file": False}

PII_CONFIG = {
    "pii_type": "email",
    "strategy": "redact",
    "apply_to_input": True,
}

# Order matters:
#   PII first (input transform)
#   Summarization (squishes post-PII messages)
#   HITL (gates dangerous tools)
#   ToolFailureMiddleware before ToolTiming so timing measures the actual tool work,
#   not the failure handler overhead. (Either order works; this is a readability choice.)
MIDDLEWARE_ORDER = [
    PIIMiddleware(**PII_CONFIG),
    SummarizationMiddleware(model=SUMMARY_MODEL, trigger=TRIGGER, keep=KEEP),
    HumanInTheLoopMiddleware(interrupt_on=INTERRUPT_ON),
    ToolFailureMiddleware(),
    ToolTimingMiddleware(),
]


# ============================================================
# Agent construction — don't touch
# ============================================================
agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    tools=[remember_about_user, recall_about_user, current_date, read_file, write_file, web_search],
    checkpointer=checkpointer,
    store=store,
    middleware=MIDDLEWARE_ORDER,
    system_prompt=(
        "You are a concise research assistant with file system + web access. "
        "Use read_file to read text files. "
        "Use write_file to save content (requires human approval). "
        "Use web_search for up-to-date info from the web. "
        "Use remember_about_user to save personal facts. "
        "Use recall_about_user to retrieve saved facts. "
        "Use current_date when the user asks about today's date."
    ),
)


# ============================================================
# TODO 5: run() — five single-tool phases
# ============================================================
# Each phase exercises one real tool (or one failure mode). No chaining.
#
# Phase 1: write a real file (HITL will pause, you approve)
# Phase 2: read it back (no HITL)
# Phase 3: try to read a nonexistent file (ToolFailureMiddleware should catch it)
# Phase 4: web_search for something real (works if network is up)
# Phase 5: web_search that may or may not work (DuckDuckGo is rate-limited)

def run():
    from langgraph.types import Command

    # ---- Phase 1: write a real file ----
    print("=== Phase 1: write_file (HITL pause expected) ===")
    cfg = {"configurable": {"thread_id": "v8-demo"}}
    test_path = "/tmp/v8_notes.md"
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": f"Use write_file to save exactly: 'hello from v8' to {test_path}"}]},
        config=cfg,
    )
    interrupts = r1.get("__interrupt__", [])
    if not interrupts:
        print("FAIL: no interrupt raised on write_file.")
        return

    payload = interrupts[0].value
    for req in payload.get("action_requests", []):
        print(f"  Agent wants: {req['name']}({req['args']})")

    decision = input("\n[approve | reject]: ").strip().lower()
    if decision == "approve":
        resume = Command(resume={"decisions": [{"type": "approve"}]})
    else:
        resume = Command(resume={"decisions": [{"type": "reject", "message": "Action rejected."}]})

    final1 = agent.invoke(resume, config=cfg)
    print("\nFinal:", final1["messages"][-1].content)

    # Verify the file actually exists on disk
    if Path(test_path).exists():
        actual = Path(test_path).read_text()
        print(f"\n[disk check] {test_path} contains: {actual!r}")
    else:
        print(f"\n[disk check] {test_path} does NOT exist (write was rejected or failed)")

    # ---- Phase 2: read it back ----
    print("\n=== Phase 2: read_file (no HITL) ===")
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": f"Use read_file to read {test_path}"}]},
        config=cfg,
    )
    print("AGENT:", r2["messages"][-1].content)

    # ---- Phase 3: read a nonexistent file (failure mode) ----
    print("\n=== Phase 3: read_file on a missing path (ToolFailureMiddleware should catch it) ===")
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "Use read_file to read /tmp/this_does_not_exist_xyz_123.md"}]},
        config=cfg,
    )
    print("AGENT:", r3["messages"][-1].content)
    print("(Expected: agent sees a 'FileNotFoundError' message and recovers)")

    # ---- Phase 4: web_search (real network call) ----
    print("\n=== Phase 4: web_search (real DuckDuckGo) ===")
    r4 = agent.invoke(
        {"messages": [{"role": "user", "content": "Use web_search to find 1 fact about Python's history. Query: 'python programming language history'"}]},
        config=cfg,
    )
    print("AGENT:", r4["messages"][-1].content)

    # ---- Phase 5: web_search with garbage query (failure mode) ----
    print("\n=== Phase 5: web_search with garbage query (likely returns 0 results, not a crash) ===")
    r5 = agent.invoke(
        {"messages": [{"role": "user", "content": "Use web_search with query 'asdkjfhasdkjfhsadkjf random gibberish' and max_results=1"}]},
        config=cfg,
    )
    print("AGENT:", r5["messages"][-1].content)


if __name__ == "__main__":
    run()
