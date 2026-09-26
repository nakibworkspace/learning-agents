"""
Stage S6 — Token streaming (cumulative on v6).

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v7.py

What this adds over agent_v6.py:
    - Replace agent.invoke(...) with agent.stream(..., stream_mode="messages")
    - Yield (chunk, metadata) tuples as the LLM generates tokens
    - All middlewares (PII, Summarization, HITL, ToolTiming) are still active
    - The interesting question: where do timing prints land relative to streamed tokens?
"""

# ============================================================
# IMPORTS — don't touch
# ============================================================
import time
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# Store + checkpointer — IDENTICAL to v6 (don't touch)
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()
USER_ID = "u-001"


# ============================================================
# Tools — same as v6 (don't touch)
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

@tool
def research_doc(query: str) -> str:
    """Look up information from the research corpus. READ-ONLY.
    Use this whenever you need facts about a topic — does NOT require human approval."""
    return f"[Research stub results for: {query}]"

@tool
def write_doc(path: str, content: str) -> str:
    """Write content to a document at the given path. DESTRUCTIVE — overwrites any existing file.
    Requires human approval before execution."""
    return f"[Stubbed write: {len(content)} chars -> {path}]"


# ============================================================
# ToolTimingMiddleware — same as v6 (don't touch)
# ============================================================
class ToolTimingMiddleware(AgentMiddleware):
    """Log how long each tool call takes."""

    def wrap_tool_call(self, request, handler):
        start = time.perf_counter()
        result = handler(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n[timing] tool={request.tool_call['name']} args={request.tool_call['args']} -> {elapsed_ms:.1f}ms")
        return result


# ============================================================
# Middleware config — same as v6 (don't touch)
# ============================================================
SUMMARY_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)
TRIGGER = ("tokens", 300)
KEEP = ("messages", 5)
INTERRUPT_ON = {"write_doc": True, "research_doc": False}

PII_CONFIG = {
    "pii_type": "email",
    "strategy": "redact",
    "apply_to_input": True,
}

MIDDLEWARE_ORDER = [
    PIIMiddleware(**PII_CONFIG),
    SummarizationMiddleware(model=SUMMARY_MODEL, trigger=TRIGGER, keep=KEEP),
    HumanInTheLoopMiddleware(interrupt_on=INTERRUPT_ON),
    ToolTimingMiddleware(),
]


# ============================================================
# Agent construction — don't touch
# ============================================================
agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    tools=[remember_about_user, recall_about_user, current_date, research_doc, write_doc],
    checkpointer=checkpointer,
    store=store,
    middleware=MIDDLEWARE_ORDER,
    system_prompt=(
        "You are a concise research assistant with access to a document store. "
        "Use research_doc for read-only lookups (no approval needed). "
        "Use write_doc to save content (will pause for human approval). "
        "Use remember_about_user to save personal facts. "
        "Use recall_about_user to retrieve saved facts. "
        "Use current_date when the user asks about today's date."
    ),
)


# ============================================================
# TODO 1: stream_tokens() helper
# ============================================================
# Reference: agents/streaming/1_token_stream.py
#
# Goal: turn agent.stream(..., stream_mode="messages") output into printed text.
#
# Pattern:
#   for chunk, meta in agent.stream(input, config=cfg, stream_mode="messages"):
#       text = getattr(chunk, "content", "") or getattr(chunk, "content_tokens", "")
#       if text:
#           print(text, end="", flush=True)
#
# Questions:
#   - Why `end=""`? (because streamed tokens must NOT have newlines between them)
#   - Why `flush=True`? (otherwise Python buffers and you don't see streaming)
#   - Why `getattr(chunk, "content", "") or getattr(chunk, "content_tokens", "")`?
#     (different chunk types have different attributes; the `or` picks whichever has text)

def stream_tokens(input_messages, cfg):
    """
    Stream an agent invocation, printing tokens as they arrive.

    Yields nothing — this is a side-effecting function that prints.
    Returns the final state once streaming completes.
    """
    # TODO 1a: call agent.stream(input_messages, config=cfg, stream_mode="messages")
    stream = agent.stream(input_messages, config=cfg, stream_mode="messages")
    # TODO 1b: for each (chunk, meta): extract text and print with end="", flush=True
    for chunk, meta in stream:
        text = getattr(chunk, "content", "") or getattr(chunk, "content_tokens", "")
        if text:
            print(text, end="", flush=True)
    # TODO 1c: print a final newline after the loop
    print()
    print()


# ============================================================
# TODO 2: run() — three streaming phases
# ============================================================
def run():
    if not callable(stream_tokens) or stream_tokens.__code__.co_consts is None:
        # Will fail to None check; the goal is to surface TODO 1
        raise RuntimeError("Finish TODO 1: implement stream_tokens.")

    # ---- Phase 1: stream a query that hits a tool (research_doc) ----
    print("=== Phase 1: streaming a tool-using query ===")
    cfg1 = {"configurable": {"thread_id": "v7-demo-1"}}
    print("USER: What's the research on LangChain middleware?\n")
    print("AGENT (streamed):", end=" ")
    final1 = stream_tokens(
        {"messages": [{"role": "user", "content": "Use research_doc to look up LangChain middleware. Then summarize in 2 sentences."}]},
        cfg1,
    )

    # ---- Phase 2: stream a HITL query (write_doc) ----
    print("\n=== Phase 2: streaming a write_doc query (HITL pause expected) ===")
    cfg2 = {"configurable": {"thread_id": "v7-demo-2"}}
    print("USER: Look up email protocols, then write 1 line to /tmp/notes.md.\n")
    print("AGENT (streamed):", end=" ")
    final2 = stream_tokens(
        {"messages": [{"role": "user", "content": "Use research_doc to look up email protocols. Then call write_doc with path='/tmp/notes.md' and a 1-line summary."}]},
        cfg2,
    )

    # ---- Phase 3: stream a HITL + interrupt resume ----
    print("\n=== Phase 2b: handle the interrupt, then resume streaming ===")
    interrupts = final2.get("__interrupt__", []) if isinstance(final2, dict) else []
    if interrupts:
        payload = interrupts[0].value
        for req in payload.get("action_requests", []):
            print(f"  Agent wants: {req['name']}({req['args']})")
        decision = input("\n[approve | reject]: ").strip().lower()
        from langgraph.types import Command
        if decision == "approve":
            resume = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            resume = Command(resume={"decisions": [{"type": "reject", "message": "Action rejected."}]})

        print("\nRESUMED STREAM:", end=" ")
        stream_tokens(resume, cfg2)

    # ---- Phase 3: clean read-only stream ----
    print("\n=== Phase 3: streaming a read-only query ===")
    cfg3 = {"configurable": {"thread_id": "v7-demo-3"}}
    print("USER: What's today's date?\n")
    print("AGENT (streamed):", end=" ")
    stream_tokens(
        {"messages": [{"role": "user", "content": "What's today's date?"}]},
        cfg3,
    )
    print()


if __name__ == "__main__":
    run()
