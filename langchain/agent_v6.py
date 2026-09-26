"""
Stage S5 — Custom middleware (cumulative on v5).

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v6.py

What this adds over agent_v5.py:
    - ToolTimingMiddleware (custom, wrap_tool_call hook)
    - PIIMiddleware        (built-in, before_model hook)
    - Two different hook points in the agent pipeline.
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
# Store + checkpointer — IDENTICAL to v5 (don't touch)
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()
USER_ID = "u-001"


# ============================================================
# Tools — same as v5 (don't touch)
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
# TODO 1: ToolTimingMiddleware (custom, wrap_tool_call hook)
# ============================================================
# Reference: agents/middleware/3_custom_middleware.py
#
# Pattern:
#   class ToolTimingMiddleware(AgentMiddleware):
#       def wrap_tool_call(self, request, handler):
#           # 1. start a timer
#           # 2. result = handler(request)
#           # 3. stop the timer, print "[timing] tool={name} -> {ms}ms"
#           # 4. return result
#
# Two questions to answer in TODOs:
#   - What is `handler`? (a callable you must invoke to actually run the tool)
#   - What is `request.tool_call`? (a dict with "name" and "args")

class ToolTimingMiddleware(AgentMiddleware):
    """TODO 1: implement wrap_tool_call to log per-tool timing."""

    def wrap_tool_call(self, request, handler):
        start = time.perf_counter()
        result = handler(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[timing] tool={request.tool_call['name']} args={request.tool_call['args']} -> {elapsed_ms:.1f}ms")
        return result


# ============================================================
# TODO 2: PIIMiddleware (built-in, before_model hook)
# ============================================================
# Reference: agents/middleware/4_pii_redaction.py
#
# Built-in pii_types: email, credit_card, ip, mac_address, url
# Built-in strategies: redact, mask, hash, block

PII_CONFIG = {
    "pii_type": "email",
    "strategy": "mask",   # show a bit of the email address rather than just redact it
    "apply_to_input": True,
}


# ============================================================
# TODO 3: middleware order
# ============================================================
# Pipeline reminder:
#   user input → before_model → LLM → wrap_tool_call → tool → ... loop
#
#   - PII should run BEFORE the LLM sees anything (apply at front of input flow)
#   - Summarization should see post-PII messages so the summary doesn't preserve PII
#   - ToolTiming wraps every tool call — position doesn't matter for correctness
#   - HITL gates dangerous tools — must run before the tool actually executes

SUMMARY_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)
TRIGGER = ("tokens", 300)
KEEP = ("messages", 5)
INTERRUPT_ON = {"write_doc": True, "research_doc": False}


# TODO 3: write MIDDLEWARE_ORDER as a list. Try (b) first if you're not sure.
MIDDLEWARE_ORDER = [
    # fill these in as middleware instances in the right order
    PIIMiddleware(**PII_CONFIG),
    SummarizationMiddleware(
        model = SUMMARY_MODEL,
        trigger = TRIGGER,
        keep = KEEP,
    ),
    HumanInTheLoopMiddleware(interrupt_on= INTERRUPT_ON),
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
# TODO 4: write run() — three phases
# ============================================================
# Three-phase demo:
#   Phase 1 — query with PII (email) → should be redacted before LLM
#   Phase 2 — query triggering write_doc → HITL pauses, handle approve
#   Phase 3 — read-only query → no interrupt, just timing logs

def run():
    from langgraph.types import Command

    if not MIDDLEWARE_ORDER:
        raise RuntimeError("Finish TODO 3: fill in MIDDLEWARE_ORDER.")

    cfg = {"configurable": {"thread_id": "v6-demo"}}

    # ---- Phase 1: PII redaction in action ----
    print("=== Phase 1: query containing PII (should be redacted before LLM) ===")
    pii_query = "My email is alice@example.com — please research what email protocols she might use."
    r1 = agent.invoke({"messages": [{"role": "user", "content": pii_query}]}, config=cfg)
    print("AGENT:", r1["messages"][-1].content)

    # ---- Phase 2: HITL + ToolTiming together (app-level chained) ----
    print("\n=== Phase 2a: research (read-only, no interrupt expected) ===")
    r2a = agent.invoke(
        {"messages": [{"role": "user", "content": "Research LangChain middleware."}]},
        config=cfg,
    )
    research_output = r2a["messages"][-1].content
    print("RESEARCH:", research_output)

    print("\n=== Phase 2b: write_doc (should HITL) ===")
    r2b = agent.invoke(
        {"messages": [{"role": "user", "content": f"Write this to /tmp/notes.md as a 2-line summary: {research_output}"}]},
        config=cfg,
    )
    interrupts = r2b.get("__interrupt__", [])
    if not interrupts:
        print("FAIL: no interrupt raised on write_doc.")
        return

    print(f"\n>>> Interrupt fired. {len(interrupts)} interrupt(s) pending.")
    payload = interrupts[0].value
    for req in payload.get("action_requests", []):
        print(f"  Agent wants: {req['name']}({req['args']})")

    # TODO 4a: handle the interrupt (copy from v5 — approve is fine for v6)
    decision = input("\n[approve | edit | reject]: ").strip().lower()
    from langgraph.types import Command
    if decision == "approve":
        resume = Command(resume={"decisions": [{"type": "approve"}]})
    elif decision == "edit":
        new_path = input("New path: ")
        original = payload["action_requests"][0]
        resume = Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": original["name"],
                    "args": {**original["args"], "path": new_path},
                },
            }]
        })
    else:
        resume = Command(resume= {
            "decisions": [{
                "type": "reject",
                "message": "Action rejected by user. Do NOT claim it succeeded. Tell the user the action was blocked.",
            }]
        })

    # TODO 4b: resume, print final, observe timing line for write_doc
    final = agent.invoke(resume, config=cfg)
    print("\nFinal:", final["messages"][-1].content)

    # ---- Phase 3: clean read-only pass ----
    print("\n=== Phase 3: read-only query (no interrupt, timing on recall_about_user) ===")
    r3 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's today's date? And what is my favorite animal?"}]},
        config=cfg,
    )
    if r3.get("__interrupt__"):
        print("FAIL: false interrupt on read-only tools.")
    else:
        print("PASS: no interrupt.")
        print("AGENT:", r3["messages"][-1].content)


if __name__ == "__main__":
    run()
