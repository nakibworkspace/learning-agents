"""
Stage S4 — HumanInTheLoopMiddleware added on top of v4.

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v5.py

What this adds over agent_v4.py:
    - HumanInTheLoopMiddleware wired in
    - 'write_file'-style tools (anything destructive) pause for human approval
    - 'read_file'-style tools (anything read-only) run without approval
    - We use stub research/summary tools so the demo doesn't need real APIs

Order of operations:
    TODO 1 — Decide which tools are dangerous (interrupt_on=True) vs safe
    TODO 2 — Wire HumanInTheLoopMiddleware with your interrupt_on map
    TODO 3 — Write run(): a query that triggers the dangerous tool, handle the interrupt
"""

# ============================================================
# IMPORTS — don't touch
# ============================================================
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# Store + checkpointer — IDENTICAL to v4 (don't touch)
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()
USER_ID = "u-001"


# ============================================================
# Tools — same as v4 (remember + recall + current_date)
# Plus 2 new stub tools that simulate file/research operations.
# These two are what HITL will gate on.
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


# ---- New tools for v5: simulate "read" and "write" of a fake document store ----

@tool
def research_doc(query: str) -> str:
    """Look up information from the research corpus. READ-ONLY.
    Use this whenever you need facts about a topic — does NOT require human approval."""
    # Stub: would call a search API or query a vector store in production.
    return f"[Research stub results for: {query}]"

@tool
def write_doc(path: str, content: str) -> str:
    """Write content to a document at the given path. DESTRUCTIVE — overwrites any existing file.
    Requires human approval before execution."""
    # Stub: in production this would actually write to disk / DB / vector store.
    return f"[Stubbed write: {len(content)} chars -> {path}]"


# ============================================================
# TODO 1 + 2: build the agent with HITL + Summarization middleware
# ============================================================
# Decisions to make:
#
# (a) Which tools need human approval?
#     - write_doc      — definitely YES (destructive)
#     - research_doc   — NO (read-only)
#     - remember_about_user — your call: does saving a fact count as destructive?
#                            In most apps: NO (you can always delete later).
#     - recall_about_user  — NO (read-only)
#     - current_date      — NO (read-only)
#
# (b) Same trigger/keep from v4 (summarization is orthogonal).

SUMMARY_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)
TRIGGER = ("tokens", 300)
KEEP    = ("messages", 5)

# TODO 2: build the interrupt_on map.
# Format: {"tool_name": True (require approval) or False (don't)}
INTERRUPT_ON = {"write_doc": True, "research_doc": False}


agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    tools=[remember_about_user, recall_about_user, current_date, research_doc, write_doc],
    checkpointer=checkpointer,
    store=store,
    middleware=[
        SummarizationMiddleware(
            model=SUMMARY_MODEL,
            trigger=TRIGGER,
            keep=KEEP,
        ),
        HumanInTheLoopMiddleware(
            interrupt_on=INTERRUPT_ON,
        ),
    ],
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
# TODO 3: write run() — handle the interrupt correctly
# ============================================================
def run():
    """
    Two-phase demo:
      Phase 1 — drive an invoke that triggers the dangerous tool (write_doc).
                Handle the interrupt: approve / edit / reject.
      Phase 2 — drive an invoke that only uses safe tools (research_doc).
                No interrupt should fire — proves the interrupt_on map is selective.
    """
    if not INTERRUPT_ON:
        raise RuntimeError("Finish TODO 2: fill in INTERRUPT_ON.")

    cfg = {"configurable": {"thread_id": "v5-demo"}}

    # ---- Phase 1: a query that should trigger write_doc ----
    print("=== Phase 1: query that triggers write_doc (should HITL) ===")
    query = "Research LangChain middleware, then write a 2-line summary to /tmp/notes.md"
    r = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=cfg)
    interrupts = r.get("__interrupt__", [])

    if not interrupts:
        print("FAIL: no interrupt raised. Either your INTERRUPT_ON map is wrong, "
              "or llama3.2 didn't pick write_doc. Try a more direct query.")
        return

    # langchain 1.x: payload = {action_requests: [...], review_configs: [...]}
    print(f"\n>>> Interrupt fired. {len(interrupts)} interrupt(s) pending.")
    payload = interrupts[0].value
    for req in payload.get("action_requests", []):
        print(f"  Agent wants: {req['name']}({req['args']})")

    # TODO 3a: collect the user's decision.
    #   - input() → "approve" | "edit" | "reject"
    #   - For "edit": also ask for the new path, build edited_action dict
    #                (langchain 1.x requires full {name, args} restatement)
    #   - For "reject": always include a "message" telling the model to NOT
    #                   claim it succeeded (small models hallucinate success)
    #   - Build the Command(resume={"decisions": [...]}) and re-invoke.
    #
    # Pattern (verify against agents/middleware/1_human_in_the_loop.py):
    #   from langgraph.types import Command
    #   resume = Command(resume={"decisions": [...]})
    #   final = agent.invoke(resume, config=cfg)
    #
    # Don't forget: same cfg (same thread_id) is what makes the resume work.
    decision = input("\n[approve | edit | reject]: ").strip().lower()
    from langgraph.types import Command
    if decision == "approve":
        resume = Command(resume={"decisions": [{"type": "approve"}]})
    elif decision == "edit":
        # langchain 1.x: edit requires a full edited_action dict (name + all args).
        # The older positional-tuple form `{"type": "edit", "args": (new_path,)}` is stale.
        new_path = input("New path: ")
        # We know the original tool was write_file(path, content) — preserve content, swap path.
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
        # Reject should ALWAYS carry an explicit message — small models hallucinate success otherwise.
        resume = Command(resume={
            "decisions": [{
                "type": "reject",
                "message": "Action rejected by human. Do NOT claim it succeeded. Tell the user the action was blocked.",
            }]
        })

    # TODO 3b: print the final agent message.
    # print("\nFinal:", final["messages"][-1].content)
    final = agent.invoke(resume, config=cfg)
    print("\nFinal:", final["messages"][-1].content)


    # ---- Phase 2: a query that should NOT trigger any interrupt ----
    print("\n=== Phase 2: query that only uses safe tools (should NOT HITL) ===")
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's today's date? And what is my favorite animal?"}]},
        config=cfg,
    )
    interrupts2 = r2.get("__interrupt__", [])
    if interrupts2:
        print(f"FAIL: {len(interrupts2)} interrupt(s) raised on a read-only query. "
              "Your INTERRUPT_ON map is too aggressive.")
    else:
        print("PASS: no interrupt on read-only tools.")
        print("AGENT:", r2["messages"][-1].content)


if __name__ == "__main__":
    run()
