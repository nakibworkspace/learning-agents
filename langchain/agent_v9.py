"""
Stage S8 — Synthesis subgraph (cumulative on v8).

Run after completion:
    cd /Users/nakibahmed/workspace/agents-dive
    python agent_v9.py

What this adds over agent_v8.py:
    - The PLANNER pattern: an outer agent that dispatches to sub-agents.
    - Two sub-agents, each a separate create_agent(...) invocation:
        * research_subagent  — has read_file + web_search + ToolFailureMiddleware.
                              Returns a JSON-serialized report.
        * writer_subagent    — has write_file (HITL-gated). Returns a confirmation.
    - The main (planner) agent has only two tools: dispatch_research, dispatch_write.
      Each tool internally calls the appropriate sub-agent.
    - This sidesteps the v6-era "llama3.2 won't chain tools" problem architecturally:
      we DECOMPOSE at the planner layer, not the model layer.

What this REPLACES from v8:
    - The main agent no longer has read_file/write_file/web_search directly.
      It only has dispatch_research and dispatch_write.
"""

# ============================================================
# IMPORTS
# ============================================================
import json
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
# Store + checkpointer — IDENTICAL to v8 (don't touch)
# ============================================================
store = InMemoryStore()
checkpointer = InMemorySaver()
USER_ID = "u-001"


# ============================================================
# Shared tools — visible to sub-agents (don't touch)
# ============================================================
@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. DESTRUCTIVE — overwrites existing files.
    Requires human approval before execution."""
    Path(path).write_text(content)
    return f"wrote {len(content)} chars to {path}"


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return the top results."""
    from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return f"No results found for: {query}"
    formatted = []
    for r in results:
        formatted.append(f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}")
    return "\n\n---\n\n".join(formatted)


# ============================================================
# Memory tools — IDENTICAL to v8 (don't touch)
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
# ToolFailureMiddleware — IDENTICAL to v8 (don't touch)
# ============================================================
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
# TODO 1: Build the two SUB-AGENTS
# ============================================================
# Each sub-agent is itself a complete create_agent(...) invocation. They have their
# own system_prompt, their own tools, their own middleware.
#
# IMPORTANT: sub-agents do NOT get a checkpointer or store here. They are stateless
# invocations from the planner's perspective. The planner owns the thread.
#
# research_subagent: read_file + web_search + ToolFailureMiddleware
# writer_subagent:   write_file only + HumanInTheLoopMiddleware

SUBAGENT_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)


# TODO 1a: build research_subagent
research_subagent = create_agent(
    model=SUBAGENT_MODEL,
    tools=[read_file, web_search],
    middleware=[ToolFailureMiddleware()],
    system_prompt=(
        # TODO 1b: write a system prompt that tells this agent it is a RESEARCH specialist
        # It should use web_search for current info, read_file for files on disk,
        # and ALWAYS return its findings as a structured report (so the planner can parse it).
        "You are a RESEARCH specialist. you should be using web_search for current infor and read_file tool for reading files on disk. Always response as a structured report"
    ),
)


# TODO 1c: build writer_subagent
writer_subagent = create_agent(
    model=SUBAGENT_MODEL,
    tools=[write_file],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"write_file": True}),
    ],
    system_prompt=(
        # TODO 1d: write a system prompt that tells this agent it is a WRITER specialist
        # It takes a path and content, calls write_file, returns a confirmation.
        "You are a WRITER specialist. When the user provides a path and content, "
        "call the write_file tool EXACTLY ONCE with those two arguments. "
        "Do not invent new content — write what the user gave you. "
        "Return a short confirmation message."
    ),
)


# ============================================================
# TODO 2: Build the dispatch tools (the PLANNER's tools)
# ============================================================
# The planner (main agent) calls dispatch_research or dispatch_write. Each dispatch
# tool internally invokes the corresponding sub-agent.
#
# Key idea: each sub-agent.invoke() takes a fresh `messages` payload. We do NOT pass
# the planner's full conversation to the sub-agent — that would be too much context.
# We pass a single, focused task message.

@tool
def dispatch_research(task: str) -> str:
    """Dispatch a research task to the research sub-agent.
    Use this when you need facts from the web or files on disk.
    The sub-agent will return a structured report."""
    result = research_subagent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    return result["messages"][-1].content


@tool
def dispatch_write(path: str, content: str) -> str:
    """Dispatch a write task to the writer sub-agent (HITL-gated).
    Use this when you need to save content to a file.
    The writer sub-agent has HITL enabled, so it will pause for approval."""
    task = f"Use write_file to save this content to {path}: {content}"
    result = writer_subagent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    # The writer sub-agent's HITL will return an interrupt. Surface and handle it.
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        from langgraph.types import Command
        payload = interrupts[0].value
        for req in payload.get("action_requests", []):
            print(f"  [nested HITL] Writer wants: {req['name']}({req['args']})")
        decision = input("[approve | reject]: ").strip().lower()
        if decision == "approve":
            resume = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            resume = Command(resume={"decisions": [{"type": "reject", "message": "Rejected by user."}]})
        final = writer_subagent.invoke(resume)
        return final["messages"][-1].content
    return result["messages"][-1].content


# ============================================================
# TODO 3: Build the main PLANNER agent
# ============================================================
# The planner has a tight set of tools: dispatch_research + dispatch_write +
# the memory tools (remember/recall/current_date).
#
# It does NOT have direct file or web access — those are sub-agent concerns.

SUMMARY_MODEL = ChatOllama(model="llama3.2:latest", temperature=0.0)
TRIGGER = ("tokens", 300)
KEEP = ("messages", 5)
INTERRUPT_ON = {"dispatch_write": True, "dispatch_research": False}

PII_CONFIG = {
    "pii_type": "email",
    "strategy": "redact",
    "apply_to_input": True,
}

# TODO 3a: write the planner's system prompt
PLANNER_SYSTEM_PROMPT = (
    "You are a PLANNER and COORDINATOR. You do not do research or write files yourself. "
    "Use dispatch_research for factual questions — it returns a structured report. "
    "Use dispatch_write when the user wants content saved to a file (it requires human approval). "
    "Use remember_about_user / recall_about_user for personal facts. "
    "Use current_date when the user asks about today's date. "
    "Be concise: summarize the sub-agent's findings for the user."
)


# TODO 3b: build the planner agent
planner_agent = create_agent(
    model=ChatOllama(model="llama3.2:latest", temperature=0.0),
    tools=[dispatch_research, dispatch_write, remember_about_user, recall_about_user, current_date],
    checkpointer=checkpointer,
    store=store,
    middleware=[
        PIIMiddleware(**PII_CONFIG),
        SummarizationMiddleware(model=SUMMARY_MODEL, trigger=TRIGGER, keep=KEEP),
        HumanInTheLoopMiddleware(interrupt_on=INTERRUPT_ON),
    ],
    system_prompt=PLANNER_SYSTEM_PROMPT,
)


# ============================================================
# TODO 4: write run() — three-phase synthesis demo
# ============================================================
# Phase 1: a query that needs RESEARCH only → dispatch_research
# Phase 2: a query that needs WRITE only → dispatch_write (with HITL)
# Phase 3: a query that needs BOTH (research then write) → two planner invokes (app-level)
#
# The v9 win: the planner can express multi-step tasks at the message level.
# The architecture handles each piece through a sub-agent that has only one job.

def run():
    from langgraph.types import Command

    cfg = {"configurable": {"thread_id": "v9-demo"}}

    # ---- Phase 1: research only ----
    print("=== Phase 1: research only (dispatch_research) ===")
    r1 = planner_agent.invoke(
        {"messages": [{"role": "user", "content": "Find out when Python was first released. Use dispatch_research."}]},
        config=cfg,
    )
    print("PLANNER:", r1["messages"][-1].content)

    # ---- Phase 2: write only ----
    print("\n=== Phase 2: write only (dispatch_write — HITL pause expected) ===")
    r2 = planner_agent.invoke(
        {"messages": [{"role": "user", "content": "Use dispatch_write to save 'plan complete' to /tmp/v9_plan.md"}]},
        config=cfg,
    )
    interrupts = r2.get("__interrupt__", [])
    if not interrupts:
        print("FAIL: no interrupt raised on dispatch_write.")
        return

    payload = interrupts[0].value
    for req in payload.get("action_requests", []):
        print(f"  Planner wants: {req['name']}({req['args']})")

    decision = input("\n[approve | reject]: ").strip().lower()
    if decision == "approve":
        resume = Command(resume={"decisions": [{"type": "approve"}]})
    else:
        resume = Command(resume={"decisions": [{"type": "reject", "message": "Action rejected."}]})

    final2 = planner_agent.invoke(resume, config=cfg)
    print("\nPLANNER:", final2["messages"][-1].content)

    # Verify the file
    if Path("/tmp/v9_plan.md").exists():
        print(f"\n[disk check] /tmp/v9_plan.md contains: {Path('/tmp/v9_plan.md').read_text()!r}")
    else:
        print("\n[disk check] /tmp/v9_plan.md does NOT exist")

    # ---- Phase 3: research THEN write (app-level chaining at planner layer) ----
    print("\n=== Phase 3: research then write (two planner invokes) ===")
    r3a = planner_agent.invoke(
        {"messages": [{"role": "user", "content": "Use dispatch_research to find out what Guido van Rossum's favorite hobby is. Then come back and tell me."}]},
        config=cfg,
    )
    research_output = r3a["messages"][-1].content
    print("PLANNER (research):", research_output)

    print("\n[now asking planner to write the answer to a file]")
    r3b = planner_agent.invoke(
        {"messages": [{"role": "user", "content": f"Use dispatch_write to save this to /tmp/v9_hobby.md: {research_output}"}]},
        config=cfg,
    )
    interrupts3 = r3b.get("__interrupt__", [])
    if interrupts3:
        decision3 = input("\n[approve | reject]: ").strip().lower()
        if decision3 == "approve":
            resume3 = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            resume3 = Command(resume={"decisions": [{"type": "reject", "message": "Rejected."}]})
        final3 = planner_agent.invoke(resume3, config=cfg)
        print("PLANNER (write):", final3["messages"][-1].content)
    else:
        print("PLANNER (write):", r3b["messages"][-1].content)


if __name__ == "__main__":
    run()
