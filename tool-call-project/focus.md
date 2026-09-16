# My LangGraph Mastery Plan — Plug & Play Research Agent

> Goal: Learn LangGraph deeply by building, end with a production-quality research agent I can plug any tool into.

## Top Resources (In Order)

1. **DeepLearning.AI "AI Agents in LangGraph"** — Free, ~1h 42m, taught by Harrison Chase (LangChain CEO). Canonical intro.
   → `https://www.deeplearning.ai/courses/ai-agents-in-langgraph`

2. **Cloned repo: `langchain-ai/agents-from-scratch`**
   → `~/workspace/ai-agents-building/repos/langchain-agents-from-scratch/`

3. **Cloned repo: `langchain-ai/deep-agents-from-scratch`**
   → `~/workspace/ai-agents-building/repos/deep-agents-from-scratch/`

4. **Official LangGraph docs** — reference only, lookup as needed
   → `docs.langchain.com/oss/python/langgraph/overview`

5. **LangGraph Studio** — run `langgraph dev` daily for visual debugging

**Skip / ignore:**
- ❌ `microsoft/ai-agents-for-beginners` (uses MAF / Azure)
- ❌ `microsoft/Building-AI-Agents-From-Zero-To-Production` (uses MAF / Azure)

Keep those two cloned for reference only. Do NOT follow them.

---

## Notebook Reading Order

### `langchain-agents-from-scratch/`
1. `notebooks/langgraph_101.ipynb` — first read, run in Studio
2. `notebooks/agent.ipynb` — your baseline production pattern
3. `notebooks/hitl.ipynb` + `src/email_assistant/email_assistant_hitl.py` — Week 2
4. `notebooks/memory.ipynb` + `src/email_assistant/email_assistant_hitl_memory.py` — Week 3
5. `notebooks/evaluation.ipynb` — Week 5+ (nice-to-have)

### `deep-agents-from-scratch/`
- Read top to bottom
- Focus on the planning + sub-agent composition pattern

---

## 4-Week Build Path

### Week 1 — LangGraph Foundations + Persistence
**Build:** Convert `my-agent.py` into a real `StateGraph` with `ToolNode` and `tools_condition`. Add `MemorySaver` + `thread_id`.

**Key concepts:**
- `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`
- `Annotated` state reducers
- `ToolNode`, `tools_condition`
- `MemorySaver` checkpointer
- `thread_id` for session memory

**Use:** DLAI course L1–L4 · `langgraph_101.ipynb` · `agent.ipynb`

---

### Week 2 — Human-in-the-Loop + Streaming
**Build:** Add `interrupt_before=["tools"]` to the research agent. CLI approval prompt that resumes via `Command(resume=...)`. Stream tokens with `stream_mode="messages"`.

**Key concepts:**
- `interrupt` / `Command`
- `stream_mode ∈ {values, updates, events, messages}`
- `get_state_history()`
- Approval patterns before destructive/expensive actions

**Use:** DLAI L5–L7 · `hitl.ipynb` · `src/email_assistant/email_assistant_hitl.py`

---

### Week 3 — Long-Term Memory + Context Management
**Build:** Add `InMemoryStore` (then `PostgresStore` later). Store user preferences across threads. Add a summarize-and-trim node so research context doesn't explode.

**Pattern:** Every 5 search results → summarize → store digest → keep top-K.

**Key concepts:**
- `BaseStore`
- `store.put` / `store.search`
- Cross-thread memory namespaces
- Context window management

**Use:** `memory.ipynb` · Andrew Ng's "Long-Term Agentic Memory with LangGraph" (~1h, free)

---

### Week 4 — The Research Agent (The Goal)
**Build:** Research agent with 3 sub-graphs:
- **`searcher`** — DuckDuckGo (already have it) + Tavily for LLM-optimized snippets
- **`reflector`** — critiques results, generates gap-fill queries, loops back to searcher
- **`synthesizer`** — writes final cited report

Compose them via a parent graph.

**Key concepts:**
- Sub-graphs (state passed via channels)
- `Send` / fan-out for parallel searches
- `ToolNode(handle_tool_errors=True)`
- Multi-agent supervisor pattern

**Use:** `deep-agents-from-scratch/notebooks/` · planning + sub-agent patterns

---

## Tool Stack for the Research Agent

| Tool | Why | Cost |
|---|---|---|
| **DuckDuckGo** | Already have it, free | Free |
| **Tavily** | LLM-optimized snippets | Free tier w/ API key |
| **Exa** | Semantic search (optional) | Free tier |
| ~~SerpAPI~~ | Skip — overkill | — |

---

## Build-vs-Buy Rules

- ❌ **DON'T** lean on `create_react_agent` from `langgraph.prebuilt`
- ✅ **DO** use `ToolNode`, `tools_condition`, `MemorySaver` as primitives
- ✅ **DO** run `langgraph dev` (Studio) every day — fastest feedback loop

---

## Rules of the Road

1. **Build, don't just read.** Every concept gets code in the same week.
2. **One concept per build.** Don't try to add HITL + memory + streaming all at once.
3. **Use the cloned repos as references, not courses.** Skim, copy patterns, move on.
4. **Keep my-agent.py simple.** Use it as the "tool registry" — register tools there once, import them into every graph.
5. **Each week ends with a working runnable artifact**, even if minimal.

---

## Progress Tracker

- [ ] Week 1 — StateGraph + MemorySaver + thread_id
- [ ] Week 2 — HITL interrupt + Command + streaming
- [ ] Week 3 — InMemoryStore + summarize-and-trim
- [ ] Week 4 — Searcher + Reflector + Synthesizer (THE research agent)
- [ ] Week 5+ — Evaluation, PostgresStore, deployment

---

## File Layout I'll Build Into

```
tool-call-project/
├── my-agent.py          # Tool registry (calculate, get_time, get_weather, search_web)
├── react-agent.py       # Manual ReAct text parsing (DONE — reference only)
├── focus.md             # ← this file
├── week1_graph.py       # StateGraph version of my-agent.py
├── week2_hitl.py        # + interrupt + streaming
├── week3_memory.py      # + InMemoryStore
└── research_agent.py    # Week 4 finale — the real research agent
```
