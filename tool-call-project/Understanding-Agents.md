# Understanding Agents — A Senior Engineer's Field Guide

> Written so that 6-months-from-now you can re-read this in 10 minutes and *remember*.
> Mental models first. Code as illustration. Real-world anchors second.

---

## Table of Contents

1. [What](#1-what-my-agentpy-is-usually-called) `my-agent.py` [is usually called](#1-what-my-agentpy-is-usually-called)
2. [Is ReAct an AI Agent pattern?](#2-is-react-an-ai-agent-pattern)
3. `my-agent.py` [vs](#3-my-agentpy-vs-react-agentpy--architecture-data-flow-configuration) `react-agent.py` [— Architecture, Data Flow, Configuration](#3-my-agentpy-vs-react-agentpy--architecture-data-flow-configuration)
4. [The Major Agent Patterns](#4-the-major-agent-patterns)
5. [How to Choose an Agent Pattern for a Real Situation](#5-how-to-choose-an-agent-pattern-for-a-real-situation)
6. [Building These Agents in LangGraph](#6-building-these-agents-in-langgraph)

---



## 1. What `my-agent.py` is usually called

It's called a **Tool-Calling Agent** (or **Function-Calling Agent**). Some people also say **Native Tool-Calling Agent** because it relies on the model's *native* ability to emit structured tool calls.

The shape is dead simple:

```
User → LLM (decides) → Tool (runs) → LLM (observes result) → ... → Final Answer
```

Three names you'll see in the wild for *exactly* this same pattern:


| Name                             | Where you'll see it                                       |
| -------------------------------- | --------------------------------------------------------- |
| **Tool-Calling Agent**           | OpenAI docs, Anthropic docs                               |
| **Function-Calling Agent**       | Older OpenAI docs, LangChain `AgentType.OPENAI_FUNCTIONS` |
| **ReAct Agent (OpenAI variant)** | LangChain `create_react_agent` — same idea, OpenAI-style  |


The pattern itself is sometimes lumped under **"ReAct"** as well, but in modern usage:

- **ReAct (text)** = `react-agent.py` (the original 2022 paper, plain-text reasoning)
- **Tool-Calling Agent (function-calling)** = `my-agent.py` (post-2023, structured outputs)

Most production code today is the tool-calling variant. The original ReAct paper is still cited because the *idea* — Reason + Act interleaved — is the same; only the wire format changed.

---



## 2. Is ReAct an AI Agent pattern?

**Yes. It's THE foundational agent pattern.** The Yao et al. 2022 paper *"ReAct: Synergizing Reasoning and Acting in Language Models"* is the paper that turned "LLM in a loop" into a named, studied pattern.

The thesis: instead of asking the model to *think and then* answer (chain-of-thought), you let it **think → act → observe → think → act → observe → …** in a loop. Each Action affects the world; each Observation grounds the next Thought.

```
ReAct = Reason + Act, interleaved, with a loop.
```

Both your files are ReAct agents. They differ only in **how the Reason and Act are encoded**:


| File             | Reason encoded as                       | Act encoded as                           |
| ---------------- | --------------------------------------- | ---------------------------------------- |
| `my-agent.py`    | hidden inside the model's tool-decision | structured `tool_calls` field            |
| `react-agent.py` | visible `Thought:` lines in text        | parsed `Action:` / `Action Input:` lines |


The *loop* is identical. The *wire format* differs.

**Rule of thumb to remember:** Every modern agent pattern you'll ever build — ReAct, Plan-and-Execute, Multi-Agent, Reflexion, Voyager — is a ReAct loop with extra structure layered on top. Master ReAct first, everything else is a remix.

---



## 3. `my-agent.py` vs `react-agent.py` — Architecture, Data Flow, Configuration

This is the deep-dive. We'll walk through:

1. What the two files *actually do*, turn by turn
2. What the LLM *sees* in each case (the wire format)
3. What the LLM *emits* in each case
4. Why the `tools=[...]` argument exists in one and not the other
5. The full comparison table

---



### 3.1 What `tools=[...]` actually does (and why only one file uses it)

#### The `tools` argument — a schema sent *to* the LLM, not code

When you call `chat(model=..., messages=..., tools=[...])`, you're not registering Python functions with the LLM. You're sending the LLM a **JSON schema** that describes what tools *exist*, what each one is *for*, and what *arguments* it accepts. The LLM reads this schema the same way it reads any other context — as tokens.

Here's the exact `get_current_weather` schema from `my-agent.py`:

```python
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city":    {"type": "string", "description": "The name of the city, e.g. Athens"},
                "unit":    {"type": "string", "enum": ["celsius", "fahrenheit"], ...}
            },
            "required": ["city"]
        }
    }
}
```

Ollama (and OpenAI, Anthropic, etc.) inject this schema into the prompt behind the scenes, often with a header like:

```
You have access to the following functions. To use them, respond with a JSON
object like: {"tool_calls": [{"function": {"name": "...", "arguments": "..."}}]}
```

The model has been **fine-tuned** during post-training to recognize this schema-in-prompt pattern and to emit the matching JSON structure in its output. That's why `tools=[...]` only works on models trained to do tool calling — it's not magic, it's a contract.

#### Why `react-agent.py` doesn't pass `tools=[...]`

Two reasons:

1. **The model is too small / too old.** `llama3.2` (and any model < ~13B without tool-calling fine-tuning) is unreliable at emitting the exact `tool_calls` JSON shape. The few-shot text approach is more forgiving because the model just has to generate text that *looks like* the example.
2. **We want visible reasoning.** With `tools=[...]`, the model's *why* for picking a tool is hidden — it just outputs `tool_calls`. With text ReAct, the model writes `Thought: ...` first, which we can log, print, and feed back to it. For debugging and audit trails, that's gold.

So the rule is:

> Pass `tools=[...]` when your model is good at tool calling and you don't need visible reasoning.
> Skip it (use text ReAct) when your model can't reliably emit structured tool calls, or when you need to see *why* it picked a tool.

---



### 3.2 Full turn-by-turn flow — `my-agent.py`



#### Step 1 — Build the initial state

```python
messages = [{"role": "user", "content": user_prompt}]
```

A list with one message. The `messages` list is the agent's **state** — it grows every turn.

#### Step 2 — Send state + tool schema to LLM

```python
response = chat(
    model="llama3.2:latest",
    messages=messages,    # state: the conversation so far
    tools=tools           # schema: what tools exist (NOT code)
)
```

The `tools` argument is **deserialized into the prompt** by Ollama. The model sees something like:

```
System: You are a helpful assistant. You have access to the following functions:
[get_current_weather(city, unit), calculate(expression), get_time(city), search_web(query, max_results)]
When the user asks something, decide if you need a function. If so, respond with a JSON
object containing "tool_calls".

User: What is the time in Berlin now?
```



#### Step 3 — Inspect the response (the magic part)

```python
if not response.message.tool_calls:
    return response.message.content  # model said "I have an answer"
```

`response.message` is a typed object. `tool_calls` is either `None` (model is done) or a **list of** `ToolCall` **objects**. Each `ToolCall` has `.id`, `.function.name`, `.function.arguments`. **No parsing.** The library has already extracted these from the model's raw output (which, under the hood, looks like JSON — but you never see that JSON).

If `tool_calls` is empty, the model decided it had enough info and emitted a normal text answer. Done.

#### Step 4 — Execute the tools

```python
for tool_call in response.message.tool_calls:
    result = execute_tool(tool_call)   # your Python function runs here
```

`execute_tool` looks up the function by name in `available_functions` (your local Python dict — *this* is where actual code lives) and calls it with the arguments from the tool_call.

#### Step 5 — Append the tool result as a typed message

```python
messages.append({
    "role": "tool",
    "tool_name": function_name,
    "tool_call_id": tool_call.id,
    "content": str(result)
})
```

The `role: "tool"` message is **what the model actually reads** as "the result of your last tool call." The `tool_call_id` links it back to the assistant message that requested it. The model is fine-tuned to understand this role.

#### Step 6 — Loop back to step 2

State has grown: `[user, assistant(tool_calls), tool(result), ...]`. Send it back to the LLM. The LLM sees the `tool` role message, decides if it has enough info, either calls more tools or emits a final answer.

#### The full state diagram

```
                ┌─────────────────────────────────────────────┐
                │              messages: list                 │
                │                                             │
 Turn 1 ────►   │  [user: "What is the time in Berlin?"]     │
                │           │                                 │
                │           ▼ chat(messages, tools=...)       │
                │           │                                 │
                │  [user,                                    │
                │   assistant: {content: null,                │
                │              tool_calls: [                  │
                │                {id: "abc",                  │
                │                 function: {name: "get_time", │
                │                           arguments:        │
                                '{"city":"Berlin"}'}}, ...]}]  │
                │           │                                 │
                │           ▼ execute_tool(...)               │
                │           │                                 │
                │  [user, assistant(tool_calls),              │
                │   tool: {tool_call_id: "abc",               │
                │          content: '{"datetime":"..."}'}]    │
                │           │                                 │
                │           ▼ chat(messages, tools=...)       │
                │           │                                 │
                │  [user, assistant(tool_calls),              │
                │   tool,                                    │
                │   assistant: {content: "It is 17:45...",    │
                │              tool_calls: None}]             │ ◄── no tool_calls → return
                └─────────────────────────────────────────────┘
```

The state is **typed at every step**. The model sees:

- "user" → human
- "assistant" → itself (with `tool_calls` being its *decisions*)
- "tool" → runtime results, linked by `tool_call_id`

---



### 3.3 Full turn-by-turn flow — `react-agent.py`



#### Step 1 — Build the prompt

```python
system_prompt = build_react_prompt()   # huge prompt with format rules + few-shot
transcript = f"Question: {user_prompt}\n"
```

Two strings: a static system prompt (built once) and a growing transcript string.

#### Step 2 — Send state to LLM (NO tools schema)

```python
response = chat(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": transcript},
    ],
    # NO tools=[...]   ← the model has no idea tools exist as a structured concept
)
```

The system prompt **manually** tells the model:

- What tools exist (via `render_tool_descriptions()`)
- The exact output format (`Thought: / Action: / Action Input: / Observation:`)
- A few-shot example showing what good output looks like

The model treats all of this as **text instructions**. There's no special "tool calling" channel — just plain text generation.

#### Step 3 — Inspect the response (the fragile part)

```python
model_text = (response.message.content or "").strip()
parsed = parse_react_step(model_text)
```

`response.message.content` is just a string. You **regex it** to find:

- `Final Answer: ...` → done
- `Action: <name>` + `Action Input: {<json>}` → run the tool
- Anything else → parse error, feed the error back as `Observation`



#### Step 4 — Execute the tool

```python
result = dispatch_tool(tool_name, args)
observation_json = json.dumps(result)
```

Same as `my-agent.py`. The tool is real Python. The result is real data.

#### Step 5 — Append the tool result as fake text

```python
transcript += f"Observation: {observation_json}\n"
```

This is the **critical difference**. We append a string that **looks like** an Observation line. The model has to figure out, from text context alone, that this line came from the runtime and not from itself. There's no `tool_call_id`, no `role: "tool"`, no structural marker.

#### Step 6 — Loop back to step 2

Send the **entire transcript as a single user message**. The model reads the whole history — including its own past ThActions and Observations — and generates the next step.

#### The full state diagram

```
   ┌─────────────────────────────────────────────────────────────┐
   │                    transcript: str                          │
   │                                                             │
   │ "Question: What is the time in Berlin?                      │
   │  Thought: I need the time in Berlin.                        │
   │  Action: get_time                                           │
   │  Action Input: {"city": "Berlin"}                           │
   │  Observation: {"datetime": "2026-09-16T17:45"}              │
   │  Thought: Now I need the weather...                         │
   │  Action: get_current_weather                                │
   │  Action Input: {"city": "Berlin", "unit": "celsius"}        │
   │  Observation: {"temperature": 18.9, "unit": "celsius"}      │
   │  Thought: I have both answers.                              │
   │  Final Answer: The time in Berlin is 17:45 and..."          │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       chat(messages)
                              │
                              ▼
                  one big string in, one big string out
```

**The model cannot distinguish:**

- A `Thought:` line it wrote two turns ago
- A `Thought:` line it just hallucinated
- An `Observation:` line that came from the runtime
- An `Observation:` line it hallucinated (this is what broke your Berlin run)

That's the entire class of bugs that disappears the moment you switch to `my-agent.py`'s approach.

---



### 3.4 Side-by-side comparison table (expanded)


| Aspect                                          | `my-agent.py` (Tool-Calling)                                                            | `react-agent.py` (ReAct text)                                              | Why it matters                                       |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------- |
| `tools=[...]` **passed to LLM?**                | Yes — schema injected into prompt                                                       | No — model only sees text tool descriptions                                | Only one gives the model a "structured tool channel" |
| **What the model sees**                         | System prompt + conversation + JSON schema of tools + format hint                       | System prompt with format rules + few-shot + tool descriptions in markdown | Different prompt strategies                          |
| **What the model emits**                        | A typed object: `response.message.tool_calls = [ToolCall(...), ...]` or `content="..."` | A plain string you must parse                                              | Parsing is where bugs live                           |
| **How we extract tool calls**                   | `response.message.tool_calls[i].function.name`                                          | Regex: `ACTION_RE.search(model_text)`                                      | Typed access vs. string parsing                      |
| **How we feed results back**                    | `messages.append({"role": "tool", "tool_call_id": ..., "content": ...})`                | `transcript += f"Observation: {json}\n"`                                   | Typed roles vs. text masquerading                    |
| **Can the model hallucinate a tool result?**    | **No** — only the runtime can append `role: "tool"` messages                            | **Yes** — model can write `Observation:` lines itself                      | This is your Berlin bug                              |
| **Can the model emit two Actions in one turn?** | Yes (multiple `tool_calls`), and we run all of them                                     | Yes (multiple `Action:` lines), and our regex picks the **first** only     | Different loss modes                                 |
| **Reasoning visibility**                        | Hidden inside the model's decision to emit `tool_calls`                                 | Visible: `Thought:` lines are real text in the transcript                  | Audit / debug trade-off                              |
| **Prompt complexity**                           | Low: just the user message + history                                                    | High: format rules, few-shot example, tool descriptions                    | Text ReAct demands careful prompting                 |
| **Model size needed**                           | Larger models (tool-calling is a fine-tuned skill)                                      | Smaller models OK (any text generator works)                               | Text ReAct is more forgiving                         |
| **Parsing errors**                              | Library handles malformed JSON, returns no `tool_calls`                                 | You must write regex, handle errors, feed them back                        | Maintenance burden                                   |
| `max_steps` **safety net**                      | Yes — without it, infinite loops possible                                               | Yes — same                                                                 | Both need it                                         |


---



### 3.5 The single sentence to remember

> `my-agent.py` **gives the LLM a *structured channel* for tool calls;** `react-agent.py` **makes the LLM *describe* tool calls in natural language and we parse them. The first eliminates hallucinated results; the second preserves visible reasoning.**

That's the entire architectural difference. Everything else flows from this one choice.

### 3.4 When the model hallucinates: side-by-side

What you saw in your Berlin run:

```
Model output:        "Observation: {time: 14:30, ...}"      ← hallucinated
Our execution:       get_time(...)                            ← ran real call
Our appended line:   "Observation: {datetime: 2026-09-16...}" ← real result
```

In `my-agent.py`, the model **physically cannot** write an Observation. Only `ToolNode` (or your `execute_tool`) can append a `role: "tool"` message. The model can only emit `tool_calls` and content. This single constraint eliminates the entire class of bugs you just saw.

---



## 4. The Major Agent Patterns

Here's the landscape. I'll group them by **what extra structure they add on top of ReAct**.

### 4.1 The core family tree

```
                         ┌──────────────────┐
                         │   ReAct Loop     │  ← the base: think → act → observe
                         └────────┬─────────┘
                                  │
        ┌─────────────────┬───────┼───────┬──────────────────┐
        │                 │       │       │                  │
        ▼                 ▼       ▼       ▼                  ▼
   Tool-Calling     Reflexion  Plan-&-  Multi-Agent      Tool-Use
   Agent            (Self-      Execute  (Supervisor /    Optimization
                   Critique)           Swarm)             (function
                                                           calling v2)
```



### 4.2 The seven patterns worth knowing



#### ① **Tool-Calling Agent** (`my-agent.py`)

- **What:** LLM in a loop, calls tools via native function calling.
- **Use when:** Default choice. 80% of real-world agents.
- **Examples:** ChatGPT plugins, Claude with tools, most LangChain `create_react_agent(..., tools=...)` setups.



#### ② **ReAct (text-based)** (`react-agent.py`)

- **What:** Same loop, but the model emits `Thought:/Action:/Observation:` as text and you parse it.
- **Use when:** Small/old models without tool support, or when you need visible reasoning traces for debugging/audit.



#### ③ **Plan-and-Execute**

- **What:** First pass: LLM writes a **plan** (a list of steps). Second pass: a worker LLM executes each step, possibly with replanning.
- **Use when:** Long-horizon tasks where you want stability ("don't get distracted after 5 tool calls").
- **Examples:** BabyAGI, AutoGPT (sort of), LangChain's `PlanAndExecute` agent.

```
User ──► Planner ──► [Step1, Step2, Step3] ──► Executor ──► Step1 result
                                  ▲                        │
                                  └──── replan if needed ─┘
```



#### ④ **Reflexion** (Self-Critique)

- **What:** ReAct loop + a **reflector** that critiques failures and stores them in memory, so the next attempt avoids the same mistake.
- **Use when:** Tasks that benefit from trial-and-error (code generation, reasoning puzzles).
- **Examples:** Shinn et al. 2023 paper, "Reflexion: Language Agents with Verbal Reinforcement Learning."

```
Attempt → Fail → Reflect: "I forgot to escape the quotes"
Attempt → Fail → Reflect: "The function signature was wrong"
Attempt → Success
```



#### ⑤ **Multi-Agent (Supervisor)**

- **What:** A **supervisor** agent decides which **worker** agent to call next. Each worker has its own tools/persona.
- **Use when:** Tasks that decompose cleanly into specialized roles ("researcher + writer + reviewer").
- **Examples:** LangGraph multi-agent tutorials, CrewAI, AutoGen.

```
              ┌──► Researcher ──┐
Supervisor ───┤                  ├──► Synthesizer
              └──► Critic ───────┘
```



#### ⑥ **Multi-Agent (Swarm / Peer-to-Peer)**

- **What:** Agents hand off to each other; no fixed supervisor. Each can call any other.
- **Use when:** Open-ended collaboration, brainstorming.
- **Examples:** AutoGen group chat, CrewAI.



#### ⑦ **Tool-Use Optimized / Routing Agent**

- **What:** A small router LLM picks the tool, a large LLM only sees the result. Or: the same LLM but with tool descriptions heavily optimized (DSPy-style).
- **Use when:** Cost/latency-sensitive production systems where you want to minimize "expensive" LLM calls.



### 4.3 Mental table for fast recall


| Pattern                | Memory?           | Visible reasoning? | Multi-step?         | Best for                 |
| ---------------------- | ----------------- | ------------------ | ------------------- | ------------------------ |
| Tool-Calling           | No                | No                 | Yes                 | Default, most production |
| ReAct (text)           | No                | Yes                | Yes                 | Debugging, small models  |
| Plan-and-Execute       | Yes (plan)        | Yes                | Yes (long)          | Long-horizon, stable     |
| Reflexion              | Yes (reflections) | Yes                | Yes (tries N times) | Trial-and-error          |
| Multi-Agent Supervisor | Optional          | Yes                | Yes                 | Specialized roles        |
| Multi-Agent Swarm      | Optional          | Yes                | Yes                 | Open-ended collab        |
| Routing                | No                | No                 | Yes                 | Cost-sensitive           |


---



## 5. How to Choose an Agent Pattern for a Real Situation

This is the **most important section** of this doc. Bookmark it.

### 5.1 The four questions you ask before building

```
                         ┌───────────────────────┐
                         │  What's the task?     │
                         └───────────┬───────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
          Single-shot?         Multi-step?          Long-horizon?
                │                    │                    │
                ▼                    ▼                    ▼
         Just LLM call      Does plan help?     Plan-and-Execute?
                            (yes/no)            Multi-Agent?
                │              │                       │
                ▼              ▼                       ▼
            Done.         Tool-Calling             Add memory?
                          ReAct                    Reflexion?
                          Multi-Agent
```



#### Q1: **How many distinct steps will this take?**

- **1 step (or zero):** Don't build an agent. A regular LLM call (or function) is enough.
- **2–10 steps:** Tool-Calling / ReAct agent. Default.
- **10+ steps or "I'll lose focus after a while":** Plan-and-Execute.



#### Q2: **Will the agent benefit from seeing its own reasoning?**

- **Yes — for debugging, audit, or trust:** ReAct (text) or Plan-and-Execute with explicit plans.
- **No — production, cost-sensitive:** Tool-Calling with structured outputs.



#### Q3: **Will the agent fail and need to retry?**

- **Yes (code gen, math, complex reasoning):** Reflexion. Store failed attempts, reflect on why, retry with the reflection in context.
- **No (mostly straightforward tool use):** Plain Tool-Calling is fine.



#### Q4: **Are there distinct roles / domains the agent switches between?**

- **Yes ("research" vs. "write" vs. "review"):** Multi-Agent Supervisor.
- **No (one coherent task):** Single agent is simpler and often better.



### 5.2 The decision factors (expanded)


| Factor                 | Why it matters                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **Latency budget**     | Tool-Calling: ~1 LLM call per step. Plan-and-Execute: 2x. Multi-Agent: 3–5x.                          |
| **Cost budget**        | Same as latency. Multi-agent = expensive.                                                             |
| **Observability need** | Regulated/medical/finance? Need visible reasoning → ReAct text or Plan-and-Execute with logged plans. |
| **Determinism need**   | Production CI/CD? Lean toward Plan-and-Execute (plans are inspectable) over free-form ReAct.          |
| **Model capability**   | Tiny models (3B, 7B) → ReAct text (can't do structured tool calls well). Big models → Tool-Calling.   |
| **Failure tolerance**  | High-stakes → add Reflexion, retries, human-in-the-loop. Low-stakes → plain agent.                    |
| **Tool ecosystem**     | Many similar tools (e.g., 6 search APIs) → add a router agent to pick the right one.                  |
| **Memory needs**       | Single-session → no memory needed. Cross-session → InMemoryStore or PostgresStore (Week 3).           |




### 5.3 Anti-patterns to avoid


| Anti-pattern                                     | Why it's bad                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **Multi-agent for everything**                   | 3x the cost, 3x the bugs. Most tasks are fine as a single Tool-Calling agent.        |
| **ReAct text when you have native tool calling** | Strictly worse — hallucination-prone, slower, fragile. Only use it when you *must*.  |
| **Plan-and-Execute for short tasks**             | Overkill. The planning overhead exceeds the benefit.                                 |
| **No observability**                             | If you can't see *why* the agent did X, you can't fix it. Always log messages/state. |
| **Skipping evals**                               | "It worked on my 5 prompts" is not a system. Week 5+ in focus.md.                    |




### 5.4 Real-world anchors (so you remember)


| Situation                                         | Pattern to use                                   | Why                                         |
| ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------- |
| Customer-support chatbot that looks up orders     | Tool-Calling                                     | Short, deterministic, low cost              |
| Coding copilot that tests & retries               | Reflexion                                        | Benefits from self-critique on failed tests |
| "Research agent" that searches, reads, summarizes | Multi-Agent (Searcher + Reflector + Synthesizer) | Clean role separation, parallelizable       |
| Long-running ops agent (multi-day tasks)          | Plan-and-Execute + Memory                        | Stability across many steps                 |
| Mobile/edge, must run on small model              | ReAct (text)                                     | No native tool support                      |
| Regulated industry, audit trail required          | Plan-and-Execute + logged plans                  | Inspectable decisions                       |


---



## 6. Building These Agents in LangGraph

**Short note:** LangGraph doesn't change the patterns above. It gives you a **better substrate** to build them on.

### 6.1 The key idea

Each pattern maps to a graph:


| Pattern                | Graph shape                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| Tool-Calling Agent     | `START → agent → (tools? → tools → agent : END)`                            |
| ReAct (text)           | Same as above, but agent is your custom parser instead of native tool calls |
| Plan-and-Execute       | `START → planner → [step1 → executor, step2 → executor, ...] → synth → END` |
| Reflexion              | Tool-Calling agent + a `reflect` node that loops back                       |
| Multi-Agent Supervisor | Supervisor node fans out to worker sub-graphs                               |
| Routing                | Same as Tool-Calling, but with a router node in front                       |




### 6.2 Why LangGraph makes it easier

1. **The loop is the graph.** You draw the cycle once; LangGraph handles routing.
2. **State is typed.** No more transcript-as-string soup. `messages: Annotated[list, add_messages]` is the canonical reducer.
3. **Tools are a node.** `ToolNode(tools)` runs every tool call and returns typed `ToolMessage`s.
4. **Edges are explicit.** `tools_condition` is the built-in "did the LLM call a tool?" router.
5. **Checkpoints are free.** `MemorySaver` + `thread_id` gives you session memory in 2 lines.
6. **Subgraphs compose.** A multi-agent system is just multiple graphs whose states are channels in a parent graph.



### 6.3 The Week 1 → Week 4 mapping (from `focus.md`)

```
my-agent.py    ──►  week1_graph.py     (StateGraph + ToolNode + MemorySaver)
react-agent.py ──►  week1_graph.py     (same — both are tool-calling under LangGraph)

Week 2 ──► + interrupt_before + Command(resume=...) + streaming
Week 3 ──► + InMemoryStore for cross-thread memory + summarize-and-trim
Week 4 ──► + 3 sub-graphs (searcher, reflector, synthesizer) in a parent graph
```

The Berlin run you saw? `week1_graph.py` would have:

- Run `get_time` once. Returned `{"datetime": "..."}` as a typed `ToolMessage`.
- The model would have seen it, decided on `get_current_weather`, called it.
- The model would have seen it, decided it had both answers, and emitted `Final Answer`.
- **No** `(3.2 + 32)/2` **math loop. No hallucinated Observations.** The model physically can't do that.



### 6.4 One-line summary

> LangGraph = a graph runtime where nodes are functions and edges are conditions. Every agent pattern becomes "draw the graph, attach a state schema, optionally checkpoint it." That's the whole game.

---



## Quick recall card (for your wall)

```
┌──────────────────────────────────────────────────────────┐
│  Agent Pattern Cheat Sheet                                │
│                                                            │
│  • Default choice      → Tool-Calling Agent               │
│  • Need visible logic → ReAct (text) or Plan-and-Execute  │
│  • Long-horizon       → Plan-and-Execute                  │
│  • Trial-and-error    → Reflexion                         │
│  • Specialized roles  → Multi-Agent Supervisor            │
│  • Small model        → ReAct (text)                      │
│  • Regulated/audit    → Plan-and-Execute + logged plans   │
│                                                            │
│  ALL of them are ReAct loops with extra structure.        │
│  LangGraph is the substrate that makes them all cleaner.  │
└──────────────────────────────────────────────────────────┘
```

---

### resources
![Choose a design pattern for your agentic AI system](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
![Learning Agentic Patterns](https://www.philschmid.de/agentic-pattern)
![5 Agentic AI Design Patterns](https://blog.dailydoseofds.com/p/5-agentic-ai-design-patterns)

