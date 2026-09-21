from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

RESEARCH_SYSTEM = "You are a researcher. Gather relevant facts. Do not answer the user directly."
SUMMARY_SYSTEM = "You are a writer. Summarize the research above for the user in clear language."
AGENT_SYSTEM = """You are an agent. Decide what to do next.
  Reply with EXACTLY one of:
    RESEARCH: <what to look up>
    FINAL: <your final answer>"""

def agent_node(state: State):
    msgs= state["messages"] + [SystemMessage(content=AGENT_SYSTEM)]
    response = llm.invoke(msgs)
    return {"messages": [response]}

def research_node(state: State):
    msgs = state["messages"] + [SystemMessage(content=RESEARCH_SYSTEM)]
    response = llm.invoke(msgs)
    return {"messages": [response]}

def summary_node(state: State):
    msgs = state["messages"] + [SystemMessage(content=SUMMARY_SYSTEM)]
    response = llm.invoke(msgs)
    return {"messages": [response]}

def should_continue(state: State) -> Literal["research", "summary", END]:
    last = state["messages"][-1].content
    if "RESEARCH" in last:
        return "research"
    if "FINAL" in last:
        return END
    return "summary"

graph = StateGraph(State)

graph.add_node("agent", agent_node)
graph.add_node("research", research_node)
graph.add_node("summary", summary_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("research", "agent")
graph.add_edge("summary", "agent")

agent = graph.compile()

result = agent.invoke(
    {"messages": [("user", "What is LangGraph?")]}
)

print(result)
