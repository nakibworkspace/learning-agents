from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

RESEARCH_SYSTEM = "You are a researcher. Gather relevant facts. Do not answer the user directly."
SUMMARY_SYSTEM = "You are a writer. Summarize the research above for the user in clear language."

def research_node(state: State):
    msgs = state["messages"] + [SystemMessage(content=RESEARCH_SYSTEM)]
    response = llm.invoke(msgs)
    return {"messages": [response]}

def summary_node(state: State):
    msgs = state["messages"] + [SystemMessage(content=SUMMARY_SYSTEM)]
    response = llm.invoke(msgs)
    return {"messages": [response]}

graph = StateGraph(State)

graph.add_node("research", research_node)
graph.add_node("summary", summary_node)

graph.add_edge(START, "research")
graph.add_edge("research", "summary")
graph.add_edge("summary", END)

agent = graph.compile()

result = agent.invoke(
    {"messages": [("user", "What is LangGraph?")]}
)

print(result)
