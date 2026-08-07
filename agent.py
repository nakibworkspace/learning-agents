from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama

class State(TypedDict):
    question: str
    research: str
    summary: str

llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

def research_node(state: State):

    question = state["question"]

    response = llm.invoke(
        f"""
        Research the following topic.

        Topic:
        {question}

        Write detailed research notes.
        """
    )

    state["research"] = response.content

    return state

def summary_node(state: State):

    notes = state["research"]

    response = llm.invoke(
        f"""
        Summarize the following research notes.

        Notes:
        {notes}
        """
    )

    state["summary"] = response.content
    return state

graph = StateGraph(State)

graph.add_node("research", research_node)
graph.add_node("summary", summary_node)

graph.add_edge(START, "research")
graph.add_edge("research", "summary")
graph.add_edge("summary", END)

agent = graph.compile()

result = agent.invoke(
    {
        "question": "What is LangGraph?",
        "research": "",
        "summary": "",
    }
)

print(result)

