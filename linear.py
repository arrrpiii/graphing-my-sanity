from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    final: str


def create_outline(state: State):
    topic = state["topic"]
    return {"outline": f"Outline for {topic}: intro, body, conclusion"}


def write_draft(state: State):
    outline = state["outline"]
    return {"draft": f"Draft based on: {outline}"}


def polish_draft(state: State):
    draft = state["draft"]
    return {"final": f"Polished version: {draft}"}


builder = StateGraph(State)

builder.add_node("create_outline", create_outline)
builder.add_node("write_draft", write_draft)
builder.add_node("polish_draft", polish_draft)

builder.add_edge(START, "create_outline")
builder.add_edge("create_outline", "write_draft")
builder.add_edge("write_draft", "polish_draft")
builder.add_edge("polish_draft", END)

graph = builder.compile()

result = graph.invoke({"topic": "LangGraph implementation"})

print(result["topic"])
print(result["outline"])
print(result["draft"])
print(result["final"])