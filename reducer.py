from typing import Annotated
from operator import add
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    input: str
    logs: Annotated[list[str], add]


def step_1(state: State):
    return {"logs": ["Step 1 completed"]}


def step_2(state: State):
    return {"logs": ["Step 2 completed"]}


builder = StateGraph(State)

builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", END)

graph = builder.compile()

result = graph.invoke({
    "input": "run",
    "logs": [],
})

print(result["logs"])