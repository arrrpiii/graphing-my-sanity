'''
builder.add_conditional_edges(
    "source_node",
    routing_function, # Evaluates the current state and returns a specific route name (a string)
    {
        "new_val_1": "next_node_1", # If routing_function returns "new_val_1", go to "next_node_1"
        "new_val_2": "next_node_2", # If routing_function returns "new_val_2", go to "next_node_2"
    },
)
'''


from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    intent: str
    response: str


def classify_intent(state: State):
    text = state["user_input"].lower()

    if "price" in text or "cost" in text:
        return {"intent": "pricing"}
    elif "error" in text or "bug" in text:
        return {"intent": "support"}
    else:
        return {"intent": "general"}


def pricing_node(state: State):
    return {"response": "Here is the pricing information."}


def support_node(state: State):
    return {"response": "Let me help you debug the issue."}


def general_node(state: State):
    return {"response": "Here is a general response."}


def route_by_intent(state: State):
    intent = state["intent"]

    if intent == "pricing":
        return "pricing"
    elif intent == "support":
        return "support"
    else:
        return "general"


builder = StateGraph(State)

builder.add_node("classify_intent", classify_intent)
builder.add_node("pricing", pricing_node)
builder.add_node("support", support_node)
builder.add_node("general", general_node)

builder.add_edge(START, "classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "pricing": "pricing",
        "support": "support",
        "general": "general",
    },
)

builder.add_edge("pricing", END)
builder.add_edge("support", END)
builder.add_edge("general", END)

graph = builder.compile()

print(graph.invoke({"user_input": "What is the cost?"}))
print(graph.invoke({"user_input": "I found an error"}))