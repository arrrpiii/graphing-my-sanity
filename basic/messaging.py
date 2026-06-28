from langgraph.graph import StateGraph, START, END, MessagesState


def chatbot_node(state: MessagesState):
    messages = state["messages"]

    last_message = messages[-1]
    user_text = last_message.content

    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"You said: {user_text}",
            }
        ]
    }


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot_node)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

result = graph.invoke({
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
})

print(result["messages"][-1].content)