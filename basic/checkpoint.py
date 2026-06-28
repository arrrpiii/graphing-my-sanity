# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver


# llm = ChatOpenAI(model="gpt-4o-mini")
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


checkpointer = InMemorySaver()

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "user-123"
    }
}

result_1 = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "My name is Ayush."}
        ]
    },
    config=config,
)

result_2 = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is my name?"}
        ]
    },
    config=config,
)

# print(result_2["messages"][-1].content)

# print("\n" + "="*50)
# print("FULL STATE MEMORY (STEP-BY-STEP)")
# print("="*50 + "\n")

# Iterate through every message stored in the graph's final state
for message in result_2["messages"]:
    message.pretty_print()