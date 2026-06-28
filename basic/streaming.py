'''
streaming node update -
for chunk in graph.stream(
    {
        "messages": [
            {"role": "user", "content": "Explain agents"}
        ]
    },
    stream_mode="updates",
):
    print(chunk)

streaming full state values - 
for chunk in graph.stream(
    {
        "messages": [
            {"role": "user", "content": "Explain agents"}
        ]
    },
    stream_mode="values",
):
    print(chunk)


streaming LLM tokens -
for chunk in graph.stream(
    {
        "messages": [
            {"role": "user", "content": "Write a short poem about AI"}
        ]
    },
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]

        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
            
'''


from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()

print("Streaming response: \n")

for msg_chunk, metadata in graph.stream(
    {
        "messages": [
            {"role": "user", "content": "Explain agents in AI."}
        ]
    },
    stream_mode="messages",
):
    # msg_chunk.content contains the raw token/word.
    # We use end="" and flush=True to print it smoothly on the same line.
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)

print("\n")