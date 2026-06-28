# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI



# llm = ChatGoogleGenerativeAI(model="gpt-4o-mini")
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

result = graph.invoke({
    "messages": [
        {"role": "user", "content": "Explain LangGraph in one sentence."}
    ]
})

print(result["messages"][-1].content)