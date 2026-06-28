from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    intent: str
    answer: str


# llm = ChatOpenAI(model="gpt-4o-mini")
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def classify(state: State):
    prompt = f"""
Classify the user request into one of these categories:

- code
- theory
- other

User request:
{state["user_input"]}

Return only one word.
"""

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    if intent not in {"code", "theory", "other"}:
        intent = "other"

    return {"intent": intent}


def answer_code_question(state: State):
    prompt = f"""
You are a programming assistant.
Answer with practical code examples.

Question:
{state["user_input"]}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


def answer_theory_question(state: State):
    prompt = f"""
Explain the concept clearly and briefly.

Question:
{state["user_input"]}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


def answer_other_question(state: State):
    prompt = f"""
Answer the user helpfully.

Question:
{state["user_input"]}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


def route(state: State) -> Literal["code", "theory", "other"]:
    return state["intent"]


builder = StateGraph(State)

builder.add_node("classify", classify)
builder.add_node("code", answer_code_question)
builder.add_node("theory", answer_theory_question)
builder.add_node("other", answer_other_question)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    route,
    {
        "code": "code",
        "theory": "theory",
        "other": "other",
    },
)

builder.add_edge("code", END)
builder.add_edge("theory", END)
builder.add_edge("other", END)

graph = builder.compile()

result = graph.invoke({
    "user_input": "Show me Python syntax for LangGraph nodes and edges."
})

print(result["answer"])