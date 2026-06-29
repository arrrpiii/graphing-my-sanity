from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


SCHEMA = """
Table: books (id, title, author_id, published_year, genre)
Table: authors (id, name, country)
Table: borrowers (id, name, email)
Table: loans (id, book_id, borrower_id, loan_date, return_date)
"""


class State(TypedDict):
    question: str
    plan: str
    query: str
    issues: str
    score: int
    revision_count: int
    final: str


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def plan_query(state: State):
    prompt = (
        f"Schema:\n{SCHEMA}\n\n"
        f"Question: {state['question']}\n\n"
        "Plan the SQL briefly:\n"
        "1. Which tables?\n"
        "2. JOIN keys?\n"
        "3. Columns to select?\n"
        "4. WHERE filters?\n"
        "5. GROUP BY / ORDER BY / LIMIT?"
    )
    response = llm.invoke(prompt)
    return {"plan": response.content.strip()}


def draft_query(state: State):
    is_revision = state.get("revision_count", 0) > 0
    if is_revision:
        prompt = (
            f"Schema:\n{SCHEMA}\n\n"
            f"Question: {state['question']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Current query:\n{state['query']}\n\n"
            f"Issues to fix:\n{state['issues']}\n\n"
            "Return the corrected SQL only."
        )
    else:
        prompt = (
            f"Schema:\n{SCHEMA}\n\n"
            f"Question: {state['question']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            "Return the SQL only, no markdown."
        )
    response = llm.invoke(prompt)
    text = response.content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("sql").strip()
    return {"query": text}


def critique_query(state: State):
    prompt = (
        "You are a strict SQL reviewer.\n\n"
        f"Schema:\n{SCHEMA}\n\n"
        f"Question: {state['question']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        f"Query:\n{state['query']}\n\n"
        "Check:\n"
        "- Do all referenced columns exist in the schema?\n"
        "- Are JOIN keys correct?\n"
        "- Does the query actually answer the question?\n"
        "- Any missing filters, GROUP BY, or aggregations?\n\n"
        "Respond EXACTLY in this format:\n"
        "Score: <integer 1-10>\n"
        "Issues: <comma-separated issues, or 'none'>"
    )
    response = llm.invoke(prompt)
    text = response.content.strip()
    score = 5
    issues = "none"
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("score:"):
            digits = "".join(ch for ch in line.split(":", 1)[1] if ch.isdigit())
            if digits:
                score = max(1, min(10, int(digits)))
        elif lower.startswith("issues:"):
            issues = line.split(":", 1)[1].strip() or "none"
    return {"score": score, "issues": issues}


def bump_revision(state: State):
    return {"revision_count": state.get("revision_count", 0) + 1}


def finalize(state: State):
    return {"final": state["query"]}


def should_continue(state: State) -> Literal["revise", "finalize"]:
    if state["score"] >= 8:
        return "finalize"
    if state.get("revision_count", 0) >= 2:
        return "finalize"
    return "revise"


builder = StateGraph(State)

builder.add_node("plan_query", plan_query)
builder.add_node("draft_query", draft_query)
builder.add_node("critique_query", critique_query)
builder.add_node("bump_revision", bump_revision)
builder.add_node("finalize", finalize)

builder.add_edge(START, "plan_query")
builder.add_edge("plan_query", "draft_query")
builder.add_edge("draft_query", "critique_query")

builder.add_conditional_edges(
    "critique_query",
    should_continue,
    {
        "revise": "bump_revision",
        "finalize": "finalize",
    },
)

builder.add_edge("bump_revision", "draft_query")
builder.add_edge("finalize", END)

graph = builder.compile()

result = graph.invoke({
    "question": "List the top 5 most-borrowed books with their authors.",
    "revision_count": 0,
})

print("FINAL QUERY:")
print(result["final"])
print()
print(f"Revisions used: {result['revision_count']}")
print(f"Final score: {result['score']}/10")
print()
print("PLAN:")
print(result["plan"])
print()
print("FINAL ISSUES (if any):")
print(result["issues"])