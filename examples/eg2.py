from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    critique: str
    score: int
    revision_count: int
    final: str


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def plan_outline(state: State):
    prompt = (
        "Create a clear three-section outline for a short article on: "
        f"{state['topic']}\n\n"
        "Sections: Introduction, Main body, Conclusion.\n"
        "Return only bullet points, one per line."
    )
    response = llm.invoke(prompt)
    return {"outline": response.content.strip()}


def write_draft(state: State):
    if state.get("critique"):
        prompt = (
            "Revise the draft using the critique below.\n\n"
            f"Outline:\n{state['outline']}\n\n"
            f"Current draft:\n{state['draft']}\n\n"
            f"Critique:\n{state['critique']}\n\n"
            "Return the improved draft only."
        )
    else:
        prompt = (
            f"Write a short article (about 200 words) following this outline:\n\n"
            f"{state['outline']}"
        )
    response = llm.invoke(prompt)
    return {"draft": response.content.strip()}


def critique_draft(state: State):
    prompt = (
        "You are a strict editor. Critique the following draft. "
        "Score it 1-10 for clarity, accuracy, and completeness.\n\n"
        f"Draft:\n{state['draft']}\n\n"
        "Respond EXACTLY in this format:\n"
        "Score: <integer>\n"
        "Critique: <one paragraph of concrete feedback>"
    )
    response = llm.invoke(prompt)
    text = response.content.strip()
    score = 5
    critique = "Needs improvement."
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("score:"):
            digits = "".join(ch for ch in line.split(":", 1)[1] if ch.isdigit())
            if digits:
                score = max(1, min(10, int(digits)))
        elif lower.startswith("critique:"):
            critique = line.split(":", 1)[1].strip() or critique
    return {"score": score, "critique": critique}


def bump_revision(state: State):
    return {"revision_count": state.get("revision_count", 0) + 1}


def finalize(state: State):
    return {"final": state["draft"]}


def should_continue(state: State) -> Literal["revise", "finalize"]:
    if state["score"] >= 8:
        return "finalize"
    if state.get("revision_count", 0) >= 2:
        return "finalize"
    return "revise"


builder = StateGraph(State)

builder.add_node("plan_outline", plan_outline)
builder.add_node("write_draft", write_draft)
builder.add_node("critique_draft", critique_draft)
builder.add_node("bump_revision", bump_revision)
builder.add_node("finalize", finalize)

builder.add_edge(START, "plan_outline")
builder.add_edge("plan_outline", "write_draft")
builder.add_edge("write_draft", "critique_draft")

builder.add_conditional_edges(
    "critique_draft",
    should_continue,
    {
        "revise": "bump_revision",
        "finalize": "finalize",
    },
)

builder.add_edge("bump_revision", "write_draft")
builder.add_edge("finalize", END)

graph = builder.compile()

result = graph.invoke({
    "topic": "How transformer attention mechanisms work",
    "revision_count": 0,
})

print("FINAL ARTICLE:")
print(result["final"])
print()
print(f"Revisions used: {result['revision_count']}")
print(f"Final score: {result['score']}/10")
print()
print("OUTLINE THAT GUIDED THE WRITING:")
print(result["outline"])