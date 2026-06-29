# best try to demonstrate chain of throughts using langchain

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def make_extract_chain(llm):
    """
    A specialist sub-prompt that only does one thing: pull the explicit
    facts out of the puzzle. It is forbidden from solving — just extraction.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a logic extraction engine. Read the puzzle and list ONLY "
         "the explicit constraints as bullet points. Do NOT solve the puzzle. "
         "Do NOT infer anything that isn't stated. Just extract facts.\n"
         "Example output:\n"
         "  - C is between A and B\n"
         "  - E is at position 5 (far right)\n"
         "  - D is at position 4 (immediately left of E)"),
        ("user", "Puzzle:\n{puzzle}"),
    ])
    return prompt | llm | StrOutputParser()


def make_deduce_chain(llm):
    """
    Receives BOTH the original puzzle AND the constraints extracted in stage 1.
    That redundancy is intentional — it forces the model to reconcile the two
    and notice if stage 1 missed something.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are solving a seating-arrangement puzzle. Using the extracted "
         "constraints below, deduce each person's EXACT position (1 through 5, "
         "left to right). Show your work step by step. If the constraints are "
         "ambiguous, list every valid arrangement."),
        ("user",
         "Original puzzle:\n{puzzle}\n\n"
         "Extracted constraints (from Stage 1):\n{constraints}\n\n"
         "Now deduce the position of each person."),
    ])
    return prompt | llm | StrOutputParser()


def make_identify_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the puzzle and the position mapping produced so far, identify "
         "who sits exactly in the middle. Respond on the final line as:\n"
         "FINAL_ANSWER: <letter>"),
        ("user",
         "Original puzzle:\n{puzzle}\n\n"
         "Position mapping (from Stage 2):\n{mapping}\n\n"
         "Who sits exactly in the middle?"),
    ])
    return prompt | llm | StrOutputParser()


def make_verify_chain(llm):
    """
    The verifier stage. This is the new piece the original file did not have.
    It independently re-checks every constraint against the deduced mapping
    and either confirms the answer or flags a contradiction.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an independent verifier. Given the original puzzle and a "
         "proposed arrangement, check each constraint one by one. For each "
         "constraint, write PASS or FAIL and a one-line justification.\n"
         "After checking all constraints, end with exactly one of:\n"
         "VERIFIED: YES\n"
         "VERIFIED: NO  — <which constraints failed>"),
        ("user",
         "Original puzzle:\n{puzzle}\n\n"
         "Proposed arrangement (from Stage 2):\n{mapping}\n\n"
         "Verify every constraint."),
    ])
    return prompt | llm | StrOutputParser()


def run_chain_of_thought():
    puzzle = (
        "Five students (A, B, C, D, E) are sitting in a row facing North. "
        "C is strictly between A and B. "
        "E is at the far right end of the row. "
        "D is sitting immediately to the left of E. "
        "Based on this data, who is sitting exactly in the middle?"
    )

    llm = build_llm()

    extract  = make_extract_chain(llm)
    deduce   = make_deduce_chain(llm)
    identify = make_identify_chain(llm)
    verify   = make_verify_chain(llm)

    chain = (
        RunnablePassthrough.assign(constraints=extract)
        | RunnablePassthrough.assign(mapping=deduce)
        | RunnablePassthrough.assign(final=identify)
        | RunnablePassthrough.assign(verdict=verify)
    )

    result = chain.invoke({"puzzle": puzzle})

    for key in ("constraints", "mapping", "final", "verdict"):
        print("=" * 70)
        print(key.upper())
        print("=" * 70)
        print(result[key], "\n")

    return result


if __name__ == "__main__":
    result = run_chain_of_thought()
