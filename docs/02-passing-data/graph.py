from typing import Annotated, Literal, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    ticket: str
    category: str
    draft: str
    events: Annotated[list[str], add]


def category_for(ticket: str) -> Literal["question", "feedback"]:
    return "question" if "?" in ticket else "feedback"


def classify(state: State) -> dict:
    category = category_for(state["ticket"])
    return {
        "category": category,
        "events": [f"classified:{category}"],
    }


def draft_reply(state: State) -> dict:
    templates = {
        "question": "We will answer this question.",
        "feedback": "We will review this feedback.",
    }
    return {
        "draft": templates[state["category"]],
        "events": ["drafted"],
    }


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node(draft_reply)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "draft_reply")
    builder.add_edge("draft_reply", END)
    return builder.compile()