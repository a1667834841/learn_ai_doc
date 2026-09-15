from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    ticket: str
    category: str


def classify(state: dict) -> dict:
    if "?" in state["ticket"]:
        return {"category": "question"}
    return {"category": "feedback"}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    return builder.compile()