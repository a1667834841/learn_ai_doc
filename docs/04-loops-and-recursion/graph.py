from typing import Annotated, Literal, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    ticket: str
    category: str
    draft: str
    events: Annotated[list[str], add]
    search_results: list[str]          # 新增：默认 reducer 即可
    attempts: int                      # 新增：跟踪重试次数

KNOWLEDGE_BASE = [
    "Reset password via Settings > Security > Change Password.",
    "Passwords must be at least 12 characters.",
]

def route_by_category(
    state: State,
) -> Literal["classify", "search_docs", "draft_reply"]:
    category = state["category"]
    if category == "uncertain":
        return "classify"
    if category == "question":
        return "search_docs"
    return "draft_reply"



def search_docs(state: State) -> dict:
    return {"search_results": KNOWLEDGE_BASE, "events": ["searched"]}


def classify(state: State) -> dict:
    attempts = state.get("attempts", 0)
    lowered = state["ticket"].lower()
    if "maybe" in lowered and attempts == 0:
        return {
            "category": "uncertain",
            "attempts": attempts + 1,
            "events": ["classified:uncertain"],
        }
    if "who knows" in lowered:
        return {"category": "uncertain", "events": ["classified:uncertain"]}
    # 终审不再经过 category_for：它没有轮次概念，maybe 会再次返回 uncertain 造成死循环
    category = "question" if "?" in state["ticket"] else "feedback"
    return {"category": category, "events": [f"classified:{category}"]}


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
    builder.add_node("search_docs", search_docs)
    builder.add_edge(START, "classify")
    # builder.add_edge("classify", "search_docs")
    builder.add_conditional_edges("classify", route_by_category)
    builder.add_edge("search_docs", "draft_reply")
    builder.add_edge("draft_reply", END)
    return builder.compile()

