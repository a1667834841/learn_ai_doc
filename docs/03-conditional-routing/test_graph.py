from graph import build_graph


def test_ticket_graph_classifies_question():
    graph = build_graph()
    result = graph.invoke({"ticket": "How do I reset my password?"})
    assert result["category"] == "question"

def test_statement_ticket_is_feedback():
    graph = build_graph()
    result = graph.invoke({"ticket": "Please add dark mode to the app."})
    assert result["category"] == "feedback"



def test_draft_node_uses_category_from_classify_node():
    graph = build_graph()

    result = graph.invoke({"ticket": "How do I reset my password?"})

    assert result["category"] == "question"
    assert result["draft"] == "We will answer this question."

import pytest


@pytest.mark.parametrize(
    ("ticket", "expected_events"),
    [
        (
            "How do I reset my password?",
            ["classified:question", "searched", "drafted"],
        ),
        (
            "Please add dark mode to the app.",
            ["classified:feedback", "drafted"],
        ),
    ],
)
def test_routing_events_by_ticket(ticket, expected_events):
    graph = build_graph()

    result = graph.invoke({"ticket": ticket})

    assert result["events"] == expected_events



