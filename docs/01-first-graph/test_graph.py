from graph import build_graph


def test_ticket_graph_classifies_question():
    graph = build_graph()
    result = graph.invoke({"ticket": "How do I reset my password?"})
    assert result["category"] == "question"

def test_statement_ticket_is_feedback():
    graph = build_graph()
    result = graph.invoke({"ticket": "Please add dark mode to the app."})
    assert result["category"] == "feedback"