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

def test_graph_keeps_processing_events():
    graph = build_graph()

    result = graph.invoke({"ticket": "How do I reset my password?"})

    assert result["events"] == ["classified:question", "drafted"]


def test_graph_feedback_processing_events():
    graph = build_graph()

    result = graph.invoke({"ticket": "Please add dark mode to the app."})
    assert result["category"] == "feedback"
    assert result["draft"] == "We will review this feedback."
    assert result["events"] == ["classified:feedback", "drafted"]
