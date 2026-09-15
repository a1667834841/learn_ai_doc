from langgraph.checkpoint.memory import InMemorySaver
from graph import create_ticket_graph


def test_ticket_pauses_for_approval():
    # 准备图与内存持久化存储
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "ticket-101"}}
    initial_input = {
        "ticket_id": "T-101",
        "amount": 500.0,
        "status": "pending",
        "reviewer": None,
        "reject_reason": None,
    }

    # 首次触发执行
    result = graph.invoke(initial_input, config=config)

    # 1. 断言图已经挂起：返回字典包含 __interrupt__ 键
    assert "__interrupt__" in result
    assert len(result["__interrupt__"]) == 1

    # 2. 检查中断暴露出来的数据载荷
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["amount"] == 500.0
    assert "T-101" in interrupt_payload["question"]

    # 3. 检查图当前状态依然是 pending，且下一步等待在 human_approval 节点
    current_state = graph.get_state(config)
    assert current_state.next == ("human_approval",)


from langgraph.types import Command


def test_ticket_resumes_with_approval():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "ticket-102"}}
    initial_input = {
        "ticket_id": "T-102",
        "amount": 300.0,
        "status": "pending",
        "reviewer": None,
        "reject_reason": None,
    }

    # 1. 触发暂停
    first_result = graph.invoke(initial_input, config=config)
    assert "__interrupt__" in first_result

    # 2. 模拟人工审核员传入批准决策
    approval_decision = {"approved": True, "reviewer": "Alice"}
    resumed_result = graph.invoke(Command(resume=approval_decision), config=config)

    # 3. 验证退款完成，审核人被记录
    assert resumed_result["status"] == "completed"
    assert resumed_result["reviewer"] == "Alice"

    # 4. 图已经执行完毕，next 为空
    state = graph.get_state(config)
    assert state.next == ()



def test_ticket_resumes_with_rejection():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "ticket-103"}}
    initial_input = {
        "ticket_id": "T-103",
        "amount": 1200.0,
        "status": "pending",
        "reviewer": None,
        "reject_reason": None,
    }

    # 首次触发并暂停
    graph.invoke(initial_input, config=config)

    # 模拟拒绝
    reject_decision = {"approved": False, "reason": "已超过7天无理由退货期限"}
    resumed_result = graph.invoke(
        Command(resume=reject_decision), config=config
    )

    assert resumed_result["status"] == "rejected"
    assert resumed_result["reject_reason"] == "已超过7天无理由退货期限"
