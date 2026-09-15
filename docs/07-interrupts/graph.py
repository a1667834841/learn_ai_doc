from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


class TicketState(TypedDict):
    ticket_id: str
    amount: float
    status: str
    reviewer: Optional[str]
    reject_reason: Optional[str]


def human_approval_node(state: TicketState):
    ticket_id = state["ticket_id"]
    # 核心：调用 interrupt 挂起当前节点，并将待审信息暴露给外部
    decision = interrupt(
        {
            "question": f"是否批准工单 {ticket_id} 的退款申请？",
            "amount": state["amount"],
        }
    )

    if decision.get("approved"):
        return {"status": "approved", "reviewer": decision.get("reviewer")}
    else:
        return {
            "status": "rejected",
            "reject_reason": decision.get("reason", "无理由拒绝"),
        }

def execute_refund_node(state: TicketState):
    # 仅在审批通过时才真正执行退款结算
    if state["status"] == "approved":
        return {"status": "completed"}
    return state

def create_ticket_graph(checkpointer=None):
    builder = StateGraph(TicketState)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_refund", execute_refund_node)
   
    builder.add_edge(START, "human_approval")
    builder.add_edge("human_approval", "execute_refund")
    builder.add_edge("execute_refund", END)

    return builder.compile(checkpointer=checkpointer)
