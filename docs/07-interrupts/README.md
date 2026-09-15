# 第 07 章：人在回路（Human-in-the-Loop）：暂停与恢复

在第 06 章中，我们学习了如何使用 Checkpointer（持久化检查点）和 `thread_id` 保存与恢复图的状态。有了持久化能力，我们就能进入一个非常关键的工程场景：**人在回路（Human-in-the-Loop, HITL）**。

在实际的**客服工单处理智能体（Support Ticket Agent）**中，并非所有操作都能全自动执行。比如：
- 客户申请了一笔 **500 元以上的大额退款**；
- 智能体需要修改客户的核心账户数据；
- 工单需要升级交由高级主管审批。

这些操作具有高风险或不可逆性。系统不能直接“自作主张”，而必须**暂停当前执行**，把待决断的事项和上下文抛出给人工审核员；等到审核员在后台完成审批（批准或驳回）后，系统再**原地唤醒并恢复流转**。

LangGraph 提供了现代的原语来支持这一机制：
1. `interrupt()`：在图节点内动态挂起执行，向调用方输出需要人工核对的数据；
2. `Command(resume=...)`：向已暂停的图输入外部决策，唤醒图继续向下流转；
3. **节点重放规则（Node Replay Rule）**：理解图恢复时的执行机制，避免重复执行副作用。

本章我们将遵循测试驱动开发（TDD）的节奏，一步一步用测试驱动出人在回路的完整生命周期。

---

## 来源契约

本章内容严格基于 LangGraph 官方文档与 Python API 规范：
- **官方文档**：[Interrupts | LangGraph](https://docs.langchain.com/oss/python/langgraph/interrupts)
- **API 参考**：[Command | Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api#resume)
- **核心契约**：
  1. `interrupt(payload)` 函数接受任意 JSON 可序列化的值，图会在此处暂停，将 payload 暴露在返回结果的 `__interrupt__` 字段中。
  2. 暂停依赖 Checkpointer：必须配置持久化层并在执行时传入 `thread_id`；若无 Checkpointer，调用 `Command(resume=...)` 会抛出 `RuntimeError`。
  3. 恢复契约：外部调用 `graph.invoke(Command(resume=value), config=config)`，传入的 `value` 将作为节点内部 `interrupt()` 调用的返回值。
  4. **节点重放规则**：恢复执行时，包含 `interrupt()` 的节点会**从函数第一行重新执行**，直到再次遇到 `interrupt()` 时注入 resume 值并继续。因此，`interrupt()` 之前的逻辑必须具备幂等性。
  5. 严禁在节点内部使用通用的 `try...except Exception:` 包裹 `interrupt()`，因为它的底层是通过抛出特定中断异常来实现挂起的。

---

## 迭代一：遇到高危操作时主动暂停

### 1. 先写测试

我们希望为工单图引入退款审核逻辑：当工单包含退款申请时，图必须在审核节点停下来，而不是一路冲到终点；并且返回的数据中应该携带人工审核所需的问题和金额。

在 `docs/07-interrupts` 目录下新建 `test_graph.py`：

```python
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
```

### 2. 运行测试（红灯）

此时我们还没有编写 `graph.py`，运行 pytest：

```bash
python -m pytest -q
```

实测输出：

```text
ERROR test_graph.py - ModuleNotFoundError: No module named 'graph'
```

这符合预期。缺少模块，让我们开始编写最小实现。

### 3. 编写最小实现

在同一目录下新建 `graph.py`。我们需要：
1. 定义工单的状态结构 `TicketState`；
2. 定义审批节点 `human_approval_node`，在其中调用 `interrupt()`；
3. 构建图并编译（注意必须接收 `checkpointer` 参数）：

```python
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

    # 后续逻辑暂时返回
    return {"status": "in_review"}


def create_ticket_graph(checkpointer=None):
    builder = StateGraph(TicketState)
    builder.add_node("human_approval", human_approval_node)

    builder.add_edge(START, "human_approval")
    builder.add_edge("human_approval", END)

    return builder.compile(checkpointer=checkpointer)
```

### 4. 运行测试（绿灯）

再次运行测试：

```bash
python -m pytest -q
```

实测输出：

```text
.                                                                        [100%]
1 passed in 0.95s
```

测试通过了！我们观察到了什么？
1. 当调用 `interrupt(...)` 时，LangGraph 保存了当前图状态到 `checkpointer` 中，并停止继续向下执行。
2. `graph.invoke(...)` 返回了一个包含 `__interrupt__` 的字典。每个元素都是一个 `Interrupt(value=...)` 对象，我们可以读取其 `value` 获取暴露的内容。
3. 通过 `graph.get_state(config)`，我们可以查看到 `state.next == ("human_approval",)`，表示图目前正阻塞在 `human_approval` 节点上，静待恢复。

---

## 迭代二：人工审核批准，使用 `Command(resume=...)` 恢复执行

### 1. 先写测试

一旦工单暂停，审核员在后台界面看到了这条退款申请并点击了“批准”。系统应该能够将审核员的决定注入回图流程中，执行后续的实际退款逻辑，将工单状态置为 `completed`。

在 `test_graph.py` 中追加测试用例：

```python
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
```

### 2. 运行测试（红灯）

运行测试：

```bash
python -m pytest -q
```

实测输出：

```text
.F                                                                       [100%]
=================================== FAILURES ===================================
______________________ test_ticket_resumes_with_approval _______________________
...
E       AssertionError: assert 'in_review' == 'completed'
E         - completed
E         + in_review
```

测试失败了！因为我们的 `human_approval_node` 目前固定返回 `{"status": "in_review"}`，并且缺少真正执行退款的后续节点。

### 3. 编写最小实现

在 `graph.py` 中：
1. 接收 `interrupt()` 的返回值——当恢复时，传入 `Command(resume=...)` 的对象会直接变成 `decision` 的值；
2. 依据审批结果更新状态；
3. 增加真正的执行结算节点 `execute_refund_node`：

```python
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
    # resume 传入的值将在恢复时作为 decision 返回
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
```

### 4. 运行测试（绿灯）

再次执行测试：

```bash
python -m pytest -q
```

实测输出：

```text
..                                                                       [100%]
2 passed in 0.94s
```

绿灯！我们用一行优雅的 `graph.invoke(Command(resume=...), config=config)` 就成功恢复了被挂起的图。

---

## 迭代三：人工审核拒绝流转

### 1. 先写测试

如果人工审核员发现该退款不符合政策并予以拒绝，工单应被标记为 `rejected`，记录拒绝原因，并且绝不能执行退款。

在 `test_graph.py` 中追加测试：

```python
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
```

### 2. 运行测试（绿灯）

运行测试：

```bash
python -m pytest -q
```

实测输出：

```text
...                                                                      [100%]
3 passed in 0.96s
```

测试直接通过！因为我们在上一步的代码中已经为 `decision.get("approved") == False` 准备好了分支逻辑。

---

## 迭代四：机制深挖——节点重放规则（Node Replay）与副作用隔离

这是人在回路中最容易被忽视、却在生产中最致命的概念：**节点重放规则（Node Replay Rule）**。

### 1. 什么是节点重放？

在我们的常识中，“恢复执行”似乎应该像操作系统的线程一样，从断点那一行继续向下执行。但 LangGraph 的底层工作机制是：
> **当图恢复时，包含 `interrupt()` 的节点会从该函数的开头重新执行（Restart from the beginning of the node）。**

为什么会这样？因为图状态是持久化在数据库（或内存）中的。在现实中，人工审核可能在 3 分钟后，也可能在 3 天后；此时原本的进程甚至机器早就重启了，Python 无法凭空还原函数中途的局部调用栈。因此，LangGraph 重新执行该节点，当代码执行到 `interrupt()` 时，框架检测到该中断已有 resume 值，于是直接返回该值继续向下执行。

### 2. 反模式测试：在 `interrupt()` 之前执行非幂等副作用

如果在 `interrupt()` 之前写了非幂等操作（例如发送邮件、累加计数器、写审计日志），恢复时就会被**执行两次**！

我们在 `test_graph.py` 中写一个测试来暴露这个问题：

```python
# 模拟一个外部审计系统调用记录
external_audit_log = []


def test_node_replay_causes_duplicate_side_effects():
    external_audit_log.clear()

    class AuditState(TypedDict):
        count: int

    def faulty_node(state: AuditState):
        # ❌ 反模式：在 interrupt 前调用非幂等外部副作用
        external_audit_log.append("audit_record_created")

        res = interrupt("Approve?")
        return {"count": 1}

    builder = StateGraph(AuditState)
    builder.add_node("faulty", faulty_node)
    builder.add_edge(START, "faulty")
    builder.add_edge("faulty", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "audit-1"}}

    # 1. 第一次调用：执行到 interrupt，此时产生了一条日志
    graph.invoke({"count": 0}, config=config)
    assert external_audit_log == ["audit_record_created"]

    # 2. 第二次调用（恢复）：节点重头执行，又执行了一次 append！
    graph.invoke(Command(resume=True), config=config)

    # 观察结果：日志里赫然出现了两条一模一样的记录！
    assert external_audit_log == [
        "audit_record_created",
        "audit_record_created",
    ]
```

运行测试验证：

```bash
python -m pytest -q -k "test_node_replay"
```

实测输出：

```text
.                                                                        [100%]
1 passed in 0.88s
```

断言 `external_audit_log == ["audit_record_created", "audit_record_created"]` 通过，证实了**前置代码确实重复执行了**！如果在真实生产中，这就是“发送了两封审批提醒邮件”或“扣了两次账户手续费”的线上事故。

### 3. 正确设计：重构与副作用隔离

如何彻底解决这个问题？官方推荐的最佳实践是：**将副作用与中断解耦**。

有两种重构策略：
1. **策略一（推荐）：将前置副作用抽离为独立的前置节点**。上游节点执行完毕后，Checkpointer 会保存上游节点执行后的状态。当图在审批节点暂停并恢复时，**上游节点绝对不会被重复执行**！
2. **策略二：将副作用移动到 `interrupt()` 返回之后**。

我们来把工单图重构为标准的架构，增加 `prepare_approval` 节点：

修改 `graph.py`：

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


class TicketState(TypedDict):
    ticket_id: str
    amount: float
    status: str
    reviewer: Optional[str]
    reject_reason: Optional[str]


def prepare_approval_node(state: TicketState):
    """前置准备节点：只执行一次，负责初始化审核状态或发出通知。"""
    return {"status": "pending_approval"}


def human_approval_node(state: TicketState):
    """审核节点：纯粹的等待交互，保持幂等。"""
    tid = state["ticket_id"]
    decision = interrupt(
        {
            "question": f"是否批准工单 {tid} 的退款申请？",
            "amount": state["amount"],
        }
    )

    if decision.get("approved"):
        return {"status": "approved", "reviewer": decision.get("reviewer")}
    return {
        "status": "rejected",
        "reject_reason": decision.get("reason", "无理由拒绝"),
    }


def execute_refund_node(state: TicketState):
    """后置执行节点：仅在审核通过后结算。"""
    if state["status"] == "approved":
        return {"status": "completed"}
    return state


def create_ticket_graph(checkpointer=None):
    builder = StateGraph(TicketState)
    builder.add_node("prepare_approval", prepare_approval_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_refund", execute_refund_node)

    builder.add_edge(START, "prepare_approval")
    builder.add_edge("prepare_approval", "human_approval")
    builder.add_edge("human_approval", "execute_refund")
    builder.add_edge("execute_refund", END)

    return builder.compile(checkpointer=checkpointer)
```

在 `test_graph.py` 中编写验证测试：

```python
def test_preceding_node_is_not_replayed():
    # 验证独立的前置准备节点在 resume 时不会被重复执行
    audit_counter = []

    def mock_prepare_approval(state: TicketState):
        audit_counter.append(state["ticket_id"])
        return {"status": "pending_approval"}

    builder = StateGraph(TicketState)
    builder.add_node("prepare_approval", mock_prepare_approval)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_refund", execute_refund_node)

    builder.add_edge(START, "prepare_approval")
    builder.add_edge("prepare_approval", "human_approval")
    builder.add_edge("human_approval", "execute_refund")
    builder.add_edge("execute_refund", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "isolated-1"}}

    # 首次执行
    graph.invoke(
        {
            "ticket_id": "T-999",
            "amount": 100.0,
            "status": "new",
            "reviewer": None,
            "reject_reason": None,
        },
        config=config,
    )
    assert audit_counter == ["T-999"]

    # 恢复执行
    graph.invoke(
        Command(resume={"approved": True, "reviewer": "Charlie"}),
        config=config,
    )

    # 核心断言：前置节点没有被重复调用！
    assert audit_counter == ["T-999"]
```

运行全部测试：

```bash
python -m pytest -q
```

实测输出：

```text
.....                                                                    [100%]
5 passed in 0.98s
```

5 个测试全部通过！通过结构化节点设计，我们彻底排除了重放带来的副作用隐患。

---

## 避坑指南（Gotchas）

在实现人在回路流程时，有三个最容易踩坑的规则，请务必记牢：

### 1. 绝对不要用通用的 `try...except` 包裹 `interrupt()`
`interrupt()` 的本质是在运行时抛出一个特殊的控制流异常（如 `GraphInterrupt`）。如果你写了：
```python
# ❌ 错误示范：异常被吞，图无法挂起！
try:
    ans = interrupt("请确认")
except Exception as e:
    logger.error("出错了")
```
通用的 `except Exception` 会将暂停异常捕获，导致图认为该节点正常完成并直接往下走，中断彻底失效！
> **规则**：若节点内有需要捕获异常的业务逻辑，请将其放在单独的函数中，或精确捕获特定的业务异常（如 `ValueError`、`KeyError`）。

### 2. 避免在单节点内使用 `while True: interrupt(...)` 循环
若想让用户反复输入直至校验合法，不要在单节点内写死循环：
```python
# ❌ 错误示范：每次 resume 都会导致前面的循环体全部重新重放
while True:
    code = interrupt("请输入 6 位验证码")
    if len(code) == 6:
        break
```
因为节点重放规则，如果用户输错了 3 次，第 4 次恢复时会导致前面的循环迭代全部重新执行一遍，造成严重的性能浪费和潜在逻辑错乱。
> **正确做法**：单次调用 `interrupt()`，并在节点退出后通过**条件边（Conditional Edges）**判断是否需要循环回到该节点重新请求输入。

### 3. 没有 Checkpointer 就无法使用 `Command(resume=...)`
如果在 `compile()` 时没有传入 `checkpointer`，调用 `Command(resume=...)` 会立即抛出：
```text
RuntimeError: Cannot use Command(resume=...) without checkpointer
```
必须为图配备持久化层（测试时使用 `InMemorySaver()`，生产时使用持久化数据库存储），并且两次调用必须传入完全相同的 `thread_id`。

---

## 完整代码清单

为方便你在本地随时复现，本章完整可运行代码如下：

### `graph.py`

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


class TicketState(TypedDict):
    ticket_id: str
    amount: float
    status: str
    reviewer: Optional[str]
    reject_reason: Optional[str]


def prepare_approval_node(state: TicketState):
    """前置准备节点：更新工单状态为等待审核。"""
    return {"status": "pending_approval"}


def human_approval_node(state: TicketState):
    """人在回路审批节点：挂起执行，等待外部人工输入。"""
    tid = state["ticket_id"]
    decision = interrupt(
        {
            "question": f"是否批准工单 {tid} 的退款申请？",
            "amount": state["amount"],
        }
    )

    if decision.get("approved"):
        return {"status": "approved", "reviewer": decision.get("reviewer")}
    return {
        "status": "rejected",
        "reject_reason": decision.get("reason", "无理由拒绝"),
    }


def execute_refund_node(state: TicketState):
    """退款执行节点：仅在审批通过后执行结算。"""
    if state["status"] == "approved":
        return {"status": "completed"}
    return state


def create_ticket_graph(checkpointer=None):
    builder = StateGraph(TicketState)
    builder.add_node("prepare_approval", prepare_approval_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_refund", execute_refund_node)

    builder.add_edge(START, "prepare_approval")
    builder.add_edge("prepare_approval", "human_approval")
    builder.add_edge("human_approval", "execute_refund")
    builder.add_edge("execute_refund", END)

    return builder.compile(checkpointer=checkpointer)
```

### `test_graph.py`

```python
from typing import TypedDict
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from graph import (
    create_ticket_graph,
    human_approval_node,
    execute_refund_node,
    TicketState,
)


def test_ticket_pauses_for_approval():
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

    result = graph.invoke(initial_input, config=config)

    assert "__interrupt__" in result
    assert len(result["__interrupt__"]) == 1
    assert result["__interrupt__"][0].value["amount"] == 500.0
    assert "T-101" in result["__interrupt__"][0].value["question"]

    current_state = graph.get_state(config)
    assert current_state.next == ("human_approval",)


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

    graph.invoke(initial_input, config=config)
    resumed_result = graph.invoke(
        Command(resume={"approved": True, "reviewer": "Alice"}), config=config
    )

    assert resumed_result["status"] == "completed"
    assert resumed_result["reviewer"] == "Alice"

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

    graph.invoke(initial_input, config=config)
    resumed_result = graph.invoke(
        Command(resume={"approved": False, "reason": "已超过7天无理由退货期限"}),
        config=config,
    )

    assert resumed_result["status"] == "rejected"
    assert resumed_result["reject_reason"] == "已超过7天无理由退货期限"


def test_node_replay_causes_duplicate_side_effects():
    external_audit_log = []

    class AuditState(TypedDict):
        count: int

    def faulty_node(state: AuditState):
        external_audit_log.append("audit_record_created")
        res = interrupt("Approve?")
        return {"count": 1}

    builder = StateGraph(AuditState)
    builder.add_node("faulty", faulty_node)
    builder.add_edge(START, "faulty")
    builder.add_edge("faulty", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "audit-1"}}

    graph.invoke({"count": 0}, config=config)
    assert external_audit_log == ["audit_record_created"]

    graph.invoke(Command(resume=True), config=config)
    assert external_audit_log == [
        "audit_record_created",
        "audit_record_created",
    ]


def test_preceding_node_is_not_replayed():
    audit_counter = []

    def mock_prepare_approval(state: TicketState):
        audit_counter.append(state["ticket_id"])
        return {"status": "pending_approval"}

    builder = StateGraph(TicketState)
    builder.add_node("prepare_approval", mock_prepare_approval)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_refund", execute_refund_node)

    builder.add_edge(START, "prepare_approval")
    builder.add_edge("prepare_approval", "human_approval")
    builder.add_edge("human_approval", "execute_refund")
    builder.add_edge("execute_refund", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "isolated-1"}}

    graph.invoke(
        {
            "ticket_id": "T-999",
            "amount": 100.0,
            "status": "new",
            "reviewer": None,
            "reject_reason": None,
        },
        config=config,
    )
    assert audit_counter == ["T-999"]

    graph.invoke(
        Command(resume={"approved": True, "reviewer": "Charlie"}),
        config=config,
    )
    assert audit_counter == ["T-999"]
```

运行验证：

```bash
python -m pytest -q
```

---

## 总结

| 概念 / API | 核心作用 | 注意事项 |
|---|---|---|
| `interrupt(payload)` | 挂起当前节点，暴露审批载荷给调用者 | 依赖持久化，必须指定 `thread_id`；不可用裸 `try...except` 吞掉 |
| `Command(resume=value)` | 作为 `invoke`/`stream` 的参数恢复图运行 | 传入的 `value` 将作为节点内 `interrupt()` 调用的返回值 |
| **节点重放规则** | 恢复执行时节点从头重新运行 | `interrupt()` 前的逻辑必须具备幂等性；非幂等副作用应抽离至独立前置节点 |

恭喜！你已经完全掌握了 LangGraph 中人在回路的核心交互模式。在下一章中，我们将进入**流式输出（Streaming）**，探索如何在图运行过程中实时把事件、中间状态和 token 展现给用户。

---

## 官方一手参考

- [LangGraph Documentation - Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Documentation - Graph API (Command & Resume)](https://docs.langchain.com/oss/python/langgraph/graph-api#resume)
- [LangGraph Documentation - Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
