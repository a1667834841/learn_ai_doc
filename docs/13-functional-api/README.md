# 第 13 章：Functional API：抛开图仪式感的命令式工作流

> 来源：[Functional API Overview](https://docs.langchain.com/oss/python/langgraph/functional-api) | [Use the Functional API](https://docs.langchain.com/oss/python/langgraph/use-functional-api) | [Choosing APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)  
> 适用版本：`langgraph >= 0.2.20`（本章实测环境：`langgraph 0.6.11`，`python 3.12.12`）  
> 验证环境：macOS Darwin arm64, Python 3.12.12, langgraph 0.6.11, langgraph-checkpoint 2.1.2

在前面的章节中，我们深入掌握了基于状态图的 **Graph API**（`StateGraph`）：声明全局状态 Schema、划分离散节点、编织条件边与循环边，最后编译为图。这套声明式图架构在应对复杂的网状拓扑、多轮状态机和多智能体交互时表现出色。

然而，在实际的**客服工单处理智能体（Support Ticket Agent）**中，很多业务流程本质上是高度线性的：
1. **解析工单**：提取客户问题、判断是否为退款及金额大小；
2. **人工审核**：如果属于高风险大额退款，暂停等待主管审批（人在回路）；
3. **办结执行**：根据审核意见完成退款或答复，生成处理报告。

如果为这样一条清晰的流水线使用 `StateGraph`，你必须遵循一系列“仪式感极强”的套路：
- 定义继承自 `TypedDict` 的全局 `TicketState`，塞入一堆只在某一步临时生效的字段，并处理复杂的 `Optional` 类型；
- 将业务步骤拆碎成多个入参为 `state`、返回更新字典的节点函数；
- 定义专门的条件路由函数，在图构建器中反复调用 `add_node`、`add_edge`、`add_conditional_edges`；
- 必须记住将节点编译后的图暴露给调用方。

很多开发者不禁会问：**“我能不能就像写日常 Python 代码一样，用原生的 `if/else`、函数调用和变量传递来写工作流，同时又享受 LangGraph 的检查点持久化（Checkpointing）和人在回路（HITL）暂停恢复能力呢？”**

答案正是 LangGraph 的 **Functional API**。它通过 `@entrypoint` 和 `@task` 两个核心装饰器，抹平了图的仪式感，带来极其丝滑的命令式开发体验。

---

## 来源契约

在动手编写代码之前，我们先从官方文档提取核心技术契约，作为实现与测试的准绳：

1. **装饰器职责划分**：
   - `@entrypoint(checkpointer=...)`：标记工作流的顶层入口函数，返回可执行的 `Pregel` 对象。它负责管理全局执行生命周期、检查点持久化、人在回路挂起与恢复；对外提供与编译图完全相同的 `invoke`、`stream` 等接口。
   - `@task`：标记离散的子任务（例如调用外部工具、解析文本、数据库读写）。任务执行结果会被持久化保存到检查点中，供工作流重放时直接读取。
2. **单入参规范与序列化**：
   - `@entrypoint` 装饰的函数**必须只接受一个位置参数**作为工作流的输入。如果需要传递多个参数，必须打包为一个 `dict`。
   - 入参、返回值以及 `@task` 的返回值必须是 **JSON 可序列化** 的基本数据类型（字典、列表、字符串、数值、布尔值），以便 Checkpointer 持久化与反序列化。
3. **`@task` 返回 Future-like 对象**：
   - 在同步环境中调用被 `@task` 装饰的函数，它不会直接返回函数计算结果，而是立即返回一个 `Future` 对象（`concurrent.futures.Future` 或 `PregelTask`）。必须显式调用 `.result()` 获取真实计算值；在异步环境中则使用 `await`。
4. **工作流重放（Replay）与任务缓存契约**：
   - 当遇到 `interrupt()` 挂起并在恢复时调用 `workflow.invoke(Command(resume=...))` 时，`@entrypoint` 入口函数会**从第一行重新开始执行（Replay）**。
   - 在重放过程中，之前已经成功执行并持久化的 `@task` **不会重复执行**，而是直接从 Checkpoint 中取出上一次的结果；而未包裹在 `@task` 中的普通 Python 代码或副作用（如 print、直接写文件、未隔离的计数器）会被再次触发。
5. **人在回路（HITL）机制**：
   - 支持在流程任意处调用 `interrupt(payload)` 挂起执行，挂起信息暴露在返回结果的 `__interrupt__` 字段中；
   - 外部通过 `workflow.invoke(Command(resume=value), config=config)` 唤醒工作流；
   - 严禁在 `@task` 或 `@entrypoint` 中使用泛型的 `try ... except Exception:` 包裹 `interrupt()`，因为其底层依托于抛出特定的中断异常 `GraphInterrupt` 实现挂起。

---

## 迭代一：最小命令式工作流（告别图仪式感）

### 1. 先写测试

首先明确期望的可观察行为：输入一个普通的客服咨询工单，工作流能够自动解析意图并执行办结，返回办结报告。

我们期望调用一个普通的 Python 函数式工作流 `support_ticket_workflow.invoke(...)`，传入工单字典，立即得到办结结果。

新建测试用例：

```python
def test_normal_inquiry_flow():
    """测试用例 1: 普通咨询工单无需审核，线性命令式执行到底"""
    workflow = create_ticket_workflow()
    ticket = {"ticket_id": "T-001", "content": "我想咨询软件的使用方法"}

    result = workflow.invoke(ticket)

    assert result["status"] == "completed"
    assert result["action"] == "常规答复"
    assert "工单 [T-001] 办结: 常规答复" in result["detail"]
```

### 2. 运行测试（红灯）

由于我们还没有定义 `create_ticket_workflow`，运行测试时必定报错：

```bash
pytest -q
```

实测输出：

```text
FAILED test_workflow.py::test_normal_inquiry_flow - NameError: name 'create_ticket_workflow' is not defined
```

测试因缺少实现符号变红，这正是我们期望的真实红灯。

### 3. 编写最小实现

我们利用 Functional API 的 `@entrypoint` 和 `@task` 构建最小实现：

```python
from langgraph.func import entrypoint, task


@task
def parse_ticket(ticket: dict) -> dict:
    """子任务：解析工单内容"""
    content = ticket.get("content", "")
    is_refund = "退款" in content
    amount = ticket.get("amount", 0.0)
    return {
        "ticket_id": ticket["ticket_id"],
        "category": "refund" if is_refund else "inquiry",
        "amount": amount,
        "needs_approval": is_refund and amount > 100.0,
    }


@task
def resolve_ticket(ticket_id: str, action: str) -> str:
    """子任务：执行工单办结"""
    return f"工单 [{ticket_id}] 办结: {action}"


def create_ticket_workflow(checkpointer=None):
    @entrypoint(checkpointer=checkpointer)
    def support_ticket_workflow(ticket: dict) -> dict:
        # 注意：task 调用返回 Future 对象，必须调用 .result() 解包获取值
        parsed = parse_ticket(ticket).result()

        action = "常规答复"
        result = resolve_ticket(parsed["ticket_id"], action).result()

        return {
            "ticket_id": parsed["ticket_id"],
            "action": action,
            "status": "completed",
            "detail": result,
        }

    return support_ticket_workflow
```

### 4. 运行测试（绿灯）

再次执行测试：

```bash
pytest -q
```

实测输出：

```text
.                                                                        [100%]
1 passed in 0.28s
```

测试顺利通过！

### 5. 重构与思考：对比 StateGraph

回过头来看，如果用传统的 `StateGraph` 实现上述简单流转，需要写成这样：

```python
# --- 基于 StateGraph 的传统写法 ---
class TicketState(TypedDict):
    ticket_id: str
    content: str
    amount: float
    category: Optional[str]
    action: Optional[str]
    status: Optional[str]
    detail: Optional[str]

def parse_node(state: TicketState):
    ...
    return {"category": "inquiry"}

def resolve_node(state: TicketState):
    ...
    return {"status": "completed", "detail": "..."}

builder = StateGraph(TicketState)
builder.add_node("parse", parse_node)
builder.add_node("resolve", resolve_node)
builder.add_edge(START, "parse")
builder.add_edge("parse", "resolve")
builder.add_edge("resolve", END)
graph = builder.compile()
```

对比两者：
1. **没有状态 Schema 仪式感**：Functional API 中不需要声明 `TypedDict` 或庞大的数据总线，变量只在需要的作用域内传递；
2. **直观的命令式流程**：就像阅读标准 Python 函数一样：先执行 A，再得到结果，然后执行 B；
3. **保持了相同的可调用能力**：包装后的 `support_ticket_workflow` 依然是一个包含 `invoke`、`stream`、`batch` 完整能力的执行器。

---

## 迭代二：人在回路——大额退款工单的暂停与恢复

### 1. 先写测试

在客服实际场景中，并非所有工单都能自动处理。如果客户申请的是**超过 100 元的大额退款**，系统必须：
1. 主动暂停（挂起），向后台审核员暴露工单 ID、申请金额与待审核原因；
2. 人工审核员审阅后，将决断结果（批准或驳回）通过 `Command(resume=...)` 送回；
3. 工作流从挂起点继续向下运行，执行退款或驳回，完成最终办结。

我们编写针对人在回路生命周期的测试：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def test_refund_interrupt_and_resume():
    """测试用例 2: 大额退款遇到 interrupt 挂起，通过 Command(resume=...) 唤醒并办结"""
    checkpointer = InMemorySaver()
    workflow = create_ticket_workflow(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-refund-101"}}
    ticket = {"ticket_id": "T-101", "content": "申请退款，机器故障", "amount": 699.0}

    # 1. 第一次调用：应当在 interrupt 处挂起，返回 __interrupt__ 信息
    init_res = workflow.invoke(ticket, config=config)

    assert "__interrupt__" in init_res
    assert len(init_res["__interrupt__"]) == 1
    interrupt_payload = init_res["__interrupt__"][0].value
    assert interrupt_payload["ticket_id"] == "T-101"
    assert interrupt_payload["amount"] == 699.0
    assert "大额退款" in interrupt_payload["reason"]

    # 2. 第二次调用：人工介入，传入批准决策
    resumed_res = workflow.invoke(Command(resume={"approved": True}), config=config)

    assert resumed_res["status"] == "completed"
    assert resumed_res["action"] == "同意退款 699.0 元"
    assert "工单 [T-101] 办结: 同意退款 699.0 元" in resumed_res["detail"]
```

### 2. 运行测试（红灯）

运行测试：

```bash
pytest -q
```

实测输出：

```text
FAILED test_workflow.py::test_refund_interrupt_and_resume - AssertionError: assert '__interrupt__' in {'ticket_id': 'T-101', 'action': '常规答复', 'status': 'completed', 'detail': '工单 [T-101] 办结: 常规答复'}
```

失败原因非常明确：我们的工作流目前没有检查 `needs_approval`，也没有触发 `interrupt`，大额退款被当成普通咨询直接跑到底了！

### 3. 编写最小实现

在 Functional API 中，接入人在回路非常符合直觉：直接在函数体中书写 `if` 判断，并在需要停下的地方调用 `interrupt(payload)`。它的返回值就是未来外部 resume 时注入的内容：

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt


def create_ticket_workflow(checkpointer=None):
    @entrypoint(checkpointer=checkpointer)
    def support_ticket_workflow(ticket: dict) -> dict:
        parsed = parse_ticket(ticket).result()
        amount = parsed["amount"]

        action = "常规答复"

        # 自然的 Python if-else 控制流
        if parsed["needs_approval"]:
            # 挂起流程，等待人工审批
            decision = interrupt(
                {
                    "ticket_id": parsed["ticket_id"],
                    "amount": amount,
                    "reason": "大额退款需要主管审批",
                }
            )

            # 恢复后，decision 即为外部 Command(resume=...) 传入的值
            if decision.get("approved"):
                action = f"同意退款 {amount} 元"
            else:
                reason = decision.get("reject_reason", "不符合退款条件")
                action = f"驳回退款: {reason}"
        elif parsed["category"] == "refund":
            action = f"小额自动退款 {amount} 元"

        result = resolve_ticket(parsed["ticket_id"], action).result()

        return {
            "ticket_id": parsed["ticket_id"],
            "action": action,
            "status": "completed",
            "detail": result,
        }

    return support_ticket_workflow
```

### 4. 运行测试（绿灯）

再次执行测试：

```bash
pytest -q
```

实测输出：

```text
..                                                                       [100%]
2 passed in 0.31s
```

测试全部变绿！

没有定义条件边路由函数，没有声明 `human_approval` 节点，只是在一个普通的 Python 函数中写了一个 `if` 和一个 `interrupt()`，人在回路就完整建立并持久化了。

---

## 迭代三：任务重放保护与副作用幂等性（深挖 `@task` 机制）

此时你可能会产生一个强烈的疑问：
> “既然 `@entrypoint` 里面写普通 Python 代码这么方便，为什么官方还要提供 `@task`？我直接写普通的 Python 辅助函数不行吗？”

这是一个直击 Functional API 底层核心的问题。

### 1. 探究 Replay 执行模型

官方文档明确指出：**当工作流被唤醒（Resume）时，`@entrypoint` 函数会从第一行重新执行（Replay）！**

如果你的工单处理中包含外部 API 调用（如扣款、调用大模型、发送短信），或者计算开销巨大的逻辑：
- 如果它们只是普通函数，当工作流从 `interrupt` 恢复时，这些代码**会被重复执行一遍**！
- 如果用 `@task` 包装，LangGraph 会在 Checkpoint 中记录该 Task 的调用指纹与执行结果。当工作流重放时，LangGraph 检查到该任务在当前线程上已经完成，**直接从 Checkpoint 读取缓存结果，跳过函数实际执行**！

### 2. 先写测试证明这一契约

我们引入调用计数器，设计测试来观察以下行为：
1. 首次触发大额退款挂起：`parse_ticket` 被调用 1 次，`resolve_ticket` 尚未被调用（0 次）；
2. 传入 `Command(resume=...)` 恢复：
   - `@entrypoint` 从头重新执行；
   - 被 `@task` 保护的 `parse_ticket` **调用次数保持为 1（直接从缓存恢复）**；
   - 后续的 `resolve_ticket` 此时才被执行（调用次数变为 1）；
   - 如果存在**未被 `@task` 保护的副作用**，它的调用计数会变成 2！

编写测试：

```python
# 用于追踪函数执行次数的全局计数器
call_counters = {
    "parse": 0,
    "resolve": 0,
    "unprotected_side_effect": 0,
}


@task
def parse_ticket(ticket: dict) -> dict:
    call_counters["parse"] += 1
    content = ticket.get("content", "")
    is_refund = "退款" in content
    amount = ticket.get("amount", 0.0)
    return {
        "ticket_id": ticket["ticket_id"],
        "category": "refund" if is_refund else "inquiry",
        "amount": amount,
        "needs_approval": is_refund and amount > 100.0,
    }


@task
def resolve_ticket(ticket_id: str, action: str) -> str:
    call_counters["resolve"] += 1
    return f"工单 [{ticket_id}] 办结: {action}"


def test_task_checkpointing_and_replay_protection():
    """测试用例 3: 证明 @task 提供了检查点缓存，防止重放时重复执行副作用"""
    call_counters["parse"] = 0
    call_counters["resolve"] = 0
    call_counters["unprotected_side_effect"] = 0

    checkpointer = InMemorySaver()
    workflow = create_ticket_workflow(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-replay-202"}}
    ticket = {"ticket_id": "T-202", "content": "申请退款服务费", "amount": 500.0}

    # 阶段 1: 首次调用，在 interrupt 处暂停
    workflow.invoke(ticket, config=config)
    assert call_counters["parse"] == 1
    assert call_counters["resolve"] == 0
    assert call_counters["unprotected_side_effect"] == 1

    # 阶段 2: resume 恢复执行
    workflow.invoke(
        Command(resume={"approved": False, "reject_reason": "超过质保期"}),
        config=config,
    )

    # 关键断言：
    # 1. parse_ticket 受到 @task 保护，重放时直接从 checkpoint 取值，没有再次执行！
    assert call_counters["parse"] == 1

    # 2. resolve_ticket 在 resume 之后才真正被调用 1 次
    assert call_counters["resolve"] == 1

    # 3. 未被 @task 保护的代码随着 entrypoint 重放被重复调用了 2 次！
    assert call_counters["unprotected_side_effect"] == 2
```

### 3. 在实现中增加未保护代码进行对照

在 `support_ticket_workflow` 内部增加一行未受保护的副作用计数：

```python
def create_ticket_workflow(checkpointer=None):
    @entrypoint(checkpointer=checkpointer)
    def support_ticket_workflow(ticket: dict) -> dict:
        # 故意记录一次未被 @task 包装的代码执行
        call_counters["unprotected_side_effect"] += 1

        parsed = parse_ticket(ticket).result()
        amount = parsed["amount"]
        ...
```

### 4. 运行测试（绿灯通过）

运行 pytest：

```bash
pytest -q
```

实测输出：

```text
...                                                                      [100%]
3 passed in 0.32s
```

通过这组断言，我们用硬核的测试数据彻底摸清了 `@task` 与 `@entrypoint` 的协作秘密：
- `@entrypoint` 负责宏观的恢复重放（Replay）；
- `@task` 负责微观的任务级状态持久化（Task Checkpointing）；
- 任何有副作用的调用（如扣款、调 LLM、发通知）都**必须封入 `@task`**，否则人在回路恢复时会产生重复调用的灾难！

---

## 架构选型权衡：Graph API vs Functional API

在实际生产项目中，两种 API 各自该在什么时候登场？官方提供了清晰的指引，下表总结了两者的核心权衡：

| 评估维度 | Graph API (`StateGraph`) | Functional API (`@entrypoint` + `@task`) |
|---|---|---|
| **代码范式** | 声明式（Declarative），节点与边拓扑构图 | 命令式（Imperative），原生 Python 控制流 |
| **状态管理** | 全局共享 State，支持 `Annotated` 与 Reducer 增量合并 | 局部变量传递，函数内作用域，无显式 Schema |
| **检查点粒度** | 每个 Superstep（步骤批次）生成检查点快照 | 入口处持久化，Task 粒度保存结果并在重放时复用 |
| **分支与循环** | 条件边（`add_conditional_edges`）、图递归环路 | Python 原生 `if/else`、`for/while`、递归函数 |
| **并发支持** | 分支 Fan-out 自动并行，多节点并发合并 | 原生 Future 模式：`[task_fn(x) for x in list]` 并发执行 |
| **可视化支持** | 原生支持：直接输出 Mermaid 流程图，无缝集成 Studio 调试 | 动态运行时构图，不支持静态图可视化导出 |
| **最佳适用场景** | 复杂网状拓扑、循环状态机、多智能体协同、团队协作需架构图 | 线性流水线、简单决策分支、现有过程式代码改造、快速原型 |

### 混用模式：两者可以无缝协同

两者并不是非此即彼的对立关系。因为它们的底层都基于同一个执行引擎（`Pregel`），你可以在项目中自由混用：

1. **在 Graph API 节点中调用 Functional API**：
   ```python
   @entrypoint()
   def clean_data_pipeline(raw_data: dict) -> dict:
       ...
       return cleaned

   def data_process_node(state: AgentState):
       # 直接在普通节点中调用 Functional API 工作流
       cleaned = clean_data_pipeline.invoke(state["raw"])
       return {"cleaned": cleaned}
   ```

2. **在 Functional API 中调用编译好的 StateGraph**：
   ```python
   subgraph = builder.compile()

   @entrypoint(checkpointer=checkpointer)
   def main_workflow(user_input: dict):
       # 用命令式流程编排子图
       graph_res = subgraph.invoke(user_input)
       return {"final": graph_res}
   ```

---

## 避坑指南（Gotchas）

### 1. Future-like 陷阱：忘了写 `.result()`

`@task` 修饰的函数在调用时立即返回一个 `Future` 对象。以下是极其隐蔽的 Bug 写法：

```python
# ❌ 错误写法：直接在 if 中判断 task 返回值
is_valid_future = validate_ticket(ticket)
if is_valid_future:  # Future 对象永远是 True！即使里面包裹的是 False！
    process()

# ❌ 错误写法：直接把 Future 当字典下发索引
ticket_info = parse_ticket(ticket)
print(ticket_info["ticket_id"])  # 运行时报错：TypeError: 'Future' object is not subscriptable

# ✅ 正确写法：显式调用 .result() 或使用 await
ticket_info = parse_ticket(ticket).result()
print(ticket_info["ticket_id"])
```

### 2. 宽泛异常捕获破坏人在回路

`interrupt()` 的底层实现是抛出一个特殊的 `GraphInterrupt` 异常，从而让执行引擎捕获并生成检查点：

```python
# ❌ 致命错误：用通用 Exception 吞掉了 interrupt
@entrypoint(checkpointer=checkpointer)
def bad_workflow(ticket: dict):
    try:
        decision = interrupt("需要审批吗？")
    except Exception as e:
        decision = "默认通过"  # GraphInterrupt 被直接当作普通错误吞掉，流程无法挂起！
    return decision

# ✅ 正确做法：不要用宽泛的 try...except 包裹 interrupt，或者显式放行 BaseException
```

### 3. 未被 `@task` 保护的非确定性逻辑与副作用

由于工作流恢复时入口函数会从头重放，**任何具有随机性、时间依赖或副作用的操作都必须放在 `@task` 内部**：

```python
# ❌ 错误写法：时间在重放时会改变，导致分支逻辑不一致
@entrypoint(checkpointer=checkpointer)
def flow(ticket: dict):
    now = time.time()  # 每次重放 now 都在变！
    if now - ticket["created_at"] > 3600:
        ans = interrupt("超时审批")

# ✅ 正确写法：包裹在 @task 中，第一次执行后结果被固定保存在检查点
@task
def get_current_time() -> float:
    return time.time()
```

### 4. `@entrypoint` 的单入参规范与序列化约束

- 入口函数必须接收单个位置参数，如 `def workflow(inputs: dict):`；
- 入参、返回值及任务返回值必须是 JSON 可序列化的类型。切勿直接在任务之间返回数据库游标连接或复杂的未序列化类实例。

---

## 完整代码清单

### 业务实现代码：`workflow.py`

```python
"""
Support Ticket Workflow - Functional API Implementation
基于 LangGraph Functional API (@entrypoint 与 @task) 构建的客服工单工作流
"""

from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.types import interrupt

# 模块级全局计数器（便于测试可观察性与重放验证）
call_counters = {
    "parse": 0,
    "resolve": 0,
    "unprotected_side_effect": 0,
}


@task
def parse_ticket(ticket: dict) -> dict:
    """子任务：解析工单内容并判断是否需要人工审批"""
    call_counters["parse"] += 1
    content = ticket.get("content", "")
    is_refund = "退款" in content
    amount = float(ticket.get("amount", 0.0))
    return {
        "ticket_id": ticket["ticket_id"],
        "category": "refund" if is_refund else "inquiry",
        "amount": amount,
        "needs_approval": is_refund and amount > 100.0,
    }


@task
def resolve_ticket(ticket_id: str, action: str) -> str:
    """子任务：执行工单办结处理"""
    call_counters["resolve"] += 1
    return f"工单 [{ticket_id}] 办结: {action}"


def create_ticket_workflow(checkpointer=None):
    """构建工单处理的命令式函数工作流"""

    @entrypoint(checkpointer=checkpointer)
    def support_ticket_workflow(ticket: dict) -> dict:
        # 追踪未经 @task 保护的代码执行（用于演示重放副作用）
        call_counters["unprotected_side_effect"] += 1

        # 1. 调用任务，获得 Future 并同步提取结果
        parsed = parse_ticket(ticket).result()
        amount = parsed["amount"]

        # 2. 自然的 Python 分支控制流
        action = "常规答复"
        if parsed["needs_approval"]:
            # 3. 人在回路挂起：向外部抛出审批详情
            decision = interrupt(
                {
                    "ticket_id": parsed["ticket_id"],
                    "amount": amount,
                    "reason": "大额退款需要主管审批",
                }
            )

            # 外部恢复唤醒后继续执行
            if decision.get("approved"):
                action = f"同意退款 {amount} 元"
            else:
                reason = decision.get("reject_reason", "不符合退款条件")
                action = f"驳回退款: {reason}"
        elif parsed["category"] == "refund":
            action = f"小额自动退款 {amount} 元"

        # 4. 执行办结任务
        result = resolve_ticket(parsed["ticket_id"], action).result()

        return {
            "ticket_id": parsed["ticket_id"],
            "action": action,
            "status": "completed",
            "detail": result,
        }

    return support_ticket_workflow
```

### 测试验证代码：`test_workflow.py`

```python
"""
Support Ticket Workflow - TDD 测试套件
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from workflow import create_ticket_workflow, call_counters


def test_normal_inquiry_flow():
    """测试用例 1: 普通咨询无需审核，线性命令式执行到底"""
    workflow = create_ticket_workflow()
    ticket = {"ticket_id": "T-001", "content": "我想咨询软件的使用方法"}

    result = workflow.invoke(ticket)

    assert result["status"] == "completed"
    assert result["action"] == "常规答复"
    assert "工单 [T-001] 办结: 常规答复" in result["detail"]


def test_refund_interrupt_and_resume():
    """测试用例 2: 大额退款触发挂起，传入 Command(resume=...) 恢复"""
    checkpointer = InMemorySaver()
    workflow = create_ticket_workflow(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-refund-101"}}
    ticket = {"ticket_id": "T-101", "content": "申请退款，机器故障", "amount": 699.0}

    # 阶段 1: 首次调用，应在 interrupt 处挂起
    init_res = workflow.invoke(ticket, config=config)
    assert "__interrupt__" in init_res
    assert len(init_res["__interrupt__"]) == 1

    payload = init_res["__interrupt__"][0].value
    assert payload["ticket_id"] == "T-101"
    assert payload["amount"] == 699.0
    assert "大额退款" in payload["reason"]

    # 阶段 2: 外部人工审批同意后唤醒
    resumed_res = workflow.invoke(Command(resume={"approved": True}), config=config)

    assert resumed_res["status"] == "completed"
    assert resumed_res["action"] == "同意退款 699.0 元"
    assert "工单 [T-101] 办结: 同意退款 699.0 元" in resumed_res["detail"]


def test_task_checkpointing_and_replay_protection():
    """测试用例 3: 证明 @task 提供了检查点缓存，防止重放时重复执行副作用"""
    call_counters["parse"] = 0
    call_counters["resolve"] = 0
    call_counters["unprotected_side_effect"] = 0

    checkpointer = InMemorySaver()
    workflow = create_ticket_workflow(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-replay-202"}}
    ticket = {"ticket_id": "T-202", "content": "申请退款服务费", "amount": 500.0}

    # 首次触发挂起
    workflow.invoke(ticket, config=config)
    assert call_counters["parse"] == 1
    assert call_counters["resolve"] == 0
    assert call_counters["unprotected_side_effect"] == 1

    # 恢复执行至结束
    workflow.invoke(
        Command(resume={"approved": False, "reject_reason": "超时申请"}),
        config=config,
    )

    # 验证：受保护的 parse_ticket 仅执行 1 次；未受保护代码执行了 2 次
    assert call_counters["parse"] == 1
    assert call_counters["resolve"] == 1
    assert call_counters["unprotected_side_effect"] == 2
```

### 运行验证与实测输出

在激活虚拟环境的终端中运行 pytest：

```bash
source .venv/bin/activate && pytest -v
```

实测输出：

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.12, pytest-8.4.1, pluggy-1.6.0
rootdir: /Users/wuwenjing/codes/nodes/demo/langgraph-learning
collected 3 items

test_workflow.py::test_normal_inquiry_flow PASSED                        [ 33%]
test_workflow.py::test_refund_interrupt_and_resume PASSED               [ 66%]
test_workflow.py::test_task_checkpointing_and_replay_protection PASSED  [100%]

============================== 3 passed in 0.32s ===============================
```

所有测试 100% 真实执行通过！

---

## 收工总结

通过本章的测试驱动探索，我们全面掌握了 LangGraph Functional API 的核心价值：

1. **去仪式感的轻量编程**：
   - `@entrypoint` 将日常 Python 函数直接升级为持久化工作流，免去了声明 TypedDict 状态总线、拆解节点和绑定边的构图成本；
   - 可以直接使用 Python 自带的 `if/else`、`for`、三元表达式等基础控制流表达业务决策。
2. **`@task` 的设计智慧**：
   - 不仅仅是一个装饰器，而是提供**重放缓存保护（Task Checkpointing）**、**并发支持（Future-like）**与**重试机制**的核心单元；
   - 恢复执行时入口函数从头重放，已完成的 `@task` 命中检查点缓存直接返回，杜绝副作用重复触发。
3. **架构选型心智模型**：
   - **选 Functional API**：线性管道、现有业务流程轻量嵌入持久化、简单分支、无需 Studio 绘图的场景；
   - **选 Graph API**：多智能体循环对话、网状复杂状态机、跨节点增量 Reducer 合并、强依赖可视化排查的场景。

---

## 来源与一手参考

- [Functional API Overview - LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [Use the Functional API - LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/use-functional-api)
- [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)
- [Interrupts and Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts)
