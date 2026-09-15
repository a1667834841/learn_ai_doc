# 第 06 章：持久化：让对话记得住

在前几章中，我们构建的图无论逻辑多么复杂，都有一个共同的特点：**每一次 `invoke()` 都是一次性的**。一旦函数执行完毕返回结果，图的内存状态就彻底消失了。

但在实际的**客服工单处理智能体（Support Ticket Agent）**中，用户的交互几乎都是多轮的：
- 用户第 1 轮说：“我想查询退款进度，工单号是 T-1001。” 智能体提取了工单号并进行了登记。
- 用户第 2 轮追问：“距离申请已经过去三天了，到底到账没有？”
- 用户在第 2 轮对话中**根本不会重复提供工单号**。如果智能体是“无状态”的，它就会瞬间“失忆”，甚至由于找不到工单 ID 而直接报错。

为了让智能体拥有记忆、能够持续跟进会话，LangGraph 引入了核心能力——**持久化层（Persistence）与检查点（Checkpointer）**。

通过本章，你将掌握：
1. **Checkpointer**：如何在图编译期注入状态检查点保存器；
2. **`thread_id`**：如何使用线程 ID 作为会话的唯一游标，实现多轮记忆与多租户隔离；
3. **`get_state` 与 `get_state_history`**：如何透视图的内部快照与检查点历史链条。

---

## 来源契约

本章内容严格基于 LangGraph 官方文档与 API 规范：
- **官方文档**：[Persistence | LangGraph](https://docs.langchain.com/oss/python/langgraph/persistence) 与 [Checkpointers | LangGraph](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- **核心契约**：
  1. **Checkpointer**：在图的每个 **super-step** 边界，自动保存当前图状态的快照（Checkpoint）。开发和测试阶段使用标准库内置的 `InMemorySaver`，生产环境可替换为 `SqliteSaver` 或 `PostgresSaver`。
  2. **`thread_id`**：Checkpointer 的主键与持久化指针。调用图时必须在 `config={"configurable": {"thread_id": ...}}` 中指定。使用相同的 `thread_id` 会继承上一轮的状态；使用不同的 `thread_id` 则状态严格隔离。
  3. **`graph.get_state(config)`**：返回指定线程最新的 `StateSnapshot` 对象，包含 `values`（当前状态）、`next`（下一步待执行节点，空元组 `()` 表示执行完毕）、`config`、`metadata` 等。
  4. **`graph.get_state_history(config)`**：返回指定线程的历史快照迭代器，按时间倒序排列。
  5. 状态保留与更新规则：新一轮调用传入的字典作为状态增量更新；未提供的字段保留历史值，有 Reducer（如 `operator.add`）修饰的字段按 Reducer 规则追加，普通字段若提供则覆盖。

---

## 迭代一：多轮对话中的“失忆”危机

### 1. 先写测试

我们希望工单智能体具备记忆能力：用户在第 1 轮提供工单号与第一句话，第 2 轮仅发送补充消息，智能体能够自动保留第一轮的工单号，并将新消息追加到消息列表中。

在 `docs/06-persistence` 目录下新建 `test_graph.py`：

```python
from graph import create_ticket_graph


def test_multi_turn_conversation_retains_memory():
    # 暂时不传入 checkpointer，创建朴素图
    graph = create_ticket_graph()
    config = {"configurable": {"thread_id": "thread-alice-101"}}

    # 第 1 轮交互：创建工单并提问
    round_1 = graph.invoke(
        {
            "ticket_id": "T-1001",
            "customer_id": "C-888",
            "messages": ["我想查询退款进度"],
            "status": "new",
        },
        config=config,
    )
    assert round_1["ticket_id"] == "T-1001"
    assert round_1["messages"] == ["我想查询退款进度"]

    # 第 2 轮交互：用户仅追加消息，不再重复提供 ticket_id 和 customer_id
    round_2 = graph.invoke(
        {"messages": ["距离申请已经过去三天了"]},
        config=config,
    )

    # 核心期望：智能体必须记得历史工单 ID，且消息列表累加
    assert "ticket_id" in round_2
    assert round_2["ticket_id"] == "T-1001"
    assert round_2["customer_id"] == "C-888"
    assert round_2["messages"] == ["我想查询退款进度", "距离申请已经过去三天了"]
```

### 2. 运行测试（红灯）

创建 `graph.py` 提供最基础的无状态图：

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END


class TicketState(TypedDict):
    ticket_id: str
    customer_id: str
    messages: Annotated[list[str], operator.add]
    status: str


def process_ticket_node(state: TicketState):
    return {"status": "in_progress"}


def create_ticket_graph(checkpointer=None):
    builder = StateGraph(TicketState)
    builder.add_node("process_ticket", process_ticket_node)
    builder.add_edge(START, "process_ticket")
    builder.add_edge("process_ticket", END)

    # 此时尚未接入 checkpointer
    return builder.compile(checkpointer=checkpointer)
```

运行 pytest：

```bash
python -m pytest -q
```

实测输出：

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_________________ test_multi_turn_conversation_retains_memory __________________
...
E       AssertionError: assert 'ticket_id' in {'messages': ['距离申请已经过去三天了'], 'status': 'in_progress'}
```

失败了！仔细看实测输出：
`round_2` 返回的字典只有 `{'messages': ['距离申请已经过去三天了'], 'status': 'in_progress'}`。
- `ticket_id` 彻底丢了；
- `customer_id` 彻底丢了；
- `messages` 也只剩下了第二轮的单条消息，第一轮的沟通记录化为乌有。

因为没有持久化层，即便传入了 `thread_id`，图也只是在每次调用时新建一个瞬时状态，执行完毕后直接丢弃。

### 3. 编写最小实现（引入 Checkpointer）

让测试通过只需要两步：
1. 引入内存检查点保存器：`from langgraph.checkpoint.memory import InMemorySaver`；
2. 在编译图时将 `checkpointer` 传给 `builder.compile(checkpointer=checkpointer)`。

修改 `test_graph.py` 中的测试调用，传入 `InMemorySaver`：

```python
from langgraph.checkpoint.memory import InMemorySaver
from graph import create_ticket_graph


def test_multi_turn_conversation_retains_memory():
    # 核心改动：为图配备检查点保存器
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-alice-101"}}

    # 第 1 轮交互
    round_1 = graph.invoke(
        {
            "ticket_id": "T-1001",
            "customer_id": "C-888",
            "messages": ["我想查询退款进度"],
            "status": "new",
        },
        config=config,
    )
    assert round_1["ticket_id"] == "T-1001"
    assert round_1["messages"] == ["我想查询退款进度"]

    # 第 2 轮交互：仅追加消息
    round_2 = graph.invoke(
        {"messages": ["距离申请已经过去三天了"]},
        config=config,
    )

    # 断言记忆保留
    assert "ticket_id" in round_2
    assert round_2["ticket_id"] == "T-1001"
    assert round_2["customer_id"] == "C-888"
    assert round_2["messages"] == ["我想查询退款进度", "距离申请已经过去三天了"]
    assert round_2["status"] == "in_progress"
```

同时，我们检查 `graph.py` 中的 `create_ticket_graph`：它已经支持接收 `checkpointer` 参数并传递给 `builder.compile(checkpointer=checkpointer)`。

### 4. 运行测试（绿灯）

再次执行测试：

```bash
python -m pytest -q
```

实测输出：

```text
.                                                                        [100%]
1 passed in 0.14s
```

绿灯通过！
为什么仅仅加上 `checkpointer` 和相同的 `thread_id`，智能体就记住了数据？
- 当第 1 轮运行结束时，`InMemorySaver` 将最终状态（包含 `ticket_id`、`customer_id`、`messages`、`status`）以 `thread-alice-101` 为键存入检查点；
- 当第 2 轮调用时，LangGraph 根据 `thread-alice-101` 自动从 Checkpointer 中加载上一轮的历史快照作为底模，然后将第 2 轮的增量输入应用到状态中；
- `messages` 字段被 `Annotated[list[str], operator.add]` 修饰，因此新消息被追加到旧列表末尾；未声明增量修改的 `ticket_id` 和 `customer_id` 则原样保留。

---

## 迭代二：会话隔离（多用户与多线程安全）

客服系统同时会有成千上万个客户在咨询。Alice 的工单和 Bob 的工单必须完全独立，绝对不能发生串号或串消息。

### 1. 先写测试

在 `test_graph.py` 中增加会话隔离测试：

```python
def test_threads_are_isolated():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)

    alice_config = {"configurable": {"thread_id": "thread-alice"}}
    bob_config = {"configurable": {"thread_id": "thread-bob"}}

    # Alice 发起工单咨询
    graph.invoke(
        {
            "ticket_id": "T-ALICE",
            "customer_id": "C-1",
            "messages": ["Alice 的退款请求"],
            "status": "new",
        },
        config=alice_config,
    )

    # Bob 发起工单咨询
    graph.invoke(
        {
            "ticket_id": "T-BOB",
            "customer_id": "C-2",
            "messages": ["Bob 的登录异常"],
            "status": "new",
        },
        config=bob_config,
    )

    # 分别调用 get_state 检查状态
    alice_state = graph.get_state(alice_config)
    bob_state = graph.get_state(bob_config)

    # 验证 Alice 的状态未受 Bob 影响
    assert alice_state.values["ticket_id"] == "T-ALICE"
    assert alice_state.values["customer_id"] == "C-1"
    assert alice_state.values["messages"] == ["Alice 的退款请求"]

    # 验证 Bob 的状态未受 Alice 影响
    assert bob_state.values["ticket_id"] == "T-BOB"
    assert bob_state.values["customer_id"] == "C-2"
    assert bob_state.values["messages"] == ["Bob 的登录异常"]
```

### 2. 运行测试（绿灯）

运行测试：

```bash
python -m pytest -q
```

实测输出：

```text
..                                                                       [100%]
2 passed in 0.14s
```

通过！
这展示了 `thread_id` 的本质定位：**它就是持久化存储的主键索引（Primary Key）**。不同 `thread_id` 的状态完全独立存储，天然互不干扰。

---

## 迭代三：透视内部状态与检查点历史

在系统运维、排查故障或外部审计时，我们经常需要在不触发图执行的前提下，回答这两个问题：
1. **当前图执行到哪一步了？最新的状态数据是什么？**
2. **这个会话经历了哪些演变历史（时间旅行）？**

LangGraph 提供了 `graph.get_state()` 和 `graph.get_state_history()` 两个强大的原语。

### 1. 先写测试

在 `test_graph.py` 中编写针对状态快照与历史链条的测试：

```python
def test_inspect_state_and_history():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-inspect"}}

    # 执行一次调用
    graph.invoke(
        {
            "ticket_id": "T-2002",
            "customer_id": "C-9",
            "messages": ["工单已提交"],
            "status": "new",
        },
        config=config,
    )

    # 1. 检查最新快照 get_state
    snapshot = graph.get_state(config)

    # 状态数据
    assert snapshot.values["ticket_id"] == "T-2002"
    assert snapshot.values["status"] == "in_progress"

    # 下一步执行节点：图已执行完毕，next 为空元组 ()
    assert snapshot.next == ()

    # 配置信息中包含 thread_id 和系统生成的 checkpoint_id
    assert snapshot.config["configurable"]["thread_id"] == "thread-inspect"
    assert "checkpoint_id" in snapshot.config["configurable"]

    # 2. 检查历史链条 get_state_history
    history = list(graph.get_state_history(config))

    # 一次完整的执行会在每个 super-step 边界落盘，因此包含多个历史检查点
    assert len(history) > 1

    # 检查点按时间倒序排列：history[0] 就是最新快照
    latest_checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    assert (
        history[0].config["configurable"]["checkpoint_id"]
        == latest_checkpoint_id
    )
```

### 2. 运行测试（绿灯）

运行测试：

```bash
python -m pytest -q
```

实测输出：

```text
...                                                                      [100%]
3 passed in 0.14s
```

### 3. 解剖 `StateSnapshot` 结构

让我们仔细看 `get_state()` 返回的 `StateSnapshot`，它是一个核心数据结构，包含以下关键字段：

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `values` | `dict` | 当前检查点所有通道的状态数据 | `{'ticket_id': 'T-2002', 'messages': [...]}` |
| `next` | `tuple[str, ...]` | 下一个将要执行的节点名称；若已到达终点则为 `()` | `()` 或 `('human_approval',)` |
| `config` | `dict` | 包含 `thread_id`、`checkpoint_ns`、`checkpoint_id` | `{'configurable': {'thread_id': '...', ...}}` |
| `metadata` | `dict` | 执行元数据，包含触发源 `source`、当前步骤 `step` 等 | `{'source': 'loop', 'step': 1, ...}` |
| `created_at`| `str` | ISO 8601 格式的时间戳 | `'2026-09-13T10:00:00.000000+00:00'` |
| `tasks` | `tuple` | 当前 step 下待执行的任务元组（含任务 ID、中断信息等）| `()` |

在后续第 07 章《人在回路》中，当图被挂起时，我们正是通过 `snapshot.next` 来观察到图阻塞在哪个审核节点；而在后续的《时间旅行》中，也是通过 `snapshot.config["configurable"]["checkpoint_id"]` 回溯到历史某一刻重新分叉执行。

---

## 避坑指南（Gotchas）

### 1. `InMemorySaver` 仅存在于进程内存中
`InMemorySaver` 极其快速且轻量，非常适合自动化测试和本地实验。但一旦 Python 进程退出或服务重启，内存中的所有检查点将全部消失。
> **生产建议**：在生产部署时，应根据基础设施替换为官方提供的持久化 Checkpointer：
> - `langgraph-checkpoint-sqlite`：基于本地 SQLite 文件存储，适合单节点或桌面端应用；
> - `langgraph-checkpoint-postgres`：基于 PostgreSQL 的分布式存储，支持高并发与异步操作（`AsyncPostgresSaver`）。

### 2. 调用已配置 Checkpointer 的图时忘记传 `thread_id`
如果编译时提供了 `checkpointer`，但在调用时写成了：
```python
# ❌ 错误示范：未提供 thread_id
graph.invoke({"messages": ["hello"]})
```
Checkpointer 将无法得知该把状态存到哪张会话表中，会导致该次调用无法正确读取历史，也无法供后续轮次复用。
> **规则**：凡是启用了持久化的图，每次调用务必显式传入 `config={"configurable": {"thread_id": "your-unique-id"}}`。

### 3. Reducer 是多轮记忆累加的关键
如果状态定义中写的是：
```python
class TicketState(TypedDict):
    messages: list[str]  # ❌ 没有 Annotated 与 Reducer
```
当第 2 轮调用传入 `{"messages": ["新消息"]}` 时，由于没有定义合并规则，后一次的值会直接**覆盖**前一次的值，历史消息依然会丢失！
只有使用 `Annotated[list[str], operator.add]`（或用于消息体系的 `add_messages`），LangGraph 才知道新输入的列表应该追加到历史列表末尾。

---

## 完整代码清单

### `graph.py`

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END


class TicketState(TypedDict):
    ticket_id: str
    customer_id: str
    messages: Annotated[list[str], operator.add]
    status: str


def process_ticket_node(state: TicketState):
    # 只要处理了工单，就将状态流转为 in_progress
    return {"status": "in_progress"}


def create_ticket_graph(checkpointer=None):
    builder = StateGraph(TicketState)
    builder.add_node("process_ticket", process_ticket_node)

    builder.add_edge(START, "process_ticket")
    builder.add_edge("process_ticket", END)

    return builder.compile(checkpointer=checkpointer)
```

### `test_graph.py`

```python
from langgraph.checkpoint.memory import InMemorySaver
from graph import create_ticket_graph


def test_multi_turn_conversation_retains_memory():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-alice-101"}}

    # 第 1 轮交互
    round_1 = graph.invoke(
        {
            "ticket_id": "T-1001",
            "customer_id": "C-888",
            "messages": ["我想查询退款进度"],
            "status": "new",
        },
        config=config,
    )
    assert round_1["ticket_id"] == "T-1001"
    assert round_1["messages"] == ["我想查询退款进度"]

    # 第 2 轮交互：用户仅追加消息
    round_2 = graph.invoke(
        {"messages": ["距离申请已经过去三天了"]},
        config=config,
    )

    assert "ticket_id" in round_2
    assert round_2["ticket_id"] == "T-1001"
    assert round_2["customer_id"] == "C-888"
    assert round_2["messages"] == ["我想查询退款进度", "距离申请已经过去三天了"]
    assert round_2["status"] == "in_progress"


def test_threads_are_isolated():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)

    alice_config = {"configurable": {"thread_id": "thread-alice"}}
    bob_config = {"configurable": {"thread_id": "thread-bob"}}

    graph.invoke(
        {
            "ticket_id": "T-ALICE",
            "customer_id": "C-1",
            "messages": ["Alice 的退款请求"],
            "status": "new",
        },
        config=alice_config,
    )

    graph.invoke(
        {
            "ticket_id": "T-BOB",
            "customer_id": "C-2",
            "messages": ["Bob 的登录异常"],
            "status": "new",
        },
        config=bob_config,
    )

    alice_state = graph.get_state(alice_config)
    bob_state = graph.get_state(bob_config)

    assert alice_state.values["ticket_id"] == "T-ALICE"
    assert alice_state.values["messages"] == ["Alice 的退款请求"]

    assert bob_state.values["ticket_id"] == "T-BOB"
    assert bob_state.values["messages"] == ["Bob 的登录异常"]


def test_inspect_state_and_history():
    checkpointer = InMemorySaver()
    graph = create_ticket_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-inspect"}}

    graph.invoke(
        {
            "ticket_id": "T-2002",
            "customer_id": "C-9",
            "messages": ["工单已提交"],
            "status": "new",
        },
        config=config,
    )

    snapshot = graph.get_state(config)
    assert snapshot.values["ticket_id"] == "T-2002"
    assert snapshot.values["status"] == "in_progress"
    assert snapshot.next == ()
    assert snapshot.config["configurable"]["thread_id"] == "thread-inspect"
    assert "checkpoint_id" in snapshot.config["configurable"]

    history = list(graph.get_state_history(config))
    assert len(history) > 1
    assert (
        history[0].config["configurable"]["checkpoint_id"]
        == snapshot.config["configurable"]["checkpoint_id"]
    )
```

验证命令：

```bash
python -m pytest -q
```

---

## 总结

| 核心概念 | 作用 | 关键代码 / 配置 |
|---|---|---|
| **Checkpointer** | 在每个 super-step 边界将状态持久化到存储中 | `builder.compile(checkpointer=InMemorySaver())` |
| **`thread_id`** | 会话持久化的唯一主键索引，实现记忆与多租户隔离 | `config={"configurable": {"thread_id": "id"}}` |
| **`get_state`** | 检视当前指定线程的最新状态快照（`StateSnapshot`） | `graph.get_state(config)` |
| **`get_state_history`** | 获取指定线程的全部历史检查点链条，用于时间旅行与审计 | `list(graph.get_state_history(config))` |

有了第 06 章的持久化基础，图就拥有了“等待与被唤醒”的物理前提。在接下来的**第 07 章《人在回路：暂停与恢复》**中，我们将学习如何利用 Checkpointer，在图执行中途主动调用 `interrupt()` 挂起，等待人工审核员审批后恢复流转。

---

## 官方一手参考

- [LangGraph Documentation - Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Documentation - Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph API Reference - StateSnapshot](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.StateSnapshot)
