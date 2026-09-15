# 第 08 章：流式：边跑边看进度

> 来源：[Streaming | LangGraph](https://docs.langchain.com/oss/python/langgraph/streaming)  
> 适用版本：LangGraph >= 1.2（教程基于 1.2.11 实测）  
> 验证环境：Python 3.12.12 / pytest 9.1.1  

在之前的章节中，我们一直使用 `graph.invoke(...)` 来运行图。调用 `invoke()` 会一直阻塞当前线程，直到图中所有的节点、边和条件分支全部流转结束，最后一次性把最终的 State 返回给调用方。

但在真实的**客服工单处理智能体（Support Ticket Agent）**业务中，这种“黑盒阻塞”模式会遇到严重的工程痛点：
1. **长耗时操作导致用户焦虑**：工单处理通常包含耗时的阶段——例如工单意图分析、检索包含数万条文档的知识库、草拟针对性解决方案。如果整个过程耗时 3~8 秒，前端用户只看到一个不断旋转的 loading 图标，无法判断系统是卡死了、挂了，还是正在努力计算；
2. **网络超时与防御性重试**：很多 API 网关、微服务反向代理（如 Nginx、API Gateway）或前端客户端设置了严格的超时门限。一个长时间没有任何数据包返回的 HTTP 请求极易触发断连，导致用户重复提交工单；
3. **缺少过程可观察性**：调用方不仅想看到“最终结果”，往往还需要监听“中间状态”——比如实时展示工单流转到了哪个节点、各节点的局部更新，或者节点内部排查处理的百分比进度。

为了解决这些问题，LangGraph 提供了强大的流式输出接口：`graph.stream(...)`。本章我们将通过测试驱动开发（TDD），系统掌握 LangGraph 的核心流式模式（Stream Modes）以及标准化的事件分发机制。

---

## 来源契约

在动手写代码前，我们先建立官方文档规范的来源契约：

- **官方文档**：[Streaming | LangGraph](https://docs.langchain.com/oss/python/langgraph/streaming)
- **API 参考**：[Pregel.stream | Graph API](https://reference.langchain.com/python/langgraph/pregel/#langgraph.pregel.Pregel.stream)
- **核心契约**：
  1. **调用签名**：`graph.stream(input, config=None, stream_mode=..., version="v2")`。该方法返回一个 Python 生成器（Iterator），每产生一个流事件就 `yield` 一个 chunk；
  2. **核心模式 `stream_mode`**：
     - `"values"`：在每个 Super-step 结束后，输出**整张图的完整状态快照（Full State Snapshot）**；注意：在图进入第 1 个节点之前，Super-step 0 会首先发射初始输入的快照；
     - `"updates"`：在每个节点执行完成后，仅输出该节点返回的**局部状态增量更新字典**，格式形如 `{node_name: {updated_fields}}`；
     - `"custom"`：在节点执行过程中，通过 `get_stream_writer()` 主动发送用户自定义的细粒度数据（如检索进度百分比、日志、中间片段等）；
  3. **自定义流发射器 `get_stream_writer()`**：
     - 位于 `langgraph.config` 模块；
     - 仅当调用 `graph.stream(...)` 时配置了 `"custom"` 模式，`get_stream_writer()` 发送的数据才会被外层捕获；否则写入的数据会被静默丢弃（作为 no-op 处理）；
  4. **版本规范 `version="v2"`（StreamPart 契约）**：
     - 在 LangGraph 1.1+ 中引入。当传入 `version="v2"` 时，所有输出统一打包为标准的 `StreamPart` 字典：`{"type": "updates"|"custom"|"values", "ns": (...), "data": ...}`；
     - 若未指定 `version="v2"`（即 v1 默认行为），单一模式返回裸数据，多模式联合流式则返回 `(mode, data)` 的元组。生产实践中推荐显式指定 `version="v2"` 以获得一致的结构与类型收窄支持。

---

## 迭代一：状态快照流式（`stream_mode="values"`）

### 1. 先写测试

我们希望为工单处理图建立一个流式监听器：调用方传入初始工单后，能够随着图的每一步执行，逐步观察到完整工单状态的演进过程。

具体行为期望：
- 节点包含两个阶段：`classify`（分类）与 `resolve`（解决）；
- 当使用 `stream_mode="values"` 时，外部收集到的事件序列应严格按时间产生 3 个状态快照：
  1. **快照 0**：接收到初始输入时的工单状态（状态为 `new`，类别为空）；
  2. **快照 1**：分类节点执行完毕后的状态（状态变为 `classified`，类别更新为 `billing`）；
  3. **快照 2**：解决节点执行完毕后的终态（状态变为 `resolved`，携带生成的解决方案）。

编写测试用例 `test_graph.py`：

```python
from graph import create_ticket_graph


def test_stream_values_emits_state_progression():
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    # 以 values 模式启动流式消费
    chunks = list(graph.stream(initial_input, stream_mode="values"))

    # 断言 1: 总共产生 3 个状态快照（初始输入 1 个 + 两个节点各 1 个）
    assert len(chunks) == 3

    # 断言 2: Step 0 是接收到初始输入后的快照
    assert chunks[0]["status"] == "new"
    assert chunks[0]["category"] == ""

    # 断言 3: Step 1 是 classify 节点执行后的状态快照
    assert chunks[1]["status"] == "classified"
    assert chunks[1]["category"] == "billing"

    # 断言 4: Step 2 是 resolve 节点执行后的最终状态快照
    assert chunks[2]["status"] == "resolved"
    assert chunks[2]["solution"] == "请提供订单号为您办理退款"
```

### 2. 让测试能够运行并确认红灯

在 `graph.py` 中先写出最简骨架，只连一条从 `START` 到 `END` 的直通图，尚未挂载具体的业务节点：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TicketState(TypedDict):
    ticket_id: str
    issue: str
    category: str
    solution: str
    status: str


def create_ticket_graph():
    builder = StateGraph(TicketState)
    builder.add_edge(START, END)
    return builder.compile()
```

运行 pytest：

```bash
pytest -q test_graph.py
```

实测输出：

```text
F                                                                        [100%]
=================================== FAILURES ===================================
__________________ test_stream_values_emits_state_progression __________________

    def test_stream_values_emits_state_progression():
        graph = create_ticket_graph()
        initial_input = {
            "ticket_id": "T-001",
            "issue": "申请退款",
            "category": "",
            "solution": "",
            "status": "new",
        }
        chunks = list(graph.stream(initial_input, stream_mode="values"))

>       assert len(chunks) == 3
E       AssertionError: assert 1 == 3
E        +  where 1 = len([{'category': '', 'issue': '申请退款', 'solution': '', 'status': 'new', 'ticket_id': 'T-001'}])
```

测试变红，且失败原因极其关键：`assert 1 == 3`！  
因为图直接从 `START` 走到 `END`，没有经历中间节点的计算，因此仅仅输出了初始状态这 1 个快照。

### 3. 最小实现变绿

现在我们在 `graph.py` 中实现 `classify_node` 和 `resolve_node`，并将它们加入图中：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TicketState(TypedDict):
    ticket_id: str
    issue: str
    category: str
    solution: str
    status: str


def classify_node(state: TicketState):
    """根据工单内容判断工单类型。"""
    issue = state["issue"].lower()
    category = "billing" if "refund" in issue or "退款" in issue else "technical"
    return {"category": category, "status": "classified"}


def resolve_node(state: TicketState):
    """根据工单类型生成解决方案。"""
    if state["category"] == "billing":
        solution = "请提供订单号为您办理退款"
    else:
        solution = "请重启客户端尝试"
    return {"solution": solution, "status": "resolved"}


def create_ticket_graph():
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify_node)
    builder.add_node("resolve", resolve_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "resolve")
    builder.add_edge("resolve", END)

    return builder.compile()
```

再次运行测试：

```bash
pytest -q test_graph.py
```

实测输出：

```text
.                                                                        [100%]
1 passed in 0.12s
```

绿灯！

### 4. 机制深挖：为什么是 3 个 Snapshot？

在 LangGraph 的执行模型（Pregel 架构）中，执行被划分为离散的 **Super-step（超级步）**：
- **Super-step 0**：引擎接收传入的初始字典，与图的 Schema 融合，形成图的初始状态。此时整个图处于起点，`values` 模式在此刻发射第 1 个快照；
- **Super-step 1**：执行 `classify` 节点，更新状态中的 `category` 与 `status`。节点完成后，触发 Super-step 1 的状态快照发射；
- **Super-step 2**：执行 `resolve` 节点，更新 `solution` 与 `status`。节点完成后，触发 Super-step 2 的状态快照发射。

对于包含 $N$ 个顺次执行节点的图，`stream_mode="values"` 必定输出 $N + 1$ 个快照。

---

## 迭代二：节点增量流式（`stream_mode="updates"`）

### 1. 先写测试

`values` 模式虽然完整直观，但在大型应用中存在明显缺陷：如果你的 State 中保存了庞大的对话历史、长篇排查文档或巨大的上下文向量，每个 step 都把整份 State 重新序列化并广播一遍，会造成**大量的网络传输冗余与内存带宽浪费**。

在很多时候，前端 UI 只需要知道：**刚才到底是哪一个节点执行完了？它返回了什么修改？**

这正是 `stream_mode="updates"` 的设计使命：它只输出每个节点的增量更新字典，格式为 `{node_name: {updated_fields}}`，并且**绝不输出初始状态**。

在 `test_graph.py` 中追加测试：

```python
def test_stream_updates_emits_node_deltas():
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    chunks = list(graph.stream(initial_input, stream_mode="updates"))

    # 断言 1: updates 模式不包含初始输入，只有产生变更的节点字典
    assert len(chunks) == 2

    # 断言 2: 每个 chunk 是以节点名称为 key 的局部更新字典
    assert chunks[0] == {
        "classify": {"category": "billing", "status": "classified"}
    }
    assert chunks[1] == {
        "resolve": {"solution": "请提供订单号为您办理退款", "status": "resolved"}
    }
```

### 2. 运行测试（绿灯验证）

运行测试：

```bash
pytest -q test_graph.py
```

实测输出：

```text
..                                                                       [100%]
2 passed in 0.13s
```

测试直接通过！

### 3. 深度对比：`values` vs `updates`

| 对比维度 | `stream_mode="values"` | `stream_mode="updates"` |
|---|---|---|
| **输出粒度** | 整图完整状态快照（Full State） | 单节点局部增量字典（Node Delta） |
| **首个事件** | Super-step 0 的初始输入状态 | 第一个完成的节点更新（无初始输入） |
| **事件结构** | `{"ticket_id": ..., "status": ...}` | `{"classify": {"status": ...}}` |
| **事件数量** | 节点步骤数 + 1 | 产生更新的节点步骤数 |
| **适用场景** | 每次都需要全量刷新状态面板的轻量 UI | 大型状态图、需要展示“节点完成步骤卡片”的工作流界面 |

---

## 迭代三：节点内长任务进度（`get_stream_writer` 与 `stream_mode="custom"`）

### 1. 先写测试

`updates` 解决了节点完成后的通知问题。但如果某一个节点本身就是一个**高耗时节点**呢？

例如在客服工单中，`resolve_node` 需要：
1. 连接远程向量库检索相似案例（耗时 1.5 秒）；
2. 检索业务退款政策规则集（耗时 1.5 秒）；
3. 综合判断并草拟回复（耗时 1.5 秒）。

在整个节点运行的 4.5 秒内，由于函数还没有 `return`，`updates` 流完全是静止的！外部客户端依然会陷入漫长的等待。我们希望节点在内部执行的不同阶段，能够像心跳一样主动往外广播进度事件（如 `progress: 30%`、`progress: 70%`）。

在 `test_graph.py` 中追加测试：

```python
def test_stream_custom_emits_in_node_progress():
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    # 使用 custom 模式消费流
    chunks = list(graph.stream(initial_input, stream_mode="custom"))

    # 断言 1: 捕获到节点内部发射的 3 次细粒度进度事件
    assert len(chunks) == 3

    # 断言 2: 验证每次自定义事件的内容
    assert chunks[0] == {"stage": "query_kb", "progress": 30, "message": "正在检索知识库..."}
    assert chunks[1] == {"stage": "draft_solution", "progress": 70, "message": "正在草拟解决方案..."}
    assert chunks[2] == {"stage": "done", "progress": 100, "message": "方案已生成"}
```

### 2. 运行测试（确认红灯）

运行测试：

```bash
pytest -q test_graph.py
```

实测输出：

```text
..F                                                                      [100%]
=================================== FAILURES ===================================
__________________ test_stream_custom_emits_in_node_progress ___________________

    def test_stream_custom_emits_in_node_progress():
        graph = create_ticket_graph()
        initial_input = {
            "ticket_id": "T-001",
            "issue": "申请退款",
            "category": "",
            "solution": "",
            "status": "new",
        }
        chunks = list(graph.stream(initial_input, stream_mode="custom"))
>       assert len(chunks) == 3
E       assert 0 == 3
E        +  where 0 = len([])
```

红灯非常直观：`assert 0 == 3`。因为此时 `resolve_node` 内部并没有发射任何自定义数据，在 `stream_mode="custom"` 下，生成器没有产出任何 chunk，`chunks` 是个空列表。

### 3. 最小实现变绿

我们需要引入 `get_stream_writer()`。它位于 `langgraph.config` 模块中。在节点内部调用 `get_stream_writer()` 会返回一个可调用对象 `writer`，我们直接传入任意自定义字典或数据即可发射事件。

更新 `graph.py` 中的 `resolve_node`：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer  # 导入流发射器获取函数


class TicketState(TypedDict):
    ticket_id: str
    issue: str
    category: str
    solution: str
    status: str


def classify_node(state: TicketState):
    """根据工单内容判断工单类型。"""
    issue = state["issue"].lower()
    category = "billing" if "refund" in issue or "退款" in issue else "technical"
    return {"category": category, "status": "classified"}


def resolve_node(state: TicketState):
    """根据工单类型生成解决方案，并在处理过程中发射进度。"""
    writer = get_stream_writer()
    if writer:
        writer({"stage": "query_kb", "progress": 30, "message": "正在检索知识库..."})
        writer({"stage": "draft_solution", "progress": 70, "message": "正在草拟解决方案..."})
        writer({"stage": "done", "progress": 100, "message": "方案已生成"})

    if state["category"] == "billing":
        solution = "请提供订单号为您办理退款"
    else:
        solution = "请重启客户端尝试"

    return {"solution": solution, "status": "resolved"}


def create_ticket_graph():
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify_node)
    builder.add_node("resolve", resolve_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "resolve")
    builder.add_edge("resolve", END)

    return builder.compile()
```

再次运行测试：

```bash
pytest -q test_graph.py
```

实测输出：

```text
...                                                                      [100%]
3 passed in 0.14s
```

测试全部通过！通过 `get_stream_writer()`，我们在不改动图的状态 Schema（`TicketState`）的前提下，打通了一条带外的实时进度通道。

---

## 迭代四：生产标准实践——联合流式与 `version="v2"`

### 1. 痛点：多路事件混合分发

在实际的前端对接中（例如基于 Server-Sent Events / WebSocket 的工单流转控制台），调用方往往**同时需要**两类事件：
1. 节点完成时广播的增量状态（`updates`）；
2. 耗时节点内部发出的细粒度实时进度（`custom`）。

如果分别调用两次 `graph.stream(...)`，整张图就会被执行两遍，造成严重的算力浪费和副作用重复。因此，我们需要将多个模式合并传入：`stream_mode=["updates", "custom"]`。

但在早期版本（v1）中，传入多模式会使返回值结构变成二元元组：`("updates", {...})` 或 `("custom", {...})`；而单模式返回的是裸数据。这种不一致的返回值结构极易在系统重构或模式扩展时导致解析逻辑崩溃。

为此，LangGraph 1.1+ 推出了标准化的 `version="v2"` 协议（即 `StreamPart`）：
无论你配置了多少个 stream mode、是否包含子图，返回的每一个 chunk 必定是具有统一字段的结构化字典：
```python
{
    "type": "updates" | "custom" | "values",  # 事件类型标签
    "ns": (),                                  # 命名空间（子图标识）
    "data": ...                                # 对应模式的具体载荷
}
```

### 2. 先写测试

在 `test_graph.py` 中编写联合流式测试：

```python
def test_stream_combined_v2_stream_part():
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    # 同时启用 updates 与 custom，并显式指定 version="v2"
    chunks = list(
        graph.stream(
            initial_input,
            stream_mode=["updates", "custom"],
            version="v2",
        )
    )

    # 断言 1: 总事件数应为 5（classify 的 1 个 updates + resolve 的 3 个 custom + resolve 的 1 个 updates）
    assert len(chunks) == 5

    # 断言 2: 每个 chunk 严格符合 v2 的 StreamPart 协议结构
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "type" in chunk
        assert "ns" in chunk
        assert "data" in chunk

    # 断言 3: 事件产出的严格时间先后顺序
    event_types = [c["type"] for c in chunks]
    assert event_types == ["updates", "custom", "custom", "custom", "updates"]

    # 断言 4: 细粒度更新与局部节点输出各自的数据一致性
    assert chunks[0]["data"] == {
        "classify": {"category": "billing", "status": "classified"}
    }
    assert chunks[1]["data"]["stage"] == "query_kb"
    assert chunks[4]["data"] == {
        "resolve": {"solution": "请提供订单号为您办理退款", "status": "resolved"}
    }
```

### 3. 运行测试（绿灯验证）

运行测试：

```bash
pytest -q test_graph.py
```

实测输出：

```text
....                                                                     [100%]
4 passed in 0.15s
```

4 个测试全部绿灯通过！

### 4. 生产消费者模式：类型路由（Type Narrowing）

有了 `version="v2"`，在编写 FastAPI SSE 接口或前端消费逻辑时，代码会变得极其优雅和严谨：

```python
# 生产消费逻辑范例
for chunk in graph.stream(input_data, stream_mode=["updates", "custom"], version="v2"):
    event_type = chunk["type"]
    payload = chunk["data"]

    if event_type == "custom":
        # 细粒度进度通知：更新前端进度条
        send_sse_event(event="progress", data=payload)
    elif event_type == "updates":
        # 节点状态更新：更新工单当前阶段
        for node_name, state_update in payload.items():
            send_sse_event(event="node_finished", data={"node": node_name, "update": state_update})
```

---

## 避坑指南（Gotchas）

在落地流式架构时，有以下四个最常见的陷阱，请务必留意：

### 1. `values` 模式的首个 chunk 是初始输入（Superstep 0 陷阱）
很多初学者在使用 `stream_mode="values"` 时，理所当然地认为“图里有 2 个节点，因此只会产出 2 个状态块”，进而在消费流时把第 1 个 chunk 当作第 1 个节点（如分类节点）的执行产物。
```python
# ❌ 错误做法：误以为第 1 个 chunk 是 classify 节点的产出
for chunk in graph.stream(initial_input, stream_mode="values"):
    first_node_output = chunk["category"] # 踩坑！第一次循环时 category 还是空字符串！
    break
```
> **规则**：`values` 会先发送图接收到输入时的初始快照。若业务只关心“经过节点加工后的状态”，应该使用 `stream_mode="updates"`，或者通过计数跳过第 1 个快照。

### 2. 忘记在 `stream_mode` 中声明 `"custom"` 导致 `writer` 静默丢弃
在节点内部编写了 `writer = get_stream_writer(); writer(...)`，但在调用 `graph.stream(...)` 时只写了 `stream_mode="updates"`。此时系统**不会抛出任何异常**，而是返回一个 no-op writer，写入的数据被全部默默丢弃！
> **规则**：只要节点内部使用了 `get_stream_writer()`，调用 `stream()` 或 `astream()` 时必须显式声明包含 `"custom"`，例如 `stream_mode=["updates", "custom"]`。

### 3. v1 与 v2 协议不兼容引发的 `TypeError`
如果调用联合流式时漏掉了 `version="v2"`：
```python
# ❌ 漏写 version="v2"
for chunk in graph.stream(input_data, stream_mode=["updates", "custom"]):
    # 在 v1 联合流中，chunk 是一个 ('updates', {...}) 的 tuple！
    event_type = chunk["type"]  # 💥 抛出 TypeError: tuple indices must be integers or slices, not str
```
实测报错：
```text
TypeError: tuple indices must be integers or slices, not str
```
> **规则**：所有新开发的代码强烈建议显式加上 `version="v2"`，彻底摆脱 tuple 与 dict 的类型不确定性。

### 4. 在 Web 服务中使用 `list(graph.stream(...))` 强转造成的“假流式”
有的开发者在编写 FastAPI 或 Flask 流式响应时，习惯性地把迭代器用 `list(...)` 包裹或者在外层加了全局缓冲：
```python
# ❌ 严重反模式：将整个生成器一次性耗尽，流式退化为阻塞批处理
all_events = list(graph.stream(input_data, stream_mode="updates"))
return EventSourceResponse(iter(all_events))
```
这样做会导致所有节点计算全部完成后才把数据批量发出，前端完全感受不到中间流式效果。
> **规则**：在网络传输层，应使用 `for chunk in graph.stream(...): yield format_sse(chunk)`，随算随发。

---

## 核心概念对比与总结表

| 流模式（Stream Mode） | 单条事件数据类型（v2 Data） | 产生时机 | 核心价值 |
|---|---|---|---|
| **`"values"`** | 完整的 `TicketState` 字典 | 每个 Super-step 完成时（含初始输入） | 适合需要无脑全量同步整图状态的前端视图 |
| **`"updates"`** | `{node_name: {diff_keys}}` | 每个节点执行完成时 | 适合长生命周期工作流，轻量传输状态增量 |
| **`"custom"`** | 任意 JSON 可序列化载荷 | 节点内部调用 `writer(...)` 时 | 适合节点内耗时排查任务的进度条、心跳广播 |
| **`"messages"`** | `(message_chunk, metadata)` 二元组 | LLM 模型生成 token 时 | 真实大模型逐字打字机流式（第 14 章详述） |

---

## 完整代码清单

为方便本地独立复现与验证，本章完整可运行代码如下：

### `graph.py`

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


class TicketState(TypedDict):
    ticket_id: str
    issue: str
    category: str
    solution: str
    status: str


def classify_node(state: TicketState):
    """工单分类节点：根据工单内容判断类别。"""
    issue = state["issue"].lower()
    category = "billing" if "refund" in issue or "退款" in issue else "technical"
    return {"category": category, "status": "classified"}


def resolve_node(state: TicketState):
    """工单解决节点：检索知识库并草拟方案，过程中发送细粒度进度事件。"""
    writer = get_stream_writer()
    if writer:
        writer({"stage": "query_kb", "progress": 30, "message": "正在检索知识库..."})
        writer({"stage": "draft_solution", "progress": 70, "message": "正在草拟解决方案..."})
        writer({"stage": "done", "progress": 100, "message": "方案已生成"})

    if state["category"] == "billing":
        solution = "请提供订单号为您办理退款"
    else:
        solution = "请重启客户端尝试"

    return {"solution": solution, "status": "resolved"}


def create_ticket_graph():
    """构建并编译工单流式处理图。"""
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify_node)
    builder.add_node("resolve", resolve_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "resolve")
    builder.add_edge("resolve", END)

    return builder.compile()
```

### `test_graph.py`

```python
from graph import create_ticket_graph


def test_stream_values_emits_state_progression():
    """验证 values 模式：输出每一步的完整状态快照序列。"""
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    chunks = list(graph.stream(initial_input, stream_mode="values"))

    # 断言 1: 总共产生 3 个快照（1 个初始输入 + 2 个节点执行结果）
    assert len(chunks) == 3

    # 断言 2: Step 0 是初始状态快照
    assert chunks[0]["status"] == "new"
    assert chunks[0]["category"] == ""

    # 断言 3: Step 1 是 classify 执行后的状态快照
    assert chunks[1]["status"] == "classified"
    assert chunks[1]["category"] == "billing"

    # 断言 4: Step 2 是 resolve 执行后的终态快照
    assert chunks[2]["status"] == "resolved"
    assert chunks[2]["solution"] == "请提供订单号为您办理退款"


def test_stream_updates_emits_node_deltas():
    """验证 updates 模式：仅输出节点的局部更新增量字典。"""
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    chunks = list(graph.stream(initial_input, stream_mode="updates"))

    # 断言 1: updates 模式不包含初始输入，只有产生变更的节点字典
    assert len(chunks) == 2

    # 断言 2: 每个 chunk 是以节点名称为 key 的局部更新字典
    assert chunks[0] == {
        "classify": {"category": "billing", "status": "classified"}
    }
    assert chunks[1] == {
        "resolve": {"solution": "请提供订单号为您办理退款", "status": "resolved"}
    }


def test_stream_custom_emits_in_node_progress():
    """验证 custom 模式：仅捕获节点内部通过 get_stream_writer 发出的自定义事件。"""
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    chunks = list(graph.stream(initial_input, stream_mode="custom"))

    # 断言 1: 仅捕获节点内部 writer 发出的 3 次进度事件
    assert len(chunks) == 3

    # 断言 2: 验证每个进度事件的载荷
    assert chunks[0] == {"stage": "query_kb", "progress": 30, "message": "正在检索知识库..."}
    assert chunks[1] == {"stage": "draft_solution", "progress": 70, "message": "正在草拟解决方案..."}
    assert chunks[2] == {"stage": "done", "progress": 100, "message": "方案已生成"}


def test_stream_combined_v2_stream_part():
    """验证联合流式与 version='v2'：返回统一的 StreamPart 协议结构。"""
    graph = create_ticket_graph()
    initial_input = {
        "ticket_id": "T-001",
        "issue": "申请退款",
        "category": "",
        "solution": "",
        "status": "new",
    }

    chunks = list(
        graph.stream(
            initial_input,
            stream_mode=["updates", "custom"],
            version="v2",
        )
    )

    # 断言 1: 总共 5 个事件（2 个 updates + 3 个 custom）
    assert len(chunks) == 5

    # 断言 2: 每个 chunk 遵循统一的 StreamPart 结构 (type, ns, data)
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "type" in chunk
        assert "ns" in chunk
        assert "data" in chunk

    # 断言 3: 事件产出的严格时间先后顺序
    event_types = [c["type"] for c in chunks]
    assert event_types == ["updates", "custom", "custom", "custom", "updates"]

    # 断言 4: 细粒度更新与局部节点输出各自的数据一致性
    assert chunks[0]["data"] == {"classify": {"category": "billing", "status": "classified"}}
    assert chunks[1]["data"]["stage"] == "query_kb"
    assert chunks[4]["data"] == {"resolve": {"solution": "请提供订单号为您办理退款", "status": "resolved"}}
```

本地运行命令与验证：

```bash
pytest -q test_graph.py
```

实测输出：

```text
....                                                                     [100%]
4 passed in 0.15s
```

---

## 官方一手参考

- [LangGraph Documentation - Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph Reference - Pregel Stream API](https://reference.langchain.com/python/langgraph/pregel/#langgraph.pregel.Pregel.stream)
- [LangGraph Reference - get_stream_writer](https://reference.langchain.com/python/langgraph/config/get_stream_writer)
- [LangGraph Documentation - Event Streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
