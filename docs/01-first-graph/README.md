# 01 · 第一个可运行的图

## 学习目标

让一张 LangGraph 图接收一条客服工单文本，经过**一个节点**的处理，在最终 state 里产出可断言的分类结果。你将亲手经历：缺模块的失败 → 缺入口的失败 → 第一次绿色。

## 当前状态

这是课程起点，什么都还没有。我们的演进项目"客服工单处理智能体"面临第一个最朴素的需求：

> **跑通一个工作单元**：给图一个输入 `{"ticket": "How do I reset my password?"}`，它要返回一个包含 `{"category": ...}` 的最终 state。

先不分类、不路由、不调 LLM——本章唯一的敌人是"图跑不起来"。

> 环境准备见根目录 [README.md](../../README.md)：Python 3.11+ 虚拟环境 + `pip install -U langgraph pytest`。以下命令默认你已 `source .venv/bin/activate`。

## 先写验证

在 `docs/01-first-graph/` 下新建 **`test_graph.py`**：

```python
from graph import build_graph


def test_ticket_graph_classifies_question():
    graph = build_graph()
    result = graph.invoke({"ticket": "How do I reset my password?"})
    assert result["category"] == "question"
```

这条测试表达的契约是：**给我一个编译好的图，我喂它一条疑问句工单，它最终告诉我这属于 `question`**。测试不知道也不关心图内部有几个节点、叫什么名字——这就是"测行为，不测实现"。

## 运行验证

```bash
cd docs/01-first-graph
python -m pytest -q
```

## 观察失败（一）：没有图

预期失败：

```text
ERROR collecting test_graph.py
E   ModuleNotFoundError: No module named 'graph'
```

这是最有信息量的失败类型之一：pytest 连测试函数都没进到，因为它依赖的 `graph` 模块不存在。失败证据告诉你——下一步只需要造出 `build_graph`，别的都不用想。

## 让代码可以运行：最小骨架

新建 **`graph.py`**——先只搭到"能 import、能调用"，**故意不加任何边**：

```python
from typing import TypedDict

from langgraph.graph import StateGraph


class State(TypedDict):
    ticket: str
    category: str


def classify(state: dict) -> dict:
    return {"category": "question"}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    # 暂时不加任何边
    return builder.compile()
```

这里已经出现了三个必要决定的最小代价：

- `StateGraph(State)` 必须接收一个 **state schema** 作为第一个参数——它声明这张图里流转的数据有哪些 key。
- `add_node("classify", classify)` 注册一个节点：节点就是一个"接收 state、返回更新"的函数。
- 节点先**硬编码**返回 `"question"`。这是合法的 TDD 手法：最小通过实现可以"撒谎"，只要它诚实到能被下一条测试拆穿（见"亲自尝试"）。

再跑一次：

```bash
python -m pytest -q
```

## 观察失败（二）：图没有入口

预期失败（不同 langgraph 版本措辞略有差异，关键词是 `entrypoint`）：

```text
E   ValueError: Graph must have an entrypoint, add a conditional edge
E   (add_conditional_edges(START, lambda s: ...) or a normal edge
E   (add_edge(START, 'node_name'))
```

这次失败发生在 `compile()` 内部，堆栈直接指向你刚写的那一行。它在教你 LangGraph 的一条核心规则：**节点不会自动按书写顺序相连**。图是有向图，运行时必须知道从哪里进入——所以你要显式声明"入口是 `classify`"。

## 让验证通过

修改 `graph.py`：从 `langgraph.graph` 导入 `START` 和 `END`，补两条边：

```python
from langgraph.graph import StateGraph, START, END


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    return builder.compile()
```

再运行：

```bash
python -m pytest -q
```

预期绿色：

```text
1 passed
```

也可以跳出测试，亲眼看一次图的输出：

```bash
python -c "
from graph import build_graph
g = build_graph()
print(g.invoke({'ticket': 'How do I reset my password?'}))
"
```

```text
{'ticket': 'How do I reset my password?', 'category': 'question'}
```

注意两点：输入里的 `ticket` 保留在结果里；节点返回的 `category` 被**合并**进了最终 state。节点返回的不是"新 state"，而是"对 state 的更新"。

## 刚刚学到了什么

### 是什么

对照官方术语（[Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)）：

- **State / state schema**：所有节点共享的数据对象。用 `TypedDict`（也可以是 dataclass / Pydantic 模型）声明有哪些 key。
- **Node**：接收当前 state、返回**部分更新**（一个 dict）的 Python 函数。
- **Edge**：节点之间的静态连接。`add_edge(START, "classify")` 表示运行时入口；`add_edge("classify", END)` 表示该节点的完成标志。
- **`START` / `END`**：两个保留的虚拟节点名，标记图的入口与出口。
- **`StateGraph`（builder）→ `compile()` → 编译后的图**：builder 可修改，编译产物可执行。对编译后的图调用 **`invoke(input)`** 会同步跑完整个图并返回最终 state。

### 为什么

不这样做会怎样，你今天已经各体验了一次：不给 state schema，`StateGraph` 不知道 key 的合并规则；不连 `START` 边，运行时找不到入口（失败二）；节点如果直接返回整个 state 或返回非 dict，会触发另一类错误（见思考题 2）。

### 怎么工作

LangGraph 的执行模型（官方称受 Pregel 启发）按 **superstep（超步）** 推进：从 `START` 边决定第一步执行哪些节点 → 每个节点收到当前 state 快照、返回更新 → 运行时把更新按 schema 规则合并进 state → 沿边找下一批节点 → 没有后继节点时结束，`invoke` 返回最终 state。你本章的图是它最小的退化形态：一个超步、一个节点。

### 什么时候使用

任何逻辑可以拆成"离散步骤 + 步骤间共享数据"的场景都适用图原语；但如果你的流程只是固定顺序的函数调用，直接写函数更简单——官方建议只有当你需要分支、循环、持久化、流式这些编排能力时才引入 LangGraph（参见 [choosing-apis](https://docs.langchain.com/oss/python/langgraph/choosing-apis) 对"何时值得用框架"的讨论）。

## 重构

测试保持绿色的前提下，改进两处结构（改完再跑一次 `python -m pytest -q` 确认仍然绿）：

1. **把 `category` 的合法值写进 schema**。`Literal` 让 state schema 顺便成为文档——读者不用翻节点函数就知道分类只有两种取值：

```python
from typing import Literal, TypedDict


class State(TypedDict):
    ticket: str
    category: Literal["question", "feedback"]
```

2. **节点参数标注上真实 schema 类型**，并利用 `add_node` 可省略名字的形态（节点名默认取函数名），让 builder 少一处字符串：

```python
def build_graph():
    builder = StateGraph(State)
    builder.add_node(classify)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    return builder.compile()
```

（节点名仍是 `"classify"`，边的字符串引用不用改。）

重构没有改变任何被验证的行为：测试从头到尾没动，`1 passed` 贯穿始终。

## 亲自尝试

红→绿循环还差最后一轮。这次你自己走：

1. 在 `test_graph.py` 里**先**加一条新测试：

```python
def test_statement_ticket_is_feedback():
    graph = build_graph()
    result = graph.invoke({"ticket": "Please add dark mode to the app."})
    assert result["category"] == "feedback"
```

2. 运行——预测它怎么失败（当前硬编码会返回 `"question"`）。
3. 把 `classify` 泛化成最小真实规则，例如：

```python
def classify(state: dict) -> dict:
    if "?" in state["ticket"]:
        return {"category": "question"}
    return {"category": "feedback"}
```

4. 跑回绿色。

这个"先撒谎、被测试拆穿、再泛化"的节奏，就是之后每一章反复使用的循环。

## 下一个需求

现在分类器会区分两种工单了。但真实工单流程不止一步：**检索知识库的节点需要用到分类节点产出的数据**——两个节点之间怎么传？如果两个节点都写同一个 key，会发生什么？这正是第 02 章 reducer 要解决的问题。

## 本章小结

- **概念**：state schema（`TypedDict`）、node（收 state、回更新）、edge、`START`/`END`、builder 与编译产物、superstep 执行模型。
- **API**：`StateGraph(State)`、`add_node` / `add_edge`、`compile()`、`invoke(input)`。
- **命令**：`python -m pytest -q`（章节内跑测试）、`python -c "..."`（直接观察图输出）。
- **工程经验**：两类失败各有分工——`ModuleNotFoundError` 告诉你缺哪个协作者，`ValueError: Graph must have an entrypoint` 告诉你运行时结构不完整；最小通过实现可以先硬编码，前提是下一条测试会逼你泛化。

## 自测题

三道题都按"先预测，再验证，最后解释"的顺序作答；只用到本章写过的代码和 `python -m pytest -q`。

1. **节点一定要连到 `END` 吗？**
   预测：删掉 `builder.add_edge("classify", END)` 这一行后跑测试，会报错、会失败，还是仍然通过？
   删掉验证后解释：没连 `END` 的节点和连了的节点，运行时行为有什么差别？这条边的价值在哪里？

2. **输入少了 `category` 为什么不报错？**
   `State` 声明了 `ticket` 和 `category` 两个 key，但 `invoke({"ticket": "..."})` 只传了 `ticket`。为什么运行不报错？最终结果里的 `category` 是谁、在什么时候写进去的？
   先口头回答，再用 `print(g.invoke({"ticket": "hi"}))` 验证你的说法。

3. **扩展：state 活在哪里？**
   对同一个编译后的图连续 `invoke` 两次，每次只传一条新工单：

   ```python
   g = build_graph()
   print(g.invoke({"ticket": "Can I get a refund?"}))
   print(g.invoke({"ticket": "Ship to Shanghai please"}))
   ```

   预测第二次输出里会出现哪些字段，运行并对照。用一句话总结：图的 state 的生命周期是什么？你的结论会在第 06 章"持久化"里升级成正式概念（checkpointer）。

---

下一章：[02 · 节点之间传数据 →](../02-passing-data/README.md)（待生成）
