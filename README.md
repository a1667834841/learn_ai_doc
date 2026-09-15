# LangGraph 测试驱动学习课程（Python / Graph API）

基于 LangGraph 官方文档（<https://docs.langchain.com/oss/python/langgraph>）设计的中文 TDD 学习课程。方法论参考 [quii/learn-go-with-tests](https://github.com/quii/learn-go-with-tests)：每个概念都因为一个可执行的失败而出现，先观察失败，再引入使行为通过的最小知识。

## 演进项目

全书围绕一个项目逐步生长：**客服工单处理智能体（Support Ticket Agent）**。

前半程（01–12 章）使用**确定性 fake 组件**驱动，**不需要任何 LLM API key**，所有验证由 pytest 在秒级完成。真实 LLM 集成放在可选章 14。

## 前置要求

| 项目 | 要求 |
|---|---|
| Python | **3.11+**（系统自带的 3.9.6 不满足：LangGraph 当前版本要求 3.10+，且 async/contextvar 相关能力要求 3.11+） |
| 包管理 | `pip` + `venv`（标准库自带） |
| LLM API key | 01–13 章不需要；14 章可选 |
| 测试框架 | `pytest` |

若没有 3.11+ 的 Python，可先安装，例如 Homebrew：

```bash
brew install python@3.11
```

## 环境搭建（只做一次）

在本仓库根目录执行：

```bash
# 1. 创建虚拟环境
python3.11 -m venv .venv

# 2. 激活（每次打开新终端都要执行）
source .venv/bin/activate

# 3. 安装依赖
pip install -U langgraph pytest

# 4. 验证安装成功
python -c "import langgraph; from langgraph.graph import StateGraph; print('langgraph OK')"
```

预期输出：

```text
langgraph OK
```

## 目录约定

```text
README.md                      # 本文件
docs/
  00-learning-map.md           # 学习地图与知识依赖图
  01-first-graph/              # 每章一个文件夹
    README.md                  # 章节教程（本技能只生成这个）
    ...                        # 你后续在此创建 graph.py、test_graph.py 等
  02-passing-data/
    README.md                  # 第 02 章：节点之间传数据
  03-conditional-routing/
    README.md                  # 第 03 章：按数据选择分支
  04-loops-and-limits/
    README.md                  # 第 04 章：循环与递归限制
  05-parallel-fanout/
    README.md                  # 第 05 章：并行 fan-out
  06-persistence/
    README.md                  # 第 06 章：持久化
  07-interrupts/
    README.md                  # 第 07 章：人在回路
  08-streaming/
    README.md                  # 第 08 章：流式
  09-testing/
    README.md                  # 第 09 章：测试
  10-fault-tolerance/
    README.md                  # 第 10 章：容错
  11-subgraphs/
    README.md                  # 第 11 章：子图
  12-store-memory/
    README.md                  # 第 12 章：长期记忆
  13-functional-api/
    README.md                  # 第 13 章：Functional API
  14-real-llm/
    README.md                  # 第 14 章：真实 LLM 与工具
```

- 每个章节文件夹是一个**检查点**：内含该阶段可独立运行的代码。
- 章节教程会列出你需要新建/修改的文件，并给出完整代码片段；代码文件由你自己创建。
- 从第 02 章起，教程会指明"从上一章复制哪些文件、改哪几处"，避免整项目重复拷贝。

## 常用命令

```bash
source .venv/bin/activate                     # 激活环境
cd docs/01-first-graph && python -m pytest -q # 运行当前章节的测试
python -m pytest -q test_graph.py -k runs     # 只跑某一条测试
deactivate                                    # 退出虚拟环境
```

## 学习路径

| 章节 | 主题 | 核心概念 |
|---|---|---|
| [01](docs/01-first-graph/README.md) | 第一个可运行的图 | `StateGraph` / node / edge / `START` / `END` / `compile` / `invoke` |
| [02](docs/02-passing-data/README.md) | 节点之间传数据 | state schema、reducer（`Annotated` + `operator.add` / `add_messages`） |
| [03](docs/03-conditional-routing/README.md) | 按数据选择分支 | 条件边 `add_conditional_edges` |
| [04](docs/04-loops-and-limits/README.md) | 循环与递归限制 | 循环边、`GRAPH_RECURSION_LIMIT`、`recursion_limit` |
| [05](docs/05-parallel-fanout/README.md) | 并行 fan-out | 并行分支、superstep、`INVALID_CONCURRENT_GRAPH_UPDATE` |
| [06](docs/06-persistence/README.md) | 持久化：让对话记得住 | checkpointer、`thread_id`、`get_state` |
| [07](docs/07-interrupts/README.md) | 人在回路：暂停与恢复 | `interrupt()`、`Command(resume=...)`、节点重放规则 |
| [08](docs/08-streaming/README.md) | 流式：边跑边看进度 | stream modes、`get_stream_writer`、`version="v2"` |
| [09](docs/09-testing/README.md) | 图怎么写测试 | 单节点测试、`update_state(as_node=...)`、部分执行 |
| [10](docs/10-fault-tolerance/README.md) | 容错：外部服务会挂 | `RetryPolicy`、`timeout`、`error_handler` |
| [11](docs/11-subgraphs/README.md) | 子图 | 子图作为节点、state 共享、`subgraphs=True` |
| [12](docs/12-store-memory/README.md) | 长期记忆 Store | `Store` vs checkpointer、跨 thread 记忆 |
| [13](docs/13-functional-api/README.md)（可选） | Functional API | `@entrypoint`、`@task`、API 取舍 |
| [14](docs/14-real-llm/README.md)（可选） | 接上真实 LLM 与工具 | `init_chat_model`、tool calling、本地 server / Studio |

完整章节动机与官方页面映射见 [docs/00-learning-map.md](docs/00-learning-map.md)。

## 官方文档来源

- LangGraph Overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- Thinking in LangGraph: <https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph>
- Graph API overview: <https://docs.langchain.com/oss/python/langgraph/graph-api>
- Persistence / Checkpointers / Stores: <https://docs.langchain.com/oss/python/langgraph/persistence>
- Interrupts: <https://docs.langchain.com/oss/python/langgraph/interrupts>
- Streaming: <https://docs.langchain.com/oss/python/langgraph/streaming>
- Test: <https://docs.langchain.com/oss/python/langgraph/test>
- Fault tolerance: <https://docs.langchain.com/oss/python/langgraph/fault-tolerance>

课程中的概念性论断以上述当前官方页面为准；如官方 API 变更，请以官方文档为准并欢迎反馈更新。
