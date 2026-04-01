"""LangGraph 入门示例：用一个可离线运行的小图学习核心概念。

适合学习：
- `StateGraph` / `START` / `END` 的最小骨架
- 节点函数如何读取 `state`，并返回“部分状态更新”
- 条件边（conditional edges）如何决定下一步节点
- `compile()`、`invoke()`、`stream()` 这三个常见入口怎么用
- 为什么 LangGraph 很适合表达多步骤、可分支的 agent / workflow

这个脚本刻意**不接真实 LLM**，而是使用一个内置的小型知识库，
这样你可以把注意力放在“图是怎么跑起来的”，而不是模型调用细节上。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from _debug_common import print_kv, print_title
from langgraph.graph import END, START, StateGraph

# 一个极小的“知识库”。
#
# 在真实项目里，这一步往往来自：
# - 向量检索
# - 数据库查询
# - 调用外部工具
# - 调用另一个子图
#
# 这里我们直接用 dict，是为了保证例子稳定、离线、可重复。
TOPIC_FACTS = {
    "langgraph": "LangGraph 是一个用图（graph）来编排 LLM / 工具 / 业务步骤的框架，适合需要分支、循环和状态管理的工作流。",
    "rag": "RAG（Retrieval-Augmented Generation）会先检索相关资料，再把检索结果交给模型生成答案，从而降低幻觉并利用私有知识。",
    "lasso": "Lasso 是一种 lookup argument protocol，常出现在零知识证明或 proof systems 的上下文中。",
}


class DemoState(TypedDict, total=False):
    """LangGraph 在这个例子里共享的状态对象。

    可以把它理解成“在图中流动的一份上下文”。
    每个节点都可以：
    1. 读取已有字段
    2. 返回自己想更新的字段

    LangGraph 会把这些局部更新合并成新的整体状态，再传给下一个节点。
    """

    user_question: str
    normalized_question: str
    intent: Literal["retrieve", "clarify"]
    topic: str
    fact: str
    answer: str
    trace: list[str]


def append_trace(state: DemoState, step_name: str) -> list[str]:
    """把当前步骤追加到 trace，方便观察图的执行路径。"""
    return [*state.get("trace", []), step_name]


def detect_topic(question: str) -> str:
    """根据问题中的关键词，粗略识别用户想问的主题。

    这是一个非常朴素的规则函数，只用于教学。
    实际项目中你可以把这里替换成：
    - LLM 分类
    - 检索路由器
    - 规则 + 模型混合判断
    """
    lowered = question.lower()
    if "langgraph" in lowered:
        return "langgraph"
    if "rag" in lowered or "检索增强" in question or "检索" in question:
        return "rag"
    if "lasso" in lowered:
        return "lasso"
    return ""


def analyze_question(state: DemoState) -> DemoState:
    """第一个节点：分析输入问题，并决定后续走向。

    这个节点只做两件事：
    - 规范化问题文本（去掉首尾空白）
    - 初步判断：问题是否足够明确，可以直接进入“检索”分支
    """
    normalized_question = state["user_question"].strip()
    topic = detect_topic(normalized_question)
    intent: Literal["retrieve", "clarify"] = "retrieve" if topic else "clarify"

    return {
        "normalized_question": normalized_question,
        "intent": intent,
        "topic": topic,
        "trace": append_trace(state, "analyze_question"),
    }


def route_after_analysis(state: DemoState) -> Literal["retrieve_fact", "ask_for_clarification"]:
    """条件路由函数：根据 `intent` 决定下一个节点。"""
    return "retrieve_fact" if state["intent"] == "retrieve" else "ask_for_clarification"


def retrieve_fact(state: DemoState) -> DemoState:
    """第二个节点：模拟“检索”。

    这里会根据 `topic` 去内置知识库里取一段说明。
    真实的 RAG 场景中，这一步常常对应：
    - embedding + 向量库检索
    - BM25 / keyword search
    - 外部 API / 数据库调用
    """
    topic = state.get("topic", "")
    fact = TOPIC_FACTS.get(topic, "")

    return {
        "fact": fact,
        "trace": append_trace(state, "retrieve_fact"),
    }


def route_after_retrieval(state: DemoState) -> Literal["answer_question", "ask_for_clarification"]:
    """第二个条件路由：检索到了内容就回答，否则转入澄清。"""
    return "answer_question" if state.get("fact") else "ask_for_clarification"


def answer_question(state: DemoState) -> DemoState:
    """第三个节点：基于检索到的事实生成答案。

    注意这里依然没有使用 LLM，而是直接拼装一个可读答案。
    这样做是为了让你更容易看清楚：
    “LangGraph 负责的是流程控制与状态管理，而不是强制你一定要接模型。”
    """
    answer = (
        f"问题：{state['normalized_question']}\n"
        f"识别主题：{state['topic']}\n"
        f"检索结果：{state['fact']}\n\n"
        "学习提示：在真实 agent 中，这里通常会把检索结果交给 LLM，"
        "让模型把资料整理成更自然的最终回答。"
    )

    return {
        "answer": answer,
        "trace": append_trace(state, "answer_question"),
    }


def ask_for_clarification(state: DemoState) -> DemoState:
    """兜底节点：当问题过于模糊时，让用户补充信息。"""
    answer = (
        "我暂时无法判断你想问哪个主题。\n"
        "请在问题里明确提到 `LangGraph`、`RAG` 或 `Lasso`。\n"
        "例如：\n"
        "- LangGraph 是什么？\n"
        "- RAG 为什么能减少幻觉？\n"
        "- What is Lasso?"
    )

    return {
        "answer": answer,
        "trace": append_trace(state, "ask_for_clarification"),
    }


def build_demo_graph():
    """构建并编译图。

    图的结构如下：

        START
          -> analyze_question
              -> (条件分支)
                 -> retrieve_fact
                     -> (条件分支)
                        -> answer_question
                        -> ask_for_clarification
                 -> ask_for_clarification
          -> END
    """
    builder = StateGraph(DemoState)

    # 注册节点：给每个 Python 函数取一个图中的节点名。
    builder.add_node("analyze_question", analyze_question)
    builder.add_node("retrieve_fact", retrieve_fact)
    builder.add_node("answer_question", answer_question)
    builder.add_node("ask_for_clarification", ask_for_clarification)

    # 从 START 进入第一个节点。
    builder.add_edge(START, "analyze_question")

    # 第一个条件分支：问题明确 -> 检索；问题模糊 -> 澄清。
    builder.add_conditional_edges(
        "analyze_question",
        route_after_analysis,
        {
            "retrieve_fact": "retrieve_fact",
            "ask_for_clarification": "ask_for_clarification",
        },
    )

    # 第二个条件分支：检索命中 -> 回答；未命中 -> 澄清。
    builder.add_conditional_edges(
        "retrieve_fact",
        route_after_retrieval,
        {
            "answer_question": "answer_question",
            "ask_for_clarification": "ask_for_clarification",
        },
    )

    # 无论是成功回答还是要求澄清，最后都结束。
    builder.add_edge("answer_question", END)
    builder.add_edge("ask_for_clarification", END)

    # compile() 之后得到的是一个可执行对象。
    return builder.compile()


def print_final_state(question: str, final_state: DemoState) -> None:
    """把 invoke 的最终状态打印出来，便于对照学习。"""
    print_kv("question", question)
    print_kv("intent", final_state.get("intent", "<none>"))
    print_kv("topic", final_state.get("topic", "<none>"))
    print_kv("trace", " -> ".join(final_state.get("trace", [])))
    print("answer:")
    print(final_state["answer"])


def run_invoke_demo(app) -> None:
    """演示 `invoke()`：一次性跑完整张图，并拿到最终状态。"""
    print_title("invoke：问题明确，走检索分支")
    final_state = app.invoke({"user_question": "LangGraph 是什么？"})
    print_final_state("LangGraph 是什么？", final_state)

    print_title("invoke：问题模糊，走澄清分支")
    final_state = app.invoke({"user_question": "这个东西到底有什么用？"})
    print_final_state("这个东西到底有什么用？", final_state)


def run_stream_demo(app) -> None:
    """演示 `stream()`：逐步观察每个节点返回了什么更新。"""
    print_title("stream：逐步观察每一步更新")
    for step_index, event in enumerate(app.stream({"user_question": "RAG 为什么能减少幻觉？"}), start=1):
        node_name, update = next(iter(event.items()))
        print_kv(f"step_{step_index}", node_name)
        print_kv("update", update)
        print("-" * 80)


def main() -> None:
    print_title("debug_langgraph")

    app = build_demo_graph()

    # Mermaid 文本非常适合快速理解图结构。
    # 如果你把它粘到支持 Mermaid 的 Markdown 工具里，就能看到流程图。
    print_title("Mermaid 图结构")
    print(app.get_graph().draw_mermaid())

    run_invoke_demo(app)
    run_stream_demo(app)


if __name__ == "__main__":
    main()

