"""LangGraph + LLM 示例：演示真正的“分析 -> 检索 -> 生成”。

适合学习：
- 如何把一个 RAG 闭环拆成多个 LangGraph 节点
- `analyze_question` / `retrieve_context` / `generate_answer` 的职责边界
- 如何把 LangGraph 接到真实 LLM（OpenAI、本地 OpenAI-compatible、Ollama）
- 如何在图里保留 `trace`、检索上下文、来源列表等中间状态

这个脚本会：
1. 先构造一组稳定的 demo chunk
2. 写入临时 Chroma 向量库
3. 用 LangGraph 编排“分析 -> 检索 -> 生成”
4. 调用真实 LLM 后端生成最终回答

推荐运行方式：

1) OpenAI 官方接口：
   OPENAI_API_KEY=你的密钥 \
   PDF2MD_RAG_LLM_PROVIDER=openai-compatible \
   PDF2MD_RAG_LLM_BASE_URL=https://api.openai.com \
   PDF2MD_RAG_LLM_MODEL=gpt-4o-mini \
   python examples/debug_langgraph_rag.py

2) 本地 OpenAI-compatible 服务（LM Studio / vLLM / llama.cpp server）：
   PDF2MD_RAG_LLM_PROVIDER=openai-compatible \
   PDF2MD_RAG_LLM_BASE_URL=http://localhost:1234 \
   PDF2MD_RAG_LLM_MODEL=qwen2.5-7b-instruct \
   python examples/debug_langgraph_rag.py

3) Ollama：
   PDF2MD_RAG_LLM_PROVIDER=ollama \
   PDF2MD_RAG_LLM_BASE_URL=http://localhost:11434 \
   PDF2MD_RAG_LLM_MODEL=qwen2.5:3b \
   python examples/debug_langgraph_rag.py

默认 embedding 仍使用项目里的 `sentence-transformers`，因为这样更适合演示“检索”阶段。
如果你只想快速跑通结构，也可以把 `PDF2MD_RAG_EMBEDDER=hash`，但检索质量会明显下降。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from _debug_common import get_embedding_model_name, preview_text, print_kv, print_title
from langgraph.graph import END, START, StateGraph

import pdf2md_rag.simple_qa as simple_qa_module
from pdf2md_rag.embeddings import build_embedder
from pdf2md_rag.models import Chunk
from pdf2md_rag.search import SearchResult, search_chunks
from pdf2md_rag.vectorstore import upsert_chunks


@dataclass(slots=True)
class RuntimeConfig:
    """这个示例运行时需要的静态配置。

    这些配置本身不会在图中“流动”，所以我们把它们放在 state 外面。
    这样更容易区分：

    - 什么是“每次执行都会变化的状态”（用户问题、检索结果、答案）
    - 什么是“本次运行固定不变的配置”（模型名、base URL、top_k）
    """

    embedder_type: str
    embedding_model: str
    hash_dimensions: int
    collection_name: str
    top_k: int
    max_context_chars: int
    llm_provider: str
    llm_model: str
    llm_base_url: str
    api_key: str | None
    system_prompt: str
    temperature: float
    max_tokens: int


class RagState(TypedDict, total=False):
    """LangGraph 图里流动的状态。"""

    user_question: str
    normalized_question: str
    analysis: str
    intent: Literal["retrieve", "clarify"]
    search_result: SearchResult
    context_text: str
    sources: list[str]
    answer: str
    raw_llm_response: dict[str, Any]
    trace: list[str]


def build_learning_chunks() -> list[Chunk]:
    """构造一组专门服务于 LangGraph + RAG 学习的 demo chunk。

    这里的内容是手工编写的“学习资料”，不是从真实 PDF 提取而来。
    这么做有两个好处：
    - 内容稳定，便于教学和重复实验
    - 不需要先准备真实论文或文档，也能演示完整流程
    """
    source_name = "LangGraph Learning Notes"
    source_path = "/virtual/langgraph-learning-notes.md"

    payloads = [
        (
            "demo-lg-1",
            "LangGraph 用 graph 的方式组织多步骤工作流。每个节点负责一个明确动作，"
            "例如分析问题、调用检索、或生成答案；边和条件边负责决定下一步去哪。",
            "LangGraph Overview",
            1,
        ),
        (
            "demo-lg-2",
            "在 RAG 系统里，一个常见拆分是：先分析用户问题，再检索相关资料，最后把检索结果交给 LLM 生成答案。"
            "这种拆分能让流程更可解释，也更容易单独调试每个阶段。",
            "RAG Workflow",
            2,
        ),
        (
            "demo-lg-3",
            "检索阶段通常会返回多个 chunk。一个好的生成节点不应该忽略来源，而应该把来源标签和上下文一起传给 LLM，"
            "这样最终回答才能带上 Sources，并且更容易排查幻觉。",
            "Retrieval and Citations",
            3,
        ),
        (
            "demo-lg-4",
            "LangGraph 本身不限制你必须使用哪种模型。你可以接 OpenAI 官方接口，也可以接本地 OpenAI-compatible 服务，"
            "或者使用 Ollama 这样的本地模型服务。关键是把“生成”封装为图中的一个节点。",
            "LLM Backends",
            4,
        ),
    ]

    chunks: list[Chunk] = []
    for chunk_index, (chunk_id, text, heading, page) in enumerate(payloads):
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "source_path": source_path,
                    "source_name": source_name,
                    "heading": heading,
                    "page": page,
                    "chunk_index": chunk_index,
                    "text_length": len(text),
                },
            )
        )
    return chunks


def append_trace(state: RagState, step_name: str) -> list[str]:
    """给 trace 追加一步，方便观察图到底经过了哪些节点。"""
    return [*state.get("trace", []), step_name]


def resolve_runtime_config() -> RuntimeConfig:
    """从环境变量解析一组对学习友好的默认配置。

    这里故意做了一点“智能默认值”：
    - 如果检测到 `OPENAI_API_KEY`，默认走 OpenAI-compatible + api.openai.com
    - 否则默认走本地 `ollama`

    这样同一个脚本既能对接远程 OpenAI，也能直接对接本地模型。
    """
    provider = os.getenv("PDF2MD_RAG_LLM_PROVIDER", "").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY")

    if not provider:
        provider = "openai-compatible" if api_key else "ollama"

    if provider in {"openai", "openai-compatible", "openai_compatible", "openai-compatible-http"}:
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", "https://api.openai.com")
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "gpt-4o-mini")
    elif provider == "ollama":
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "qwen2.5:3b")
    else:
        raise ValueError(f"Unsupported llm provider: {provider}")

    return RuntimeConfig(
        embedder_type=os.getenv("PDF2MD_RAG_EMBEDDER", "sentence-transformers"),
        embedding_model=os.getenv("PDF2MD_RAG_EMBEDDING_MODEL", get_embedding_model_name()),
        hash_dimensions=int(os.getenv("PDF2MD_RAG_HASH_DIMENSIONS", "384")),
        collection_name=os.getenv("PDF2MD_RAG_COLLECTION", "debug-langgraph-rag"),
        top_k=int(os.getenv("PDF2MD_RAG_TOP_K", "3")),
        max_context_chars=int(os.getenv("PDF2MD_RAG_MAX_CONTEXT_CHARS", "1800")),
        llm_provider=provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        api_key=os.getenv("PDF2MD_RAG_API_KEY") or api_key,
        system_prompt=(
            "You are a careful RAG assistant inside a LangGraph tutorial. "
            "First read the question analysis, then answer only from the retrieved context. "
            "If the context is insufficient, say so clearly. Always end with a Sources line."
        ),
        temperature=float(os.getenv("PDF2MD_RAG_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("PDF2MD_RAG_MAX_TOKENS", "700")),
    )


def prepare_demo_collection(runtime: RuntimeConfig, persist_directory: Path) -> None:
    """把 demo chunk 写入临时 Chroma。

    这一步模拟真实项目里的 ingest：
    文本 -> embedding -> upsert 到向量库。
    """
    chunks = build_learning_chunks()
    embedder = build_embedder(
        embedder_type=runtime.embedder_type,
        model_name=runtime.embedding_model,
        hash_dimensions=runtime.hash_dimensions,
    )
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=runtime.collection_name,
    )


def analyze_question(state: RagState) -> RagState:
    """第一个节点：分析用户问题。

    注意：
    这里的“分析”不是让 LLM 思考，而是一个轻量的预处理节点。
    它的任务是把问题规范化，并判断是否值得进入检索阶段。

    真实项目中，你也可以把这里换成：
    - Query rewriting
    - 意图分类
    - 工具选择 / router
    - 是否走 web search / database / vector search 的判定
    """
    normalized_question = state["user_question"].strip()
    intent: Literal["retrieve", "clarify"] = "retrieve" if len(normalized_question) >= 6 else "clarify"

    analysis = (
        f"原始问题长度={len(state['user_question'])}，规范化后长度={len(normalized_question)}。"
        "这个示例把长度足够的问题送入检索节点，把过短的问题送入澄清节点。"
    )

    return {
        "normalized_question": normalized_question,
        "analysis": analysis,
        "intent": intent,
        "trace": append_trace(state, "analyze_question"),
    }


def route_after_analysis(state: RagState) -> Literal["retrieve_context", "ask_for_clarification"]:
    """根据分析结果决定是否进入检索。"""
    return "retrieve_context" if state["intent"] == "retrieve" else "ask_for_clarification"


def build_demo_graph(runtime: RuntimeConfig, persist_directory: Path) -> Any:
    """构建 LangGraph 图，并把静态运行配置闭包进节点。

    这是一种很常见的写法：
    - 可变数据走 state
    - 固定配置走 closure
    """

    def retrieve_context(state: RagState) -> RagState:
        """第二个节点：真正调用项目里的 `search_chunks` 做检索。"""
        search_result = search_chunks(
            question=state["normalized_question"],
            collection_name=runtime.collection_name,
            persist_directory=persist_directory,
            top_k=runtime.top_k,
            embedder_type=runtime.embedder_type,
            embedding_model=runtime.embedding_model,
            hash_dimensions=runtime.hash_dimensions,
            max_context_chars=runtime.max_context_chars,
        )
        return {
            "search_result": search_result,
            "context_text": search_result.context_text,
            "sources": search_result.sources,
            "trace": append_trace(state, "retrieve_context"),
        }

    def route_after_retrieval(state: RagState) -> Literal["generate_answer", "ask_for_clarification"]:
        """检索到了内容才进入生成节点。"""
        search_result = state.get("search_result")
        if search_result and search_result.hits:
            return "generate_answer"
        return "ask_for_clarification"

    def generate_answer(state: RagState) -> RagState:
        """第三个节点：把检索结果交给真实 LLM 生成答案。"""
        search_result = state["search_result"]

        # 这里直接复用项目现有 QA 层中的 prompt 组装函数，避免重复发明一套格式。
        # 你可以把它理解为：LangGraph 负责编排，而 prompt 拼装仍由现有模块复用。
        user_prompt = (
            "Question analysis:\n"
            f"{state['analysis']}\n\n"
            f"{simple_qa_module._build_user_prompt(state['normalized_question'], search_result)}"
        )

        try:
            raw_response, answer = call_llm(runtime, user_prompt)
        except Exception as exc:  # pragma: no cover - 这里主要是给真实调试时提供可读错误
            raise RuntimeError(build_llm_help_message(runtime)) from exc

        return {
            "answer": answer,
            "raw_llm_response": raw_response,
            "trace": append_trace(state, "generate_answer"),
        }

    def ask_for_clarification(state: RagState) -> RagState:
        """兜底节点：没有足够信息时要求用户补充问题。"""
        answer = (
            "你的问题太短，示例图把它路由到了澄清节点。\n"
            "请尝试问得更具体一些，例如：\n"
            "- LangGraph 在 RAG 中有什么用？\n"
            "- 为什么 LangGraph 适合做 analysis -> retrieval -> generation？"
        )
        return {
            "answer": answer,
            "trace": append_trace(state, "ask_for_clarification"),
        }

    builder = StateGraph(cast(Any, RagState))
    builder.add_node("analyze_question", cast(Any, analyze_question))
    builder.add_node("retrieve_context", cast(Any, retrieve_context))
    builder.add_node("generate_answer", cast(Any, generate_answer))
    builder.add_node("ask_for_clarification", cast(Any, ask_for_clarification))

    builder.add_edge(START, "analyze_question")
    builder.add_conditional_edges(
        "analyze_question",
        cast(Any, route_after_analysis),
        {
            "retrieve_context": "retrieve_context",
            "ask_for_clarification": "ask_for_clarification",
        },
    )
    builder.add_conditional_edges(
        "retrieve_context",
        cast(Any, route_after_retrieval),
        {
            "generate_answer": "generate_answer",
            "ask_for_clarification": "ask_for_clarification",
        },
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("ask_for_clarification", END)
    return builder.compile()


def call_llm(runtime: RuntimeConfig, user_prompt: str) -> tuple[dict[str, Any], str]:
    """统一调用不同类型的 LLM 后端。

    这里故意把“访问模型”的细节集中在一个函数中：
    - LangGraph 节点只需要关心“我要生成答案”
    - 后端差异（OpenAI-compatible / Ollama）在这里被屏蔽掉
    """
    provider = runtime.llm_provider.strip().lower()

    if provider in {"openai", "openai-compatible", "openai_compatible", "openai-compatible-http"}:
        raw = simple_qa_module._call_openai_compatible(
            base_url=runtime.llm_base_url,
            model=runtime.llm_model,
            system_prompt=runtime.system_prompt,
            user_prompt=user_prompt,
            api_key=runtime.api_key,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
        )
        return raw, simple_qa_module._extract_openai_answer(raw).strip()

    if provider == "ollama":
        raw = simple_qa_module._call_ollama(
            base_url=runtime.llm_base_url,
            model=runtime.llm_model,
            system_prompt=runtime.system_prompt,
            user_prompt=user_prompt,
            temperature=runtime.temperature,
        )
        return raw, simple_qa_module._extract_ollama_answer(raw).strip()

    raise ValueError(f"Unsupported llm_provider: {runtime.llm_provider}")


def build_llm_help_message(runtime: RuntimeConfig) -> str:
    """当真实 LLM 请求失败时，给出更适合学习场景的提示。"""
    return (
        "LLM 调用失败。请检查下面几项：\n"
        f"- llm_provider={runtime.llm_provider}\n"
        f"- llm_base_url={runtime.llm_base_url}\n"
        f"- llm_model={runtime.llm_model}\n"
        "- 如果你要调用 OpenAI，请确认 OPENAI_API_KEY 已设置\n"
        "- 如果你要调用本地模型，请确认本地服务已经启动，并且接口路径兼容\n"
        "  * OpenAI-compatible: /v1/chat/completions\n"
        "  * Ollama: /api/chat"
    )


def print_runtime_summary(runtime: RuntimeConfig) -> None:
    """打印本次运行配置，方便理解这个示例到底连到了什么后端。"""
    print_kv("embedder", runtime.embedder_type)
    print_kv("embedding_model", runtime.embedding_model)
    print_kv("collection", runtime.collection_name)
    print_kv("top_k", runtime.top_k)
    print_kv("llm_provider", runtime.llm_provider)
    print_kv("llm_model", runtime.llm_model)
    print_kv("llm_base_url", runtime.llm_base_url)


def print_final_state(final_state: RagState) -> None:
    """把图的最终状态做成更适合教学的终端输出。"""
    print_kv("trace", " -> ".join(final_state.get("trace", [])))
    print_kv("analysis", final_state.get("analysis", "<none>"))
    print_kv("sources", final_state.get("sources", []))
    print_kv("context_preview", preview_text(final_state.get("context_text", ""), limit=260))
    print("answer:")
    print(final_state["answer"])

    raw_response = final_state.get("raw_llm_response") or {}
    if raw_response:
        usage = raw_response.get("usage")
        if usage:
            print_kv("raw_usage", usage)


def run_invoke_demo(app: Any) -> None:
    """执行一次真正的 analysis -> retrieval -> generation。"""
    print_title("invoke：完整 RAG 路径")
    question = "LangGraph 为什么适合做 analysis、retrieval 和 generation 的编排？"
    final_state = app.invoke({"user_question": question})
    print_kv("question", question)
    print_final_state(cast(RagState, final_state))


def run_stream_demo(app: Any) -> None:
    """演示 `stream()`，但用一个短问题走澄清分支，避免再额外调用一次真实 LLM。"""
    print_title("stream：逐步观察分支路由")
    for step_index, event in enumerate(app.stream({"user_question": "短问"}), start=1):
        node_name, update = next(iter(event.items()))
        print_kv(f"step_{step_index}", node_name)
        print_kv("update", update)
        print("-" * 80)


def main() -> None:
    print_title("debug_langgraph_rag")
    runtime = resolve_runtime_config()
    print_runtime_summary(runtime)

    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = Path(temp_dir) / "chroma"
        prepare_demo_collection(runtime, persist_directory)
        app = build_demo_graph(runtime, persist_directory)

        print_title("Mermaid 图结构")
        print(app.get_graph().draw_mermaid())

        run_invoke_demo(app)
        run_stream_demo(app)


if __name__ == "__main__":
    main()

