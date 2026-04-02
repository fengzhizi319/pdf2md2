"""LangGraph + 真实本地 Chroma 知识库示例。

适合学习：
- 如何让 LangGraph 直接连接已经落盘的本地 Chroma 向量库
- 如何把“分析问题 -> 校验知识库 -> 检索上下文 -> 生成答案”拆成多个节点
- 如何处理 collection 不存在、Chroma 目录不存在、检索无命中等真实工程问题
- 如何在不依赖真实 LLM 的情况下先把图跑通，再切到 OpenAI / 本地模型 / Ollama

和 `debug_langgraph_rag.py` 的区别：
- `debug_langgraph_rag.py` 会先创建临时 demo 向量库，适合学习“完整闭环”
- 当前脚本会直接连接你本地已经存在的 `data/chroma/`，适合学习“图如何接真实知识库”

默认行为：
- 默认连接仓库里的 `data/chroma/`
- 默认自动挑选一个可用 collection（也可手动指定）
- 默认使用 `fake` 生成模式，避免你必须先启动真实 LLM

当你想切到真实模型时，可以设置：
- `PDF2MD_RAG_LLM_PROVIDER=openai-compatible`
- `PDF2MD_RAG_LLM_PROVIDER=ollama`

推荐运行方式：

1) 先离线学习图结构（默认 fake LLM）：
   python examples/debug_langgraph_local_chroma.py

2) 指定真实 collection：
   PDF2MD_RAG_COLLECTION=understanding-lasso \
   python examples/debug_langgraph_local_chroma.py

3) 接 OpenAI-compatible：
   OPENAI_API_KEY=<your-key> \
   PDF2MD_RAG_LLM_PROVIDER=openai-compatible \
   PDF2MD_RAG_LLM_BASE_URL=https://api.openai.com \
   PDF2MD_RAG_LLM_MODEL=gpt-4o-mini \
   python examples/debug_langgraph_local_chroma.py

4) 接 Ollama：
   PDF2MD_RAG_LLM_PROVIDER=ollama \
   PDF2MD_RAG_LLM_BASE_URL=http://localhost:11434 \
   PDF2MD_RAG_LLM_MODEL=qwen3.5:0.8b \
   python examples/debug_langgraph_local_chroma.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from _debug_common import get_embedding_model_name, preview_text, print_kv, print_title
from langgraph.graph import END, START, StateGraph

import pdf2md_rag.simple_qa as simple_qa_module
from pdf2md_rag.config import DEFAULT_CHROMA_DIR
from pdf2md_rag.search import SearchResult, search_chunks


@dataclass(slots=True)
class RuntimeConfig:
    """这个示例运行时的固定配置。"""

    chroma_dir: Path
    collection_name: str
    available_collections: list[str]
    embedder_type: str
    embedding_model: str
    hash_dimensions: int
    top_k: int
    max_context_chars: int
    llm_provider: str
    llm_model: str
    llm_base_url: str
    api_key: str | None
    system_prompt: str
    temperature: float
    max_tokens: int


class LocalKbState(TypedDict, total=False):
    """在 LangGraph 图中流动的状态。"""

    user_question: str
    normalized_question: str
    analysis: str
    intent: Literal["retrieve", "clarify"]
    store_ok: bool
    collection_summary: str
    store_error: str
    retrieval_error: str
    search_result: SearchResult
    context_text: str
    sources: list[str]
    answer: str
    raw_llm_response: dict[str, Any]
    trace: list[str]


PREFERRED_COLLECTIONS = [
    "pdf-knowledge-base",
    "understanding-lasso",
    "understanding-lasso-hash-debug",
    "understanding-lasso-hash",
]


def append_trace(state: LocalKbState, step_name: str) -> list[str]:
    """给状态里的 trace 追加一步，帮助观察图的执行路径。"""
    return [*state.get("trace", []), step_name]


def list_collections(chroma_dir: Path) -> list[str]:
    """读取本地 Chroma 中当前可见的 collection 列表。"""
    if not chroma_dir.exists():
        return []

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    return sorted(collection.name for collection in client.list_collections())


def choose_default_collection(available_collections: list[str]) -> str:
    """从本地已有 collection 中选一个学习时最容易上手的默认值。"""
    for name in PREFERRED_COLLECTIONS:
        if name in available_collections:
            return name
    return available_collections[0] if available_collections else "pdf-knowledge-base"


def infer_embedder_defaults(collection_name: str) -> tuple[str, str]:
    """返回本示例统一使用的默认 embedding 设置。"""
    return "sentence-transformers", get_embedding_model_name()


def resolve_runtime_config() -> RuntimeConfig:
    """从环境变量和本地知识库状态中解析配置。"""
    chroma_dir = Path(os.getenv("PDF2MD_RAG_CHROMA_DIR", str(DEFAULT_CHROMA_DIR))).expanduser().resolve()
    available_collections = list_collections(chroma_dir)

    requested_collection = os.getenv("PDF2MD_RAG_COLLECTION", "").strip()
    collection_name = requested_collection or choose_default_collection(available_collections)

    inferred_embedder, inferred_model = infer_embedder_defaults(collection_name)

    provider = os.getenv("PDF2MD_RAG_LLM_PROVIDER", "fake").strip().lower()
    api_key = os.getenv("PDF2MD_RAG_API_KEY") or os.getenv("OPENAI_API_KEY")

    if provider in {"fake", "mock", "offline"}:
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "fake-rag-explainer")
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", "<disabled>")
    elif provider in {"openai", "openai-compatible", "openai_compatible", "openai-compatible-http"}:
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "gpt-4o-mini")
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", "https://api.openai.com")
    elif provider == "ollama":
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "qwen3.5:0.8b")
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    else:
        raise ValueError(f"Unsupported llm provider: {provider}")

    return RuntimeConfig(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        available_collections=available_collections,
        embedder_type=os.getenv("PDF2MD_RAG_EMBEDDER", inferred_embedder),
        embedding_model=os.getenv("PDF2MD_RAG_EMBEDDING_MODEL", inferred_model),
        hash_dimensions=int(os.getenv("PDF2MD_RAG_HASH_DIMENSIONS", "384")),
        top_k=int(os.getenv("PDF2MD_RAG_TOP_K", "3")),
        max_context_chars=int(os.getenv("PDF2MD_RAG_MAX_CONTEXT_CHARS", "2200")),
        llm_provider=provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        api_key=api_key,
        system_prompt=(
            "You are a careful RAG assistant inside a LangGraph local-Chroma tutorial. "
            "Read the question analysis and the retrieved context carefully. "
            "Answer only from the retrieved context. If the context is insufficient, say so clearly. "
            "Always end with a Sources line."
        ),
        temperature=float(os.getenv("PDF2MD_RAG_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("PDF2MD_RAG_MAX_TOKENS", "700")),
    )


def analyze_question(state: LocalKbState) -> LocalKbState:
    """第一个节点：分析用户问题。

    这里仍然保持“轻分析”原则：
    - 不强依赖 LLM
    - 只负责清理输入、写一段分析说明、决定是否值得进入检索
    """
    normalized_question = state["user_question"].strip()
    intent: Literal["retrieve", "clarify"] = "retrieve" if len(normalized_question) >= 4 else "clarify"
    analysis = (
        f"原始问题长度={len(state['user_question'])}，规范化后长度={len(normalized_question)}。"
        "本示例把足够具体的问题送入检索节点，把过短的问题送入澄清节点。"
    )
    return {
        "normalized_question": normalized_question,
        "analysis": analysis,
        "intent": intent,
        "trace": append_trace(state, "analyze_question"),
    }


def route_after_analysis(state: LocalKbState) -> Literal["validate_store", "ask_for_clarification"]:
    """问题太短时先走澄清，否则继续检查知识库。"""
    return "validate_store" if state["intent"] == "retrieve" else "ask_for_clarification"


def build_demo_graph(runtime: RuntimeConfig) -> Any:
    """构建连接真实本地 Chroma 的 LangGraph。"""

    def validate_store(state: LocalKbState) -> LocalKbState:
        """第二个节点：校验本地 Chroma 和 collection 状态。"""
        chroma_dir = runtime.chroma_dir
        if not chroma_dir.exists():
            return {
                "store_ok": False,
                "store_error": (
                    f"Chroma 目录不存在：{chroma_dir}\n"
                    "请先运行 ingest，把 PDF 写入本地向量库。"
                ),
                "trace": append_trace(state, "validate_store"),
            }

        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        available_names = sorted(collection.name for collection in client.list_collections())
        if runtime.collection_name not in available_names:
            return {
                "store_ok": False,
                "store_error": (
                    f"找不到 collection: {runtime.collection_name}\n"
                    f"当前可用 collection: {available_names or ['<none>']}\n"
                    "你可以通过环境变量 PDF2MD_RAG_COLLECTION 指定正确的 collection。"
                ),
                "trace": append_trace(state, "validate_store"),
            }

        collection = client.get_collection(runtime.collection_name)
        count = collection.count()
        if count <= 0:
            return {
                "store_ok": False,
                "store_error": (
                    f"collection '{runtime.collection_name}' 目前是空的。\n"
                    "请重新 ingest 文档，确认向量已经成功写入。"
                ),
                "trace": append_trace(state, "validate_store"),
            }

        return {
            "store_ok": True,
            "collection_summary": (
                f"chroma_dir={chroma_dir} | collection={runtime.collection_name} | vector_count={count}"
            ),
            "trace": append_trace(state, "validate_store"),
        }

    def route_after_store_validation(
        state: LocalKbState,
    ) -> Literal["retrieve_context", "report_store_error"]:
        """校验通过就检索，否则输出可读错误。"""
        return "retrieve_context" if state.get("store_ok") else "report_store_error"

    def retrieve_context(state: LocalKbState) -> LocalKbState:
        """第三个节点：真正从本地 Chroma 做检索。"""
        try:
            search_result = search_chunks(
                question=state["normalized_question"],
                collection_name=runtime.collection_name,
                persist_directory=runtime.chroma_dir,
                top_k=runtime.top_k,
                embedder_type=runtime.embedder_type,
                embedding_model=runtime.embedding_model,
                hash_dimensions=runtime.hash_dimensions,
                max_context_chars=runtime.max_context_chars,
            )
        except Exception as exc:
            return {
                "retrieval_error": (
                    "检索失败。常见原因包括：\n"
                    f"- embedder_type={runtime.embedder_type}\n"
                    f"- embedding_model={runtime.embedding_model}\n"
                    "- 当前查询使用的 embedder 与当初写入 collection 时不一致\n"
                    f"- 原始异常：{exc}"
                ),
                "trace": append_trace(state, "retrieve_context"),
            }

        return {
            "search_result": search_result,
            "context_text": search_result.context_text,
            "sources": search_result.sources,
            "trace": append_trace(state, "retrieve_context"),
        }

    def route_after_retrieval(
        state: LocalKbState,
    ) -> Literal["generate_answer", "report_retrieval_error"]:
        """检索成功才进入生成，否则输出检索错误或无命中提示。"""
        if state.get("retrieval_error"):
            return "report_retrieval_error"
        search_result = state.get("search_result")
        if search_result and search_result.hits:
            return "generate_answer"
        return "report_retrieval_error"

    def generate_answer(state: LocalKbState) -> LocalKbState:
        """第四个节点：基于真实本地知识库的检索结果生成答案。"""
        search_result = state["search_result"]

        user_prompt = (
            "Question analysis:\n"
            f"{state['analysis']}\n\n"
            f"{simple_qa_module._build_user_prompt(state['normalized_question'], search_result)}"
        )

        try:
            raw_response, answer = call_llm(runtime, state, user_prompt)
        except Exception as exc:  # pragma: no cover - 真实联调时更重要
            raise RuntimeError(build_llm_help_message(runtime)) from exc

        return {
            "answer": answer,
            "raw_llm_response": raw_response,
            "trace": append_trace(state, "generate_answer"),
        }

    def ask_for_clarification(state: LocalKbState) -> LocalKbState:
        """问题过短时先要求澄清。"""
        return {
            "answer": (
                "你的问题太短，图把它路由到了澄清节点。\n"
                "可以尝试问得更具体，例如：\n"
                "- What is Lasso?\n"
                "- Lasso 这篇论文主要讲什么？\n"
                "- 这篇文档里如何描述 lookup argument protocol?"
            ),
            "trace": append_trace(state, "ask_for_clarification"),
        }

    def report_store_error(state: LocalKbState) -> LocalKbState:
        """把知识库层面的错误包装成用户可读答案。"""
        available_text = ", ".join(runtime.available_collections) or "<none>"
        answer = (
            "本地知识库校验没有通过。\n\n"
            f"原因：\n{state.get('store_error', '<unknown>')}\n\n"
            "建议排查：\n"
            f"- 当前 chroma_dir: {runtime.chroma_dir}\n"
            f"- 当前 collection: {runtime.collection_name}\n"
            f"- 本地可见 collections: {available_text}\n"
            "- 如果你还没 ingest 过文档，请先执行 pdf2md-rag ingest ..."
        )
        return {
            "answer": answer,
            "trace": append_trace(state, "report_store_error"),
        }

    def report_retrieval_error(state: LocalKbState) -> LocalKbState:
        """把检索失败或无命中的情况包装成教学型输出。"""
        if state.get("retrieval_error"):
            answer = state["retrieval_error"]
        else:
            answer = (
                "检索已成功执行，但没有拿到命中结果。\n"
                "这通常意味着：\n"
                "- 问题和知识库内容相关性太弱\n"
                "- top_k 太小\n"
                "- 当前 embedder 与写库时不一致"
            )
        return {
            "answer": answer,
            "trace": append_trace(state, "report_retrieval_error"),
        }

    builder = StateGraph(cast(Any, LocalKbState))
    builder.add_node("analyze_question", cast(Any, analyze_question))
    builder.add_node("validate_store", cast(Any, validate_store))
    builder.add_node("retrieve_context", cast(Any, retrieve_context))
    builder.add_node("generate_answer", cast(Any, generate_answer))
    builder.add_node("ask_for_clarification", cast(Any, ask_for_clarification))
    builder.add_node("report_store_error", cast(Any, report_store_error))
    builder.add_node("report_retrieval_error", cast(Any, report_retrieval_error))

    builder.add_edge(START, "analyze_question")
    builder.add_conditional_edges(
        "analyze_question",
        cast(Any, route_after_analysis),
        {
            "validate_store": "validate_store",
            "ask_for_clarification": "ask_for_clarification",
        },
    )
    builder.add_conditional_edges(
        "validate_store",
        cast(Any, route_after_store_validation),
        {
            "retrieve_context": "retrieve_context",
            "report_store_error": "report_store_error",
        },
    )
    builder.add_conditional_edges(
        "retrieve_context",
        cast(Any, route_after_retrieval),
        {
            "generate_answer": "generate_answer",
            "report_retrieval_error": "report_retrieval_error",
        },
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("ask_for_clarification", END)
    builder.add_edge("report_store_error", END)
    builder.add_edge("report_retrieval_error", END)
    return builder.compile()


def call_llm(
    runtime: RuntimeConfig,
    state: LocalKbState,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    """统一调用 fake / OpenAI-compatible / Ollama 三种生成方式。"""
    provider = runtime.llm_provider.strip().lower()
    search_result = state["search_result"]

    if provider in {"fake", "mock", "offline"}:
        raw = {
            "provider": "fake",
            "model": runtime.llm_model,
            "reason": "No live LLM backend required for this learning example.",
        }
        answer = build_fake_answer(state, search_result)
        return raw, answer

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


def build_fake_answer(state: LocalKbState, search_result: SearchResult) -> str:
    """构造一个离线可用的教学型答案。

    目的不是替代真实 LLM，而是让你在没有启动模型服务时，
    仍然可以观察到“图已经完成了分析 -> 检索 -> 生成”的结构。
    """
    if not search_result.hits:
        return "我没有在本地知识库里检索到相关内容。\n\nSources: <none>"

    top_hit = search_result.hits[0]
    return (
        "[fake-llm] 下面这段回答不是模型生成，而是教学模式下的本地格式化结果。\n\n"
        f"问题分析：{state['analysis']}\n"
        f"最相关来源：{top_hit.citation}\n"
        f"最高相关片段预览：{preview_text(top_hit.text, limit=220)}\n\n"
        "如果你想体验真实 LLM，把 `PDF2MD_RAG_LLM_PROVIDER` 改成 `openai-compatible` 或 `ollama` 即可。\n\n"
        f"Sources: {', '.join(search_result.sources) if search_result.sources else '<none>'}"
    )


def build_llm_help_message(runtime: RuntimeConfig) -> str:
    """真实 LLM 请求失败时输出更适合学习的提示。"""
    return (
        "LLM 调用失败。请检查下面几项：\n"
        f"- llm_provider={runtime.llm_provider}\n"
        f"- llm_base_url={runtime.llm_base_url}\n"
        f"- llm_model={runtime.llm_model}\n"
        "- 如果你要调用 OpenAI，请确认 OPENAI_API_KEY 已设置\n"
        "- 如果你要调用本地模型，请确认服务已经启动，并且接口路径兼容\n"
        "  * OpenAI-compatible: /v1/chat/completions\n"
        "  * Ollama: /api/chat"
    )


def print_runtime_summary(runtime: RuntimeConfig) -> None:
    """打印本次示例的关键运行参数。"""
    print_kv("chroma_dir", runtime.chroma_dir)
    print_kv("collection", runtime.collection_name)
    print_kv("collections", runtime.available_collections)
    print_kv("embedder", runtime.embedder_type)
    print_kv("embedding_model", runtime.embedding_model)
    print_kv("top_k", runtime.top_k)
    print_kv("llm_provider", runtime.llm_provider)
    print_kv("llm_model", runtime.llm_model)
    print_kv("llm_base_url", runtime.llm_base_url)


def print_final_state(final_state: LocalKbState) -> None:
    """用更适合学习的方式打印最终状态。"""
    print_kv("trace", " -> ".join(final_state.get("trace", [])))
    print_kv("analysis", final_state.get("analysis", "<none>"))
    print_kv("collection_summary", final_state.get("collection_summary", "<none>"))
    print_kv("sources", final_state.get("sources", []))
    print_kv("context_preview", preview_text(final_state.get("context_text", ""), limit=300))
    print("answer:")
    print(final_state["answer"])

    raw_response = final_state.get("raw_llm_response") or {}
    if raw_response:
        print_kv("raw_response", raw_response)


def run_invoke_demo(app: Any) -> None:
    """执行一次完整的真实本地知识库查询。"""
    print_title("invoke：真实本地 Chroma 知识库")
    question = os.getenv("PDF2MD_RAG_DEMO_QUESTION", "What is Lasso?")
    final_state = app.invoke({"user_question": question})
    print_kv("question", question)
    print_final_state(cast(LocalKbState, final_state))


def run_stream_demo(app: Any) -> None:
    """演示 stream，但走澄清分支，方便观察路由。"""
    print_title("stream：逐步观察分支路由")
    for step_index, event in enumerate(app.stream({"user_question": "短问"}), start=1):
        node_name, update = next(iter(event.items()))
        print_kv(f"step_{step_index}", node_name)
        print_kv("update", update)
        print("-" * 80)


def main() -> None:
    print_title("debug_langgraph_local_chroma")
    runtime = resolve_runtime_config()
    print_runtime_summary(runtime)

    app = build_demo_graph(runtime)

    print_title("Mermaid 图结构")
    print(app.get_graph().draw_mermaid())

    run_invoke_demo(app)
    run_stream_demo(app)


if __name__ == "__main__":
    main()

