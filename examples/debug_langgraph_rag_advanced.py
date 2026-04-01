"""更像 production 的 LangGraph RAG 示例。

这个脚本在前面的 LangGraph 示例基础上，再加入四类真实工程里很常见的节点：

- `rewrite_question`：把原始问题改写成更适合检索的查询
- `grade_retrieval`：检查这次检索是否真的拿到了足够相关的上下文
- `retry_search`：如果检索质量不够，就换一个查询再试一次
- `fallback_answer`：多次重试后仍然不够好，就优雅降级，而不是胡乱回答

适合学习：
- 为什么 production RAG 往往不是“一次检索 + 一次生成”就结束
- 为什么“检索成功”不等于“检索质量足够好”
- LangGraph 如何表达循环、重试和兜底
- 如何先用离线 fake LLM 学图，再切到真实 OpenAI-compatible / Ollama

默认行为：
- 默认使用临时 demo Chroma，避免依赖外部数据
- 默认使用 `hash` embedder，保证示例离线、稳定、可重复
- 默认使用 `fake` 生成模式，方便先关注图编排

如果你想切到真实 LLM，可以设置：
- `PDF2MD_RAG_LLM_PROVIDER=openai-compatible`
- `PDF2MD_RAG_LLM_PROVIDER=ollama`
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from _debug_common import preview_text, print_kv, print_title
from langgraph.graph import END, START, StateGraph

import pdf2md_rag.simple_qa as simple_qa_module
from pdf2md_rag.embeddings import build_embedder
from pdf2md_rag.models import Chunk
from pdf2md_rag.search import SearchResult, search_chunks
from pdf2md_rag.vectorstore import upsert_chunks


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "into",
    "that",
    "the",
    "this",
    "what",
    "when",
    "why",
    "with",
    "workflow",
    "system",
}


@dataclass(slots=True)
class RuntimeConfig:
    """本示例运行时的固定配置。"""

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
    max_retries: int
    min_overlap_for_good: int


class ProductionRagState(TypedDict, total=False):
    """在高级 RAG 图里流动的状态。"""

    user_question: str
    normalized_question: str
    query_focus: Literal["grounding", "robustness", "hardware", "general"]
    analysis: str
    intent: Literal["retrieve", "clarify"]
    candidate_queries: list[str]
    active_question: str
    retry_count: int
    max_retries: int
    retrieval_grade: Literal["strong", "weak", "empty"]
    grade_reason: str
    decision: Literal["generate_answer", "retry_search", "fallback_answer", "ask_for_clarification"]
    fallback_reason: str
    search_result: SearchResult
    context_text: str
    sources: list[str]
    answer: str
    raw_llm_response: dict[str, Any]
    trace: list[str]


def build_learning_chunks() -> list[Chunk]:
    """构造一组专门用于讲解 advanced RAG graph 的 demo chunk。"""
    source_name = "Production RAG Notes"
    source_path = "/virtual/production-rag-notes.md"

    payloads = [
        (
            "adv-rag-1",
            "A production RAG workflow often separates analyze, rewrite, retrieve, grade, generate, and fallback into explicit steps. "
            "LangGraph is a good fit because graphs make branching and retries visible.",
            "RAG Graph Overview",
            1,
        ),
        (
            "adv-rag-2",
            "Query rewriting improves retrieval when the user question is vague. A rewrite can add domain terms such as retrieval, context, citations, and hallucinations.",
            "Query Rewriting",
            2,
        ),
        (
            "adv-rag-3",
            "Retrieval grading checks whether the returned chunks actually match the question. Useful signals include keyword overlap, source quality, and whether the context supports citations.",
            "Retrieval Grading",
            3,
        ),
        (
            "adv-rag-4",
            "Retry search is useful when the first retrieval is weak. The second attempt often expands the query with more concrete terms like retry, fallback, relevance, and source labels.",
            "Retry Search",
            4,
        ),
        (
            "adv-rag-5",
            "Fallback answers should clearly explain that the knowledge base does not contain enough evidence. They should avoid hallucination and suggest a better query to the user.",
            "Fallback Answer",
            5,
        ),
        (
            "adv-rag-6",
            "RAG reduces hallucinations by retrieving context before generation and by requiring the answer to cite the supporting sources.",
            "Grounding and Citations",
            6,
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


def append_trace(state: ProductionRagState, step_name: str) -> list[str]:
    """把当前步骤追加到 trace，便于观察图的执行轨迹。"""
    return [*state.get("trace", []), step_name]


def tokenize_for_overlap(text: str) -> set[str]:
    """把文本转成一组适合做粗粒度重叠判断的 token。"""
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    tokens = set("".join(cleaned).split())
    return {token for token in tokens if len(token) >= 4 and token not in STOPWORDS}


def infer_focus(question: str) -> Literal["grounding", "robustness", "hardware", "general"]:
    """根据用户问题推断这次更关注哪类主题。"""
    lowered = question.lower()
    if any(keyword in lowered for keyword in ["halluc", "citation", "source", "ground", "context"]):
        return "grounding"
    if any(keyword in lowered for keyword in ["retry", "retries", "fallback", "robust", "failure", "recover"]):
        return "robustness"
    if any(keyword in lowered for keyword in ["gpu", "memory", "hardware", "latency", "deploy"]):
        return "hardware"
    return "general"


def build_candidate_queries(question: str, focus: str) -> list[str]:
    """为重试准备多组候选查询。

    设计思路：
    - 第一个查询尽量保留用户原意
    - 后续查询逐步加入更明确的领域词
    - 如果还是失败，就交给 fallback，而不是无限重试
    """
    candidates = [question]

    if focus == "grounding":
        candidates.append(f"{question} retrieval context citations hallucinations sources")
        candidates.append(f"{question} RAG grounding citations supporting context")
    elif focus == "robustness":
        candidates.append(f"{question} retry fallback answer retrieval relevance")
        candidates.append(f"{question} LangGraph retries fallback weak retrieval recovery")
    elif focus == "hardware":
        candidates.append(f"{question} hardware gpu memory deployment requirements")
        candidates.append(f"{question} serving model hardware latency memory")
    else:
        candidates.append(f"{question} LangGraph production RAG rewrite retrieve grade fallback")
        candidates.append(f"{question} retrieval grading retry answer citations")

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def resolve_runtime_config() -> RuntimeConfig:
    """解析一组对学习友好的默认配置。"""
    provider = os.getenv("PDF2MD_RAG_LLM_PROVIDER", "fake").strip().lower()
    api_key = os.getenv("PDF2MD_RAG_API_KEY") or os.getenv("OPENAI_API_KEY")

    if provider in {"fake", "mock", "offline"}:
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "fake-production-rag")
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", "<disabled>")
    elif provider in {"openai", "openai-compatible", "openai_compatible", "openai-compatible-http"}:
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "gpt-4o-mini")
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", "https://api.openai.com")
    elif provider == "ollama":
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "qwen2.5:3b")
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    else:
        raise ValueError(f"Unsupported llm provider: {provider}")

    return RuntimeConfig(
        embedder_type=os.getenv("PDF2MD_RAG_EMBEDDER", "hash"),
        embedding_model=os.getenv("PDF2MD_RAG_EMBEDDING_MODEL", "unused"),
        hash_dimensions=int(os.getenv("PDF2MD_RAG_HASH_DIMENSIONS", "128")),
        collection_name=os.getenv("PDF2MD_RAG_COLLECTION", "debug-langgraph-rag-advanced"),
        top_k=int(os.getenv("PDF2MD_RAG_TOP_K", "3")),
        max_context_chars=int(os.getenv("PDF2MD_RAG_MAX_CONTEXT_CHARS", "1800")),
        llm_provider=provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        api_key=api_key,
        system_prompt=(
            "You are a careful RAG assistant inside an advanced LangGraph tutorial. "
            "Use only the retrieved context. If evidence is insufficient, say so clearly. "
            "Always end with a Sources line."
        ),
        temperature=float(os.getenv("PDF2MD_RAG_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("PDF2MD_RAG_MAX_TOKENS", "700")),
        max_retries=int(os.getenv("PDF2MD_RAG_MAX_RETRIES", "2")),
        min_overlap_for_good=int(os.getenv("PDF2MD_RAG_MIN_OVERLAP_FOR_GOOD", "2")),
    )


def prepare_demo_collection(runtime: RuntimeConfig, persist_directory: Path) -> None:
    """把 demo chunk 写入临时 Chroma。"""
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


def analyze_question(state: ProductionRagState) -> ProductionRagState:
    """第一个节点：分析用户问题，并初始化重试相关状态。"""
    normalized_question = state["user_question"].strip()
    query_focus = infer_focus(normalized_question)
    intent: Literal["retrieve", "clarify"] = "retrieve" if len(normalized_question) >= 6 else "clarify"
    candidate_queries = build_candidate_queries(normalized_question, query_focus)
    analysis = (
        f"focus={query_focus}，候选查询数={len(candidate_queries)}。"
        "这个示例会先尝试原始问题，再在检索较弱时切换到改写后的查询。"
    )
    return {
        "normalized_question": normalized_question,
        "query_focus": query_focus,
        "analysis": analysis,
        "intent": intent,
        "candidate_queries": candidate_queries,
        "retry_count": 0,
        "max_retries": state.get("max_retries", 0),
        "trace": append_trace(state, "analyze_question"),
    }


def route_after_analysis(state: ProductionRagState) -> Literal["rewrite_question", "ask_for_clarification"]:
    """问题足够具体时进入 rewrite 节点，否则先澄清。"""
    return "rewrite_question" if state["intent"] == "retrieve" else "ask_for_clarification"


def build_demo_graph(runtime: RuntimeConfig, persist_directory: Path) -> Any:
    """构建一个带 rewrite / grade / retry / fallback 的高级 RAG 图。"""

    def rewrite_question(state: ProductionRagState) -> ProductionRagState:
        """第二个节点：根据当前重试轮次选择 active query。"""
        candidate_queries = state["candidate_queries"]
        retry_count = state.get("retry_count", 0)
        index = min(retry_count, len(candidate_queries) - 1)
        active_question = candidate_queries[index]
        return {
            "active_question": active_question,
            "trace": append_trace(state, "rewrite_question"),
        }

    def retrieve_context(state: ProductionRagState) -> ProductionRagState:
        """第三个节点：执行实际检索。"""
        search_result = search_chunks(
            question=state["active_question"],
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

    def grade_retrieval(state: ProductionRagState) -> ProductionRagState:
        """第四个节点：给检索结果打一个工程化的“够不够用”分数。

        这里不用 LLM 做 grading，而是使用稳定的规则：
        - 查看 active query 和命中文本的关键词重叠度
        - 如果重叠太弱且还有剩余重试次数，就路由到 retry_search
        - 如果多次重试后仍然不够好，就路由到 fallback_answer
        """
        search_result = state.get("search_result")
        if not search_result or not search_result.hits:
            return {
                "retrieval_grade": "empty",
                "grade_reason": "没有检索到任何 hits。",
                "decision": "retry_search" if has_remaining_retry(state) else "fallback_answer",
                "fallback_reason": "知识库没有返回任何相关片段。",
                "trace": append_trace(state, "grade_retrieval"),
            }

        query_tokens = tokenize_for_overlap(state["active_question"])
        best_overlap = 0
        best_citation = "<none>"
        for hit in search_result.hits:
            overlap = len(query_tokens & tokenize_for_overlap(hit.text))
            if overlap > best_overlap:
                best_overlap = overlap
                best_citation = hit.citation

        if best_overlap >= runtime.min_overlap_for_good:
            decision: Literal["generate_answer", "retry_search", "fallback_answer", "ask_for_clarification"] = "generate_answer"
            retrieval_grade: Literal["strong", "weak", "empty"] = "strong"
            fallback_reason = ""
        elif has_remaining_retry(state):
            decision = cast(
                Literal["generate_answer", "retry_search", "fallback_answer", "ask_for_clarification"],
                "retry_search",
            )
            retrieval_grade = cast(Literal["strong", "weak", "empty"], "weak")
            fallback_reason = ""
        else:
            decision = cast(
                Literal["generate_answer", "retry_search", "fallback_answer", "ask_for_clarification"],
                "fallback_answer",
            )
            retrieval_grade = cast(Literal["strong", "weak", "empty"], "weak")
            fallback_reason = "重试次数已用尽，但检索结果的关键词覆盖度仍然不足。"

        return {
            "retrieval_grade": retrieval_grade,
            "grade_reason": (
                f"best_overlap={best_overlap}，best_citation={best_citation}，"
                f"min_overlap_for_good={runtime.min_overlap_for_good}。"
            ),
            "decision": decision,
            "fallback_reason": fallback_reason,
            "trace": append_trace(state, "grade_retrieval"),
        }

    def route_after_grading(
        state: ProductionRagState,
    ) -> Literal["generate_answer", "retry_search", "fallback_answer"]:
        """根据 grading 结果选择下一步。"""
        return cast(Literal["generate_answer", "retry_search", "fallback_answer"], state["decision"])

    def retry_search(state: ProductionRagState) -> ProductionRagState:
        """第五个节点：增加 retry_count，并准备下一次改写查询。"""
        next_retry_count = state.get("retry_count", 0) + 1
        candidate_queries = state["candidate_queries"]
        next_index = min(next_retry_count, len(candidate_queries) - 1)
        next_query = candidate_queries[next_index]
        return {
            "retry_count": next_retry_count,
            "active_question": next_query,
            "trace": append_trace(state, "retry_search"),
        }

    def generate_answer(state: ProductionRagState) -> ProductionRagState:
        """第六个节点：检索质量足够时生成答案。"""
        search_result = state["search_result"]
        user_prompt = (
            "Question analysis:\n"
            f"{state['analysis']}\n"
            f"Retrieval grade: {state.get('retrieval_grade')}\n"
            f"Grade reason: {state.get('grade_reason')}\n"
            f"Retry count: {state.get('retry_count', 0)}\n\n"
            f"{simple_qa_module._build_user_prompt(state['normalized_question'], search_result)}"
        )
        raw_response, answer = call_llm(runtime, state, user_prompt)
        return {
            "answer": answer,
            "raw_llm_response": raw_response,
            "trace": append_trace(state, "generate_answer"),
        }

    def fallback_answer(state: ProductionRagState) -> ProductionRagState:
        """第七个节点：多次重试仍然不理想时优雅降级。"""
        search_result = state.get("search_result")
        sources = search_result.sources if search_result else []
        answer = (
            "我没有足够把握直接回答这个问题，因为当前知识库里缺少足够强的证据。\n\n"
            f"原因：{state.get('fallback_reason', '检索相关性不足。')}\n"
            f"最后一次检索评分：{state.get('retrieval_grade', '<none>')}\n"
            f"评分解释：{state.get('grade_reason', '<none>')}\n\n"
            "建议你尝试：\n"
            "- 把问题改得更贴近知识库里的术语\n"
            "- 明确加入 retrieval / citations / fallback / retry 等关键词\n"
            "- 缩小问题范围，只问一个子问题\n\n"
            f"Sources: {', '.join(sources) if sources else '<none>'}"
        )
        return {
            "answer": answer,
            "trace": append_trace(state, "fallback_answer"),
        }

    def ask_for_clarification(state: ProductionRagState) -> ProductionRagState:
        """问题太短时先要求补充。"""
        return {
            "answer": (
                "你的问题太短，图把它路由到了澄清节点。\n"
                "你可以试试这些更具体的问题：\n"
                "- How does LangGraph organize a production RAG workflow?\n"
                "- How can rewrite and retry improve retrieval?\n"
                "- What should a fallback answer do?"
            ),
            "trace": append_trace(state, "ask_for_clarification"),
        }

    builder = StateGraph(cast(Any, ProductionRagState))
    builder.add_node("analyze_question", cast(Any, analyze_question))
    builder.add_node("rewrite_question", cast(Any, rewrite_question))
    builder.add_node("retrieve_context", cast(Any, retrieve_context))
    builder.add_node("grade_retrieval", cast(Any, grade_retrieval))
    builder.add_node("retry_search", cast(Any, retry_search))
    builder.add_node("generate_answer", cast(Any, generate_answer))
    builder.add_node("fallback_answer", cast(Any, fallback_answer))
    builder.add_node("ask_for_clarification", cast(Any, ask_for_clarification))

    builder.add_edge(START, "analyze_question")
    builder.add_conditional_edges(
        "analyze_question",
        cast(Any, route_after_analysis),
        {
            "rewrite_question": "rewrite_question",
            "ask_for_clarification": "ask_for_clarification",
        },
    )
    builder.add_edge("rewrite_question", "retrieve_context")
    builder.add_edge("retrieve_context", "grade_retrieval")
    builder.add_conditional_edges(
        "grade_retrieval",
        cast(Any, route_after_grading),
        {
            "generate_answer": "generate_answer",
            "retry_search": "retry_search",
            "fallback_answer": "fallback_answer",
        },
    )
    builder.add_edge("retry_search", "rewrite_question")
    builder.add_edge("generate_answer", END)
    builder.add_edge("fallback_answer", END)
    builder.add_edge("ask_for_clarification", END)
    return builder.compile()


def has_remaining_retry(state: ProductionRagState) -> bool:
    """判断是否还允许继续重试。"""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 0)
    candidate_queries = state.get("candidate_queries", [])
    return retry_count < max_retries and retry_count + 1 < len(candidate_queries)


def build_fake_answer(state: ProductionRagState, search_result: SearchResult) -> str:
    """构造一个适合教学的 fake LLM 答案。"""
    top_hit = search_result.hits[0] if search_result.hits else None
    top_citation = top_hit.citation if top_hit else "<none>"
    top_preview = preview_text(top_hit.text, limit=180) if top_hit else "<none>"
    return (
        "[fake-llm] 这是一段离线教学答案，用来证明图已经成功走到了 generate_answer。\n\n"
        f"原始问题：{state['normalized_question']}\n"
        f"实际查询：{state['active_question']}\n"
        f"重试次数：{state.get('retry_count', 0)}\n"
        f"检索评分：{state.get('retrieval_grade')}\n"
        f"评分解释：{state.get('grade_reason')}\n"
        f"最相关来源：{top_citation}\n"
        f"最相关片段预览：{top_preview}\n\n"
        "在真实项目里，这里通常会把检索结果交给真正的 LLM，由模型把证据整合成更自然的最终回答。\n\n"
        f"Sources: {', '.join(search_result.sources) if search_result.sources else '<none>'}"
    )


def call_llm(
    runtime: RuntimeConfig,
    state: ProductionRagState,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    """统一调用 fake / OpenAI-compatible / Ollama 三种后端。"""
    provider = runtime.llm_provider.strip().lower()

    if provider in {"fake", "mock", "offline"}:
        raw = {
            "provider": "fake",
            "model": runtime.llm_model,
            "reason": "Offline tutorial mode.",
        }
        return raw, build_fake_answer(state, state["search_result"])

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


def print_runtime_summary(runtime: RuntimeConfig) -> None:
    """打印本次运行配置。"""
    print_kv("embedder", runtime.embedder_type)
    print_kv("embedding_model", runtime.embedding_model)
    print_kv("collection", runtime.collection_name)
    print_kv("top_k", runtime.top_k)
    print_kv("max_retries", runtime.max_retries)
    print_kv("good_overlap", runtime.min_overlap_for_good)
    print_kv("llm_provider", runtime.llm_provider)
    print_kv("llm_model", runtime.llm_model)
    print_kv("llm_base_url", runtime.llm_base_url)


def print_final_state(final_state: ProductionRagState) -> None:
    """把最终状态整理成适合学习的输出。"""
    print_kv("trace", " -> ".join(final_state.get("trace", [])))
    print_kv("active_question", final_state.get("active_question", "<none>"))
    print_kv("retry_count", final_state.get("retry_count", 0))
    print_kv("retrieval_grade", final_state.get("retrieval_grade", "<none>"))
    print_kv("grade_reason", final_state.get("grade_reason", "<none>"))
    print_kv("sources", final_state.get("sources", []))
    print_kv("context_preview", preview_text(final_state.get("context_text", ""), limit=260))
    print("answer:")
    print(final_state["answer"])


def run_single_case(app: Any, question: str, title: str) -> None:
    """执行一个问题，并打印结构化结果。"""
    print_title(title)
    final_state = cast(ProductionRagState, app.invoke({"user_question": question, "max_retries": 2}))
    print_kv("question", question)
    print_final_state(final_state)


def run_invoke_demos(app: Any) -> None:
    """跑三组场景：直接命中、重试后命中、走 fallback。"""
    run_single_case(
        app,
        question="What does retrieval grading check?",
        title="invoke：直接命中并生成答案",
    )
    run_single_case(
        app,
        question="How do retries help retrieval?",
        title="invoke：第一次较弱，重试后改善",
    )
    run_single_case(
        app,
        question="What GPU memory is required to deploy this stack?",
        title="invoke：重试后仍然不足，走 fallback",
    )


def run_stream_demo(app: Any) -> None:
    """用 stream 观察“重试后改善”的完整节点流转。"""
    print_title("stream：逐步观察 retry 路径")
    question = "How do retries help retrieval?"
    for step_index, event in enumerate(app.stream({"user_question": question, "max_retries": 2}), start=1):
        node_name, update = next(iter(event.items()))
        print_kv(f"step_{step_index}", node_name)
        print_kv("update", update)
        print("-" * 80)


def main() -> None:
    print_title("debug_langgraph_rag_advanced")
    runtime = resolve_runtime_config()
    print_runtime_summary(runtime)

    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = Path(temp_dir) / "chroma"
        prepare_demo_collection(runtime, persist_directory)
        app = build_demo_graph(runtime, persist_directory)

        print_title("Mermaid 图结构")
        print(app.get_graph().draw_mermaid())

        run_invoke_demos(app)
        run_stream_demo(app)


if __name__ == "__main__":
    main()




