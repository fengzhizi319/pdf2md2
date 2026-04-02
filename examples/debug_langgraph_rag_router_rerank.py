"""带 router + rerank 的 advanced LangGraph RAG 示例。

这个脚本是在 `debug_langgraph_rag_advanced.py` 的基础上继续增强的版本，
重点演示两个 production RAG 中非常常见的环节：

- `route_question`：先判断这类问题更适合走哪种检索策略
- `rerank_hits`：检索回来以后，不直接拿原始排序，而是再做一次重排

它同时保留 advanced 版里已经出现的思想：
- `rewrite_question`
- `grade_retrieval`
- `retry_search`
- `fallback_answer`

适合学习：
- 为什么生产级 RAG 往往不只做一次 naive retrieval
- 为什么 router 和 rerank 可以显著改善最终上下文质量
- LangGraph 如何表达“路由 -> 检索 -> 重排 -> 评分 -> 重试 -> 兜底”
- 如何先用离线规则学清楚工作流，再切到真实 LLM

默认行为：
- 默认使用临时 demo Chroma
- 默认使用 `hash` embedder，保证离线、稳定、可重复
- 默认使用 `fake` LLM，方便先关注 graph 编排而不是模型调用
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
from pdf2md_rag.search import SearchHit, SearchResult, search_chunks
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

RouteLabel = Literal["grounding", "robustness", "metadata", "hardware", "general"]
RetrievalPlan = Literal["citation-heavy", "recovery-heavy", "metadata-aware", "hardware-scout", "general-rag"]
Decision = Literal["generate_answer", "retry_search", "fallback_answer", "ask_for_clarification"]

ROUTE_TO_PLAN: dict[RouteLabel, RetrievalPlan] = {
    "grounding": "citation-heavy",
    "robustness": "recovery-heavy",
    "metadata": "metadata-aware",
    "hardware": "hardware-scout",
    "general": "general-rag",
}

ROUTE_HEADING_BONUS: dict[RouteLabel, dict[str, float]] = {
    "grounding": {
        "Grounding and Citations": 2.5,
        "Retrieval Grading": 1.4,
        "Query Rewriting": 0.8,
    },
    "robustness": {
        "Retry Search": 2.6,
        "Fallback Answer": 1.6,
        "RAG Graph Overview": 1.0,
    },
    "metadata": {
        "Metadata Queries": 2.8,
        "RAG Graph Overview": 0.6,
    },
    "hardware": {},
    "general": {
        "RAG Graph Overview": 1.8,
        "Query Rewriting": 1.0,
        "Retrieval Grading": 0.9,
    },
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


class RouterRerankState(TypedDict, total=False):
    """在 router + rerank 图中流动的状态。"""

    user_question: str
    normalized_question: str
    analysis: str
    intent: Literal["retrieve", "clarify"]
    route_label: RouteLabel
    route_reason: str
    retrieval_plan: RetrievalPlan
    candidate_queries: list[str]
    active_question: str
    retry_count: int
    max_retries: int
    raw_search_result: SearchResult
    search_result: SearchResult
    top_hit_before_rerank: str
    top_hit_after_rerank: str
    rerank_reason: str
    retrieval_grade: Literal["strong", "weak", "empty"]
    grade_reason: str
    decision: Decision
    fallback_reason: str
    context_text: str
    sources: list[str]
    answer: str
    raw_llm_response: dict[str, Any]
    trace: list[str]


@dataclass(slots=True)
class ScoredHit:
    """给 rerank 临时使用的打分结构。"""

    hit: SearchHit
    final_score: float
    overlap: int
    heading_bonus: float



def build_learning_chunks() -> list[Chunk]:
    """构造一组用于讲解 router + rerank 的 demo chunk。"""
    source_name = "Router Rerank RAG Notes"
    source_path = "/virtual/router-rerank-rag-notes.md"

    payloads = [
        (
            "rr-1",
            "A production RAG workflow often routes the question before retrieval. "
            "The router decides whether the question is mainly about grounding, robustness, metadata, or a general answer.",
            "RAG Graph Overview",
            1,
        ),
        (
            "rr-2",
            "Query rewriting improves retrieval when the user question is vague. A rewrite can add terms such as retrieval, citations, relevance, and context.",
            "Query Rewriting",
            2,
        ),
        (
            "rr-3",
            "Retrieval grading checks whether the returned chunks actually match the question. Useful signals include keyword overlap, source quality, and citation support.",
            "Retrieval Grading",
            3,
        ),
        (
            "rr-4",
            "Retry search is useful when the first retrieval is weak. A later query may include retry, fallback, recovery, and relevance terms to pull better evidence.",
            "Retry Search",
            4,
        ),
        (
            "rr-5",
            "Fallback answers should say the evidence is insufficient, avoid hallucination, and suggest a better question to the user.",
            "Fallback Answer",
            5,
        ),
        (
            "rr-6",
            "Grounding reduces hallucinations by retrieving supporting context before generation and by citing the supporting sources explicitly.",
            "Grounding and Citations",
            6,
        ),
        (
            "rr-7",
            "Metadata queries often ask about source names, headings, page numbers, or which section a chunk came from. "
            "A router may send these questions to a metadata-aware retrieval strategy.",
            "Metadata Queries",
            7,
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



def append_trace(state: RouterRerankState, step_name: str) -> list[str]:
    """把当前步骤写入 trace，方便观察图的流转。"""
    return [*state.get("trace", []), step_name]



def tokenize_for_overlap(text: str) -> set[str]:
    """把文本切成适合做粗粒度规则判断的 token 集合。"""
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    tokens = set("".join(cleaned).split())
    return {token for token in tokens if len(token) >= 4 and token not in STOPWORDS}



def infer_route(question: str) -> tuple[RouteLabel, str]:
    """根据问题内容推断更适合的路由。

    这是一个规则 router：
    - grounding：更关心证据、引用、幻觉
    - robustness：更关心重试、恢复、fallback
    - metadata：更关心来源、页码、heading、section
    - hardware：更关心部署、GPU、显存、延迟等问题
    - general：一般性的 RAG / LangGraph 工作流问题
    """
    lowered = question.lower()
    if any(keyword in lowered for keyword in ["heading", "page", "section", "metadata", "document", "source name"]):
        return "metadata", "问题包含 heading/page/section 等词，更像元信息查询。"
    if any(keyword in lowered for keyword in ["citation", "citations", "source", "sources", "ground", "halluc", "context"]):
        return "grounding", "问题包含 citations/source/context 等词，更像证据归因问题。"
    if any(keyword in lowered for keyword in ["retry", "retries", "fallback", "recover", "robust", "failure"]):
        return "robustness", "问题包含 retry/fallback/recover 等词，更像鲁棒性问题。"
    if any(keyword in lowered for keyword in ["gpu", "memory", "hardware", "latency", "deploy", "deployment"]):
        return "hardware", "问题包含 gpu/memory/deploy 等词，更像部署与硬件问题。"
    return "general", "没有明显的专属路由信号，因此走 general 路径。"



def build_candidate_queries(question: str, route_label: RouteLabel) -> list[str]:
    """根据 router 结果，生成一组候选查询。"""
    candidates = [question]

    if route_label == "grounding":
        candidates.append(f"{question} grounding citations supporting context sources")
        candidates.append(f"{question} retrieval evidence hallucinations citations")
    elif route_label == "robustness":
        candidates.append(f"{question} retry fallback recovery retrieval relevance")
        candidates.append(f"{question} robust retrieval retry fallback answer")
    elif route_label == "metadata":
        candidates.append(f"{question} heading page source metadata section")
        candidates.append(f"{question} document section page number heading")
    elif route_label == "hardware":
        candidates.append(f"{question} gpu memory hardware deployment latency")
        candidates.append(f"{question} serving hardware inference memory requirements")
    else:
        candidates.append(f"{question} LangGraph production RAG routing reranking")
        candidates.append(f"{question} retrieval rerank route rewrite answer")

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique



def resolve_runtime_config() -> RuntimeConfig:
    """解析一组对学习友好的默认配置。"""
    provider = os.getenv("PDF2MD_RAG_LLM_PROVIDER", "fake").strip().lower()
    api_key = os.getenv("PDF2MD_RAG_API_KEY") or os.getenv("OPENAI_API_KEY")

    if provider in {"fake", "mock", "offline"}:
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "fake-router-rerank-rag")
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
        embedder_type=os.getenv("PDF2MD_RAG_EMBEDDER", "sentence-transformers"),
        embedding_model=os.getenv("PDF2MD_RAG_EMBEDDING_MODEL", get_embedding_model_name()),
        hash_dimensions=int(os.getenv("PDF2MD_RAG_HASH_DIMENSIONS", "128")),
        collection_name=os.getenv("PDF2MD_RAG_COLLECTION", "debug-langgraph-rag-router-rerank"),
        top_k=int(os.getenv("PDF2MD_RAG_TOP_K", "7")),
        max_context_chars=int(os.getenv("PDF2MD_RAG_MAX_CONTEXT_CHARS", "2200")),
        llm_provider=provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        api_key=api_key,
        system_prompt=(
            "You are a careful RAG assistant inside a LangGraph router-and-rerank tutorial. "
            "Use only the reranked retrieved context. If evidence is insufficient, say so clearly. "
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



def analyze_question(state: RouterRerankState) -> RouterRerankState:
    """第一个节点：做最小化的问题分析。"""
    normalized_question = state["user_question"].strip()
    intent: Literal["retrieve", "clarify"] = "retrieve" if len(normalized_question) >= 6 else "clarify"
    analysis = (
        f"问题长度={len(normalized_question)}。"
        "本示例会先做 router，再根据路由结果执行 rewrite / retrieval / rerank。"
    )
    return {
        "normalized_question": normalized_question,
        "analysis": analysis,
        "intent": intent,
        "retry_count": 0,
        "max_retries": state.get("max_retries", 0),
        "trace": append_trace(state, "analyze_question"),
    }



def route_after_analysis(state: RouterRerankState) -> Literal["route_question", "ask_for_clarification"]:
    """问题足够具体时进入 router 节点。"""
    return "route_question" if state["intent"] == "retrieve" else "ask_for_clarification"



def build_context_text(question: str, hits: list[SearchHit], max_context_chars: int) -> str:
    """在示例内部重建一个适合给 LLM 的上下文文本。"""
    sections: list[str] = [f"Question: {question}", "Relevant context:"]
    remaining = max_context_chars
    for hit in hits:
        prefix = f"[Source {hit.rank}] {hit.citation}\n"
        body_budget = max(0, remaining - len(prefix) - 2)
        if body_budget <= 0:
            break
        excerpt = hit.text[:body_budget].strip()
        if not excerpt:
            continue
        section = f"{prefix}{excerpt}"
        sections.append(section)
        remaining -= len(section)
        if remaining <= 0:
            break
    return "\n\n".join(sections).strip()



def rerank_search_result(
    search_result: SearchResult,
    active_question: str,
    route_label: RouteLabel,
    max_context_chars: int,
) -> tuple[SearchResult, str, str, str]:
    """根据 router 结果和 query-overlap 对 hits 做规则重排。"""
    query_tokens = tokenize_for_overlap(active_question)
    scored_hits: list[ScoredHit] = []
    for hit in search_result.hits:
        overlap = len(query_tokens & tokenize_for_overlap(hit.text))
        heading_bonus = ROUTE_HEADING_BONUS[route_label].get(hit.heading or "", 0.0)
        final_score = overlap * 3.0 + heading_bonus + max(0.0, 1.0 - hit.distance) * 0.2
        scored_hits.append(
            ScoredHit(
                hit=hit,
                final_score=final_score,
                overlap=overlap,
                heading_bonus=heading_bonus,
            )
        )

    scored_hits.sort(
        key=lambda item: (
            item.final_score,
            item.overlap,
            item.heading_bonus,
            -item.hit.rank,
        ),
        reverse=True,
    )

    reranked_hits: list[SearchHit] = []
    for new_rank, scored in enumerate(scored_hits, start=1):
        original = scored.hit
        reranked_hits.append(
            SearchHit(
                rank=new_rank,
                chunk_id=original.chunk_id,
                text=original.text,
                distance=original.distance,
                score=scored.final_score,
                source_path=original.source_path,
                source_name=original.source_name,
                heading=original.heading,
                page=original.page,
                chunk_index=original.chunk_index,
                text_length=original.text_length,
            )
        )

    reranked_result = SearchResult(
        question=search_result.question,
        collection_name=search_result.collection_name,
        top_k=search_result.top_k,
        hits=reranked_hits,
        context_text=build_context_text(
            question=search_result.question,
            hits=reranked_hits,
            max_context_chars=max_context_chars,
        ),
        sources=[hit.citation for hit in reranked_hits],
        retrieval_meta={
            **search_result.retrieval_meta,
            "reranked": True,
            "route_label": route_label,
        },
    )

    top_before = search_result.hits[0].citation if search_result.hits else "<none>"
    top_after = reranked_hits[0].citation if reranked_hits else "<none>"
    reason = (
        f"route={route_label}；top_before={top_before}；top_after={top_after}。"
        "规则分数由 overlap、route-heading bonus 和原始距离共同组成。"
    )
    return reranked_result, top_before, top_after, reason



def has_remaining_retry(state: RouterRerankState) -> bool:
    """判断是否还允许继续重试。"""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 0)
    candidate_queries = state.get("candidate_queries", [])
    return retry_count < max_retries and retry_count + 1 < len(candidate_queries)



def build_demo_graph(runtime: RuntimeConfig, persist_directory: Path) -> Any:
    """构建一个带 router + rerank + retry 的高级 RAG 图。"""

    def route_question(state: RouterRerankState) -> RouterRerankState:
        """第二个节点：先决定这类问题走哪种检索路线。"""
        route_label, route_reason = infer_route(state["normalized_question"])
        retrieval_plan = ROUTE_TO_PLAN[route_label]
        candidate_queries = build_candidate_queries(state["normalized_question"], route_label)
        return {
            "route_label": route_label,
            "route_reason": route_reason,
            "retrieval_plan": retrieval_plan,
            "candidate_queries": candidate_queries,
            "trace": append_trace(state, "route_question"),
        }

    def rewrite_question(state: RouterRerankState) -> RouterRerankState:
        """第三个节点：根据 route 和 retry_count 选择当前查询。"""
        candidate_queries = state["candidate_queries"]
        retry_count = state.get("retry_count", 0)
        index = min(retry_count, len(candidate_queries) - 1)
        active_question = candidate_queries[index]
        return {
            "active_question": active_question,
            "trace": append_trace(state, "rewrite_question"),
        }

    def retrieve_context(state: RouterRerankState) -> RouterRerankState:
        """第四个节点：执行原始检索。"""
        raw_search_result = search_chunks(
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
            "raw_search_result": raw_search_result,
            "trace": append_trace(state, "retrieve_context"),
        }

    def rerank_hits(state: RouterRerankState) -> RouterRerankState:
        """第五个节点：对原始 hits 做一次规则 rerank。"""
        raw_search_result = state["raw_search_result"]
        reranked_result, top_before, top_after, rerank_reason = rerank_search_result(
            search_result=raw_search_result,
            active_question=state["active_question"],
            route_label=state["route_label"],
            max_context_chars=runtime.max_context_chars,
        )
        return {
            "search_result": reranked_result,
            "context_text": reranked_result.context_text,
            "sources": reranked_result.sources,
            "top_hit_before_rerank": top_before,
            "top_hit_after_rerank": top_after,
            "rerank_reason": rerank_reason,
            "trace": append_trace(state, "rerank_hits"),
        }

    def grade_retrieval(state: RouterRerankState) -> RouterRerankState:
        """第六个节点：基于 rerank 后的结果判断是否足够好。"""
        search_result = state.get("search_result")
        if not search_result or not search_result.hits:
            return {
                "retrieval_grade": "empty",
                "grade_reason": "rerank 后没有任何 hits。",
                "decision": cast(Decision, "retry_search" if has_remaining_retry(state) else "fallback_answer"),
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
            retrieval_grade: Literal["strong", "weak", "empty"] = "strong"
            decision: Decision = "generate_answer"
            fallback_reason = ""
        elif has_remaining_retry(state):
            retrieval_grade = cast(Literal["strong", "weak", "empty"], "weak")
            decision = cast(Decision, "retry_search")
            fallback_reason = ""
        else:
            retrieval_grade = cast(Literal["strong", "weak", "empty"], "weak")
            decision = cast(Decision, "fallback_answer")
            fallback_reason = "重试次数已用尽，但 rerank 后的上下文仍然没有足够高的关键词覆盖度。"

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

    def route_after_grading(state: RouterRerankState) -> Literal["generate_answer", "retry_search", "fallback_answer"]:
        """根据 grading 决定下一步。"""
        return cast(Literal["generate_answer", "retry_search", "fallback_answer"], state["decision"])

    def retry_search(state: RouterRerankState) -> RouterRerankState:
        """第七个节点：准备下一轮 rewrite + retrieve。"""
        next_retry_count = state.get("retry_count", 0) + 1
        candidate_queries = state["candidate_queries"]
        next_index = min(next_retry_count, len(candidate_queries) - 1)
        next_query = candidate_queries[next_index]
        return {
            "retry_count": next_retry_count,
            "active_question": next_query,
            "trace": append_trace(state, "retry_search"),
        }

    def generate_answer(state: RouterRerankState) -> RouterRerankState:
        """第八个节点：使用 rerank 后的上下文生成答案。"""
        user_prompt = (
            "Question analysis:\n"
            f"{state['analysis']}\n"
            f"Route label: {state['route_label']}\n"
            f"Route reason: {state['route_reason']}\n"
            f"Retrieval plan: {state['retrieval_plan']}\n"
            f"Rerank reason: {state['rerank_reason']}\n"
            f"Retry count: {state.get('retry_count', 0)}\n"
            f"Retrieval grade: {state.get('retrieval_grade')}\n"
            f"Grade reason: {state.get('grade_reason')}\n\n"
            f"{simple_qa_module._build_user_prompt(state['normalized_question'], state['search_result'])}"
        )
        raw_response, answer = call_llm(runtime, state, user_prompt)
        return {
            "answer": answer,
            "raw_llm_response": raw_response,
            "trace": append_trace(state, "generate_answer"),
        }

    def fallback_answer(state: RouterRerankState) -> RouterRerankState:
        """第九个节点：重试后仍不足时诚实降级。"""
        sources = state.get("sources", [])
        answer = (
            "我没有足够把握直接回答这个问题，因为 router + rerank + retry 之后，知识库里仍然缺少足够强的证据。\n\n"
            f"route={state.get('route_label', '<none>')}\n"
            f"retrieval_plan={state.get('retrieval_plan', '<none>')}\n"
            f"原因：{state.get('fallback_reason', '检索相关性不足。')}\n"
            f"rerank：{state.get('rerank_reason', '<none>')}\n"
            f"评分：{state.get('grade_reason', '<none>')}\n\n"
            "建议你尝试：\n"
            "- 把问题改得更贴近知识库里的术语\n"
            "- 明确说明你更关心 citations、retry、metadata 还是 general explanation\n"
            "- 缩小问题范围，只问一个子问题\n\n"
            f"Sources: {', '.join(sources) if sources else '<none>'}"
        )
        return {
            "answer": answer,
            "trace": append_trace(state, "fallback_answer"),
        }

    def ask_for_clarification(state: RouterRerankState) -> RouterRerankState:
        """问题太短时先要求补充。"""
        return {
            "answer": (
                "你的问题太短，图把它路由到了澄清节点。\n"
                "你可以试试这些更具体的问题：\n"
                "- How do citations help grounding?\n"
                "- How do retries help retrieval?\n"
                "- Which heading or page should I show for a source?"
            ),
            "trace": append_trace(state, "ask_for_clarification"),
        }

    builder = StateGraph(cast(Any, RouterRerankState))
    builder.add_node("analyze_question", cast(Any, analyze_question))
    builder.add_node("route_question", cast(Any, route_question))
    builder.add_node("rewrite_question", cast(Any, rewrite_question))
    builder.add_node("retrieve_context", cast(Any, retrieve_context))
    builder.add_node("rerank_hits", cast(Any, rerank_hits))
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
            "route_question": "route_question",
            "ask_for_clarification": "ask_for_clarification",
        },
    )
    builder.add_edge("route_question", "rewrite_question")
    builder.add_edge("rewrite_question", "retrieve_context")
    builder.add_edge("retrieve_context", "rerank_hits")
    builder.add_edge("rerank_hits", "grade_retrieval")
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



def build_fake_answer(state: RouterRerankState, search_result: SearchResult) -> str:
    """构造一个适合教学的 fake LLM 答案。"""
    top_hit = search_result.hits[0] if search_result.hits else None
    top_preview = preview_text(top_hit.text, limit=180) if top_hit else "<none>"
    return (
        "[fake-llm] 这是一段离线教学答案，用来证明图已经成功完成了 router + rerank + generate。\n\n"
        f"原始问题：{state['normalized_question']}\n"
        f"route_label：{state['route_label']}\n"
        f"retrieval_plan：{state['retrieval_plan']}\n"
        f"实际查询：{state['active_question']}\n"
        f"重试次数：{state.get('retry_count', 0)}\n"
        f"rerank 前 top1：{state.get('top_hit_before_rerank', '<none>')}\n"
        f"rerank 后 top1：{state.get('top_hit_after_rerank', '<none>')}\n"
        f"评分：{state.get('retrieval_grade')}\n"
        f"评分解释：{state.get('grade_reason')}\n"
        f"rerank 解释：{state.get('rerank_reason')}\n"
        f"最相关片段预览：{top_preview}\n\n"
        "在真实项目里，这里通常会把 rerank 后的上下文交给真正的 LLM，再生成最终回答。\n\n"
        f"Sources: {', '.join(search_result.sources) if search_result.sources else '<none>'}"
    )



def call_llm(
    runtime: RuntimeConfig,
    state: RouterRerankState,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    """统一调用 fake / OpenAI-compatible / Ollama 三种后端。"""
    provider = runtime.llm_provider.strip().lower()

    if provider in {"fake", "mock", "offline"}:
        raw = {
            "provider": "fake",
            "model": runtime.llm_model,
            "reason": "Offline router-rerank tutorial mode.",
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



def print_final_state(final_state: RouterRerankState) -> None:
    """用更适合学习的方式打印最终状态。"""
    print_kv("trace", " -> ".join(final_state.get("trace", [])))
    print_kv("route_label", final_state.get("route_label", "<none>"))
    print_kv("route_reason", final_state.get("route_reason", "<none>"))
    print_kv("retrieval_plan", final_state.get("retrieval_plan", "<none>"))
    print_kv("active_question", final_state.get("active_question", "<none>"))
    print_kv("retry_count", final_state.get("retry_count", 0))
    print_kv("top_before", final_state.get("top_hit_before_rerank", "<none>"))
    print_kv("top_after", final_state.get("top_hit_after_rerank", "<none>"))
    print_kv("retrieval_grade", final_state.get("retrieval_grade", "<none>"))
    print_kv("grade_reason", final_state.get("grade_reason", "<none>"))
    print_kv("rerank_reason", final_state.get("rerank_reason", "<none>"))
    print_kv("sources", final_state.get("sources", []))
    print_kv("context_preview", preview_text(final_state.get("context_text", ""), limit=280))
    print("answer:")
    print(final_state["answer"])



def run_single_case(app: Any, question: str, title: str) -> None:
    """执行一个问题，并打印结构化结果。"""
    print_title(title)
    final_state = cast(RouterRerankState, app.invoke({"user_question": question, "max_retries": 2}))
    print_kv("question", question)
    print_final_state(final_state)



def run_invoke_demos(app: Any) -> None:
    """跑三组场景：grounding、metadata、fallback。"""
    run_single_case(
        app,
        question="How do citations help grounding?",
        title="invoke：router 选择 grounding，并在 rerank 后生成答案",
    )
    run_single_case(
        app,
        question="Which heading or page should I show for a source?",
        title="invoke：router 选择 metadata，并让 rerank 提升元信息 chunk",
    )
    run_single_case(
        app,
        question="What GPU memory is required to deploy this stack?",
        title="invoke：router 后多次重试仍不足，走 fallback",
    )



def run_stream_demo(app: Any) -> None:
    """用 stream 观察 metadata 路径里的 router + rerank 行为。"""
    print_title("stream：逐步观察 metadata 路径")
    question = "Which heading or page should I show for a source?"
    for step_index, event in enumerate(app.stream({"user_question": question, "max_retries": 2}), start=1):
        node_name, update = next(iter(event.items()))
        print_kv(f"step_{step_index}", node_name)
        print_kv("update", update)
        print("-" * 80)



def main() -> None:
    print_title("debug_langgraph_rag_router_rerank")
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

