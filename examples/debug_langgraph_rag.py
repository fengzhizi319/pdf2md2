"""LangGraph + LLM 示例：演示真正的"分析 -> 检索 -> 生成"。

适合学习：
- 如何把一个 RAG 闭环拆成多个 LangGraph 节点
- `analyze_question` / `retrieve_context` / `generate_answer` 的职责边界
- 如何把 LangGraph 接到真实 LLM（OpenAI、本地 OpenAI-compatible、Ollama）
- 如何在图里保留 `trace`、检索上下文、来源列表等中间状态

这个脚本会：
1. 先构造一组稳定的 demo chunk
2. 写入临时 Chroma 向量库
3. 用 LangGraph 编排"分析 -> 检索 -> 生成"
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
   PDF2MD_RAG_LLM_MODEL=qwen3.5:0.8b \
   python examples/debug_langgraph_rag.py

默认 embedding 在 `openai-compatible` 场景下仍使用项目里的 `sentence-transformers`，因为这样更适合演示"检索"阶段。
但如果你走的是本地 `ollama`，脚本会默认切到 `hash` embedding，避免在离线环境里额外下载 Hugging Face 模型。
如果你希望在 Ollama 场景里仍使用真实语义 embedding，也可以显式设置 `PDF2MD_RAG_EMBEDDER` 和 `PDF2MD_RAG_EMBEDDING_MODEL`。
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

    这些配置本身不会在图中"流动"，所以我们把它们放在 state 外面。
    这样更容易区分：

    - 什么是"每次执行都会变化的状态"（用户问题、检索结果、答案）
    - 什么是"本次运行固定不变的配置"（模型名、base URL、top_k）

    使用 dataclass 的好处：
    - 自动生成 __init__ 方法，简化对象创建
    - slots=True 优化内存占用（Python 3.10+）
    - 类型安全，IDE 提供自动补全
    """

    embedder_type: str              # Embedding 模型类型："sentence-transformers" 或 "hash"
    embedding_model: str            # 具体使用的 embedding 模型名称
    hash_dimensions: int            # 哈希 embedding 的维度（仅当使用 hash 时有效）
    collection_name: str            # Chroma 向量库中的集合名称
    top_k: int                      # 检索时返回的 chunk 数量
    max_context_chars: int          # 喂给 LLM 的最大上下文字符数
    llm_provider: str               # LLM 提供商："openai-compatible" 或 "ollama"
    llm_model: str                  # LLM 模型名称
    llm_base_url: str               # LLM API 的基础 URL
    api_key: str | None             # API 密钥（可选，某些本地服务不需要）
    system_prompt: str              # 系统提示词，设定 LLM 的角色和行为准则
    temperature: float              # LLM 生成的温度值（0.0-2.0），越高越随机
    max_tokens: int                 # LLM 生成的最大 token 数


class RagState(TypedDict, total=False):
    """LangGraph 图里流动的状态。

    这是整个 RAG 流程的数据容器，定义了所有节点共享的上下文信息。

    total=False 的含义：
    - 所有字段都是可选的（不需要在初始化时提供所有值）
    - 每个节点可以只返回它更新的字段
    - LangGraph 会自动合并各节点的返回值

    状态流转过程：
    1. analyze_question 节点：设置 normalized_question, analysis, intent
    2. retrieve_context 节点：设置 search_result, context_text, sources
    3. generate_answer 节点：设置 answer, raw_llm_response
    4. 所有节点都会追加 trace
    """

    user_question: str              # 【输入】用户的原始问题
    normalized_question: str        # 【处理】规范化后的问题（去除首尾空白）
    analysis: str                   # 【处理】问题分析文本，包含长度等信息
    intent: Literal["retrieve", "clarify"]  # 【决策】意图：检索还是澄清
    search_result: SearchResult     # 【检索】完整的检索结果对象（包含 hits、context_text 等）
    context_text: str               # 【检索】拼接好的上下本文本（用于喂给 LLM）
    sources: list[str]              # 【检索】来源标签列表（用于引用和溯源）
    answer: str                     # 【输出】最终生成的答案
    raw_llm_response: dict[str, Any]  # 【调试】LLM 的原始响应（包含 usage 等信息）
    trace: list[str]                # 【追踪】执行路径追踪，记录经过了哪些节点


def build_learning_chunks() -> list[Chunk]:
    """构造一组专门服务于 LangGraph + RAG 学习的 demo chunk。

    这里的内容是手工编写的"学习资料"，不是从真实 PDF 提取而来。
    这么做有两个好处：
    - 内容稳定，便于教学和重复实验
    - 不需要先准备真实论文或文档，也能演示完整流程

    Chunk 数据结构说明：
    - chunk_id: 唯一标识符
    - text: 实际的文本内容（用于 embedding 和检索）
    - metadata: 元数据字典，包含来源、标题、页码等信息

    Returns:
        list[Chunk]: Chunk 对象列表，每个对象包含文本和元数据
    """
    source_name = "LangGraph Learning Notes"  # 虚拟的源文档名称
    source_path = "/virtual/langgraph-learning-notes.md"  # 虚拟的文件路径

    # 定义 4 个 demo chunk 的负载数据
    # 每个 tuple 包含：(chunk_id, text, heading, page)
    payloads = [
        (
            "demo-lg-1",
            "LangGraph 用 graph 的方式组织多步骤工作流。每个节点负责一个明确动作，"
            "例如分析问题、调用检索、或生成答案；边和条件边负责决定下一步去哪。",
            "LangGraph Overview",  # 所属章节标题
            1,  # 页码
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
                    "chunk_index": chunk_index,  # chunk 在列表中的索引
                    "text_length": len(text),  # 文本长度（可用于过滤或排序）
                },
            )
        )
    return chunks


def append_trace(state: RagState, step_name: str) -> list[str]:
    """给 trace 追加一步，方便观察图到底经过了哪些节点。

    Args:
        state: 当前状态对象
        step_name: 当前执行的节点名称

    Returns:
        list[str]: 更新后的 trace 列表

    示例：
        初始 trace: ["analyze_question"]
        调用 append_trace(state, "retrieve_context")
        返回：["analyze_question", "retrieve_context"]
    """
    return [*state.get("trace", []), step_name]


def resolve_runtime_config() -> RuntimeConfig:
    """从环境变量解析一组对学习友好的默认配置。

    这里故意做了一点"智能默认值"：
    - 如果检测到 `OPENAI_API_KEY`，默认走 OpenAI-compatible + api.openai.com
    - 否则默认走本地 `ollama`

    这样同一个脚本既能对接远程 OpenAI，也能直接对接本地模型。

    优先级规则：
    1. 显式指定的环境变量优先于默认值
    2. PDF2MD_RAG_LLM_PROVIDER 未设置时，根据 OPENAI_API_KEY 自动推断

    Returns:
        RuntimeConfig: 填充好的运行时配置对象
    """
    # 从环境变量读取 LLM 提供商类型
    provider = os.getenv("PDF2MD_RAG_LLM_PROVIDER", "").strip().lower()
    # 检查是否设置了 OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")

    # 如果没有显式指定 provider，则根据 API Key 自动推断
    if not provider:
        provider = "openai-compatible" if api_key else "ollama"

    # 根据 provider 类型设置对应的 base URL、model 和 embedding 默认值
    if provider in {"openai", "openai-compatible", "openai_compatible", "openai-compatible-http"}:
        # OpenAI 官方或兼容接口
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", "https://api.openai.com")
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "gpt-4o-mini")
        default_embedder_type = "sentence-transformers"
        default_embedding_model = get_embedding_model_name()
    elif provider == "ollama":
        # Ollama 本地服务
        llm_base_url = os.getenv("PDF2MD_RAG_LLM_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        llm_model = os.getenv("PDF2MD_RAG_LLM_MODEL", "qwen3.5:0.8b")
        default_embedder_type = "hash"
        default_embedding_model = "unused"
    else:
        raise ValueError(f"Unsupported llm provider: {provider}")

    return RuntimeConfig(
        embedder_type=os.getenv("PDF2MD_RAG_EMBEDDER", default_embedder_type),
        embedding_model=os.getenv("PDF2MD_RAG_EMBEDDING_MODEL", default_embedding_model),
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

    执行流程：
    1. 构建学习用的 demo chunk
    2. 根据配置创建 embedding 模型实例
    3. 对所有 chunk 文本进行向量化
    4. 将 vectors 和 metadata 一起存入 Chroma

    Args:
        runtime: 运行时配置对象
        persist_directory: Chroma 持久化目录路径
    """
    chunks = build_learning_chunks()  # 获取预定义的 chunk 列表
    embedder = build_embedder(
        embedder_type=runtime.embedder_type,
        model_name=runtime.embedding_model,
        hash_dimensions=runtime.hash_dimensions,
    )
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])  # 批量计算 embedding
    upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=runtime.collection_name,
    )


def analyze_question(state: RagState) -> RagState:
    """第一个节点：分析用户问题。

    注意：
    这里的"分析"不是让 LLM 思考，而是一个轻量的预处理节点。
    它的任务是把问题规范化，并判断是否值得进入检索阶段。

    真实项目中，你也可以把这里换成：
    - Query rewriting（查询重写，改进检索效果）
    - 意图分类（判断用户想做什么类型的任务）
    - 工具选择 / router（决定使用哪个工具或子流程）
    - 是否走 web search / database / vector search 的判定

    Args:
        state: 当前状态，至少包含 user_question 字段

    Returns:
        RagState: 部分状态更新，包含 normalized_question, analysis, intent, trace
    """
    # 规范化问题：去除首尾空白字符
    normalized_question = state["user_question"].strip()

    # 根据问题长度决定意图
    # 这是一个简化的策略：长度>=6 认为是有意义的问题，可以进入检索
    # 真实场景中可能需要更复杂的判断逻辑（关键词匹配、语义分析等）
    intent: Literal["retrieve", "clarify"] = "retrieve" if len(normalized_question) >= 6 else "clarify"

    # 生成分析文本，记录问题和长度信息
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
    """根据分析结果决定是否进入检索。

    这是第一个条件路由函数，在 analyze_question 节点之后被调用。

    Args:
        state: 当前状态，包含 intent 字段

    Returns:
        Literal: 下一个节点的名称
            - "retrieve_context": 当 intent="retrieve" 时
            - "ask_for_clarification": 当 intent="clarify" 时
    """
    return "retrieve_context" if state["intent"] == "retrieve" else "ask_for_clarification"


def build_demo_graph(runtime: RuntimeConfig, persist_directory: Path) -> Any:
    """构建 LangGraph 图，并把静态运行配置闭包进节点。

    这是一种很常见的写法：
    - 可变数据走 state（在节点间流动的用户问题、检索结果等）
    - 固定配置走 closure（embedding 配置、LLM 配置等）

    为什么使用闭包？
    - 避免在每个节点函数中重复传递相同的配置参数
    - 保持节点函数签名简洁（只需要 state 作为参数）
    - 配置在编译时就已经确定，运行时不会改变

    Args:
        runtime: 运行时配置对象
        persist_directory: Chroma 持久化目录

    Returns:
        CompiledGraph: 编译好的可执行图对象
    """

    def retrieve_context(state: RagState) -> RagState:
        """第二个节点：真正调用项目里的 `search_chunks` 做检索。

        这个节点是整个 RAG 流程的核心：
        1. 使用规范化的问题作为查询
        2. 从 Chroma 向量库中检索相关 chunk
        3. 返回搜索结果、拼接好的上下文、来源列表

        Args:
            state: 当前状态，包含 normalized_question 字段

        Returns:
            RagState: 包含 search_result, context_text, sources, trace
        """
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
        """检索到了内容才进入生成节点。

        这是第二个条件路由函数，在 retrieve_context 节点之后被调用。

        判断逻辑：
        - 如果 search_result 存在且 hits 非空 → 生成答案
        - 否则 → 请求澄清

        Args:
            state: 当前状态，包含 search_result 字段

        Returns:
            Literal: 下一个节点的名称
        """
        search_result = state.get("search_result")
        if search_result and search_result.hits:
            return "generate_answer"
        return "ask_for_clarification"

    def generate_answer(state: RagState) -> RagState:
        """第三个节点：把检索结果交给真实 LLM 生成答案。

        这个节点负责：
        1. 组装 prompt（包含问题分析和检索上下文）
        2. 调用 LLM API
        3. 提取并返回最终答案

        Prompt 组成：
        - Question analysis: 来自 analyze_question 节点的分析
        - Relevant context: 来自 retrieve_context 的检索结果
        - System prompt: 在 RuntimeConfig 中定义的行为准则

        Args:
            state: 当前状态，包含 analysis, normalized_question, search_result

        Returns:
            RagState: 包含 answer, raw_llm_response, trace
        """
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
        """兜底节点：没有足够信息时要求用户补充问题。

        这个节点在两种情况下被触发：
        1. 问题太短（长度<6），无法判断意图
        2. 检索结果为空，无法生成有依据的答案

        Args:
            state: 当前状态

        Returns:
            RagState: 包含 answer, trace
        """
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

    # 创建状态图构建器，指定状态类型为 RagState
    builder = StateGraph(cast(Any, RagState))

    # 注册 4 个节点到图中
    # add_node 的第一个参数是节点 ID（字符串），第二个参数是实际执行的函数
    builder.add_node("analyze_question", cast(Any, analyze_question))
    builder.add_node("retrieve_context", cast(Any, retrieve_context))
    builder.add_node("generate_answer", cast(Any, generate_answer))
    builder.add_node("ask_for_clarification", cast(Any, ask_for_clarification))

    # 定义图的拓扑结构

    # 1. 从 START 进入第一个节点（analyze_question）
    builder.add_edge(START, "analyze_question")

    # 2. 第一个条件分支：根据 analyze_question 的结果决定下一步
    builder.add_conditional_edges(
        "analyze_question",
        cast(Any, route_after_analysis),
        {
            "retrieve_context": "retrieve_context",
            "ask_for_clarification": "ask_for_clarification",
        },
    )

    # 3. 第二个条件分支：根据 retrieve_context 的结果决定下一步
    builder.add_conditional_edges(
        "retrieve_context",
        cast(Any, route_after_retrieval),
        {
            "generate_answer": "generate_answer",
            "ask_for_clarification": "ask_for_clarification",
        },
    )

    # 4. 无论成功回答还是请求澄清，最后都结束
    builder.add_edge("generate_answer", END)
    builder.add_edge("ask_for_clarification", END)

    # 编译图，返回可执行对象
    return builder.compile()


def call_llm(runtime: RuntimeConfig, user_prompt: str) -> tuple[dict[str, Any], str]:
    """统一调用不同类型的 LLM 后端。

    这里故意把"访问模型"的细节集中在一个函数中：
    - LangGraph 节点只需要关心"我要生成答案"
    - 后端差异（OpenAI-compatible / Ollama）在这里被屏蔽掉

    这是一个典型的"适配器模式"：
    - 对外提供统一的接口（接收 prompt，返回 response）
    - 内部根据 provider 分发到不同的实现

    Args:
        runtime: 运行时配置，包含 provider、model、URL 等信息
        user_prompt: 组装好的用户 prompt（包含问题和分析）

    Returns:
        tuple[dict[str, Any], str]:
            - raw_response: LLM 的原始响应对象（包含 usage、finish_reason 等）
            - answer: 提取出的纯文本答案
    """
    provider = runtime.llm_provider.strip().lower()

    # OpenAI 或兼容接口
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

    # Ollama 接口
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


def probe_ollama_connectivity(runtime: RuntimeConfig) -> tuple[bool, str]:
    """探测本地 Ollama 是否可连通。

    这个函数会直接复用项目现有的 Ollama 调用链路：
    - 复用 `simple_qa_module._call_ollama(...)`
    - 间接验证 `/api/chat` 是否可访问
    - 间接验证 `llm_base_url` 和 `llm_model` 是否可用

    它的定位是“探活”，不是正式问答：
    - prompt 尽量短
    - 温度固定为 0
    - 输出以是否能拿到非空响应为准
    """
    if runtime.llm_provider != "ollama":
        return False, f"connectivity probe is only for ollama, current provider={runtime.llm_provider}"

    try:
        raw = simple_qa_module._call_ollama(
            base_url=runtime.llm_base_url,
            model=runtime.llm_model,
            system_prompt="You are a connectivity probe. Reply with OK only.",
            user_prompt="Reply with OK only.",
            temperature=0.0,
        )
        answer = simple_qa_module._extract_ollama_answer(raw).strip()
        if answer:
            return True, f"Ollama reachable: {preview_text(answer, limit=120)}"
        return False, "Ollama responded but returned an empty message"
    except Exception as exc:
        return False, f"Ollama connectivity check failed: {exc}"


def build_llm_help_message(runtime: RuntimeConfig) -> str:
    """当真实 LLM 请求失败时，给出更适合学习场景的提示。

    这个函数体现了"防御性编程"的思想：
    - 不仅抛出异常，还提供可操作的调试建议
    - 帮助初学者快速定位问题（配置错误、服务未启动等）

    Args:
        runtime: 运行时配置对象

    Returns:
        str: 包含当前配置和排查建议的帮助文本
    """
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
    """打印本次运行配置，方便理解这个示例到底连到了什么后端。

    这是一个很好的调试实践：
    - 程序启动时明确告知用户当前的配置
    - 避免因为默认值不符合预期而产生的困惑

    Args:
        runtime: 运行时配置对象
    """
    print_kv("embedder", runtime.embedder_type)
    print_kv("embedding_model", runtime.embedding_model)
    print_kv("collection", runtime.collection_name)
    print_kv("top_k", runtime.top_k)
    print_kv("llm_provider", runtime.llm_provider)
    print_kv("llm_model", runtime.llm_model)
    print_kv("llm_base_url", runtime.llm_base_url)


def print_final_state(final_state: RagState) -> None:
    """把图的最终状态做成更适合教学的终端输出。

    这个函数帮助用户理解：
    1. 图经过了哪些节点（trace）
    2. 问题分析是什么（analysis）
    3. 引用了哪些来源（sources）
    4. 检索到的上下文预览（context_preview）
    5. 最终答案（answer）
    6. Token 使用情况（usage，如果有）

    Args:
        final_state: 图执行完毕后的完整状态
    """
    print_kv("trace", " -> ".join(final_state.get("trace", [])))
    print_kv("analysis", final_state.get("analysis", "<none>"))
    print_kv("sources", final_state.get("sources", []))
    print_kv("context_preview", preview_text(final_state.get("context_text", ""), limit=260))
    print("answer:")
    print(final_state["answer"])

    # 如果有 LLM 原始响应，打印 token 使用情况
    raw_response = final_state.get("raw_llm_response") or {}
    if raw_response:
        usage = raw_response.get("usage")
        if usage:
            print_kv("raw_usage", usage)


def run_invoke_demo(app: Any) -> None:
    """执行一次真正的 analysis -> retrieval -> generation。

    这个演示展示了完整的 RAG 流程：
    1. 提出一个足够长的问题（触发 retrieve 分支）
    2. 一次性执行完整张图（invoke）
    3. 打印最终状态和答案

    Args:
        app: 编译好的图对象
    """
    print_title("invoke：完整 RAG 路径")
    question = "LangGraph 为什么适合做 analysis、retrieval 和 generation 的编排？"
    final_state = app.invoke({"user_question": question})
    print_kv("question", question)
    print_final_state(cast(RagState, final_state))


def run_stream_demo(app: Any) -> None:
    """演示 `stream()`，但用一个短问题走澄清分支，避免再额外调用一次真实 LLM。

    stream() vs invoke() 的区别：
    - invoke(): 阻塞式执行，等待整张图完成后返回最终状态
    - stream(): 流式执行，每完成一个节点就返回该节点的更新

    使用短问题的原因：
    - 避免触发完整的 RAG 流程（节省时间和资源）
    - 专注于展示 stream() 的执行过程

    Args:
        app: 编译好的图对象
    """
    print_title("stream：逐步观察分支路由")
    for step_index, event in enumerate(app.stream({"user_question": "短问"}), start=1):
        node_name, update = next(iter(event.items()))
        print_kv(f"step_{step_index}", node_name)
        print_kv("update", update)
        print("-" * 80)


def main() -> None:
    """主函数：协调整个示例的执行流程。

    执行步骤：
    1. 解析运行时配置（从环境变量）
    2. 打印配置摘要（让用户知道连接了什么后端）
    3. 创建临时 Chroma 向量库（使用临时目录，程序结束后自动清理）
    4. 准备 demo 数据（构建 chunk、计算 embedding、存入向量库）
    5. 构建并编译 LangGraph
    6. 打印 Mermaid 流程图（可视化图结构）
    7. 运行 invoke 演示（完整 RAG 流程）
    8. 运行 stream 演示（流式执行过程）
    """
    print_title("debug_langgraph_rag")
    runtime = resolve_runtime_config()
    print_runtime_summary(runtime)

    if runtime.llm_provider == "ollama":
        ok, detail = probe_ollama_connectivity(runtime)
        print_kv("ollama_probe", "ok" if ok else "failed")
        print_kv("ollama_probe_detail", detail)
        if not ok:
            print_kv("ollama_probe_hint", build_llm_help_message(runtime))

    # 使用临时目录存储 Chroma 向量库
    """
    # TemporaryDirectory 会在 with 块结束时自动清理文件
    TemporaryDirectory 实现了 Python 的上下文管理协议，也就是我们常说的 with 语句支持。
    1.调用 tempfile.TemporaryDirectory()，创建临时目录对象
    2.调用 __enter__() 方法，创建实际的临时目录，返回路径字符串
    3.执行 with 块内的代码，在临时目录中存储 Chroma 数据
    4.with 块结束（正常或异常），无论何种原因退出，都会触发清理
    5.调用 __exit__() 方法，删除临时目录及其内容，确保没有残留文件
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 在临时目录中创建文件
        persist_directory = Path(temp_dir) / "chroma"
        prepare_demo_collection(runtime, persist_directory)
        app = build_demo_graph(runtime, persist_directory)

        print_title("Mermaid 图结构")
        # 使用临时目录
        print(app.get_graph().draw_mermaid())

        run_invoke_demo(app)
        run_stream_demo(app)
    # ← 程序执行到这里时，自动触发清理逻辑

if __name__ == "__main__":
    main()
