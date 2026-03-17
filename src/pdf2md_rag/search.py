"""结构化检索层。

    这个模块位于“向量库查询”和“问答/生成”之间，主要职责是把底层检索结果
    转换成更适合 RAG 使用的稳定结构。

    可以把它理解成三段处理链路：

    1. 问题 -> embedding
       使用配置指定的 embedder，把用户问题编码成向量。

    2. embedding -> 向量库检索结果
       调用 `query_collection(...)` 从 Chroma 中取回最相近的若干 chunk。

    3. 原始检索结果 -> RAG 友好结构
       - 单条结果转成 `SearchHit`
       - 多条结果汇总成 `SearchResult`
       - 把文本片段压缩成适合直接交给 LLM 的 `context_text`

    这样做的好处是：
    - 上层 QA / prompt 组装逻辑不需要直接理解 Chroma 的返回格式
    - 底层向量库实现如果将来替换，上层接口可以保持稳定
    - 检索阶段与问答阶段之间的边界更清晰
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import build_embedder
from .vectorstore import query_collection


@dataclass(slots=True)
class SearchHit:
    """一次检索命中的标准化表示。

    这个对象表示“向量库返回的一条命中结果”，但它已经不是底层数据库原始格式，
    而是项目内部统一后的结构。

    字段含义：
    - `rank`:
        命中排名，从 1 开始。这里的顺序通常继承自向量库返回顺序，
        即默认认为前面的结果更相关。
    - `chunk_id`:
        chunk 的唯一标识，通常来自 ingest 阶段写入向量库时的 id。
    - `text`:
        命中的正文内容，后续会被拼进 `context_text`。
    - `distance`:
        向量库返回的距离值。越小通常表示越相近。
    - `score`:
        一个便于展示的粗略分数，这里简单使用 `1 - distance`。
        它不是严格概率，也不是训练出来的置信度，仅用于展示。
    - `source_path` / `source_name`:
        来源文件路径与来源名，方便后续做引用与展示。
    - `heading` / `page` / `chunk_index` / `text_length`:
        这些都来自 chunk metadata，用于增强可解释性与调试体验。
    """

    rank: int
    chunk_id: str
    text: str
    distance: float
    score: float
    source_path: str
    source_name: str
    heading: str | None
    page: int | None
    chunk_index: int | None
    text_length: int | None

    @property
    def citation(self) -> str:
        """生成适合展示给用户或注入 prompt 的来源引用文本。

        设计思路：
        - 只拼接真正有信息量的字段
        - 优先展示“文档名 -> 页码 -> 小节标题”
        - 让最终引用既短，又能帮助人快速定位来源

        例如：
        - "lasso | p.1 | Abstract"
        - "paper-name | p.3"
        - "unknown"

        这里返回的是纯展示字符串，不承担机器可解析格式的职责。
        """
        parts = [self.source_name]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.heading:
            parts.append(self.heading)
        return " | ".join(parts)


@dataclass(slots=True)
class SearchResult:
    """面向 RAG 的完整检索结果结构。

    相比直接把向量库原始返回值继续往上传，这个对象有几个优势：

    1. `hits`
       使用统一的 `SearchHit` 列表，方便上层稳定处理。
    2. `context_text`
       已经是适合直接交给 LLM 的上下文，不需要 QA 层重复拼接。
    3. `sources`
       单独提取来源摘要，方便终端输出、UI 展示、引用列表等场景。
    4. `retrieval_meta`
       保留这次检索的关键配置与统计信息，便于调试和日志记录。

    这相当于把“检索结果”和“给问答层的输入材料”打包到同一个对象中。
    """

    question: str
    collection_name: str
    top_k: int
    hits: list[SearchHit]
    context_text: str
    sources: list[str]
    retrieval_meta: dict[str, Any]


def search_chunks(
    question: str,
    collection_name: str,
    persist_directory: str | Path = "data/chroma",
    top_k: int = 5,
    embedder_type: str = "sentence-transformers",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    hash_dimensions: int = 384,
    max_context_chars: int = 6000,
) -> SearchResult:
    """执行一次完整检索：问题 embedding -> Chroma 查询 -> RAG 上下文拼装。

    这是检索层的主入口，整体流程如下：

    第 1 步：构造 query embedder
    - 根据配置选择具体 embedding 后端
    - 可以是真正的语义模型，也可以是 hash 模式

    第 2 步：把用户问题编码为向量
    - `embed_query(question)` 返回一个 query embedding

    第 3 步：去向量库里查 top-k 结果
    - 底层调用 `query_collection(...)`
    - 返回值仍然是面向数据库格式的原始行结构

    第 4 步：把每一行结果标准化为 `SearchHit`
    - 做字段清洗
    - 提取 metadata
    - 计算便于展示的 score

    第 5 步：把 hits 压缩成 LLM 可直接使用的上下文文本
    - 添加问题说明
    - 添加来源标记
    - 控制总字符预算

    最终返回 `SearchResult`，供问答层或 CLI 直接消费。
    """
    # 创建 query 用的 embedding 实现。
    # 这里故意通过统一工厂函数创建，而不是在 search 层直接依赖某个具体模型类，
    # 这样整个检索层就不会和特定 embedding 库强耦合。
    embedder = build_embedder(
        embedder_type=embedder_type,
        model_name=embedding_model,
        hash_dimensions=hash_dimensions,
    )

    # 向量检索的核心：先把问题编码成向量，再交给底层向量库查询。
    # `query_collection` 返回的是项目内部的“轻量行结构”，但仍然偏底层。
    rows = query_collection(
        question=question,
        query_embedding=embedder.embed_query(question),
        persist_directory=persist_directory,
        collection_name=collection_name,
        top_k=top_k,
    )

    # 把每一条底层 row 转成统一的 SearchHit。
    # 排名从 1 开始更适合展示给用户，也更适合后续生成 [Source 1] 这种引用格式。
    hits = [_row_to_hit(index + 1, row) for index, row in enumerate(rows)]

    # 将多个命中片段拼成最终给 LLM 使用的上下文。
    # 这里控制字符预算，而不是无脑拼接所有文本，避免上下文过长。
    context_text = _build_context_text(
        question=question,
        hits=hits,
        max_context_chars=max_context_chars,
    )

    # 单独提取 citation 列表，方便 CLI / UI 直接展示来源。
    sources = [hit.citation for hit in hits]

    # 返回完整检索结果对象。
    # `retrieval_meta` 主要用于调试、日志和实验记录。
    return SearchResult(
        question=question,
        collection_name=collection_name,
        top_k=top_k,
        hits=hits,
        context_text=context_text,
        sources=sources,
        retrieval_meta={
            "embedding_model": embedding_model,
            "embedder_type": embedder_type,
            "persist_directory": str(Path(persist_directory)),
            "hit_count": len(hits),
            "max_context_chars": max_context_chars,
        },
    )


def _row_to_hit(rank: int, row: dict[str, Any]) -> SearchHit:
    """把向量库返回的一行结果标准化成 `SearchHit`。

    为什么需要这一步？
    因为向量库层返回的是底层结构，字段组织形式偏数据库风格；
    但上层检索展示、QA、日志、调试都更适合使用语义明确的对象结构。

    这里做的事情包括：
    - 从 `row["metadata"]` 中提取来源信息
    - 对缺失字段做兜底
    - 规范化 distance 为 float
    - 计算一个简单的展示分数 score

    注意：
    - `score = 1 - distance` 只是“展示型分数”
    - 它不是严格意义上的相似度概率
    - 如果底层距离定义发生变化，这个分数也只是辅助信息
    """
    metadata = row.get("metadata") or {}

    # 向量库通常返回 distance，值越小表示越相近。
    # 这里统一转成 float，避免后续展示或排序时出现类型不一致。
    distance = float(row.get("distance", 1.0))

    # 这里的 score 不是模型校准后的置信度，而是一个方便展示的粗略值。
    # 例如 distance=0.2 时得到 score=0.8。
    # 若 distance > 1，则最低截断到 0，避免出现负值。
    score = max(0.0, 1.0 - distance)

    return SearchHit(
        rank=rank,
        chunk_id=str(row.get("id", "")),
        text=str(row.get("document", "")).strip(),
        distance=distance,
        score=score,
        source_path=str(metadata.get("source_path", "")),
        source_name=str(metadata.get("source_name", "unknown")),
        heading=metadata.get("heading"),
        page=metadata.get("page"),
        chunk_index=metadata.get("chunk_index"),
        text_length=metadata.get("text_length"),
    )


def _build_context_text(question: str, hits: list[SearchHit], max_context_chars: int) -> str:
    """把多个命中片段压缩成适合直接喂给 LLM 的上下文文本。

    设计目标：
    1. 保留用户问题，帮助 prompt 自解释
    2. 保留来源标签，便于模型引用和回答后溯源
    3. 控制最大上下文长度，避免 prompt 过长
    4. 优先保留排名更高的结果

    这里采用“顺序贪心拼接”的策略：
    - 按 hits 排名顺序遍历
    - 每条命中先计算 prefix（来源标签）会占多少长度
    - 剩下的预算分配给正文 excerpt
    - 一旦预算耗尽就停止

    这种实现简单、稳定，而且非常适合当前项目的轻量 RAG 场景。
    """
    # 先放入问题与一个固定标题，帮助后续 LLM 理解下面内容的语义角色。
    sections: list[str] = [f"Question: {question}", "Relevant context:"]

    # 用字符数作为预算控制单位。
    # 这样不需要引入 tokenizer，也足够适合本项目当前规模。
    remaining = max_context_chars

    for hit in hits:
        # 每条来源前缀都包含来源编号和引用信息，
        # 例如：[Source 1] lasso | p.1 | Abstract
        prefix = f"[Source {hit.rank}] {hit.citation}\n"

        # 预留 prefix 的长度后，剩余预算用于正文。
        # `- 2` 是给拼接时的空行/分隔留一点余量。
        body_budget = max(0, remaining - len(prefix) - 2)
        if body_budget <= 0:
            break

        # 如果正文过长，只截取当前预算范围内的前缀部分。
        # 这里优先保证“尽可能多地塞进高排名结果”。
        excerpt = hit.text[:body_budget].strip()
        if not excerpt:
            continue

        section = f"{prefix}{excerpt}"
        sections.append(section)

        # 当前 section 已经被加入上下文，因此从总预算中扣除。
        remaining -= len(section)
        if remaining <= 0:
            break

    # 用空行拼接，让不同来源之间的结构更清晰，
    # 同时也更适合直接放进 LLM prompt。
    return "\n\n".join(sections).strip()
