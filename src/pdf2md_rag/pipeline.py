"""主编排层：把各个处理阶段串成一次完整 ingest。

如果只想理解项目主线，优先看这个文件。
它会依次调用：
- `extract_markdown`
- `chunk_markdown`
- `build_embedder`
- `upsert_chunks`

最终产出 Markdown、manifest 和 Chroma 向量库数据。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunking import chunk_markdown
from .config import PipelineConfig
from .embeddings import build_embedder
from .pdf_to_markdown import extract_markdown
from .vectorstore import upsert_chunks


@dataclass(slots=True)
class IngestResult:
    """一次 ingest 执行结束后返回给 CLI / 调试脚本的摘要结果。"""

    pdf_path: str
    markdown_path: str
    manifest_path: str
    collection_name: str
    chunk_count: int
    vector_count: int
    page_count: int
    embedder_type: str
    embedding_model: str


def ingest_pdf(pdf_path: str | Path, config: PipelineConfig) -> IngestResult:
    """执行项目主流程：PDF -> Markdown -> Chunks -> Embeddings -> Chroma。"""
    # 先确保所有输出目录存在。后续每个阶段都依赖这些目录落盘结果。
    config.markdown_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF 提取为 Markdown。
    document = extract_markdown(pdf_path)
    markdown_path = config.markdown_dir / f"{document.source_path.stem}.md"
    markdown_path.write_text(document.text, encoding="utf-8")

    # 2) Markdown 切块，得到更适合检索的语义片段。
    chunks = chunk_markdown(
        document,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # 3) 为每个 chunk 生成 embedding。embedding 的实现由配置决定：
    #    可以是本地语义模型，也可以是纯 hash 的离线快速测试模式。
    texts = [chunk.text for chunk in chunks]
    embedder = build_embedder(
        embedder_type=config.embedder_type,
        model_name=config.embedding_model,
        hash_dimensions=config.hash_embedding_dimensions,
    )
    embeddings = embedder.embed_texts(texts)

    # 4) 把 chunk + embedding 一起写入本地 Chroma。
    vector_summary = upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=config.chroma_dir,
        collection_name=config.collection_name,
    )

    # 5) 汇总结果，供 CLI、调试脚本和测试直接消费。
    result = IngestResult(
        pdf_path=str(document.source_path),
        markdown_path=str(markdown_path),
        manifest_path=str(config.manifest_dir / f"{document.source_path.stem}.json"),
        collection_name=config.collection_name,
        chunk_count=len(chunks),
        vector_count=vector_summary["count"],
        page_count=document.page_count,
        embedder_type=config.embedder_type,
        embedding_model=config.embedding_model,
    )

    # manifest 主要是为了方便人工排查：
    # - 记录最终输出路径
    # - 保留前几个 chunk 的预览
    # - 遇到检索效果异常时可以快速检查切块结果是否合理
    manifest_payload: dict[str, Any] = asdict(result)
    manifest_payload["chunk_preview"] = [
        {
            "chunk_id": chunk.chunk_id,
            "metadata": chunk.metadata,
            "preview": chunk.text[:240],
        }
        for chunk in chunks[:20]
    ]
    Path(result.manifest_path).write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result
