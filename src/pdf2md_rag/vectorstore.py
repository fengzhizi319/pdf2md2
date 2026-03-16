"""向量存储适配层。

这个模块只做最薄的一层封装：
- 把内部 `Chunk`/embedding 结构映射到 Chroma
- 把 Chroma 的查询结果整理回项目自己的轻量结构

这样上层模块可以依赖稳定接口，而不是直接耦合 Chroma 的原始返回格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Chunk


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    persist_directory: str | Path,
    collection_name: str,
) -> dict[str, Any]:
    """把 chunk 和对应向量一起写入本地 Chroma。"""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    import chromadb

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )
    count = collection.count()
    return {
        "collection_name": collection_name,
        "persist_directory": str(persist_path),
        "count": count,
    }


def query_collection(
    question: str,
    query_embedding: list[float],
    persist_directory: str | Path,
    collection_name: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """在本地 Chroma 中执行向量检索，并整理成统一的行结构。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(Path(persist_directory)))
    collection = client.get_collection(collection_name)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    rows: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    # 这里故意转成项目自己的轻量结构，而不是把 Chroma 原始返回值继续往上传，
    # 这样 search / QA 层就不需要关心底层数据库的具体字段组织方式。
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        rows.append(
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "question": question,
            }
        )
    return rows
