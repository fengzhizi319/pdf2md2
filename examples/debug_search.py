"""调试 `search_chunks` 的离线脚本。

适合学习：
- `SearchHit` / `SearchResult` 的结构
- `context_text` 是如何拼出来的
- search 层如何站在向量库之上再做一层整理
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _debug_common import build_demo_chunks, preview_text, print_kv, print_title

from pdf2md_rag.embeddings import build_embedder
from pdf2md_rag.search import search_chunks
from pdf2md_rag.vectorstore import upsert_chunks


def main() -> None:
    print_title("debug_search")

    chunks = build_demo_chunks()
    embedder = build_embedder(embedder_type="hash", model_name="unused", hash_dimensions=64)

    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = Path(temp_dir) / "chroma"
        upsert_chunks(
            chunks=chunks,
            embeddings=embedder.embed_texts([chunk.text for chunk in chunks]),
            persist_directory=persist_directory,
            collection_name="debug-search",
        )

        result = search_chunks(
            question="What is Lasso?",
            collection_name="debug-search",
            persist_directory=persist_directory,
            top_k=2,
            embedder_type="hash",
            embedding_model="unused",
            hash_dimensions=64,
            max_context_chars=500,
        )

        print_kv("hit_count", len(result.hits))
        print_kv("sources", result.sources)
        print_kv("context_preview", preview_text(result.context_text, limit=320))
        for hit in result.hits:
            print("-" * 80)
            print_kv("rank", hit.rank)
            print_kv("citation", hit.citation)
            print_kv("score", round(hit.score, 4))
            print_kv("preview", preview_text(hit.text, limit=220))


if __name__ == "__main__":
    main()
