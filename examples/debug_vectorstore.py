"""调试 `upsert_chunks` / `query_collection` 的离线脚本。

适合学习：
- chunk + embedding 是如何写入 Chroma 的
- query_collection 返回的数据结构长什么样
- 向量库查询结果如何映射回文本和 metadata
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _debug_common import build_demo_chunks, preview_text, print_kv, print_title

from pdf2md_rag.embeddings import build_embedder
from pdf2md_rag.vectorstore import query_collection, upsert_chunks


def main() -> None:
    print_title("debug_vectorstore")

    chunks = build_demo_chunks()
    embedder = build_embedder(embedder_type="hash", model_name="unused", hash_dimensions=64)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    # 用临时目录避免污染真实数据目录；脚本结束后自动清理。
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = Path(temp_dir) / "chroma"
        summary = upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name="debug-vectorstore",
        )
        rows = query_collection(
            question="What is Lasso?",
            query_embedding=embedder.embed_query("What is Lasso?"),
            persist_directory=persist_directory,
            collection_name="debug-vectorstore",
            top_k=2,
        )

        print_kv("persist_dir", persist_directory)
        print_kv("stored_count", summary["count"])
        print_kv("row_count", len(rows))
        for row in rows:
            print("-" * 80)
            print_kv("id", row["id"])
            print_kv("distance", row["distance"])
            print_kv("preview", preview_text(row["document"], limit=200))


if __name__ == "__main__":
    main()
