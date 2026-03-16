"""调试 `build_embedder` / `embed_texts` 的最小脚本。

适合学习：
- embedding 接口长什么样
- 向量数量和维度如何对应输入文本
- `hash` 模式下的完全离线调试方式
"""

from __future__ import annotations

from _debug_common import build_demo_chunks, print_kv, print_title

from pdf2md_rag.embeddings import build_embedder


def main() -> None:
    print_title("debug_embeddings")

    chunks = build_demo_chunks()
    # 用 hash 后端，避免下载模型，方便离线学习数据结构。
    embedder = build_embedder(embedder_type="hash", model_name="unused", hash_dimensions=64)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    print_kv("embedder", type(embedder).__name__)
    print_kv("vector_count", len(embeddings))
    print_kv("vector_dim", len(embeddings[0]) if embeddings else 0)
    print_kv("first_vector_head", [round(value, 4) for value in embeddings[0][:8]])


if __name__ == "__main__":
    main()
