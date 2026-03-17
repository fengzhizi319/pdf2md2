"""调试 `build_embedder` / `embed_texts` 的最小脚本。

适合学习：
- embedding 接口长什么样
- 向量数量和维度如何对应输入文本
- 真实 `sentence-transformers` embedding 是如何生成的

首次运行会下载模型；如需切换模型，可设置环境变量
`PDF2MD_RAG_EMBEDDING_MODEL`。
"""

from __future__ import annotations

from _debug_common import build_demo_chunks, get_embedding_model_name, print_kv, print_title

from pdf2md_rag.embeddings import build_embedder


def main() -> None:
    print_title("debug_embeddings")

    chunks = build_demo_chunks()
    # BAAI / bge - small - en - v1.5 是一个小型的开源 embedding 模型，适合在 debug 脚本里快速演示。实际使用时可以换成更强大的模型，比如 `BAAI/bge-base-en-v1.5`。
    '''
    model_name="BAAI/bge-base-en-v1.5"
    embedder: SentenceTransformerEmbedder
    model_name: BAAI/bge-base-en-v1.5
    vector_count: 2
    vector_dim: 768
    '''

    '''
    model_name = BAAI/bge-small-base-en-v1.5"
    embedder      : SentenceTransformerEmbedder
    model_name    : BAAI/bge-small-en-v1.5
    vector_count  : 2
    vector_dim    : 384
    '''

    model_name = get_embedding_model_name()
    embedder = build_embedder(embedder_type="sentence-transformers", model_name=model_name)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    print_kv("embedder", type(embedder).__name__)
    print_kv("model_name", model_name)
    print_kv("vector_count", len(embeddings))
    print_kv("vector_dim", len(embeddings[0]) if embeddings else 0)
    print_kv("first_vector_head", [round(value, 4) for value in embeddings[0][:8]])


if __name__ == "__main__":
    main()
