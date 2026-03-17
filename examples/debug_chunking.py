"""调试 `chunk_markdown` 的最小脚本。

适合学习：
- 一个 `MarkdownDocument` 会被切成多少 chunk
- chunk 的 heading / page / length 如何变化
- overlap 和 chunk 元数据的效果
"""

from __future__ import annotations

from _debug_common import PDF_PATH, get_embedding_model_name, preview_text, print_kv, print_title

from pdf2md_rag.chunking import chunk_markdown
from pdf2md_rag.pdf_to_markdown import extract_markdown
from pdf2md_rag.embeddings import build_embedder

def main() -> None:
    print_title("debug_chunking")
    print_kv("pdf_path", PDF_PATH)

    document = extract_markdown(PDF_PATH)
    # 这里刻意把 chunk_size 调小一点，方便观察更多切块结果。
    chunks = chunk_markdown(document, chunk_size=800, chunk_overlap=120)

    print_kv("page_count", document.page_count)
    print_kv("chunk_count", len(chunks))

    for chunk in chunks[:5]:
        print("-" * 80)
        print_kv("chunk_id", chunk.chunk_id)
        print_kv("heading", chunk.metadata.get("heading"))
        print_kv("page", chunk.metadata.get("page"))
        print_kv("length", chunk.metadata.get("text_length"))
        print_kv("preview", preview_text(chunk.text, limit=240))
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
