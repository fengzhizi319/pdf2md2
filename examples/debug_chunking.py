"""调试 `chunk_markdown` 的最小脚本。

适合学习：
- 一个 `MarkdownDocument` 会被切成多少 chunk
- chunk 的 heading / page / length 如何变化
- overlap 和 chunk 元数据的效果
"""

from __future__ import annotations

from _debug_common import PDF_PATH, preview_text, print_kv, print_title

from pdf2md_rag.chunking import chunk_markdown
from pdf2md_rag.pdf_to_markdown import extract_markdown


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


if __name__ == "__main__":
    main()
