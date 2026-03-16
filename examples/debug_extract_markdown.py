"""调试 `extract_markdown` 的最小脚本。

适合学习：
- Marker 设备选择
- PDF 提取后的页数和文本规模
- Markdown 结果预览应该长什么样
"""

from __future__ import annotations

from _debug_common import PDF_PATH, preview_text, print_kv, print_title

from pdf2md_rag.pdf_to_markdown import extract_markdown, get_marker_device


def main() -> None:
    print_title("debug_extract_markdown")
    print_kv("pdf_path", PDF_PATH)
    print_kv("device", get_marker_device())

    # 核心调用：直接看 PDF -> Markdown 的结果，不经过后续切块和向量化。
    document = extract_markdown(PDF_PATH)

    print_kv("page_count", document.page_count)
    print_kv("text_length", len(document.text))
    print_kv("preview", preview_text(document.text, limit=320))


if __name__ == "__main__":
    main()
