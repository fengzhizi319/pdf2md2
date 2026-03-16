"""面向本地演示和学习的脚本入口。

这个文件不是项目唯一入口，而是一个更直观的脚本版 demo：
- 复用共享的 Marker 提取逻辑
- 在终端里打印更友好的进度信息
- 演示 PDF -> Markdown -> 轻量 chunking 的最短路径

如果想看正式主流程，请优先阅读 `pdf2md_rag/pipeline.py`。
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf2md_rag.pdf_to_markdown import extract_markdown, get_marker_device


# ==========================================
# 步骤 1：配置 Mac 专属的 MPS 硬件加速
# ==========================================
def setup_device() -> str:
    """这个脚本版入口只负责给用户一个更直观的本地运行体验。"""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = get_marker_device()

    if device == "mps":
        print("🚀 检测到 Apple Silicon (M系列芯片)，已开启 Marker + MPS 硬件加速！")
    else:
        print("⚠️ 未检测到 MPS，将使用 CPU 运行 Marker。")

    return device


# ==========================================
# 步骤 2：使用 Marker 将 PDF 解析为 Markdown (保留公式)
# ==========================================
def parse_pdf_to_md(pdf_path, output_md_path, device: str | None = None):
    """对共享 `extract_markdown` 做一层脚本友好的包装。"""
    pdf_path = Path(pdf_path).expanduser().resolve()
    output_md_path = Path(output_md_path).expanduser().resolve()
    marker_device = device or os.environ.get("TORCH_DEVICE") or setup_device()

    print(f"\n[1/3] 开始解析论文: {pdf_path}")
    print(f"使用 Marker 提取 Markdown，并尽量保留 LaTeX 数学公式。当前设备: {marker_device}")

    markdown_document = extract_markdown(pdf_path, device=marker_device)
    markdown_text = markdown_document.text

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown_text, encoding="utf-8")

    print(f"✅ 解析完成！Markdown 已保存至: {output_md_path}")
    print(f"📄 总页数: {markdown_document.page_count}")
    return markdown_text


# ==========================================
# 步骤 3：针对公式优化的文档切分 (Chunking)
# ==========================================
def chunk_markdown_with_math_protection(md_text):
    """给脚本版演示准备的轻量 chunking：重点优先照顾 `$$...$$` 数学块。"""
    print("\n[2/3] 开始进行智能切分 (Chunking)...")

    custom_separators = [
        "\n$$",
        "$$\n",
        "$$",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=custom_separators,
        keep_separator=True,
    )

    chunks = text_splitter.split_text(md_text)
    print(f"✅ 切分完成！共生成了 {len(chunks)} 个 RAG 文本块 (Chunks)。")
    return chunks


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    project_root_dir = Path(__file__).resolve().parents[1]
    project_pdf_dir = project_root_dir / "pdf"
    input_pdf = project_pdf_dir / "Understanding Lasso – A Novel Lookup Argument Protocol.pdf"
    output_md = project_pdf_dir / "math-demo.marker.md"

    device = setup_device()

    if not input_pdf.exists():
        print(f"❌ 找不到文件 {input_pdf}，请修改代码中的 input_pdf 路径。")
    else:
        markdown_content = parse_pdf_to_md(input_pdf, output_md, device=device)
        document_chunks = chunk_markdown_with_math_protection(markdown_content)

        print("\n[3/3] 预览前 2 个 Chunk")
        for i, chunk in enumerate(document_chunks[:2], start=1):
            print(f"\n【Chunk {i}】(长度: {len(chunk)}):")
            print("-" * 40)
            print(chunk)
            print("-" * 40)