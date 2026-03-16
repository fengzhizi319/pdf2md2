"""Markdown 语义切块层。

这个模块的职责是把提取后的 Markdown 组织成更适合检索的 chunk：
- 尽量保留段落边界
- 跟踪标题和页码
- 在相邻 chunk 之间保留少量 overlap
- 生成稳定的 chunk_id 和 metadata

它是 Markdown 提取和 embedding / 向量库之间的关键中间层。
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from .models import Chunk, MarkdownDocument

# 标题和分页标记会影响 chunk 的元数据归属；比如某段内容属于哪个章节、哪一页。
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_PAGE_RE = re.compile(r"^##\s+Page\s+(\d+)\s*$", re.IGNORECASE)


def chunk_markdown(document: MarkdownDocument, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Chunk]:
    """把 Markdown 文本切成适合向量检索的 chunk。

    这里不是简单地按固定字符数切，而是尽量保留：
    - 段落边界
    - 章节标题
    - 页码信息
    这样检索结果更容易回溯到原文上下文。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    blocks = _split_blocks(document.text)
    chunks: list[Chunk] = []
    current_heading = "Document"
    current_page: int | None = None
    current_parts: list[str] = []
    current_length = 0

    for block in blocks:
        heading_match = _HEADING_RE.match(block)
        page_match = _PAGE_RE.match(block)
        block_heading = heading_match.group(2).strip() if heading_match else current_heading
        block_page = int(page_match.group(1)) if page_match else current_page

        # 当前 chunk 再加这个 block 会超长时，先把已有内容封成一个 chunk。
        if current_parts and current_length + len(block) + 2 > chunk_size:
            chunks.append(
                _build_chunk(
                    document=document,
                    heading=current_heading,
                    page=current_page,
                    text="\n\n".join(current_parts).strip(),
                    index=len(chunks),
                )
            )
            # 如果遇到新标题/新页码，就从头开始；否则保留一小段 overlap，
            # 让相邻 chunk 共享少量上下文，提升召回时的连续性。
            current_parts = [] if (heading_match or page_match) else _carry_overlap(current_parts, chunk_overlap)
            current_length = sum(len(part) for part in current_parts) + max(0, len(current_parts) - 1) * 2

        # 某个 block 本身就超过 chunk_size 时，走单独拆分逻辑。
        if len(block) > chunk_size:
            if current_parts:
                chunks.append(
                    _build_chunk(
                        document=document,
                        heading=current_heading,
                        page=current_page,
                        text="\n\n".join(current_parts).strip(),
                        index=len(chunks),
                    )
                )
                current_parts = []
                current_length = 0
            for piece in _split_large_block(block, chunk_size, chunk_overlap):
                chunks.append(
                    _build_chunk(
                        document=document,
                        heading=block_heading,
                        page=block_page,
                        text=piece,
                        index=len(chunks),
                    )
                )
            current_heading = block_heading
            current_page = block_page
            continue

        if not current_parts:
            current_heading = block_heading
            current_page = block_page

        current_parts.append(block)
        current_length += len(block) + 2

    if current_parts:
        chunks.append(
            _build_chunk(
                document=document,
                heading=current_heading,
                page=current_page,
                text="\n\n".join(current_parts).strip(),
                index=len(chunks),
            )
        )

    return chunks


def _split_blocks(markdown: str) -> list[str]:
    """按空行把 Markdown 粗分成语义 block。"""
    return [part.strip() for part in re.split(r"\n\s*\n+", markdown) if part.strip()]


def _split_large_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """处理超长段落：退化为固定窗口滑动切分。"""
    pieces: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(block):
        end = min(len(block), start + chunk_size)
        pieces.append(block[start:end].strip())
        if end == len(block):
            break
        start += step
    return [piece for piece in pieces if piece]


def _carry_overlap(parts: list[str], chunk_overlap: int) -> list[str]:
    """从上一块尾部回收少量文本，作为下一个 chunk 的重叠上下文。"""
    if chunk_overlap == 0:
        return []

    carried: list[str] = []
    total = 0
    for part in reversed(parts):
        carried.insert(0, part)
        total += len(part)
        if total >= chunk_overlap:
            break
    return carried


def _build_chunk(document: MarkdownDocument, heading: str, page: int | None, text: str, index: int) -> Chunk:
    """构造最终的 `Chunk` 对象，并补齐稳定 metadata。"""
    source_name = Path(document.source_path).stem
    # chunk_id 不直接用随机 uuid，而是让相同文本在同一来源下得到稳定 id，
    # 这样更方便做测试、去重和重复 ingest 场景的比对。
    chunk_key = f"{source_name}-{index}-{uuid.uuid5(uuid.NAMESPACE_URL, text)}"
    return Chunk(
        chunk_id=chunk_key,
        text=text,
        metadata={
            "source_path": str(document.source_path),
            "source_name": source_name,
            "heading": heading,
            "page": page,
            "chunk_index": index,
            "text_length": len(text),
        },
    )
