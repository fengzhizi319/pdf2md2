"""Embedding 抽象层。

这个模块统一了两类 embedding 后端：
- 真正用于语义检索的 `sentence-transformers`
- 用于离线测试和快速调试的 `hash`

主流程只依赖 `build_embedder` 返回的统一接口，不需要关心具体后端实现。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol


class Embedder(Protocol):
    """项目里所有 embedding 实现都遵守的最小接口。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class HashingEmbedder:
    """纯本地、零依赖的快速 embedder。

    它不具备真正的语义理解能力，但很适合：
    - 离线验证整条管线是否能跑通
    - 测试中避免下载模型
    - 调试向量库写入/查询逻辑
    """

    dimensions: int = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # 通过重复哈希构造一个稳定的伪向量；相同输入总会得到相同结果。
        values = [0.0] * self.dimensions
        encoded = text.encode("utf-8")
        digest_stream = b""
        while len(digest_stream) < self.dimensions * 4:
            encoded = hashlib.sha256(encoded).digest()
            digest_stream += encoded
        for index in range(self.dimensions):
            raw = digest_stream[index * 4 : (index + 1) * 4]
            value = int.from_bytes(raw, byteorder="big", signed=False)
            values[index] = (value / 2**32) * 2 - 1
        # 最后做单位化，让它更接近真实 embedding 常见的余弦相似度使用方式。
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """真正具备语义检索能力的 embedding 实现。"""

    model_name: str
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers and torch are required for the default embedder"
            ) from exc

        # Apple Silicon 上优先尝试 MPS；否则退回 CPU。
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_embedder(
    embedder_type: str,
    model_name: str,
    hash_dimensions: int = 384,
) -> Embedder:
    """按配置创建 embedding 实现，隐藏具体后端差异。"""
    normalized = embedder_type.strip().lower()
    if normalized == "hash":
        return HashingEmbedder(dimensions=hash_dimensions)
    if normalized in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unsupported embedder_type: {embedder_type}")
