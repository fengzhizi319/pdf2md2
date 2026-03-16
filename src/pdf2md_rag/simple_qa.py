"""最小化 RAG 问答层。

这个模块站在检索层之上：
- 先调用 `search_chunks` 取回上下文
- 再把上下文包装成 prompt
- 最后调用本地或远程 LLM 后端

它不负责训练或复杂 agent 逻辑，只演示“检索增强问答”的最短闭环。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

import typer

from .search import SearchResult, search_chunks

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful RAG assistant. Answer only from the provided context. "
    "If the context is insufficient, say so clearly. Always cite the sources you used."
)

app = typer.Typer(help="Ask questions against the local Chroma store with a local or remote LLM.")


@dataclass(slots=True)
class QAResult:
    """一次问答执行后的结构化结果。"""

    question: str
    answer: str
    search_result: SearchResult
    provider: str
    model: str
    raw_response: dict[str, Any]


def ask_question(
    question: str,
    collection_name: str,
    chroma_dir: str | Path = "data/chroma",
    top_k: int = 5,
    embedder_type: str = "sentence-transformers",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    hash_dimensions: int = 384,
    llm_provider: str = "openai-compatible",
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "http://localhost:11434",
    api_key: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 800,
    max_context_chars: int = 6000,
) -> QAResult:
    """执行一次简化版 RAG 问答。

    流程很直接：
    1. 先检索相关 chunk
    2. 把上下文拼进 prompt
    3. 把 prompt 发给指定的 LLM 后端
    """
    search_result = search_chunks(
        question=question,
        collection_name=collection_name,
        persist_directory=chroma_dir,
        top_k=top_k,
        embedder_type=embedder_type,
        embedding_model=embedding_model,
        hash_dimensions=hash_dimensions,
        max_context_chars=max_context_chars,
    )
    prompt = _build_user_prompt(question, search_result)
    provider = llm_provider.strip().lower()

    if provider in {"openai", "openai-compatible", "openai_compatible", "openai-compatible-http"}:
        raw = _call_openai_compatible(
            base_url=llm_base_url,
            model=llm_model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = _extract_openai_answer(raw)
    elif provider == "ollama":
        raw = _call_ollama(
            base_url=llm_base_url,
            model=llm_model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=temperature,
        )
        answer = _extract_ollama_answer(raw)
    else:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    return QAResult(
        question=question,
        answer=answer.strip(),
        search_result=search_result,
        provider=provider,
        model=llm_model,
        raw_response=raw,
    )


def _build_user_prompt(question: str, search_result: SearchResult) -> str:
    """把检索结果包装成给 LLM 的用户提示词。"""
    return (
        f"Answer the question using only the context below.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{search_result.context_text}\n\n"
        "Instructions:\n"
        "- Be concise but complete.\n"
        "- If the context is insufficient, say what is missing.\n"
        "- End with a 'Sources:' line that cites the relevant source labels."
    )


def _call_openai_compatible(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """调用兼容 `/v1/chat/completions` 的后端。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return _post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, headers)


def _call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> dict[str, Any]:
    """调用本地或远程的 Ollama `/api/chat` 接口。"""
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    return _post_json(
        f"{base_url.rstrip('/')}/api/chat",
        payload,
        {"Content-Type": "application/json"},
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """发送一个最小的 JSON POST 请求。"""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_openai_answer(raw: dict[str, Any]) -> str:
    """从 OpenAI 风格响应中取出文本答案。"""
    choices = raw.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content", ""))


def _extract_ollama_answer(raw: dict[str, Any]) -> str:
    """从 Ollama 风格响应中取出文本答案。"""
    return str((raw.get("message") or {}).get("content", ""))


@app.command()
def main(
    question: str = typer.Argument(..., help="Question to ask over the local knowledge base."),
    collection: str = typer.Option("pdf-knowledge-base", help="Chroma collection name."),
    chroma_dir: Path = typer.Option(Path("data/chroma"), help="Directory where Chroma is persisted."),
    top_k: int = typer.Option(5, min=1, max=20, help="How many chunks to retrieve."),
    embedder: str = typer.Option("sentence-transformers", help="Embedding backend used for retrieval."),
    embedding_model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Embedding model name."),
    hash_dimensions: int = typer.Option(384, min=8, help="Hash embedder vector dimension for offline testing."),
    llm_provider: str = typer.Option("openai-compatible", help="LLM backend: openai-compatible or ollama."),
    llm_model: str = typer.Option("gpt-4o-mini", help="Remote/local LLM model name."),
    llm_base_url: str = typer.Option("http://localhost:11434", help="Base URL of the chat completion server."),
    api_key: str | None = typer.Option(None, help="API key for OpenAI-compatible providers."),
    max_context_chars: int = typer.Option(6000, min=500, help="Max characters of retrieval context passed to the LLM."),
) -> None:
    """CLI 入口：把问答结果打印到终端。"""
    result = ask_question(
        question=question,
        collection_name=collection,
        chroma_dir=chroma_dir,
        top_k=top_k,
        embedder_type=embedder,
        embedding_model=embedding_model,
        hash_dimensions=hash_dimensions,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        api_key=api_key,
        max_context_chars=max_context_chars,
    )
    typer.echo("Answer:\n")
    typer.echo(result.answer)
    typer.echo("\nSources:")
    for source in result.search_result.sources:
        typer.echo(f"- {source}")


if __name__ == "__main__":
    app()
