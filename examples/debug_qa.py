"""调试 `ask_question` 的示例脚本。

适合学习：
- QA 层如何复用 search 结果
- prompt 是如何送到 LLM provider 的
- 最终 `QAResult` 会包含哪些关键信息

这里使用假的 OpenAI-compatible 响应，避免依赖真实远程服务。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _debug_common import build_demo_chunks, get_embedding_model_name, preview_text, print_kv, print_title

import pdf2md_rag.simple_qa as simple_qa_module
from pdf2md_rag.embeddings import build_embedder
from pdf2md_rag.simple_qa import ask_question
from pdf2md_rag.vectorstore import upsert_chunks


def main() -> None:
    print_title("debug_qa")

    chunks = build_demo_chunks()
    model_name = get_embedding_model_name()
    embedder = build_embedder(embedder_type="sentence-transformers", model_name=model_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = Path(temp_dir) / "chroma"
        upsert_chunks(
            chunks=chunks,
            embeddings=embedder.embed_texts([chunk.text for chunk in chunks]),
            persist_directory=persist_directory,
            collection_name="debug-qa",
        )

        original_post_json = simple_qa_module._post_json

        # 用假的 HTTP 响应代替真实 LLM 服务，方便离线学习 ask_question 的流程。
        def fake_post_json(url: str, payload: dict, headers: dict) -> dict:
            user_prompt = payload["messages"][1]["content"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Lasso is a lookup argument protocol used in proof systems. "
                                "Sources: [Source 1]\n\n"
                                f"Prompt preview: {preview_text(user_prompt, limit=120)}"
                            )
                        }
                    }
                ]
            }

        simple_qa_module._post_json = fake_post_json
        try:
            result = ask_question(
                question="What is Lasso?",
                collection_name="debug-qa",
                chroma_dir=persist_directory,
                top_k=2,
                embedder_type="sentence-transformers",
                embedding_model=model_name,
                llm_provider="openai-compatible",
                llm_model="debug-model",
                llm_base_url="http://debug.local",
            )
        finally:
            simple_qa_module._post_json = original_post_json

        print_kv("provider", result.provider)
        print_kv("model", result.model)
        print_kv("embedding_model", model_name)
        print_kv("sources", result.search_result.sources)
        print_kv("answer_preview", preview_text(result.answer, limit=320))


if __name__ == "__main__":
    main()
