import importlib.util
import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "debug_langgraph_rag.py"



def _load_module():
    module_name = "debug_langgraph_rag_for_test"
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))

    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def test_resolve_runtime_config_defaults_ollama_model_to_qwen35(monkeypatch) -> None:
    monkeypatch.setenv("PDF2MD_RAG_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("PDF2MD_RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("PDF2MD_RAG_EMBEDDER", raising=False)
    monkeypatch.delenv("PDF2MD_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module = _load_module()
    runtime = module.resolve_runtime_config()

    assert runtime.llm_provider == "ollama"
    assert runtime.llm_model == "qwen3.5:0.8b"
    assert runtime.embedder_type == "hash"
    assert runtime.embedding_model == "unused"



def test_resolve_runtime_config_defaults_openai_compatible_embedding_stays_real(monkeypatch) -> None:
    monkeypatch.setenv("PDF2MD_RAG_LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("PDF2MD_RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("PDF2MD_RAG_EMBEDDER", raising=False)
    monkeypatch.delenv("PDF2MD_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module = _load_module()
    runtime = module.resolve_runtime_config()

    assert runtime.llm_provider == "openai-compatible"
    assert runtime.llm_model == "gpt-4o-mini"
    assert runtime.embedder_type == "sentence-transformers"
    assert runtime.embedding_model == "BAAI/bge-base-en-v1.5"



def test_probe_ollama_connectivity_uses_ollama_chat_endpoint(monkeypatch) -> None:
    module = _load_module()
    runtime = module.RuntimeConfig(
        embedder_type="hash",
        embedding_model="unused",
        hash_dimensions=128,
        collection_name="demo",
        top_k=3,
        max_context_chars=1800,
        llm_provider="ollama",
        llm_model="qwen3.5:0.8b",
        llm_base_url="http://localhost:11434",
        api_key=None,
        system_prompt="prompt",
        temperature=0.2,
        max_tokens=16,
    )

    captured: dict[str, object] = {}

    def fake_call_ollama(base_url: str, model: str, system_prompt: str, user_prompt: str, temperature: float) -> dict:
        captured["base_url"] = base_url
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["temperature"] = temperature
        return {"message": {"content": "OK"}}

    monkeypatch.setattr(module.simple_qa_module, "_call_ollama", fake_call_ollama)

    ok, detail = module.probe_ollama_connectivity(runtime)

    assert ok is True
    assert "reachable" in detail
    assert captured == {
        "base_url": "http://localhost:11434",
        "model": "qwen3.5:0.8b",
        "system_prompt": "You are a connectivity probe. Reply with OK only.",
        "user_prompt": "Reply with OK only.",
        "temperature": 0.0,
    }

