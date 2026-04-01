import importlib.util
import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
ADVANCED_EXAMPLE_PATH = EXAMPLES_DIR / "debug_langgraph_rag_advanced.py"


def _load_module():
    module_name = "debug_langgraph_rag_advanced"
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))

    spec = importlib.util.spec_from_file_location(module_name, ADVANCED_EXAMPLE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_app(tmp_path: Path):
    module = _load_module()
    runtime = module.RuntimeConfig(
        embedder_type="hash",
        embedding_model="unused",
        hash_dimensions=128,
        collection_name="advanced-test",
        top_k=3,
        max_context_chars=1800,
        llm_provider="fake",
        llm_model="fake-production-rag",
        llm_base_url="<disabled>",
        api_key=None,
        system_prompt="Test prompt",
        temperature=0.0,
        max_tokens=300,
        max_retries=2,
        min_overlap_for_good=2,
    )
    persist_directory = tmp_path / "chroma"
    module.prepare_demo_collection(runtime, persist_directory)
    app = module.build_demo_graph(runtime, persist_directory)
    return module, app


def test_advanced_graph_direct_hit_generates_without_retry(tmp_path: Path) -> None:
    _, app = _build_app(tmp_path)

    final_state = app.invoke({"user_question": "What does retrieval grading check?", "max_retries": 2})

    assert final_state["retrieval_grade"] == "strong"
    assert final_state["retry_count"] == 0
    assert "retry_search" not in final_state["trace"]
    assert final_state["trace"][-1] == "generate_answer"



def test_advanced_graph_retry_path_eventually_generates(tmp_path: Path) -> None:
    _, app = _build_app(tmp_path)

    final_state = app.invoke({"user_question": "How do retries help retrieval?", "max_retries": 2})

    assert final_state["retry_count"] == 1
    assert "retry_search" in final_state["trace"]
    assert final_state["trace"][-1] == "generate_answer"
    assert final_state["retrieval_grade"] == "strong"



def test_advanced_graph_falls_back_when_retries_are_exhausted(tmp_path: Path) -> None:
    _, app = _build_app(tmp_path)

    final_state = app.invoke({"user_question": "What GPU memory is required to deploy this stack?", "max_retries": 2})

    assert final_state["trace"][-1] == "fallback_answer"
    assert final_state["retrieval_grade"] == "weak"
    assert final_state["retry_count"] == 2
    assert "缺少足够强的证据" in final_state["answer"]

