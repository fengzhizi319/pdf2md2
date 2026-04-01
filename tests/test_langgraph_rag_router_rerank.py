import importlib.util
import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_PATH = EXAMPLES_DIR / "debug_langgraph_rag_router_rerank.py"



def _load_module():
    module_name = "debug_langgraph_rag_router_rerank"
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))

    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_PATH)
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
        collection_name="router-rerank-test",
        top_k=7,
        max_context_chars=2200,
        llm_provider="fake",
        llm_model="fake-router-rerank-rag",
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



def test_router_rerank_grounding_question_routes_and_generates(tmp_path: Path) -> None:
    _, app = _build_app(tmp_path)

    final_state = app.invoke({"user_question": "How do citations help grounding?", "max_retries": 2})

    assert final_state["route_label"] == "grounding"
    assert final_state["retrieval_plan"] == "citation-heavy"
    assert final_state["top_hit_after_rerank"].endswith("Grounding and Citations")
    assert final_state["trace"][-1] == "generate_answer"



def test_router_rerank_metadata_question_promotes_metadata_hit(tmp_path: Path) -> None:
    _, app = _build_app(tmp_path)

    final_state = app.invoke({"user_question": "Which heading or page should I show for a source?", "max_retries": 2})

    assert final_state["route_label"] == "metadata"
    assert final_state["retrieval_plan"] == "metadata-aware"
    assert final_state["top_hit_before_rerank"] != final_state["top_hit_after_rerank"]
    assert final_state["top_hit_after_rerank"].endswith("Metadata Queries")
    assert "rerank_hits" in final_state["trace"]



def test_router_rerank_hardware_question_falls_back_after_retries(tmp_path: Path) -> None:
    _, app = _build_app(tmp_path)

    final_state = app.invoke({"user_question": "What GPU memory is required to deploy this stack?", "max_retries": 2})

    assert final_state["route_label"] == "hardware"
    assert final_state["retrieval_plan"] == "hardware-scout"
    assert final_state["retry_count"] == 2
    assert final_state["retrieval_grade"] == "weak"
    assert final_state["trace"][-1] == "fallback_answer"
    assert "缺少足够强的证据" in final_state["answer"]

