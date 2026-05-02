from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_prometheus_eval():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "prometheus-eval"
        / "prometheus-eval.py"
    )
    spec = spec_from_file_location("prometheus_eval", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    prometheus_eval = load_prometheus_eval()
    sample_context = {
        "domain_format": "summary",
        "domain_task": "Summarize the article faithfully.",
        "safe_source": "The source text goes here.",
        "summary": "The generated summary goes here.",
    }

    for name in prometheus_eval.PROMPT_DICT:
        print(f"=== {name} ===")
        print(prometheus_eval.render_prompt(name, **sample_context))
        print()
