import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


def load_prometheus_module():
    module_path = Path(__file__).resolve().parent.parent / "src" / "prometheus-eval" / "prometheus-eval.py"
    spec = importlib.util.spec_from_file_location("prometheus_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    load_dotenv()
    api_base = os.getenv("PROMETHEUS_API_BASE", "http://127.0.0.1:8000")
    module = load_prometheus_module()
    evaluator = module.PrometheusEvaluator(api_base=api_base)
    console.log(f"Using Prometheus API at {api_base}")

    source = (
        "The city council approved a $2 million park renovation plan. "
        "Supporters said the upgrade would improve safety and accessibility, "
        "while critics argued the budget should prioritize road repairs first."
    )
    summary = (
        "The city council approved a $2 million renovation for the park. "
        "Backers said it would improve safety and accessibility, while critics "
        "said road repairs should come first."
    )
    task = "Summarize the article faithfully."

    results = evaluator.evaluate_all(
        summary=summary,
        safe_source=source,
        domain_task=task,
    )

    summary_table = Table(title="Prometheus Smoke Test Scores")
    summary_table.add_column("Domain", style="cyan")
    summary_table.add_column("Score", style="magenta")

    for name, result in results.items():
        summary_table.add_row(name, str(result.score))
        console.print(
            Panel(
                result.feedback or result.raw_output,
                title=f"{name} | score={result.score}",
                expand=False,
            )
        )

    console.print(summary_table)


if __name__ == "__main__":
    main()
