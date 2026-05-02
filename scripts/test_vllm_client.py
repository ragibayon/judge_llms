import importlib.util
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich_logger import (
    console,
    log_api_base,
    log_progress,
    print_feedback_panel,
    setup_logger,
)


def load_prometheus_module():
    module_path = Path(__file__).resolve().parent.parent / "src" / "prometheus-eval" / "prometheus-eval.py"
    spec = importlib.util.spec_from_file_location("prometheus_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    load_dotenv()
    setup_logger("test_vllm_client")
    api_base = os.getenv("PROMETHEUS_API_BASE", "http://127.0.0.1:8000")
    output_path = Path(os.getenv("TEST_OUTPUT_PATH", "results/test_vllm_client_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = load_prometheus_module()
    evaluator = module.PrometheusEvaluator(api_base=api_base)
    log_api_base(api_base)

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

    from rich.table import Table

    summary_table = Table(title="Prometheus Smoke Test Scores")
    summary_table.add_column("Domain", style="cyan")
    summary_table.add_column("Score", style="magenta")

    for name, result in results.items():
        summary_table.add_row(name, str(result.score))
        print_feedback_panel(f"{name} | score={result.score}", result.feedback or result.raw_output)

    console.print(summary_table)

    row = {
        "source": source,
        "summary": summary,
        "domain_task": task,
    }
    for name, result in results.items():
        row[f"{name}_score"] = result.score
        row[f"{name}_feedback"] = result.feedback

    output_path.write_text(json.dumps([row], indent=2))
    log_progress(f"Saved results to {output_path}")

    csv_path = output_path.with_suffix(".csv")
    fieldnames = list(row.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    log_progress(f"Saved CSV results to {csv_path}")


if __name__ == "__main__":
    main()
