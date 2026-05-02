import importlib.util
import json
import os
from pathlib import Path

from data.dataset_loader import DatasetLoader
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
    sample_count = int(os.getenv("SAMPLE_COUNT", "3"))
    output_path = Path(os.getenv("OUTPUT_PATH", "results/summeval_vllm_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.log(f"Using Prometheus API at {api_base}")
    console.log(f"Evaluating {sample_count} sampled SummEval rows")

    loader = DatasetLoader("summeval", None, 42)
    dataset = loader.load_dataset()
    samples = loader.sample_posts(dataset, sample_count)

    module = load_prometheus_module()
    evaluator = module.PrometheusEvaluator(api_base=api_base)

    results = []
    for idx, sample in enumerate(samples, start=1):
        console.rule(f"Sample {idx}/{sample_count}")
        console.print(
            Panel(
                sample.get("summary", ""),
                title=f"Summary | system={sample.get('system_id', '')}",
                expand=False,
            )
        )
        evaluation = evaluator.evaluate_all(
            summary=sample.get("summary", ""),
            safe_source=sample.get("source", ""),
            domain_task="Evaluate the quality of the summary against the source document.",
            domain_format="summary",
        )
        score_table = Table(title=f"Prometheus Scores for Sample {idx}")
        score_table.add_column("Domain", style="cyan")
        score_table.add_column("Score", style="magenta")

        for domain, result in evaluation.items():
            score_table.add_row(domain, str(result.score))
            console.print(
                Panel(
                    result.feedback or result.raw_output,
                    title=f"{domain} | score={result.score}",
                    expand=False,
                )
            )

        console.print(score_table)
        row = {
            "index": idx,
            "source": sample.get("source", ""),
            "reference": sample.get("reference", ""),
            "summary": sample.get("summary", ""),
            "human_scores": sample.get("human_scores", {}),
            "baseline_scores": sample.get("baseline_scores", {}),
            "system_id": sample.get("system_id", ""),
            "prometheus_scores": {
                domain: {
                    "score": result.score,
                    "feedback": result.feedback,
                }
                for domain, result in evaluation.items()
            },
        }
        results.append(row)
        console.log(f"Evaluated sample {idx}/{sample_count}")

    output_path.write_text(json.dumps(results, indent=2))
    console.log(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
