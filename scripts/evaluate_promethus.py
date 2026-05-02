from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset_loader import prepare_fixed_sample

DATASET_ORDER = ["summeval", "dialsummeval"]

HUMAN_SCORE_KEY_MAP = {
    "coherence": "coherence",
    "factual_consistency": "consistency",
    "fairness": None,
    "fluency": "fluency",
    "relevance": "relevance",
}

console = Console()
log_console: Console | None = None


def setup_logger() -> Path:
    global log_console

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"evaluate_promethus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_console = Console(file=log_path.open("w", encoding="utf-8"), force_terminal=False)
    console.log(f"Writing log to {log_path}")
    return log_path


def log_message(message: str) -> None:
    console.log(message)
    if log_console is not None:
        log_console.print(message)


def load_prometheus_module():
    module_path = ROOT / "src" / "prometheus-eval" / "prometheus-eval.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Prometheus module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("prometheus_eval", module_path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")

    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    rows = read_jsonl(jsonl_path)

    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_path, index=False)


def make_run_key(sample: Dict[str, Any]) -> str:
    dataset = sample.get("dataset", "")
    sample_index = sample.get("sample_index", "")
    original_index = sample.get("original_index", "")
    sample_id = sample.get("sample_id", "")
    system_id = sample.get("system_id", "")

    return f"{dataset}::{sample_index}::{original_index}::{sample_id}::{system_id}"


def load_completed_keys(result_jsonl_path: Path) -> set[str]:
    completed = set()

    for row in read_jsonl(result_jsonl_path):
        run_key = row.get("run_key")
        if run_key:
            completed.add(run_key)

    return completed


def get_human_score(sample: Dict[str, Any], prometheus_domain: str) -> Optional[float]:
    human_key = HUMAN_SCORE_KEY_MAP.get(prometheus_domain)

    if human_key is None:
        return None

    return sample.get("human_scores", {}).get(human_key)


def flatten_human_scores(sample: Dict[str, Any]) -> Dict[str, Any]:
    scores = sample.get("human_scores", {}) or {}

    return {
        "human_consistency": scores.get("consistency"),
        "human_relevance": scores.get("relevance"),
        "human_fluency": scores.get("fluency"),
        "human_coherence": scores.get("coherence"),
    }


def flatten_baseline_scores(sample: Dict[str, Any]) -> Dict[str, Any]:
    scores = sample.get("baseline_scores", {}) or {}

    return {f"baseline_{key}": value for key, value in scores.items()}


def flatten_prometheus_results(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}

    for domain, result in evaluation.items():
        flat[f"prometheus_{domain}_score"] = result.score
        flat[f"prometheus_{domain}_feedback"] = result.feedback
        flat[f"prometheus_{domain}_raw_output"] = result.raw_output

    return flat


def evaluate_sample(
    *,
    sample: Dict[str, Any],
    evaluator: Any,
    global_index: int,
) -> Dict[str, Any]:
    evaluation = evaluator.evaluate_all(
        summary=sample.get("summary", ""),
        safe_source=sample.get("source", ""),
        domain_task="Evaluate the quality of the summary against the source document.",
        domain_format="summary",
    )

    row = {
        "index": global_index,
        "run_key": make_run_key(sample),
        "dataset": sample.get("dataset", ""),
        "sample_index": sample.get("sample_index", ""),
        "original_index": sample.get("original_index", ""),
        "sample_id": sample.get("sample_id", ""),
        "system_id": sample.get("system_id", ""),
        "source": sample.get("source", ""),
        "reference": sample.get("reference", ""),
        "summary": sample.get("summary", ""),
        "human_scores_json": json.dumps(
            sample.get("human_scores", {}), ensure_ascii=False
        ),
    }

    row.update(flatten_human_scores(sample))
    row.update(flatten_baseline_scores(sample))

    for domain in evaluation:
        row[f"human_for_{domain}"] = get_human_score(sample, domain)

    row.update(flatten_prometheus_results(evaluation))

    return row


def ensure_prepared_file(
    *,
    dataset_name: str,
    prepared_dir: Path,
    seed: int,
) -> Path:
    fixed_sample_count = 100
    sample_path = prepared_dir / f"{dataset_name}_{fixed_sample_count}_seed{seed}.jsonl"

    if sample_path.exists():
        print(f"Using prepared sample file: {sample_path}")
        return sample_path

    print(f"Prepared sample file not found. Creating: {sample_path}")

    prepare_fixed_sample(
        dataset_name=dataset_name,
        sample_count=fixed_sample_count,
        seed=seed,
        output_dir=prepared_dir,
    )

    return sample_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["summeval", "dialsummeval", "both"],
        default=os.getenv("DATASET", "both"),
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=int(os.getenv("SAMPLE_COUNT", "100")),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
    )

    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path(os.getenv("PREPARED_DIR", "data/prepared")),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.getenv("RESULTS_DIR", "results/prometheus")),
    )

    parser.add_argument(
        "--api-base",
        default=os.getenv("PROMETHEUS_API_BASE", "http://127.0.0.1:8000"),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("PROMETHEUS_API_KEY", "EMPTY"),
    )

    return parser.parse_args()


def selected_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg == "both":
        return DATASET_ORDER
    return [dataset_arg]


def main() -> None:
    load_dotenv()
    setup_logger()
    args = parse_args()

    log_message(f"Prometheus API base: {args.api_base}")
    log_message(
        f"Rows to evaluate from prepared 100-sample file: {args.sample_count}"
    )
    log_message(f"Seed: {args.seed}")

    module = load_prometheus_module()
    evaluator = module.PrometheusEvaluator(
        api_base=args.api_base,
        api_key=args.api_key,
    )

    global_index = 1

    for dataset_name in selected_datasets(args.dataset):
        console.rule(f"Running dataset: {dataset_name}")
        if log_console is not None:
            log_console.print(f"\n===== Running dataset: {dataset_name} =====")

        sample_path = ensure_prepared_file(
            dataset_name=dataset_name,
            prepared_dir=args.prepared_dir,
            seed=args.seed,
        )

        samples = read_jsonl(sample_path)
        samples = samples[: args.sample_count]

        result_jsonl_path = (
            args.results_dir / f"{dataset_name}_prometheus_results.jsonl"
        )
        result_csv_path = args.results_dir / f"{dataset_name}_prometheus_results.csv"

        completed_keys = load_completed_keys(result_jsonl_path)

        log_message(f"Prepared rows: {len(samples)}")
        log_message(f"Already completed: {len(completed_keys)}")
        log_message(f"Result JSONL: {result_jsonl_path}")
        log_message(f"Result CSV:   {result_csv_path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(f"{dataset_name}", total=len(samples))

            for local_idx, sample in enumerate(samples, start=1):
                run_key = make_run_key(sample)
                sample_id = sample.get("sample_id", "")
                system_id = sample.get("system_id", "")
                progress.update(
                    task_id,
                    description=(
                        f"{dataset_name} sample={sample_id} system={system_id} "
                        f"row={local_idx}/{len(samples)}"
                    ),
                )

                if run_key in completed_keys:
                    log_message(
                        f"[SKIP] dataset={dataset_name} row={local_idx}/{len(samples)} "
                        f"sample_id={sample_id} system_id={system_id}"
                    )
                    global_index += 1
                    progress.advance(task_id)
                    continue

                log_message(
                    f"[RUN] dataset={dataset_name} row={local_idx}/{len(samples)} "
                    f"sample_id={sample_id} system_id={system_id} run_key={run_key}"
                )
                log_message(f"Summary preview: {sample.get('summary', '')[:200]}")

                try:
                    result_row = evaluate_sample(
                        sample=sample,
                        evaluator=evaluator,
                        global_index=global_index,
                    )
                    result_row["status"] = "ok"
                except Exception as exc:
                    result_row = {
                        "index": global_index,
                        "run_key": run_key,
                        "dataset": sample.get("dataset", dataset_name),
                        "sample_index": sample.get("sample_index", ""),
                        "original_index": sample.get("original_index", ""),
                        "sample_id": sample.get("sample_id", ""),
                        "system_id": sample.get("system_id", ""),
                        "source": sample.get("source", ""),
                        "reference": sample.get("reference", ""),
                        "summary": sample.get("summary", ""),
                        "human_scores_json": json.dumps(
                            sample.get("human_scores", {}),
                            ensure_ascii=False,
                        ),
                        "status": "error",
                        "error": repr(exc),
                    }
                    log_message(
                        f"[ERROR] dataset={dataset_name} row={local_idx}/{len(samples)} "
                        f"sample_id={sample_id} system_id={system_id}: {exc}"
                    )

                append_jsonl(result_row, result_jsonl_path)
                completed_keys.add(run_key)

                write_csv_from_jsonl(result_jsonl_path, result_csv_path)

                log_message(
                    f"[SAVED] dataset={dataset_name} row={local_idx}/{len(samples)} "
                    f"sample_id={sample_id} system_id={system_id}"
                )
                global_index += 1
                progress.advance(task_id)

        write_csv_from_jsonl(result_jsonl_path, result_csv_path)
        log_message(f"Finished dataset: {dataset_name}")
        log_message(f"Saved JSONL: {result_jsonl_path}")
        log_message(f"Saved CSV:   {result_csv_path}")


if __name__ == "__main__":
    main()
