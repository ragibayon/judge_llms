"""
Dataset loader for SummEval and DialSummEval.

What this does:
1. Loads SummEval from Hugging Face.
2. Loads DialSummEval from local JSONL.
3. Normalizes both datasets into one common format.
4. Samples fixed 100 rows with seed 42.
5. Saves prepared samples as JSONL and CSV.

Normalized row format:
{
    "dataset": "summeval" | "dialsummeval",
    "sample_id": "...",
    "source": "...",       # source article or dialogue
    "reference": "...",    # gold/reference summary if available
    "summary": "...",      # system summary to evaluate
    "system_id": "...",
    "human_scores": {
        "consistency": float | None,
        "relevance": float | None,
        "fluency": float | None,
        "coherence": float | None
    }
}
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


class DatasetLoader:
    SUPPORTED_DATASETS = {
        "summeval": {
            "hf_path": "KnutJaegersberg/summeval_pairs",
            "split": None,
        },
        "dialsummeval": {
            "local_path": "data/DialSummEval.jsonl",
            "split": "train",
        },
    }

    def __init__(
        self,
        dataset_name: str,
        supported_datasets: Optional[dict] = None,
        random_seed: int = 42,
        domain_filter: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.supported_datasets = supported_datasets or self.SUPPORTED_DATASETS
        self.random_seed = random_seed
        self.domain_filter = domain_filter
        self.rng = random.Random(random_seed)

        if self.dataset_name not in self.supported_datasets:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {list(self.supported_datasets.keys())}"
            )

    @staticmethod
    def _safe_score(value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, list):
            numeric_values = []
            for item in value:
                try:
                    numeric_values.append(float(item))
                except (TypeError, ValueError):
                    continue
            return sum(numeric_values) / len(numeric_values) if numeric_values else None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _average(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def load_dataset(self) -> List[Dict[str, Any]]:
        if self.dataset_name == "summeval":
            return self._load_summeval()

        if self.dataset_name == "dialsummeval":
            return self._load_dialsummeval()

        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_summeval(self) -> List[Dict[str, Any]]:
        info = self.supported_datasets["summeval"]
        hf_path = info["hf_path"]

        print(f"Loading SummEval from Hugging Face: {hf_path}")
        dataset_obj = load_dataset(hf_path)

        dataset = self._select_split(dataset_obj, preferred_split=info.get("split"))
        records = [dict(row) for row in dataset]

        print(f"Loaded SummEval: {len(records)} rows")
        return records

    def _load_dialsummeval(self) -> List[Dict[str, Any]]:
        info = self.supported_datasets["dialsummeval"]
        local_path = Path(info["local_path"])

        if not local_path.exists():
            raise FileNotFoundError(
                f"DialSummEval file not found: {local_path}\n"
                "Place the dataset at data/DialSummEval.jsonl"
            )

        print(f"Loading DialSummEval from local JSONL: {local_path}")
        dataset = load_dataset("json", data_files=str(local_path), split=info["split"])
        records = [dict(row) for row in dataset]

        print(f"Loaded DialSummEval: {len(records)} rows")
        return records

    @staticmethod
    def _select_split(
        dataset_obj: Dataset | DatasetDict,
        preferred_split: Optional[str] = None,
    ) -> Dataset:
        if isinstance(dataset_obj, Dataset):
            return dataset_obj

        if preferred_split and preferred_split in dataset_obj:
            return dataset_obj[preferred_split]

        for split_name in ("test", "validation", "train"):
            if split_name in dataset_obj:
                print(f"Using split: {split_name}")
                return dataset_obj[split_name]

        first_split = list(dataset_obj.keys())[0]
        print(f"Using first available split: {first_split}")
        return dataset_obj[first_split]

    def sample_posts(
        self, dataset: List[Dict[str, Any]], num_posts: int
    ) -> List[Dict[str, Any]]:
        if num_posts <= 0:
            raise ValueError("num_posts must be greater than 0")

        if num_posts > len(dataset):
            print(
                f"Requested {num_posts} rows, but dataset only has {len(dataset)}. "
                "Using all rows."
            )
            num_posts = len(dataset)

        indices = self.rng.sample(range(len(dataset)), num_posts)

        samples = []
        for sampled_index, original_index in enumerate(indices, start=1):
            raw_row = dataset[original_index]
            processed = self._process_post(raw_row)
            processed["sample_index"] = sampled_index
            processed["original_index"] = original_index
            samples.append(processed)

        print(
            f"Sampled {len(samples)} rows from {self.dataset_name} "
            f"with seed={self.random_seed}"
        )
        return samples

    def process_all_posts(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for idx, raw_row in enumerate(dataset, start=1):
            processed = self._process_post(raw_row)
            processed["sample_index"] = idx
            processed["original_index"] = idx - 1
            rows.append(processed)
        return rows

    def _process_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset_name == "summeval":
            return self._process_summeval(post)

        if self.dataset_name == "dialsummeval":
            return self._process_dialsummeval(post)

        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _process_summeval(self, post: Dict[str, Any]) -> Dict[str, Any]:
        source = str(post.get("text", ""))

        references = post.get("references", [])
        if isinstance(references, list) and references:
            reference = str(references[0])
        else:
            reference = str(references or "")

        summary = str(post.get("decoded", ""))

        human_scores = self._average_annotation_scores(
            annotations=post.get("expert_annotations", []),
            metrics=("consistency", "relevance", "fluency", "coherence"),
        )

        sample_id = post.get("id") or post.get("uid") or post.get("sample_id") or ""

        system_id = (
            post.get("system_id")
            or post.get("model_id")
            or post.get("model")
            or post.get("model_name")
            or "summeval_system"
        )

        row = {
            "dataset": "summeval",
            "sample_id": str(sample_id),
            "source": source,
            "reference": reference,
            "summary": summary,
            "system_id": str(system_id),
            "human_scores": human_scores,
        }

        baseline_scores = self._extract_summeval_baseline_scores(post)
        if baseline_scores:
            row["baseline_scores"] = baseline_scores

        return row

    def _process_dialsummeval(self, post: Dict[str, Any]) -> Dict[str, Any]:
        human_scores = self._average_annotation_scores(
            annotations=post.get("annotations", []),
            metrics=("consistency", "relevance", "fluency", "coherence"),
        )

        return {
            "dataset": "dialsummeval",
            "sample_id": str(post.get("id", "")),
            "source": str(post.get("dialogue", "")),
            "reference": "",
            "summary": str(post.get("summary", "")),
            "system_id": str(post.get("model_id", "unknown_model")),
            "human_scores": human_scores,
        }

    def _average_annotation_scores(
        self,
        annotations: Any,
        metrics: Iterable[str],
    ) -> Dict[str, Optional[float]]:
        score_lists: Dict[str, List[float]] = {metric: [] for metric in metrics}

        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue

                for metric in metrics:
                    score = self._safe_score(annotation.get(metric))
                    if score is not None:
                        score_lists[metric].append(score)

        return {metric: self._average(scores) for metric, scores in score_lists.items()}

    def _extract_summeval_baseline_scores(self, post: Dict[str, Any]) -> Dict[str, Any]:
        raw_metrics = post.get("metric_scores_11", {})

        if isinstance(raw_metrics, str):
            try:
                metrics = ast.literal_eval(raw_metrics)
            except (ValueError, SyntaxError):
                metrics = {}
        elif isinstance(raw_metrics, dict):
            metrics = raw_metrics
        else:
            metrics = {}

        rouge = metrics.get("rouge", {})
        if not isinstance(rouge, dict):
            rouge = {}

        return {
            "rouge_1_f": rouge.get("rouge_1_f_score"),
            "rouge_2_f": rouge.get("rouge_2_f_score"),
            "rouge_l_f": rouge.get("rouge_l_f_score"),
            "rouge_we_1_f": metrics.get("rouge_we_1_f"),
            "rouge_we_2_f": metrics.get("rouge_we_2_f"),
            "rouge_we_3_f": metrics.get("rouge_we_3_f"),
            "bert_score_f1": metrics.get("bert_score_f1"),
            "bleu": metrics.get("bleu"),
            "meteor": metrics.get("meteor"),
            "chrf": metrics.get("chrf"),
            "mover_score": metrics.get("mover_score"),
            "cider": metrics.get("cider"),
            "s3_pyr": metrics.get("s3_pyr"),
            "s3_resp": metrics.get("s3_resp"),
            "sms": metrics.get("sentence_movers_glove_sms"),
        }


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    flat_rows = [flatten_row(row) for row in rows]

    fieldnames = []
    seen = set()
    for row in flat_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


def flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}

    for key, value in row.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}_{nested_key}"] = nested_value
        else:
            flat[key] = value

    return flat


def prepare_fixed_sample(
    dataset_name: str,
    sample_count: int,
    seed: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    loader = DatasetLoader(dataset_name, random_seed=seed)
    raw_dataset = loader.load_dataset()
    samples = loader.sample_posts(raw_dataset, sample_count)

    jsonl_path = output_dir / f"{dataset_name}_{sample_count}_seed{seed}.jsonl"
    csv_path = output_dir / f"{dataset_name}_{sample_count}_seed{seed}.csv"

    write_jsonl(samples, jsonl_path)
    write_csv(samples, csv_path)

    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved CSV:   {csv_path}")

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["summeval", "dialsummeval", "both"],
        default="both",
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_names = (
        ["summeval", "dialsummeval"] if args.dataset == "both" else [args.dataset]
    )

    for dataset_name in dataset_names:
        prepare_fixed_sample(
            dataset_name=dataset_name,
            sample_count=args.sample_count,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
