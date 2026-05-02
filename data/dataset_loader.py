"""
Dataset loader module for various summarization datasets.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset, load_from_disk


class DatasetLoader:
    """Handles loading and sampling from various summarization datasets."""

    SUPPORTED_DATASETS = {
        "multi_news": {"path": "multi_news", "config": None},
        "cnndm": {"path": "cnn_dailymail", "config": "3.0.0"},
        "wikihow": {"path": "gursi26/wikihow-cleaned", "config": None},
        "govreport": {"path": "ccdv/govreport-summarization", "config": None},
        "meqsum": {"path": "albertvillanova/meqsum", "config": None},
        "squality": {"path": "pszemraj/SQuALITY-v1.3-flat", "config": None},
        "tweetsum": {"path": "Andyrasika/TweetSumm-tuned", "config": None},
        "dialogsum": {"path": "knkarthick/dialogsum", "config": None},
        "qmsum": {"path": "pszemraj/qmsum-cleaned", "config": None},
        "samsum": {"path": "knkarthick/samsum", "config": None},
        "unisumeval_all": {"path": "json", "config": None},
        "summeval": {
            "path": "KnutJaegersberg/summeval_pairs",
            "config": None,
            "local_path": "data/datasets/KnutJaegersberg__summeval_pairs",
            "split": "train",
        },
        "dialsummeval": {"path": "json", "config": None},
    }

    def __init__(
        self,
        dataset_name: str,
        supported_datasets: Optional[dict],
        random_seed: int,
        domain_filter: str = None,
    ) -> None:
        """
        Initialize dataset loader.

        Args:
            dataset_name: Name of the dataset to load
            supported_datasets: Optional override for supported dataset mapping
            random_seed: Seed for random sampling
            domain_filter: Optional domain filter for UniSumEval
        """
        self.dataset_name = dataset_name.lower()
        self.supported_datasets = supported_datasets or self.SUPPORTED_DATASETS
        self.random_seed = random_seed
        self.domain_filter = domain_filter
        random.seed(random_seed)

        if self.dataset_name not in self.supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Supported: {list(self.supported_datasets.keys())}"
            )

    @staticmethod
    def _safe_score(val) -> Optional[float]:
        """
        Convert a raw dataset value to a single float score.
        """
        if val is None:
            return None

        if isinstance(val, list):
            numeric = []
            for item in val:
                try:
                    numeric.append(float(item))
                except (TypeError, ValueError):
                    pass
            return sum(numeric) / len(numeric) if numeric else None

        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def load_dataset(self) -> List[Dict]:
        """Load the specified dataset."""
        if self.dataset_name.startswith("unisumeval"):
            raise NotImplementedError(
                "UniSumEval loading is not wired in this repo yet. "
                "Add the external config/data dependencies first."
            )

        if self.dataset_name == "dialsummeval":
            print("Loading DialSummEval from local JSONL...")
            try:
                dataset = load_dataset("json", data_files="data/DialSummEval.jsonl", split="train")
                dataset_records = [record for record in dataset]
                print(f"✓ DialSummEval loaded! Total samples: {len(dataset_records)}")
                return dataset_records
            except Exception as exc:
                print(f"Error loading DialSummEval: {exc}")
                raise

        ds_info = self.supported_datasets[self.dataset_name]
        local_path = ds_info.get("local_path")
        local_split = ds_info.get("split", "train")
        if local_path:
            local_dir = Path(local_path)
            if local_dir.exists():
                print(f"Loading '{self.dataset_name}' from local disk ({local_dir})...")
                dataset_dict = load_from_disk(str(local_dir))
                dataset = dataset_dict[local_split]
                dataset_records = [record for record in dataset]
                print(f"✓ Dataset loaded successfully. Total target samples: {len(dataset_records)}")
                return dataset_records

        path = ds_info["path"]
        config = ds_info.get("config")

        print(f"Loading '{self.dataset_name}' from Hugging Face Hub ({path})...")

        try:
            if config:
                dataset = load_dataset(path, config, split="test")
            else:
                dataset = load_dataset(path, split="test")
        except Exception as exc:
            print(f"  Test split not found ({exc}). Falling back to 'train' split...")
            if config:
                dataset = load_dataset(path, config, split="train")
            else:
                dataset = load_dataset(path, split="train")

        dataset_records = [record for record in dataset]
        print(f"✓ Dataset loaded successfully. Total target samples: {len(dataset_records)}")
        return dataset_records

    def sample_posts(self, dataset, num_posts: int) -> List[Dict]:
        """
        Randomly sample posts from dataset.
        """
        if num_posts > len(dataset):
            print(
                f"Warning: Requested {num_posts} posts but dataset has {len(dataset)}. "
                "Sampling all available."
            )
            num_posts = len(dataset)

        indices = random.sample(range(len(dataset)), num_posts)
        sampled_posts = []

        for idx in indices:
            post = dataset[idx]
            processed_post = self._process_post(post)
            sampled_posts.append(processed_post)

        print(f"Sampled {len(sampled_posts)} posts")
        return sampled_posts

    def _process_post(self, post: Dict) -> Dict:
        """
        Process a raw dataset row into a normalised dict with keys:
            source
            reference
        """
        processed = {"source": "", "reference": "", "human_scores": {}}

        if self.dataset_name == "multi_news":
            processed["source"] = str(post.get("document", ""))
            processed["reference"] = str(post.get("summary", ""))

        elif self.dataset_name == "cnndm":
            processed["source"] = str(post.get("article", ""))
            processed["reference"] = str(post.get("highlights", ""))

        elif self.dataset_name == "wikihow":
            processed["source"] = str(post.get("text", ""))
            processed["reference"] = str(post.get("summary", ""))

        elif self.dataset_name == "govreport":
            processed["source"] = str(post.get("report", ""))
            processed["reference"] = str(post.get("summary", ""))

        elif self.dataset_name == "squality":
            processed["source"] = str(post.get("document", ""))
            processed["reference"] = str(post.get("response", ""))

        elif self.dataset_name == "meqsum":
            processed["source"] = str(post.get("CHQ", ""))
            processed["reference"] = str(post.get("Summary", ""))

        elif self.dataset_name == "tweetsum":
            raw_text = str(post.get("text", post.get("conversation", "")))
            if "### Input" in raw_text and "### Response" in raw_text:
                try:
                    after_input = raw_text.split("### Input")[1]
                    clean_input = after_input.split("### Response")[0]
                    processed["source"] = clean_input.lstrip(":\n- ").strip()
                except IndexError:
                    processed["source"] = raw_text.strip()
            else:
                processed["source"] = raw_text.strip()
            processed["reference"] = str(post.get("summary", ""))

        elif self.dataset_name in ["dialogsum", "samsum"]:
            processed["source"] = str(post.get("dialogue", ""))
            processed["reference"] = str(post.get("summary", ""))

        elif self.dataset_name == "qmsum":
            processed["source"] = str(post.get("input", ""))
            processed["reference"] = str(post.get("output", ""))

        elif self.dataset_name.startswith("unisumeval"):
            processed["source"] = str(
                post.get("input_context", post.get("source", post.get("document", "")))
            )
            processed["reference"] = str(post.get("reference", ""))
            processed["summary"] = str(post.get("summary", ""))

            raw_domain = str(
                post.get("dataset", post.get("source", post.get("domain", "")))
            ).lower()
            unisumeval_to_dataset_key = {
                "news": "cnndm",
                "lifestyle": "wikihow",
                "booking": "tweetsum",
                "tweetsumm": "tweetsum",
                "multiwoz": "tweetsum",
                "daily life": "dialogsum",
                "report": "govreport",
                "med lit": "meqsum",
                "medical": "meqsum",
                "pubmed": "meqsum",
                "mimic": "meqsum",
                "sci-fi": "squality",
                "interview": "qmsum",
                "mediasum": "qmsum",
                "meeting": "samsum",
                "ami": "samsum",
                "meetingbank": "samsum",
            }

            mapped_key = None
            for key, value in unisumeval_to_dataset_key.items():
                if key in raw_domain:
                    mapped_key = value
                    break

            processed["original_dataset"] = mapped_key or raw_domain
            processed["human_scores"] = {
                "consistency": self._safe_score(
                    post.get("faithfulness", post.get("faithfulness_score"))
                ),
                "relevance": self._safe_score(post.get("completeness")),
                "fluency": self._safe_score(post.get("conciseness")),
                "coherence": "",
            }

        elif self.dataset_name == "summeval":
            processed["source"] = str(post.get("text", ""))

            references = post.get("references", [])
            processed["reference"] = str(
                references[0] if isinstance(references, list) and references else references
            )

            processed["summary"] = str(post.get("decoded", ""))

            expert_annots = post.get("expert_annotations", [])
            raw_scores = {"consistency": [], "relevance": [], "fluency": [], "coherence": []}

            if isinstance(expert_annots, list):
                for annot in expert_annots:
                    if isinstance(annot, dict):
                        for metric in raw_scores.keys():
                            if metric in annot and annot[metric] is not None:
                                raw_scores[metric].append(float(annot[metric]))

            def avg_score(score_list):
                return sum(score_list) / len(score_list) if score_list else None

            processed["human_scores"] = {
                "consistency": avg_score(raw_scores["consistency"]),
                "relevance": avg_score(raw_scores["relevance"]),
                "fluency": avg_score(raw_scores["fluency"]),
                "coherence": avg_score(raw_scores["coherence"]),
            }

            raw_metrics = post.get("metric_scores_11", {})
            if isinstance(raw_metrics, str):
                try:
                    m11 = ast.literal_eval(raw_metrics)
                except (ValueError, SyntaxError):
                    m11 = {}
            else:
                m11 = raw_metrics if isinstance(raw_metrics, dict) else {}

            rouge_metrics = m11.get("rouge", {})
            processed["baseline_scores"] = {
                "rouge_1_f": rouge_metrics.get("rouge_1_f_score"),
                "rouge_2_f": rouge_metrics.get("rouge_2_f_score"),
                "rouge_l_f": rouge_metrics.get("rouge_l_f_score"),
                "rouge_we_1_f": m11.get("rouge_we_1_f"),
                "rouge_we_2_f": m11.get("rouge_we_2_f"),
                "rouge_we_3_f": m11.get("rouge_we_3_f"),
                "bert_score_f1": m11.get("bert_score_f1"),
                "bleu": m11.get("bleu"),
                "meteor": m11.get("meteor"),
                "chrf": m11.get("chrf"),
                "mover_score": m11.get("mover_score"),
                "cider": m11.get("cider"),
                "s3_pyr": m11.get("s3_pyr"),
                "s3_resp": m11.get("s3_resp"),
                "sms": m11.get("sentence_movers_glove_sms"),
            }

            processed["system_id"] = str(post.get("model_id", "summeval_system"))

        elif self.dataset_name == "dialsummeval":
            processed["source"] = str(post.get("dialogue", ""))
            processed["summary"] = str(post.get("summary", ""))
            processed["reference"] = ""

            annotations = post.get("annotations", [])
            if isinstance(annotations, list) and len(annotations) > 0:
                def avg_score(metric):
                    scores = [a.get(metric) for a in annotations if a.get(metric) is not None]
                    return sum(scores) / len(scores) if scores else None

                processed["human_scores"] = {
                    "consistency": avg_score("consistency"),
                    "relevance": avg_score("relevance"),
                    "fluency": avg_score("fluency"),
                    "coherence": avg_score("coherence"),
                }

            processed["system_id"] = str(post.get("model_id", "unknown_model"))

        else:
            processed["source"] = str(
                post.get("text", post.get("document", post.get("article", "")))
            )
            processed["reference"] = str(
                post.get("summary", post.get("highlights", ""))
            )

        sys_id = post.get(
            "system_id",
            post.get("model_id", post.get("model", post.get("model_name", post.get("system")))),
        )
        if sys_id:
            processed["system_id"] = str(sys_id)
        else:
            processed["system_id"] = "human_reference"

        return processed
