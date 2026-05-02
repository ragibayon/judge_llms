from __future__ import annotations
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "KnutJaegersberg/summeval_pairs"


def main() -> None:
    dataset = load_dataset(DATASET_NAME)

    output_dir = (
        Path(__file__).resolve().parent
        / "datasets"
        / DATASET_NAME.replace("/", "__")
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_dir))
    print(f"Saved dataset '{DATASET_NAME}' to {output_dir}")


if __name__ == "__main__":
    main()
