"""Step 1 of RL dataset construction: load raw public RL datasets and filter.

Reads sparse-label dermatology datasets from ``$SKIN_R1_DATA_ROOT/RL/`` via
``data.load_dataset("RL")``, resolves taxonomy paths and benign/malignant labels
through ``RLDatasetForGRPO`` (backed by ``report_RL_diagnosis_hits``), and writes:

- ``RL_dataset_all.json``      — all loaded records (field-normalized)
- ``RL_dataset_filtered.json`` — records that pass taxonomy / b_or_m resolution

Run before ``RL_data_construct.py`` (or use ``scripts/build_rl_dataset.sh``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Union

from .data import load_dataset
from .rl_dataset_wrapper import RLDatasetForGRPO

logger = logging.getLogger(__name__)

from .paths import default_data_root

DATA_ROOT = default_data_root()


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rename GRPO wrapper fields to the schema expected by RL_data_construct."""
    record = dict(record)
    record.pop("prompt", None)
    record = {("b_or_m" if k == "taxonomy" else k): v for k, v in record.items()}
    record = {("diagnosis" if k == "final_diagnosis" else k): v for k, v in record.items()}
    return record


def save_rl_records(
    records: Union[List[Dict[str, Any]], RLDatasetForGRPO],
    file_name: str,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Serialize records to JSON and return the normalized list."""
    if isinstance(records, RLDatasetForGRPO):
        processed_records = [_normalize_record(records[i]) for i in range(len(records))]
    else:
        processed_records = [_normalize_record(r) for r in records]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d records to %s", len(processed_records), output_path)
    _print_source_split_stats(processed_records)
    return processed_records


def _print_source_split_stats(records: List[Dict[str, Any]]) -> None:
    source_counts: Dict[str, int] = {}
    for record in records:
        source = record["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in sorted(source_counts.items()):
        logger.info("Source %s: %d records", source, count)
        split_counts: Dict[str, int] = {}
        for record in records:
            if record["source"] != source:
                continue
            split = record.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1
        for split_name in ("train", "valid", "test"):
            if split_name in split_counts:
                logger.info("  %s: %d", split_name, split_counts[split_name])


def run(output_dir: str) -> str:
    """Load raw RL data, filter, and return the path to the filtered JSON file."""
    logger.info("Loading raw RL datasets from %s/RL/", output_dir)
    dataset = load_dataset("RL")
    logger.info("Loaded %d raw records", len(dataset))

    all_records = save_rl_records(dataset, "RL_dataset_all.json", output_dir)

    wrapped = RLDatasetForGRPO(all_records)
    filtered_records = save_rl_records(wrapped, "RL_dataset_filtered.json", output_dir)

    logger.info(
        "Filtering summary: raw=%d kept=%d removed=%d",
        len(all_records),
        len(filtered_records),
        len(all_records) - len(filtered_records),
    )
    return os.path.join(output_dir, "RL_dataset_filtered.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and filter raw RL training datasets.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_ROOT,
        help="Directory for RL_dataset_all.json and RL_dataset_filtered.json "
             "(default: $SKIN_R1_DATA_ROOT or model_training/data).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
