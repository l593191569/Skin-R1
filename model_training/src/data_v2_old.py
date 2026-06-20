"""SkinRationale dataset loader for SFT.

Used by `train_sft_trajectory.py`. Loads SkinRationale from
`$SKIN_R1_DATA_ROOT/trajectory_v2/`. Keeps raw {image, prompt, response, type}
fields and feeds them to `VisionLanguageCollator` in `data.py`.
"""

import logging
import os
from typing import Any, Dict, List
import pickle
import pandas as pd
from PIL import Image
import json
import random
from collections import defaultdict
from datasets import Dataset as HFDataset, concatenate_datasets

from .prompts import get_trajectory_prompt_v2, get_prompt

logger = logging.getLogger(__name__)

from .paths import default_data_root

DATA_ROOT = default_data_root()


def _open_image(path: str) -> Image.Image:
    """Open an image path as an RGB PIL image."""
    with Image.open(path) as im:
        return im.convert("RGB")


def preprocess_trajectory_v2(images_dir: str, trajectory_path: str) -> List[Dict[str, Any]]:
    """Preprocess the trajectory dataset from JSONL files and an images directory.

    Args:
        images_dir: Directory containing image files.
        trajectory_path: Path to a trajectory JSONL file (or a directory of JSONL files).

    Returns:
        A list of record dicts with at least: type, image_key, images, prompt, response.
    """
    processed_data = []

    # Handle both a single file and a directory of files
    if os.path.isfile(trajectory_path):
        jsonl_files = [trajectory_path]
    elif os.path.isdir(trajectory_path):
        jsonl_files = [os.path.join(trajectory_path, f) for f in os.listdir(trajectory_path) if f.endswith('.jsonl')]
    else:
        raise ValueError(f"trajectory_path must be a file or directory: {trajectory_path}")

    logger.info(f"Processing {len(jsonl_files)} JSONL files: {jsonl_files}")

    for jsonl_file in jsonl_files:
        logger.info(f"Processing file: {jsonl_file}")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())

                    # Extract fields
                    record_type = record.get('type', 'unknown')
                    record_id = record.get('record_id', f'line_{line_num}')

                    # Extract diagnosis / image fields depending on record_type
                    if record_type == "type1":
                        image_key = record.get('image_key', '')
                        diagnosis = record.get('diagnosis', '')
                    elif record_type == "type2" or record_type == "type4":
                        # type2/4 have two images; use the first one
                        image_key = record.get('record1_image_key', '')
                        diagnosis = record.get('dx1', '')
                        record['image_key'] = image_key  # normalize field
                        record['diagnosis'] = diagnosis  # normalize field
                    elif record_type == "type3" or record_type == "type5":
                        # type3/5 have two images; use the second one
                        image_key = record.get('record2_image_key', '')
                        diagnosis = record.get('dx2', '')
                        record['image_key'] = image_key  # normalize field
                        record['diagnosis'] = diagnosis  # normalize field

                    # 1. Fetch and validate the text content up front
                    prompt_text = get_trajectory_prompt_v2(record_type)
                    response_text = record.get('text')  # use .get() to read safely (None if missing)

                    if not image_key or not prompt_text or not response_text:
                        logger.warning(
                            f"Skipping record {record_id} at line {line_num} due to missing image_key, prompt, or response text."
                        )

                        try:
                            problematic_record_info = json.dumps(record, indent=4, ensure_ascii=False)
                            logger.warning(f"Problematic record content:\n{problematic_record_info}")
                        except Exception as e:
                            logger.error(f"Could not serialize and print the problematic record: {e}")

                        continue  # skip this invalid record

                    image_path = os.path.join(images_dir, image_key)
                    if not (os.path.exists(image_path) and os.path.getsize(image_path) > 0):
                        logger.warning(f"Image not found or empty for {image_key} in record {record_id}")
                        continue

                    record['prompt'] = prompt_text
                    record['response'] = response_text
                    record['images'] = [_open_image(image_path)]

                    processed_data.append(record)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num} in {jsonl_file}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {jsonl_file}: {e}")
                    continue

    if not processed_data:
        raise ValueError("No valid records found in trajectory files")

    return processed_data


def load_trajectory_v2(dataset_source: str) -> List[Dict[str, Any]]:
    if dataset_source in ("trajectory", "trajectory_v2"):
        preprocessed_dataset_path = os.path.join(DATA_ROOT, "trajectory_v2_preprocessed.pkl")
        if os.path.exists(preprocessed_dataset_path):
            return pd.read_pickle(preprocessed_dataset_path)
        else:
            trajectory_path = os.path.join(DATA_ROOT, "trajectory_v2")
            trajectory_images_dir = os.path.join(trajectory_path, "images")
            if os.path.exists(trajectory_path):
                records = preprocess_trajectory_v2(trajectory_images_dir, trajectory_path)
                with open(preprocessed_dataset_path, "wb") as f:
                    pickle.dump(records, f)
            return records
    raise ValueError(
        f"Unknown dataset_source '{dataset_source}'. Expected 'trajectory' "
        f"(loaded from {os.path.join(DATA_ROOT, 'trajectory_v2')})."
    )


def format_for_sft(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format a single sample into the chat structure SFTTrainer expects.

    NOTE: the SFT pipeline keeps the raw fields and uses VisionLanguageCollator
    instead, so this helper is retained only for reference/optional use.
    """
    example['messages'] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["prompt"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example["response"]},
            ],
        },
    ]
    return example


def _load_and_split_dataset_v2(dataset_source: str, val_ratio: float, split: bool, seed: int) -> Dict[str, Any]:
    """Load the local dataset and optionally split it into stratified (by type)
    train / validation sets. Caches the formatted dataset to disk to avoid
    reprocessing.
    """

    cache_path = os.path.join(DATA_ROOT, "sft_formatted_dataset_cache")

    # Check whether a cache already exists
    if os.path.exists(cache_path):
        logger.info(f"Loaded formatted dataset from cache: {cache_path}")
        formatted_dataset = HFDataset.load_from_disk(cache_path)
    else:
        logger.info("No cache found; processing data from scratch...")
        # 1. Load raw data as a list of dicts
        all_samples = load_trajectory_v2(dataset_source)

        # 2. Convert the sample list into a HuggingFace Dataset
        full_dataset = HFDataset.from_list(all_samples)

        # 3. Keep the raw fields (the SFT stage uses VisionLanguageCollator, not
        #    format_for_sft chat-template mapping).
        formatted_dataset = full_dataset

        # Save to cache
        logger.info(f"Processing complete; saving results to cache: {cache_path}")
        formatted_dataset.save_to_disk(cache_path)

    # If no split is requested, return the full formatted dataset
    if not split:
        return formatted_dataset

    # --- Stratified split logic (operates on the formatted dataset) ---

    # Group by 'type' for stratified sampling
    type_to_indices = defaultdict(list)
    for i, sample in enumerate(formatted_dataset):
        record_type = sample.get("type", "unknown")
        type_to_indices[record_type].append(i)

    train_parts = []
    val_parts = []

    # Split each type independently
    for type_name, indices in type_to_indices.items():
        if not indices:
            logger.warning(f"No data found for type {type_name}")
            continue

        random.seed(seed)
        random.shuffle(indices)

        split_idx = int(len(indices) * (1 - val_ratio))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        if train_idx:
            train_parts.append(formatted_dataset.select(train_idx))
        if val_idx:
            val_parts.append(formatted_dataset.select(val_idx))

    # Concatenate all per-type shards and shuffle the final datasets
    train_dataset = concatenate_datasets(train_parts).shuffle(seed=seed)
    val_dataset = concatenate_datasets(val_parts).shuffle(seed=seed + 1)

    logger.info(f"Final train set size: {len(train_dataset)}, validation set size: {len(val_dataset)}")

    return {"train": train_dataset, "eval": val_dataset}
