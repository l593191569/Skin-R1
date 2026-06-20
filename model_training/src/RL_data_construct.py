"""Step 2 of RL dataset construction: MCQ prompts, distractors, and hierarchical scores.

Reads ``RL_dataset_filtered.json`` (from ``load_RL_data_raw.py``), then:

1. ``sampling_options`` — sample 4-way MCQ options (correct paths + distractors)
   and assign ``score_A``…``score_D`` (wrong = 0; correct = depth-based × 0.75).
2. ``add_prompt`` — build the instruction via ``prompts.get_question_prompt`` (type 4
   by default) and ``get_message_verl``.
3. ``warp_for_verl`` — pack VERL JSON consumed by ``train_rl_grpo.py``.
4. ``split_dataset`` — per-source train/valid/test split.
5. Write ``RL_dataset_prompt_format_<N>/RL_dataset_verl_{train,valid,test}.json``.

Optionally writes fractional train subsets (``RL_dataset_verl_train_01pct.json``, …).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from .prompts import get_message_verl, get_question_prompt

logger = logging.getLogger(__name__)

from .paths import default_data_root

DATA_ROOT = default_data_root()


def warp_for_verl(rl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in rl_data:
        item["data_source"] = item["source"]
        item["image"] = [{"image_url": item["image"]}]
        item["extra_info"] = {"path_list": item["path_list"], "b_or_m": item["b_or_m"]}
        for key, value in list(item.items()):
            if key.startswith("option_") or key.startswith("score_"):
                item["extra_info"][key] = value
        item["extra_info"]["data_source"] = item["data_source"]
    return rl_data


def sampling_options(
    rl_data: List[Dict[str, Any]],
    n_options: int,
    diagnosis_set_by_source: Dict[str, set],
) -> List[Dict[str, Any]]:
    for item in rl_data:
        current_source = item["source"]
        diagnosis_set = diagnosis_set_by_source[current_source]
        path_list = item["path_list"]
        right_diagnosis_num = random.randint(1, min(n_options - 1, len(path_list)))
        wrong_diagnosis_num = n_options - right_diagnosis_num

        path_idxs = random.sample(range(len(path_list)), right_diagnosis_num)
        diagnosis_set_exclusive = diagnosis_set - set(path_list)
        wrong_diagnosis_idx = random.sample(range(len(diagnosis_set_exclusive)), wrong_diagnosis_num)
        wrong_diagnosis = [list(diagnosis_set_exclusive)[i] for i in wrong_diagnosis_idx]
        wrong_score = [0.0] * wrong_diagnosis_num
        right_diagnosis = [path_list[i] for i in path_idxs]
        right_score = [(idx + 1) / (max(path_idxs) + 1) * 0.75 for idx in path_idxs]
        diagnosis_options = right_diagnosis + wrong_diagnosis
        score = right_score + wrong_score

        shuffle_idx = random.sample(range(n_options), n_options)

        for i in range(n_options):
            item[f"option_{chr(65 + i)}"] = diagnosis_options[shuffle_idx[i]]
            item[f"score_{chr(65 + i)}"] = score[shuffle_idx[i]]
    return rl_data


def construct_diagnosis_set_by_source(rl_data: List[Dict[str, Any]]) -> Dict[str, set]:
    diagnosis_set_by_source: Dict[str, set] = defaultdict(set)
    for item in rl_data:
        diagnosis_set_by_source[item["source"]].update(item["path_list"])
    return dict(diagnosis_set_by_source)


def construct_question(item: Dict[str, Any], prompt_type: int) -> str:
    question_text = "What type of abnormality is present in this image?"
    n_options = 4
    options = []
    for i in range(n_options):
        option_key = f"option_{chr(65 + i)}"
        option_value = item.get(option_key)
        if option_value:
            options.append(f"{chr(65 + i)}: {option_value}")
    options_str = " ".join(options)
    return get_question_prompt(question_text, options_str, prompt_type)


def add_prompt(rl_data: List[Dict[str, Any]], prompt_type: int) -> List[Dict[str, Any]]:
    for item in rl_data:
        item["prompt"] = get_message_verl(construct_question(item, prompt_type))
    return rl_data


def split_dataset(rl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split by source: keep existing splits, or derive train/valid/test."""
    source_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in rl_data:
        source_data[item["source"]].append(item)

    result_data: List[Dict[str, Any]] = []

    for source, items in source_data.items():
        logger.info("Splitting source %s (%d samples)", source, len(items))

        splits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            split = (item.get("split") or "").strip() or "unknown"
            splits[split].append(item)

        if "train" in splits and "valid" in splits and "test" in splits:
            logger.info("  keeping existing train/valid/test splits")
            result_data.extend(items)

        elif "train" in splits and "test" in splits:
            train_items = splits["train"]
            test_items = splits["test"]
            random.shuffle(train_items)
            train_size = int(len(train_items) * 0.75)
            new_train = train_items[:train_size]
            new_valid = train_items[train_size:]
            for item in new_train:
                item["split"] = "train"
            for item in new_valid:
                item["split"] = "valid"
            for item in test_items:
                item["split"] = "test"
            logger.info("  train=%d valid=%d test=%d", len(new_train), len(new_valid), len(test_items))
            result_data.extend(new_train + new_valid + test_items)

        else:
            all_items: List[Dict[str, Any]] = []
            for split_items in splits.values():
                all_items.extend(split_items)
            random.shuffle(all_items)
            train_size = int(len(all_items) * 0.6)
            valid_size = int(len(all_items) * 0.2)
            new_train = all_items[:train_size]
            new_valid = all_items[train_size : train_size + valid_size]
            new_test = all_items[train_size + valid_size :]
            for item in new_train:
                item["split"] = "train"
            for item in new_valid:
                item["split"] = "valid"
            for item in new_test:
                item["split"] = "test"
            logger.info("  train=%d valid=%d test=%d", len(new_train), len(new_valid), len(new_test))
            result_data.extend(new_train + new_valid + new_test)

    return result_data


def create_subset_trainset(train_data: List[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
    source_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in train_data:
        source_data[item["source"]].append(item)

    subset_data: List[Dict[str, Any]] = []
    for source, items in source_data.items():
        sample_size = max(1, int(len(items) * ratio))
        sampled_items = random.sample(items, sample_size)
        subset_data.extend(sampled_items)
        logger.info(
            "Subset %.1f%% — source %s: %d -> %d",
            ratio * 100,
            source,
            len(items),
            len(sampled_items),
        )
    return subset_data


def statistics_dataset(rl_data: List[Dict[str, Any]]) -> None:
    counter = Counter((item["source"], item["split"]) for item in rl_data)
    for (source, split), count in sorted(counter.items()):
        logger.info("%s:%s: %d", source, split, count)


def _write_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s (%d records)", path, len(data))


def run(
    input_path: str,
    output_root: str,
    prompt_type: int = 4,
    n_options: int = 4,
    seed: int = 42,
    subset_ratios: Optional[Sequence[float]] = (0.01, 0.05, 0.1, 0.2),
) -> str:
    random.seed(seed)

    with open(input_path, encoding="utf-8") as f:
        rl_data = json.load(f)
    logger.info("Loaded %d filtered records from %s", len(rl_data), input_path)

    diagnosis_set_by_source = construct_diagnosis_set_by_source(rl_data)
    rl_data = sampling_options(rl_data, n_options, diagnosis_set_by_source)
    rl_data = add_prompt(rl_data, prompt_type)
    rl_data = warp_for_verl(rl_data)

    logger.info("Before split:")
    statistics_dataset(rl_data)

    rl_data = split_dataset(rl_data)

    logger.info("After split:")
    statistics_dataset(rl_data)

    save_dir = os.path.join(output_root, f"RL_dataset_prompt_format_{prompt_type}")
    os.makedirs(save_dir, exist_ok=True)

    train_data = [item for item in rl_data if item["split"] == "train"]
    valid_data = [item for item in rl_data if item["split"] == "valid"]
    test_data = [item for item in rl_data if item["split"] == "test"]
    logger.info(
        "Final split sizes: train=%d valid=%d test=%d",
        len(train_data),
        len(valid_data),
        len(test_data),
    )

    _write_json(os.path.join(save_dir, "RL_dataset_verl_train.json"), train_data)
    _write_json(os.path.join(save_dir, "RL_dataset_verl_valid.json"), valid_data)
    _write_json(os.path.join(save_dir, "RL_dataset_verl_test.json"), test_data)

    combined_path = os.path.join(output_root, f"RL_dataset_verl_prompt_format_{prompt_type}.json")
    _write_json(combined_path, rl_data)

    if subset_ratios:
        for ratio in subset_ratios:
            ratio_str = f"{int(ratio * 100):02d}pct"
            subset = create_subset_trainset(train_data, ratio)
            _write_json(
                os.path.join(save_dir, f"RL_dataset_verl_train_{ratio_str}.json"),
                subset,
            )

    return save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VERL-format RL GRPO datasets (prompts, options, scores)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(DATA_ROOT, "RL_dataset_filtered.json"),
        help="Filtered RL JSON from load_RL_data_raw.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_ROOT,
        help="Root directory; writes RL_dataset_prompt_format_<N>/ under here.",
    )
    parser.add_argument(
        "--prompt-type",
        type=int,
        default=int(os.environ.get("SKIN_R1_RL_PROMPT_FORMAT", "4")),
        help="Prompt template id passed to get_question_prompt (paper uses 4).",
    )
    parser.add_argument("--n-options", type=int, default=4, help="Number of MCQ options.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-subsets",
        action="store_true",
        help="Skip writing fractional train subsets (01pct, 05pct, …).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    subset_ratios = None if args.no_subsets else (0.01, 0.05, 0.1, 0.2)
    out_dir = run(
        input_path=args.input,
        output_root=args.output_dir,
        prompt_type=args.prompt_type,
        n_options=args.n_options,
        seed=args.seed,
        subset_ratios=subset_ratios,
    )
    logger.info("RL VERL datasets written under %s", out_dir)


if __name__ == "__main__":
    main()
