"""Build standardized evaluation JSON files consumed by ``run_eval.sh``.

Each builder writes ``$SKIN_R1_DATA_ROOT/standardized_datasets/<name>_standardized.json``.

Sources
-------
- **omnimedvqa** — OmniMedVQA (Hugging Face); prompt type 5 (reflection CoT).
- **indomain** — RL test split; prompt patched for in-domain MCQ eval.
- **indomain_b_or_m** — RL test split; benign / malignant / precancerous MCQ.
- **hierarchical** — RL test split; options from ``path_list`` (fine-grained GT).
- **ddx** — RL test split + DDx graph from ``data_construction``; differential options.

Prerequisites
-------------
- ``bash scripts/build_rl_dataset.sh`` (RL ``RL_dataset_verl_test.json``).
- ``synonym_and_subtype2.json`` under ``SKIN_R1_DATA_ROOT``.
- OmniMedVQA download for ``omnimedvqa`` (``$SKIN_R1_DATA_ROOT/OmniMedVQA/``).
- ``SKIN_R1_DDX_GRAPH`` (``ddx_graph_merged.json`` from ``data_construction``) for ``ddx``.
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from .paths import default_data_root, MODEL_TRAINING_ROOT
from .prompts import get_question_prompt

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(MODEL_TRAINING_ROOT, ".."))
ALL_DATASETS = ("omnimedvqa", "indomain", "indomain_b_or_m", "hierarchical", "ddx")

OMNIMED_QA_FILES = (
    "Fitzpatrick 17k.json",
    "ISBI2016.json",
    "ISIC2018.json",
    "ISIC2019.json",
    "ISIC2020.json",
    "Monkeypox Skin Image 2022.json",
    "PAD-UFES-20.json",
)

B_OR_M_PROMPT = """You are a medical vision-language assistant specializing in dermatology.
Given the dermatology image, answer: What type of lesion condition (benign, malignant or precancerous in situ) is present in this image?
A: benign B: malignant C: precancerous in situ
Provide necessary reasoning and only answer the question in the following format:
<thinking>Describe the key clinical features and visual observations that support your diagnosis.</thinking>
<final diagnosis>Only output one option: A, B, or C.</final diagnosis>
Ensure your response is medically accurate, concise, and strictly follows the specified format."""


def _prompt_format() -> int:
    return int(os.environ.get("SKIN_R1_RL_PROMPT_FORMAT", "4"))


def _rl_test_path(data_root: str) -> str:
    return os.path.join(
        data_root,
        f"RL_dataset_prompt_format_{_prompt_format()}",
        "RL_dataset_verl_test.json",
    )


def _standardized_dir(data_root: str) -> str:
    return os.path.join(data_root, "standardized_datasets")


def _synonym_path(data_root: str) -> str:
    return os.path.join(data_root, "synonym_and_subtype2.json")


def _save_standardized(
    output_dir: str,
    name: str,
    samples: List[Dict[str, Any]],
    description: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_standardized.json")
    payload = {
        "dataset_name": name,
        "total_samples": len(samples),
        "description": description,
        "format_version": "1.0",
        "samples": samples,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d samples → %s", len(samples), out_path)
    return out_path


def _maybe_limit(items: Sequence[Any], limit: int) -> List[Any]:
    if limit > 0:
        return list(items[:limit])
    return list(items)


def _patch_indomain_prompt(text: str) -> str:
    text = text.replace(
        "Provide only the single most likely option without reasoning",
        "Provide only the single most likely option (A/B/C/D) without reasoning",
    )
    text = text.replace(
        "Also provide the lesion condition (benign, malignant or precancerous in situ).",
        "",
    )
    return text.strip()


def _verl_image_path(item: Dict[str, Any]) -> str:
    images = item.get("image") or []
    if images and isinstance(images[0], dict):
        return str(images[0].get("image_url", "") or "")
    return ""


def _verl_prompt_text(item: Dict[str, Any]) -> str:
    prompts = item.get("prompt") or []
    if not prompts or not isinstance(prompts[0], dict):
        return ""
    return str(prompts[0].get("content", "") or "")


def _verl_answer_letter(item: Dict[str, Any]) -> str:
    for key, value in item.items():
        if not key.startswith("score_"):
            continue
        try:
            if abs(float(value) - 0.75) < 1e-8:
                return key.split("score_", 1)[-1]
        except (TypeError, ValueError):
            continue
    return ""


def _load_rl_test(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"RL test split not found: {path}\nRun: bash scripts/build_rl_dataset.sh"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _omnimed_answer_letter(item: Dict[str, Any]) -> Optional[str]:
    gt = item.get("gt_answer")
    for letter in ("A", "B", "C", "D"):
        if item.get(f"option_{letter}") == gt:
            return letter
    return None


def build_omnimedvqa(
    data_root: str,
    omni_root: Optional[str] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Build omnimedvqa from a local OmniMedVQA download."""
    root = omni_root or os.path.join(data_root, "OmniMedVQA")
    image_dir = os.path.join(root, "OmniMedVQA")
    qa_dir = os.path.join(image_dir, "QA_information", "Open-access")
    if not os.path.isdir(qa_dir):
        raise FileNotFoundError(
            f"OmniMedVQA QA directory not found: {qa_dir}\n"
            "Download: huggingface-cli download foreverbeliever/OmniMedVQA "
            f"--repo-type dataset --local-dir {root}"
        )

    samples: List[Dict[str, Any]] = []
    for qa_file in OMNIMED_QA_FILES:
        qa_path = os.path.join(qa_dir, qa_file)
        if not os.path.isfile(qa_path):
            logger.warning("Skipping missing QA file: %s", qa_path)
            continue
        with open(qa_path, encoding="utf-8") as f:
            records = json.load(f)
        for item in records:
            if item.get("question_type") != "Disease Diagnosis":
                continue
            rel_path = item.get("image_path", "")
            image_path = os.path.join(image_dir, rel_path)
            if not os.path.isfile(image_path):
                logger.debug("Image not found: %s", image_path)
                continue
            options = []
            for letter in ("A", "B", "C", "D"):
                opt = item.get(f"option_{letter}")
                if opt:
                    options.append(f"{letter}: {opt}")
            question = get_question_prompt(
                item.get("question", ""),
                " ".join(options),
                5,
            )
            answer = _omnimed_answer_letter(item)
            if not answer:
                continue
            samples.append(
                {
                    "dataset_name": "OmniMedVQA",
                    "question_id": item.get("question_id", ""),
                    "image_path": image_path,
                    "question_completed": question,
                    "answer_letter": answer,
                }
            )

    return _maybe_limit(samples, limit)


def build_indomain(raw_test: List[Dict[str, Any]], limit: int = 0) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_test):
        image_path = _verl_image_path(item)
        question = _patch_indomain_prompt(_verl_prompt_text(item))
        gt = _verl_answer_letter(item)
        if not image_path or not question or not gt:
            continue
        samples.append(
            {
                "dataset_name": "in_domain",
                "question_id": f"{item.get('source', 'unknown')}_sample_{i}",
                "image_path": image_path,
                "question_completed": question,
                "answer_letter": gt,
            }
        )
    return _maybe_limit(samples, limit)


def build_indomain_b_or_m(raw_test: List[Dict[str, Any]], limit: int = 0) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_test):
        image_path = _verl_image_path(item)
        if not image_path:
            continue
        b_or_m = item.get("b_or_m", "")
        if b_or_m == "benign":
            gt = "A"
        elif b_or_m == "malignant":
            gt = "B"
        elif b_or_m == "precancerous_in_situ":
            gt = "C"
        else:
            continue
        samples.append(
            {
                "dataset_name": "in_domain_b_or_m",
                "question_id": f"{item.get('source', 'unknown')}_sample_{i}",
                "image_path": image_path,
                "question_completed": B_OR_M_PROMPT,
                "answer_letter": gt,
            }
        )
    return _maybe_limit(samples, limit)


def build_hierarchical(
    raw_test: List[Dict[str, Any]],
    random_seed: int = 42,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    random.seed(random_seed)
    samples: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_test):
        path_list = item.get("path_list") or []
        if len(path_list) <= 1:
            continue
        groundtruth = path_list[-1]
        diseases = [d.strip() for d in path_list if d and str(d).strip()]
        if not diseases:
            continue

        image_path = _verl_image_path(item)
        if not image_path or not os.path.isfile(image_path):
            continue

        seen: set[str] = set()
        unique: List[str] = []
        for disease in diseases:
            key = disease.lower()
            if key not in seen:
                seen.add(key)
                unique.append(disease)
        if len(unique) > 26:
            unique = unique[:26]

        shuffled = list(unique)
        random.shuffle(shuffled)
        letters = [chr(ord("A") + i) for i in range(len(shuffled))]
        if groundtruth not in shuffled:
            continue
        answer = letters[shuffled.index(groundtruth)]
        options_text = " ".join(f"{l}: {o}" for l, o in zip(letters, shuffled))

        question = (
            "You are a medical vision-language assistant specializing in dermatology. "
            "Given the dermatology image, answer: What type of abnormality is present in this image? "
            f"{options_text}\n"
            "Provide necessary reasoning and only answer the question in the following format:\n"
            "<thinking>Begin by describing the characteristic clinical features and visual observations of the lesion."
            "Summarize the key diagnostic criteria or typical findings for each provided option."
            "Compare these features systematically and reason which most fine-grained option best matches the observed findings.</thinking>\n"
            "<final diagnosis>Provide only the single most likely and fine-grained option without reasoning. "
            "</final diagnosis>\n"
            "Ensure your response is medically accurate, concise, and strictly follows the specified format."
        )
        source = item.get("source") or item.get("data_source") or "unknown"
        samples.append(
            {
                "dataset_name": "Hierarchical",
                "question_id": f"{source}_sample_{idx}",
                "image_path": image_path,
                "question_completed": question,
                "answer_letter": answer,
            }
        )

    return _maybe_limit(samples, limit)


def build_ddx(
    raw_test: List[Dict[str, Any]],
    synonyms_path: str,
    ddx_graph_path: str,
    random_seed: int = 42,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    if not os.path.isfile(ddx_graph_path):
        raise FileNotFoundError(
            f"DDx graph not found: {ddx_graph_path}\n"
            "Set SKIN_R1_DDX_GRAPH to ddx_graph_merged.json from data_construction "
            "(e.g. data/outputs/<run>/ddx_graph_merged.json)."
        )

    dc_root = os.path.join(REPO_ROOT, "data_construction")
    if dc_root not in sys.path:
        sys.path.insert(0, dc_root)
    from training_sample_generator import Record, TrainingSampleGenerator  # noqa: WPS433

    random.seed(random_seed)
    generator = TrainingSampleGenerator({})
    generator.load_synonyms(synonyms_path)
    generator.load_ddxgraph(ddx_graph_path)

    diag_index: Dict[str, List[Record]] = {}
    for idx, item in enumerate(raw_test):
        path_list = item.get("path_list") or []
        if not path_list:
            continue
        groundtruth = path_list[-1]
        gt_std, mapped = generator.normalize_diagnosis(groundtruth, generator.synonyms)
        gt_std = (gt_std or "").lower().strip()
        if not gt_std:
            continue
        image_path = _verl_image_path(item)
        if not image_path:
            continue
        diag_index.setdefault(gt_std, []).append(
            Record(
                record_id=f"sample_{idx}",
                image_key=image_path,
                rule="",
                diagnosis=groundtruth,
                diagnosis_mapped=gt_std,
                mapped=mapped,
                taxonomy_text="",
            )
        )

    generator.diag_index = diag_index
    samples: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_test):
        path_list = item.get("path_list") or []
        if not path_list:
            continue
        groundtruth = path_list[-1]
        gt_std, _ = generator.normalize_diagnosis(groundtruth, generator.synonyms)
        gt_std = (gt_std or "").lower().strip()
        if not gt_std:
            continue
        image_path = _verl_image_path(item)
        if not image_path or not os.path.isfile(image_path):
            continue

        neighbors = generator.neighbors_with_records(generator.ddx_graph, gt_std, generator.diag_index)
        if not neighbors:
            continue

        options = [gt_std] + [n for n in neighbors if n != gt_std]
        seen: set[str] = set()
        unique: List[str] = []
        for opt in options:
            if opt not in seen:
                seen.add(opt)
                unique.append(opt)
        if len(unique) > 26:
            unique = unique[:26]
        if gt_std not in unique:
            continue

        shuffled = list(unique)
        random.shuffle(shuffled)
        letters = [chr(ord("A") + i) for i in range(len(shuffled))]
        answer = letters[shuffled.index(gt_std)]
        options_text = " ".join(f"{l}: {o}" for l, o in zip(letters, shuffled))

        question = (
            "You are a medical vision-language assistant specializing in dermatology. "
            "Given the dermatology image, answer: What type of abnormality is present in this image? "
            f"{options_text}\n"
            "Provide necessary reasoning and only answer the question in the following format:\n"
            "<thinking>Describe the key clinical features and visual observations that support your diagnosis.</thinking>\n"
            "<final diagnosis>Provide only the single most likely option without reasoning. "
            "</final diagnosis>\n"
            "Ensure your response is medically accurate, concise, and strictly follows the specified format."
        )
        source = item.get("source") or item.get("data_source") or "unknown"
        samples.append(
            {
                "dataset_name": "DDx",
                "question_id": f"{source}_sample_{idx}",
                "image_path": image_path,
                "question_completed": question,
                "answer_letter": answer,
            }
        )

    return _maybe_limit(samples, limit)


def organize_eval_datasets(
    data_root: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    limit: int = 0,
    omni_root: Optional[str] = None,
    ddx_graph_path: Optional[str] = None,
    random_seed: int = 42,
) -> Dict[str, str]:
    root = data_root or default_data_root()
    out_dir = _standardized_dir(root)
    chosen = list(datasets or ALL_DATASETS)
    unknown = set(chosen) - set(ALL_DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Choose from: {ALL_DATASETS}")

    outputs: Dict[str, str] = {}
    raw_test: Optional[List[Dict[str, Any]]] = None

    builders: Dict[str, Callable[[], List[Dict[str, Any]]]] = {
        "omnimedvqa": lambda: build_omnimedvqa(root, omni_root, limit),
    }

    def _get_raw_test() -> List[Dict[str, Any]]:
        nonlocal raw_test
        if raw_test is None:
            raw_test = _load_rl_test(_rl_test_path(root))
        return raw_test

    if "indomain" in chosen:
        builders["indomain"] = lambda: build_indomain(_get_raw_test(), limit)
    if "indomain_b_or_m" in chosen:
        builders["indomain_b_or_m"] = lambda: build_indomain_b_or_m(_get_raw_test(), limit)
    if "hierarchical" in chosen:
        builders["hierarchical"] = lambda: build_hierarchical(_get_raw_test(), random_seed, limit)
    if "ddx" in chosen:
        graph = ddx_graph_path or os.environ.get("SKIN_R1_DDX_GRAPH", "")
        builders["ddx"] = lambda: build_ddx(
            _get_raw_test(),
            _synonym_path(root),
            graph,
            random_seed,
            limit,
        )

    descriptions = {
        "omnimedvqa": "Standardized omnimedvqa dataset for model testing",
        "indomain": "Standardized indomain dataset (RL test split, patched prompts)",
        "indomain_b_or_m": "Standardized benign/malignant/precancerous dataset (RL test split)",
        "hierarchical": "Hierarchical test dataset from RL test split path_list",
        "ddx": "DDx test dataset from RL test split and DDx graph",
    }

    for name in chosen:
        logger.info("Building %s …", name)
        samples = builders[name]()
        outputs[name] = _save_standardized(out_dir, name, samples, descriptions[name])

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standardized evaluation JSON files for run_eval.sh."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="SKIN_R1_DATA_ROOT (default: model_training/data).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(ALL_DATASETS),
        choices=list(ALL_DATASETS),
        help="Which benchmarks to build (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples per dataset (0 = all).",
    )
    parser.add_argument(
        "--omni-root",
        default=None,
        help="OmniMedVQA download root (default: $SKIN_R1_DATA_ROOT/OmniMedVQA).",
    )
    parser.add_argument(
        "--ddx-graph",
        default=None,
        help="Path to ddx_graph_merged.json (or set SKIN_R1_DDX_GRAPH).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for hierarchical / ddx option shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    organize_eval_datasets(
        data_root=args.data_root,
        datasets=args.datasets,
        limit=args.limit,
        omni_root=args.omni_root,
        ddx_graph_path=args.ddx_graph,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
