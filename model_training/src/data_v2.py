"""RL (verl-format) dataset loader.

Used by `train_rl_grpo.py`. SkinRationale (SFT) loading lives in `data_v2_old.py`.
"""

import os
import json
from typing import Any, Dict

from .paths import default_data_root

DATA_ROOT = default_data_root()


def load_dataset_verl(dataset_source: str) -> Dict[str, Any]:
    """Load RL train/valid/test splits from the verl-format JSON files."""
    prompt_type = int(os.environ.get("SKIN_R1_RL_PROMPT_FORMAT", "4"))
    rl_dir = os.path.join(DATA_ROOT, f"RL_dataset_prompt_format_{prompt_type}")
    with open(os.path.join(rl_dir, f"{dataset_source}.json"), encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(rl_dir, "RL_dataset_verl_valid.json"), encoding="utf-8") as f:
        valid_data = json.load(f)
    with open(os.path.join(rl_dir, "RL_dataset_verl_test.json"), encoding="utf-8") as f:
        test_data = json.load(f)
    return {"train": train_data, "eval": valid_data, "test": test_data}
