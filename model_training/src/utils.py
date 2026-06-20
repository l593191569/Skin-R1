"""Utility helpers for logging, seeding, metrics, and device selection."""

import json
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = logging.INFO) -> None:
    """Configure root logger with a simple format.

    Args:
        verbosity: Logging level, defaults to INFO.
    """

    logging.basicConfig(
        level=verbosity,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_device() -> torch.device:
    """Return the best available device (CUDA, MPS, or CPU)."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed random number generators for reproducibility.

    Args:
        seed: The seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""

    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Save a dictionary as JSON to the specified path."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def print_gpu_memory_usage(stage: str = "", reset_stats: bool = False):
    """Print current GPU memory usage for all available GPUs.
    
    Args:
        stage: Stage name for logging
        reset_stats: Whether to reset peak memory statistics after printing
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"GPU Memory [{stage}]: {num_gpus} GPU(s) available")
        
        for gpu_id in range(num_gpus):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
            free_memory = total_memory - reserved
            
            logger.info(f"  GPU {gpu_id}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                       f"Free={free_memory:.2f}GB, Max={max_allocated:.2f}GB, Total={total_memory:.2f}GB")
        
        if reset_stats:
            torch.cuda.reset_peak_memory_stats()
            logger.info("  Reset peak memory statistics")
    else:
        logger.info(f"GPU Memory [{stage}]: No GPU available")

