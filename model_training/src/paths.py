"""Default paths for the model_training package."""

import os

# model_training/ (parent of src/)
MODEL_TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_data_root() -> str:
    """Training data root. Override with the SKIN_R1_DATA_ROOT environment variable."""
    default = os.path.join(MODEL_TRAINING_ROOT, "data")
    legacy = os.path.join(MODEL_TRAINING_ROOT, "dataset")
    root = os.environ.get("SKIN_R1_DATA_ROOT", default)
    if os.path.normpath(root) == os.path.normpath(legacy):
        return default
    return root
