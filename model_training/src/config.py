import os
from dataclasses import dataclass, field
from typing import Optional, List

# Default HuggingFace cache directory. Override with SKIN_R1_CACHE_DIR.
_DEFAULT_CACHE_DIR = os.environ.get(
    "SKIN_R1_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
)


@dataclass
class Config:
    """Dataclass containing configuration for training and evaluation.

    Attributes:

    """

    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_dir: Optional[str] = _DEFAULT_CACHE_DIR
    checkpoint_path: Optional[str] = None  # Direct path to specific checkpoint file (e.g., module_epoch_6.pt)
    output_dir: str = ""

    evaluate_dataset_path: Optional[str] = None  # optional CSV for evaluation
    task_type: str = ""
    dataset_source: Optional[str] = None
    timestamp: Optional[str] = None
    task_name: Optional[str] = None
    
    num_train_epochs: int = 80
    per_device_train_batch_size: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500  # interval (in steps) between evaluations
    warmup_steps: int = 300
    max_steps: Optional[int] = None

    # Tokenization / generation
    max_seq_length: int = 1024
    max_new_tokens: int = 1024
    num_generations: int = 2


    # Reproducibility
    seed: int = 42
    
    # Data split for concept training (train/val split ratio)
    val_ratio: float = 0.2

    # Optional dataset subsampling for quick tests
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Debug printing of generations
    debug_print: bool = False
    debug_print_samples: int = 3
    
    # Training strategy
    train_only_projector: bool = False  # Only train selected module(s)
    train_module_prefixes: List[str] = field(default_factory=lambda: None)  # List of parameter prefixes to train (if None, defaults to ["model.visual.merger."])
    sft_test: bool = False
    gradient_accumulation_steps: int = 1


    # Save total limit
    save_total_limit: int = 10
    save_merger_every_n_epochs: int = 100
    save_model_every_n_epochs: int = 100
    
    # RL training specific
    # reward_funcs: List[str] = field(default_factory=lambda: ["format", "accuracy"])  # List of reward functions: ["format", "accuracy"]
    
    # LoRA configuration
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.1  # LoRA dropout

DEFAULT_CONFIG = Config()


