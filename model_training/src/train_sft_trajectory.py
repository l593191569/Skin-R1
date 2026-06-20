"""Unified supervised fine-tuning for image-text pairs."""

import logging
import os
import json
import sys
from glob import glob
import torch
from typing import Optional, Dict, Any, List

from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments, TrainerCallback
from trl import SFTTrainer
from peft import LoraConfig, TaskType
from .config import DEFAULT_CONFIG, Config
from .data import SkinConDataset, TrajectoryDataset, VisionLanguageCollator, load_dataset, _load_and_split_dataset, debug_collator
from .utils import ensure_dir, seed_everything, print_gpu_memory_usage
from .prompts import get_message
from .loss import evaluate_concept_accuracy, evaluate_trajectory_accuracy
from .train_utils import (
    get_torch_dtype,
    load_model_and_processor,
    load_checkpoint,
    setup_model_parameters,
    create_callbacks,
    evaluate_epoch_checkpoints,
    debug_print_samples,
    convert_RL_dataset,
    ModuleCheckpointCallback
)
import wandb
from .data_v2_old import _load_and_split_dataset_v2
import argparse
import logging
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


    

def train_sft(config: Optional[Config] = None) -> None:
    """Unified SFT training for image-text pairs."""
    cfg = config
    seed_everything(cfg.seed)
    
    torch_dtype = get_torch_dtype()
    logger.info(f"torch_dtype: {torch_dtype}")


    # Load datasets
    loaded_dataset = _load_and_split_dataset_v2(cfg.dataset_source, cfg.val_ratio, True, cfg.seed)

    train_dataset, eval_dataset = loaded_dataset["train"], loaded_dataset["eval"]

    # Apply subsampling if specified
    if cfg.max_train_samples:
        if len(train_dataset) > cfg.max_train_samples:
            train_dataset = train_dataset.select(range(cfg.max_train_samples))

    if cfg.max_eval_samples:
        if len(eval_dataset) > cfg.max_eval_samples:
            eval_dataset = eval_dataset.select(range(cfg.max_eval_samples))

   
    output_dir = cfg.output_dir
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    
    # Load model and processor
    model, processor = load_model_and_processor(cfg, torch_dtype)

    # Optionally apply a previously saved SFT adapter on top of the base model.
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        from peft import PeftModel
        
        try:
            model = PeftModel.from_pretrained(
                model,
                cfg.checkpoint_path,
                is_trainable=True
            )
            logger.info(f"Successfully applied SFT adapters from: {cfg.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load SFT adapters from {cfg.checkpoint_path}: {e}")
            logger.error("Exiting due to SFT model loading failure.")
            sys.exit(1)
    else:
        logger.info(f"No available SFT adapters ")

    # Create the vision-language collator (handles chat templating + label masking).
    collator = VisionLanguageCollator(processor=processor, max_length=cfg.max_seq_length)
    debug_collator(collator, train_dataset, idx=0)
    ensure_dir(output_dir)

    callbacks = create_callbacks(cfg, output_dir)
    for callback in callbacks:
        if isinstance(callback, ModuleCheckpointCallback):
            logger.info("Injecting processor into ModuleCheckpointCallback...")
            callback.processor = processor
            break # We found it, no need to keep looping

    args = create_training_arguments(cfg, output_dir)
    args.num_train_epochs = cfg.num_train_epochs
    args.max_steps = cfg.max_steps if cfg.max_steps is not None else -1
    args.remove_unused_columns = False

    from peft import LoraConfig, get_peft_model
    import torch.nn as nn

    # Enumerate the LoRA target layers on the language side and the visual merger
    # (only Linear layers under model.language_model.* / model.visual.merger.*).
    def collect_lm_linear_targets(model):
        targets = []
        for name, module in model.named_modules():
            if (name.startswith("model.language_model.") or name.startswith("model.visual.merger.")) and isinstance(module, nn.Linear):
                # Only attach LoRA to the common attention / MLP projection layers.
                if name.endswith(( "up_proj",
                    "gate_proj",
                    "mlp.0",
                    "k_proj",
                    "mlp.2",
                    "qkv",
                    "q_proj",
                    "attn.proj",
                    "v_proj",
                    "o_proj",
                    "down_proj"
                )):
                    targets.append(name)  # use the full module path to avoid matching same-named vision-side layers
        return targets

    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                callbacks=callbacks,
            )
    else:
        lm_targets = collect_lm_linear_targets(model)

        if cfg.lora_r is not None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=lm_targets,
                bias="none"
            )

        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                callbacks=callbacks,
                peft_config=peft_config,
            )

    trainer.train()
    logger.info("Training complete")


def create_training_arguments(cfg: Config, output_dir: str):
    """Create training arguments."""
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    
    logger.info(f"Using bf16: {use_bf16}")
    logger.info(f"Using fp16: {use_fp16}")
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        save_steps=cfg.save_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,   # 3% of total update steps used for warmup
        warmup_steps=0,  
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps, 
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,  
        save_strategy="no",
        save_total_limit=0,
        load_best_model_at_end=False,
        logging_steps=cfg.logging_steps,
        logging_first_step=True if cfg.sft_test else False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["wandb"],
        run_name=cfg.model_name_or_path + "_" + cfg.task_type + "_" + cfg.dataset_source + "_" + cfg.timestamp  # wandb run name
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SFT training on SkinRationale (trajectory_v2)")
    
    # Model and data paths
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Path to the base model")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("SKIN_R1_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
                       help="Cache directory for model and dataset")
    parser.add_argument("--dataset_source", type=str, default="trajectory",
                       help="Dataset source — SkinRationale under $SKIN_R1_DATA_ROOT/trajectory_v2")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for training")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=4,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=5,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save steps")
    parser.add_argument("--warmup_steps", type=int, default=300,
                       help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Evaluation steps")
    
    # Dataset parameters
    parser.add_argument("--val_ratio", type=float, default=0.05,
                       help="Validation split ratio")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Maximum evaluation samples")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Debug and testing
    parser.add_argument("--debug_print", action="store_true", default=False,
                       help="Enable debug printing")
    parser.add_argument("--sft_test", action="store_true", default=False,
                       help="Enable SFT test mode")
    
    # Save configuration
    parser.add_argument("--save_total_limit", type=int, default=10,
                       help="Maximum number of checkpoints to save")
    
    # Experiment tracking
    parser.add_argument("--timestamp", type=str, default=None,
                       help="Timestamp for the experiment")
    parser.add_argument("--task_name", type=str, default="SFT trajectory_v2",
                       help="Task name for the experiment")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                       help="Gradient accumulation steps")
    parser.add_argument("--save_model_every_n_epochs", type=int, default=10,
                       help="Save model every N epochs")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Checkpoint path")
    return parser.parse_args()


def build_config_from_args(args):
    """Build a Config object from parsed command-line arguments."""
    return Config(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        dataset_source=args.dataset_source,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        debug_print=args.debug_print,
        sft_test=args.sft_test,
        save_total_limit=args.save_total_limit,
        timestamp=args.timestamp,
        task_name=args.task_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_merger_every_n_epochs=args.save_model_every_n_epochs,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments and build the config
    args = parse_args()
    cfg = build_config_from_args(args)
    
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "Skin-R1-SFT"),
        name=f"{cfg.task_name}_{cfg.timestamp}",
        config={
            "lr": cfg.learning_rate,
            "batch_size": cfg.per_device_train_batch_size,
            "epochs": cfg.num_train_epochs,
            "model": cfg.model_name_or_path,
            "max_seq_length": cfg.max_seq_length,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
        },
        tags=["SFT", "trajectory_v2", "LoRA"]
    )
    
    try:
        # Start training
        train_sft(cfg)
    finally:
        # Ensure wandb shuts down cleanly
        wandb.finish()
