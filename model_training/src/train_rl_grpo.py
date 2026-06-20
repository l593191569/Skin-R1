"""Reinforcement Learning training using GRPO for dermatology diagnosis."""

import logging
import os
import re
import sys
from datetime import datetime
from typing import Optional, List
import math
import wandb

import gc
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainerCallback
from trl import GRPOConfig, get_peft_config
from open_r1.trainer import Qwen2VLGRPOTrainer
from collections import deque
import random
from .config import Config
from .data import load_dataset, _load_and_split_dataset, VisionLanguageCollator
from .utils import ensure_dir, seed_everything, print_gpu_memory_usage
from .prompts import get_message, get_prompt
from .rl_dataset_wrapper import RLDatasetForGRPO
from .train_utils import (
    get_torch_dtype,
    load_model_and_processor, 
    load_checkpoint,
    setup_model_parameters,
    create_callbacks,
    resolve_path_and_bm_simple
)
from peft import PeftModel, LoraConfig, TaskType

logger = logging.getLogger(__name__)


class LoRACheckpointCallback_step_and_epoch(TrainerCallback):
    """Save LoRA adapter every N steps and at each epoch end."""

    def __init__(self, output_dir: str, save_every_n_steps: int = 0, save_every_n_epochs: int = 1, processor=None):
        self.output_dir = output_dir
        self.module_dir = os.path.join(output_dir, "lora_checkpoint")
        os.makedirs(self.module_dir, exist_ok=True)
        self.save_every_n_steps = int(save_every_n_steps)
        self.save_every_n_epochs = int(save_every_n_epochs)
        self.processor = processor

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.save_every_n_steps <= 0:
            return
        if state.global_step > 0 and state.global_step % self.save_every_n_steps == 0:
            save_path = os.path.join(self.module_dir, f"lora_step_{state.global_step}")
            self._save_lora(model, save_path)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        cur_epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        int_epoch = int(round(cur_epoch))
        is_last_epoch = math.isclose(cur_epoch, float(args.num_train_epochs), rel_tol=0.0, abs_tol=1e-9)

        should_save = (
            (self.save_every_n_epochs > 0 and int_epoch % self.save_every_n_epochs == 0)
            or is_last_epoch
        )
        if should_save:
            save_path = os.path.join(self.module_dir, f"lora_epoch_{int_epoch}")
            self._save_lora(model, save_path)

    def _save_lora(self, model, save_path):
        os.makedirs(save_path, exist_ok=True)
        # Save only the LoRA adapter
        try:
            model.save_pretrained(save_path, safe_serialization=True)
            if self.processor is not None:
                self.processor.save_pretrained(save_path)
            logger.info(f"Saved LoRA adapter checkpoint to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save LoRA checkpoint: {e}")



class PrintRewardsCallback(TrainerCallback):
    """Print individual reward components and totals every logging_steps for train and eval.
    
    Adapted for new reward function structure that returns multiple reward components.
    """

    def __init__(self, reward_funcs, log_every_n_steps=None):
        super().__init__()
        # Handle both single function and function list cases
        if reward_funcs is None:
            self.reward_func_name = "reward_fn"
        elif callable(reward_funcs):
            # Single function case
            self.reward_func_name = reward_funcs.__name__
        elif isinstance(reward_funcs, (list, tuple)) and len(reward_funcs) > 0:
            # Function list case
            self.reward_func_name = reward_funcs[0].__name__
        else:
            self.reward_func_name = "reward_fn"
            
        self.log_every_n_steps = log_every_n_steps
        
        # Expected reward components based on the new structure
        self.expected_components = ["total", "format", "option", "b_or_m"]

    def _fmt(self, v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        n = self.log_every_n_steps or getattr(args, "logging_steps", None)
        if n and state.global_step % int(n) != 0:
            return

        # Train rewards - look for individual components
        train_parts = []
        for component in self.expected_components:
            key = f"rewards/{self.reward_func_name}/{component}"
            if key in logs:
                train_parts.append(f"{component}={self._fmt(logs[key])}")
        
        # Also check for total reward
        if "reward" in logs:
            train_parts.append(f"total={self._fmt(logs['reward'])}")
        
        if train_parts:
            logger.info(f"[Train step {state.global_step}] " + ", ".join(train_parts))

        # Eval rewards - look for individual components
        eval_parts = []
        for component in self.expected_components:
            key = f"eval_rewards/{self.reward_func_name}/{component}"
            if key in logs:
                eval_parts.append(f"{component}={self._fmt(logs[key])}")
        
        # Also check for total eval reward
        if "eval_reward" in logs:
            eval_parts.append(f"total={self._fmt(logs['eval_reward'])}")
        
        if eval_parts:
            logger.info(f"[Eval  step {state.global_step}] " + ", ".join(eval_parts))

def length_suppress_reward(completions, **kwargs):
    """
    Length-suppression reward in [0,1].
    - Prefer using tokenizer for token count; otherwise fall back to char count.
    Kwargs:
      tokenizer: HF tokenizer (optional, recommended)
      target_len: target upper bound (int, default 128 tokens/chars)
      tau: softness (float, default 64). Larger => slower decay after target.
      use_chars: bool, if True use len(text), else use tokenizer (if provided)
      min_floor: float in [0,1], minimum reward floor to avoid all-zero variance
    """
    tokenizer = kwargs.get("tokenizer", None)
    target_len = int(kwargs.get("target_len", 256))
    tau = float(kwargs.get("tau", 64.0))
    use_chars = bool(kwargs.get("use_chars", False))
    min_floor = float(kwargs.get("min_floor", 0.7))

    # This part is correct: extract the full completion strings
    full_texts = [c[0]["content"] for c in completions]

    # --- FIX ---
    # Isolate just the assistant's response from the full completion string.
    # The response is everything that comes after the "<|im_start|>assistant\n" marker.
    assistant_marker = "<|im_start|>assistant\n"
    assistant_responses = []
    for text in full_texts:
        parts = text.split(assistant_marker)
        if len(parts) > 1:
            # If the marker is found, the response is the last part
            assistant_responses.append(parts[-1])
        else:
            # If marker not found, it's an unexpected format.
            # Append an empty string for a length of 0 to avoid errors.
            assistant_responses.append("")

    # Calculate length on the isolated assistant responses only
    if tokenizer is not None and not use_chars:
        # Use tokenizer to count tokens, which is more accurate
        lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in assistant_responses]
    else:
        # Fallback to character count if no tokenizer is provided
        lens = [len(t) for t in assistant_responses]

    # The rest of the reward calculation logic remains the same
    rewards = []
    for length in lens:
        if length <= target_len:
            reward = 1.0
        else:
            x = float(length - target_len)
            reward = math.exp(-x / tau)
        
        # Apply the minimum floor to ensure some gradient signal
        rewards.append(max(reward, min_floor))

    return rewards

def format_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for response in completions:
        # normalize
        if isinstance(response, list) and response and "content" in response[0]:
            response_str = response[0]["content"]
        else:
            response_str = str(response)

        pattern = r'^\s*<thinking>.*?</thinking>\s*<final diagnosis>.*?</final diagnosis>\s*$'
        match = re.match(pattern, response_str.strip(), re.IGNORECASE | re.DOTALL)
        rewards.append(1.0 if match else 0.0)
    return rewards

def option_reward_fn(prompts, completions, native_reward=False, **kwargs):
    rewards = []
    extra_infos = kwargs.get("extra_info", [{} for _ in range(len(completions))])

    for response, extra_info in zip(completions, extra_infos):
        # extract answer
        if isinstance(response, list) and response and "content" in response[0]:
            response_str = response[0]["content"]
        else:
            response_str = str(response)

        m = re.search(r'<final diagnosis>(.*?)</final diagnosis>', response_str, re.I | re.DOTALL)
        option = None
        if m:
            ans = m.group(1)
            opt_m = re.search(r'[:\(\s]*([A-Z])[:\)\s]*', ans, re.I)
            if opt_m:
                option = opt_m.group(1).upper()

        reward = 0.0
        if option and f"score_{option}" in extra_info:
            score = extra_info[f"score_{option}"]
            if native_reward:
                # Only grant a reward when the score equals 0.75
                reward = score if score == 0.75 else 0.0
            else:
                # Default logic: use the score directly
                reward = score
        rewards.append(reward)
    return rewards

def create_option_reward_fn_with_native_reward(native_reward):
    """Create a wrapper that passes the native_reward flag to option_reward_fn."""
    def wrapped_option_reward_fn(prompts, completions, **kwargs):
        return option_reward_fn(prompts, completions, native_reward=native_reward, **kwargs)
    wrapped_option_reward_fn.__name__ = f"option_reward_fn(native_reward={native_reward})"
    return wrapped_option_reward_fn

def b_or_m_reward_fn(prompts, completions, **kwargs):
    rewards = []
    extra_infos = kwargs.get("extra_info", [{} for _ in range(len(completions))])

    for response, extra_info in zip(completions, extra_infos):
        # extract answer
        if isinstance(response, list) and response and "content" in response[0]:
            response_str = response[0]["content"]
        else:
            response_str = str(response)

        m = re.search(r'<final diagnosis>(.*?)</final diagnosis>', response_str, re.I | re.DOTALL)
        b_or_m = None
        if m:
            ans = m.group(1)
            bm_m = re.search(r'\b(benign|precancerous|malignant)\b', ans, re.I)
            if bm_m:
                b_or_m = bm_m.group(1).lower()
                if b_or_m == "precancerous":
                    b_or_m = "precancerous_in_situ"

        reward = 0.25 if b_or_m and b_or_m == extra_info.get("b_or_m") else 0.0
        rewards.append(reward)
    return rewards

class GenerationStatusCallback(TrainerCallback):
    """
    Prints generation status for a few fixed samples during RL training.
    
    IMPROVEMENTS:
    - Switches model to eval() mode for generation to get stable output.
    - Refactored into a helper method for clarity.
    - Improved error handling and logging.
    """
    
    def __init__(self, train_dataset, processor, log_every_n_steps=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.processor = processor
        self.log_every_n_steps = log_every_n_steps
        self.tokenizer = getattr(processor, "tokenizer", None)
        self.trainer_ref = None
        
        # Sample fixed indices for consistent tracking
        num_samples = min(3, len(train_dataset))
        if len(train_dataset) > num_samples:
            self.fixed_indices = random.sample(range(len(train_dataset)), num_samples)
        else:
            self.fixed_indices = list(range(len(train_dataset)))
        logger.info(f"GenerationStatusCallback initialized with fixed indices: {self.fixed_indices}")
    
    def set_trainer(self, trainer):
        """A method to give the callback a reference to the trainer."""
        self.trainer_ref = trainer
    
    def _generate_from_sample(self, sample, step):
        """Helper function to handle generation for a single sample."""
        if not self.trainer_ref or not self.processor or not self.tokenizer:
            logger.warning("Trainer, processor, or tokenizer not set. Skipping generation.")
            return

        model = self.trainer_ref.model
        
        # --- CRITICAL FIX: Switch to eval mode for generation ---
        is_training = model.training
        model.eval()

        try:
            print_gpu_memory_usage("before generate")
            prompt_messages = sample.get("prompt", [])
            image = sample.get("image")

            # Prepare text input using the chat template
            text_input = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs for the model
            inputs = self.processor(
                text=[text_input],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            
            # Generate response with no gradients
            with torch.no_grad():
                prompt_len = inputs["input_ids"].shape[1]
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens
            new_tokens = generated_ids[0][prompt_len:]
            generated_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # --- Log the results ---
            logger.info("=" * 80)
            logger.info(f"Generation Status at Step {step} - Sample {sample.get('id', 'N/A')}")
            logger.info(f"  - Ground Truth Diagnosis: {sample.get('final_diagnosis', 'N/A')}")
            logger.info(f"  - Generated Response: {generated_response.strip()}")
            logger.info("=" * 80)
            print_gpu_memory_usage("after generate")
        except Exception as e:
            logger.error(f"Generation failed for sample: {e}", exc_info=True)
        finally:
            # --- CRITICAL FIX: Switch back to the original mode ---
            if is_training:
                model.train()

    def on_step_end(self, args, state, control, **kwargs):
        """Trigger generation at specified step intervals."""
        n = self.log_every_n_steps or args.logging_steps
        # Trigger less frequently to avoid cluttering logs, e.g., every 20 logging steps
        if not n or state.global_step == 0 or state.global_step % int(20 * n) != 0:
        # if not n or state.global_step == 0 or state.global_step % int(n) != 0:
            return
        
        for idx in self.fixed_indices:
            try:
                sample = self.train_dataset[idx]
                self._generate_from_sample(sample, state.global_step)
            except Exception as e:
                logger.error(f"Failed to process sample {idx} in callback: {e}")

def train_rl_v2_with_ref(config: Optional[Config] = None) -> None:
    """Reinforcement Learning training for dermatology diagnosis using GRPO."""
    cfg = config
    seed_everything(cfg.seed)
    
    torch_dtype = get_torch_dtype()
    logger.info(f"torch_dtype: {torch_dtype}")
    
    # TODO: necessary? Load processor (model will be loaded by Qwen2VLGRPOTrainer)
    _, processor = load_model_and_processor(cfg, torch_dtype, only_processor=True)
    logger.info("Processor loaded for RL training")
    
    # Load datasets 
    logger.info(f"Loading RL dataset: {cfg.dataset_source}")
    # For RL training, we always want split=True to get train/eval/test datasets
    from .data_v2 import load_dataset_verl
    loaded_dataset = load_dataset_verl(cfg.dataset_source)
    
    if isinstance(loaded_dataset, dict):
        train_dataset = loaded_dataset["train"]
        eval_dataset = loaded_dataset["eval"]
        test_dataset = loaded_dataset["test"]
        

    if isinstance(train_dataset, list):
        random.seed(42)
        random.shuffle(train_dataset)
        logger.info(f"Shuffled the training dataset; {len(train_dataset)} samples total")



        from .rl_dataset_wrapper import RLDatasetForGRPO_from_verl_type
        # Wrap datasets for GRPO compatibility
        train_dataset = RLDatasetForGRPO_from_verl_type(train_dataset)
        eval_dataset = RLDatasetForGRPO_from_verl_type(eval_dataset)
        test_dataset = RLDatasetForGRPO_from_verl_type(test_dataset)
        
        logger.info("✅ Wrapped datasets for GRPO compatibility")

    # Choose which version of option_reward_fn to use based on native_reward
    native_reward = getattr(cfg, 'native_reward', False)
    if native_reward:
        option_reward_fn_to_use = create_option_reward_fn_with_native_reward(native_reward=True)
        logger.info("Using reward functions: format, option (native_reward=True, only score=0.75 gives reward), b_or_m")
    else:
        option_reward_fn_to_use = option_reward_fn
        logger.info(f"Using reward functions: format, option (native_reward=False, uses score value directly), b_or_m")
    
    reward_funcs = [format_reward_fn, option_reward_fn_to_use, b_or_m_reward_fn]


    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=0.0,
        logging_steps=cfg.logging_steps,
        save_steps=500,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,   # 3% of total update steps used for warmup
        warmup_steps=0,  
        max_steps=-1,
        eval_strategy="no",  
        save_strategy="no",
        save_total_limit=0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,  # enable gradient checkpointing to save GPU memory
        dataloader_pin_memory=False,  # disable pin_memory to reduce GPU memory usage
        report_to=["wandb"],
        run_name=cfg.task_name + "_" + cfg.model_name_or_path + "_" + cfg.task_type + "_" + cfg.dataset_source + "_" + cfg.timestamp,
        remove_unused_columns=False,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        # GRPO specific parameters
        num_generations=cfg.num_generations,  # Number of generations per sample
        max_completion_length=cfg.max_seq_length,  # Maximum completion length
        temperature=1,  # Generation temperature
        gradient_accumulation_steps=int(16/cfg.per_device_train_batch_size) if int(16/cfg.per_device_train_batch_size) > 0 else 1,
    )


    ensure_dir(cfg.output_dir)

    sft_checkpoint_dir = cfg.checkpoint_path
    # Check whether a checkpoint is provided: non-empty, not None
    has_checkpoint = (
        sft_checkpoint_dir 
        and isinstance(sft_checkpoint_dir, str) 
        and sft_checkpoint_dir.strip() 

    )
    print(f"has_checkpoint: {has_checkpoint}")

    logger.info(f"Loading base model from: {cfg.model_name_or_path}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        # device_map=None,
        trust_remote_code=True
    )
    
    # Load the adapter only when a checkpoint is provided
    if has_checkpoint:
        logger.info(f"Loading SFT adapter from: {sft_checkpoint_dir}")
        sft_tuned_model = PeftModel.from_pretrained(
            base_model,
            sft_checkpoint_dir,
            is_trainable=True
        )
        logger.info(f"Successfully applied SFT adapters from: {sft_checkpoint_dir}")
    else:
        logger.info("No checkpoint provided, using base model directly (no adapter)")
        sft_tuned_model = base_model
    
    # Load ref_model only when a checkpoint is provided
    ref_model = None
    if has_checkpoint:
        logger.info(f"Loading ref model from: {sft_checkpoint_dir}")
        ref_base_model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch_dtype,
            # device_map=None,
            device_map="auto",
            trust_remote_code=True
        )

        ref_model = PeftModel.from_pretrained(
            ref_base_model,
            sft_checkpoint_dir
        )

        ref_model = ref_model.merge_and_unload() 
        logger.info(f"Successfully applied ref model from: {sft_checkpoint_dir}")
    else:
        logger.info("No checkpoint provided, skipping ref model (will not be used in trainer)")
        print(f"ref_model is None")
        # ref_model = AutoModelForVision2Seq.from_pretrained(
        #     cfg.model_name_or_path,
        #     torch_dtype=torch_dtype,
        #     device_map="auto",
        #     trust_remote_code=True
        # )


    print_gpu_memory_usage("after load model")    
    

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,  # Low-rank dimension from config
        lora_alpha=cfg.lora_alpha,  # LoRA scaling parameter from config
        lora_dropout=cfg.lora_dropout,  # LoRA dropout from config
        target_modules="all-linear",
        bias="none",
        inference_mode=False,
    )
    print_gpu_memory_usage("after build LoRA config")

    # callbacks = create_callbacks(cfg, cfg.output_dir)

    callbacks = []
    callback_save = LoRACheckpointCallback_step_and_epoch(
        output_dir=cfg.output_dir,
        save_every_n_steps=100,
        save_every_n_epochs=1,
        processor=processor,
    )
    callbacks.append(callback_save)

    callbacks.append(PrintRewardsCallback(reward_funcs=reward_funcs, log_every_n_steps=cfg.logging_steps))
    
    # Add generation status callback for RL training
    gen_callback = GenerationStatusCallback(train_dataset, processor, cfg.logging_steps)
    callbacks.append(gen_callback)
    
    logger.info(f"LoRA config: r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}")
    


    # Initialize GRPO trainer with Qwen2VL specific trainer
    logger.info("Initializing Qwen2VL GRPO trainer with LoRA...")
    trainer_kwargs = {
        "model": sft_tuned_model,
        "reward_funcs": reward_funcs,
        "args": grpo_cfg,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": processor, 
        # "peft_config": lora_config,
        "callbacks": callbacks,
        "max_pixels": 448*448,  # Match SFT training    
        "attn_implementation": "sdpa",
    }
    
    # Pass ref_model only when one is available
    if ref_model is not None:
        trainer_kwargs["ref_model"] = ref_model
        logger.info("Trainer initialized with ref_model")
    else:
        logger.info("Trainer initialized without ref_model")
    if not has_checkpoint:
        trainer_kwargs["peft_config"] = lora_config
        logger.info("Trainer initialized with LoRA config")
        print(f"trainer_kwargs['peft_config']: {trainer_kwargs['peft_config']}")
    else:
        logger.info("Trainer initialized without LoRA config")
    
    trainer = Qwen2VLGRPOTrainer(**trainer_kwargs)

    print(f"isinstance(trainer.model, PeftModel): {isinstance(trainer.model, PeftModel)}")
    print(f"isinstance(trainer.model.base_model, PeftModel): {isinstance(trainer.model.base_model, PeftModel)}")
    # print(trainer.args.beta, trainer.args.lam, trainer.args.gamma)

    # Give callback a reference to trainer for model access
    gen_callback.set_trainer(trainer)
    
    print_gpu_memory_usage("after init trainer")
    
    


    logger.info(f"Starting RL training for task_type='{cfg.task_type}' on dataset: {cfg.dataset_source}")
    print_gpu_memory_usage("before training start")
    
    # Train the model
    trainer.train()
    print_gpu_memory_usage("after training end")

    logger.info("RL training complete.")
    
    # Save the model
    trainer.save_model(cfg.output_dir)
    logger.info(f"Saved RL model to: {cfg.output_dir}")
    print_gpu_memory_usage("after save model")
    
    # Save processor
    processor_dir = os.path.join(cfg.output_dir, "processor")
    processor.save_pretrained(processor_dir)
    logger.info(f"Saved processor to: {processor_dir}")
    print_gpu_memory_usage("after save processor")
    
    logger.info("RL training and evaluation complete.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model performance on OmniMedVQA dataset")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--dataset_source", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--num_generations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--max_train_samples", type=int, required=True)
    parser.add_argument("--max_eval_samples", type=int, required=True)
    parser.add_argument("--save_merger_every_n_epochs", type=int, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--native_reward", action="store_true", help="Ablation: option_reward only grants reward when score equals 0.75")

    args = parser.parse_args()

    run = wandb.init(
    name=args.task_name + "_" + args.model_name_or_path + "_" + args.task_type + "_" + args.dataset_source + "_" + args.timestamp,  # wandb run name
    config={
        "lr": args.learning_rate, "batch_size": args.per_device_train_batch_size, "epochs": args.num_train_epochs,
        "model": args.model_name_or_path, "max_pixels": args.max_seq_length
    },
    tags=["RL","VL","A100"],
)

    train_rl_v2_with_ref(args)
