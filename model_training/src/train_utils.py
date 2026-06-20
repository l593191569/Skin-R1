import logging
import os
import json
from glob import glob
import torch
from typing import Optional, Dict, Any, List
from typing import Set, Tuple

import pandas as pd

from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments, TrainerCallback
from .config import DEFAULT_CONFIG, Config
from .data import SkinConDataset, TrajectoryDataset, VisionLanguageCollator, load_dataset, _load_and_split_dataset
from .utils import ensure_dir, seed_everything
from .prompts import get_message
from .loss import evaluate_concept_accuracy, evaluate_trajectory_accuracy
import math

logger = logging.getLogger(__name__)

class ModuleCheckpointCallback(TrainerCallback):
    """Save only parameters under given prefixes at the end of each epoch."""

    def __init__(self, output_dir: str, module_prefixes: List[str], save_every_n_epochs: int = 10) -> None:
        self.output_dir = output_dir
        self.module_dir = os.path.join(output_dir, "module_checkpoint")
        os.makedirs(self.module_dir, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs

        if module_prefixes is None:
            self.module_prefixes = ()
        elif isinstance(module_prefixes, str):
            self.module_prefixes = (module_prefixes,)          # single string
        else:
            # list / iterable -> keep only non-empty strings and convert to a tuple
            self.module_prefixes = tuple(
                str(p) for p in module_prefixes if isinstance(p, str) and p
            )

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        cur_epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        # Decide whether this is the "last epoch"
        # 1) Approximately equal to args.num_train_epochs
        is_last_epoch = math.isclose(cur_epoch, float(args.num_train_epochs), rel_tol=0.0, abs_tol=1e-9)
        # 2) Fallback: if max_steps info is available, judge by step count too
        if hasattr(state, "max_steps") and hasattr(state, "global_step"):
            is_last_epoch = is_last_epoch or (state.global_step >= getattr(state, "max_steps", 0))

        # Use round to avoid 1.999 -> 1 truncation
        int_epoch = int(round(cur_epoch))  # typically 1, 2, 3, ...

        should_save = False
        if self.save_every_n_epochs:
            should_save = (int_epoch % self.save_every_n_epochs == 0)
        # Always save once more on the last epoch (even if the modulo above missed)
        if is_last_epoch:
            should_save = True

        if not should_save:
            return

        os.makedirs(self.module_dir, exist_ok=True)
        # save_path = os.path.join(self.module_dir, f"module_epoch_{int_epoch}.pt")


        # Save only parameters with the selected prefixes; if unspecified, save all (CPU state_dict)
        # if self.module_prefixes:
        #     module_state = {
        #         name: tensor.detach().cpu()
        #         for name, tensor in model.state_dict().items()
        #         if name.startswith(self.module_prefixes)
        #     }
        # else:
            # module_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        # if module_state:
        #     torch.save(module_state, save_path)
        #     logger.info("Saved module-only checkpoint: %s", save_path)

        save_path = os.path.join(self.module_dir, f"module_epoch_{int_epoch}")
        model.save_pretrained(save_path)
        if hasattr(self, 'processor') and self.processor is not None:
            self.processor.save_pretrained(save_path)
            logger.info(f"Saved full model and processor checkpoint to: {save_path}")
        else:
            logger.warning("Callback does not have a processor reference. Processor not saved.")

        



def get_torch_dtype():
    """Determine optimal torch dtype based on hardware."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def load_model_and_processor(cfg: Config, torch_dtype: torch.dtype, only_processor: bool = False):
    """Load model and processor."""
    logger.info(f"Loading base model from: {cfg.model_name_or_path}")
    if only_processor:
        return None, load_processor(cfg)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_name_or_path,
        cache_dir=cfg.cache_dir,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    processor = load_processor(cfg)
    
    return model, processor


def load_processor(cfg: Config):
    """Load processor from checkpoint or base model."""
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        checkpoint_path = os.path.dirname(cfg.checkpoint_path)
        processor_dir = os.path.join(os.path.dirname(checkpoint_path), "processor")
        if os.path.exists(processor_dir):
            logger.info(f"Loading processor from: {processor_dir}")
            return AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)
        else:
            logger.warning(f"Processor directory not found: {processor_dir}, using base model processor")
    
    logger.info(f"Loading processor from: {cfg.model_name_or_path}")
    max_pixels = getattr(cfg, "max_pixels", 448*448)
    processor = AutoProcessor.from_pretrained(
        cfg.model_name_or_path,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
        max_pixels=max_pixels,
    )
    logger.info(f"Processor pixel range: {max_pixels:,} pixels")
    return processor


def load_checkpoint(model, cfg: Config):
    """Load checkpoint if specified."""
    if not cfg.checkpoint_path or not os.path.exists(cfg.checkpoint_path):
        return
    
    logger.info(f"Loading SFT checkpoint from: {cfg.checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_state = torch.load(cfg.checkpoint_path, map_location=device)
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    
    logger.info(f"Successfully loaded SFT checkpoint with {len(checkpoint_state)} parameter groups")


def setup_model_parameters(model, cfg: Config):
    """Setup model parameters (freeze/unfreeze based on config)."""

    if cfg.dataset_source == "RL":
        return
    
    if not cfg.train_only_projector:
        logger.info("Training all model parameters")
        return
    
    train_prefixes = cfg.train_module_prefixes or ["model.visual.merger."]
    logger.info(f"Freezing all parameters except those starting with: {train_prefixes}")
    
    trainable_count = 0
    frozen_count = 0
    
    for name, param in model.named_parameters():
        is_trainable = any(name.startswith(prefix) for prefix in train_prefixes)
        param.requires_grad = is_trainable
        if is_trainable:
            logger.info(f"Training parameter: {name}")
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    
    logger.info(f"Trainable parameters: {trainable_count:,}")
    logger.info(f"Frozen parameters: {frozen_count:,}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")


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
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=0,
        load_best_model_at_end=False,
        logging_steps=cfg.logging_steps,
        logging_first_step=True if cfg.sft_test else False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["wandb"],
        run_name=cfg.model_name_or_path + "_" + cfg.task_type + "_" + cfg.dataset_source + "_" + cfg.timestamp  # wandb run name
    )


def create_callbacks(cfg: Config, output_dir: str):
    """Create training callbacks."""
    callbacks = []
    callbacks.append(ModuleCheckpointCallback(
            output_dir=output_dir,
            module_prefixes=None,
            save_every_n_epochs=cfg.save_merger_every_n_epochs
        ))
    # if cfg.sft_test:
    #     checkpoint_prefixes = cfg.train_module_prefixes
    #     callbacks.append(ModuleCheckpointCallback(
    #         output_dir=output_dir,
    #         module_prefixes=checkpoint_prefixes,
    #         save_every_n_epochs=cfg.save_merger_every_n_epochs
    #     ))
    # else:
    #     callbacks.append(ModuleCheckpointCallback(
    #         output_dir=output_dir,
    #         module_prefixes=None,
    #         save_every_n_epochs=cfg.save_merger_every_n_epochs
    #     ))
    return callbacks



def evaluate_epoch_checkpoints(eval_model, processor, cfg, output_dir, eval_dataset, eval_dataset1=None, eval_dataset23=None):
    """Evaluate model on epoch checkpoints."""
    logger.info("Starting per-epoch module evaluation on validation set")
    
    epoch_metrics = {}
    
    if cfg.dataset_source == "trajectory":
        evaluate_trajectory_checkpoints(eval_model, processor, cfg, output_dir, eval_dataset1, eval_dataset23, epoch_metrics)
    elif cfg.dataset_source == "RL":
        evaluate_regular_checkpoints(eval_model, processor, cfg, output_dir, eval_dataset, epoch_metrics)
    else:
        evaluate_regular_checkpoints(eval_model, processor, cfg, output_dir, eval_dataset, epoch_metrics)
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, "module_eval_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(epoch_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved per-epoch evaluation summary to %s", summary_path)


def evaluate_trajectory_checkpoints(eval_model, processor, cfg, output_dir, eval_dataset1, eval_dataset23, epoch_metrics):
    """Evaluate trajectory checkpoints."""
    device = next(eval_model.parameters()).device
    
    # Phase 1 checkpoints
    phase1_module_dir = os.path.join(output_dir, "phase1_type1", "module_checkpoint")
    if os.path.exists(phase1_module_dir):
        phase1_module_files = sorted(glob(os.path.join(phase1_module_dir, "module_epoch_*.pt")))
        logger.info(f"Found {len(phase1_module_files)} Phase 1 checkpoints")
        
        for module_path in phase1_module_files:
            checkpoint_name = extract_checkpoint_name(module_path, "phase1")
            load_and_evaluate_checkpoint(eval_model, module_path, device, processor, cfg, 
                                       eval_dataset1, eval_dataset23, checkpoint_name, epoch_metrics)
    
    # Phase 2 checkpoints
    phase2_module_dir = os.path.join(output_dir, "phase2_type23", "module_checkpoint")
    if os.path.exists(phase2_module_dir):
        phase2_module_files = sorted(glob(os.path.join(phase2_module_dir, "module_epoch_*.pt")))
        logger.info(f"Found {len(phase2_module_files)} Phase 2 checkpoints")
        
        for module_path in phase2_module_files:
            checkpoint_name = extract_checkpoint_name(module_path, "phase2")
            load_and_evaluate_checkpoint(eval_model, module_path, device, processor, cfg, 
                                       eval_dataset1, eval_dataset23, checkpoint_name, epoch_metrics)


def evaluate_regular_checkpoints(eval_model, processor, cfg, output_dir, eval_dataset, epoch_metrics):
    """Evaluate regular checkpoints."""
    device = next(eval_model.parameters()).device
    module_dir = os.path.join(output_dir, "module_checkpoint")
    module_files = sorted(glob(os.path.join(module_dir, "module_epoch_*.pt")))
    
    if cfg.dataset_source == "skincon":
        for module_path in module_files:
            epoch_num = extract_epoch_number(module_path)
            load_checkpoint_weights(eval_model, module_path)
            metrics = evaluate_concept_accuracy(eval_model, eval_dataset, processor, device, cfg.max_eval_samples)
            logger.info("Epoch %s concept-based metrics: %s", epoch_num, metrics)
            epoch_metrics[str(epoch_num)] = metrics


def extract_checkpoint_name(module_path: str, phase: str) -> str:
    """Extract checkpoint name from path."""
    try:
        epoch_str = os.path.splitext(os.path.basename(module_path))[0].split("_")[-1]
        epoch_num = int(epoch_str)
        return f"{phase}_epoch_{epoch_num}"
    except Exception:
        return f"{phase}_epoch_unknown"


def extract_epoch_number(module_path: str) -> int:
    """Extract epoch number from path."""
    try:
        epoch_str = os.path.splitext(os.path.basename(module_path))[0].split("_")[-1]
        return int(epoch_str)
    except Exception:
        return -1


def load_checkpoint_weights(eval_model, module_path: str):
    """Load checkpoint weights into evaluation model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module_state = torch.load(module_path, map_location=device)
    eval_model.load_state_dict(module_state, strict=False)


def load_and_evaluate_checkpoint(eval_model, module_path: str, device, processor, cfg, 
                                eval_dataset1, eval_dataset23, checkpoint_name: str, epoch_metrics: Dict):
    """Load checkpoint and evaluate on both datasets."""
    load_checkpoint_weights(eval_model, module_path)
    
    metrics_type1 = evaluate_trajectory_accuracy(eval_model, eval_dataset1, processor, device, cfg.max_eval_samples)
    metrics_type23 = evaluate_trajectory_accuracy(eval_model, eval_dataset23, processor, device, cfg.max_eval_samples)
    
    epoch_metrics[checkpoint_name] = {
        "type1_metrics": metrics_type1,
        "type23_metrics": metrics_type23
    }
    
    logger.info(f"{checkpoint_name} - Type1 metrics: {metrics_type1}")
    logger.info(f"{checkpoint_name} - Type23 metrics: {metrics_type23}")


def debug_print_samples(eval_model, eval_dataset, processor, cfg):
    """Print debug samples."""
    if not cfg.debug_print:
        return
    
    logger.info("Debug print: sampling %d examples from eval set for generation", cfg.debug_print_samples)
    sample_count = 0
    
    for batch in eval_dataset:
        if sample_count >= cfg.debug_print_samples:
            break
        
        try:
            if cfg.dataset_source == "skincon":
                prompt = "List the observed clinical concepts in a single sentence starting with 'Presence of', and separate the concepts with commas."
            elif cfg.dataset_source == "trajectory":
                prompt = "You are a medical vision-language assistant specializing in dermatology."
            elif cfg.dataset_source == "RL":
                prompt = batch["prompt"]
            else:
                prompt = "Describe this image."
            
            messages = get_message(prompt)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[batch["image"]], return_tensors="pt").to(eval_model.device)
            
            with torch.no_grad():
                outputs = eval_model(**inputs, labels=inputs.get("input_ids"))
                loss = outputs.loss.item() if hasattr(outputs, "loss") else None
                gen_ids = eval_model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
                text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            logger.info("Sample %d | loss=%s | prompt=%s | response=%s", 
                      sample_count + 1, 
                      f"{loss:.4f}" if loss is not None else "n/a", 
                      f"prompt: {prompt}", 
                      f"response: {text_out.strip()}")
        except Exception as e:
            logger.warning("Debug print failed on sample %d: %s", sample_count + 1, e)
        
        sample_count += 1

def convert_RL_dataset(dataset):    
    """Convert RL dataset to SFT dataset."""
    
    def _normalize_to_string(value):
        """Convert dict/list/str/None to a readable string for SFT text."""
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, dict):
            # Prefer common textual keys if present
            for key in ("final_diagnosis", "diagnosis", "label", "name", "text"):
                v = value.get(key)
                if isinstance(v, str):
                    return v
            import json
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            return " / ".join(_normalize_to_string(v) for v in value)
        return str(value)


    converted = []
    length = len(dataset) if hasattr(dataset, "__len__") else 0
    for idx in range(length):
        sample = dataset[idx]
        # Clone sample to avoid mutating original dataset object
        out = dict(sample) if isinstance(sample, dict) else sample

        # Extract data source and diagnosis from the sample's 'diagnosis' field
        source = sample.get("source", "")
        diagnosis_raw = sample.get("diagnosis", "") if isinstance(sample, dict) else ""
        diagnosis_text = _normalize_to_string(diagnosis_raw).strip()

        # Taxonomy via resolver
        res = resolve_path_and_bm_simple(source, diagnosis_text)
        path_list = res.get("path", []) or []
        b_or_m = res.get("b_or_m", "") or ""

        if len(path_list) == 1 and b_or_m:
            taxonomy_text = f"{diagnosis_text} is generally classified as {b_or_m}."
        elif len(path_list) == 1 and not b_or_m:
            taxonomy_text = None
        elif len(path_list) > 1 and b_or_m:
            path_str = ", ".join(map(str, path_list))
            taxonomy_text = f"The taxonomy of path of {diagnosis_text} is {path_str} and is generally classified as {b_or_m}."
        elif len(path_list) > 1 and not b_or_m:
            path_str = ", ".join(map(str, path_list))
            taxonomy_text = f"The taxonomy of path of {diagnosis_text} is {path_str}."
  

        # Assemble SFT text with required tags
        sft_text = f"<rule></rule><diagnosis>{diagnosis_text}</diagnosis><taxonomy>{taxonomy_text}</taxonomy>"
        # Attach new field
        if isinstance(out, dict):
            out["response"] = sft_text
        converted.append(out)

    return converted


def stratified_subsample_by_diagnosis(dataset, target_size: int, seed: int = 42, min_per_class: int = 1, dataset_name: str = None):
    """Return a stratified Subset of the dataset by diagnosis label.

    - Preserves class proportions approximately.
    - Ensures at least `min_per_class` samples per class when possible.
    - Works with RL-style samples where diagnosis can be str or dict with 'label'/'diagnosis'.
    """
    from torch.utils.data import Subset
    import math
    import random
    from collections import defaultdict

    total = len(dataset)
    if target_size is None or target_size >= total:
        return dataset

    def _label_from_sample(sample):
        diag = sample.get("diagnosis") if isinstance(sample, dict) else None
        if isinstance(diag, dict):
            # Prefer the finest-grained field by dataset
            if dataset_name == "derm12345" and "label" in diag:
                return str(diag.get("label"))
            if dataset_name == "dermnet" and "diagnosis" in diag:
                return str(diag.get("diagnosis"))
            # Fallback preference: diagnosis text first, then label code
            if "diagnosis" in diag:
                return str(diag.get("diagnosis"))
            if "label" in diag:
                return str(diag.get("label"))
            return str(diag)
        return str(diag)

    # collect indices per class
    cls_to_indices = defaultdict(list)
    for idx in range(total):
        try:
            sample = dataset[idx]
        except Exception:
            continue
        label = _label_from_sample(sample)
        cls_to_indices[label].append(idx)

    classes = list(cls_to_indices.keys())
    counts = {c: len(cls_to_indices[c]) for c in classes}
    rnd = random.Random(seed)

    if target_size <= len(classes):
        # pick one per class up to target deterministically
        selected = [cls_to_indices[c][0] for c in classes[:target_size] if cls_to_indices[c]]
        return Subset(dataset, selected)

    # proportional allocation with min_per_class
    frac = {c: counts[c] / total for c in classes}
    floors = {c: max(min_per_class, int(math.floor(frac[c] * target_size))) for c in classes}
    sum_floors = sum(floors.values())
    fr_part = {c: (frac[c] * target_size) - (floors[c] if floors[c] > 0 else 0) for c in classes}

    if sum_floors < target_size:
        need = target_size - sum_floors
        for c, _ in sorted(fr_part.items(), key=lambda x: x[1], reverse=True):
            if need <= 0:
                break
            floors[c] += 1
            need -= 1
    elif sum_floors > target_size:
        excess = sum_floors - target_size
        for c, _ in sorted(fr_part.items(), key=lambda x: x[1]):
            if excess <= 0:
                break
            if floors[c] > min_per_class:
                floors[c] -= 1
                excess -= 1

    selected = []
    for c in classes:
        k = min(floors[c], counts[c])
        if k <= 0:
            continue
        pool = cls_to_indices[c]
        chosen = pool if k >= len(pool) else rnd.sample(pool, k)
        selected.extend(chosen)

    # adjust length strictly to target_size
    if len(selected) > target_size:
        selected = selected[:target_size]
    elif len(selected) < target_size:
        need = target_size - len(selected)
        for c in sorted(classes, key=lambda x: counts[x], reverse=True):
            if need <= 0:
                break
            pool = [i for i in cls_to_indices[c] if i not in selected]
            take = min(need, len(pool))
            selected.extend(pool[:take])
            need -= take

    return Subset(dataset, selected)


def resolve_path_and_bm_simple(dataset: str, diag_raw: Any) -> Dict[str, Any]:
    """Simple entry: only dataset and diagnosis are required.

    Internally loads config and builds indices using the same resources
    as the reporting script, then returns only:
      { 'path': List[str], 'b_or_m': Optional[str] }
    """




    def _normalize(s: str) -> str:
        return (s or '').strip().lower()


    def _load_config_json(json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # Normalize all text content in cfg
        return _normalize_cfg_text(cfg)


    def _normalize_cfg_text(obj):
        """Recursively normalize all string values in the config"""
        if isinstance(obj, dict):
            return {k: _normalize_cfg_text(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_normalize_cfg_text(item) for item in obj]
        elif isinstance(obj, str):
            return _normalize(obj)
        else:
            return obj


    def _build_subtype_index(subtype_map: Dict[str, Any]) -> Tuple[Set[str], Dict[str, str], Set[str]]:
        """Build subtype indices:
        - family_keys: set of family names (keys of subtype_map)
        - value_to_family: map of subtype value -> its family
        - all_names: union of family_keys and all subtype values
        """
        family_keys: Set[str] = set()
        value_to_family: Dict[str, str] = {}
        all_names: Set[str] = set()
        for fam, values in (subtype_map or {}).items():
            fam_lc = _normalize(fam)
            family_keys.add(fam_lc)
            all_names.add(fam_lc)
            if isinstance(values, list):
                for v in values:
                    v_lc = _normalize(v)
                    all_names.add(v_lc)
                    value_to_family[v_lc] = fam_lc
        return family_keys, value_to_family, all_names


    def _collect_buckets(benign_prec_malig: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for bucket, items in benign_prec_malig.items():
            out[bucket] = set(_normalize(x) for x in items)
        return out


    def _build_synonym_lookup(synonym_map: Dict[str, str]) -> Dict[str, str]:
        # normalize keys and values
        return { _normalize(k): _normalize(v) for k, v in synonym_map.items() }


    def _build_shortname_lookup(short_name_map_all: Dict[str, Dict[str, str]], dataset: str) -> Dict[str, str]:
        m = short_name_map_all.get(dataset, {}) or {}
        return { _normalize(k): _normalize(v) for k, v in m.items() }


    def _taxonomy_find_parents(term: str, tree: Any, parents: Optional[List[str]] = None) -> List[List[str]]:
        """Find all parent chains for a term within a mixed dict/list taxonomy tree.
        Returns list of chains (each chain is a list of parent names).
        Matches case-insensitively on list elements or dict keys within lists.
        """
        term_lc = _normalize(term)
        if parents is None:
            parents = []
        chains: List[List[str]] = []
        if isinstance(tree, dict):
            for k, v in tree.items():
                new_parents = parents + [k]
                chains.extend(_taxonomy_find_parents(term_lc, v, new_parents))
        elif isinstance(tree, list):
            # elements may be strings or further nested dicts
            for elem in tree:
                if isinstance(elem, str):
                    if _normalize(elem) == term_lc:
                        chains.append(parents[:])
                else:
                    chains.extend(_taxonomy_find_parents(term_lc, elem, parents))
        else:
            # primitive non-container; ignore
            pass
        return chains


    def _taxonomy_build_code_to_path(tree: Any, path: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Traverse derm12345 taxonomy and build reverse map code -> full name path.
        Leaves are string codes (e.g., "acb"). Keys along the way are names.
        """
        if path is None:
            path = []
        mapping: Dict[str, List[str]] = {}
        if isinstance(tree, dict):
            for k, v in tree.items():
                mapping.update(_taxonomy_build_code_to_path(v, path + [k]))
        else:
            # leaf: code string
            if isinstance(tree, str):
                mapping[_normalize(tree)] = [p for p in path]
        return mapping


    def _expand_candidates(base_term: str, short_map: Dict[str, str], synonym_map: Dict[str, str]) -> List[str]:
        base = _normalize(base_term)
        cands: Set[str] = set()
        if base:
            cands.add(base)
            # dataset short-name first
            if base in short_map:
                cands.add(short_map[base])
            # global synonym
            # apply synonym on both original and the short map expansion
            for t in list(cands):
                if t in synonym_map:
                    cands.add(synonym_map[t])
        return list(cands)


    def _check_hits_for_text(
        term: str,
        family_keys: Set[str],
        value_to_family: Dict[str, str],
        subtype_all_names: Set[str],
        buckets: Dict[str, Set[str]],
        short_map: Dict[str, str],
        synonym_map: Dict[str, str],
        dataset_taxonomy: Dict[str, Any],
    ) -> Dict[str, Any]:
        cands = _expand_candidates(term, short_map, synonym_map)
        subtype_hit_name = None
        subtype_family = None
        # 1) direct: candidate in subtype names (family or value)
        for t in cands:
            if t in subtype_all_names:
                subtype_hit_name = t
                subtype_family = t if t in family_keys else value_to_family.get(t)
                break
        # 2) taxonomy parents → subtype
        parent_names: Set[str] = set()
        if subtype_hit_name is None and dataset_taxonomy:
            for t in cands:
                chains = _taxonomy_find_parents(t, dataset_taxonomy)
                for chain in chains:
                    for p in chain:
                        p_lc = _normalize(p)
                        parent_names.add(p_lc)
            for p in parent_names:
                if p in subtype_all_names:
                    subtype_hit_name = p
                    subtype_family = p if p in family_keys else value_to_family.get(p)
                    break
        # 3) benign/precancerous/malignant: check cands + parents
        b_or_m_hit: Optional[Tuple[str, str]] = None  # (bucket, matched_name)
        names_to_check = set(cands) | parent_names
        # Also include the subtype-hit family and name in the benign/malignant matching scope
        if subtype_family:
            names_to_check.add(subtype_family)
        if subtype_hit_name:
            names_to_check.add(subtype_hit_name)
        for name in list(names_to_check):
            if name in synonym_map:
                names_to_check.add(synonym_map[name])
        for bucket, item_set in buckets.items():
            for name in names_to_check:
                if name in item_set:
                    b_or_m_hit = (bucket, name)
                    break
            if b_or_m_hit is not None:
                break
        return {
            'candidates': sorted(cands),
            'parents': sorted(parent_names),
            'subtype_hit': subtype_hit_name,
            'subtype_family': subtype_family,
            'b_or_m_hit': b_or_m_hit,
        }


    def _check_hits_for_derm12345(
        label_code: str,
        family_keys: Set[str],
        value_to_family: Dict[str, str],
        subtype_all_names: Set[str],
        buckets: Dict[str, Set[str]],
        synonym_map: Dict[str, str],
        code_to_path: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        code = _normalize(label_code)
        path_names = code_to_path.get(code, [])
        path_lc = [_normalize(x) for x in path_names]
        subtype_hit_name = None
        subtype_family = None
        # Check along the entire path (e.g., 'dysplastic nevus', 'melanocytic nevus', ...)
        for name in path_lc:
            if name in subtype_all_names:
                subtype_hit_name = name
                subtype_family = name if name in family_keys else value_to_family.get(name)
                break
            syn = synonym_map.get(name)
            if syn and syn in subtype_all_names:
                subtype_hit_name = syn
                subtype_family = syn if syn in family_keys else value_to_family.get(syn)
                break
        # Benign/precancerous/malignant: path names + one-level synonyms
        b_or_m_hit: Optional[Tuple[str, str]] = None
        names_to_check = set(path_lc)
        # Also include the subtype-hit family and name in the benign/malignant matching scope
        if subtype_family:
            names_to_check.add(subtype_family)
        if subtype_hit_name:
            names_to_check.add(subtype_hit_name)
        for n in list(names_to_check):
            if n in synonym_map:
                names_to_check.add(synonym_map[n])
        for bucket, item_set in buckets.items():
            for name in names_to_check:
                if name in item_set:
                    b_or_m_hit = (bucket, name)
                    break
            if b_or_m_hit is not None:
                break
        return {
            'path': path_names,
            'subtype_hit': subtype_hit_name,
            'subtype_family': subtype_family,
            'b_or_m_hit': b_or_m_hit,
        }


    def resolve_hits(
        dataset: str,
        diag_raw: Any,
        family_keys: Set[str],
        value_to_family: Dict[str, str],
        subtype_all_names: Set[str],
        buckets: Dict[str, Set[str]],
        short_map: Dict[str, str],
        synonym_map: Dict[str, str],
        ds_taxonomy: Dict[str, Any],
        code_to_path: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Unified resolver for all RL datasets."""
        if isinstance(diag_raw, dict):
            if dataset == 'derm12345':
                label_code = str(diag_raw.get('label', '')).strip()
                return _check_hits_for_derm12345(
                    label_code=label_code,
                    family_keys=family_keys,
                    value_to_family=value_to_family,
                    subtype_all_names=subtype_all_names,
                    buckets=buckets,
                    synonym_map=synonym_map,
                    code_to_path=code_to_path or {},
                )
            if dataset == 'dermnet':
                term = str(diag_raw.get('diagnosis', '')).strip()
                return _check_hits_for_text(
                    term=term,
                    family_keys=family_keys,
                    value_to_family=value_to_family,
                    subtype_all_names=subtype_all_names,
                    buckets=buckets,
                    short_map=short_map,
                    synonym_map=synonym_map,
                    dataset_taxonomy=ds_taxonomy,
                )
            # Unknown dict shape; treat conservatively
            return {
                'candidates': [],
                'parents': [],
                'subtype_hit': None,
                'subtype_family': None,
                'b_or_m_hit': None,
            }
        # Plain text diagnosis
        term = str(diag_raw or '').strip()
        return _check_hits_for_text(
            term=term,
            family_keys=family_keys,
            value_to_family=value_to_family,
            subtype_all_names=subtype_all_names,
            buckets=buckets,
            short_map=short_map,
            synonym_map=synonym_map,
            dataset_taxonomy=ds_taxonomy,
        )


    def resolve_path_and_bm(
        dataset: str,
        diag_raw: Any,
        family_keys: Set[str],
        value_to_family: Dict[str, str],
        subtype_all_names: Set[str],
        buckets: Dict[str, Set[str]],
        short_map: Dict[str, str],
        synonym_map: Dict[str, str],
        ds_taxonomy: Dict[str, Any],
        code_to_path: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Return best-effort full taxonomy path and benign/precancerous/malignant for a diagnosis.

        Reuses resolve_hits and helper functions. Path semantics:
        - derm12345: use code_to_path[label] (original-cased names) if available
        - text-like datasets (e.g., derm7pt, dermnet): find a parent chain in ds_taxonomy for
        the diagnosis (or its short/synonym expansions) and append the diagnosis term at the end.
        Pick the first longest chain found; fallback to just [term] if not found.
        Returns:
        {
            'path': List[str],
            'b_or_m': Optional[str],  # one of keys from buckets, or None
            'subtype_hit': Optional[str],
            'subtype_family': Optional[str],
            'matched_name_for_b_or_m': Optional[str],
        }
        """
        hit = resolve_hits(
            dataset=dataset,
            diag_raw=diag_raw,
            family_keys=family_keys,
            value_to_family=value_to_family,
            subtype_all_names=subtype_all_names,
            buckets=buckets,
            short_map=short_map,
            synonym_map=synonym_map,
            ds_taxonomy=ds_taxonomy,
            code_to_path=code_to_path,
        )

        # Determine b_or_m bucket
        bm_tuple = hit.get('b_or_m_hit')
        b_or_m: Optional[str] = bm_tuple[0] if isinstance(bm_tuple, tuple) else None
        bm_match_name: Optional[str] = bm_tuple[1] if isinstance(bm_tuple, tuple) and len(bm_tuple) > 1 else None

        # Build full path
        path: List[str] = []

        # derm12345 uses code → path mapping
        if isinstance(diag_raw, dict) and dataset == 'derm12345':
            code = str(diag_raw.get('label', '')).strip().lower()
            if code_to_path:
                path = code_to_path.get(code, [])[:]
            # Fallback to label text itself if missing
            if not path:
                path = [str(diag_raw.get('label', '')).strip()]
            return {
                'path': path,
                'b_or_m': b_or_m,
                'subtype_hit': hit.get('subtype_hit'),
                'subtype_family': hit.get('subtype_family'),
                'matched_name_for_b_or_m': bm_match_name,
            }

        # For text-like datasets, try to locate parent chains in ds_taxonomy
        # Prepare candidate terms (reuse expansion logic)
        if isinstance(diag_raw, dict):
            # dermnet: use inner 'diagnosis'; otherwise join dict to string
            term_raw = str(diag_raw.get('diagnosis', '')) if dataset == 'dermnet' else str(diag_raw)
        else:
            term_raw = str(diag_raw or '')
        term_norm = _normalize(term_raw)

        # Build candidates: base + short map + synonyms
        cands = _expand_candidates(term_norm, short_map, synonym_map)

        best_chain: Optional[List[str]] = None
        if ds_taxonomy:
            # Try each candidate; keep the longest chain found
            for cand in cands:
                chains = _taxonomy_find_parents(cand, ds_taxonomy)
                if not chains:
                    continue
                # pick the longest chain
                local_best = max(chains, key=lambda ch: len(ch))
                if best_chain is None or len(local_best) > len(best_chain):
                    best_chain = local_best

        if best_chain:
            path = best_chain + [term_raw.strip()]
        else:
            # Fallback: only the term itself
            path = [term_raw.strip()]

        return {
            'path': path,
            'b_or_m': b_or_m,
            'subtype_hit': hit.get('subtype_hit'),
            'subtype_family': hit.get('subtype_family'),
            'matched_name_for_b_or_m': bm_match_name,
        }


    # Use the same config path as main()
    from .paths import default_data_root

    json_path = os.environ.get(
        "SKIN_R1_SYNONYM_JSON",
        os.path.join(default_data_root(), "synonym_and_subtype2.json"),
    )

    cfg = _load_config_json(json_path)
    synonym_map = _build_synonym_lookup(cfg.get('synonym_map', {}))
    subtype_map = cfg.get('subtype_map', {})
    family_keys, value_to_family, subtype_all_names = _build_subtype_index(subtype_map)
    buckets = _collect_buckets(cfg.get('benign or precancerous_in_situ or malignant', {}))
    taxonomy_all = cfg.get('taxonomy_RL_dataset', {})
    short_map_all = cfg.get('short_name_RL_dataset', {})

    short_map = _build_shortname_lookup(short_map_all, dataset)
    ds_taxonomy = taxonomy_all.get(dataset, {}) or {}

    code_to_path: Optional[Dict[str, List[str]]] = None
    if dataset == 'derm12345':
        code_to_path = _taxonomy_build_code_to_path(taxonomy_all.get('derm12345', {}))

    res = resolve_path_and_bm(
        dataset=dataset,
        diag_raw=diag_raw,
        family_keys=family_keys,
        value_to_family=value_to_family,
        subtype_all_names=subtype_all_names,
        buckets=buckets,
        short_map=short_map,
        synonym_map=synonym_map,
        ds_taxonomy=ds_taxonomy,
        code_to_path=code_to_path,
    )

    return { 'path': res.get('path', []), 'b_or_m': res.get('b_or_m') }


target_modules = [
    # Attention
    "qkv", "proj",
    # MLP
    "gate_proj", "up_proj", "down_proj",
    # Merger
    "mlp.0", "mlp.2"
]