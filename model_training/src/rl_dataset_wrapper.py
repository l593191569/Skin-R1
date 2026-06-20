"""Wrapper for RL dataset to make it compatible with GRPO trainer."""

from typing import Dict, Any, List
from torch.utils.data import Dataset
try:
    from .prompts import get_message, get_prompt
except ImportError:
    from prompts import get_message, get_prompt

try:
    from report_RL_diagnosis_hits import resolve_path_and_bm_only
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from report_RL_diagnosis_hits import resolve_path_and_bm_only
import logging  # It's good practice to use logging

# Setup a logger
logger = logging.getLogger(__name__)

from PIL import Image


def _open_image(path: str) -> Image.Image:
    """Open an image path as RGB PIL image."""

    with Image.open(path) as im:
        return im.convert("RGB")

class RLDatasetForGRPO(Dataset):
    
    def __init__(self, rl_dataset):
        self.rl_dataset = rl_dataset
        self.valid_indices = []
        self._filter_dataset() # Call the filtering method upon initialization

    def _filter_dataset(self):
        original_size = len(self.rl_dataset)
        logger.info(f"Starting dataset filtering... Original size: {original_size}")

        for i in range(original_size):
            sample = self.rl_dataset[i]
            diagnosis = sample.get("diagnosis", "")
            source = sample.get("source", "")
            
            final_diagnosis = self._extract_final_diagnosis(diagnosis, source)

            _res = resolve_path_and_bm_only(source, final_diagnosis)
            path_list = _res.get("path", []) or []
            b_or_m = _res.get("b_or_m", "")

            if not b_or_m:
                logger.info(f"will skip sample: In {source}, {final_diagnosis} , path_list: {path_list} , b_or_m: {b_or_m}")
                continue # Skip this sample

            # if len(path_list) == 1 and path_list[0] == final_diagnosis:
            #     continue # Skip this sample
            
            # If both checks pass, the index is valid
            self.valid_indices.append(i)
            
        filtered_size = len(self.valid_indices)
        logger.info(
            f"Filtering complete. Kept {filtered_size} of {original_size} samples "
            f"({original_size - filtered_size} removed)."
        )

    def __len__(self):
        """Returns the number of _valid_ samples."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """Fetches a valid sample using the pre-computed index map."""
        # Get the original index from our list of valid indices
        original_idx = self.valid_indices[idx]
        original_sample = self.rl_dataset[original_idx]
        
        # Extract fields
        image = original_sample["image"]
        prompt_text = get_prompt("trajectory_type1_v2")
        diagnosis = original_sample.get("diagnosis", "")
        source = original_sample.get("source", "")
        split = original_sample.get("split", "")
        
        # Create GRPO-compatible format
        prompt_messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        final_diagnosis = self._extract_final_diagnosis(diagnosis, source)
        _res = resolve_path_and_bm_only(source, final_diagnosis)
        path_list = _res.get("path", []) or []
        b_or_m = _res.get("b_or_m", "")

        return {
            "prompt": prompt_messages,
            "image": image,
            "final_diagnosis": final_diagnosis,
            "taxonomy": b_or_m,
            "source": source,
            "path_list": path_list,
            "split": split,
        }
    
    def _extract_final_diagnosis(self, diagnosis, source):
        """Extract final diagnosis for reward calculation."""
        if isinstance(diagnosis, dict):
            if source == "derm12345":
                return diagnosis.get("label", "")
            elif source == "dermnet":
                return diagnosis.get("diagnosis", "")
        return str(diagnosis) if diagnosis else ""

class RLDatasetForGRPO_from_verl_type(Dataset):
    """Wrapper for VERL-type RL dataset to make it compatible with GRPO trainer."""
    
    def __init__(self, data):
        self.data = data
        self.valid_indices = []
        self._filter_dataset()
    
    def _filter_dataset(self):
        """Filter dataset to keep only valid samples."""
        original_size = len(self.data)
        logger.info(f"Starting VERL dataset filtering... Original size: {original_size}")

        for i in range(original_size):
            sample = self.data[i]
            
            # Check if required fields exist
            if not self._is_valid_sample(sample):
                continue
                
            self.valid_indices.append(i)
            
        filtered_size = len(self.valid_indices)
        logger.info(
            f"VERL filtering complete. Kept {filtered_size} of {original_size} samples "
            f"({original_size - filtered_size} removed)."
        )
    
    def _is_valid_sample(self, sample):
        """Check if sample has required fields."""
        required_fields = ["prompt", "image", "data_source", "extra_info"]
        for field in required_fields:
            if field not in sample:
                logger.info(f"Skipping sample missing field: {field}")
                return False
        
        # Check if prompt has content
        prompt = sample.get("prompt", [])
        if not prompt or not isinstance(prompt, list) or len(prompt) == 0:
            logger.info("Skipping sample with empty prompt")
            return False
            
        # Check if image has content
        image = sample.get("image", [])
        if not image or not isinstance(image, list) or len(image) == 0:
            logger.info("Skipping sample with empty image")
            return False
            
        return True
    
    def __len__(self):
        """Returns the number of valid samples."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """Fetches a valid sample and converts to GRPO-compatible format."""
        # Get the original index from our list of valid indices
        original_idx = self.valid_indices[idx]
        original_sample = self.data[original_idx]
        
        # Extract and convert prompt from VERL format to GRPO format
        prompt_messages = self._convert_prompt_format(original_sample["prompt"])
        
        # Extract image (already in correct format)
        image = original_sample["image"][0]["image_url"]
        image = _open_image(image)
        # Extract other fields
        data_source = original_sample.get("data_source", "")
        extra_info = original_sample.get("extra_info", {})
        
        # Extract diagnosis information from extra_info
        path_list = extra_info.get("path_list", [])
        b_or_m = extra_info.get("b_or_m", "")
        
        # Get final diagnosis from path_list (usually the first item)
        final_diagnosis = path_list[0] if path_list else ""
        # Derive gt option from extra_info: pick the option whose score_* equals 0.75
        gt = ""
        try:
            for k, v in extra_info.items():
                if isinstance(k, str) and k.startswith("score_"):
                    # value may be int/float/str; cast to float safely
                    try:
                        score_val = float(v)
                    except Exception:
                        continue
                    if abs(score_val - 0.75) < 1e-8:
                        gt = k.split("score_", 1)[-1]
                        break
        except Exception:
            pass
        # Store back into extra_info for downstream use
        extra_info["gt"] = gt
        
        return {
            "prompt": prompt_messages,
            "image": image,
            "final_diagnosis": final_diagnosis,
            "taxonomy": b_or_m,
            "source": data_source,
            "path_list": path_list,
            "split": "train",  # VERL datasets are typically for training
            "extra_info": extra_info,  # Keep original extra_info for reference
            "gt": gt,
        }
    
    def _convert_prompt_format(self, verl_prompt):
        """Convert VERL prompt format to GRPO-compatible format."""
        
        # VERL format: [{"role": "user", "content": "text"}]
        # GRPO format: [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}]
        
        message = verl_prompt[0]
        prompt_text = message.get("content", "")
        prompt_messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
 
        
        return prompt_messages