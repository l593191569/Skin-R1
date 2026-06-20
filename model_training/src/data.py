"""Dataset and data collator utilities for multimodal SFT.

This module provides:
- `SkinConDataset`: wrapper over a Hugging Face dataset with `image` and `caption` fields
- `VisionLanguageCollator`: batch collator using the model's AutoProcessor
"""

import csv
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pickle
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any
import torch
import json
from .prompts import get_trajectory_prompt, get_prompt


logger = logging.getLogger(__name__)


def _open_image(path: str) -> Image.Image:
    """Open an image path as RGB PIL image."""

    with Image.open(path) as im:
        return im.convert("RGB")
        

def preprocess_skincon(images_dir: str, annotation_path: str) -> pd.DataFrame:
    """Preprocess skincon dataset from annotation CSV and images directory.
    
    Args:
        images_dir: Directory containing image files
        annotation_path: Path to annotation CSV file
    
    Returns:
        DataFrame with columns: imagekey, image, text
    """
    import random
    from .prompts import get_prompt
    
    # Read annotation CSV
    df = pd.read_csv(annotation_path)
    
    # Get concept columns (exclude index, ImageID, and "Do not consider this image")
    concept_columns = [col for col in df.columns if col not in ['', 'ImageID', 'Do not consider this image']]
    
    processed_data = []
    
    for _, row in df.iterrows():
        image_id = row['ImageID']
        
        # Find image file (try different extensions)
        image_found = False
        image_path = os.path.join(images_dir, image_id)
        if os.path.exists(image_path):
            image_found = True
        
        if not image_found:
            logger.warning(f"Image not found for {image_id}")
            continue
        
        # Get concepts that are present (value == 1)
        present_concepts = []
        for col in concept_columns:
            if row[col] == 1:
                present_concepts.append(col)
        
        # Create SFT format: prompt + response
        prompt = get_prompt('skincon')
        
        if not present_concepts:
            logger.warning(f"No concepts found for {image_id}")
            response = "<rule>Presence of no concept.</rule>"
        else:
            # Shuffle concepts for variety
            random.shuffle(present_concepts)
            response = f"<rule>Presence of {', '.join(present_concepts)}.</rule>"
        
        # Add to processed data
        processed_data.append({
            'imagekey': image_id,
            'image': image_path,
            'prompt': prompt,
            'response': response
        })
    
    result_df = pd.DataFrame(processed_data)
    logger.info(f"Processed {len(result_df)} samples from {annotation_path}")
    return result_df

def preprocess_trajectory(images_dir: str, trajectory_path: str) -> pd.DataFrame:
    """Preprocess trajectory dataset from JSONL files and images directory.
    
    Args:
        images_dir: Directory containing image files
        trajectory_path: Path to trajectory JSONL file (or directory containing multiple JSONL files)
    
    Returns:
        DataFrame with columns: type, imagekey, image, prompt, response
    """
    import random
    
    processed_data = []
    
    # Handle both single file and directory of files
    if os.path.isfile(trajectory_path):
        jsonl_files = [trajectory_path]
    elif os.path.isdir(trajectory_path):
        jsonl_files = [os.path.join(trajectory_path, f) for f in os.listdir(trajectory_path) if f.endswith('.jsonl')]
    else:
        raise ValueError(f"trajectory_path must be a file or directory: {trajectory_path}")
    
    logger.info(f"Processing {len(jsonl_files)} JSONL files: {jsonl_files}")
    
    for jsonl_file in jsonl_files:
        logger.info(f"Processing file: {jsonl_file}")
        
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    record = json.loads(line.strip())
                    
                    # Extract fields
                    text = record.get('text', '')
                    record_type = record.get('type', 'unknown')
                    record_id = record.get('record_id', f'line_{line_num}')
                    
                    # Extract diagnosis / image fields depending on record_type
                    if record_type == "type1":
                        image_key = record.get('image_key', '')
                        diagnosis = record.get('diagnosis', '')
                    elif record_type == "type2":
                        image_key = record.get('record1_image_key', '')
                        diagnosis = record.get('dx1', '')
                    elif record_type == "type3":
                        # type3 has two images; use the first one
                        image_key = record.get('record2_image_key', '')
                        diagnosis = record.get('dx2', '')
                    else:
                        image_key = record.get('image_key', '')
                        diagnosis = record.get('diagnosis', '')  # fall back to the diagnosis field
                    
                    if not image_key:
                        logger.warning(f"Missing image_key in record {record_id} at line {line_num}")
                        continue
                    
                    # Find image file (try different extensions)
                    image_found = False
                    image_path = None
                    
                    # First try the image_key as-is (in case it already has extension)
                    test_path = os.path.join(images_dir, image_key)
                    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                        image_path = test_path
                        image_found = True
                    else:
                        # Try adding different extensions
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                            test_path = os.path.join(images_dir, image_key + ext)
                            
                            if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                                image_path = test_path
                                image_found = True
                                break
                    
                    if not image_found:
                        logger.warning(f"Image not found for {image_key} in record {record_id}")
                        continue
                    
                    # Create SFT format: prompt + response
                    prompt = get_trajectory_prompt(record_type)

                    # Use the text field as response (already in the required format)
                    response = text
                    
                    # Add to processed data
                    processed_data.append({
                        'type': record_type,
                        'imagekey': image_key,
                        'image': image_path,
                        'prompt': prompt,
                        'response': response,
                        'diagnosis': diagnosis,
                        'record_id': record_id
                    })
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num} in {jsonl_file}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {jsonl_file}: {e}")
                    continue
    
    if not processed_data:
        raise ValueError("No valid records found in trajectory files")
    
    result_df = pd.DataFrame(processed_data)
    logger.info(f"Successfully processed {len(result_df)} trajectory records")
    
    return result_df

def preprocess_RL_dataset(dataset_source: str, root_path: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if dataset_source == "BCN20000":
        train_image_dir = os.path.join(root_path, "BCN20000", "bcn_20k_train")
        train_meta_path = os.path.join(root_path, "BCN20000", "bcn_20k_train.csv")
        # test_image_dir = os.path.join(root_path, "BCN20000", "bcn_20k_test")
        # test_meta_path = os.path.join(root_path, "BCN20000", "bcn_20k_test.csv")
        train_df = pd.read_csv(train_meta_path)
        train_df["image"] = train_df["bcn_filename"].apply(lambda x: os.path.join(train_image_dir, x))
        train_df["diagnosis"] = train_df["diagnosis"]
        df = pd.concat([df, train_df[["image", "diagnosis", "split"]]], ignore_index=True)
        # test_df = pd.read_csv(test_meta_path)
        # test_df["image"] = test_df["bcn_filename"].apply(lambda x: os.path.join(test_image_dir, x))
        # test_df["diagnosis"] = test_df["diagnosis"]
        # df = pd.concat([df, test_df[["image", "diagnosis", "split"]]], ignore_index=True)
        # Remove rows where image files don't exist
        valid_indices = []
        for idx, row in df.iterrows():
            if os.path.exists(row["image"]):
                valid_indices.append(idx)
            else:
                logger.warning(f"Image not found: {row['image']}")
        
        df = df.loc[valid_indices].reset_index(drop=True)
        return df
 
    elif dataset_source == "derm7pt":
        image_dir = os.path.join(root_path, "derm7pt", "images")
        meta_path = os.path.join(root_path, "derm7pt", "meta", "meta.csv")
        indexes_dir = os.path.join(root_path, "derm7pt", "meta")
        meta_df = pd.read_csv(meta_path)

        def _read_case_set(path: str) -> set:
            if not os.path.exists(path):
                logger.warning(f"Missing split index file: {path}")
                return set()
            idx_df = pd.read_csv(path)
            if "case_num" in idx_df.columns:
                return set(idx_df["case_num"].astype(str).tolist())
            # fallback: take first column
            return set(idx_df.iloc[:, 0].astype(str).tolist())

        train_cases = _read_case_set(os.path.join(indexes_dir, "train_indexes.csv"))
        valid_cases = _read_case_set(os.path.join(indexes_dir, "valid_indexes.csv"))
        test_cases  = _read_case_set(os.path.join(indexes_dir, "test_indexes.csv"))

        def _case_to_split(case_num: Any) -> str:
            c = str(case_num)
            if c in train_cases:
                return "train"
            if c in valid_cases:
                return "valid"
            if c in test_cases:
                return "test"
            return "unknown"


        records: List[Dict[str, Any]] = []
        for _, row in meta_df.iterrows():
            diagnosis = row.get("diagnosis", "")
            case_num = row.get("case_num", "")
            split = _case_to_split(case_num)

            for view_key in ["clinic", "derm"]:
                img_name = row.get(view_key, None)
                
                img_path = os.path.join(image_dir, img_name)
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path} (case_num={case_num}, view={view_key})")
                    continue
                records.append({
                    "image": img_path,
                    "diagnosis": diagnosis,
                    "split": split,
                })

        if not records:
            logger.warning("No valid derm7pt records constructed")
            return df

        df = pd.DataFrame(records)
        return df
    
    elif dataset_source == "derm12345":
        
        train_image_dir1 = os.path.join(root_path, "derm12345", "derm12345_train_part_1")
        train_image_dir2 = os.path.join(root_path, "derm12345", "derm12345_train_part_2")
        train_meta_path  = os.path.join(root_path, "derm12345", "derm12345_metadata_train.csv")
        test_image_dir   = os.path.join(root_path, "derm12345", "derm12345_test")
        test_meta_path   = os.path.join(root_path, "derm12345", "derm12345_metadata_test.csv")

        def _resolve_image_path(label: str, image_id: str, split: str) -> Optional[str]:
            """Build image path under label/image_id by probing known roots and extensions."""
            # Train may reside in either part1 or part2
            candidate_roots = []
            if split == "train":
                candidate_roots = [train_image_dir1, train_image_dir2]
            elif split == "test":
                candidate_roots = [test_image_dir]
            else:
                candidate_roots = [train_image_dir1, train_image_dir2, test_image_dir]

            # Try as-is and common extensions
            exts = [".jpg"]
            for root in candidate_roots:
                for ext in exts:
                    candidate = os.path.join(root, str(label), f"{image_id}{ext}")
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                        return candidate
                    # else:
                    #     logger.warning(f"derm12345 image not found: {candidate}")
            return None

        def _build_records(meta_csv: str, expected_split: str) -> List[Dict[str, Any]]:
            if not os.path.exists(meta_csv):
                logger.warning(f"Missing derm12345 meta file: {meta_csv}")
                return []
            mdf = pd.read_csv(meta_csv)
            required_cols = [
                "image_id","split","super_class","malignancy",
                "main_class_1","main_class_2","sub_class","label"
            ]
            for col in required_cols:
                if col not in mdf.columns:
                    raise ValueError(f"Column {col} missing in {meta_csv}")

            recs: List[Dict[str, Any]] = []
            for _, row in mdf.iterrows():
                image_id = row["image_id"]
                split = str(row.get("split", expected_split) or expected_split)
                label = row["label"]
                img_path = _resolve_image_path(label=label, image_id=image_id, split=split)
                if img_path is None:
                    logger.warning(
                        f"derm12345 image not found: label={label}, image_id={image_id}, split={split}"
                    )
                    continue

                recs.append({
                    "image": img_path,
                    "split": split,
                    "diagnosis": {
                        "super_class": row["super_class"],
                        "malignancy": row["malignancy"],
                        "main_class_1": row["main_class_1"],
                        "main_class_2": row["main_class_2"],
                        "sub_class": row["sub_class"],
                        "label": label,
                    }
                })
            return recs

        all_records: List[Dict[str, Any]] = []
        all_records.extend(_build_records(train_meta_path, expected_split="train"))
        all_records.extend(_build_records(test_meta_path, expected_split="test"))

        if not all_records:
            logger.warning("No derm12345 records constructed")
            return df

        df = pd.DataFrame(all_records)
        return df
    
    elif dataset_source == "dermnet":
        train_image_dir = os.path.join(root_path, "dermnet", "train")
        test_image_dir = os.path.join(root_path, "dermnet", "test")

        def _scan_split(split_dir: str, split_name: str) -> List[Dict[str, Any]]:
            records: List[Dict[str, Any]] = []
            if not os.path.exists(split_dir):
                logger.warning(f"Missing dermnet split dir: {split_dir}")
                return records
            for category in sorted(os.listdir(split_dir)):
                category_dir = os.path.join(split_dir, category)
                if not os.path.isdir(category_dir):
                    continue
                for fname in os.listdir(category_dir):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    image_path = os.path.join(category_dir, fname)
                    if not os.path.exists(image_path) or os.path.getsize(image_path) <= 0:
                        logger.warning(f"dermnet image not found: {image_path}")
                        continue
                    stem = os.path.splitext(fname)[0]
                    # Enforce rule: disease-name-index.jpg → must contain a hyphen and end with numeric index
                    if "-" not in stem:
                        # skip non-conforming (e.g., 03EczemaExcoriated.jpg)
                        continue
                    name_part, last_part = stem.rsplit("-", 1)
                    if not last_part.isdigit():
                        # skip non-conforming (e.g., disease-name-suffix)
                        continue
                    diagnosis_base = name_part
                    diagnosis = diagnosis_base.replace("-", " ").strip()
                    records.append({
                        "image": image_path,
                        "diagnosis": {
                            "disease_category": category,
                            "diagnosis": diagnosis,
                        },
                        "split": split_name,
                    })
            return records

        all_records: List[Dict[str, Any]] = []
        all_records.extend(_scan_split(train_image_dir, "train"))
        all_records.extend(_scan_split(test_image_dir, "test"))
        if not all_records:
            logger.warning("No dermnet records constructed")
            return df
        df = pd.DataFrame(all_records)
        return df
    
    elif dataset_source == "HAM10000":
        train_image_dir1 = os.path.join(root_path, "HAM10000", "HAM10000_images_part_1")
        train_image_dir2 = os.path.join(root_path, "HAM10000", "HAM10000_images_part_2")
        train_meta_path  = os.path.join(root_path, "HAM10000", "HAM10000_metadata")
        test_image_dir   = os.path.join(root_path, "HAM10000", "ISIC2018_Task3_Test_Images")
        test_meta_path   = os.path.join(root_path, "HAM10000", "ISIC2018_Task3_Test_GroundTruth.csv")
       
        # Train: use HAM10000_metadata (image_id, dx)
        def _scan_split(image_dir: Any, meta_path: str, split: str):
            records: List[Dict[str, Any]] = []
            if not os.path.exists(meta_path):
                logger.warning(f"Missing HAM10000 metadata: {meta_path}")
                return records
            meta_df = pd.read_csv(meta_path)
            if "image_id" not in meta_df.columns or "dx" not in meta_df.columns:
                raise ValueError("HAM10000 metadata must contain image_id and dx columns")
            # Normalize image_dir to a list of roots
            if isinstance(image_dir, str):
                roots = [image_dir]
            elif isinstance(image_dir, (list, tuple)):
                roots = list(image_dir)
            else:
                raise TypeError(f"image_dir must be str or list/tuple of str, got: {type(image_dir)}")
            for _, row in meta_df.iterrows():
                image_id = str(row["image_id"])  # e.g., ISIC_0027419
                diagnosis = row["dx"]
                # train images are split across two folders
                img_path = None
                for root in roots:
                    candidate = os.path.join(root, f"{image_id}.jpg")
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                        img_path = candidate
                        break
                if img_path is None:
                    logger.warning(f"HAM10000 train image not found: {image_id}.jpg")
                    continue
                records.append({
                    "image": img_path,
                    "diagnosis": diagnosis,
                    "split": split,
                })
            return records
        
        all_records: List[Dict[str, Any]] = []
        all_records.extend(_scan_split([train_image_dir1, train_image_dir2], train_meta_path, "train"))
        all_records.extend(_scan_split(test_image_dir, test_meta_path, "test"))
        
        if not all_records:
            logger.warning("No HAM10000 records constructed")
            return df
        df = pd.DataFrame(all_records)
        return df

    elif dataset_source == "PAD-UFES-20":
        image_dir = os.path.join(root_path, "PAD-UFES-20", "images")
        meta_path = os.path.join(root_path, "PAD-UFES-20", "metadata.csv")
        if not os.path.exists(meta_path):
            logger.warning(f"Missing PAD-UFES-20 metadata: {meta_path}")
            return df
        meta_df = pd.read_csv(meta_path)
        if "img_id" not in meta_df.columns or "diagnostic" not in meta_df.columns:
            raise ValueError("PAD-UFES-20 metadata must contain img_id and diagnostic columns")

        # Gather candidate subfolders under images/
        candidate_dirs: List[str] = []
        if os.path.isdir(image_dir):
            for d in os.listdir(image_dir):
                subdir = os.path.join(image_dir, d)
                if os.path.isdir(subdir):
                    candidate_dirs.append(subdir)

        records: List[Dict[str, Any]] = []
        for _, row in meta_df.iterrows():
            fname = str(row["img_id"])  # already includes suffix
            diagnosis = row["diagnostic"]
            if not fname or fname == "nan":
                continue

            found_path: Optional[str] = None
            for root in candidate_dirs:
                candidate = os.path.join(root, fname)
                if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                    found_path = candidate
                    break
            if found_path is None:
                logger.warning(f"PAD-UFES-20 image not found: {fname}")
                continue

            records.append({
                "image": found_path,
                "diagnosis": diagnosis,
                "split": "",
            })

        if not records:
            logger.warning("No PAD-UFES-20 records constructed")
            return df
        df = pd.DataFrame(records)
        return df
        

def load_dataset(dataset_source: str) -> List[Dict[str, Any]]:

    from .paths import default_data_root

    data_root = default_data_root() + os.sep
    
    if dataset_source == "skincon":
        preprocessed_dataset_path = data_root + "skincon_preprocessed.pkl"
    elif dataset_source == "trajectory":
        preprocessed_dataset_path = data_root + "trajectory_preprocessed.pkl"
    elif dataset_source == "RL":
        preprocessed_dataset_path = data_root + "RL_preprocessed.pkl"
    else:
        raise ValueError(f"Unknown dataset source: {dataset_source}")
    
    if os.path.exists(preprocessed_dataset_path): 
        return pd.read_pickle(preprocessed_dataset_path)
    
    if dataset_source == "skincon":
        skincon_path = data_root + "skincon"
        ddi_path = skincon_path + "/DDI"
        fitzpatrick17k_path = skincon_path + "/fitzpatrick17k"
        
        preprocessed_datasets = []
        
        if os.path.exists(ddi_path):
            ddi_images_dir = ddi_path + "/images"
            ddi_annotation_path = ddi_path + "/annotation.csv"
            ddi_df = preprocess_skincon(ddi_images_dir, ddi_annotation_path)
            preprocessed_datasets.append(ddi_df)
            
        if os.path.exists(fitzpatrick17k_path):
            fitzpatrick17k_images_dir = fitzpatrick17k_path + "/images"
            fitzpatrick17k_annotation_path = fitzpatrick17k_path + "/annotation.csv"
            fitzpatrick17k_df = preprocess_skincon(fitzpatrick17k_images_dir, fitzpatrick17k_annotation_path)
            preprocessed_datasets.append(fitzpatrick17k_df)

        if preprocessed_datasets:
            preprocessed_dataset = pd.concat(preprocessed_datasets, ignore_index=True)
            records = preprocessed_dataset.to_dict('records')
            with open(preprocessed_dataset_path, "wb") as f:
                pickle.dump(records, f)
            return records
        else:
            raise ValueError("No valid datasets found")
            
    elif dataset_source == "trajectory":
        trajectory_path = data_root + "trajectory"
        trajectory_images_dir = trajectory_path + "/images"
        if os.path.exists(trajectory_path):
            preprocessed_dataset = preprocess_trajectory(trajectory_images_dir, trajectory_path)
            records = preprocessed_dataset.to_dict('records')
            with open(preprocessed_dataset_path, "wb") as f:
                pickle.dump(records, f)
        return records
    
    elif dataset_source == "RL":
        RL_dataset_names = ["BCN20000", "derm7pt", "derm12345", "dermnet", "HAM10000", "PAD-UFES-20"]
        RL_dataset_dir = data_root + "RL/"
        RL_preprocessed_datasets = []
        for RL_dataset_name in RL_dataset_names:
            df = preprocess_RL_dataset(RL_dataset_name, RL_dataset_dir)
            if not df.empty:
                df["source"] = RL_dataset_name
                # Use RL prompt for RL datasets
                df["prompt"] = get_prompt("rl")
                RL_preprocessed_datasets.append(df)
        RL_preprocessed_dataset = pd.concat(RL_preprocessed_datasets, ignore_index=True)
        records = RL_preprocessed_dataset.to_dict('records')
        with open(preprocessed_dataset_path, "wb") as f:
            pickle.dump(records, f)
        return records

def _load_and_split_dataset(dataset_source: str, val_ratio: float, split: bool, seed: int) -> Dict[str, Any]:
    """Load local dataset and split into train/val(if skincon)."""
    import random
    from torch.utils.data import Subset
    
    # Load all samples
    all_samples = load_dataset(dataset_source)
    
    if dataset_source == "skincon":    
        if split:
            # Set random seed for reproducible split
            random.seed(seed)
            
            # Create indices and shuffle
            indices = list(range(len(all_samples)))
            random.shuffle(indices)
            
            # Calculate split point
            split_idx = int(len(indices) * (1 - val_ratio))
            
            # Split indices
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # Create single dataset and apply splits using Subset
            full_dataset = SkinConDataset(dataset_source)
            train_subset = Subset(full_dataset, train_indices)
            val_subset = Subset(full_dataset, val_indices)
            
            return {
                "train": train_subset,
                "eval": val_subset,
            }
        else:
            return SkinConDataset(dataset_source)
                
    elif dataset_source == "trajectory":    

        if split:
            # Load trajectory dataset
            trajectory_dataset = TrajectoryDataset(dataset_source)
            
            # Group by record type
            type1_data = []
            type2_data = []
            type3_data = []
            
            for i, sample in enumerate(trajectory_dataset):
                record_type = sample.get('type', 'unknown')
                if record_type == 'type1':
                    type1_data.append(i)
                elif record_type == 'type2':
                    type2_data.append(i)
                elif record_type == 'type3':
                    type3_data.append(i)
            
            # Split each type separately
            result = {}
            
            for type_name, type_indices in [('type1', type1_data), ('type2', type2_data), ('type3', type3_data)]:
                if not type_indices:
                    logger.warning(f"No data found for {type_name}")
                    continue
                    
                # Set random seed for reproducible split
                random.seed(seed)
                random.shuffle(type_indices)
                
                # Calculate split point
                split_idx = int(len(type_indices) * (1 - val_ratio))
                
                # Split indices
                train_indices = type_indices[:split_idx]
                val_indices = type_indices[split_idx:]
                
                # Create subsets
                train_subset = Subset(trajectory_dataset, train_indices)
                val_subset = Subset(trajectory_dataset, val_indices)
                
                result[f"{type_name}_train"] = train_subset
                result[f"{type_name}_eval"] = val_subset
                
                logger.info(f"{type_name}: train {len(train_indices)} / eval {len(val_indices)} samples")
            
            return result
        else:
            return TrajectoryDataset(dataset_source)
        
    elif dataset_source == "RL":

        if split:
            RL_dataset = RLDataset(dataset_source)
            

            # Find the indices of all derm7pt samples
            derm7pt_indices = [i for i, s in enumerate(RL_dataset.samples) if s.get("source") == "derm7pt"]

            # Collect split assignments for the derm7pt samples
            train_indices = []
            eval_indices = []
            test_indices = []
            
            for idx in derm7pt_indices:
                sample = RL_dataset.samples[idx]
                split_value = sample.get("split", "")
                if split_value == "train":
                    train_indices.append(idx)
                elif split_value == "valid":
                    eval_indices.append(idx)
                elif split_value == "test":
                    test_indices.append(idx)
                else:
                    # Default to train if split is unknown
                    train_indices.append(idx)

            train_dataset = Subset(RL_dataset, train_indices)
            eval_dataset = Subset(RL_dataset, eval_indices)
            test_dataset = Subset(RL_dataset, test_indices)
          
            
            logger.info(f"derm7pt train: {len(train_dataset)}, eval: {len(eval_dataset)}, test: {len(test_dataset)}")
            
            return {
                "train": train_dataset,
                "eval": eval_dataset,
                "test": test_dataset,
            }
            
        else:
            return RLDataset(dataset_source)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_source}")


class SkinConDataset(Dataset):
    """Dataset wrapper for the local `skincon` image-caption dataset.

    Loads data from local CSV files and image directories.
    """

    def __init__(self, dataset_source: str = "skincon") -> None:
        super().__init__()
        self.samples = load_dataset(dataset_source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image = _open_image(item["image"])
        prompt = item["prompt"]
        response = item["response"]
        return {"image": image, "prompt": prompt, "response": response}

class TrajectoryDataset(Dataset):
    """Dataset wrapper for the local `trajectory` image-caption dataset.

    Loads data from local JSON files and image directories.
    """

    def __init__(self, dataset_source: str = "trajectory") -> None:
        super().__init__()
        self.samples = load_dataset(dataset_source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        type = item["type"]
        image = _open_image(item["image"])
        prompt = item["prompt"]
        response = item["response"]
        diagnosis = item.get("diagnosis", "")      
        return {"type": type, "image": image, "prompt": prompt, "response": response, "diagnosis": diagnosis}

class RLDataset(Dataset):
    """Dataset wrapper for the local `RL` image-caption dataset.

    Loads data from local JSON files and image directories.
    """

    def __init__(self, dataset_source: str = "RL") -> None:
        super().__init__()
        self.samples = load_dataset(dataset_source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image = _open_image(item["image"])
        prompt = item["prompt"]
        split = item.get("split", "")
        diagnosis = item.get("diagnosis", "") 
        source = item.get("source", "")     
        return {"image": image, "prompt": prompt, "diagnosis": diagnosis, "split": split, "source": source}


def _find_last_sublist(main_list: list, sub_list: list) -> Optional[int]:
    """
    Find the last occurrence of a sub-list within the main list and return its
    starting index. Returns None if not found.
    """
    if not sub_list:
        return len(main_list)
    sub_len = len(sub_list)
    for i in range(len(main_list) - sub_len, -1, -1):
        if main_list[i:i + sub_len] == sub_list:
            return i
    return None

@dataclass
class VisionLanguageCollator:
    """
    Data collator for instruction-tuning the Qwen2-VL family of models.

    The collator implements the following core logic:
    1.  Convert the message list into formatted strings via
        tokenizer.apply_chat_template.
    2.  Encode the formatted strings and images into model inputs with the
        processor.
    3.  Generate `labels` according to the Qwen2-VL chat-template format.
    4.  In `labels`, set the tokens of the user prompt and padding to -100 so
        that loss is only computed on the assistant's response.
    """
    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None)
        assert self.tokenizer is not None, "tokenizer not found in processor"
        self.max_length = max_length

        self.assistant_prompt_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(self.im_end_id, int):
             self.im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]


    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in batch]
        prompts = [ex["prompt"] for ex in batch]
        replies = [ex["response"] for ex in batch]

        # 1) Build the message list
        messages_list = []
        for p, r in zip(prompts, replies):
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]},
                {"role": "assistant", "content": [{"type": "text", "text": r}]},
            ]
            messages_list.append(messages)

        # 2) Apply the chat template first to turn messages into formatted strings.
        #    The placeholders (e.g. <|image|>) are recognized by the processor below.
        formatted_texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in messages_list
        ]
        
        # 3) Encode text and images together with the processor (text is a list of strings)
        model_inputs = self.processor(
            text=formatted_texts,  # pass the list of formatted strings
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=None,
        )

        # 4) Build labels from the final input_ids
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = input_ids.clone()
        
        labels[attention_mask == 0] = -100

        for i, sample_ids in enumerate(input_ids.tolist()):
            seq_len = int(attention_mask[i].sum())
            
            assistant_prompt_start_idx = _find_last_sublist(
                sample_ids[:seq_len], self.assistant_prompt_ids
            )
            
            if assistant_prompt_start_idx is not None:
                mask_end_idx = assistant_prompt_start_idx + len(self.assistant_prompt_ids)
                labels[i, :mask_end_idx] = -100
                
                if sample_ids[seq_len - 1] == self.im_end_id:
                    labels[i, seq_len - 1] = -100
            else:
                print(
                    f"Warning: assistant start marker not found in sample {i}. "
                    "The labels for this sample may be incorrect."
                )

        model_inputs["labels"] = labels
        return model_inputs

def debug_collator(collator: VisionLanguageCollator, dataset: List[Dict], idx: int = 0):
    """
    Print one collated sample to inspect the alignment of input_ids and labels.
    """
    print(f"\n--- Debugging sample {idx} of the dataset ---")
    
    # Collate a single sample (wrapped in a list to simulate a batch)
    batch = collator([dataset[idx]])
    tokenizer = collator.processor.tokenizer

    # Extract the data of the first (and only) sample
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    
    # Decode for readability
    full_decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    # Decode the part the model is supervised on from labels
    label_ids_to_decode = [label_id for label_id in labels if label_id != -100]
    supervised_decoded_text = tokenizer.decode(label_ids_to_decode, skip_special_tokens=False)

    print("="*80)
    print("[Full input decode] (everything the model sees):")
    print(full_decoded_text)
    print("="*80)
    print("[Supervised label decode] (what the model must learn to predict):")
    print(supervised_decoded_text)
    # print("="*80)
    # print("[Per-token alignment details]:")
    # print(f"{'Index':<5} | {'Token ID':<8} | {'Token String':<20} | {'Label':<10}")
    # print("-"*80)
    
    # tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # for i, (tid, tok_str, lab) in enumerate(zip(input_ids, tokens, labels)):
    #     label_str = str(lab) if lab != -100 else "MASKED"
    #     print(f"{i:<5} | {tid:<8} | {tok_str:<20} | {label_str:<10}")
    # print("="*80)