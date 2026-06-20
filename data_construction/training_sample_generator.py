#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkinRationale generator.

Builds SkinRationale samples (image + diagnosis-reasoning + diagnosis) from
textbook extraction results and the DDx/taxonomy graph files.
"""

import pandas as pd
import json
import re
import hashlib
import random
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI
from textwrap import dedent
import logging
import sys
import os

# Import resolve_path_and_bm_only from model_training (package or src fallback).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_TRAINING_ROOT = os.environ.get(
    "SKIN_R1_MODEL_TRAINING_ROOT",
    os.path.join(_REPO_ROOT, "model_training"),
)
_MODEL_TRAINING_SRC = os.environ.get(
    "SKIN_R1_MODEL_TRAINING_SRC",
    os.path.join(_MODEL_TRAINING_ROOT, "src"),
)
for _p in (_MODEL_TRAINING_ROOT, _MODEL_TRAINING_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.report_RL_diagnosis_hits import resolve_path_and_bm_only
except ImportError:
    from report_RL_diagnosis_hits import resolve_path_and_bm_only
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Lower the httpx/openai log level to avoid logging every HTTP request
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

@dataclass
class TaxoInfo:
    """Taxonomy info"""
    parent_class: Optional[str] = None
    malignancy_level: Optional[str] = None
    chapter_path: Optional[str] = None
    
    def is_empty(self) -> bool:
        return not any([self.parent_class, self.malignancy_level, self.chapter_path])

@dataclass
class Record:
    """Data record"""
    record_id: str
    image_key: str
    rule: str
    diagnosis: str
    diagnosis_mapped: str
    mapped: bool
    taxonomy_text: str

class TrainingSampleGenerator:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SkinRationale generator."""
        self.config = config
        self.synonyms = {}
        self.subtype_info = {}
        self.taxonomy_tree = {}
        self.ddx_graph = {}
        self.diag_index = {}
        self.used_samples = set()
        # OpenAI client (initialized lazily)
        load_dotenv()
        self.openai_client: Optional[OpenAI] = None
        api_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")
        
        # Statistics
        self.stats = {
            'total_records': 0,
            'mapped_records': 0,
            'unmapped_records': 0,
            'type1_samples': 0,
            'type2_samples': 0,
            'type3_samples': 0,
            'type4_samples': 0,
            'type5_samples': 0,
            'no_neighbors': 0,
            'empty_taxonomy': 0
        }
    
    def load_refined_data(self, path_csv: str) -> pd.DataFrame:
        """Load refined_data.csv"""
        logger.info(f"Loading CSV data: {path_csv}")
        df = pd.read_csv(path_csv)
        
        # Ensure the required columns exist
        required_cols = ['image_key', 'rule', 'diagnosis']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add a record_id column if it does not exist
        if 'record_id' not in df.columns:
            df['record_id'] = [f"record_{i}" for i in range(len(df))]
        
        self.stats['total_records'] = len(df)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def load_synonyms(self, path_json: str) -> Dict[str, Any]:
        """Load synonym and subtype info"""
        logger.info(f"Loading synonym data: {path_json}")
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_syn = data.get('synonym_map', {})
        # Lowercase keys and values for case-insensitive matching
        self.synonyms = {str(k).lower().strip(): str(v).lower().strip() for k, v in raw_syn.items()}
        # Lowercase subtypes and benign/malignant categories
        raw_subtype = data.get('subtype_map', {})
        self.subtype_info = {str(parent).lower().strip(): [str(child).lower().strip() for child in children]
                             for parent, children in raw_subtype.items()}
        raw_bm = data.get('benign or precancerous_in_situ or malignant', {})
        self.benign_malignant = {str(level).lower().strip(): [str(dx).lower().strip() for dx in dlist]
                                  for level, dlist in raw_bm.items()}
        
        logger.info(f"Loaded {len(self.synonyms)} synonym mappings")
        logger.info(f"Loaded {len(self.subtype_info)} subtype mappings")
        return data
    
    def load_taxonomy_tree(self, path_json: str) -> Dict[str, Any]:
        """Load the taxonomy tree"""
        logger.info(f"Loading taxonomy tree: {path_json}")
        try:
            with open(path_json, 'r', encoding='utf-8') as f:
                self.taxonomy_tree = json.load(f)
            logger.info(f"Loaded taxonomy tree data")
        except FileNotFoundError:
            logger.warning(f"Taxonomy tree file not found: {path_json}")
            self.taxonomy_tree = {}
        return self.taxonomy_tree
    
    def load_ddxgraph(self, path_graph: str) -> Dict[str, Any]:
        """Load the DDx graph"""
        logger.info(f"Loading DDx graph: {path_graph}")
        with open(path_graph, 'r', encoding='utf-8') as f:
            self.ddx_graph = json.load(f)
        
        nodes = self.ddx_graph.get('nodes', [])
        edges = self.ddx_graph.get('edges', [])
        logger.info(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
        return self.ddx_graph
    
    def set_random_seed(self, seed: int) -> None:
        """Set the random seed"""
        random.seed(seed)
        logger.info(f"Set random seed: {seed}")
    
    def clean_text(self, s: str) -> str:
        """Clean text: collapse whitespace and remove invisible characters"""
        if pd.isna(s) or s == '':
            return ''
        
        # Remove invisible characters and normalize whitespace
        cleaned = re.sub(r'\s+', ' ', str(s).strip())
        return cleaned
    
    def normalize_diagnosis(self, name: str, synonyms: Dict[str, str]) -> Tuple[str, bool]:
        """
        Synonym mapping: case-insensitive normalization.
        
        Returns:
            (std_name, mapped): standardized name and whether mapping succeeded
        """
        if pd.isna(name) or name == '':
            return '', False
        
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in synonyms:
            return synonyms[name_lower], True
        
        # Reverse lookup
        for original, mapped in synonyms.items():
            if mapped.lower() == name_lower:
                return mapped, True
        
        # No mapping found; return the original value
        return name, False
    
    def taxonomy_from_subtype(self, std_name: str, subtype_dict: Dict[str, List[str]]) -> Optional[TaxoInfo]:
        """Build taxonomy info from subtype data"""
        # Helper: collect variants via synonyms (both forward and reverse)
        def variants_for(name: str) -> Set[str]:
            name_l = str(name).lower().strip()
            vars_set: Set[str] = {name_l}
            if name_l in self.synonyms:
                vars_set.add(self.synonyms[name_l])
            for orig, mapped in self.synonyms.items():
                if mapped == name_l:
                    vars_set.add(orig)
            return vars_set

        # Helper: resolve malignancy by direct hit or via parents recursively
        visited: Set[str] = set()
        def resolve_malignancy(name: str) -> Optional[str]:
            name_l = str(name).lower().strip()
            if name_l in visited:
                return None
            visited.add(name_l)
            # Direct check on variants
            for var in variants_for(name_l):
                for level, diseases in self.benign_malignant.items():
                    if var in diseases:
                        return level
            # Recurse to parents
            for parent, children in subtype_dict.items():
                if name_l in children:
                    level = resolve_malignancy(parent)
                    if level:
                        return level
            return None

        # Find immediate parent (first hit)
        parent_class = None
        for parent, children in subtype_dict.items():
            if std_name in children:
                parent_class = parent
                break

        malignancy_level = resolve_malignancy(std_name)

        if parent_class or malignancy_level:
            return TaxoInfo(parent_class=parent_class, malignancy_level=malignancy_level)
        return None
    
    def taxonomy_from_tree(self, std_name: str, tree: Dict[str, Any]) -> Optional[TaxoInfo]:
        """Build taxonomy info from the taxonomy tree.
        Recursively find where std_name appears and return the parent chapter
        path (the last two levels).
        """
        if not tree:
            return None

        std_name_l = str(std_name).lower().strip()

        def dfs(node: Any, path: List[str]) -> Optional[List[str]]:
            if isinstance(node, dict):
                for k, v in node.items():
                    subpath = path + [str(k)]
                    found = dfs(v, subpath)
                    if found is not None:
                        return found
                return None
            else:
                # Leaf level is a diagnosis dict or string mapping; try matching keys or values
                # If it is a string, node is the value and the parent key should hold the diagnosis
                return None

        # Custom scan: the tree is a multi-level dict whose innermost keys are diagnosis names
        found_path: Optional[List[str]] = None

        def scan_for_key(d: Dict[str, Any], path: List[str]) -> Optional[List[str]]:
            for k, v in d.items():
                k_l = str(k).lower().strip()
                if isinstance(v, dict):
                    # If this level already contains the target diagnosis key
                    if std_name_l in [str(x).lower().strip() for x in v.keys()]:
                        return path + [str(k)]
                    res = scan_for_key(v, path + [str(k)])
                    if res is not None:
                        return res
                else:
                    # v is a string; match the key name
                    if k_l == std_name_l:
                        return path
            return None

        # Start from the top
        found_path = scan_for_key(tree, [])
        if found_path:
            # Keep only the last two chapter names (or as many as available)
            tail = found_path[-2:] if len(found_path) >= 2 else found_path
            chapter_path = " > ".join(tail)
            return TaxoInfo(parent_class=None, malignancy_level=None, chapter_path=chapter_path)
        return None
    
    def compose_taxonomy_text(self, info: TaxoInfo, std_name: str) -> str:
        """Compose taxonomy description in English, including diagnosis name.
        Examples:
          "Sebaceous carcinoma is a subtype of adnexal tumor (appendage) and is generally classified as malignant."
          "Melanoma is categorized under Neoplasia > Melanoma."
        """
        if info.is_empty():
            return ""

        dx = (std_name[:1].upper() + std_name[1:]) if std_name else "This entity"

        # Map malignancy to English phrase
        mal_map = {
            "benign": "benign",
            "precancerous_in_situ": "precancerous (in situ)",
            "malignant": "malignant",
        }
        mal_text = None
        if info.malignancy_level:
            mal_text = mal_map.get(str(info.malignancy_level).lower().strip(), str(info.malignancy_level))

        if info.parent_class and mal_text:
            return f"{dx} is a subtype of {info.parent_class} and is generally classified as {mal_text}."
        if info.parent_class and not mal_text:
            return f"{dx} is a subtype of {info.parent_class}."
        if not info.parent_class and mal_text:
            return f"{dx} is generally classified as {mal_text}."
        if info.chapter_path:
            return f"{dx} is categorized under {info.chapter_path}."
        return ""
    
    def compose_taxonomy_text_v2(self, _res: Dict[str, Any], std_name: str) -> str:
        """Compose taxonomy description in English, including diagnosis name.
        Examples:
          "Sebaceous carcinoma is a subtype of adnexal tumor (appendage) and is generally classified as malignant."
          "Melanoma is categorized under Neoplasia > Melanoma."
        """
   

        path_list = _res.get("path", []) or []
        b_or_m = _res.get("b_or_m", "")

        dx = (std_name[:1].upper() + std_name[1:]) if std_name else "This entity"
        path_exist = len(path_list) > 1
        # Map malignancy to English phrase
        mal_map = {
            "benign": "benign",
            "precancerous_in_situ": "precancerous (in situ)",
            "malignant": "malignant",
        }
        mal_text = None
        if b_or_m:
            mal_text = mal_map.get(str(b_or_m))

        if path_exist:
            ancestors = list(reversed(path_list[:-1]))  # from the second-to-last up to the first
            if len(ancestors) == 1:
                text = f"{dx} is a subtype of {ancestors[0]}"
            else:
                text = (
                    f"{dx} is a subtype of "
                    + ", ".join(ancestors[:-1])
                    + f", and {ancestors[-1]}"
                )

            if mal_text:
                text = f"{text}, and is generally classified as {mal_text}."
            else:
                text = f"{text}."
            return text
        if not path_exist and mal_text:
            return f"{dx} is generally classified as {mal_text}."
        return ""
    

    def build_diag_index(self, df: pd.DataFrame) -> Dict[str, List[Record]]:
        """Build the diagnosis inverted index"""
        logger.info("Building the diagnosis inverted index...")
        
        diag_index = defaultdict(list)
        
        for _, row in df.iterrows():
            # Clean and map
            rule = self.clean_text(row['rule'])
            diagnosis = self.clean_text(row['diagnosis'])
            
            if not rule or not diagnosis:
                continue
            
            std_name, mapped = self.normalize_diagnosis(diagnosis, self.synonyms)
            std_name = std_name.lower().strip() if std_name else std_name
            
            if mapped:
                self.stats['mapped_records'] += 1
            else:
                self.stats['unmapped_records'] += 1
            
            _res = resolve_path_and_bm_only("trajectory", std_name)
            taxonomy_text = self.compose_taxonomy_text_v2(_res, std_name)
            
            # Build taxonomy
            # taxo_info = self.taxonomy_from_subtype(std_name, self.subtype_info)
            # if not taxo_info:
            #     taxo_info = self.taxonomy_from_tree(std_name, self.taxonomy_tree)
            
            # taxonomy_text = self.compose_taxonomy_text(taxo_info, std_name) if taxo_info else ""
            
            if not taxonomy_text:
                self.stats['empty_taxonomy'] += 1
            
            # Create the record
            record = Record(
                record_id=str(row.get('record_id', '')),
                image_key=str(row['image_key']),
                rule=rule,
                diagnosis=diagnosis,
                diagnosis_mapped=std_name,
                mapped=mapped,
                taxonomy_text=taxonomy_text
            )
            
            diag_index[std_name].append(record)
        
        self.diag_index = dict(diag_index)
        logger.info(f"Built an index for {len(self.diag_index)} diagnoses")
        return self.diag_index
    
    def standardize_graph_nodes(self, graph: Dict[str, Any], normalize_fn) -> Dict[str, Any]:
        """Standardize graph node names"""
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        # Standardize node names
        node_mapping = {}
        for node in nodes:
            std_name, _ = normalize_fn(node, self.synonyms)
            node_mapping[node] = std_name
        
        # Update the edges
        standardized_edges = []
        for edge in edges:
            if len(edge) >= 2:
                source = node_mapping.get(edge[0], edge[0])
                target = node_mapping.get(edge[1], edge[1])
                standardized_edges.append([source, target])
        
        return {
            'nodes': list(set(node_mapping.values())),
            'edges': standardized_edges
        }
    
    def neighbors_with_records(self, graph: Dict[str, Any], dx: str, diag_index: Dict[str, List[Record]]) -> List[str]:
        """Get neighbors that have records (relaxed: parents/children + synonym variants).
        - Start set = {dx, parents(dx), children(dx)} and their synonym variants (forward/reverse)
        - When matching edges, normalize the opposite endpoint via synonyms; if any variant
          exists in diag_index it counts as a valid neighbor
        - Returned neighbor names use the normalized standard name (matching diag_index keys)
        """
        edges = graph.get('edges', [])
        neighbors = set()

        # Build a child->parents reverse index
        child_to_parents: Dict[str, List[str]] = defaultdict(list)
        for parent, children in self.subtype_info.items():
            for child in children:
                child_to_parents[child].append(parent)

        dx_l = str(dx).lower().strip()
        # Helper returning the set of synonym variants for a name
        def variants_for(name: str) -> Set[str]:
            name_l = str(name).lower().strip()
            vars_set: Set[str] = {name_l}
            # Forward mapping
            if name_l in self.synonyms:
                vars_set.add(self.synonyms[name_l])
            # Reverse mapping
            for orig, mapped in self.synonyms.items():
                if mapped == name_l:
                    vars_set.add(orig)
            return vars_set

        # Collect equivalent start nodes: dx itself + parents + children (and their synonym variants)
        start_nodes = set(variants_for(dx_l))
        # Parents
        for p in child_to_parents.get(dx_l, []):
            start_nodes.update(variants_for(p))
        # Children
        for parent, children in self.subtype_info.items():
            if parent == dx_l:
                for c in children:
                    start_nodes.update(variants_for(c))

        for edge in edges:
            if len(edge) >= 2:
                source_raw, target_raw = str(edge[0]), str(edge[1])
                source, target = source_raw.lower().strip(), target_raw.lower().strip()
                # If source is in the start set, check whether target normalizes to an existing record
                if source in start_nodes:
                    target_std, _ = self.normalize_diagnosis(target, self.synonyms)
                    target_std_l = target_std.lower().strip()
                    if target_std_l in diag_index:
                        neighbors.add(target_std_l)
                    else:
                        # Try the variant set
                        for v in variants_for(target):
                            v_std, _ = self.normalize_diagnosis(v, self.synonyms)
                            v_std_l = v_std.lower().strip()
                            if v_std_l in diag_index:
                                neighbors.add(v_std_l)
                                break
                # If target is in the start set, do the same for the source endpoint
                elif target in start_nodes:
                    source_std, _ = self.normalize_diagnosis(source, self.synonyms)
                    source_std_l = source_std.lower().strip()
                    if source_std_l in diag_index:
                        neighbors.add(source_std_l)
                    else:
                        for v in variants_for(source):
                            v_std, _ = self.normalize_diagnosis(v, self.synonyms)
                            v_std_l = v_std.lower().strip()
                            if v_std_l in diag_index:
                                neighbors.add(v_std_l)
                                break

        return list(neighbors)
    
    def extract_rule_keys(self, rule_text: str) -> Set[str]:
        """Extract keywords from a rule"""
        # Simple keyword extraction
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        words = re.findall(r'\b\w+\b', rule_text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return set(keywords)
    
    def compare_rules(self, keys1: Set[str], keys2: Set[str]) -> Dict[str, Set[str]]:
        """Compare the differences between two rules"""
        common = keys1 & keys2
        unique1 = keys1 - keys2
        unique2 = keys2 - keys1
        
        return {
            'common': common,
            'unique1': unique1,
            'unique2': unique2
        }
    
    def write_reason_keep(self, dx1: str, dx2: str, diff: Dict[str, Set[str]], image_ctx: str = None) -> str:
        """Generate fallback reasoning that keeps the original diagnosis"""
        unique1 = diff['unique1']
        unique2 = diff['unique2']
        
        reason_parts = []
        
        if unique1:
            features1 = ', '.join(list(unique1)[:3])  # take the first 3 features
            reason_parts.append(f"This case shows features of {dx1}: {features1}")
        
        if unique2:
            features2 = ', '.join(list(unique2)[:3])
            reason_parts.append(f"while features of {dx2} ({features2}) are not prominent here")
        
        if reason_parts:
            return "; ".join(reason_parts) + "."
        
        return f"Based on the clinical presentation, this case better fits the diagnostic criteria for {dx1}."
    
    def write_reason_switch(self, dx1: str, dx2: str, diff: Dict[str, Set[str]], image_ctx: str = None) -> str:
        """Generate fallback reasoning that switches the diagnosis"""
        unique1 = diff['unique1']
        unique2 = diff['unique2']
        
        reason_parts = []
        
        if unique2:
            features2 = ', '.join(list(unique2)[:3])
            reason_parts.append(f"This case shows features of {dx2}: {features2}")
        
        if unique1:
            features1 = ', '.join(list(unique1)[:3])
            reason_parts.append(f"while features of {dx1} ({features1}) are atypical here")
        
        if reason_parts:
            return "; ".join(reason_parts) + "."
        
        return f"Based on the clinical presentation, this case better fits {dx2} rather than {dx1}."

    def generate_ddx_reasoning_via_llm(self, rule1: str, dx1: str, rule2: str, dx2: str, mode: str) -> Optional[str]:
        """Call OpenAI Chat Completions (OpenAI SDK) to produce short DDx reasoning in English.
        mode: 'keep' (type2) or 'switch' (type3)
        Returns plain text string or None on failure.
        """
        if self.openai_client is None:
            api_key = self.config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                return None
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.warning(f"Init OpenAI client failed: {e}")
                return None

        model = self.config.get('openai_model', 'gpt-4o-mini')
        max_tokens = int(self.config.get('llm_max_tokens', 120))

        # Simple prompts (editable later) — fix formatting using dedent triple-quoted strings
        if mode == 'keep':
            user_content = dedent(f"""
                You are a dermatology expert. You are given two clinical diagnostic rules extracted from a dermatology textbook, each describing different possible diagnoses for a skin lesion based on visible symptoms.

                Your task:
                - Compare the two rules based ONLY on the clinical features and visible signs described (ignore any histologic, pathologic, or laboratory findings).
                - Determine why the FIRST rule better explains the case and supports keeping the original diagnosis.
                - Write a short, professional differential diagnosis reasoning explaining which clinical features from the first rule are clearly present, and which key features from the second rule are absent or inconsistent with the case.

                Formatting rules:
                - Mention specific symptoms/signs that match the original diagnosis.
                - Mention the missing or absent features for the alternative diagnosis.
                - Avoid generic statements like "characteristic features" or "aid in diagnosis."
                - Output in this format:
                <differential diagnosis reasoning>Your reasoning here</differential diagnosis reasoning>

                Example:
                <differential diagnosis reasoning>This case shows keratotic papules with a stuck-on appearance, consistent with seborrheic keratosis, while the irregular sun-exposed macules typical of solar lentigo are absent.</differential diagnosis reasoning>

                Input:
                primary diagnosis (Diagnosis: {dx1}): {rule1}
                differential diagnosis (Diagnosis: {dx2}): {rule2}
                Output:
                <differential diagnosis reasoning>Your reasoning here</differential diagnosis reasoning>    
            """)
        else:
            user_content = dedent(f"""
                You are a dermatology expert. You are given two clinical diagnostic rules extracted from a dermatology textbook, each describing different possible diagnoses for a skin lesion based on visible symptoms.

                Your task:
                - Compare the two rules based ONLY on the clinical features and visible signs described (ignore any histologic, pathologic, or laboratory findings).
                - Determine why the SECOND rule better explains the case and supports replacing the original diagnosis with the differential diagnosis.
                - Write a short, professional differential diagnosis reasoning explaining which clinical features from the second rule are clearly present, and which key features from the first rule are absent or inconsistent with the case.

                Formatting rules:
                - Mention specific symptoms/signs that match the differential diagnosis.
                - Mention the missing or absent features for the original diagnosis.
                - Avoid generic statements like "characteristic features" or "aid in diagnosis."
                - Output in this format:
                <differential diagnosis reasoning>Your reasoning here</differential diagnosis reasoning>

                Example:
                <differential diagnosis reasoning>The lesion shows light-brown macules with irregular outlines on sun-exposed skin, consistent with solar lentigo, while the stuck-on keratotic surface typical of seborrheic keratosis is absent.</differential diagnosis reasoning>
                Input:
                primary diagnosis (Diagnosis: {dx1}): {rule1}
                differential diagnosis (Diagnosis: {dx2}): {rule2}
                Output:
                <differential diagnosis reasoning>Your reasoning here</differential diagnosis reasoning>

            """)

        try:
            resp = self.openai_client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a concise, professional dermatology assistant."},
                    {"role": "user", "content": user_content},
                ],
            )
            content = resp.choices[0].message.content.strip()
            return content
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}")
            return None
    
    def make_sample_type1(self, record: Record, taxo_text: str) -> str:
        """Generate a type-1 sample (basic sample)"""
        sample = f"""<thinking>{record.rule}</thinking>
        <diagnosis>{record.diagnosis_mapped}, {taxo_text}</diagnosis>"""
        return sample
    
    def make_sample_type2(self, record1: Record, record2: Record, taxo_text: str, reason: str) -> str:
        """Generate a type-2 sample (DDx that keeps the original diagnosis)"""
        ddx_text = self.build_ddx_summary(record1.diagnosis_mapped, record2.diagnosis_mapped, record2.rule)
        clean_reason = self.sanitize_reason_text(reason)
        sample = f"""<thinking>
        Based on the rule: {record1.rule} We can give a primary diagnosis that {record1.diagnosis_mapped}. {ddx_text} 
        {clean_reason} Therefore, the most likely condition corresponds to "{record1.diagnosis_mapped}". 
        </thinking>
        <diagnosis>{record1.diagnosis_mapped}. {taxo_text}</diagnosis>"""
        return sample
    
    def make_sample_type3(self, record1_ddx: Record, record2_true: Record, taxo_text: str, reason: str) -> str:
        """Generate a type-3 sample (DDx that switches the diagnosis)"""
        ddx_text = self.build_ddx_summary(record1_ddx.diagnosis_mapped, record2_true.diagnosis_mapped, record2_true.rule)
        clean_reason = self.sanitize_reason_text(reason)
        sample = f"""<thinking>
        Based on the rule: {record1_ddx.rule} We can give a primary diagnosis that {record1_ddx.diagnosis_mapped}. {ddx_text} 
        {clean_reason} Therefore, the most likely condition corresponds to "{record2_true.diagnosis_mapped}". 
        </thinking>
        <diagnosis>{record2_true.diagnosis_mapped}. {taxo_text}</diagnosis>"""
        return sample
    
    def make_sample_type4(self, record1: Record, record2: Record, taxo_text: str, reason: str) -> str:
        """Generate a type-4 sample (random DDx that keeps the original diagnosis)"""
        ddx_text = self.build_non_ddx_summary(record1.diagnosis_mapped, record2.diagnosis_mapped, record2.rule)
        clean_reason = self.sanitize_reason_text(reason)
        sample = f"""<thinking>
        Based on the rule: {record1.rule} We can give a primary diagnosis that {record1.diagnosis_mapped}. {ddx_text} 
        {clean_reason} Therefore, the most likely condition corresponds to "{record1.diagnosis_mapped}". 
        </thinking>
        <diagnosis>{record1.diagnosis_mapped}. {taxo_text}</diagnosis>"""
        return sample
    
    def make_sample_type5(self, record1_ddx: Record, record2_true: Record, taxo_text: str, reason: str) -> str:
        """Generate a type-5 sample (random DDx that switches the diagnosis)"""
        ddx_text = self.build_non_ddx_summary(record1_ddx.diagnosis_mapped, record2_true.diagnosis_mapped, record2_true.rule)
        clean_reason = self.sanitize_reason_text(reason)
        sample = f"""<thinking>
        Based on the rule: {record1_ddx.rule} We can give a primary diagnosis that {record1_ddx.diagnosis_mapped}. {ddx_text} 
        {clean_reason} Therefore, the most likely condition corresponds to "{record2_true.diagnosis_mapped}". 
        </thinking>
        <diagnosis>{record2_true.diagnosis_mapped}. {taxo_text}</diagnosis>"""
        return sample

    def build_ddx_summary(self, dx1: str, dx2: str, rule2: str) -> str:
        """Build a coherent English description for the <differential diagnosis> field."""
        dx1_en = dx1
        dx2_en = dx2
        rule2_clean = rule2.strip()
        return (
            f"Considering the differential diagnosis for {dx1_en}, namely {dx2_en}, "
            f"we compare against the diagnostic rule for {dx2_en}: {rule2_clean}"
        )

    def build_non_ddx_summary(self, dx1: str, dx2: str, rule2: str) -> str:
        """Build a coherent English description for the <non-differential diagnosis> field."""
        dx1_en = dx1
        dx2_en = dx2
        rule2_clean = rule2.strip()
        return (
            f"To enhance the diagnostic reasoning, compare the primary diagnosis '{dx1_en}' "
            f"with the diagnostic rule of '{dx2_en}': {rule2_clean}"
        )

    def sanitize_reason_text(self, text: Optional[str]) -> str:
        """Strip any <differential diagnosis reasoning> tags from the LLM output, keeping the content."""
        if not text:
            return ""
        t = str(text)
        # Unwrap to the inner text
        try:
            import re
            t = re.sub(r"<\s*differential diagnosis reasoning\s*>", "", t, flags=re.IGNORECASE)
            t = re.sub(r"<\s*/\s*differential diagnosis reasoning\s*>", "", t, flags=re.IGNORECASE)
        except Exception:
            pass
        return t.strip()
    
    def pick_ddx_record(self, dx: str, diag_index: Dict[str, List[Record]], avoid_image_keys: Set[str]) -> Optional[Record]:
        """Pick a DDx record, avoiding duplicate images"""
        if dx not in diag_index:
            return None
        
        available_records = [r for r in diag_index[dx] if r.image_key not in avoid_image_keys]
        
        if not available_records:
            # If no usable record is available, return any of them
            available_records = diag_index[dx]
        
        if available_records:
            return random.choice(available_records)
        
        return None
    
    def pick_random_ddx(self, current_dx: str, diag_index: Dict[str, List[Record]]) -> Optional[str]:
        """Randomly pick a DDx from all possible diagnoses (excluding the current one)"""
        all_diagnoses = list(diag_index.keys())
        if len(all_diagnoses) <= 1:
            return None
        
        # Exclude the current diagnosis
        available_diagnoses = [dx for dx in all_diagnoses if dx != current_dx]
        if not available_diagnoses:
            return None
        
        return random.choice(available_diagnoses)
    
    def is_duplicate(self, sample_text: str, hash_set: Set[str]) -> bool:
        """Check for duplicates"""
        sample_hash = hashlib.md5(sample_text.encode()).hexdigest()
        if sample_hash in hash_set:
            return True
        hash_set.add(sample_hash)
        return False
    
    def generate_samples(self) -> Dict[str, List[str]]:
        """Generate all sample types"""
        logger.info("Generating training samples...")
        
        # Standardize graph nodes
        self.ddx_graph = self.standardize_graph_nodes(self.ddx_graph, self.normalize_diagnosis)
        
        samples = {'type1': [], 'type2': [], 'type3': [], 'type4': [], 'type5': []}
        samples_jsonl = {'type1': [], 'type2': [], 'type3': [], 'type4': [], 'type5': []}
        
        sample_hashes = set()
        
        # Generate type-1 samples (basic samples)
        logger.info("Generating type-1 samples...")
        for dx, records in self.diag_index.items():
            for record in records:
                sample = self.make_sample_type1(record, record.taxonomy_text)
                if self.config.get('output_format', 'text') == 'jsonl':
                    obj = {
                        'type': 'type1',
                        'record_id': record.record_id,
                        'image_key': record.image_key,
                        'diagnosis': record.diagnosis_mapped,
                        'text': sample
                    }
                    samples_jsonl['type1'].append(obj)
                    # Also count type-1 samples in JSONL mode
                    self.stats['type1_samples'] += 1
                else:
                    if not self.is_duplicate(sample, sample_hashes):
                        samples['type1'].append(sample)
                        self.stats['type1_samples'] += 1
        
        # Generate type-2 and type-3 samples
        logger.info("Generating type-2 and type-3 samples...")
        max_t2 = int(self.config.get('max_per_diag_type2', 3))
        max_t3 = int(self.config.get('max_per_diag_type3', 3))
        min_repeat_if_single = int(self.config.get('min_repeats_if_single', 2))
        type2_count_per_dx = defaultdict(int)
        type3_count_per_dx = defaultdict(int)
        # Global example caps (0 or negative means no global cap)
        limit_t2 = int(self.config.get('limit_type2_examples', 0))
        limit_t3 = int(self.config.get('limit_type3_examples', 0))
        total_t2 = 0
        total_t3 = 0
        for dx, records in self.diag_index.items():
            # Get neighbor nodes
            neighbors = self.neighbors_with_records(self.ddx_graph, dx, self.diag_index)

            if not neighbors:
                self.stats['no_neighbors'] += 1
                continue

            # Use a while loop so each diagnosis reaches its per-diagnosis or global cap when possible
            attempts = 0
            max_attempts = max(10, len(records) * max(max_t2, max_t3) * 3)
            while attempts < max_attempts:
                limited2 = (limit_t2 > 0 and total_t2 >= limit_t2)
                limited3 = (limit_t3 > 0 and total_t3 >= limit_t3)
                if (limited2 and limited3) or (type2_count_per_dx[dx] >= max_t2 and type3_count_per_dx[dx] >= max_t3):
                    break

                record1 = random.choice(records)
                dx2 = random.choice(neighbors)
                record2 = self.pick_ddx_record(dx2, self.diag_index, {record1.image_key})
                if not record2:
                    attempts += 1
                    continue

                # Type-2 sample (keep the original diagnosis)
                if type2_count_per_dx[dx] < max_t2 and not (limit_t2 > 0 and total_t2 >= limit_t2):
                    reason_keep = self.generate_ddx_reasoning_via_llm(
                        record1.rule, record1.diagnosis_mapped, record2.rule, record2.diagnosis_mapped, mode='keep'
                    )
                    if not reason_keep:
                        reason_keep = f"The visible clinical features better fit {record1.diagnosis_mapped} than {record2.diagnosis_mapped}."
                    clean_keep = self.sanitize_reason_text(reason_keep)
                    # logger.info(f"Type2 DDx reasoning: {clean_keep}")
                    sample2 = self.make_sample_type2(record1, record2, record1.taxonomy_text, clean_keep)
                    allow_dup = False
                    if len(self.diag_index.get(dx2, [])) == 1 and type2_count_per_dx[dx] < min_repeat_if_single:
                        allow_dup = True
                    if self.config.get('output_format', 'text') == 'jsonl':
                        obj2 = {
                            'type': 'type2',
                            'record1_id': record1.record_id,
                            'record1_image_key': record1.image_key,
                            'record2_id': record2.record_id,
                            'record2_image_key': record2.image_key,
                            'dx1': record1.diagnosis_mapped,
                            'dx2': record2.diagnosis_mapped,
                            'text': sample2
                        }
                        samples_jsonl['type2'].append(obj2)
                        self.stats['type2_samples'] += 1
                        type2_count_per_dx[dx] += 1
                        total_t2 += 1
                    else:
                        if allow_dup or not self.is_duplicate(sample2, sample_hashes):
                            samples['type2'].append(sample2)
                            self.stats['type2_samples'] += 1
                            type2_count_per_dx[dx] += 1
                            total_t2 += 1

                # Type-3 sample (switch the diagnosis)
                if type3_count_per_dx[dx] < max_t3 and not (limit_t3 > 0 and total_t3 >= limit_t3):
                    reason_switch = self.generate_ddx_reasoning_via_llm(
                        record1.rule, record1.diagnosis_mapped, record2.rule, record2.diagnosis_mapped, mode='switch'
                    )
                    if not reason_switch:
                        reason_switch = f"The visible clinical features better fit {record2.diagnosis_mapped} than {record1.diagnosis_mapped}."
                    clean_switch = self.sanitize_reason_text(reason_switch)
                    # logger.info(f"Type3 DDx reasoning: {clean_switch}")
                    sample3 = self.make_sample_type3(record1, record2, record2.taxonomy_text, clean_switch)
                    allow_dup3 = False
                    if len(self.diag_index.get(dx2, [])) == 1 and type3_count_per_dx[dx] < min_repeat_if_single:
                        allow_dup3 = True
                    if self.config.get('output_format', 'text') == 'jsonl':
                        obj3 = {
                            'type': 'type3',
                            'record1_id': record1.record_id,
                            'record1_image_key': record1.image_key,
                            'record2_id': record2.record_id,
                            'record2_image_key': record2.image_key,
                            'dx1': record1.diagnosis_mapped,
                            'dx2': record2.diagnosis_mapped,
                            'text': sample3
                        }
                        samples_jsonl['type3'].append(obj3)
                        self.stats['type3_samples'] += 1
                        type3_count_per_dx[dx] += 1
                        total_t3 += 1
                    else:
                        if allow_dup3 or not self.is_duplicate(sample3, sample_hashes):
                            samples['type3'].append(sample3)
                            self.stats['type3_samples'] += 1
                            type3_count_per_dx[dx] += 1
                            total_t3 += 1

                attempts += 1
        
        # Generate type-4 and type-5 samples (using a random DDx)
        logger.info("Generating type-4 and type-5 samples...")
        max_t4 = int(self.config.get('max_per_diag_type4', 3))
        max_t5 = int(self.config.get('max_per_diag_type5', 3))
        limit_t4 = int(self.config.get('limit_type4_examples', 0))
        limit_t5 = int(self.config.get('limit_type5_examples', 0))
        total_t4 = 0
        total_t5 = 0
        type4_count_per_dx = defaultdict(int)
        type5_count_per_dx = defaultdict(int)
        
        for dx, records in self.diag_index.items():
            # Use a while loop so each diagnosis reaches its per-diagnosis or global cap when possible
            attempts = 0
            max_attempts = max(10, len(records) * max(max_t4, max_t5) * 3)
            while attempts < max_attempts:
                limited4 = (limit_t4 > 0 and total_t4 >= limit_t4)
                limited5 = (limit_t5 > 0 and total_t5 >= limit_t5)
                if (limited4 and limited5) or (type4_count_per_dx[dx] >= max_t4 and type5_count_per_dx[dx] >= max_t5):
                    break

                record1 = random.choice(records)
                # Use a random DDx instead of a graph neighbor
                dx2 = self.pick_random_ddx(dx, self.diag_index)
                if not dx2:
                    attempts += 1
                    continue
                    
                record2 = self.pick_ddx_record(dx2, self.diag_index, {record1.image_key})
                if not record2:
                    attempts += 1
                    continue

                # Type-4 sample (keep the original diagnosis)
                if type4_count_per_dx[dx] < max_t4 and not (limit_t4 > 0 and total_t4 >= limit_t4):
                    reason_keep = self.generate_ddx_reasoning_via_llm(
                        record1.rule, record1.diagnosis_mapped, record2.rule, record2.diagnosis_mapped, mode='keep'
                    )
                    if not reason_keep:
                        reason_keep = f"The visible clinical features better fit {record1.diagnosis_mapped} than {record2.diagnosis_mapped}."
                    clean_keep = self.sanitize_reason_text(reason_keep)
                    sample4 = self.make_sample_type4(record1, record2, record1.taxonomy_text, clean_keep)
                    if self.config.get('output_format', 'text') == 'jsonl':
                        obj4 = {
                            'type': 'type4',
                            'record1_id': record1.record_id,
                            'record1_image_key': record1.image_key,
                            'record2_id': record2.record_id,
                            'record2_image_key': record2.image_key,
                            'dx1': record1.diagnosis_mapped,
                            'dx2': record2.diagnosis_mapped,
                            'text': sample4
                        }
                        samples_jsonl['type4'].append(obj4)
                        self.stats['type4_samples'] += 1
                        type4_count_per_dx[dx] += 1
                        total_t4 += 1
                    else:
                        if not self.is_duplicate(sample4, sample_hashes):
                            samples['type4'].append(sample4)
                            self.stats['type4_samples'] += 1
                            type4_count_per_dx[dx] += 1
                            total_t4 += 1

                # Type-5 sample (switch the diagnosis)
                if type5_count_per_dx[dx] < max_t5 and not (limit_t5 > 0 and total_t5 >= limit_t5):
                    reason_switch = self.generate_ddx_reasoning_via_llm(
                        record1.rule, record1.diagnosis_mapped, record2.rule, record2.diagnosis_mapped, mode='switch'
                    )
                    if not reason_switch:
                        reason_switch = f"The visible clinical features better fit {record2.diagnosis_mapped} than {record1.diagnosis_mapped}."
                    clean_switch = self.sanitize_reason_text(reason_switch)
                    sample5 = self.make_sample_type5(record1, record2, record2.taxonomy_text, clean_switch)
                    if self.config.get('output_format', 'text') == 'jsonl':
                        obj5 = {
                            'type': 'type5',
                            'record1_id': record1.record_id,
                            'record1_image_key': record1.image_key,
                            'record2_id': record2.record_id,
                            'record2_image_key': record2.image_key,
                            'dx1': record1.diagnosis_mapped,
                            'dx2': record2.diagnosis_mapped,
                            'text': sample5
                        }
                        samples_jsonl['type5'].append(obj5)
                        self.stats['type5_samples'] += 1
                        type5_count_per_dx[dx] += 1
                        total_t5 += 1
                    else:
                        if not self.is_duplicate(sample5, sample_hashes):
                            samples['type5'].append(sample5)
                            self.stats['type5_samples'] += 1
                            type5_count_per_dx[dx] += 1
                            total_t5 += 1

                attempts += 1
        
        logger.info("Sample generation complete")
        if self.config.get('output_format', 'text') == 'jsonl':
            return samples_jsonl
        return samples
    
    def write_samples(self, samples: Dict[str, List[Any]], output_dir: str):
        """Write the sample files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        outfmt = self.config.get('output_format', 'text')
        for sample_type, sample_list in samples.items():
            ext = 'jsonl' if outfmt == 'jsonl' else 'txt'
            output_file = os.path.join(output_dir, f"train_{sample_type}.{ext}")
            with open(output_file, 'w', encoding='utf-8') as f:
                if outfmt == 'jsonl':
                    for obj in sample_list:
                        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                else:
                    for sample in sample_list:
                        f.write(sample + '\n\n')
            logger.info(f"Wrote {len(sample_list)} {sample_type} samples to {output_file}")
    
    def log_stats(self):
        """Log the statistics"""
        logger.info("=== Statistics ===")
        logger.info(f"Total records: {self.stats['total_records']}")
        logger.info(f"Mapped: {self.stats['mapped_records']}")
        logger.info(f"Unmapped: {self.stats['unmapped_records']}")
        logger.info(f"Mapping success rate: {self.stats['mapped_records']/self.stats['total_records']*100:.1f}%")
        logger.info(f"Empty taxonomy: {self.stats['empty_taxonomy']}")
        logger.info(f"No neighbors: {self.stats['no_neighbors']}")
        logger.info(f"Type-1 samples: {self.stats['type1_samples']}")
        logger.info(f"Type-2 samples: {self.stats['type2_samples']}")
        logger.info(f"Type-3 samples: {self.stats['type3_samples']}")
        logger.info(f"Type-4 samples: {self.stats['type4_samples']}")
        logger.info(f"Type-5 samples: {self.stats['type5_samples']}")
        logger.info(f"Total samples: {sum([self.stats['type1_samples'], self.stats['type2_samples'], self.stats['type3_samples'], self.stats['type4_samples'], self.stats['type5_samples']])}")

def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description='Training-sample generator')
    parser.add_argument('--csv', default='refined_data.csv', help='Path to refined_data.csv')
    parser.add_argument('--synonyms', default='synonym_and_subtype2.json', help='Path to the synonym file')
    parser.add_argument('--taxonomy', default='taxonomy_tree.json', help='Path to the taxonomy tree file')
    parser.add_argument('--ddxgraph', default='ddx_graph_merged.json', help='Path to the DDx graph file')
    parser.add_argument('--output', default='training_samples', help='Output directory')
    parser.add_argument('--format', dest='outfmt', choices=['text','jsonl'], default='text', help='Output format')
    parser.add_argument('--max_per_diag_type2', type=int, default=3, help='Max type-2 samples per diagnosis')
    parser.add_argument('--max_per_diag_type3', type=int, default=3, help='Max type-3 samples per diagnosis')
    parser.add_argument('--max_per_diag_type4', type=int, default=3, help='Max type-4 samples per diagnosis')
    parser.add_argument('--max_per_diag_type5', type=int, default=3, help='Max type-5 samples per diagnosis')
    parser.add_argument('--min_repeats_if_single', type=int, default=2, help='Min repeats when a DDx has only one record')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # LLM config (simple)
    parser.add_argument('--openai-model', default='gpt-4o-mini', help='OpenAI Chat Completions model name')
    parser.add_argument('--openai-api-key', default=None, help='OpenAI API Key (or env OPENAI_API_KEY)')
    parser.add_argument('--llm-max-tokens', type=int, default=120, help='Max tokens per LLM response')
    parser.add_argument('--llm-timeout', type=int, default=20, help='HTTP timeout seconds per LLM request')
    # Global caps (0 means unlimited)
    parser.add_argument('--limit_type2_examples', type=int, default=0, help='Global cap for type2 examples (0=unlimited)')
    parser.add_argument('--limit_type3_examples', type=int, default=0, help='Global cap for type3 examples (0=unlimited)')
    parser.add_argument('--limit_type4_examples', type=int, default=0, help='Global cap for type4 examples (0=unlimited)')
    parser.add_argument('--limit_type5_examples', type=int, default=0, help='Global cap for type5 examples (0=unlimited)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'csv_path': args.csv,
        'synonyms_path': args.synonyms,
        'taxonomy_path': args.taxonomy,
        'ddxgraph_path': args.ddxgraph,
        'output_dir': args.output,
        'random_seed': args.seed,
        'output_format': args.outfmt,
        'max_per_diag_type2': args.max_per_diag_type2,
        'max_per_diag_type3': args.max_per_diag_type3,
        'max_per_diag_type4': args.max_per_diag_type4,
        'max_per_diag_type5': args.max_per_diag_type5,
        'min_repeats_if_single': args.min_repeats_if_single
        , 'openai_model': args.openai_model
        , 'openai_api_key': args.openai_api_key
        , 'llm_max_tokens': args.llm_max_tokens
        , 'llm_timeout': args.llm_timeout
        , 'limit_type2_examples': args.limit_type2_examples
        , 'limit_type3_examples': args.limit_type3_examples
        , 'limit_type4_examples': args.limit_type4_examples
        , 'limit_type5_examples': args.limit_type5_examples
    }
    
    # Create the generator
    generator = TrainingSampleGenerator(config)
    
    try:
        # Set the random seed
        generator.set_random_seed(args.seed)
        
        # Load data
        df = generator.load_refined_data(args.csv)
        generator.load_synonyms(args.synonyms)
        generator.load_taxonomy_tree(args.taxonomy)
        generator.load_ddxgraph(args.ddxgraph)
        
        # Build the index
        generator.build_diag_index(df)
        
        # Generate samples
        samples = generator.generate_samples()
        
        # Write the files
        generator.write_samples(samples, args.output)
        
        # Log statistics
        generator.log_stats()
        
        logger.info("Training-sample generation complete.")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
