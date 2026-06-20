#!/usr/bin/env python3
"""
Report diagnosis hits against taxonomy/synonyms for all RL datasets.

Rules summary (per user spec):
- For text diagnosis datasets: use dataset-specific short_name map first (if any),
  then global synonym_map to get candidate expressions; check if diagnosis or any
  synonym appears in subtype_map; also check its parents in dataset taxonomy
  taxonomy_RL_dataset[dataset] to see if any parent appears in subtype_map;
  then check diagnosis/synonyms and any ascended parents (multi-level) against
  "benign or precancerous_in_situ or malignant" buckets to classify.
- For dict diagnosis datasets:
  - derm12345: only use 'label' (code like 'acb'); map code to full path using
    taxonomy_RL_dataset['derm12345'] (reverse mapping); then apply the same
    subtype/benign checks on any names along the path (and their synonyms).
  - dermnet: only use second-level 'diagnosis' i.e., record['diagnosis']['diagnosis'].

Outputs:
- Console summary per dataset and overall.
- Detailed JSON report including unmatched ratios for unique diagnoses and samples.
"""

import json
import os
import sys
from collections import defaultdict, Counter
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from .paths import default_data_root
except ImportError:  # imported from data_construction via model_training/src on sys.path
    from paths import default_data_root

DATA_ROOT = default_data_root()
SYNONYM_JSON = os.path.join(DATA_ROOT, "synonym_and_subtype2.json")


# def _install_torch_stub() -> None:
#     """Install a minimal torch stub in sys.modules to import src.data without real torch.
#     Provides torch.utils.data.Dataset symbol only.
#     """
#     if 'torch' in sys.modules:
#         return
#     torch_mod = ModuleType('torch')
#     utils_mod = ModuleType('torch.utils')
#     data_mod = ModuleType('torch.utils.data')

#     class _Dataset:  # minimal placeholder
#         pass

#     data_mod.Dataset = _Dataset
#     # wire modules
#     sys.modules['torch'] = torch_mod
#     sys.modules['torch.utils'] = utils_mod
#     sys.modules['torch.utils.data'] = data_mod


def _normalize(s: str) -> str:
    return (s or '').strip().lower()


def _normalize_cfg_text(obj: Any) -> Any:
    """Recursively normalize all string values in the config"""
    if isinstance(obj, dict):
        return {k: _normalize_cfg_text(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_cfg_text(item) for item in obj]
    elif isinstance(obj, str):
        return _normalize(obj)
    else:
        return obj


def _load_config_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # Normalize all text content in cfg
    return _normalize_cfg_text(cfg)





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
        print(f"Unknown dict shape: {diag_raw}")
        return {}
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

def resolve_path_and_bm_only(dataset: str, diag_raw: Any) -> Dict[str, Any]:
    """Simple entry: only dataset and diagnosis are required.

    Internally loads config and builds indices using the same resources
    as the reporting script, then returns only:
      { 'path': List[str], 'b_or_m': Optional[str] }
    """
    # Use the same config path as main()
    json_path = SYNONYM_JSON
    path: List[str] = []

    cfg = _load_config_json(json_path)
    synonym_map = _build_synonym_lookup(cfg.get('synonym_map', {}))
    subtype_map = cfg.get('subtype_map', {})
    family_keys, value_to_family, subtype_all_names = _build_subtype_index(subtype_map)
    child_to_parent = build_child_to_parent(subtype_map)
    buckets = _collect_buckets(cfg.get('benign or precancerous_in_situ or malignant', {}))
    if dataset != 'trajectory':
        taxonomy_all = cfg.get('taxonomy_RL_dataset', {})
        short_map_all = cfg.get('short_name_RL_dataset', {})

        short_map = _build_shortname_lookup(short_map_all, dataset)
        ds_taxonomy = taxonomy_all.get(dataset, {}) or {}

    
        if dataset == 'derm12345':
            # print(diag_raw)
            code_to_path = _taxonomy_build_code_to_path(taxonomy_all.get('derm12345', {}))
            path = code_to_path.get(diag_raw, [])[:]
            # print(f"path: {path}")
        else:
            term_norm = _normalize(diag_raw)

            if dataset in ['BCN20000', 'dermnet', 'HAM10000', 'PAD-UFES-20']:
                term_norm = short_map.get(term_norm, term_norm)
                # print(f"term_norm: {term_norm}")

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
                path = best_chain + [term_norm]
            else:
                # Fallback: use the normalized/mapped term, not the raw term
                path = [term_norm]
    else:
        term_norm = _normalize(diag_raw)
        path = [term_norm]

    path = rebuild_path_with_taxonomy_all(path, subtype_all_names, child_to_parent)
    # Find b_or_m using the path
    b_or_m = _find_b_or_m_from_path(path, buckets)
    
    return { 'path': path, 'b_or_m': b_or_m } 

def rebuild_path_with_taxonomy_all(
    path: list[str],
    subtype_all_names: set[str],
    child_to_parent: dict[str, str],
) -> list[str]:
    """
    Rebuild the path using child_to_parent, ensuring the output is coarse->fine.
    """

    def climb_to_root(term: str, mapping: dict[str, str]) -> list[str]:
        """Climb up recursively until there is no parent; return a coarse->fine chain."""
        chain = [term]
        cur = _normalize(term)
        while cur in mapping:
            parent = mapping[cur]
            chain.append(parent)
            cur = parent
        chain.reverse()  # normalize to coarse->fine
        return chain

    rebuilt_path = None
    reversed_path = list(reversed(path))  # fine->coarse

    for i, term in enumerate(reversed_path):
        term_norm = _normalize(term)

        if term_norm in child_to_parent or term_norm in subtype_all_names:
            finer_terms = reversed_path[:i]  # nodes finer than the matched term
            taxonomy_chain = climb_to_root(term_norm, child_to_parent) \
                             if term_norm in child_to_parent else [term_norm]
            rebuilt_path = taxonomy_chain + list(reversed(finer_terms))
            break

    if rebuilt_path is None:
        rebuilt_path = path  # fallback

    return rebuilt_path

   
def build_child_to_parent(subtype_map: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a direct child -> parent mapping table from subtype_map.
    """
    mapping: Dict[str, str] = {}
    for parent, children in (subtype_map or {}).items():
        parent_norm = _normalize(parent)
        if isinstance(children, list):
            for child in children:
                child_norm = _normalize(child)
                mapping[child_norm] = parent_norm
    return mapping

def _find_b_or_m_from_path(path: List[str], buckets: Dict[str, Set[str]]) -> Optional[str]:
    """
    Look up each item of the path from fine to coarse granularity; return on the
    first hit, otherwise return None.

    Args:
        path: list of diagnosis path terms, from fine to coarse granularity
        buckets: benign/malignant classification table, e.g. {"benign": set(...), "malignant": set(...), ...}

    Returns:
        The matched bucket string (e.g. "benign", "malignant"), or None if not found.
    """
    # Iterate over each path item from fine to coarse granularity
    for term in path:
        term_norm = _normalize(term)
        if not term_norm:
            continue
            
        # Look it up in each classification bucket
        for bucket_name, term_set in buckets.items():
            if term_norm in term_set:
                return bucket_name
    
    return None

def resolve_path_and_bm_simple(dataset: str, diag_raw: Any) -> Dict[str, Any]:
    """Simple entry: only dataset and diagnosis are required.

    Internally loads config and builds indices using the same resources
    as the reporting script, then returns only:
      { 'path': List[str], 'b_or_m': Optional[str] }
    """
    # Use the same config path as main()
    json_path = SYNONYM_JSON

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

def main() -> None:
    data_root = DATA_ROOT
    json_path = SYNONYM_JSON

    # Load config
    cfg = _load_config_json(json_path)
    synonym_map = _build_synonym_lookup(cfg.get('synonym_map', {}))
    subtype_map = cfg.get('subtype_map', {})
    family_keys, value_to_family, subtype_all_names = _build_subtype_index(subtype_map)
    buckets = _collect_buckets(cfg.get('benign or precancerous_in_situ or malignant', {}))
    taxonomy_all = cfg.get('taxonomy_RL_dataset', {})
    short_map_all = cfg.get('short_name_RL_dataset', {})

    # Prepare import of dataset loader. Run this report as a module from the
    # model_training/ root, e.g. `python -m src.report_RL_diagnosis_hits`.
    try:
        from .data import load_dataset  # type: ignore
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data import load_dataset  # type: ignore

    # Load all RL datasets at once using existing loader (with caching)
    records = load_dataset("RL")
    all_df = pd.DataFrame(records)
    if 'source' not in all_df.columns:
        raise RuntimeError("Loaded RL dataset is missing 'source' column")
    rl_datasets = list(sorted(all_df['source'].unique()))

    per_dataset_details: Dict[str, Any] = {}
    overall_samples = 0
    overall_any_match = 0
    overall_subtype_match = 0
    overall_bm_match = 0
    overall_diag_unique: Set[Tuple[str, str]] = set()  # (dataset, diag_key)
    overall_diag_unique_any_match = 0
    overall_diag_unique_subtype_match = 0
    overall_diag_unique_bm_match = 0

    for ds in rl_datasets:
        df = all_df[all_df['source'] == ds].reset_index(drop=True)
        if df is None or df.empty:
            per_dataset_details[ds] = {
                'status': 'empty_or_missing',
                'samples': 0,
            }
            continue

        # dataset-specific helpers
        short_map = _build_shortname_lookup(short_map_all, ds)
        ds_taxonomy = taxonomy_all.get(ds, {}) or {}

        # derm12345 code reverse map
        code_to_path: Dict[str, List[str]] = {}
        if ds == 'derm12345':
            code_to_path = _taxonomy_build_code_to_path(ds_taxonomy)

        # Pull diagnosis values + normalization per dataset rules
        sample_hits: List[Dict[str, Any]] = []
        diag_to_hit: Dict[str, Dict[str, Any]] = {}

        for _, row in df.iterrows():
            diag_raw = row.get('diagnosis', None)
            hit_info: Dict[str, Any]
            diag_key: str  # unique key per diagnosis value as string

            if isinstance(diag_raw, dict) and ds == 'derm12345':
                label_code = str(diag_raw.get('label', '')).strip()
                diag_key = f"label:{label_code}"
            elif isinstance(diag_raw, dict) and ds == 'dermnet':
                diag_key = str(diag_raw.get('diagnosis', '')).strip()
            else:
                diag_key = str(diag_raw or '').strip()

            hit_info = resolve_hits(
                dataset=ds,
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

            subtype_matched = hit_info.get('subtype_hit') is not None
            bm_matched = hit_info.get('b_or_m_hit') is not None
            any_matched = subtype_matched or bm_matched

            sample_hits.append({
                'diag_key': diag_key,
                **hit_info,
                'matched_subtype': subtype_matched,
                'matched_b_or_m': bm_matched,
                'matched_any': any_matched,
            })

            # Aggregate per-diagnosis unique
            if diag_key not in diag_to_hit:
                diag_to_hit[diag_key] = {
                    'examples': 1,
                    'matched_subtype': subtype_matched,
                    'matched_b_or_m': bm_matched,
                    'matched_any': any_matched,
                }
            else:
                d = diag_to_hit[diag_key]
                d['examples'] += 1
                d['matched_subtype'] = d['matched_subtype'] or subtype_matched
                d['matched_b_or_m'] = d['matched_b_or_m'] or bm_matched
                d['matched_any'] = d['matched_any'] or any_matched

        # Summaries
        num_samples = len(sample_hits)
        num_subtype_matched = sum(1 for h in sample_hits if h['matched_subtype'])
        num_bm_matched = sum(1 for h in sample_hits if h['matched_b_or_m'])
        num_any_matched = sum(1 for h in sample_hits if h['matched_any'])

        diag_unique_total = len(diag_to_hit)
        diag_unique_subtype = sum(1 for v in diag_to_hit.values() if v['matched_subtype'])
        diag_unique_bm = sum(1 for v in diag_to_hit.values() if v['matched_b_or_m'])
        diag_unique_any = sum(1 for v in diag_to_hit.values() if v['matched_any'])

        per_dataset_details[ds] = {
            'status': 'ok',
            'samples': num_samples,
            'sample_matched_subtype': num_subtype_matched,
            'sample_matched_b_or_m': num_bm_matched,
            'sample_matched_any': num_any_matched,
            'sample_unmatched_ratio_any': float((num_samples - num_any_matched) / num_samples) if num_samples else 0.0,
            'sample_unmatched_ratio_subtype': float((num_samples - num_subtype_matched) / num_samples) if num_samples else 0.0,
            'sample_unmatched_ratio_b_or_m': float((num_samples - num_bm_matched) / num_samples) if num_samples else 0.0,
            'diagnosis_unique_total': diag_unique_total,
            'diagnosis_unique_matched_subtype': diag_unique_subtype,
            'diagnosis_unique_matched_b_or_m': diag_unique_bm,
            'diagnosis_unique_matched_any': diag_unique_any,
            'diagnosis_unique_unmatched_ratio_any': float((diag_unique_total - diag_unique_any) / diag_unique_total) if diag_unique_total else 0.0,
            'diagnosis_unique_unmatched_ratio_subtype': float((diag_unique_total - diag_unique_subtype) / diag_unique_total) if diag_unique_total else 0.0,
            'diagnosis_unique_unmatched_ratio_b_or_m': float((diag_unique_total - diag_unique_bm) / diag_unique_total) if diag_unique_total else 0.0,
            'diagnosis_unique_details': diag_to_hit,
        }

        # Overall aggregation
        overall_samples += num_samples
        overall_subtype_match += num_subtype_matched
        overall_bm_match += num_bm_matched
        overall_any_match += num_any_matched
        for k in diag_to_hit.keys():
            overall_diag_unique.add((ds, k))
        overall_diag_unique_subtype_match += diag_unique_subtype
        overall_diag_unique_bm_match += diag_unique_bm
        overall_diag_unique_any_match += diag_unique_any

    # Build report
    overall_diag_unique_total = len(overall_diag_unique)
    report = {
        'overall': {
            'samples_total': overall_samples,
            'sample_matched_subtype': overall_subtype_match,
            'sample_matched_b_or_m': overall_bm_match,
            'sample_matched_any': overall_any_match,
            'sample_unmatched_ratio_any': float((overall_samples - overall_any_match) / overall_samples) if overall_samples else 0.0,
            'sample_unmatched_ratio_subtype': float((overall_samples - overall_subtype_match) / overall_samples) if overall_samples else 0.0,
            'sample_unmatched_ratio_b_or_m': float((overall_samples - overall_bm_match) / overall_samples) if overall_samples else 0.0,
            'diagnosis_unique_total': overall_diag_unique_total,
            'diagnosis_unique_matched_subtype': overall_diag_unique_subtype_match,
            'diagnosis_unique_matched_b_or_m': overall_diag_unique_bm_match,
            'diagnosis_unique_matched_any': overall_diag_unique_any_match,
            'diagnosis_unique_unmatched_ratio_any': float((overall_diag_unique_total - overall_diag_unique_any_match) / overall_diag_unique_total) if overall_diag_unique_total else 0.0,
            'diagnosis_unique_unmatched_ratio_subtype': float((overall_diag_unique_total - overall_diag_unique_subtype_match) / overall_diag_unique_total) if overall_diag_unique_total else 0.0,
            'diagnosis_unique_unmatched_ratio_b_or_m': float((overall_diag_unique_total - overall_diag_unique_bm_match) / overall_diag_unique_total) if overall_diag_unique_total else 0.0,
        },
        'per_dataset': per_dataset_details,
    }

    # Console summary
    print("=" * 80)
    print("RL diagnosis hit report")
    print("=" * 80)
    for ds, info in per_dataset_details.items():
        if info.get('status') != 'ok':
            print(f"{ds:12s} -> missing/empty")
            continue
        print(f"{ds:12s} -> samples={info['samples']}, any_hit={info['sample_matched_any']} ({1.0 - info['sample_unmatched_ratio_any']:.2%}), "
              f"diag_unique={info['diagnosis_unique_total']}, any_hit_unique={info['diagnosis_unique_matched_any']} ({1.0 - info['diagnosis_unique_unmatched_ratio_any']:.2%})")
    print("-" * 80)
    ov = report['overall']
    print(f"OVERALL    -> samples={ov['samples_total']}, any_hit={ov['sample_matched_any']} ({1.0 - ov['sample_unmatched_ratio_any']:.2%}), "
          f"diag_unique={ov['diagnosis_unique_total']}, any_hit_unique={ov['diagnosis_unique_matched_any']} ({1.0 - ov['diagnosis_unique_unmatched_ratio_any']:.2%})")

    # Save JSON report
    out_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'RL_diagnosis_hits.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed report to: {out_path}")


if __name__ == '__main__':
    main()


