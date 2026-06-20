from collections import defaultdict
import json
import os
from copy import deepcopy
from typing import Any, Dict, Set, Tuple


def load_synonym_subtype_maps(json_path: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Load synonym_map and subtype_map from synonym_and_subtype2.json (lowercased keys)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_syn = data.get("synonym_map", {})
    synonym_map = {
        str(k).lower().strip(): str(v).lower().strip()
        for k, v in raw_syn.items()
    }

    raw_sub = data.get("subtype_map", {})
    subtype_map: Dict[str, Set[str]] = {}
    for parent, children in raw_sub.items():
        parent_l = str(parent).lower().strip()
        if isinstance(children, (list, tuple, set)):
            subtype_map[parent_l] = {str(c).lower().strip() for c in children}
        else:
            subtype_map[parent_l] = set()

    return synonym_map, subtype_map


def normalize_graph(data, synonym_map=None, subtype_map=None):
    """
    data: dict with keys: nodes(list[str]), edges(list[[u,v]]), node_attributes(dict[name]->{"degree":int})
    synonym_map: dict[str->str], map variant/synonym -> canonical
    subtype_map: dict[parent -> set(children)], only used to PUBLISH a structure; we DO NOT merge subtypes.
    """
    data = deepcopy(data)
    nodes = set(data.get("nodes", []))
    edges = [tuple(e) for e in data.get("edges", [])]

    synonym_map = synonym_map or {}

    def canon(name):
        name_l = str(name).lower().strip()
        return synonym_map.get(name_l, name_l)

    new_edges_set = set()
    for u, v in edges:
        cu, cv = canon(u), canon(v)
        if cu == "" or cv == "":
            continue
        if cu == cv:
            continue
        new_edges_set.add(tuple(sorted((cu, cv))))

    new_edges = sorted(list(new_edges_set))

    deg = defaultdict(int)
    for u, v in new_edges:
        deg[u] += 1
        deg[v] += 1

    mapped_nodes = {canon(n) for n in nodes}
    edge_nodes = set()
    for u, v in new_edges:
        edge_nodes.add(u)
        edge_nodes.add(v)
    new_nodes = sorted(mapped_nodes.union(edge_nodes))

    new_node_attributes = {n: {"degree": int(deg.get(n, 0))} for n in new_nodes}

    subtype_map = subtype_map or {}
    present_subtypes = {}
    node_set = set(new_nodes)

    for parent, children in subtype_map.items():
        c_parent = canon(parent)
        c_children = sorted({
            canon(c) for c in children
            if canon(c) in node_set and canon(c) != c_parent
        })
        if c_children:
            present_subtypes[c_parent] = c_children

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "node_attributes": new_node_attributes,
        "subtype_structure": present_subtypes,
    }


if __name__ == "__main__":
    import argparse

    _here = os.path.dirname(os.path.abspath(__file__))
    default_synonyms = os.path.join(_here, "synonym_and_subtype2.json")

    parser = argparse.ArgumentParser(description="Merge synonyms and subtypes in the DDx graph")
    parser.add_argument("--input", type=str, default="ddx_graph.json",
                        help="Input DDx graph JSON file path")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--prefix", type=str, default="ddx_graph_merged",
                        help="Output filename prefix")
    parser.add_argument(
        "--synonyms",
        type=str,
        default=default_synonyms,
        help="synonym_and_subtype2.json (single source for synonym_map + subtype_map)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        exit(1)
    if not os.path.exists(args.synonyms):
        print(f"Synonym file not found: {args.synonyms}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== DDx graph synonym merge ===")
    print(f"Input file: {args.input}")
    print(f"Synonym/subtype config: {args.synonyms}")
    print(f"Output directory: {args.output_dir}")
    print()

    synonym_map, subtype_map = load_synonym_subtype_maps(args.synonyms)

    try:
        print("1. Loading DDx graph data...")
        with open(args.input, encoding="utf-8") as f:
            graph = json.load(f)

        print(f"   Original nodes: {len(graph['nodes'])}")
        print(f"   Original edges: {len(graph['edges'])}")

        print("2. Applying synonym and subtype mappings...")
        new_graph = normalize_graph(graph, synonym_map, subtype_map)

        print(f"   Merged nodes: {len(new_graph['nodes'])}")
        print(f"   Merged edges: {len(new_graph['edges'])}")

        output_path = os.path.join(args.output_dir, f"{args.prefix}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_graph, f, indent=2, ensure_ascii=False)

        print(f"\nMerged graph saved to: {output_path}")

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
