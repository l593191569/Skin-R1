import os
import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
import numpy as np

def load_ddx_data(json_file_path):
    """Load DDx data."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_ddx_graph(ddx_data):
    """Build the DDx graph."""
    G = nx.DiGraph()

    # Counters
    total_edges = 0
    unique_subjects = set()
    unique_ddx = set()

    # Process each DDx result
    for result in ddx_data["results"]:
        ddx_result = result["ddx_result"]

        if ddx_result.get("is_valid", False):
            subject = ddx_result.get("subject")
            ddx_list = ddx_result.get("ddx_list", [])

            if subject and ddx_list:
                # Lowercase everything
                subject_lower = subject.lower() if subject else ""
                ddx_list_lower = [ddx.lower() for ddx in ddx_list if ddx]

                unique_subjects.add(subject_lower)

                # Add nodes and edges
                for ddx in ddx_list_lower:
                    unique_ddx.add(ddx)
                    G.add_edge(subject_lower, ddx)
                    total_edges += 1

    return G, {
        "total_edges": total_edges,
        "unique_subjects": len(unique_subjects),
        "unique_ddx": len(unique_ddx),
        "total_nodes": len(unique_subjects | unique_ddx)
    }

def analyze_graph(G):
    """Analyze the graph structure."""
    analysis = {}

    # Basic statistics
    analysis["nodes"] = G.number_of_nodes()
    analysis["edges"] = G.number_of_edges()

    # Degree analysis
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Most common DDx (highest in-degree nodes)
    top_ddx = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    analysis["top_ddx"] = top_ddx

    # Most common subjects (highest out-degree nodes)
    top_subjects = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    analysis["top_subjects"] = top_subjects

    # Connectivity analysis
    analysis["connected_components"] = nx.number_strongly_connected_components(G)
    analysis["weakly_connected_components"] = nx.number_weakly_connected_components(G)

    return analysis

def visualize_graph(G, output_dir, filename="ddx_graph"):
    """Visualize the graph."""
    plt.figure(figsize=(16, 12))

    # Spring layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

    # Node sizes (based on degree)
    node_sizes = [max(100, G.degree(node) * 50) for node in G.nodes()]

    # Node colors (based on degree)
    degrees = [G.degree(node) for node in G.nodes()]
    max_degree = max(degrees) if degrees else 1
    node_colors = [plt.cm.viridis(d / max_degree) for d in degrees]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=10,
                          alpha=0.6,
                          width=1)

    # Draw labels (only for important nodes)
    important_nodes = [node for node in G.nodes() if G.degree(node) >= 2]
    labels = {node: node for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title("Differential Diagnosis Graph", fontsize=16, fontweight='bold')
    plt.axis('off')

    # Save the figure
    graph_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    return graph_path

def save_graph_data(G, output_dir, filename="ddx_graph"):
    """Save the graph data."""
    # Save as GML
    gml_path = os.path.join(output_dir, f"{filename}.gml")
    nx.write_gml(G, gml_path)

    # Save as JSON
    graph_data = {
        "nodes": list(G.nodes()),
        "edges": list(G.edges()),
        "node_attributes": {node: {"degree": G.degree(node)} for node in G.nodes()}
    }

    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    return gml_path, json_path

def generate_graph_report(analysis, output_dir, filename="ddx_graph_report"):
    """Generate a graph analysis report."""
    report = f"""
# DDx Graph Analysis Report

## Basic statistics
- Total nodes: {analysis['nodes']}
- Total edges: {analysis['edges']}
- Strongly connected components: {analysis['connected_components']}
- Weakly connected components: {analysis['weakly_connected_components']}

## Most common DDx (highest in-degree nodes)
"""

    for i, (node, degree) in enumerate(analysis['top_ddx'], 1):
        report += f"{i}. {node}: {degree}\n"

    report += "\n## Most common subjects (highest out-degree nodes)\n"
    for i, (node, degree) in enumerate(analysis['top_subjects'], 1):
        report += f"{i}. {node}: {degree}\n"

    # Save the report
    report_path = os.path.join(output_dir, f"{filename}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report_path

def main():
    parser = argparse.ArgumentParser(description="Convert DDx data into a graph structure")
    parser.add_argument("--input", type=str, required=True, help="Input DDx JSON file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--filename", type=str, default="ddx_graph", help="Output filename prefix")

    args = parser.parse_args()

    # Check the input file
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== DDx graph conversion ===")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Filename prefix: {args.filename}")
    print()

    try:
        # Load DDx data
        print("1. Loading DDx data...")
        ddx_data = load_ddx_data(args.input)
        print(f"   Loaded {len(ddx_data['results'])} results")

        # Build the graph
        print("2. Building the DDx graph...")
        G, stats = build_ddx_graph(ddx_data)
        print(f"   Nodes: {stats['total_nodes']}")
        print(f"   Edges: {stats['total_edges']}")
        print(f"   Unique subjects: {stats['unique_subjects']}")
        print(f"   Unique DDx: {stats['unique_ddx']}")

        # Analyze the graph
        print("3. Analyzing the graph structure...")
        analysis = analyze_graph(G)
        print(f"   Connected components: {analysis['connected_components']}")

        # Visualize
        print("4. Generating visualization...")
        graph_path = visualize_graph(G, args.output_dir, args.filename)
        print(f"   Figure saved to: {graph_path}")

        # Save graph data
        print("5. Saving graph data...")
        gml_path, json_path = save_graph_data(G, args.output_dir, args.filename)
        print(f"   GML: {gml_path}")
        print(f"   JSON: {json_path}")

        # Generate report
        print("6. Generating analysis report...")
        report_path = generate_graph_report(analysis, args.output_dir, args.filename)
        print(f"   Report saved to: {report_path}")

        print()
        print("Graph conversion complete.")
        print(f"Figure: {graph_path}")
        print(f"Data: {json_path}")
        print(f"Report: {report_path}")

    except Exception as e:
        print(f"Processing failed: {e}")
        return

if __name__ == "__main__":
    main()
