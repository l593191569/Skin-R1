import os
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict

def load_taxonomy(json_file_path):
    """Load the taxonomy data."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_taxonomy(taxonomy_data):
    """Analyze the taxonomy structure."""
    analysis = {}

    # Basic statistics
    total_conditions = 0
    chapter_stats = {}

    for main_category, chapters in taxonomy_data.items():
        analysis[main_category] = {}
        for chapter, conditions in chapters.items():
            condition_count = len(conditions)
            total_conditions += condition_count
            chapter_stats[chapter] = condition_count
            analysis[main_category][chapter] = condition_count

    analysis["total_conditions"] = total_conditions
    analysis["total_chapters"] = len(chapter_stats)
    analysis["chapter_stats"] = chapter_stats

    return analysis

def create_hierarchical_visualization(taxonomy_data, output_dir, filename="taxonomy_tree"):
    """Create a vertical tree diagram with one leaf node per row."""
    # Use the total number of conditions to size the figure height
    total_conditions = sum(len(conditions) for conditions in taxonomy_data['Neoplasia'].values())
    fig_height = max(30, total_conditions * 0.2)  # more height per condition
    fig, ax = plt.subplots(1, 1, figsize=(18, fig_height))

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, 20))

    # Node positions
    nodes = []
    edges = []

    # Root node
    root_x, root_y = 0.15, 0.5
    nodes.append(('root', root_x, root_y, 'Neoplasia'))

    # Chapter node positions
    chapters = list(taxonomy_data['Neoplasia'].keys())
    chapter_count = len(chapters)
    chapter_y_spacing = 0.8 / max(chapter_count, 1)

    for i, chapter in enumerate(chapters):
        chapter_x = 0.4
        chapter_y = 0.1 + i * chapter_y_spacing
        nodes.append((f'chapter_{i}', chapter_x, chapter_y, chapter))
        edges.append(('root', f'chapter_{i}'))

        # Condition node positions - one condition per row
        conditions = list(taxonomy_data['Neoplasia'][chapter].keys())
        condition_count = len(conditions)
        if condition_count > 0:
            # Vertical spacing for all conditions in this chapter
            condition_y_spacing = 0.8 / total_conditions
            current_condition_index = 0

            # Count conditions in all previous chapters
            for prev_i in range(i):
                current_condition_index += len(list(taxonomy_data['Neoplasia'][list(chapters)[prev_i]].keys()))

            for j, condition in enumerate(conditions):
                condition_x = 0.7
                condition_y = 0.1 + (current_condition_index + j) * condition_y_spacing
                nodes.append((f'condition_{i}_{j}', condition_x, condition_y, condition))
                edges.append((f'chapter_{i}', f'condition_{i}_{j}'))

    # Draw edges
    for edge in edges:
        start_node = next(n for n in nodes if n[0] == edge[0])
        end_node = next(n for n in nodes if n[0] == edge[1])
        ax.plot([start_node[1], end_node[1]], [start_node[2], end_node[2]],
                'k-', alpha=0.4, linewidth=1.5)

    # Draw nodes
    for node_id, x, y, text in nodes:
        if node_id == 'root':
            # Root node
            circle = plt.Circle((x, y), 0.04, facecolor='lightblue', alpha=0.9, linewidth=2, edgecolor='darkblue')
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold')
        elif node_id.startswith('chapter'):
            # Chapter node
            circle = plt.Circle((x, y), 0.025, facecolor=colors[int(node_id.split('_')[1]) % len(colors)],
                              alpha=0.8, linewidth=1.5, edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        else:
            # Condition node
            circle = plt.Circle((x, y), 0.02, facecolor='lightyellow', alpha=0.7,
                              linewidth=1, edgecolor='gray')
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title("Taxonomy Tree", fontsize=18, fontweight='bold', pad=20)

    # Save the figure
    graph_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    return graph_path

def create_sunburst_visualization(taxonomy_data, output_dir, filename="taxonomy_sunburst"):
    """Create a three-level sunburst visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 18), subplot_kw=dict(projection='polar'))

    # Color palette
    main_colors = plt.cm.Set1(np.linspace(0, 1, 5))
    chapter_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Compute data
    angles = []
    radii = []
    labels = []
    color_list = []
    levels = []  # 0: main category, 1: chapter, 2: condition

    current_angle = 0

    # Level 1: main category
    for main_category, chapters in taxonomy_data.items():
        total_conditions = sum(len(conditions) for conditions in chapters.values())
        main_angle = 2 * np.pi * total_conditions / sum(sum(len(conditions) for conditions in ch.values()) for ch in taxonomy_data.values())

        # Main category
        angles.append(current_angle)
        radii.append(total_conditions)
        labels.append(f"Main Category:\n{main_category}\n({total_conditions} total)")
        color_list.append(main_colors[0])
        levels.append(0)

        # Level 2: chapters
        for i, (chapter, conditions) in enumerate(chapters.items()):
            chapter_angle = main_angle * len(conditions) / total_conditions
            condition_count = len(conditions)

            angles.append(current_angle)
            radii.append(condition_count)
            labels.append(f"Chapter:\n{chapter}\n({condition_count} conditions)")
            color_list.append(chapter_colors[i % len(chapter_colors)])
            levels.append(1)

            current_angle += chapter_angle

    # Draw the sunburst
    bars = ax.bar(angles, radii, width=0.15, bottom=0,
                  color=color_list, alpha=0.8, edgecolor='white', linewidth=2)

    # Labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9, ha='center')

    # Title
    ax.set_title("Taxonomy Three-Level Sunburst Chart", fontsize=18, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        patches.Patch(color=main_colors[0], label='Main Category'),
        patches.Patch(color=chapter_colors[0], label='Chapter')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Save the figure
    sunburst_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(sunburst_path, dpi=300, bbox_inches='tight')
    plt.close()

    return sunburst_path

def create_bar_chart_visualization(analysis, output_dir, filename="taxonomy_bar_chart"):
    """Create bar-chart visualizations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Per-chapter bar chart
    chapters = list(analysis["chapter_stats"].keys())
    counts = list(analysis["chapter_stats"].values())

    bars1 = ax1.bar(range(len(chapters)), counts,
                    color=plt.cm.viridis(np.linspace(0, 1, len(chapters))))

    ax1.set_xlabel('Chapters', fontsize=12)
    ax1.set_ylabel('Number of Conditions', fontsize=12)
    ax1.set_title('Conditions per Chapter', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(chapters)))
    ax1.set_xticklabels(chapters, rotation=45, ha='right')

    # Value labels
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=8)

    # Per-main-category statistics
    main_categories = []
    main_counts = []

    for main_category, chapters_data in analysis.items():
        if main_category not in ["total_conditions", "total_chapters", "chapter_stats"]:
            total = sum(chapters_data.values())
            main_categories.append(main_category)
            main_counts.append(total)

    bars2 = ax2.bar(main_categories, main_counts,
                    color=plt.cm.Set3(np.linspace(0, 1, len(main_categories))))

    ax2.set_xlabel('Main Categories', fontsize=12)
    ax2.set_ylabel('Total Conditions', fontsize=12)
    ax2.set_title('Conditions per Main Category', fontsize=14, fontweight='bold')

    # Value labels
    for bar, count in zip(bars2, main_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save the figure
    bar_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    return bar_path

def generate_taxonomy_report(analysis, output_dir, filename="taxonomy_report"):
    """Generate a taxonomy report."""
    report = f"""
# Taxonomy Analysis Report

## Basic statistics
- Total conditions: {analysis['total_conditions']}
- Total chapters: {analysis['total_chapters']}

## Per-main-category statistics
"""

    for main_category, chapters_data in analysis.items():
        if main_category not in ["total_conditions", "total_chapters", "chapter_stats"]:
            total = sum(chapters_data.values())
            report += f"- {main_category}: {total} conditions\n"

    report += "\n## Per-chapter statistics\n"
    for chapter, count in sorted(analysis["chapter_stats"].items(), key=lambda x: x[1], reverse=True):
        report += f"- {chapter}: {count} conditions\n"

    # Save the report
    report_path = os.path.join(output_dir, f"{filename}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report_path

def main():
    parser = argparse.ArgumentParser(description="Visualize the taxonomy tree")
    parser.add_argument("--input", type=str, required=True, help="Input taxonomy JSON file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--filename", type=str, default="taxonomy", help="Output filename prefix")

    args = parser.parse_args()

    # Check the input file
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Taxonomy visualization ===")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Filename prefix: {args.filename}")
    print()

    try:
        # Load the taxonomy data
        print("1. Loading taxonomy data...")
        taxonomy_data = load_taxonomy(args.input)
        print(f"   Loaded {len(taxonomy_data)} main categories")

        # Analyze
        print("2. Analyzing the taxonomy structure...")
        analysis = analyze_taxonomy(taxonomy_data)
        print(f"   Total conditions: {analysis['total_conditions']}")
        print(f"   Total chapters: {analysis['total_chapters']}")

        # Hierarchical visualization
        print("3. Creating the hierarchical visualization...")
        tree_path = create_hierarchical_visualization(taxonomy_data, args.output_dir, args.filename)
        print(f"   Tree diagram saved to: {tree_path}")

        # Sunburst visualization
        print("4. Creating the sunburst visualization...")
        sunburst_path = create_sunburst_visualization(taxonomy_data, args.output_dir, f"{args.filename}_sunburst")
        print(f"   Sunburst saved to: {sunburst_path}")

        # Bar-chart visualization
        print("5. Creating the bar-chart visualization...")
        bar_path = create_bar_chart_visualization(analysis, args.output_dir, f"{args.filename}_bar")
        print(f"   Bar chart saved to: {bar_path}")

        # Report
        print("6. Generating the analysis report...")
        report_path = generate_taxonomy_report(analysis, args.output_dir, args.filename)
        print(f"   Report saved to: {report_path}")

        print()
        print("Taxonomy visualization complete.")
        print(f"Tree diagram: {tree_path}")
        print(f"Sunburst: {sunburst_path}")
        print(f"Bar chart: {bar_path}")
        print(f"Report: {report_path}")

    except Exception as e:
        print(f"Processing failed: {e}")
        return

if __name__ == "__main__":
    main()
