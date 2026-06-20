import pandas as pd
import json
import re
import argparse
import os
from pathlib import Path

def get_chapter_title(page_num, chapters, page_offset=1798):
    """Map a page number to a chapter title.

    CSV page numbers are relative (starting from 1); convert them to the
    absolute page numbers used in the table of contents by adding page_offset.
    For the reference Fitzpatrick textbook the absolute pages start at 1799,
    so the default offset is 1798 (1799 - 1). Adjust --page-offset for a
    different source document.
    """
    absolute_page_num = page_num + page_offset

    for ch in chapters:
        if ch["start_page"] <= absolute_page_num < ch["end_page"]:
            return ch["title"]
    return "Unknown Chapter"

def extract_diagnosis_info(rephrase):
    """Extract rule and result from LLM_rephrase"""
    if "NOT A RULE" in rephrase.upper():
        return None, None  # Skip invalid

    rule_match = re.search(r"Rule:\s*(.+?)\s*\|", rephrase)
    result_match = re.search(r"Result:\s*(.+)", rephrase)

    if rule_match and result_match:
        rule = rule_match.group(1).strip()
        result = result_match.group(1).strip()
        if rule.upper() == "NOT A RULE":
            return None, None
        return rule, result

    return None, None

def process_and_build_taxonomy(df, chapters, page_offset=1798):
    """Process dataframe and build taxonomy + paths"""
    taxonomy = {"Neoplasia": {}}
    taxonomy_paths = []

    for _, row in df.iterrows():
        taxonomy_path = ""  # Default to empty if not matched

        rephrase = str(row.get("LLM_rephrase", ""))
        page_num = row.get("image_page_num")

        try:
            page_num = int(page_num)
            rule, result = extract_diagnosis_info(rephrase)
            if rule and result:
                chapter_title = get_chapter_title(page_num, chapters, page_offset)

                # Add to taxonomy tree
                if chapter_title not in taxonomy["Neoplasia"]:
                    taxonomy["Neoplasia"][chapter_title] = {}

                taxonomy["Neoplasia"][chapter_title][result] = result

                taxonomy_path = f"Neoplasia > {chapter_title} > {result}"
        except:
            pass  # silently skip bad rows

        taxonomy_paths.append(taxonomy_path)  # Always append to maintain alignment

    df["taxonomy_path"] = taxonomy_paths
    return taxonomy, df

def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy from LLM rephrased data")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--toc", type=str, required=True, help="Table of contents JSON file path")
    parser.add_argument("--output-taxonomy", type=str, default="taxonomy_tree.json", help="Output taxonomy JSON file path")
    parser.add_argument("--output-csv", type=str, default="augmented_data_with_taxonomy.csv", help="Output augmented CSV file path")
    parser.add_argument("--page-offset", type=int, default=1798,
                        help="Offset added to CSV (relative) page numbers to get absolute pages (default: 1798)")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.input):
        print(f"Error: Input CSV file '{args.input}' not found")
        return

    if not os.path.exists(args.toc):
        print(f"Error: Table of contents file '{args.toc}' not found")
        return

    print("=== Taxonomy generation ===")
    print(f"Input file: {args.input}")
    print(f"Table of contents: {args.toc}")
    print(f"Taxonomy output: {args.output_taxonomy}")
    print(f"CSV output: {args.output_csv}")
    print()

    try:
        # Load chapters from JSON
        print("1. Loading the table of contents...")
        with open(args.toc, "r") as f:
            chapters = json.load(f)
        print(f"   Loaded {len(chapters)} chapters")

        # Load input CSV
        print("2. Loading the CSV file...")
        df = pd.read_csv(args.input)
        print(f"   Loaded {len(df)} rows")

        # Process and build taxonomy
        print("3. Processing data and building the taxonomy...")
        taxonomy_tree, df_augmented = process_and_build_taxonomy(df, chapters, args.page_offset)

        # Count valid entries
        valid_entries = df_augmented['taxonomy_path'].str.len() > 0
        valid_count = valid_entries.sum()
        print(f"   Generated {valid_count} valid taxonomy paths")

        # Save taxonomy tree
        print("4. Saving the taxonomy tree...")
        with open(args.output_taxonomy, "w") as f:
            json.dump(taxonomy_tree, f, indent=2)

        # Save augmented CSV
        print("5. Saving the augmented CSV...")
        df_augmented.to_csv(args.output_csv, index=False)

        print()
        print("Done.")
        print(f"Taxonomy tree saved to: {args.output_taxonomy}")
        print(f"Augmented CSV saved to: {args.output_csv}")

        # Show some statistics
        print()
        print("=== Statistics ===")
        print(f"Total rows: {len(df_augmented)}")
        print(f"Valid taxonomy paths: {valid_count}")
        print(f"Invalid entries: {len(df_augmented) - valid_count}")

        # Show taxonomy structure
        print()
        print("=== Taxonomy structure ===")
        for chapter, conditions in taxonomy_tree["Neoplasia"].items():
            print(f"{chapter}: {len(conditions)} conditions")
            for condition in list(conditions.keys())[:3]:  # Show first 3
                print(f"  - {condition}")
            if len(conditions) > 3:
                print(f"  ... and {len(conditions) - 3} more conditions")

    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
