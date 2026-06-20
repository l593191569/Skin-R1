#!/bin/bash

# Working directory = data_construction/; unified data directory = <repo>/data.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}

# RUN_DIR points at the bbc_<RUN_ID> directory under data/outputs/.
RUN_DIR=${RUN_DIR:-$DATA_DIR/outputs/run}
INPUT_CSV="${RUN_DIR}/pdf_outputs.matched_with_llm.csv"
TOC_FILE="table_of_contents_part20.json"

# Resolve the directory of the input CSV.
CSV_DIR=$(dirname "$INPUT_CSV")
OUTPUT_TAXONOMY="$CSV_DIR/taxonomy_tree.json"
OUTPUT_CSV="$CSV_DIR/augmented_data_with_taxonomy.csv"

# Check the input files.
if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Input CSV not found: $INPUT_CSV"
    exit 1
fi

if [[ ! -f "$TOC_FILE" ]]; then
    echo "Table-of-contents file not found: $TOC_FILE"
    exit 1
fi

# Generate the taxonomy.
echo "Generating the taxonomy..."
echo "Output directory: $CSV_DIR"
python taxonomy_generate.py \
    --input "$INPUT_CSV" \
    --toc "$TOC_FILE" \
    --output-taxonomy "$OUTPUT_TAXONOMY" \
    --output-csv "$OUTPUT_CSV"

echo "Taxonomy generation complete"
