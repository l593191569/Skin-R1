#!/bin/bash

# Working directory = data_construction/; unified data directory = <repo>/data.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}

# RUN_DIR points at the bbc_<RUN_ID> directory under data/outputs/.
RUN_DIR=${RUN_DIR:-$DATA_DIR/outputs/run}
INPUT_JSON="${RUN_DIR}/differential_diagnosis.json"
OUTPUT_JSON="${RUN_DIR}/ddx_extracted.json"

# Resolve the directory of the input JSON.
JSON_DIR=$(dirname "$INPUT_JSON")
OUTPUT_JSON="$JSON_DIR/ddx_extracted.json"

# Check the input file.
if [[ ! -f "$INPUT_JSON" ]]; then
    echo "Input JSON not found: $INPUT_JSON"
    exit 1
fi

# Skip the LLM step if the DDx extraction already exists.
if [[ -f "$OUTPUT_JSON" ]]; then
    echo "Found existing DDx extraction: $OUTPUT_JSON"
    echo "Skipping the LLM DDx step and going straight to graph construction..."
else
    # Check for the .env file with the API key.
    if [[ ! -f ".env" ]]; then
        echo ".env not found; make sure it contains OPENAI_API_KEY"
        exit 1
    fi

    # Run DDx extraction.
    echo "Extracting DDx information..."
    echo "Output directory: $JSON_DIR"
    python llm_DDx_rephrase.py \
        --input "$INPUT_JSON" \
        --output "$OUTPUT_JSON" \
        --model "gpt-4.1-mini" \
        --max-items 500

    echo "DDx extraction complete"
fi

# Build the DDx graph.
echo ""
echo "Building the DDx graph..."
python ddx_to_graph.py \
    --input "$OUTPUT_JSON" \
    --output-dir "$JSON_DIR" \
    --filename "ddx_graph"

echo "DDx graph built"

# Merge synonym nodes -> ddx_graph_merged.json (needed by SkinRationale generator).
echo ""
echo "Merging synonym nodes in the DDx graph..."
python diagnosis_merge.py \
    --input "$JSON_DIR/ddx_graph.json" \
    --output-dir "$JSON_DIR" \
    --synonyms "$SCRIPT_DIR/../synonym_and_subtype2.json"

echo "DDx graph merge complete: $JSON_DIR/ddx_graph_merged.json"
