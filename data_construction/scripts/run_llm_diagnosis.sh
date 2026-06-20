#!/bin/bash

# LLM diagnosis-rule extraction.
# Reads the matched CSV from a run directory and rephrases diagnostic rules with an LLM.

# Working directory = data_construction/; unified data directory = <repo>/data.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}

# RUN_DIR points at the bbc_<RUN_ID> directory created under data/outputs/.
LATEST_BBC_DIR="${RUN_DIR:-$DATA_DIR/outputs/run}"
INPUT_CSV="${LATEST_BBC_DIR}/pdf_outputs.matched.csv"
OUTPUT_CSV="${LATEST_BBC_DIR}/pdf_outputs.matched_with_llm.csv"

# Check that the input file exists.
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: input file not found: $INPUT_CSV"
    echo "Check that the run directory exists: $LATEST_BBC_DIR"
    exit 1
fi

echo "=== LLM diagnosis-rule extraction ==="
echo "Input: $INPUT_CSV"
echo "Output: $OUTPUT_CSV"
echo ""

# Check for the .env file with the API key.
if [ ! -f ".env" ]; then
    echo "Warning: .env not found; make sure OPENAI_API_KEY is set in the environment."
    echo "Or create a .env file containing: OPENAI_API_KEY=your_api_key_here"
fi

# Run the LLM processing.
echo "Running LLM processing..."
python3 llm_diagnosis_rephrase.py \
    --csv "$INPUT_CSV" \
    --output "$OUTPUT_CSV" \
    --model "gpt-4.1-mini" \
    --max-rows 1000  # adjust or remove this to process all rows

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Done ==="
    echo "Result saved to: $OUTPUT_CSV"
    echo ""
    echo "File info:"
    ls -lh "$OUTPUT_CSV"
else
    echo "Error: LLM processing failed"
    exit 1
fi
