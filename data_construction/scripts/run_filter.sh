#!/bin/bash

# Interactive image curation + CSV refinement.
# Run from data_construction/ (this script cd's there automatically).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}
RUN_DIR=${RUN_DIR:-$DATA_DIR/outputs/run}
# filter_and_visualize_data.py / refine_csv.py read/write the paths below.
export RUN_DIR
export FILTER_GALLERY_HTML="$RUN_DIR/image_gallery.html"
export FILTER_OUTPUT_CSV="$RUN_DIR/filtered_final_data.csv"

echo "=========================================="
echo "Image curation and refinement tool"
echo "=========================================="

# Check that we are in the right directory.
if [ ! -f "filter_and_visualize_data.py" ]; then
    echo "Error: run this script from the data_construction directory"
    exit 1
fi

# Check the Python environment.
if ! command -v python &> /dev/null; then
    echo "Error: python not found"
    exit 1
fi

# Check that pandas is installed.
python -c "import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pandas not installed; run: pip install pandas"
    exit 1
fi

echo "Environment check passed."
echo ""

# Menu.
echo "Choose an action:"
echo "1. Generate the image gallery page (step 1)"
echo "2. Apply the exclusion list and produce the final CSV (step 2)"
echo "3. Show usage notes"
echo "4. Exit"
echo ""

read -p "Enter an option (1-4): " choice

case $choice in
    1)
        echo "Generating the image gallery page..."
        python filter_and_visualize_data.py
        echo ""
        echo "Done. Open $FILTER_GALLERY_HTML in a browser to curate images."
        ;;
    2)
        if [ ! -f "excluded_images.json" ]; then
            echo "Error: excluded_images.json not found"
            echo "Run step 1 first, then curate images in the browser to export it."
            exit 1
        fi
        echo "Applying the exclusion list and producing the final CSV..."
        python filter_and_visualize_data.py --process-exclusion
        echo ""
        echo "Done. Filtered CSV: $FILTER_OUTPUT_CSV"
        echo "Refining into refined_data.csv (splitting rule / diagnosis)..."
        python refine_csv.py --input "$FILTER_OUTPUT_CSV" --output "$RUN_DIR/refined_data.csv"
        echo "Done. refined_data.csv: $RUN_DIR/refined_data.csv"
        ;;
    3)
        echo ""
        echo "=== Usage ==="
        echo "See data_construction/README.md (Stage 3: image curation + refine)."
        ;;
    4)
        echo "Exiting"
        exit 0
        ;;
    *)
        echo "Invalid option; please re-run the script"
        exit 1
        ;;
esac
