#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 0 (part A): PDF → extract → featurize → cluster, then STOP for manual
# cluster review. Does not filter or match — you choose exclude labels next.
#
# After this script:
#   1. Inspect:  $RUN_DIR/pdf_outputs.clustering/kmeans_label_lower.csv
#   2. Continue: bash scripts/run_stage0_post_cluster.sh <source_run> "<labels>"
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=_run_dir_hint.sh
source "$SCRIPT_DIR/_run_dir_hint.sh"

mkdir -p logs/output logs/error

export MONET_SRC=${MONET_SRC:-MONET-main/src}
export PYTHONPATH=$MONET_SRC:${PYTHONPATH:-}

DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}
PDF_NAME=${PDF_NAME:-"Fitzpatrick's dermatology NEOPLASIA"}
PDF_FILE="$DATA_DIR/pdfs/${PDF_NAME}.pdf"

if [[ ! -f "$PDF_FILE" ]]; then
  echo "Error: PDF not found: $PDF_FILE"
  echo "Place your textbook PDF there (see data/README.md)."
  exit 1
fi

RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUT_ROOT="$DATA_DIR/outputs/bbc_$RUN_ID"
mkdir -p "$OUT_ROOT"

TEMP_DIR="$OUT_ROOT/temp_pdf_dir"
mkdir -p "$TEMP_DIR"
cp "$PDF_FILE" "$TEMP_DIR/"

OUTPUT_DIR="$OUT_ROOT/pdf_outputs"
HDF5_PATH="$OUT_ROOT/pdf_outputs.compact.hdf5"

echo "Processing PDF: $PDF_FILE"
echo "Output directory: $OUT_ROOT"

echo "[1/4] Extracting images and text..."
python3 "$MONET_SRC/MONET/preprocess/pdf_extract.py" \
  --input "$TEMP_DIR" --output "$OUTPUT_DIR" --thread 1

echo "[2/4] Packing images into HDF5..."
python3 "$MONET_SRC/MONET/preprocess/glob_files.py" \
  --input "$OUTPUT_DIR" --output "$HDF5_PATH" \
  --field images --binary --style slash_to_underscore --extension .jpg,.jpeg,.png,.png

echo "[3/4] Featurizing images..."
python3 "$MONET_SRC/MONET/preprocess/featurize.py" \
  --input "$HDF5_PATH" \
  --output "$OUT_ROOT/pdf_outputs.featurized.pt" \
  --device "${DEVICE:-cuda}"

echo "[4/4] Clustering images..."
python3 "$MONET_SRC/MONET/preprocess/cluster.py" \
  --input "$HDF5_PATH" \
  --featurized-file "$OUT_ROOT/pdf_outputs.featurized.pt" \
  --output "$OUT_ROOT/pdf_outputs.clustering" \
  --pca --feature-to-use efficientnet -n1 4 -n2 4

rm -rf "$TEMP_DIR"

SOURCE_NAME="$(basename "$OUT_ROOT")"
write_run_dir_hint "$OUT_ROOT"

echo "Manual step — review cluster labels:"
echo "  less $OUT_ROOT/pdf_outputs.clustering/kmeans_label_lower.csv"
echo ""
echo "Then re-filter and finish Stage 0 with:"
echo "  bash scripts/run_stage0_post_cluster.sh ${SOURCE_NAME} \"02_01 02_03 01 00 03\""
echo "(Replace the label codes with your own choices.)"
