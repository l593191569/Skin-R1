#!/bin/bash
# Optional SLURM headers (only used when submitting with `sbatch`; ignored by `bash`).
# Set --partition / --account / --gres to match your cluster.
#SBATCH --job-name=bbc_pdf_process
#SBATCH --output=logs/output/bbc_pdf_%j.out
#SBATCH --error=logs/error/bbc_pdf_%j.err
#SBATCH --partition=<your_partition>
#SBATCH --account=<your_account>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Work from the data_construction/ directory (parent of scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Create log directories.
mkdir -p logs/output logs/error

# Activate your python environment here, e.g.:
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate derm1m

# Full PDF -> image-text pair extraction pipeline (MONET-style).
# Steps: extract images & text -> pack images into HDF5 -> featurize ->
#        cluster -> filter -> image-text matching -> export CSV and image paths.
# Note: install the data-construction requirements first (pip install -r requirements.txt).

# MONET package path (vendored under data_construction/MONET-main).
export MONET_SRC=${MONET_SRC:-MONET-main/src}
export PYTHONPATH=$MONET_SRC:${PYTHONPATH:-}

# Unified data directory (default <repo>/data; holds the source PDF and all outputs).
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}

# Source PDF lives under data/pdfs/.
PDF_NAME=${PDF_NAME:-"Fitzpatrick's dermatology NEOPLASIA"}
PDF_FILE="$DATA_DIR/pdfs/${PDF_NAME}.pdf"

# Create a unique, timestamped output subdirectory under data/outputs/.
RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUT_ROOT="$DATA_DIR/outputs/bbc_$RUN_ID"
mkdir -p "$OUT_ROOT"

# Stage the target PDF in a temporary directory (MONET expects a directory input).
TEMP_DIR="$OUT_ROOT/temp_pdf_dir"
mkdir -p "$TEMP_DIR"
cp "$PDF_FILE" "$TEMP_DIR/"

OUTPUT_DIR="$OUT_ROOT/pdf_outputs"
HDF5_PATH="$OUT_ROOT/pdf_outputs.compact.hdf5"
MATCHED_PATH="$OUT_ROOT/pdf_outputs.matched.csv"
CONFIG_PATH="pdf_files.config.json"  # matching config shipped in data_construction/

# Remove stale files just in case.
rm -f "$HDF5_PATH"
rm -f "${MATCHED_PATH%.csv}_image_paths.pkl"

echo "Processing PDF: $PDF_FILE"
echo "Output directory: $OUT_ROOT"
echo "Temp directory: $TEMP_DIR"

# 1. Extract images and text.
echo "[1/7] Extracting images and text from the PDF..."
python3 $MONET_SRC/MONET/preprocess/pdf_extract.py \
    --input "$TEMP_DIR" \
    --output "$OUTPUT_DIR" \
    --thread 1

# 2. Pack images into a single HDF5 file.
echo "[2/7] Packing images into HDF5..."
python3 $MONET_SRC/MONET/preprocess/glob_files.py \
    --input "$OUTPUT_DIR" \
    --output "$HDF5_PATH" \
    --field images \
    --binary \
    --style slash_to_underscore \
    --extension .jpg,.jpeg,.png,.png

# 3. Featurize images.
echo "[3/7] Extracting image features..."
python3 $MONET_SRC/MONET/preprocess/featurize.py \
    --input "$HDF5_PATH" \
    --output "$OUT_ROOT/pdf_outputs.featurized.pt" \
    --device "${DEVICE:-cuda}"

# 4. Cluster images.
echo "[4/7] Clustering images..."
python3 $MONET_SRC/MONET/preprocess/cluster.py \
    --input "$HDF5_PATH" \
    --featurized-file "$OUT_ROOT/pdf_outputs.featurized.pt" \
    --output "$OUT_ROOT/pdf_outputs.clustering" \
    --pca \
    --feature-to-use efficientnet \
    -n1 4 -n2 4

# 5. Filter out non-skin images by cluster label.
# Inspect pdf_outputs.clustering/kmeans_label_lower.csv and adjust --exclude-label.
echo "[5/7] Filtering out non-skin images..."
python3 $MONET_SRC/MONET/preprocess/filter.py \
    --input "$HDF5_PATH" \
    --label-file "$OUT_ROOT/pdf_outputs.clustering/kmeans_label_lower.csv" \
    --exclude-label 02_01 02_03 01 00 03 \
    --output "$OUT_ROOT/pdf_outputs.dermonly.hdf5"

# Continue the pipeline with the filtered HDF5.
HDF5_PATH="$OUT_ROOT/pdf_outputs.dermonly.hdf5"

# 6. Image-text matching.
echo "[6/7] Matching images with text..."
python3 $MONET_SRC/MONET/preprocess/pdf_match.py \
    --image "$HDF5_PATH" \
    --pdf-extracted "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --output "$MATCHED_PATH"

# 6.5 Extract differential-diagnosis text blocks.
DIFF_JSON="$OUT_ROOT/differential_diagnosis.json"
echo "[6.5/7] Extracting differential-diagnosis text..."
python3 extract_differential_diagnosis.py --input-dir "$OUTPUT_DIR" --output "$DIFF_JSON"

# 7. Export image paths.
IMG_PATHS_DIR="${MATCHED_PATH%.csv}_image_paths_dir"
IMG_PATHS_PKL="${IMG_PATHS_DIR}.pkl"
IMG_PATHS_TXT="${MATCHED_PATH%.csv}_image_paths.txt"

# Remove stale outputs.
rm -rf "$IMG_PATHS_DIR" "$IMG_PATHS_PKL" "$IMG_PATHS_TXT"

echo "[7/7] Exporting image paths..."
python3 $MONET_SRC/MONET/preprocess/save_as_path.py \
    --input "$HDF5_PATH" \
    --field images \
    --output "$IMG_PATHS_DIR"

# Convert the pkl mapping into a tab-separated txt file.
python3 -c "import pickle; d=pickle.load(open('$IMG_PATHS_PKL','rb')); f=open('$IMG_PATHS_TXT','w'); [f.write(f'{k}\t{v}\n') for k,v in d.items()]; f.close()"

# Clean up the temporary directory.
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "PDF processing complete."
echo "Final CSV: $MATCHED_PATH"
echo "Image paths: $IMG_PATHS_TXT"
echo "All outputs saved under: $OUT_ROOT"
echo ""
echo "=== Output files ==="
echo "1. Extracted text: $OUTPUT_DIR/"
echo "2. Features: $OUT_ROOT/pdf_outputs.featurized.pt"
echo "3. Clustering: $OUT_ROOT/pdf_outputs.clustering/"
echo "4. Filtered HDF5: $HDF5_PATH"
echo "5. Image-text CSV: $MATCHED_PATH"
echo "6. Differential-diagnosis text: $DIFF_JSON"
echo "7. Image paths: $IMG_PATHS_TXT"

# shellcheck source=scripts/_run_dir_hint.sh
source "$SCRIPT_DIR/_run_dir_hint.sh"
write_run_dir_hint "$OUT_ROOT"
