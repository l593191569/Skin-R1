#!/bin/bash
# Optional SLURM headers (only used when submitting with `sbatch`; ignored by `bash`).
# Set --partition / --account / --gres to match your cluster.
#SBATCH --job-name=bbc_pdf_continue
#SBATCH --output=logs/output/bbc_pdf_continue_%j.out
#SBATCH --error=logs/error/bbc_pdf_continue_%j.err
#SBATCH --partition=<your_partition>
#SBATCH --account=<your_account>
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Work from the data_construction/ directory (parent of scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Create log directories.
mkdir -p logs/output logs/error

# Activate your python environment here, e.g.:
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate derm1m

# Resume PDF processing from an existing run.
# Reuse the extracted/featurized/clustered files of a previous run and only
# re-run filter -> match with a (possibly different) --exclude-label set.
# Steps: copy previous outputs -> re-filter -> image-text matching -> export.
# Note: install the data-construction requirements first (pip install -r requirements.txt).

# MONET package path (vendored under data_construction/MONET-main).
export MONET_SRC=${MONET_SRC:-MONET-main/src}
export PYTHONPATH=$MONET_SRC:${PYTHONPATH:-}

# Unified data directory (default <repo>/data).
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}

# Arguments:
#   $1: name of the previous run subdirectory under data/outputs/ (e.g. bbc_<timestamp>).
SOURCE_RUN=${1:-"bbc_<timestamp>"}
#   $2: exclude labels (cluster codes from kmeans_label_lower.csv).
EXCLUDE_LABELS=${2:-"02_01 02_03 02_00 01 00 03"}

# Create a new, timestamped output subdirectory under data/outputs/.
RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUT_ROOT="$DATA_DIR/outputs/bbc_continue_$RUN_ID"
mkdir -p "$OUT_ROOT"

# Source paths (from the previous run).
SOURCE_ROOT="$DATA_DIR/outputs/$SOURCE_RUN"
SOURCE_OUTPUT_DIR="$SOURCE_ROOT/pdf_outputs"
SOURCE_HDF5_PATH="$SOURCE_ROOT/pdf_outputs.compact.hdf5"
SOURCE_FEATURIZED="$SOURCE_ROOT/pdf_outputs.featurized.pt"
SOURCE_CLUSTERING="$SOURCE_ROOT/pdf_outputs.clustering"

# New output paths.
OUTPUT_DIR="$OUT_ROOT/pdf_outputs"
HDF5_PATH="$OUT_ROOT/pdf_outputs.compact.hdf5"
MATCHED_PATH="$OUT_ROOT/pdf_outputs.matched.csv"
CONFIG_PATH="pdf_files.config.json"

echo "Resuming from existing run: $SOURCE_RUN"
echo "Exclude labels: $EXCLUDE_LABELS"
echo "New output directory: $OUT_ROOT"

# Check that the source files exist.
if [ ! -f "$SOURCE_HDF5_PATH" ]; then
    echo "Error: source HDF5 file not found: $SOURCE_HDF5_PATH"
    echo "Check that the source run directory exists, or pass the correct one."
    echo "Usage: $0 [source_run_dir] [exclude labels]"
    echo "Example: $0 bbc_20250729_030037 '02_01 02_03 01 00 03'"
    exit 1
fi

if [ ! -f "$SOURCE_FEATURIZED" ]; then
    echo "Error: source feature file not found: $SOURCE_FEATURIZED"
    exit 1
fi

if [ ! -d "$SOURCE_CLUSTERING" ]; then
    echo "Error: source clustering directory not found: $SOURCE_CLUSTERING"
    exit 1
fi

if [ ! -d "$SOURCE_OUTPUT_DIR" ]; then
    echo "Error: source output directory not found: $SOURCE_OUTPUT_DIR"
    exit 1
fi

# Copy the previous outputs into the new directory.
echo "[1/4] Copying previous run outputs..."
cp "$SOURCE_HDF5_PATH" "$HDF5_PATH"
cp "$SOURCE_FEATURIZED" "$OUT_ROOT/pdf_outputs.featurized.pt"
cp -r "$SOURCE_CLUSTERING" "$OUT_ROOT/pdf_outputs.clustering"
cp -r "$SOURCE_OUTPUT_DIR" "$OUTPUT_DIR"

# 2. Re-filter images with the chosen exclude labels.
echo "[2/4] Re-filtering images with custom exclude labels..."
python3 $MONET_SRC/MONET/preprocess/filter.py \
    --input "$HDF5_PATH" \
    --label-file "$OUT_ROOT/pdf_outputs.clustering/kmeans_label_lower.csv" \
    --exclude-label $EXCLUDE_LABELS \
    --output "$OUT_ROOT/pdf_outputs.dermonly.hdf5"

# Continue the pipeline with the filtered HDF5.
HDF5_PATH="$OUT_ROOT/pdf_outputs.dermonly.hdf5"

# 3. Image-text matching.
echo "[3/4] Matching images with text..."
python3 $MONET_SRC/MONET/preprocess/pdf_match.py \
    --image "$HDF5_PATH" \
    --pdf-extracted "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --output "$MATCHED_PATH"

# 3.5 Extract differential-diagnosis text blocks.
DIFF_JSON="$OUT_ROOT/differential_diagnosis.json"
echo "[3.5/4] Extracting differential-diagnosis text..."
python3 extract_differential_diagnosis.py --input-dir "$OUTPUT_DIR" --output "$DIFF_JSON"

# 4. Export image paths.
IMG_PATHS_DIR="${MATCHED_PATH%.csv}_image_paths_dir"
IMG_PATHS_PKL="${IMG_PATHS_DIR}.pkl"
IMG_PATHS_TXT="${MATCHED_PATH%.csv}_image_paths.txt"

# Remove stale outputs.
rm -rf "$IMG_PATHS_DIR" "$IMG_PATHS_PKL" "$IMG_PATHS_TXT"

echo "[4/4] Exporting image paths..."
python3 $MONET_SRC/MONET/preprocess/save_as_path.py \
    --input "$HDF5_PATH" \
    --field images \
    --output "$IMG_PATHS_DIR"

# Convert the pkl mapping into a tab-separated txt file.
python3 -c "import pickle; d=pickle.load(open('$IMG_PATHS_PKL','rb')); f=open('$IMG_PATHS_TXT','w'); [f.write(f'{k}\t{v}\n') for k,v in d.items()]; f.close()"

echo "Resumed PDF processing complete."
echo "Source run: $SOURCE_RUN"
echo "Exclude labels: $EXCLUDE_LABELS"
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
