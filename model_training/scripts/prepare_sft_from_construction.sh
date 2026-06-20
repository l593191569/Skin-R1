#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bridge SkinRationale (data_construction) → model_training SFT layout.
#
# Copies SkinRationale train_type*.jsonl into trajectory_v2/ and links textbook
# images from the Stage 0 run directory. Also copies synonym_and_subtype2.json.
#
# Requires:
#   SKIN_R1_DATA_DIR (default <repo>/data) — holds SkinRationale under sft_dataset/
#   RUN_DIR — Stage 0 run with pdf_outputs.matched_image_paths_dir/
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"

# shellcheck source=_data_root.sh
source "$SCRIPT_DIR/_data_root.sh"
skin_r1_set_data_root "$PROJECT_DIR"

DATA_DIR=${SKIN_R1_DATA_DIR:-"$REPO_ROOT/data"}
SFT_SRC="$DATA_DIR/sft_dataset"
TRAJ_DIR="$SKIN_R1_DATA_ROOT/trajectory_v2"
IMG_DIR="$TRAJ_DIR/images"

if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -f "$DATA_DIR/outputs/.env_run" ]]; then
    # shellcheck source=/dev/null
    source "$DATA_DIR/outputs/.env_run"
  fi
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "Error: RUN_DIR is not set (Stage 0 run with matched images)."
  echo "  export RUN_DIR=$DATA_DIR/outputs/bbc_continue_<timestamp>"
  exit 1
fi

IMAGE_SRC="$RUN_DIR/pdf_outputs.matched_image_paths_dir"
if [[ ! -d "$IMAGE_SRC" ]]; then
  echo "Error: image directory not found: $IMAGE_SRC"
  exit 1
fi

shopt -s nullglob
jsonl_files=("$SFT_SRC"/train_type*.jsonl)
if [[ ${#jsonl_files[@]} -eq 0 ]]; then
  echo "Error: no train_type*.jsonl under $SFT_SRC"
  echo "Run data_construction/scripts/generate_sft_samples.sh first (SkinRationale)."
  exit 1
fi

mkdir -p "$IMG_DIR"
echo "SKIN_R1_DATA_ROOT: $SKIN_R1_DATA_ROOT"
echo "Copying SkinRationale jsonl: $SFT_SRC → $TRAJ_DIR/"
cp -f "${jsonl_files[@]}" "$TRAJ_DIR/"

echo "Linking images: $IMAGE_SRC → $IMG_DIR/"
shopt -s dotglob
for img in "$IMAGE_SRC"/*; do
  base=$(basename "$img")
  ln -sfn "$img" "$IMG_DIR/$base"
done

SYN_SRC="$REPO_ROOT/data_construction/synonym_and_subtype2.json"
if [[ -f "$SYN_SRC" ]]; then
  cp -f "$SYN_SRC" "$SKIN_R1_DATA_ROOT/synonym_and_subtype2.json"
  echo "Copied synonym_and_subtype2.json → $SKIN_R1_DATA_ROOT/"
fi

echo ""
echo "SkinRationale ready under $SKIN_R1_DATA_ROOT/trajectory_v2/"
echo "Next: cd model_training && bash scripts/run_sft.sh"
