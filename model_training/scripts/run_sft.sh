#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 1: Supervised fine-tuning (SFT) on SkinRationale. Trains a LoRA adapter
#          on top of Qwen2.5-VL-7B.
#
# Entry point: model_training/src/train_sft_trajectory.py
# Output:      $OUTPUT_DIR/module_checkpoint/module_epoch_{N}
#
# Edit the CONFIG block below for your environment, then run:
#     bash scripts/run_sft.sh
# (A SLURM header is provided; submit with `sbatch scripts/run_sft.sh` on a cluster.)
# ---------------------------------------------------------------------------
#SBATCH --job-name=skin_r1_sft
#SBATCH --output=logs/output/skin_r1_sft_%j.out
#SBATCH --error=logs/error/skin_r1_sft_%j.err
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

# ============================ CONFIG (edit me) =============================
# Repo root is auto-detected from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Base VLM (paper uses Qwen2.5-VL-7B-Instruct).
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}

# Root that holds SkinRationale (trajectory_v2/, etc.). Consumed by src/data_v2_old.py.
# shellcheck source=_data_root.sh
source "$SCRIPT_DIR/_data_root.sh"
skin_r1_set_data_root "$PROJECT_DIR"

# HuggingFace / torch caches.
export SKIN_R1_CACHE_DIR=${SKIN_R1_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}
export HF_HOME=$SKIN_R1_CACHE_DIR
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

# Activate your python environment here, e.g.:
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate skin_r1
# ===========================================================================

mkdir -p "$PROJECT_DIR/logs/output" "$PROJECT_DIR/logs/error" "$SKIN_R1_CACHE_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

TRAJECTORY_DATASET=${TRAJECTORY_DATASET:-trajectory}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/output/SFT_trajectory_${TIMESTAMP}}
export WANDB_PROJECT=${WANDB_PROJECT:-Skin-R1-SFT}
mkdir -p "$OUTPUT_DIR"

echo "Model:   $MODEL_NAME_OR_PATH"
echo "Dataset: $TRAJECTORY_DATASET (root: $SKIN_R1_DATA_ROOT)"
echo "Output:  $OUTPUT_DIR"

# NOTE: --max_train_samples / --max_eval_samples cap the dataset for quick smoke
# runs. Remove them (or raise the limits) to train on the full trajectory set.
python -m src.train_sft_trajectory \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --cache_dir "$SKIN_R1_CACHE_DIR" \
  --dataset_source "$TRAJECTORY_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 8 \
  --per_device_train_batch_size 1 \
  --learning_rate 3e-5 \
  --logging_steps 5 \
  --max_seq_length 1024 \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --seed 42 \
  --sft_test \
  --max_train_samples 3000 \
  --max_eval_samples 150 \
  --val_ratio 0.05 \
  --debug_print \
  --timestamp "$TIMESTAMP" \
  --task_name "SFT_trajectory" \
  --eval_steps 50 \
  --gradient_accumulation_steps 16 \
  --save_model_every_n_epochs 1

echo "Done. Checkpoints under: $OUTPUT_DIR/module_checkpoint/"
# The RL stage (scripts/run_rl.sh) loads e.g. module_checkpoint/module_epoch_4
